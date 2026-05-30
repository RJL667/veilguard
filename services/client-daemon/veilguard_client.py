#!/usr/bin/env python3
"""
Veilguard Client Daemon — Local tool execution for cloud-hosted Veilguard.

Connects to the cloud sub-agents server via WebSocket. Receives tool execution
requests (file ops, commands, searches), runs them locally, returns results.

Usage:
    python veilguard_client.py                          # Uses config.yaml
    python veilguard_client.py --server ws://host:8809/ws/client --token abc123
"""

# Bump this on every release. The auto-updater compares against the
# manifest served by the cloud at GET /api/client/latest — when the
# remote version is higher, the client downloads the new installer,
# runs it silently, and exits so Inno Setup can replace files.
# Semver: MAJOR.MINOR.PATCH. 3-part only; pre-release tags not supported.
#
# 0.2.2 (2026-04-26): self-heal on revoked credentials. When the WS
# server returns "Invalid token" or "Missing user_id" the daemon now
# wipes the stored token+user_id from ~/.veilguard/config.yaml,
# launches the setup page at http://localhost:9090/, and waits for the
# user to paste a fresh QR-blob from the LibreChat cowork panel. Pre-
# 0.2.2 daemons just looped forever on auth failure, requiring the
# user to manually delete config.yaml and reinstall — that's what bit
# us during the spear-phish-incident token rotation on 2026-04-24.
#
# 0.2.3 (2026-04-28): fix the 0.2.2 self-heal that NEVER actually
# fired. The `except Exception` clause in run_daemon's connect loop
# was swallowing CredentialsRevokedError before main() could catch
# it, so the daemon just logged "Unexpected error: " and reconnected
# forever. Now CredentialsRevokedError is caught explicitly and
# re-raised. Caught by 0.2.2's smoke test on PJ's machine: he
# upgraded to 0.2.2, the cloud rejected his rotated token, and 0.2.2
# still didn't show the setup page.
#
# 0.2.4 (2026-04-28): installer hardening — no daemon code change.
# Bumping is purely to give the new VeilguardSetup.exe correct
# embedded version metadata (FileVersion / ProductVersion) and to
# anchor a stable AppId GUID for upgrade detection. Sarel hit a
# confusion where 0.2.3's installer had no FileVersion resource AND
# Inno's upgrade prompt cited the existing-installed version, so
# users couldn't tell what version of the installer they had until
# they finished installing. From 0.2.4 onwards, right-click ->
# Properties -> Details on VeilguardSetup.exe shows the version
# directly, and the AppId pin makes the upgrade flow predictable.
__version__ = "0.9.3"


class CredentialsRevokedError(Exception):
    """Raised by ``run_daemon`` when the WS server explicitly rejects
    our credentials. ``main()`` catches this, wipes the stored token /
    user_id, re-runs the setup UI, then restarts the daemon loop with
    the freshly pasted credentials. Treat distinct from network errors
    (which we want to keep retrying with backoff)."""
    pass

import argparse
import asyncio
import json
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import quote_plus, urlparse, urlunparse

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets")
    sys.exit(1)

try:
    import yaml
except ImportError:
    yaml = None

try:
    import httpx
except ImportError:
    httpx = None

def _daemon_log_path() -> str:
    """Where the daemon writes its rotating log file.

    Windows: %LOCALAPPDATA%\\Veilguard\\daemon.log
    Linux:   ~/.veilguard/daemon.log
    """
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        return os.path.join(base, "Veilguard", "daemon.log")
    return os.path.join(os.path.expanduser("~"), ".veilguard", "daemon.log")


# Console=False in the PyInstaller spec means stdout/stderr disappear
# when run as a tray-resident process.  We still want logs visible —
# the tray's "Logs" tab tails the file below.  RotatingFileHandler
# caps disk use at ~3 MB (1 MB × 3 rolls); plenty for debugging
# without blowing up over time.
_LOG_PATH = _daemon_log_path()
try:
    os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
    from logging.handlers import RotatingFileHandler

    class _FlushingRotatingFileHandler(RotatingFileHandler):
        """RotatingFileHandler that flushes after every record.

        Stock behaviour buffers up to ~4 KB; on a hard process kill the
        last ~hundred log lines are lost, which makes post-mortem
        debugging brutal.  Flushing is cheap on the daemon's volume
        (~dozens of records per minute) and the trade-off favours
        observability over a few extra fsync() calls.
        """
        def emit(self, record):
            super().emit(record)
            try:
                self.flush()
            except Exception:
                pass

    _file_handler = _FlushingRotatingFileHandler(
        _LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8",
    )
    _file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
except Exception:
    _file_handler = None

_handlers: list = [logging.StreamHandler()]   # stderr; harmless when console=False
if _file_handler is not None:
    _handlers.append(_file_handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [veilguard-client] %(message)s",
    datefmt="%H:%M:%S",
    handlers=_handlers,
)
logger = logging.getLogger("veilguard-client")
logger.info(f"[boot] daemon log file: {_LOG_PATH}")


# ── Safety Validation (embedded from utils/safety.py) ────────────────────────

AGENT_DANGEROUS_PATTERNS = [
    (re.compile(r"rm\s+-rf\s+/", re.IGNORECASE), "Recursive delete from root"),
    (re.compile(r"format\s+[a-z]:", re.IGNORECASE), "Format disk drive"),
    (re.compile(r"del\s+/[sfq]", re.IGNORECASE), "Recursive delete"),
    (re.compile(r"rmdir\s+/s", re.IGNORECASE), "Remove directory tree"),
    (re.compile(r"reg\s+delete", re.IGNORECASE), "Delete registry keys"),
    (re.compile(r"net\s+user\s+.*\s+/delete", re.IGNORECASE), "Delete user account"),
    (re.compile(r"shutdown", re.IGNORECASE), "Shutdown system"),
    (re.compile(r"bcdedit", re.IGNORECASE), "Modify boot config"),
    (re.compile(r"diskpart", re.IGNORECASE), "Disk partition tool"),
    (re.compile(r"cipher\s+/w:", re.IGNORECASE), "Secure wipe"),
    (re.compile(r"powershell\s+-enc", re.IGNORECASE), "Encoded PowerShell command"),
    (re.compile(r"\|\s*powershell", re.IGNORECASE), "Piped to PowerShell"),
]

PROTECTED_PATHS = {".env", ".git", "credentials", "secrets", "id_rsa", ".ssh"}


def validate_command(cmd: str) -> tuple:
    for pat, desc in AGENT_DANGEROUS_PATTERNS:
        if pat.search(cmd):
            return False, f"BLOCKED: {desc}"
    return True, ""


def is_path_safe(path: str) -> tuple:
    for p in PROTECTED_PATHS:
        if p in path.lower():
            return False, f"BLOCKED: Cannot write to protected path containing '{p}'"
    return True, ""


def safe_resolve(path: str, work_dir: str) -> tuple:
    """Resolve path safely within work_dir. Returns (resolved, error)."""
    if os.path.isabs(path):
        resolved = os.path.realpath(path)
    else:
        resolved = os.path.realpath(os.path.join(work_dir, path))
    # Path traversal check
    if not resolved.startswith(os.path.realpath(work_dir)):
        return "", f"BLOCKED: Path traversal — {path} resolves outside project root"
    return resolved, ""


# ── Tool Execution ───────────────────────────────────────────────────────────

class ToolExecutor:
    """Executes tools locally with safety validation.

    Supports multiple working folders. File operations are restricted to
    the allowed folders. The first folder is the default working directory.
    """

    def __init__(self, project_root: str, working_folders: list = None):
        self.project_root = os.path.realpath(project_root)
        self.working_folders = [os.path.realpath(f) for f in (working_folders or [project_root])]
        if self.project_root not in self.working_folders:
            self.working_folders.insert(0, self.project_root)

    def is_path_allowed(self, path: str) -> tuple:
        """Check if a path falls within any allowed working folder."""
        resolved = os.path.realpath(path)
        for folder in self.working_folders:
            if resolved.startswith(folder):
                return True, ""
        return False, f"BLOCKED: Path '{path}' is outside allowed working folders"

    def get_folders(self) -> list:
        """Return the list of allowed working folders."""
        return self.working_folders

    def set_folders(self, folders: list):
        """Update the allowed working folders."""
        self.working_folders = [os.path.realpath(f) for f in folders]
        if self.working_folders:
            self.project_root = self.working_folders[0]

    async def execute(self, tool: str, args: dict) -> str:
        """Dispatch tool execution. Returns result string.

        Critical detail: the sync ``_tool_*`` handlers use blocking
        ``subprocess.run()`` calls. If we invoked them directly from
        this async function, the blocking call would freeze the ENTIRE
        asyncio event loop for the duration of the subprocess. During
        that freeze:
          - the heartbeat task can't fire (app-level keepalive dies)
          - the WebSocket recv loop can't read new frames
          - the connection eventually drops
        Observed 23 Apr 2026 with Petrus's Pipedrive run: a 2-3 minute
        ``run_command`` froze the loop, heartbeat missed, daemon
        dropped three times in 10 minutes.

        Fix: dispatch sync handlers via ``asyncio.to_thread`` so they
        run in the default thread pool. The event loop stays
        responsive, heartbeat keeps ticking, websocket pings answer.
        Async handlers (web_search / web_fetch) are awaited directly —
        they're already non-blocking.
        """
        handler = getattr(self, f"_tool_{tool}", None)
        if handler is None:
            return f"Error: Unknown tool '{tool}'"
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(args)
            else:
                # Run in a worker thread so subprocess.run doesn't
                # block the event loop.
                result = await asyncio.to_thread(handler, args)
            return result
        except Exception as e:
            return f"Error executing {tool}: {e}"

    def _tool_read_file(self, args: dict) -> str:
        path = args.get("path", "")
        full, err = safe_resolve(path, self.project_root)
        if err:
            return err
        allowed, reason = self.is_path_allowed(full)
        if not allowed:
            return reason
        if not os.path.exists(full):
            return f"Error: File not found: {path}"
        offset = int(args.get("offset", 0))
        limit = int(args.get("limit", 500))
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        total_lines = len(lines)
        selected = lines[offset:offset + limit]
        body = "".join(f"{offset + i + 1}\t{l}" for i, l in enumerate(selected))
        # [F13_EXPLICIT_COMPLETENESS_2026_05_27] Critic-claim consistently
        # hallucinated "truncated at line 25" when given a 42-line file
        # with no completeness marker, then spun re-reading the same
        # range 12+ times.  Daemon was returning the full file every
        # time; the model just couldn't tell.  Now we emit an explicit
        # footer so the model has a ground-truth signal rather than
        # having to infer completeness from the absence of a marker.
        end_line = offset + len(selected)
        if end_line >= total_lines:
            footer = (
                f"\n\n[end of file — {total_lines} line(s) total, "
                f"all shown]"
            )
        else:
            remaining = total_lines - end_line
            footer = (
                f"\n\n[partial read — showed lines {offset + 1}-{end_line} "
                f"of {total_lines}; {remaining} more line(s) available, "
                f"call again with offset={end_line}]"
            )
        return body + footer

    def _tool_write_file(self, args: dict) -> str:
        path = args.get("path", "")
        content = args.get("content", "")
        full, err = safe_resolve(path, self.project_root)
        if err:
            return err
        allowed, reason = self.is_path_allowed(full)
        if not allowed:
            return reason
        safe, reason = is_path_safe(full)
        if not safe:
            return reason
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} chars to {path}"

    def _tool_edit_file(self, args: dict) -> str:
        path = args.get("path", "")
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        full, err = safe_resolve(path, self.project_root)
        if err:
            return err
        allowed, reason = self.is_path_allowed(full)
        if not allowed:
            return reason
        safe, reason = is_path_safe(full)
        if not safe:
            return reason
        if not os.path.exists(full):
            return f"Error: File not found: {path}"
        with open(full, "r", encoding="utf-8") as f:
            content = f.read()
        if old_string not in content:
            return f"Error: old_string not found in {path}. Read the file first."
        if content.count(old_string) > 1:
            return f"Error: old_string appears {content.count(old_string)} times. Provide more context."
        with open(full, "w", encoding="utf-8") as f:
            f.write(content.replace(old_string, new_string, 1))
        return f"Edited {path}: replaced {len(old_string)} chars with {len(new_string)} chars"

    def _tool_search_files(self, args: dict) -> str:
        """Glob for files under ``path``. Bounded iteration so a sloppy
        ``**/*`` doesn't take down the daemon.

        2026-05-18 fix: the previous implementation did
            ``sorted(Path(search_path).glob(pattern), key=mtime)[:50]``
        which materialized the ENTIRE result set (potentially hundreds
        of thousands of entries under C:\\Users\\<name>) AND called
        ``stat()`` on every one. With ``pattern='**/*'`` this hung the
        daemon. Now we:
          1. Cap iteration at ``MAX_SCAN`` (default 5000) — stop early
             rather than materialize the world.
          2. Wall-clock timeout (default 8s) — short-circuit on slow
             filesystems / network drives / Windows AppData jungle.
          3. Skip common noise dirs by name (``.git``, ``node_modules``,
             ``__pycache__``, ``AppData``, ``.venv``, ``.cache``, etc.).
          4. Sort only the (capped) result set, not the universe.
        """
        import time as _t
        pattern = args.get("pattern", "*")
        search_path = args.get("path", self.project_root)
        if not os.path.isabs(search_path):
            search_path = os.path.join(self.project_root, search_path)

        MAX_SCAN = int(args.get("max_results") or 5000)
        WALL_CLOCK_S = float(args.get("timeout") or 8.0)
        SKIP_DIRS = {
            ".git", "node_modules", "__pycache__", "AppData",
            ".venv", "venv", ".cache", "site-packages",
            "dist", "build", ".next", ".turbo", ".nuxt",
            ".pytest_cache", ".mypy_cache", "Cache", "Caches",
        }

        deadline = _t.monotonic() + WALL_CLOCK_S
        scanned = 0
        truncated = False
        matches: list[Path] = []

        try:
            it = Path(search_path).glob(pattern)
            for p in it:
                scanned += 1
                if any(part in SKIP_DIRS for part in p.parts):
                    continue
                matches.append(p)
                if len(matches) >= MAX_SCAN:
                    truncated = True
                    break
                if scanned & 0xFF == 0 and _t.monotonic() > deadline:
                    truncated = True
                    break
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

        def _mt(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except Exception:
                return 0.0

        matches.sort(key=_mt, reverse=True)
        matches = matches[:50]

        root = self.project_root
        body = "\n".join(
            str(m.relative_to(root)) if str(m).startswith(root) else str(m)
            for m in matches
        ) or "(no matches)"

        if truncated:
            body += (
                f"\n\n[search_files: scanned {scanned} entries, hit "
                f"MAX_SCAN={MAX_SCAN} or {WALL_CLOCK_S}s timeout. "
                f"Showing 50 most-recent matches from the capped set. "
                f"Use a narrower pattern (e.g. '**/*.py') or smaller "
                f"path for full coverage.]"
            )
        return body

    def _tool_grep(self, args: dict) -> str:
        pattern = args.get("pattern", "")
        path = args.get("path", ".")
        include = args.get("include", "")
        if os.path.isabs(path):
            try:
                search_path = os.path.relpath(path, self.project_root).replace("\\", "/")
            except ValueError:
                search_path = path.replace("\\", "/")
        elif path and path != ".":
            search_path = path.replace("\\", "/")
        else:
            search_path = "."
        grep_args = ["grep", "-rn", "-m", "30"]
        if include:
            grep_args.append(f"--include={include}")
        else:
            for ext in ["*.py", "*.md", "*.json", "*.yaml", "*.yml", "*.txt", "*.js", "*.ts"]:
                grep_args.append(f"--include={ext}")
        grep_args.extend(["--", pattern, search_path])
        # 15s grep timeout was too tight on large monorepos. 60s gives
        # slack for deep recursive scans; event loop stays free because
        # execute() now dispatches sync handlers through asyncio.to_thread.
        grep_timeout = int(args.get("timeout") or 60)
        try:
            result = subprocess.run(grep_args, capture_output=True, text=True, timeout=grep_timeout, cwd=self.project_root)
            output = result.stdout[:3000]
        except FileNotFoundError:
            search_dir = os.path.join(self.project_root, search_path)
            result = subprocess.run(
                ["findstr", "/S", "/N", "/R", pattern, os.path.join(search_dir, "*.*")],
                capture_output=True, text=True, timeout=grep_timeout, cwd=self.project_root
            )
            output = result.stdout[:3000]
        return output or "(no matches)"

    def _tool_run_command(self, args: dict) -> str:
        cmd = args.get("command", "")
        safe, reason = validate_command(cmd)
        if not safe:
            return reason
        # Timeout bumped 30s -> 600s (10 min). Real scripts (Pipedrive
        # bulk loaders, batch NLP passes, etc.) routinely take 2-5
        # minutes and were timing out mid-run with the 30s cap — users
        # saw TimeoutExpired, the script kept running in background,
        # and they had no reliable way to get the result back. Caller
        # can still override via args["timeout"] if they know they
        # need shorter. 10 min is generous enough for any reasonable
        # interactive script; anything longer should be a proper
        # background job via start_task rather than a blocking tool
        # call.
        timeout = int(args.get("timeout") or 600)
        if os.name == "nt":
            has_pipe = "|" in cmd or ">" in cmd or "&&" in cmd
            if has_pipe:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=self.project_root)
            else:
                result = subprocess.run(["cmd", "/c", cmd], capture_output=True, text=True, timeout=timeout, cwd=self.project_root)
        else:
            result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout, cwd=self.project_root)
        out = result.stdout[:2000]
        if result.stderr:
            out += f"\nstderr: {result.stderr[:500]}"
        return out or "(no output)"

    async def _tool_web_search(self, args: dict) -> str:
        if httpx is None:
            return "Error: httpx not installed on client"
        safe_query = quote_plus(args.get("query", ""))
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(
                f"https://lite.duckduckgo.com/lite/?q={safe_query}",
                headers={"User-Agent": "Veilguard-Agent/1.0"}
            )
            text = resp.text[:3000]
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return f"Web search results:\n\n{text[:2000]}"

    async def _tool_web_fetch(self, args: dict) -> str:
        if httpx is None:
            return "Error: httpx not installed on client"
        url = args.get("url", "")
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Veilguard-Agent/1.0"})
            if resp.status_code != 200:
                return f"Error: HTTP {resp.status_code}"
            text = resp.text[:3000]
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:3000]

    # ── Host-exec tools ──────────────────────────────────────────────────

    def _tool_run_cmd(self, args: dict) -> str:
        return self._tool_run_command({"command": args.get("command", "")})

    def _tool_run_powershell(self, args: dict) -> str:
        cmd = args.get("command", "")
        safe, reason = validate_command(cmd)
        if not safe:
            return reason
        # 60s was too short for real PS work (installers, bulk file
        # ops, Get-ChildItem -Recurse on big trees). Now 600s w/
        # caller override, matching _tool_run_command.
        timeout = int(args.get("timeout") or 600)
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True, text=True, timeout=timeout, cwd=self.project_root
        )
        out = result.stdout[:2000]
        if result.stderr:
            out += f"\nstderr: {result.stderr[:500]}"
        return out or "(no output)"

    def _tool_run_docker(self, args: dict) -> str:
        cmd = args.get("command", "")
        safe, reason = validate_command(cmd)
        if not safe:
            return reason
        # docker build / compose up / pull on fresh images easily
        # blow past 120s. Bump to 600s w/ caller override.
        timeout = int(args.get("timeout") or 600)
        result = subprocess.run(
            f"docker {cmd}", shell=True,
            capture_output=True, text=True, timeout=timeout, cwd=self.project_root
        )
        out = result.stdout[:2000]
        if result.stderr:
            out += f"\nstderr: {result.stderr[:500]}"
        return out or "(no output)"

    def _tool_run_git(self, args: dict) -> str:
        cmd = args.get("command", "")
        # Block destructive git ops
        git_dangerous = [
            re.compile(r"push\s+.*--force", re.IGNORECASE),
            re.compile(r"reset\s+--hard", re.IGNORECASE),
            re.compile(r"clean\s+-[fd]", re.IGNORECASE),
            re.compile(r"branch\s+-D", re.IGNORECASE),
        ]
        for pat in git_dangerous:
            if pat.search(cmd):
                return f"BLOCKED: Destructive git operation"
        # git clone / fetch / push on big repos can exceed 30s. Bump
        # to 300s w/ caller override. Matches run_command/docker pattern.
        timeout = int(args.get("timeout") or 300)
        result = subprocess.run(
            f"git {cmd}", shell=True,
            capture_output=True, text=True, timeout=timeout, cwd=self.project_root
        )
        out = result.stdout[:2000]
        if result.stderr:
            out += f"\nstderr: {result.stderr[:500]}"
        return out or "(no output)"

    def _tool_host_file_read(self, args: dict) -> str:
        return self._tool_read_file(args)

    def _tool_host_file_write(self, args: dict) -> str:
        return self._tool_write_file(args)


# ── Auto-updater ─────────────────────────────────────────────────────────────
#
# Every UPDATE_CHECK_INTERVAL seconds the daemon hits the cloud manifest
# endpoint (derived from the WebSocket URL). If the manifest advertises a
# higher version, the client:
#   1. Downloads the new installer to a temp file
#   2. Launches it detached with Inno Setup silent flags
#   3. Calls os._exit(0) so the installer can overwrite the running .exe
#      (Inno Setup's CloseApplications=yes also kills running instances
#      as a belt-and-braces measure)
#
# The installer's [Run] section relaunches VeilguardClient.exe, so the
# user never sees a "daemon stopped" state for more than ~10s.
#
# Ops flow for shipping a release:
#   1. Bump __version__ in this file
#   2. Bump AppVersion in installer.iss
#   3. Run build.bat → produces installer_output/VeilguardSetup.exe
#   4. scp VeilguardSetup.exe + version.json to the VM downloads dir
#   5. Every connected client picks up the update within UPDATE_CHECK_INTERVAL
#
# First check runs 60s after startup so a freshly-installed client doesn't
# immediately re-update in a loop if the manifest is briefly stale.

UPDATE_CHECK_INTERVAL_SEC = 30 * 60   # every 30 minutes
UPDATE_FIRST_CHECK_DELAY_SEC = 60     # first check 60s after startup


def _parse_version(v: str) -> tuple:
    """Parse '0.2.0' → (0, 2, 0). Returns (0,0,0) on malformed input."""
    try:
        parts = [int(x) for x in v.strip().split(".")[:3]]
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts)
    except Exception:
        return (0, 0, 0)


def _http_base_from_ws(ws_url: str) -> str:
    """Convert ws(s)://host:port/ws/client → http(s)://host:port."""
    parsed = urlparse(ws_url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    return urlunparse((scheme, parsed.netloc, "", "", "", ""))


def _candidate_manifest_urls(ws_url: str, explicit: str = "") -> list:
    """Return manifest URLs to try, in priority order.

    If the user set update_manifest_url in config.yaml, that wins.
    Otherwise we try two derivations because the prod Caddy reverse
    proxy routes /ws/client direct to the backend but prefixes HTTP
    routes with /api/sub-agents/. Local dev doesn't have that prefix.
    """
    if explicit:
        return [explicit]
    base = _http_base_from_ws(ws_url)
    return [
        # Caddy prod routing (phishield.com): HTTP has /api/sub-agents/ prefix
        f"{base}/api/sub-agents/api/client/latest",
        # Local dev (no Caddy) or when ws+http share the same path layout
        f"{base}/api/client/latest",
    ]


async def _download_and_launch_installer(url: str):
    """Download installer, launch silently detached, exit self.

    On Windows, uses Inno Setup silent-install flags; on non-Windows
    (no installer available) just logs and skips — Linux/macOS users
    run the daemon via ``python veilguard_client.py`` and can use
    pip/git pull for updates.
    """
    if httpx is None:
        logger.warning("[UPDATE] httpx not installed — cannot download installer")
        return

    if os.name != "nt":
        logger.info(
            "[UPDATE] Non-Windows platform — skipping auto-install. "
            "Update via `git pull` or `pip install -U veilguard-client`."
        )
        return

    try:
        tmp_path = os.path.join(tempfile.gettempdir(), "VeilguardSetup_update.exe")
        async with httpx.AsyncClient(timeout=600, follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    logger.error(f"[UPDATE] Download failed: HTTP {resp.status_code}")
                    return
                with open(tmp_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(64 * 1024):
                        f.write(chunk)
        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        logger.info(f"[UPDATE] Installer downloaded ({size_mb:.1f}MB) → {tmp_path}")
    except Exception as e:
        logger.error(f"[UPDATE] Download failed: {e}")
        return

    # DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP so the installer
    # survives this process exiting.
    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    try:
        subprocess.Popen(
            [tmp_path, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/NOCANCEL"],
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
        )
    except Exception as e:
        logger.error(f"[UPDATE] Failed to launch installer: {e}")
        return

    logger.info("[UPDATE] Installer launched silently. Exiting so it can replace files.")
    # Give Inno Setup a moment to start before we exit, so it can hold a
    # handle on the file and kill us via CloseApplications.
    await asyncio.sleep(2)
    os._exit(0)


async def auto_updater(server_ws_url: str, explicit_manifest_url: str = ""):
    """Background task — polls the manifest and triggers updates.

    Tries multiple candidate URLs on each cycle to tolerate the Caddy
    routing difference (prod has /api/sub-agents/ prefix on HTTP routes,
    local dev doesn't). The first candidate that returns 200 with a
    valid JSON manifest wins — once one works, we could pin it, but
    the cost of trying all candidates is a handful of 404s every 30min
    which is cheaper than extra config complexity.
    """
    if httpx is None:
        logger.info("[UPDATE] httpx not installed — auto-update disabled")
        return

    candidates = _candidate_manifest_urls(server_ws_url, explicit_manifest_url)
    logger.info(
        f"[UPDATE] Auto-updater active — will try manifest URLs: {candidates}"
    )

    await asyncio.sleep(UPDATE_FIRST_CHECK_DELAY_SEC)

    while True:
        manifest = None
        winning_url = ""
        for url in candidates:
            try:
                async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                    resp = await client.get(url)
                if resp.status_code == 200:
                    try:
                        manifest = resp.json()
                        winning_url = url
                        break
                    except Exception:
                        continue
                else:
                    logger.debug(f"[UPDATE] {url} → HTTP {resp.status_code}")
            except Exception as e:
                logger.debug(f"[UPDATE] {url} → {e}")

        if manifest is not None:
            remote_version = manifest.get("version", "0.0.0")
            if _parse_version(remote_version) > _parse_version(__version__):
                download_url = manifest.get("url", "")
                if download_url and not download_url.startswith(("http://", "https://")):
                    # Resolve relative URL against the base that served
                    # the manifest (NOT the raw ws->http base) so the
                    # Caddy prefix is preserved.
                    parsed = urlparse(winning_url)
                    # Keep everything up to the last /api/ segment that
                    # matches the server layout.
                    if "/api/sub-agents/" in winning_url:
                        http_base = winning_url.split("/api/sub-agents/")[0] + "/api/sub-agents"
                    else:
                        http_base = urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
                    download_url = f"{http_base}{download_url}"
                if download_url:
                    logger.info(
                        f"[UPDATE] New version available: "
                        f"{__version__} → {remote_version}. "
                        f"Downloading from {download_url}"
                    )
                    await _download_and_launch_installer(download_url)
                    # If we return here, install failed — fall through
                    # to sleep and retry on the next cycle.
                else:
                    logger.warning("[UPDATE] Manifest missing 'url' field")
            else:
                logger.debug(f"[UPDATE] Up to date (v{__version__})")
        else:
            logger.debug("[UPDATE] No manifest endpoint responded; will retry")

        await asyncio.sleep(UPDATE_CHECK_INTERVAL_SEC)


# ── WebSocket Client ─────────────────────────────────────────────────────────

async def run_daemon(config: dict):
    """Main daemon loop — orchestrates one WS session per enabled env.

    0.4+ behaviour: config can declare multiple `environments`, each
    with its own server/token/user_id.  The daemon spawns one
    asyncio task per enabled+complete env and runs them in parallel,
    so a user paired with BOTH prod and a local dev stack gets tool
    calls routed correctly to either side at the same time.

    Backward compat: old configs (top-level server/token/user_id)
    auto-migrate to a single "default" env via parse_environments().
    """
    from daemon.env import parse_environments

    client_id = config.get("client_id", "veilguard-client")
    project_root = os.path.realpath(config.get("project_root", "."))
    executor = ToolExecutor(project_root)

    logger.info(f"Veilguard Client v{__version__}")
    logger.info(f"Project root: {project_root}")

    # Register the AppUserModelID for this process so Windows binds
    # our toasts (winotify uses app_id="Veilguard") to a coherent
    # identity.  Without this, Windows may not display banners + the
    # app stays missing from Settings → Notifications even with the
    # Start Menu shortcut having an AUMID property.  Belt-and-braces
    # with the installer's [Icons] AppUserModelID setting.
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "Veilguard"
            )
            logger.info(
                "[boot] AUMID registered: SetCurrentProcessExplicitAppUserModelID('Veilguard') OK"
            )
        except Exception as _e:
            logger.warning(f"[boot] AUMID registration failed: {_e}")

    envs = parse_environments(config)
    enabled = [e for e in envs.values() if e.enabled and e.is_complete()]
    if not enabled:
        if envs:
            disabled_or_incomplete = ", ".join(
                f"{e.name}(disabled={not e.enabled},complete={e.is_complete()})"
                for e in envs.values()
            )
            logger.warning(
                f"[ENV] no enabled+complete envs to run: {disabled_or_incomplete}"
            )
        else:
            logger.warning(
                "[ENV] no environments declared — daemon will sit idle. "
                "Pair via 'Set up Veilguard' in LibreChat, or paste a "
                "veilguard://configure?… link."
            )
        # Block so the process stays alive (auto-updater + tray + handoff
        # listener still work).  Without this, an unpaired daemon would
        # exit immediately and Inno Setup's RestartApplications would
        # bounce it in a loop.
        while True:
            await asyncio.sleep(60)
        return

    logger.info(
        f"[ENV] spawning {len(enabled)} session(s): "
        f"{[(e.name, e.server) for e in enabled]}"
    )

    # Per-env outgoing message queue.  The tray's set_permission_level
    # callback writes into the env-named queue; each env's drainer
    # task only sends from its own.  Stash on config so the tray can
    # reach the right one.
    import queue as _stdq
    config["_outgoing_queues"] = {
        e.name: _stdq.Queue(maxsize=64) for e in enabled
    }

    # Auto-updater runs once per process, independent of WS reconnects.
    # We arbitrarily attach it to the first env's server URL — auto-
    # update probes the manifest at $server/api/client/latest, which
    # is the same on every env that runs sub-agents.
    if config.get("auto_update", True):
        asyncio.create_task(
            auto_updater(
                enabled[0].server,
                config.get("update_manifest_url", ""),
            )
        )

    # Per-env WS session tasks.  asyncio.gather keeps them all alive;
    # if one raises, the others keep going (return_exceptions=True).
    tasks = [
        asyncio.create_task(
            _run_env_session(env, executor, config, client_id),
            name=f"veilguard-env-{env.name}",
        )
        for env in enabled
    ]
    await asyncio.gather(*tasks, return_exceptions=True)


async def _run_env_session(
    env, executor, config: dict, client_id: str,
):
    """One environment's connect → auth → message loop → reconnect cycle.

    Extracted from the legacy single-env run_daemon body.  Same WS
    handling, same heartbeat / drainer / message dispatch — just
    parameterised on `env` (server/token/user_id) and the per-env
    outgoing queue at config["_outgoing_queues"][env.name].
    """
    server = env.server
    token = env.token
    user_id = env.user_id
    reconnect_delay = config.get("reconnect_delay", 5)
    max_delay = config.get("max_reconnect_delay", 300)
    current_delay = reconnect_delay

    logger.info(f"[ENV:{env.name}] server={server}")
    logger.info(f"[ENV:{env.name}] [AUTH] user_id: {user_id[:8]}...{user_id[-4:]}")

    while True:
        try:
            logger.info(f"Connecting to {server}...")
            # ping_interval=20s, ping_timeout=60s — native WebSocket keepalive.
            # Previously ping_interval=None disabled pings entirely; combined
            # with sync handlers that blocked the event loop, the connection
            # silently dropped after idle timeout during long subprocess.run
            # calls. With sync handlers now routed through asyncio.to_thread,
            # the recv loop stays responsive and native pings can fire every
            # 20s. ping_timeout=60s tolerates brief network hiccups.
            async with websockets.connect(
                server,
                ping_interval=20,
                ping_timeout=60,
                close_timeout=10,
            ) as ws:
                # Authenticate — user_id is REQUIRED by the server for
                # per-user token validation. version is sent so the cloud
                # can track which clients are still on old builds.
                await ws.send(json.dumps({
                    "jsonrpc": "2.0",
                    "method": "auth",
                    "params": {
                        "user_id":     user_id,
                        "token":       token,
                        "client_id":   client_id,
                        "version":     __version__,
                        # 2026-05-18: send real platform identity so
                        # sub-agents/proxy stop guessing OS from path
                        # prefixes. Receiving side has these slots
                        # already (bridge.platform, bridge.os_name,
                        # bridge.os_release, bridge.shell) — older
                        # daemons just left them empty and the LLM
                        # had to probe. ``sys.platform`` is the
                        # canonical short code (``win32`` / ``linux``
                        # / ``darwin``); ``platform.system()`` is the
                        # friendly name (``Windows`` / ``Linux`` /
                        # ``Darwin``). Shell is best-effort: COMSPEC
                        # on Windows, SHELL on Unix.
                        "platform":    sys.platform,
                        "os_name":     platform.system(),
                        "os_release":  platform.release(),
                        "shell":       (
                            os.environ.get("COMSPEC", "")
                            if os.name == "nt"
                            else os.environ.get("SHELL", "")
                        ),
                        # [PHASE_0_0_4_CAPABILITY_HANDSHAKE_2026_05_27]
                        # Per spec §0.4 + §0.0.4 — declare which JSON-RPC
                        # methods this build of the daemon implements.
                        # The server stores these on the bridge and
                        # uses ``bridge.has_capability(name)`` to gate
                        # feature calls.  Add a string here when you
                        # land a new JSON-RPC handler in the daemon;
                        # otherwise the server will see the older
                        # build's set and gracefully degrade.  Keep
                        # alphabetical for determinism.
                        "capabilities": [
                            "approval_callback",     # POSTs back to /api/client/approval_callback
                            "bypass_rules",          # honours bypass-rule add/remove
                            "edit_file",             # str-replace tool
                            "execute_remote",        # runs shell tools sent by the cloud
                            "file_io",               # read_file / write_file
                            "host_hint",             # accepts host-hint prepending on first run_command
                            "request_approval",      # renders toast + returns user decision
                            "task_dispatch_drainer", # processes pending task dispatch queue
                            "toast",                 # informational toasts (no decision required)
                        ],
                    },
                }))

                resp = json.loads(await ws.recv())
                if "error" in resp:
                    logger.error(f"Auth failed: {resp['error']}")
                    # Distinguish "credentials are bad" from "network /
                    # server hiccup". The first one isn't recoverable by
                    # retrying — looping forever just spams logs and
                    # masks the real problem from the user. Bail out so
                    # main() can wipe + relaunch setup.
                    err = resp.get("error") or {}
                    msg = (err.get("message") or "").lower()
                    if (
                        "invalid token" in msg
                        or "missing user_id" in msg
                        or "missing user-id" in msg
                        or "user_id" in msg and "missing" in msg
                    ):
                        raise CredentialsRevokedError(err.get("message", ""))
                    await asyncio.sleep(current_delay)
                    continue

                logger.info(f"Authenticated as '{client_id}'")
                current_delay = reconnect_delay  # Reset on success
                # Flip tray to "connected" — repaints from offline grey
                # back to the task-derived idle/busy/approval state.
                _tray_obj = config.get("_tray")
                if _tray_obj is not None:
                    try:
                        _tray_obj.set_connection_state(True)
                    except Exception:
                        pass

                # Start heartbeat
                async def heartbeat():
                    while True:
                        await asyncio.sleep(30)
                        try:
                            await ws.send(json.dumps({"jsonrpc": "2.0", "method": "ping"}))
                        except Exception:
                            break

                hb_task = asyncio.create_task(heartbeat())

                # Drain daemon→cloud outgoing queue.  The tray feeds
                # this when the user changes permission_level via the
                # right-click menu.  Survives backpressure: stops
                # consuming on WS error so the queue holds items until
                # the next reconnect picks them up.
                # Per-env outgoing queue.  Multi-env build (0.4+) keeps
                # one queue per env in config["_outgoing_queues"] so the
                # tray's set_permission_level callback can fan out to
                # specific servers without cross-talk.
                _q = (config.get("_outgoing_queues") or {}).get(env.name)
                if _q is None:
                    _q = config.get("_outgoing_queue")   # legacy fallback

                async def outgoing_drainer():
                    import queue as _qmod
                    if _q is None:
                        logger.warning(
                            f"[OUTGOING] no queue for env={env.name!r} "
                            "— drainer exiting"
                        )
                        return
                    logger.info(
                        f"[OUTGOING] drainer started env={env.name} qid={id(_q)} "
                        f"keys={list((config.get('_outgoing_queues') or {}).keys())}"
                    )
                    while True:
                        try:
                            item = await asyncio.to_thread(_q.get, True, 1.0)
                        except _qmod.Empty:
                            # Idle tick — no item.  Loop.
                            continue
                        except Exception as _e:
                            # Some other failure pulling from queue.
                            logger.warning(
                                f"[OUTGOING] queue.get failed: "
                                f"{type(_e).__name__}: {_e}"
                            )
                            await asyncio.sleep(0.5)
                            continue
                        logger.info(
                            f"[OUTGOING] dequeued method={item.get('method')!r} "
                            f"— attempting ws.send"
                        )
                        try:
                            await ws.send(json.dumps(item))
                            logger.info(
                                f"[OUTGOING] sent method={item.get('method')!r}"
                            )
                        except Exception as _e:
                            # Push back so we retry on reconnect.  If
                            # the queue is full, drop with warning —
                            # better than blocking the drainer.
                            logger.warning(
                                f"[OUTGOING] ws.send failed: "
                                f"{type(_e).__name__}: {_e}",
                                exc_info=True,
                            )
                            try:
                                _q.put_nowait(item)
                            except Exception:
                                logger.warning(
                                    f"[OUTGOING] discarded {item.get('method')!r}: {_e}"
                                )
                            break

                drain_task = asyncio.create_task(outgoing_drainer())

                # Message loop — tool calls run in parallel via asyncio tasks.
                # Active set keeps GC references to in-flight tasks; the
                # structured TaskRegistry mirrors the same entries with
                # full state (status / args / result preview) for the
                # tray UI and the LLM-queryable MCP tools.
                active_tasks = set()
                try:
                    from daemon.tasks import TaskRegistry as _TaskRegistry
                    _registry = _TaskRegistry.get()
                except Exception as _e:
                    _registry = None
                    logger.debug(f"[TASKS] registry unavailable: {_e}")

                async def run_tool(ws, executor, req_id, tool, args):
                    """Execute a tool and send the result back over WebSocket."""
                    logger.info(f"[TOOL] {tool}({list(args.keys())}) id={req_id}")
                    start = time.time()
                    # Register with the task tracker so the tray + MCP
                    # tools can see it.  Lookup happens before await so
                    # the registry sees RUNNING immediately.
                    if _registry is not None:
                        try:
                            _registry.start_tool(
                                req_id, tool, args,
                                future=asyncio.current_task(),
                            )
                        except Exception as e:
                            logger.debug(f"[TASKS] start_tool failed: {e}")

                    result = ""
                    error = ""
                    try:
                        result = await executor.execute(tool, args)
                    except asyncio.CancelledError:
                        if _registry is not None:
                            _registry.finish_tool(
                                req_id, result="", error="cancelled",
                                cancelled=True,
                            )
                        raise
                    except Exception as e:
                        result = f"Error: {e}"
                        error = str(e)
                    elapsed = time.time() - start

                    if len(result) > 50000:
                        result = result[:50000] + "\n... [truncated]"

                    logger.info(f"[TOOL] {tool} done in {elapsed:.1f}s ({len(result)} chars)")

                    if _registry is not None:
                        try:
                            _registry.finish_tool(req_id, result=result, error=error)
                        except Exception as e:
                            logger.debug(f"[TASKS] finish_tool failed: {e}")

                    await ws.send(json.dumps({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": result,
                    }))

                try:
                    async for raw in ws:
                        msg = json.loads(raw)

                        if msg.get("result") == "pong":
                            continue

                        method = msg.get("method", "")

                        if method == "execute_tool":
                            req_id = msg.get("id", "")
                            tool = msg["params"]["tool"]
                            args = msg["params"].get("args", {})

                            # Fire and forget — runs in parallel
                            task = asyncio.create_task(run_tool(ws, executor, req_id, tool, args))
                            active_tasks.add(task)
                            task.add_done_callback(active_tasks.discard)

                        elif method == "set_working_folders":
                            # Cloud sends updated working folders list
                            folders = msg.get("params", {}).get("folders", [])
                            executor.set_folders(folders)
                            logger.info(f"[FOLDERS] Updated working folders: {folders}")
                            await ws.send(json.dumps({
                                "jsonrpc": "2.0",
                                "id": msg.get("id", ""),
                                "result": {"folders": executor.get_folders()},
                            }))

                        elif method == "get_working_folders":
                            # Cloud requests current working folders
                            await ws.send(json.dumps({
                                "jsonrpc": "2.0",
                                "id": msg.get("id", ""),
                                "result": {"folders": executor.get_folders()},
                            }))

                        elif method == "request_approval":
                            # Cloud is asking the user to approve a client-tool
                            # call.  Phase C: route to the toast UI when
                            # available; fall back to Phase A auto-approve
                            # otherwise (headless containers, missing winotify,
                            # tray-UI degraded mode).  See daemon/toast.py for
                            # the click-capture HTTP shim.
                            req_id = msg.get("id", "")
                            params = msg.get("params", {}) or {}
                            tool = params.get("tool", "")
                            args_dict = params.get("args") or {}
                            conv_id = params.get("conv_id", "")
                            agent_id = params.get("agent_id", "user")
                            level = params.get("level", "?")
                            policy_decision = params.get("policy_decision", "?")
                            timeout_s = int(params.get("timeout_s", 60) or 60)
                            arg_keys = list(args_dict.keys())

                            # Register the pending approval so the tray sees
                            # it and the icon flips to "approval needed".
                            if _registry is not None:
                                try:
                                    _registry.start_approval(
                                        req_id, tool, args_dict,
                                        conv_id=conv_id, agent_id=agent_id,
                                        policy_decision=policy_decision,
                                        level=level, timeout_s=timeout_s,
                                    )
                                except Exception as e:
                                    logger.debug(f"[TASKS] start_approval failed: {e}")

                            logger.info(
                                f"[APPROVAL] req={req_id} tool={tool!r} "
                                f"policy={policy_decision} level={level} "
                                f"conv={conv_id[:8]} args={arg_keys}"
                            )

                            # Try the real Windows toast first.  Falls back to
                            # auto-approve (Phase A behavior) when UI is
                            # unavailable — headless containers, missing
                            # winotify, etc.
                            approved = False
                            reason = ""
                            persist_for_conv = False
                            try:
                                from daemon import toast as _toast
                                has_ui = _toast.HAS_TOAST_UI
                            except Exception as _e:
                                has_ui = False
                                logger.debug(f"[APPROVAL] toast import failed: {_e}")

                            if has_ui:
                                # Real user-driven flow.
                                try:
                                    decision = await _toast.ask_user_via_toast(
                                        tool=tool, args=args_dict,
                                        conv_id=conv_id, agent_id=agent_id,
                                        policy_decision=policy_decision,
                                        level=level, timeout_s=timeout_s,
                                        # Unify token so the viewer's
                                        # inline Approve/Deny buttons
                                        # (keyed by req_id in
                                        # TaskRegistry) can resolve the
                                        # SAME future the toast banner
                                        # would.
                                        request_id=req_id,
                                    )
                                    approved = decision.approved
                                    reason = decision.reason
                                    persist_for_conv = decision.persist_for_conv
                                except Exception as e:
                                    logger.error(f"[APPROVAL] toast flow raised: {e}")
                                    approved = False
                                    reason = f"toast_failed: {e}"
                            else:
                                # Headless / no UI — same fallback as Phase A.
                                approved = policy_decision in ("allow", "approve")
                                reason = (
                                    "auto-approved by daemon (no UI; policy-based)"
                                    if approved else
                                    f"policy {policy_decision!r} not in allow/approve set"
                                )

                            if _registry is not None:
                                try:
                                    _registry.finish_approval(
                                        req_id, approved=approved, reason=reason,
                                    )
                                except Exception as e:
                                    logger.debug(f"[TASKS] finish_approval failed: {e}")

                            await ws.send(json.dumps({
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "result": {
                                    "approved":         approved,
                                    "reason":           reason,
                                    "persist_for_conv": persist_for_conv,
                                    "tool":             tool,
                                    "request_id":       req_id,
                                },
                            }))

                        elif method == "dismiss_approval":
                            # Cloud is cancelling a pending approval (e.g.
                            # because the parent task was cancelled and we
                            # don't want the user to click "approve" for
                            # work that's already irrelevant).  Best-effort:
                            # we can't actually dismiss the Windows toast
                            # programmatically once it's on screen, but we
                            # mark the approval future so the user's click
                            # (if they ever make it) is ignored.
                            #
                            # When toast UI is unavailable (headless /
                            # sandbox), the cloud-side gate already
                            # auto-decided; this method is a no-op for
                            # those flows.
                            req_id = (msg.get("params") or {}).get("request_id", "")
                            dismissed = False
                            try:
                                from daemon import toast as _toast
                                # Pop the per-toast future so its
                                # eventual callback is dropped.
                                fut = _toast._pending.pop(req_id, None)
                                if fut is not None and not fut.done():
                                    fut.cancel()
                                    dismissed = True
                            except Exception as _e:
                                logger.debug(f"[DISMISS] toast lookup failed: {_e}")
                            # Also clean up the daemon-local TaskRegistry
                            # pending entry so the tray icon refreshes.
                            try:
                                if _registry is not None:
                                    _registry.finish_approval(
                                        req_id, approved=False,
                                        reason="dismissed by cloud",
                                    )
                            except Exception:
                                pass
                            logger.info(
                                f"[DISMISS] approval req={req_id} "
                                f"dismissed={dismissed}"
                            )
                            await ws.send(json.dumps({
                                "jsonrpc": "2.0",
                                "id": msg.get("id", ""),
                                "result": {"dismissed": dismissed,
                                           "request_id": req_id},
                            }))

                        elif method == "list_local_tasks":
                            # Daemon-local task snapshot (Phase D MCP).
                            # Cloud's `list_my_tasks` MCP tool round-trips
                            # this so the LLM can see {running, pending,
                            # completed} on the user's machine.
                            snap = (
                                _registry.snapshot()
                                if _registry is not None
                                else {"running": [], "pending": [], "completed": []}
                            )
                            await ws.send(json.dumps({
                                "jsonrpc": "2.0",
                                "id": msg.get("id", ""),
                                "result": snap,
                            }))

                        elif method == "task_state":
                            # Detail view for one task_id.  Returns None when
                            # the id has been evicted from the completed cap.
                            tid = (msg.get("params") or {}).get("task_id", "")
                            rec = (
                                _registry.get_task(tid)
                                if (_registry is not None and tid)
                                else None
                            )
                            await ws.send(json.dumps({
                                "jsonrpc": "2.0",
                                "id": msg.get("id", ""),
                                "result": rec,
                            }))

                        elif method == "cancel_task":
                            # Best-effort cancel.  Only running tool calls
                            # can be cancelled; pending approvals are
                            # cancelled by the user dismissing the toast.
                            tid = (msg.get("params") or {}).get("task_id", "")
                            ok = (
                                _registry.cancel_running(tid)
                                if (_registry is not None and tid)
                                else False
                            )
                            await ws.send(json.dumps({
                                "jsonrpc": "2.0",
                                "id": msg.get("id", ""),
                                "result": {"cancelled": ok, "task_id": tid},
                            }))

                        elif method == "list_directory":
                            # List directories for the folder picker UI
                            path = msg.get("params", {}).get("path", "")
                            if not path:
                                # List drives on Windows, / on Unix
                                if os.name == "nt":
                                    import string
                                    drives = [f"{d}:\\" for d in string.ascii_uppercase
                                              if os.path.exists(f"{d}:\\")]
                                    dirs = drives
                                else:
                                    dirs = ["/"]
                            else:
                                try:
                                    dirs = sorted([
                                        os.path.join(path, d) for d in os.listdir(path)
                                        if os.path.isdir(os.path.join(path, d))
                                        and not d.startswith(".")
                                        and d not in ("node_modules", "__pycache__", ".git", "venv", ".venv")
                                    ])[:50]
                                except PermissionError:
                                    dirs = []
                            await ws.send(json.dumps({
                                "jsonrpc": "2.0",
                                "id": msg.get("id", ""),
                                "result": {"directories": dirs},
                            }))
                finally:
                    hb_task.cancel()
                    try:
                        drain_task.cancel()
                    except NameError:
                        pass

        except CredentialsRevokedError:
            # Don't swallow this -- main() needs to wipe creds and run
            # the setup UI. The catch-all `except Exception` below would
            # otherwise treat it as a transient hiccup and loop forever.
            # This was the actual bug behind "0.2.2 fresh install does
            # not show QR screen" on PJ's machine: self-heal raised, the
            # broad except logged "Unexpected error: ", slept, retried,
            # and the user never saw the setup page.
            raise
        except (websockets.ConnectionClosed, ConnectionRefusedError, OSError) as e:
            logger.warning(f"Disconnected: {e}. Reconnecting in {current_delay}s...")
            _tray_obj = config.get("_tray")
            if _tray_obj is not None:
                try:
                    _tray_obj.set_connection_state(False)
                except Exception:
                    pass
            await asyncio.sleep(current_delay)
            current_delay = min(current_delay * 2, max_delay)
        except Exception as e:
            logger.error(f"Unexpected error: {e}. Reconnecting in {current_delay}s...")
            _tray_obj = config.get("_tray")
            if _tray_obj is not None:
                try:
                    _tray_obj.set_connection_state(False)
                except Exception:
                    pass
            await asyncio.sleep(current_delay)
            current_delay = min(current_delay * 2, max_delay)


# ── Entry Point ──────────────────────────────────────────────────────────────

def load_config(args) -> dict:
    """Load config from yaml file, overridden by CLI args.

    Resolution order:
      1. Defaults (below)
      2. yaml file at ``args.config`` (or ~/.veilguard/config.yaml)
      3. CLI args
      4. VEILGUARD_* env vars  (highest precedence — Phase B headless
         mode in the server-side sandbox container ships entirely via
         env vars, bypassing the yaml/setup-UI path)
    """
    config = {
        "server": "ws://localhost:8809/ws/client",
        "token": "",
        "client_id": "veilguard-client",
        "project_root": ".",
        "timeout": 60,
        "reconnect_delay": 5,
        "max_reconnect_delay": 300,
    }

    # Load from yaml if available
    config_path = args.config if args.config else "config.yaml"
    if os.path.exists(config_path) and yaml:
        with open(config_path, "r") as f:
            file_config = yaml.safe_load(f) or {}
        config.update(file_config)

    # CLI overrides
    if args.server:
        config["server"] = args.server
    if args.token:
        config["token"] = args.token
    if getattr(args, "user_id", None):
        config["user_id"] = args.user_id
    if args.client_id:
        config["client_id"] = args.client_id
    if args.project_root:
        config["project_root"] = args.project_root

    # Env-var overrides (highest precedence — for sandbox container)
    _env_map = {
        "VEILGUARD_SERVER":       "server",
        "VEILGUARD_TOKEN":        "token",
        "VEILGUARD_USER_ID":      "user_id",
        "VEILGUARD_CLIENT_ID":    "client_id",
        "VEILGUARD_PROJECT_ROOT": "project_root",
    }
    for env_key, cfg_key in _env_map.items():
        val = os.environ.get(env_key)
        if val:
            config[cfg_key] = val

    # Headless mode: server containers have no Windows tray, no QR setup,
    # no auto-update.  ``--headless`` (or VEILGUARD_HEADLESS=1) flips them
    # off and tells the daemon to expect identity via env vars.
    if getattr(args, "headless", False) or os.environ.get("VEILGUARD_HEADLESS"):
        config["headless"] = True
        config["auto_update"] = False    # no Inno Setup on Linux

    return config


def setup_and_run(server: str, token: str, project_root: str = ".", user_id: str = ""):
    """One-liner setup: save config and start daemon immediately.

    Called via:
        pip install veilguard-client && veilguard --setup wss://server/ws/client --token abc123 --user-id u123
    Or the combined one-liner the cloud UI generates.
    """
    import platform
    client_id = f"{os.getenv('USER', os.getenv('USERNAME', 'client'))}-{platform.node()}"
    project_root = os.path.realpath(project_root)

    # Save config to ~/.veilguard/config.yaml
    config_dir = os.path.join(os.path.expanduser("~"), ".veilguard")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "config.yaml")

    config = {
        "server": server,
        "token": token,
        "user_id": user_id,
        "client_id": client_id,
        "project_root": project_root,
        "timeout": 60,
        "reconnect_delay": 5,
        "max_reconnect_delay": 300,
    }

    if yaml:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        with open(config_path, "w") as f:
            for k, v in config.items():
                f.write(f"{k}: {json.dumps(v)}\n")

    print(f"""
    Veilguard Client Daemon — Setup Complete
    Config saved: {config_path}
    Server:       {server}
    Client ID:    {client_id}
    Project Root: {project_root}

    Starting daemon...
    """)

    asyncio.run(run_daemon(config))


def _handle_deeplink_or_handoff(argv: list[str]) -> bool:
    """Top-of-main hook: deal with ``veilguard://…`` invocations.

    Two scenarios:

      A. Daemon is NOT yet running.  We're the first instance.  Parse
         the URL, write config, fall through to normal startup so the
         daemon comes up with fresh creds.

      B. Daemon IS already running.  Hand argv off via the
         single_instance Listener, exit 0.  The running daemon's
         handoff thread picks it up and re-applies config + reconnects.

    Returns True when the caller should EXIT IMMEDIATELY (case B).
    Returns False to continue with normal main() flow (case A).
    """
    if len(argv) < 2 or not argv[1].startswith("veilguard://"):
        return False

    try:
        from daemon.deeplink import (
            parse_configure, apply_configure, parse_approve,
        )
        from daemon import single_instance
    except Exception as e:
        logger.error(f"[deeplink] handler import failed: {e}")
        return False

    url = argv[1]

    # Approve / Deny / Always toast-button URLs.  These ALWAYS require
    # a running daemon (the future to resolve lives in that process's
    # toast._pending dict).  If no daemon is running, the click is a
    # no-op — exit silently.
    approve_payload = parse_approve(url)
    if approve_payload is not None:
        if single_instance.send_handoff(argv):
            logger.info(
                f"[deeplink] approval handed off to running daemon "
                f"(action={approve_payload['action']}); exiting"
            )
        else:
            logger.info(
                "[deeplink] approval URL received but no daemon running; "
                "ignoring"
            )
        return True

    # Configure / pairing URLs.
    try:
        payload = parse_configure(url)
    except ValueError as e:
        logger.error(f"[deeplink] bad URL {url!r}: {e}")
        sys.exit(2)
    if payload is None:
        logger.warning(f"[deeplink] unrecognised URL host in {url!r}; ignoring")
        return False

    # Try handoff first.  If a daemon is already running it picks up;
    # we exit clean.
    if single_instance.send_handoff(argv):
        logger.info("[deeplink] handed off to running daemon; exiting")
        return True

    # No running daemon — we ARE the first instance.  Apply config
    # synchronously so the WS loop sees the fresh server/token.
    try:
        apply_configure(payload)
    except ValueError as e:
        # User_id mismatch / unsafe server — refuse, print, exit.
        logger.error(f"[deeplink] config rejected: {e}")
        sys.exit(2)
    except Exception as e:
        logger.error(f"[deeplink] config write failed: {e}")
        sys.exit(2)
    return False


def main():
    # First responsibility: deal with deep-link argv before any
    # parser/setup runs.  If we hand off to a running instance, we
    # exit cleanly here without touching the rest of main().
    if _handle_deeplink_or_handoff(sys.argv):
        return

    parser = argparse.ArgumentParser(
        description="Veilguard Client Daemon — local tool execution for cloud Veilguard",
        epilog="Quick start: veilguard --setup wss://your-server/ws/client --token YOUR_TOKEN",
    )
    parser.add_argument("--version", action="version",
                        version=f"Veilguard Client {__version__}")
    parser.add_argument("--setup", metavar="SERVER_URL",
                        help="One-step setup: save config to ~/.veilguard/ and start daemon")
    parser.add_argument("--server", help="WebSocket server URL")
    parser.add_argument("--token", help="Auth token")
    parser.add_argument("--user-id", dest="user_id", help="LibreChat user ID (from QR code)")
    parser.add_argument("--client-id", dest="client_id", help="Client identifier")
    parser.add_argument("--project-root", dest="project_root", help="Project root directory")
    parser.add_argument("--config", help="Config file path (default: ~/.veilguard/config.yaml)")
    parser.add_argument("--no-auto-update", action="store_true",
                        help="Disable auto-update check (for development)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without tray UI / QR setup (server / Linux). "
                             "Loads identity from VEILGUARD_SERVER + VEILGUARD_TOKEN + "
                             "VEILGUARD_USER_ID env vars. Disables auto-update.")
    args = parser.parse_args()

    # Auto-headless on non-Windows when the daemon was launched without
    # a tty (running under systemd / docker / supervisord / k8s).  Lets
    # the same binary work as both Windows tray daemon and Linux sandbox
    # without an explicit flag in the unit file.
    if not args.headless and platform.system() != "Windows":
        if not sys.stdout.isatty() or os.environ.get("VEILGUARD_HEADLESS"):
            args.headless = True

    # Quick setup mode
    if args.setup:
        setup_and_run(
            server=args.setup,
            token=args.token or "",
            user_id=args.user_id or "",
            project_root=args.project_root or ".",
        )
        return

    # Normal mode — load config or run setup
    # Default config path: ~/.veilguard/config.yaml
    if not args.config:
        home_config = os.path.join(os.path.expanduser("~"), ".veilguard", "config.yaml")
        if os.path.exists(home_config):
            args.config = home_config

    config = load_config(args)

    if args.no_auto_update:
        config["auto_update"] = False

    # Pre-flight: do we have ANY pairing creds at all?  Multi-env
    # (0.4+) means there could be N environments declared — we just
    # need at least ONE complete + enabled to proceed.  If everything
    # is empty we fall through to the setup UI (or fail-hard in
    # headless mode).
    try:
        from daemon.env import parse_environments
        _envs = parse_environments(config)
        _has_any = any(e.enabled and e.is_complete() for e in _envs.values())
    except Exception:
        # Fallback to legacy single-env check if env module fails to load.
        _has_any = bool(config.get("token") and config.get("user_id"))

    if not _has_any:
        if config.get("headless"):
            logger.error(
                "[HEADLESS] No complete env supplied via "
                "VEILGUARD_TOKEN + VEILGUARD_USER_ID env vars (or via the "
                "environments block in config.yaml).  Sandbox container "
                "will exit (server cannot auth). Set "
                "VEILGUARD_USER_ID=system:sandbox and "
                "VEILGUARD_TOKEN=<shared secret> in your compose env."
            )
            sys.exit(2)
        # Interactive: kick the QR/paste setup so the user can pair
        # their first env.  Multi-env additions happen later via the
        # tray's "Add server" dialog + Set up button on LibreChat.
        config = _run_first_run_setup()

    # ── Single-instance handoff listener (Phase: onboarding) ─────────
    # Accepts ``veilguard://configure?…`` argv from second-instance
    # launches and re-applies config to the running daemon.  Skip on
    # headless containers — they don't get URL-scheme registrations
    # and re-pair is done via env vars.
    if not config.get("headless"):
        try:
            from daemon.single_instance import start_listener
            from daemon.deeplink import (
                parse_configure, apply_configure, parse_approve,
            )

            def _on_handoff(handoff_argv: list[str]) -> None:
                if len(handoff_argv) < 2 or not handoff_argv[1].startswith("veilguard://"):
                    logger.warning(
                        f"[HANDOFF] ignoring argv with no deep link: {handoff_argv[:3]}"
                    )
                    return
                # Approval URLs — toast-button clicks coming back via
                # the protocol scheme.  Resolve the pending future in
                # this process's toast._pending and we're done.
                try:
                    approve = parse_approve(handoff_argv[1])
                except Exception as e:
                    approve = None
                    logger.warning(f"[HANDOFF] parse_approve raised: {e}")
                if approve is not None:
                    try:
                        from daemon import toast as _toast
                        ok = _toast.resolve_pending(
                            approve["token"], action=approve["action"],
                        )
                        logger.info(
                            f"[HANDOFF] approval {approve['action']!r} "
                            f"token={approve['token'][:12]} ok={ok}"
                        )
                    except Exception as e:
                        logger.error(
                            f"[HANDOFF] toast.resolve_pending failed: {e}",
                            exc_info=True,
                        )
                    return

                # Otherwise it's a configure / pairing URL.
                try:
                    payload = parse_configure(handoff_argv[1])
                    if payload is None:
                        return
                    apply_configure(payload)
                    logger.info(
                        "[HANDOFF] config refreshed from deep link; "
                        "WS will pick up on next reconnect"
                    )
                    # Force a reconnect by closing the current ws.  The
                    # outer while-True loop in run_daemon catches the
                    # disconnect and reopens with the new creds.  We
                    # rely on the natural backoff (current_delay is
                    # tiny right after auth, so the reconnect is fast).
                    #
                    # Done without a direct ws reference because the
                    # handoff thread doesn't own one — instead we mark
                    # the lock-file with a sentinel that the heartbeat
                    # checks and short-circuits, OR simply rely on the
                    # disk config being re-read on next reconnect.
                    # For now: log + trust the user to be patient or
                    # to bounce the daemon from the tray.  Hot-reload
                    # of WS creds is a Phase F follow-up.
                except Exception as e:
                    logger.error(f"[HANDOFF] could not re-apply config: {e}")

            _handoff_listener = start_listener(_on_handoff)
            if _handoff_listener is not None:
                logger.info(
                    "[HANDOFF] listening for veilguard:// handoff messages"
                )
        except Exception as e:
            logger.warning(f"[HANDOFF] could not start listener: {e}")

    # ── Outgoing notification queue (daemon → cloud) ─────────────────
    # Tray callbacks run on the pystray thread; the WS lives on the
    # asyncio loop in another thread.  We bridge them with a sync
    # ``queue.Queue`` — tray .put_nowait(), drainer task .get() with
    # short timeout.  When the WS is disconnected the drainer holds
    # items until reconnect, with a soft cap to avoid unbounded growth.
    import queue as _stdq
    _outgoing_queue: "_stdq.Queue[dict]" = _stdq.Queue(maxsize=64)
    config["_outgoing_queue"] = _outgoing_queue

    # ── Spawn Windows tray (Phase C) ──────────────────────────────────
    # Tray lives for the whole process; survives WS reconnects.  Imports
    # are conditional — headless / non-Windows / missing-dep builds skip.
    _tray = None
    if not config.get("headless") and platform.system() == "Windows":
        try:
            from daemon.tray import TrayController, HAS_TRAY_UI
            if HAS_TRAY_UI:
                def _on_set_level(level: str, scope: str) -> None:
                    """Tray menu → broadcast permission change to ALL envs.

                    Multi-env (0.4+): we fan the JSON-RPC notification
                    out to every per-env outgoing queue so a level
                    change from the tray applies to both prod + local
                    pairings at once.  Each env's drainer picks it up
                    and sends over its own WS.
                    """
                    payload = {
                        "jsonrpc": "2.0",
                        "method": "set_permission_level",
                        "params": {
                            "level": level,
                            "scope": scope,
                            "source": "tray",
                        },
                    }
                    targets = list(
                        (config.get("_outgoing_queues") or {}).items()
                    ) or [("default", _outgoing_queue)]
                    for env_name, q in targets:
                        try:
                            q.put_nowait(dict(payload))
                            logger.info(
                                f"[TRAY] queued set_permission_level "
                                f"env={env_name} level={level} scope={scope}"
                            )
                        except _stdq.Full:
                            try:
                                q.get_nowait()
                                q.put_nowait(dict(payload))
                            except Exception:
                                logger.warning(
                                    f"[TRAY] outgoing queue full for env={env_name}; "
                                    f"dropped set_permission_level level={level}"
                                )

                def _on_quit() -> None:
                    logger.info("[TRAY] user clicked Quit")
                    os._exit(0)

                _tray = TrayController(
                    on_quit=_on_quit,
                    on_set_level=_on_set_level,
                )
                _tray.start()
                # Stash on config so run_daemon can flip
                # set_connection_state(True/False) at the right
                # lifecycle points (auth-success / disconnect).
                config["_tray"] = _tray
                logger.info("[TRAY] system tray icon active")
        except Exception as e:
            import traceback
            logger.warning(
                f"[TRAY] failed to start: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )

    # Recovery loop. ``run_daemon`` raises CredentialsRevokedError when
    # the cloud rejects our token (e.g. after a security rotation). On
    # that, we wipe the bad creds, run setup so the user can paste a
    # fresh QR-blob, and restart the daemon. Any other exception (or
    # clean return) drops out of the loop. Network-level reconnects are
    # already handled inside run_daemon's own while-True.
    while True:
        print(f"""
    Veilguard Client Daemon
    Server:       {config['server']}
    Client ID:    {config.get('client_id', 'unknown')}
    Project Root: {os.path.realpath(config.get('project_root', '.'))}
    """)
        try:
            asyncio.run(run_daemon(config))
            break
        except CredentialsRevokedError as exc:
            print(f"""
    [REPAIR] The Veilguard server rejected your access token: {exc}
    [REPAIR] Wiping the stored token from {args.config or '~/.veilguard/config.yaml'}.
    [REPAIR] Opening the setup page so you can paste a fresh QR-blob from
    [REPAIR] the LibreChat 'Workspace' panel.
            """)
            _wipe_credentials_in_config(args.config)
            config = _run_first_run_setup()
            # Loop back and re-enter run_daemon with the new config.


def _wipe_credentials_in_config(config_path: str) -> None:
    """Strip ``token`` and ``user_id`` from the on-disk config so a
    subsequent ``load_config`` returns empties and the setup UI fires.

    Falls back to deleting the file if YAML parsing or rewrite fails —
    a missing file is the strongest possible signal of "first run" and
    setup_server handles it identically.
    """
    if not config_path:
        config_path = os.path.join(
            os.path.expanduser("~"), ".veilguard", "config.yaml"
        )
    if not os.path.exists(config_path):
        return
    try:
        if yaml:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f) or {}
            data.pop("token", None)
            data.pop("user_id", None)
            with open(config_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            # No yaml -> rewrite as JSON-style yaml minus the bad keys.
            with open(config_path, "r") as f:
                lines = f.readlines()
            kept = [
                ln for ln in lines
                if not ln.startswith("token:") and not ln.startswith("user_id:")
            ]
            with open(config_path, "w") as f:
                f.writelines(kept)
    except Exception as e:
        logger.warning(
            f"Could not rewrite config to drop creds ({e}); deleting it instead"
        )
        try:
            os.remove(config_path)
        except Exception:
            pass


def _run_first_run_setup() -> dict:
    """Spin up the local setup HTTP server and block until the user
    pastes a fresh QR-blob. Returns the new config dict.

    Used both for genuine first-runs (no config.yaml on disk) and for
    post-rotation recovery (CredentialsRevokedError caught above).

    Threading model: setup_server runs in a worker thread. When the
    user POSTs the new credentials, our on_setup callback runs in
    that thread and needs to wake the main thread up. Using an
    ``asyncio.Event`` for this was a 0.2.2 bug -- on Python 3.12
    ``asyncio.get_event_loop()`` from a non-main thread raises, so
    the signal silently dropped and main blocked forever even though
    config.yaml had been written. ``threading.Event`` is the
    cross-thread-safe primitive; we don't need asyncio for a flow
    that's just "block until callback fires."
    """
    print("""
    Veilguard Client Daemon — Pairing Setup
    Opening setup page in your browser at http://localhost:9090/

    1. Open https://veilguard.phishield.com/ in another tab and log in.
    2. Click the 'Workspace' side-panel.
    3. Click the grey 'Click to copy' connection-string button.
    4. Paste it into the setup page above.
    """)
    import threading
    from setup_server import run_setup_server, open_setup_page

    setup_done = threading.Event()
    setup_config: dict = {}

    def on_setup(cfg):
        nonlocal setup_config
        setup_config = cfg
        setup_done.set()

    run_setup_server(on_complete=on_setup)
    open_setup_page()
    setup_done.wait()
    return setup_config


if __name__ == "__main__":
    main()
