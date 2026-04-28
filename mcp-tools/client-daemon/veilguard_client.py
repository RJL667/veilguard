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
__version__ = "0.2.2"


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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [veilguard-client] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("veilguard-client")


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
        selected = lines[offset:offset + limit]
        return "".join(f"{offset + i + 1}\t{l}" for i, l in enumerate(selected))

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
        pattern = args.get("pattern", "*")
        search_path = args.get("path", self.project_root)
        if not os.path.isabs(search_path):
            search_path = os.path.join(self.project_root, search_path)
        matches = sorted(Path(search_path).glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)[:50]
        root = self.project_root
        return "\n".join(
            str(m.relative_to(root)) if str(m).startswith(root) else str(m) for m in matches
        ) or "(no matches)"

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
    """Main daemon loop — connect, authenticate, handle tool requests."""
    server = config["server"]
    token = config.get("token", "")
    # user_id is REQUIRED by the server for per-user token validation.
    # setup_server.save_config() writes it to ~/.veilguard/config.yaml;
    # if it's missing the user needs to re-copy their QR code from
    # LibreChat's "Connect Client" panel.
    user_id = config.get("user_id", "")
    client_id = config.get("client_id", "veilguard-client")
    project_root = os.path.realpath(config.get("project_root", "."))

    executor = ToolExecutor(project_root)
    reconnect_delay = config.get("reconnect_delay", 5)
    max_delay = config.get("max_reconnect_delay", 300)
    current_delay = reconnect_delay

    logger.info(f"Veilguard Client v{__version__}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Server: {server}")
    if not user_id:
        logger.warning(
            "[AUTH] No user_id in config — server will reject auth. "
            "Re-pair via LibreChat's 'Connect Client' panel, or add "
            "'user_id: <your-lc-user-id>' to ~/.veilguard/config.yaml."
        )
    else:
        logger.info(f"[AUTH] user_id: {user_id[:8]}...{user_id[-4:]}")

    # Auto-updater runs for the lifetime of the process, independent of
    # WebSocket reconnects. If it triggers an update it calls os._exit(0)
    # so Inno Setup can overwrite files.
    if config.get("auto_update", True):
        asyncio.create_task(
            auto_updater(server, config.get("update_manifest_url", ""))
        )

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
                        "user_id": user_id,
                        "token": token,
                        "client_id": client_id,
                        "version": __version__,
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

                # Start heartbeat
                async def heartbeat():
                    while True:
                        await asyncio.sleep(30)
                        try:
                            await ws.send(json.dumps({"jsonrpc": "2.0", "method": "ping"}))
                        except Exception:
                            break

                hb_task = asyncio.create_task(heartbeat())

                # Message loop — tool calls run in parallel via asyncio tasks
                active_tasks = set()

                async def run_tool(ws, executor, req_id, tool, args):
                    """Execute a tool and send the result back over WebSocket."""
                    logger.info(f"[TOOL] {tool}({list(args.keys())}) id={req_id}")
                    start = time.time()
                    try:
                        result = await executor.execute(tool, args)
                    except Exception as e:
                        result = f"Error: {e}"
                    elapsed = time.time() - start

                    if len(result) > 50000:
                        result = result[:50000] + "\n... [truncated]"

                    logger.info(f"[TOOL] {tool} done in {elapsed:.1f}s ({len(result)} chars)")

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

        except (websockets.ConnectionClosed, ConnectionRefusedError, OSError) as e:
            logger.warning(f"Disconnected: {e}. Reconnecting in {current_delay}s...")
            await asyncio.sleep(current_delay)
            current_delay = min(current_delay * 2, max_delay)
        except Exception as e:
            logger.error(f"Unexpected error: {e}. Reconnecting in {current_delay}s...")
            await asyncio.sleep(current_delay)
            current_delay = min(current_delay * 2, max_delay)


# ── Entry Point ──────────────────────────────────────────────────────────────

def load_config(args) -> dict:
    """Load config from yaml file, overridden by CLI args."""
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


def main():
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
    args = parser.parse_args()

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

    # If no token / no user_id configured (or one was just wiped after
    # CredentialsRevokedError), launch setup UI before entering the
    # daemon loop. After the user pastes the new QR-blob, the setup
    # callback returns the fresh config dict.
    if not config.get("token") or not config.get("user_id"):
        config = _run_first_run_setup()

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
    """
    print("""
    Veilguard Client Daemon — Pairing Setup
    Opening setup page in your browser at http://localhost:9090/

    1. Open https://veilguard.phishield.com/ in another tab and log in.
    2. Click the 'Workspace' side-panel.
    3. Click the grey 'Click to copy' connection-string button.
    4. Paste it into the setup page above.
    """)
    from setup_server import run_setup_server, open_setup_page

    setup_done = asyncio.Event()
    setup_config: dict = {}

    def on_setup(cfg):
        nonlocal setup_config
        setup_config = cfg
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(setup_done.set)

    run_setup_server(on_complete=on_setup)
    open_setup_page()

    async def wait_for_setup():
        await setup_done.wait()
        return setup_config

    return asyncio.run(wait_for_setup())


if __name__ == "__main__":
    main()
