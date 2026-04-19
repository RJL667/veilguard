#!/usr/bin/env python3
"""
Veilguard Client Daemon — Local tool execution for cloud-hosted Veilguard.

Connects to the cloud sub-agents server via WebSocket. Receives tool execution
requests (file ops, commands, searches), runs them locally, returns results.

Usage:
    python veilguard_client.py                          # Uses config.yaml
    python veilguard_client.py --server ws://host:8809/ws/client --token abc123
"""

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote_plus

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
        """Dispatch tool execution. Returns result string."""
        handler = getattr(self, f"_tool_{tool}", None)
        if handler is None:
            return f"Error: Unknown tool '{tool}'"
        try:
            result = handler(args) if not asyncio.iscoroutinefunction(handler) else await handler(args)
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
        try:
            result = subprocess.run(grep_args, capture_output=True, text=True, timeout=15, cwd=self.project_root)
            output = result.stdout[:3000]
        except FileNotFoundError:
            search_dir = os.path.join(self.project_root, search_path)
            result = subprocess.run(
                ["findstr", "/S", "/N", "/R", pattern, os.path.join(search_dir, "*.*")],
                capture_output=True, text=True, timeout=15, cwd=self.project_root
            )
            output = result.stdout[:3000]
        return output or "(no matches)"

    def _tool_run_command(self, args: dict) -> str:
        cmd = args.get("command", "")
        safe, reason = validate_command(cmd)
        if not safe:
            return reason
        if os.name == "nt":
            has_pipe = "|" in cmd or ">" in cmd or "&&" in cmd
            if has_pipe:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=self.project_root)
            else:
                result = subprocess.run(["cmd", "/c", cmd], capture_output=True, text=True, timeout=30, cwd=self.project_root)
        else:
            result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=30, cwd=self.project_root)
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
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True, text=True, timeout=60, cwd=self.project_root
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
        result = subprocess.run(
            f"docker {cmd}", shell=True,
            capture_output=True, text=True, timeout=120, cwd=self.project_root
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
        result = subprocess.run(
            f"git {cmd}", shell=True,
            capture_output=True, text=True, timeout=30, cwd=self.project_root
        )
        out = result.stdout[:2000]
        if result.stderr:
            out += f"\nstderr: {result.stderr[:500]}"
        return out or "(no output)"

    def _tool_host_file_read(self, args: dict) -> str:
        return self._tool_read_file(args)

    def _tool_host_file_write(self, args: dict) -> str:
        return self._tool_write_file(args)


# ── WebSocket Client ─────────────────────────────────────────────────────────

async def run_daemon(config: dict):
    """Main daemon loop — connect, authenticate, handle tool requests."""
    server = config["server"]
    token = config.get("token", "")
    client_id = config.get("client_id", "veilguard-client")
    project_root = os.path.realpath(config.get("project_root", "."))

    executor = ToolExecutor(project_root)
    reconnect_delay = config.get("reconnect_delay", 5)
    max_delay = config.get("max_reconnect_delay", 300)
    current_delay = reconnect_delay

    logger.info(f"Project root: {project_root}")
    logger.info(f"Server: {server}")

    while True:
        try:
            logger.info(f"Connecting to {server}...")
            async with websockets.connect(server, ping_interval=None) as ws:
                # Authenticate
                await ws.send(json.dumps({
                    "jsonrpc": "2.0",
                    "method": "auth",
                    "params": {"token": token, "client_id": client_id},
                }))

                resp = json.loads(await ws.recv())
                if "error" in resp:
                    logger.error(f"Auth failed: {resp['error']}")
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
    if args.client_id:
        config["client_id"] = args.client_id
    if args.project_root:
        config["project_root"] = args.project_root

    return config


def setup_and_run(server: str, token: str, project_root: str = "."):
    """One-liner setup: save config and start daemon immediately.

    Called via:
        pip install veilguard-client && veilguard --setup wss://server/ws/client --token abc123
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
    parser.add_argument("--setup", metavar="SERVER_URL",
                        help="One-step setup: save config to ~/.veilguard/ and start daemon")
    parser.add_argument("--server", help="WebSocket server URL")
    parser.add_argument("--token", help="Auth token")
    parser.add_argument("--client-id", dest="client_id", help="Client identifier")
    parser.add_argument("--project-root", dest="project_root", help="Project root directory")
    parser.add_argument("--config", help="Config file path (default: ~/.veilguard/config.yaml)")
    args = parser.parse_args()

    # Quick setup mode
    if args.setup:
        setup_and_run(
            server=args.setup,
            token=args.token or "",
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

    # If no token configured (and no CLI override), launch setup UI
    if not config.get("token"):
        print("""
    Veilguard Client Daemon — First Run Setup
    Opening setup page in your browser...
    Scan the QR code from LibreChat to connect.
        """)
        from setup_server import run_setup_server, open_setup_page

        setup_done = asyncio.Event()
        setup_config = {}

        def on_setup(cfg):
            nonlocal setup_config
            setup_config = cfg
            # Signal the main thread
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(setup_done.set)

        run_setup_server(on_complete=on_setup)
        open_setup_page()

        # Wait for setup to complete
        async def wait_for_setup():
            await setup_done.wait()
            return setup_config

        config = asyncio.run(wait_for_setup())

    print(f"""
    Veilguard Client Daemon
    Server:       {config['server']}
    Client ID:    {config.get('client_id', 'unknown')}
    Project Root: {os.path.realpath(config.get('project_root', '.'))}
    """)

    asyncio.run(run_daemon(config))


if __name__ == "__main__":
    main()
