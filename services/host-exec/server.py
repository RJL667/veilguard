"""Host Execution MCP Tool Server for Veilguard.

Runs ON THE WINDOWS HOST (not inside Docker).
Provides CMD, PowerShell, and Docker control from LibreChat chat.

Start: python services/host-exec/server.py
"""

import os
import subprocess
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "host-exec",
    instructions=(
        "Host execution tools for running Windows CMD, PowerShell, and Docker commands "
        "on the host machine. Use with caution — these run with full host privileges."
    ),
)

# Working directory for commands — defaults to the veilguard project root
WORK_DIR = os.environ.get("HOST_WORK_DIR", str(Path(__file__).parent.parent.parent))
TIMEOUT = int(os.environ.get("HOST_EXEC_TIMEOUT", "60"))

# [HOST_DOC_READ_2026_06_01] Read-only document tools (read_pdf/word/excel)
# may read the user's own files anywhere under their home directory — host-
# exec runs ON the user's Windows machine, so reading
# 'C:\\Users\\<user>\\Downloads\\report.pdf' is exactly the intended use.
# WRITES stay scoped to WORK_DIR via _safe_resolve; only READ-parse tools
# get this wider root. Override with HOST_DOC_READ_ROOT to narrow/widen.
_DOC_READ_ROOT = os.environ.get("HOST_DOC_READ_ROOT", os.path.expanduser("~"))

# ── Command Safety Validation ────────────────────────────────────────────────
import re as _re

_DANGEROUS_PATTERNS = [
    (r"rm\s+-rf\s+/", "Recursive delete from root"),
    (r"format\s+[a-z]:", "Format disk drive"),
    (r"del\s+/[sfq].*\\windows", "Delete Windows system files"),
    (r"rmdir\s+/s\s+/q\s+[a-z]:\\$", "Remove entire drive"),
    (r"reg\s+delete\s+hklm", "Delete system registry keys"),
    (r"net\s+user\s+.*\s+/delete", "Delete user account"),
    (r"cipher\s+/w:", "Secure wipe disk"),
    (r"shutdown\s+/[srf]", "Shutdown/restart system"),
    (r"bcdedit", "Modify boot configuration"),
    (r"diskpart", "Disk partition tool"),
    (r"schtasks\s+/delete\s+/tn\s+\\", "Delete system scheduled tasks"),
    (r"wmic\s+os\s+.*delete", "WMI destructive operation"),
]
_DANGEROUS_RE = [(_re.compile(p, _re.IGNORECASE), desc) for p, desc in _DANGEROUS_PATTERNS]


def _validate_command(cmd: str) -> tuple[bool, str]:
    """Check command against dangerous patterns. Returns (safe, reason)."""
    for pattern, desc in _DANGEROUS_RE:
        if pattern.search(cmd):
            return False, f"BLOCKED: {desc} — pattern matched in: {cmd[:100]}"
    return True, ""


def _run(args: list[str], timeout: int = TIMEOUT, cwd: str = WORK_DIR, shell: bool = False) -> str:
    """Run a subprocess and return formatted output."""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            shell=shell,
        )
        output = ""
        if result.stdout.strip():
            output += result.stdout.strip()
        if result.stderr.strip():
            if output:
                output += "\n\n--- stderr ---\n"
            output += result.stderr.strip()
        if not output:
            output = "(no output)"

        status = "OK" if result.returncode == 0 else f"EXIT CODE {result.returncode}"
        return f"[{status}]\n{output}"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except FileNotFoundError as e:
        return f"Error: Command not found: {e}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def run_cmd(command: str, timeout: int = 60, working_dir: str = "") -> str:
    """Run a Windows CMD command on the host machine.

    Args:
        command: CMD command to execute (e.g. "dir", "ipconfig", "type file.txt")
        timeout: Max seconds to wait (default 60)
        working_dir: Working directory. Empty = veilguard project root.
    """
    safe, reason = _validate_command(command)
    if not safe:
        return f"Error: {reason}"
    cwd = working_dir if working_dir else WORK_DIR
    return _run(["cmd", "/c", command], timeout=timeout, cwd=cwd)


@mcp.tool()
def run_powershell(script: str, timeout: int = 60, working_dir: str = "") -> str:
    """Run a PowerShell script/command on the host machine.

    Args:
        script: PowerShell script or command (e.g. "Get-Process", "Get-ChildItem -Recurse *.py")
        timeout: Max seconds to wait (default 60)
        working_dir: Working directory. Empty = veilguard project root.
    """
    safe, reason = _validate_command(script)
    if not safe:
        return f"Error: {reason}"
    cwd = working_dir if working_dir else WORK_DIR
    return _run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        timeout=timeout,
        cwd=cwd,
    )


@mcp.tool()
def run_docker(command: str, timeout: int = 120) -> str:
    """Run a docker or docker compose command on the host.

    Args:
        command: Docker command WITHOUT the "docker" prefix.
                 Examples: "ps", "compose logs api --tail 20", "compose restart api"
        timeout: Max seconds to wait (default 120)
    """
    safe, reason = _validate_command(f"docker {command}")
    if not safe:
        return f"Error: {reason}"
    args = ["docker"] + command.split()
    return _run(args, timeout=timeout, cwd=WORK_DIR)


_GIT_DANGEROUS = _re.compile(r"(push\s+--force|reset\s+--hard|clean\s+-[fd])", _re.IGNORECASE)


@mcp.tool()
def run_git(command: str, working_dir: str = "") -> str:
    """Run a git command on the host.

    Args:
        command: Git command WITHOUT the "git" prefix.
                 Examples: "status", "diff", "log --oneline -10", "add -A", "commit -m 'message'"
        working_dir: Repository path. Empty = veilguard project root.
    """
    if _GIT_DANGEROUS.search(command):
        return f"Error: BLOCKED — destructive git operation: {command[:50]}"
    cwd = working_dir if working_dir else WORK_DIR
    args = ["git"] + command.split()
    return _run(args, timeout=30, cwd=cwd)


def _safe_resolve(path: str) -> tuple[Path, str]:
    """Resolve path and check for traversal attacks. Returns (resolved_path, error_or_empty)."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(WORK_DIR) / p
    resolved = p.resolve()
    work_resolved = Path(WORK_DIR).resolve()
    if not str(resolved).startswith(str(work_resolved)):
        return resolved, f"Error: Path traversal denied — {path} resolves outside project root"
    return resolved, ""


@mcp.tool()
def host_file_read(path: str) -> str:
    """Read a file from the Windows host filesystem.

    Args:
        path: Absolute or relative path on the host (e.g. "C:\\Users\\rudol\\file.txt" or ".env")
    """
    p, err = _safe_resolve(path)
    if err:
        return err

    if not p.exists():
        return f"Error: File not found: {p}"
    if not p.is_file():
        return f"Error: Not a file: {p}"

    try:
        content = p.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        numbered = [f"{i + 1:>6}  {line}" for i, line in enumerate(lines)]
        return f"# {p} ({len(lines)} lines)\n" + "\n".join(numbered)
    except Exception as e:
        return f"Error reading file: {e}"


@mcp.tool()
def host_file_write(path: str, content: str) -> str:
    """Write a file to the Windows host filesystem.

    Args:
        path: Absolute or relative path on the host
        content: Content to write
    """
    p, err = _safe_resolve(path)
    if err:
        return err

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} bytes to {p}"
    except Exception as e:
        return f"Error writing file: {e}"


def _safe_read_doc_path(path: str) -> tuple[Path, str]:
    """Resolve a path for READ-ONLY document parsers (read_pdf/word/excel).

    Allows the project root OR the user's home dir (_DOC_READ_ROOT) so a
    Windows path like C:\\Users\\rudol\\Downloads\\x.pdf — which the daemon
    can already see — is readable here too, on the same host. Writes still
    use the tighter _safe_resolve (WORK_DIR only).
    """
    p = Path(path)
    if not p.is_absolute():
        p = Path(WORK_DIR) / p
    resolved = p.resolve()
    roots = [Path(WORK_DIR).resolve(), Path(_DOC_READ_ROOT).resolve()]
    if not any(str(resolved).startswith(str(r)) for r in roots):
        return resolved, (
            f"Error: read denied — {path!r} is outside the allowed roots "
            f"({WORK_DIR} and {_DOC_READ_ROOT}). Set HOST_DOC_READ_ROOT to widen."
        )
    return resolved, ""


def _extract_pdf_text(p: Path, pages: str) -> str:
    import fitz  # PyMuPDF — available in the host Python (1.26.x)
    doc = fitz.open(str(p))
    try:
        total = len(doc)
        if pages.strip():
            parts = pages.strip().split("-")
            start = max(0, int(parts[0]) - 1)
            end = int(parts[-1]) if len(parts) > 1 else start + 1
            rng = range(start, min(end, total))
        else:
            rng = range(total)
        out = [f"# {p.name} ({total} pages, reading {len(rng)})\n"]
        for i in rng:
            out.append(f"--- Page {i + 1} ---\n{doc[i].get_text().strip()}\n")
        return "\n".join(out)
    finally:
        doc.close()


@mcp.tool()
def host_read_pdf(path: str, pages: str = "") -> str:
    """Read and extract text from a PDF on the user's WINDOWS host.

    Use THIS — not the documents server's read_pdf — for PDFs on the user's
    machine (e.g. 'C:\\Users\\rudol\\Downloads\\report.pdf'). The documents
    server runs in a Linux container and cannot see Windows paths; host-exec
    runs ON the host and has a real PDF parser (PyMuPDF). Reads under the
    project root or the user's home directory.

    Args:
        path: PDF path on the host (absolute Windows path, or relative to the
              project root). Forward or back slashes both work.
        pages: Page range like "1-5" or "3". Empty = all pages.
    """
    p, err = _safe_read_doc_path(path)
    if err:
        return err
    if not p.exists():
        return f"Error: File not found: {p}"
    if not p.is_file():
        return f"Error: Not a file: {p}"
    try:
        return _extract_pdf_text(p, pages)
    except Exception as e:
        return f"Error reading PDF: {e}"


if __name__ == "__main__":
    # This server runs on the host, not in Docker
    # Start with: python services/host-exec/server.py
    # Or via SSE: python services/host-exec/server.py --sse --port 8808
    if "--sse" in sys.argv:
        port = 8808
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        import uvicorn
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route

        sse = SseServerTransport("/messages/")

        # [SSE_NONE_RETURN_FIX 2026-06-03] Starlette 0.52+ requires route
        # handlers to return a Response (dispatcher does
        # `response = await f(request); await response(scope, receive, send)`).
        # connect_sse() streams itself, so returning None crashes with
        # "TypeError: 'NoneType' object is not callable" on teardown. Return a
        # Response shim that owns the SSE lifecycle. See sub-agents/server.py.
        class _SseResponse:
            """Minimal Response shim that owns the SSE stream lifecycle."""
            async def __call__(self, scope, receive, send):
                async with sse.connect_sse(scope, receive, send) as (read, write):
                    await mcp._mcp_server.run(
                        read, write, mcp._mcp_server.create_initialization_options()
                    )

        async def handle_sse(request):
            return _SseResponse()

        # Webhook auth token — set via WEBHOOK_TOKEN env var or defaults to random
        import secrets as _secrets
        _WEBHOOK_TOKEN = os.environ.get("WEBHOOK_TOKEN", _secrets.token_hex(16))
        if not os.environ.get("WEBHOOK_TOKEN"):
            print(f"  Webhook token (auto-generated): {_WEBHOOK_TOKEN}")

        async def handle_webhook(request):
            """External webhook endpoint — forwards to sub-agents trigger.
            Requires Authorization: Bearer <WEBHOOK_TOKEN> header."""
            from starlette.responses import JSONResponse

            # Auth check
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[7:] != _WEBHOOK_TOKEN:
                return JSONResponse({"error": "Unauthorized — set Authorization: Bearer <token>"}, status_code=401)

            import httpx as _httpx
            name = request.path_params.get("name", "")
            # Validate name — alphanumeric and hyphens only
            if not _re.match(r'^[\w\-]+$', name):
                return JSONResponse({"error": "Invalid trigger name"}, status_code=400)

            body = {}
            try:
                body = await request.json()
            except Exception:
                pass

            # Forward to sub-agents server
            sub_agents_url = f"http://localhost:8809/trigger/{name}"
            try:
                async with _httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(sub_agents_url, json=body)
                    return JSONResponse(resp.json(), status_code=resp.status_code)
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=502)

        app = Starlette(
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
                Route("/webhook/{name}", endpoint=handle_webhook, methods=["POST"]),
            ],
        )

        print(f"Starting host-exec MCP server on http://0.0.0.0:{port}/sse")
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        mcp.run(transport="stdio")
