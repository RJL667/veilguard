"""Host Execution MCP Tool Server for Veilguard.

Runs ON THE WINDOWS HOST (not inside Docker).
Provides CMD, PowerShell, and Docker control from LibreChat chat.

Start: python mcp-tools/host-exec/server.py
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
    args = ["docker"] + command.split()
    return _run(args, timeout=timeout, cwd=WORK_DIR)


@mcp.tool()
def run_git(command: str, working_dir: str = "") -> str:
    """Run a git command on the host.

    Args:
        command: Git command WITHOUT the "git" prefix.
                 Examples: "status", "diff", "log --oneline -10", "add -A", "commit -m 'message'"
        working_dir: Repository path. Empty = veilguard project root.
    """
    cwd = working_dir if working_dir else WORK_DIR
    args = ["git"] + command.split()
    return _run(args, timeout=30, cwd=cwd)


@mcp.tool()
def host_file_read(path: str) -> str:
    """Read a file from the Windows host filesystem.

    Args:
        path: Absolute or relative path on the host (e.g. "C:\\Users\\rudol\\file.txt" or ".env")
    """
    p = Path(path)
    if not p.is_absolute():
        p = Path(WORK_DIR) / p

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
    p = Path(path)
    if not p.is_absolute():
        p = Path(WORK_DIR) / p

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} bytes to {p}"
    except Exception as e:
        return f"Error writing file: {e}"


if __name__ == "__main__":
    # This server runs on the host, not in Docker
    # Start with: python mcp-tools/host-exec/server.py
    # Or via SSE: python mcp-tools/host-exec/server.py --sse --port 8808
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

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as (read, write):
                await mcp._mcp_server.run(
                    read, write, mcp._mcp_server.create_initialization_options()
                )

        app = Starlette(
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        print(f"Starting host-exec MCP server on http://0.0.0.0:{port}/sse")
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        mcp.run(transport="stdio")
