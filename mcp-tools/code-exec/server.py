"""Code Execution MCP Tool Server for Veilguard.

Tools: execute_python, execute_bash
"""

import os
import subprocess
import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("code-exec", instructions="Code execution tools for running Python scripts and shell commands.")

TIMEOUT = int(os.environ.get("EXEC_TIMEOUT", "30"))
WORKSPACE = os.environ.get("WORKSPACE_ROOT", "/workspace")


@mcp.tool()
def execute_python(code: str, timeout: int = 30) -> str:
    """Execute Python code and return the output.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds (default 30)
    """
    timeout = min(timeout, TIMEOUT)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(code)
        f.flush()
        script_path = f.name

    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=WORKSPACE,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr}" if output else result.stderr

        if result.returncode != 0:
            output += f"\n[Exit code: {result.returncode}]"

        return output.strip() or "(no output)"

    except subprocess.TimeoutExpired:
        return f"Error: Execution timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing Python: {e}"
    finally:
        os.unlink(script_path)


@mcp.tool()
def execute_bash(command: str, timeout: int = 30) -> str:
    """Execute a shell command and return the output.

    Args:
        command: Shell command to execute
        timeout: Maximum execution time in seconds (default 30)
    """
    timeout = min(timeout, TIMEOUT)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=WORKSPACE,
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr}" if output else result.stderr

        if result.returncode != 0:
            output += f"\n[Exit code: {result.returncode}]"

        return output.strip() or "(no output)"

    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing command: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
