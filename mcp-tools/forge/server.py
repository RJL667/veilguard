"""Tool Forge MCP Server for Veilguard.

Create, edit, run, and manage custom tools on the fly.
Tools are Python functions stored in the forge directory.
The AI can create new tools when it encounters a task with no existing tool.

Runs ON THE WINDOWS HOST.
Start: python mcp-tools/forge/server.py --sse --port 8810
"""

import importlib.util
import json
import os
import sys
import textwrap
import time
import traceback
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "forge",
    instructions=(
        "Tool Forge — create, edit, and run custom tools on the fly. "
        "When you need a capability that doesn't exist, forge a new tool. "
        "Forged tools persist across sessions and can be reused."
    ),
)

# Forge directory — where custom tools are stored
FORGE_DIR = Path(__file__).parent / "tools"
FORGE_DIR.mkdir(exist_ok=True)

# Tool registry — metadata about forged tools
REGISTRY_FILE = FORGE_DIR / "_registry.json"


def _load_registry() -> dict:
    """Load the tool registry from disk."""
    if REGISTRY_FILE.exists():
        try:
            return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_registry(registry: dict):
    """Save the tool registry to disk."""
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2, default=str), encoding="utf-8")


def _tool_path(name: str) -> Path:
    """Get the file path for a tool."""
    safe_name = name.lower().replace(" ", "_").replace("-", "_")
    # Only allow alphanumeric and underscore
    safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
    return FORGE_DIR / f"{safe_name}.py"


@mcp.tool()
def forge_tool(name: str, description: str, code: str, args_schema: str = "") -> str:
    """Create a new custom tool. The tool is saved to disk and immediately usable via run_tool.

    Write the tool as a Python function called `run` that takes keyword arguments and returns a string.
    You have access to standard library + any packages installed on the host.

    Example:
        forge_tool(
            name="domain_age",
            description="Check how old a domain is using WHOIS",
            code='''
import subprocess
def run(domain: str) -> str:
    result = subprocess.run(["nslookup", domain], capture_output=True, text=True, timeout=10)
    return result.stdout or result.stderr
''',
            args_schema='{"domain": "string"}'
        )

    Args:
        name: Tool name (lowercase, underscores). E.g. "domain_age", "hash_check"
        description: What the tool does (shown when listing tools)
        code: Python code. Must define a `run(**kwargs) -> str` function.
        args_schema: Optional JSON describing arguments: {"arg_name": "type_or_description", ...}
    """
    tool_file = _tool_path(name)
    safe_name = tool_file.stem

    # Validate: must contain def run
    if "def run" not in code:
        return "Error: Code must define a `run` function. Example: def run(domain: str) -> str:"

    # Validate: try to compile
    try:
        compile(code, f"forge/{safe_name}.py", "exec")
    except SyntaxError as e:
        return f"Error: Syntax error in code: {e}"

    # Save code
    tool_file.write_text(code, encoding="utf-8")

    # Parse args schema
    schema = {}
    if args_schema:
        try:
            schema = json.loads(args_schema)
        except json.JSONDecodeError:
            pass

    # Update registry
    registry = _load_registry()
    registry[safe_name] = {
        "name": safe_name,
        "description": description,
        "args": schema,
        "created_at": time.time(),
        "updated_at": time.time(),
        "run_count": 0,
        "last_error": None,
    }
    _save_registry(registry)

    args_list = ", ".join(f"{k}: {v}" for k, v in schema.items()) if schema else "none"
    return (
        f"Forged tool `{safe_name}`\n"
        f"Description: {description}\n"
        f"Args: {args_list}\n"
        f"File: {tool_file}\n\n"
        f"Use: `run_tool(\"{safe_name}\", ...)` to execute."
    )


@mcp.tool()
def run_tool(name: str, args: str = "{}") -> str:
    """Run a forged custom tool.

    Args:
        name: Tool name (as created by forge_tool)
        args: JSON string of arguments to pass. E.g. '{"domain": "example.com"}'
    """
    safe_name = _tool_path(name).stem
    tool_file = _tool_path(name)

    if not tool_file.exists():
        available = [f.stem for f in FORGE_DIR.glob("*.py") if not f.stem.startswith("_")]
        return f"Error: Tool '{safe_name}' not found. Available: {available}"

    # Parse args
    try:
        kwargs = json.loads(args)
    except json.JSONDecodeError:
        return "Error: 'args' must be valid JSON"

    # Load and execute
    try:
        spec = importlib.util.spec_from_file_location(f"forge.{safe_name}", tool_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "run"):
            return f"Error: Tool '{safe_name}' has no `run` function"

        result = module.run(**kwargs)

        # Update stats
        registry = _load_registry()
        if safe_name in registry:
            registry[safe_name]["run_count"] = registry[safe_name].get("run_count", 0) + 1
            registry[safe_name]["last_error"] = None
            _save_registry(registry)

        return str(result)

    except Exception as e:
        # Save error for debugging
        error_msg = traceback.format_exc()
        registry = _load_registry()
        if safe_name in registry:
            registry[safe_name]["last_error"] = str(e)
            _save_registry(registry)

        return f"Error running tool '{safe_name}':\n{error_msg}"


@mcp.tool()
def edit_tool(name: str, code: str) -> str:
    """Edit an existing forged tool's code.

    Args:
        name: Tool name
        code: New Python code (must define `run` function)
    """
    tool_file = _tool_path(name)
    if not tool_file.exists():
        return f"Error: Tool '{name}' not found. Use forge_tool to create it."

    if "def run" not in code:
        return "Error: Code must define a `run` function."

    try:
        compile(code, f"forge/{name}.py", "exec")
    except SyntaxError as e:
        return f"Error: Syntax error: {e}"

    tool_file.write_text(code, encoding="utf-8")

    registry = _load_registry()
    safe_name = tool_file.stem
    if safe_name in registry:
        registry[safe_name]["updated_at"] = time.time()
        _save_registry(registry)

    return f"Updated tool `{safe_name}`"


@mcp.tool()
def read_tool(name: str) -> str:
    """Read the source code of a forged tool.

    Args:
        name: Tool name
    """
    tool_file = _tool_path(name)
    if not tool_file.exists():
        return f"Error: Tool '{name}' not found."

    code = tool_file.read_text(encoding="utf-8")
    registry = _load_registry()
    meta = registry.get(tool_file.stem, {})

    header = f"# Tool: {tool_file.stem}\n"
    if meta:
        header += f"# Description: {meta.get('description', '?')}\n"
        header += f"# Runs: {meta.get('run_count', 0)}\n"
        if meta.get("last_error"):
            header += f"# Last error: {meta['last_error']}\n"
    header += f"# File: {tool_file}\n\n"

    return header + code


@mcp.tool()
def list_forged_tools() -> str:
    """List all forged custom tools."""
    registry = _load_registry()
    tool_files = [f.stem for f in FORGE_DIR.glob("*.py") if not f.stem.startswith("_")]

    if not tool_files:
        return "No forged tools yet. Use `forge_tool` to create one."

    lines = ["# Forged Tools\n"]
    for name in sorted(tool_files):
        meta = registry.get(name, {})
        desc = meta.get("description", "No description")
        runs = meta.get("run_count", 0)
        args = meta.get("args", {})
        args_str = ", ".join(f"{k}: {v}" for k, v in args.items()) if args else "none"
        error = " [HAS ERROR]" if meta.get("last_error") else ""

        lines.append(f"## `{name}`{error}")
        lines.append(f"{desc}")
        lines.append(f"Args: {args_str} | Runs: {runs}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def delete_tool(name: str) -> str:
    """Delete a forged tool permanently.

    Args:
        name: Tool name
    """
    tool_file = _tool_path(name)
    safe_name = tool_file.stem

    if not tool_file.exists():
        return f"Error: Tool '{name}' not found."

    tool_file.unlink()

    registry = _load_registry()
    registry.pop(safe_name, None)
    _save_registry(registry)

    return f"Deleted tool `{safe_name}`"


@mcp.tool()
def test_tool(name: str, args: str = "{}") -> str:
    """Test a forged tool with sample args and show detailed output including timing.

    Args:
        name: Tool name
        args: JSON string of test arguments
    """
    start = time.time()
    result = run_tool(name, args)
    elapsed = time.time() - start

    return (
        f"# Test: `{name}`\n"
        f"Args: {args}\n"
        f"Time: {elapsed:.2f}s\n\n"
        f"## Output:\n{result}"
    )


if __name__ == "__main__":
    if "--sse" in sys.argv:
        port = 8810
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

        print(f"Starting forge MCP server on http://0.0.0.0:{port}/sse")
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        mcp.run(transport="stdio")
