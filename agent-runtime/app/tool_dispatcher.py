"""In-process tool dispatcher for non-SDK backends.

The Claude Agent SDK dispatches MCP tool calls internally — it parses
LLM tool_use blocks, looks up the tool by name in its MCP servers,
calls the handler, and feeds tool_result back to the LLM in the next
turn — all without exposing the dispatch step to us.

For the ScriptedBackend and SsoBackend (which don't wrap the SDK's
agent loop), runtime.py owns the loop and uses this dispatcher to
execute tools between LLM turns.

The dispatcher knows about:
  - veilguard_ledger tools (app/tools/ledger_mcp.py)
  - veilguard_memory tools (app/tools/memory_mcp.py)
  - (future) client-daemon tools via the bridge (deferred)

Each tool's handler is async and returns the MCP content envelope
(`{"content": [...], "isError": bool}`).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .tools import ledger_mcp, memory_mcp

logger = logging.getLogger("agent-runtime.tool_dispatcher")


# Build a name → handler lookup at module load.  Tool names match the
# `name` attribute on each SdkMcpTool (e.g., "create_task", "recall").
# MCP-prefixed tool names from the LLM (e.g., "mcp__veilguard_ledger__create_task")
# get stripped to the bare name before lookup.

_HANDLER_REGISTRY: dict[str, Any] = {}


def _build_registry() -> None:
    """Index all in-process MCP tool handlers by bare name."""
    for tool in (*ledger_mcp._ALL_TOOLS, *memory_mcp._ALL_TOOLS):
        # Each SdkMcpTool has .name + .handler; the shim has _mcp_tool_meta.
        name = getattr(tool, "name", None)
        handler = getattr(tool, "handler", None)
        if name and handler:
            _HANDLER_REGISTRY[name] = handler


_build_registry()


def _strip_mcp_prefix(tool_name: str) -> str:
    """`mcp__veilguard_ledger__create_task` → `create_task`."""
    if tool_name.startswith("mcp__"):
        parts = tool_name.split("__")
        if len(parts) >= 3:
            return "__".join(parts[2:])
    return tool_name


async def dispatch(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Look up the tool by name + call its handler.

    Returns the MCP content envelope.  If the tool isn't registered,
    returns an isError result with a clear message — the LLM sees that
    on its next turn and can react.
    """
    bare = _strip_mcp_prefix(tool_name)
    handler = _HANDLER_REGISTRY.get(bare)
    if handler is None:
        logger.warning(f"[tool_dispatcher] unknown tool: {tool_name!r}")
        return {
            "content": [{
                "type": "text",
                "text": (
                    f"ERROR: tool {tool_name!r} not registered in agent-runtime; "
                    f"available: {sorted(_HANDLER_REGISTRY.keys())}"
                ),
            }],
            "isError": True,
        }

    try:
        return await handler(tool_input)
    except Exception as e:
        logger.exception(f"[tool_dispatcher] {tool_name!r} raised: {e}")
        return {
            "content": [{"type": "text", "text": f"ERROR: {tool_name!r} raised {e}"}],
            "isError": True,
        }


def available_tools() -> list[str]:
    return sorted(_HANDLER_REGISTRY.keys())


def to_anthropic_tool_result_block(
    *,
    tool_use_id: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Convert a dispatcher result to an Anthropic tool_result content block.

    For Anthropic API + Claude Agent SDK, tool_results go in a user
    message as `{"type": "tool_result", "tool_use_id": "...",
    "content": "..."}` (content is a string for simple tools).
    """
    content_blocks = result.get("content", [])
    # Concatenate text blocks into one string for the simple case.
    text_parts = []
    for b in content_blocks:
        if isinstance(b, dict) and b.get("type") == "text":
            text_parts.append(b.get("text", ""))
    text = "\n".join(text_parts) or json.dumps(result)
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": text,
        "is_error": bool(result.get("isError", False)),
    }


__all__ = [
    "dispatch",
    "available_tools",
    "to_anthropic_tool_result_block",
]
