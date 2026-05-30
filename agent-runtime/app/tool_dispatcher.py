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
    """Index all in-process MCP tool handlers by bare name.

    Two flavours of tool decorator to handle:

      1. SDK-provided (`claude_agent_sdk.tool`) — exposes `.name` and
         `.handler` attributes.  Used when claude-agent-sdk is
         installed in the environment.
      2. Fallback shim (defined inside ledger_mcp.py / memory_mcp.py
         when the SDK isn't importable) — wraps the function and
         attaches `_mcp_tool_meta = {name, description, input_schema}`,
         leaving the function itself as the callable handler.

    Before 2026-05-25 only flavour 1 was handled; flavour 2 silently
    produced an empty registry on hosts without claude-agent-sdk
    installed (which is now the default after the SDK was removed in
    the cleanup pass).  Result: every tool_use from Director / IC
    returned "unknown tool: create_task" and the multi-agent flow
    was dead.  Fix unifies both paths.
    """
    for tool in (*ledger_mcp._ALL_TOOLS, *memory_mcp._ALL_TOOLS):
        name = None
        handler = None
        # Flavour 1 (SDK SdkMcpTool)
        n_attr = getattr(tool, "name", None)
        h_attr = getattr(tool, "handler", None)
        if n_attr and callable(h_attr):
            name = n_attr
            handler = h_attr
        # Flavour 2 (fallback shim: meta dict on the function itself)
        elif hasattr(tool, "_mcp_tool_meta"):
            meta = tool._mcp_tool_meta or {}
            name = meta.get("name")
            handler = tool   # the decorated function IS the handler
        if name and callable(handler):
            _HANDLER_REGISTRY[name] = handler

    # [WORKSPACE_FS_2026_05_29]  Server-side filesystem tools.  Registered
    # in Path 1 so read_file/write_file/edit_file/list_directory execute
    # against the mounted workspace dir INSTEAD of falling through to the
    # client-daemon (Path 2).  Team-workspace artifacts (team/drafts/...)
    # are server-side state per spec §3.5; routing them to the daemon was
    # the root cause of the critic file-not-found loops when sub-agents
    # was offline.  These OVERRIDE any same-named client tool — that's
    # intentional: the org's own workspace is canonical.
    from . import tools as _tools_pkg  # noqa: F401  (ensure package import)
    from .tools import workspace_fs
    for _name, _handler in workspace_fs.HANDLERS.items():
        _HANDLER_REGISTRY[_name] = _handler


_build_registry()
logger.info(
    f"[tool_dispatcher] registered {len(_HANDLER_REGISTRY)} tool(s): "
    f"{sorted(_HANDLER_REGISTRY.keys())}"
)


def _strip_mcp_prefix(tool_name: str) -> str:
    """`mcp__veilguard_ledger__create_task` → `create_task`."""
    if tool_name.startswith("mcp__"):
        parts = tool_name.split("__")
        if len(parts) >= 3:
            return "__".join(parts[2:])
    return tool_name


async def dispatch(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Look up the tool by name + call its handler.

    Resolution order:
      1. In-process handler (ledger_mcp, memory_mcp tools).
      2. Remote dispatch to sub-agents via HTTP — used for CLIENT
         tools (file_write, run_command, etc.).  Sub-agents owns the
         daemon WS bridge and runs the approval gate before forwarding
         to the user's machine.

    Returns the MCP content envelope.  If neither path knows the tool,
    returns an isError result with a clear message — the LLM sees it on
    its next turn and can react.
    """
    bare = _strip_mcp_prefix(tool_name)
    logger.info(f"[tool_dispatcher] dispatch {tool_name!r} (bare={bare!r})")

    # Path 1 — in-process (server-side ledger / memory tools).
    handler = _HANDLER_REGISTRY.get(bare)
    if handler is not None:
        try:
            res = await handler(tool_input)
            is_err = bool(res.get("isError")) if isinstance(res, dict) else False
            if is_err:
                # Extract the human-readable error text so we don't
                # have to dig into raw_content blocks every time.
                _msg = ""
                try:
                    for _b in (res.get("content") or []):
                        if isinstance(_b, dict) and _b.get("type") == "text":
                            _msg = _b.get("text", "")
                            break
                except Exception:
                    pass
                logger.warning(
                    f"[tool_dispatcher]   path=1 {tool_name!r} ERROR: "
                    f"{_msg[:240]!r}"
                )
            else:
                logger.info(f"[tool_dispatcher]   path=1 in-process {tool_name!r} OK")
            return res
        except Exception as e:
            logger.exception(f"[tool_dispatcher] {tool_name!r} raised: {e}")
            return {
                "content": [{"type": "text", "text": f"ERROR: {tool_name!r} raised {e}"}],
                "isError": True,
            }

    # Path 2 — remote dispatch via sub-agents.
    # Sub-agents owns the per-user daemon bridge + approval gate +
    # client-settings level lookup.  Agent-runtime is a pure agent
    # orchestrator; client-tool execution lives in sub-agents.
    try:
        from .middleware import tenant
        from .config import SUB_AGENTS_URL, VEILGUARD_INTERNAL_SECRET
        import httpx as _httpx
        ctx = tenant.current()
        if ctx is None:
            return {
                "content": [{
                    "type": "text",
                    "text": (
                        f"ERROR: cannot dispatch {tool_name!r} — no tenant context. "
                        "agent-runtime must run inside set_tenant_context()."
                    ),
                }],
                "isError": True,
            }
        headers = {
            "X-Internal-Secret": VEILGUARD_INTERNAL_SECRET or "",
            "x-user-id":         ctx.user_id,
            "x-conversation-id": ctx.conversation_id,
        }
        payload = {"tool_name": tool_name, "tool_input": tool_input,
                   "agent_id":  ctx.agent_id}
        async with _httpx.AsyncClient(timeout=90.0) as client:
            r = await client.post(
                f"{SUB_AGENTS_URL.rstrip('/')}/api/agent_runtime/dispatch_tool",
                headers=headers, json=payload,
            )
        if r.status_code == 404:
            # Sub-agents doesn't have the endpoint yet (old build) —
            # surface a clean fall-back error so the LLM doesn't get
            # a cryptic 404 string.
            return {
                "content": [{"type": "text", "text": (
                    f"ERROR: tool {tool_name!r} not in agent-runtime "
                    f"and sub-agents lacks /api/agent_runtime/dispatch_tool. "
                    f"available local: {sorted(_HANDLER_REGISTRY.keys())}"
                )}],
                "isError": True,
            }
        if r.status_code >= 400:
            return {
                "content": [{"type": "text", "text": (
                    f"ERROR: sub-agents dispatch returned {r.status_code}: "
                    f"{r.text[:200]}"
                )}],
                "isError": True,
            }
        out = r.json()
        is_err = bool(out.get("isError")) if isinstance(out, dict) else False
        logger.info(f"[tool_dispatcher]   path=2 remote result is_error={is_err}")
        return out
    except _httpx.ConnectError as e:
        # [GRACEFUL_DEGRADE_2026_05_29]  sub-agents is DOWN.  How we
        # report this decides whether the agent moves on or loops.
        #
        # Empirically (caught live 2026-05-29): returning isError=True
        # for web_search made the Sonnet researcher treat it as a
        # transient/retryable failure — it retried web_search, then
        # tried read_file on invented task-state paths, burned all 25
        # turns, and NEVER wrote its deliverable.  An LLM loops on
        # isError=True but ACCEPTS a successful tool_result that tells
        # it how to proceed.
        #
        # So for READ-ONLY research tools (web_search, web_fetch,
        # search_files, grep) we return a SOFT SUCCESS (isError=False)
        # whose body instructs the agent to proceed from its own
        # knowledge and mark unverifiable claims [unverified].  For
        # MUTATION/host tools (run_command, etc.) we keep isError=True
        # — those genuinely cannot be faked and the agent must raise a
        # blocker rather than pretend they ran.
        _READONLY_RESEARCH = {"web_search", "web_fetch", "search_files", "grep"}
        if bare in _READONLY_RESEARCH:
            logger.warning(
                f"[tool_dispatcher] sub-agents unreachable ({e}) — "
                f"soft-degrading read-only {tool_name!r} to success-with-"
                f"guidance so the agent proceeds instead of looping"
            )
            return {
                "content": [{"type": "text", "text": (
                    f"[{bare} unavailable in this environment — the research "
                    f"bridge is offline]\n\n"
                    f"No results were returned.  Proceed using the knowledge "
                    f"you already have.  Complete your deliverable now: write "
                    f"it with write_file to the path in your spec, mark any "
                    f"factual claim you could not verify as [unverified], "
                    f"then attach_output and submit_for_review.  Do NOT call "
                    f"{bare} again and do NOT look for task-state files — "
                    f"there are none."
                )}],
                "isError": False,
                # [GRACEFUL_DEGRADE_2026_05_29]  Not a real success — the
                # tool is offline and we returned guidance, not results.
                # The `degraded` flag lets the UI render this as amber
                # "unavailable" rather than green ✓, so the operator isn't
                # misled into thinking web_search actually worked.
                "degraded": True,
            }
        logger.warning(
            f"[tool_dispatcher] sub-agents unreachable ({e}) — telling LLM "
            f"{tool_name!r} is unavailable so it stops retrying"
        )
        return {
            "content": [{"type": "text", "text": (
                f"TOOL UNAVAILABLE: {tool_name!r} cannot be dispatched — the "
                f"sub-agents service (client-daemon bridge) is not running "
                f"in this environment.  Do NOT retry this tool.  Either "
                f"complete the deliverable with the information you already "
                f"have (note the limitation), or raise add_comment("
                f"kind=blocker_raised) and submit_for_review with what you "
                f"have.  Never loop on a failing tool."
            )}],
            "isError": True,
        }
    except Exception as e:
        logger.exception(f"[tool_dispatcher] remote dispatch {tool_name!r} failed: {e}")
        return {
            "content": [{"type": "text", "text": (
                f"ERROR: remote dispatch of {tool_name!r} failed: {e}"
            )}],
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
