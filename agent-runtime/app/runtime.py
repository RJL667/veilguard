"""Core runtime — composes middleware + backend around the agent loop.

The flow:

  request
    │
    ├─ set TenantContext (cid, user_id, tenant_id, agent_id)
    │
    ├─ fetch TCMM blob (cached per parent_cid + agent_id) — bytes pass
    │  through unchanged; TCMM owns cache_control placement
    │
    ├─ append persona system_prompt as separate FINAL block (no marker)
    │
    ├─ pick backend (sdk | scripted | sso) — env var BACKEND
    │
    ├─ agent loop:
    │      while True:
    │        for msg in backend.run_turn(config):
    │          emit_event(msg)
    │          if msg has tool_use:
    │             await tool_dispatcher.dispatch(...)
    │             append tool_result to messages
    │        if no tool_use in this turn OR stop_reason=end_turn:
    │           break
    │      (SDK backend does its own internal loop; runs once externally)
    │      (Scripted/SSO backends iterate externally via this loop)
    │
    └─ on completion → audit row → SSE close
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, AsyncIterator, Optional

from .config import (
    CACHE_TTL_DEFAULT,
    SUB_AGENTS_URL,
    VEILGUARD_INTERNAL_SECRET,
)
from .backends.base import BackendConfig
from .backends.scripted_backend import ScriptedBackend
from .middleware import audit, tcmm, tenant

# The unified agent harness — used for BACKEND=live.  We add the repo
# root to sys.path so `from agent import ...` resolves regardless of
# how agent-runtime was launched.
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
from .hooks.approval_gate import (
    approval_gate_hook,
    CLIENT_DAEMON_PREFIX,
)
from . import tool_dispatcher
from .personas.loader import PersonaSpec, PersonaRegistry

logger = logging.getLogger("agent-runtime.runtime")


# Backend selection: env var BACKEND ∈ {live, scripted}.  Default
# `live` for production (calls TCMM /generate); scripted demos + tests
# set BACKEND=scripted to use the deterministic in-process fake.
def _get_backend_name() -> str:
    name = os.environ.get("BACKEND", "live").lower()
    # Legacy aliases — older demos / docs still set these.
    if name in ("sdk", "sso"):
        return "live"
    return name


# Conservative cap on agent-loop iterations so a misbehaving script
# or LLM can't infinite-loop us.  Production SDK calls don't iterate
# externally (SDK handles it); scripted backend with N turns scripted
# obviously fits within this.
_MAX_LOOP_ITERATIONS = 50


# ── SDK availability for AgentDefinition + HookMatcher constructors ─────


_SDK_AVAILABLE = False
try:
    from claude_agent_sdk import (  # type: ignore
        AgentDefinition,
        HookMatcher,
    )
    _SDK_AVAILABLE = True
except Exception:
    pass


# ── Persona → AgentDefinition translation ────────────────────────────────


def _persona_to_agent_def(persona: PersonaSpec) -> Any:
    """Build SDK's AgentDefinition from PersonaSpec (only for SDK backend)."""
    if not _SDK_AVAILABLE:
        return None
    return AgentDefinition(
        description=f"Veilguard persona: {persona.display_name}",
        prompt=persona.system_prompt,
        tools=list(persona.tools),
        model=persona.model_for("reactive"),
    )


def _build_subagent_registry(
    foreground: PersonaSpec,
    registry: PersonaRegistry,
) -> dict[str, Any]:
    """Build the `agents` dict the SDK takes."""
    if not _SDK_AVAILABLE:
        return {}
    sub: dict[str, Any] = {}
    if foreground.role == "director":
        for p in registry.all():
            if p.agent_id == foreground.agent_id:
                continue
            if p.role in ("ic", "consultant"):
                sub[p.agent_id] = _persona_to_agent_def(p)
    elif foreground.role == "ic":
        for p in registry.consultants():
            sub[p.agent_id] = _persona_to_agent_def(p)
    return sub


def _build_mcp_server_config() -> dict[str, Any]:
    """MCP servers the SDK should connect to as a client."""
    if not _SDK_AVAILABLE:
        return {}
    from .tools.ledger_mcp import build_ledger_mcp_server
    from .tools.memory_mcp import build_memory_mcp_server

    headers = {}
    if VEILGUARD_INTERNAL_SECRET:
        headers["x-veilguard-internal-secret"] = VEILGUARD_INTERNAL_SECRET

    servers: dict[str, Any] = {}
    ledger = build_ledger_mcp_server()
    if ledger is not None:
        servers["veilguard_ledger"] = ledger
    memory = build_memory_mcp_server()
    if memory is not None:
        servers["veilguard_memory"] = memory
    servers["sub-agents"] = {
        "type": "http",
        "url": f"{SUB_AGENTS_URL}/mcp",
        "headers": headers,
    }
    return servers


def _build_hooks() -> dict[str, Any]:
    """PreToolUse hook for the approval gate.

    Kept for compat with code that still references it.  Not used now
    that the SDK backend is gone — the approval gate is enforced at
    the daemon's WS handshake side (Phase 0.3 capability matrix).
    """
    if not _SDK_AVAILABLE:
        return {}
    return {
        "PreToolUse": [
            HookMatcher(
                matcher=f"^{CLIENT_DAEMON_PREFIX}",
                hooks=[approval_gate_hook],
            ),
        ],
    }


def _build_tool_schemas(persona_tools: list[str]) -> list[dict]:
    """Return Anthropic-shape `tools` array for live TCMM /generate calls.

    Looks up each persona tool name in our in-process MCP tool
    registries (ledger_mcp + memory_mcp).  Unknown names are dropped
    silently — TCMM/Anthropic wouldn't know how to dispatch them, and
    our runtime's tool_dispatcher only routes the known set.
    """
    from .tools.ledger_mcp import _ALL_TOOLS as _LEDGER_TOOLS
    from .tools.memory_mcp import _ALL_TOOLS as _MEMORY_TOOLS

    by_name: dict[str, dict] = {}
    for tool_obj in list(_LEDGER_TOOLS) + list(_MEMORY_TOOLS):
        if hasattr(tool_obj, "name") and hasattr(tool_obj, "input_schema"):
            by_name[tool_obj.name] = {
                "name": tool_obj.name,
                "description": tool_obj.description,
                "input_schema": tool_obj.input_schema,
            }
        elif hasattr(tool_obj, "_mcp_tool_meta"):
            meta = tool_obj._mcp_tool_meta
            by_name[meta["name"]] = {
                "name": meta["name"],
                "description": meta["description"],
                "input_schema": meta["input_schema"],
            }

    out: list[dict] = []
    seen: set[str] = set()
    for t in persona_tools:
        if t in by_name and t not in seen:
            out.append(by_name[t])
            seen.add(t)
    return out


# ── Tool extraction from messages ────────────────────────────────────────


def _extract_tool_uses(messages: list[Any]) -> list[dict[str, Any]]:
    """Find every tool_use block in a list of backend messages.

    Each result: {"id": str, "name": str, "input": dict}.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        msg_type = getattr(msg, "type", None) or (
            msg.get("type") if isinstance(msg, dict) else None
        )
        if msg_type != "assistant":
            continue
        inner = getattr(msg, "message", None) or (
            msg.get("message") if isinstance(msg, dict) else None
        )
        content = (
            getattr(inner, "content", None)
            or (inner.get("content") if isinstance(inner, dict) else None)
            or []
        )
        for block in content:
            btype = (
                getattr(block, "type", None)
                or (block.get("type") if isinstance(block, dict) else None)
            )
            if btype == "tool_use":
                out.append({
                    "id": (
                        getattr(block, "id", None)
                        or (block.get("id") if isinstance(block, dict) else "")
                    ),
                    "name": (
                        getattr(block, "name", None)
                        or (block.get("name") if isinstance(block, dict) else "")
                    ),
                    "input": (
                        getattr(block, "input", None)
                        or (block.get("input") if isinstance(block, dict) else {})
                        or {}
                    ),
                })
    return out


def _last_stop_reason(messages: list[Any]) -> str:
    """Find the most-recent stop_reason in the messages."""
    for msg in reversed(messages):
        msg_type = getattr(msg, "type", None) or (
            msg.get("type") if isinstance(msg, dict) else None
        )
        if msg_type == "result":
            return (
                getattr(msg, "stop_reason", None)
                or (msg.get("stop_reason") if isinstance(msg, dict) else "")
                or ""
            )
    return ""


# ── Main entry ───────────────────────────────────────────────────────────


async def run_agent_query(
    *,
    persona: PersonaSpec,
    conversation_id: str,
    user_id: str,
    tenant_id: str,
    messages: list[dict[str, Any]],
    registry: PersonaRegistry,
    constitution: dict | None = None,
    parent_cid: Optional[str] = None,
    task_id: Optional[str] = None,
    backend_name: Optional[str] = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield SSE-shaped event dicts as the agent runs.

    backend_name overrides the env-default BACKEND var (useful for tests).
    """
    is_background = tenant.is_subagent_cid(conversation_id) or bool(parent_cid)

    with tenant.set_tenant_context(
        conversation_id=conversation_id,
        user_id=user_id,
        tenant_id=tenant_id,
        agent_id=persona.agent_id,
        parent_cid=parent_cid,
        is_background=is_background,
    ):
        # ── Resolve backend mode ────────────────────────────────────────
        # Two paths only:
        #   scripted — ScriptedBackend (deterministic; tests + scripted demos)
        #   live     — agent.Agent.run_turn (the unified 5-step pipeline)
        mode = (backend_name or _get_backend_name())
        if mode not in ("scripted", "live"):
            yield {
                "type": "error",
                "code": "unknown_backend",
                "message": f"BACKEND={mode!r}; expected 'live' or 'scripted'",
            }
            return

        scripted = ScriptedBackend() if mode == "scripted" else None

        # ── Fetch TCMM blob (only for scripted mode — live mode lets
        #    TCMM /generate do its own rendering with include_memory=True) ─
        if mode == "scripted":
            try:
                system_blocks = await tcmm.get_system_prefix(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    agent_id=persona.agent_id,
                    model=persona.model_for("reactive"),
                    ttl=CACHE_TTL_DEFAULT,
                )
            except Exception as e:
                logger.exception(f"[runtime] TCMM render failed: {e}; degraded mode")
                system_blocks = []
        else:
            system_blocks = []

        if persona.system_prompt:
            system_blocks = list(system_blocks) + [
                {"type": "text", "text": persona.system_prompt}
            ]

        # ── Pre-compute tool schemas for live mode ──────────────────────
        # Live mode passes Anthropic-shape tool schemas to TCMM /generate.
        # Scripted mode doesn't need them — it emits canned tool_use blocks.
        persona_tool_schemas: list[dict] = []
        if mode == "live":
            persona_tool_schemas = _build_tool_schemas(list(persona.tools))

        allowed_tools = list(persona.tools)

        # ── Emit run_start ──────────────────────────────────────────────
        run_id = uuid.uuid4().hex[:12]
        yield {
            "type": "run_start",
            "run_id": run_id,
            "agent_id": persona.agent_id,
            "model": persona.model_for("reactive"),
            "backend": mode,
            "started_at": time.time(),
        }

        # ── Agent loop ──────────────────────────────────────────────────
        # Both modes use the external dispatch loop in this file:
        # backend yields tool_use blocks → tool_dispatcher runs them →
        # tool_results appended to loop_messages → next iteration.
        usage = audit.TurnUsage(model=persona.model_for("reactive"))
        loop_messages = list(messages)
        iteration = 0

        # Build Agent + TurnContext for live mode (instantiated once
        # per run_agent_query; reused across loop iterations).
        if mode == "live":
            from agent import agent_for, TurnContext
            from agent import events as _agent_events
            AgentCls = agent_for(persona)
            live_agent = AgentCls(persona)
            agent_ctx = TurnContext(
                conversation_id=conversation_id,
                user_id=user_id,
                tenant_id=tenant_id,
                parent_cid=parent_cid,
                is_background=is_background,
            )
        else:
            live_agent = None
            agent_ctx = None

        while iteration < _MAX_LOOP_ITERATIONS:
            iteration += 1

            turn_messages: list[Any] = []

            try:
                if scripted is not None:
                    config = BackendConfig(
                        persona_id=persona.agent_id,
                        persona_system_prompt=persona.system_prompt,
                        persona_model=persona.model_for("reactive"),
                        persona_tools=allowed_tools,
                        persona_subagents={},
                        system_blocks=system_blocks,
                        messages=loop_messages,
                        mcp_servers={},
                        hooks={},
                        include_partial_messages=True,
                    )
                    stream = scripted.run_turn(config)
                    async for msg in audit.tap_sdk_stream(stream, usage):
                        turn_messages.append(msg)
                        async for ev in _msg_to_events(msg):
                            yield ev
                else:
                    # Live mode — drive Agent.run_turn.  The Agent emits
                    # the high-level event stream directly; we filter
                    # out its inner run_start/run_end/usage (the runtime
                    # owns those at the OUTER scope) and forward the
                    # rest.  audit middleware still gets to see the
                    # `assistant` event for token accounting.
                    assert live_agent is not None and agent_ctx is not None
                    inner_stream = live_agent.run_turn(loop_messages, agent_ctx)
                    async for ev in audit.tap_sdk_stream(inner_stream, usage):
                        if not isinstance(ev, dict):
                            # Defensive: scripted backend uses _FakeMsg
                            # objects, but Agent only yields dicts.
                            continue
                        et = ev.get("type")
                        if et in ("run_start", "run_end", "usage"):
                            # Suppressed; runtime emits outer versions.
                            continue
                        if et == "assistant":
                            # Keep for tool-use extraction below.
                            turn_messages.append(ev)
                            continue   # don't forward raw 'assistant'
                                       # — text/tool events follow
                        yield ev
            except Exception as e:
                logger.exception(f"[runtime] backend {mode} failed: {e}")
                yield {
                    "type": "error",
                    "code": "backend_error",
                    "message": str(e),
                }
                break

            # External dispatch loop (both scripted + live modes).
            tool_uses = _extract_tool_uses(turn_messages)
            if not tool_uses:
                break  # no tools → backend is done with this turn

            # Dispatch each tool, accumulate results.
            tool_result_blocks: list[dict[str, Any]] = []
            for tu in tool_uses:
                yield {"type": "tool_dispatch", "name": tu["name"], "id": tu["id"]}
                result = await tool_dispatcher.dispatch(tu["name"], tu["input"])
                yield {
                    "type": "tool_result",
                    "id": tu["id"],
                    "is_error": bool(result.get("isError", False)),
                }
                tool_result_blocks.append(
                    tool_dispatcher.to_anthropic_tool_result_block(
                        tool_use_id=tu["id"],
                        result=result,
                    )
                )

            # Append assistant + tool_result to loop_messages so the next
            # backend call sees the conversation history.
            loop_messages.append({
                "role": "assistant",
                "content": _materialize_assistant_content(turn_messages),
            })
            loop_messages.append({
                "role": "user",
                "content": tool_result_blocks,
            })

            stop_reason = _last_stop_reason(turn_messages)
            if stop_reason == "end_turn":
                break  # backend signalled the conversation is complete

        if iteration >= _MAX_LOOP_ITERATIONS:
            logger.warning(
                f"[runtime] hit max loop iterations ({_MAX_LOOP_ITERATIONS}); "
                "agent did not converge"
            )
            yield {
                "type": "error",
                "code": "max_iterations",
                "message": f"agent loop did not converge in {_MAX_LOOP_ITERATIONS} iterations",
            }

        # ── Audit + completion ──────────────────────────────────────────
        audit.record_turn(
            conversation_id=conversation_id,
            user_id=user_id,
            tenant_id=tenant_id,
            agent_id=persona.agent_id,
            task_id=task_id,
            parent_cid=parent_cid,
            usage=usage,
        )

        yield {
            "type": "usage",
            "tokens_input_total": usage.tokens_input_total,
            "tokens_input_new": usage.tokens_input_new,
            "tokens_output": usage.tokens_output,
            "cache_create": usage.cache_create,
            "cache_read": usage.cache_read,
            "cache_hit_rate": round(usage.cache_hit_rate(), 4),
            "model": usage.model,
            "iterations": iteration,
        }

        yield {
            "type": "run_end",
            "run_id": run_id,
            "ended_at": time.time(),
            "stop_reason": usage.stop_reason,
        }


# ── Conversion helpers ───────────────────────────────────────────────────


def _materialize_assistant_content(turn_messages: list[Any]) -> list[dict[str, Any]]:
    """Reconstruct the assistant message content blocks from a stream."""
    out: list[dict[str, Any]] = []
    for msg in turn_messages:
        msg_type = getattr(msg, "type", None) or (
            msg.get("type") if isinstance(msg, dict) else None
        )
        if msg_type != "assistant":
            continue
        inner = getattr(msg, "message", None) or (
            msg.get("message") if isinstance(msg, dict) else None
        )
        content = (
            getattr(inner, "content", None)
            or (inner.get("content") if isinstance(inner, dict) else None)
            or []
        )
        for block in content:
            btype = (
                getattr(block, "type", None)
                or (block.get("type") if isinstance(block, dict) else None)
            )
            if btype == "text":
                out.append({
                    "type": "text",
                    "text": (
                        getattr(block, "text", None)
                        or (block.get("text") if isinstance(block, dict) else "")
                    ),
                })
            elif btype == "tool_use":
                out.append({
                    "type": "tool_use",
                    "id": (
                        getattr(block, "id", None)
                        or (block.get("id") if isinstance(block, dict) else "")
                    ),
                    "name": (
                        getattr(block, "name", None)
                        or (block.get("name") if isinstance(block, dict) else "")
                    ),
                    "input": (
                        getattr(block, "input", None)
                        or (block.get("input") if isinstance(block, dict) else {})
                        or {}
                    ),
                })
    return out


async def _msg_to_events(msg: Any) -> AsyncIterator[dict[str, Any]]:
    """Translate one backend message into 0..N SSE events for upstream."""
    msg_type = getattr(msg, "type", None) or (
        msg.get("type") if isinstance(msg, dict) else None
    )

    if msg_type == "stream_event":
        event = getattr(msg, "event", None) or (
            msg.get("event") if isinstance(msg, dict) else None
        )
        if event is None:
            return
        ev_type = getattr(event, "type", None) or (
            event.get("type") if isinstance(event, dict) else None
        )
        if ev_type == "content_block_delta":
            delta = getattr(event, "delta", None) or (
                event.get("delta") if isinstance(event, dict) else None
            )
            text = (
                getattr(delta, "text", None)
                or (delta.get("text") if isinstance(delta, dict) else "")
                or ""
            )
            if text:
                yield {"type": "text_delta", "text": text}

    elif msg_type == "assistant":
        inner = getattr(msg, "message", None) or (
            msg.get("message") if isinstance(msg, dict) else None
        )
        content = (
            getattr(inner, "content", None)
            or (inner.get("content") if isinstance(inner, dict) else None)
            or []
        )
        for block in content:
            btype = (
                getattr(block, "type", None)
                or (block.get("type") if isinstance(block, dict) else None)
            )
            if btype == "text":
                yield {
                    "type": "assistant_text",
                    "text": (
                        getattr(block, "text", None)
                        or (block.get("text") if isinstance(block, dict) else "")
                    ),
                }
            elif btype == "tool_use":
                yield {
                    "type": "tool_call",
                    "name": (
                        getattr(block, "name", None)
                        or (block.get("name") if isinstance(block, dict) else "")
                    ),
                    "id": (
                        getattr(block, "id", None)
                        or (block.get("id") if isinstance(block, dict) else "")
                    ),
                }

    elif msg_type == "result":
        result_text = (
            getattr(msg, "result", None)
            or (msg.get("result") if isinstance(msg, dict) else None)
        )
        if result_text:
            yield {"type": "final_result", "result": result_text}

    elif msg_type == "error":
        # agent.Agent and ScriptedBackend can yield error envelopes.
        # Surface them so callers (demos, /agent/query SSE consumers)
        # don't see a silent abort.
        yield {
            "type": "error",
            "code": (
                getattr(msg, "code", None)
                or (msg.get("code") if isinstance(msg, dict) else None)
                or "backend_error"
            ),
            "message": (
                getattr(msg, "message", None)
                or (msg.get("message") if isinstance(msg, dict) else None)
                or ""
            ),
        }


__all__ = ["run_agent_query"]
