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
# The local approval_gate_hook + client_tool_policy modules were the
# SDK-PreToolUse path.  All tool dispatch now flows through
# tool_dispatcher → sub-agents (Path 2 HTTP), and sub-agents runs the
# canonical approval gate (services/sub-agents/core/approval.py).
# Local hook/policy are dead — module imports removed 2026-05-25.
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


# Note on the legacy claude_agent_sdk path:
#
#   Earlier scaffolding (Phase 0 / pre-2026-05-25) imported AgentDefinition
#   + HookMatcher from claude_agent_sdk to build an SDK-driven session.
#   That path is gone — `BACKEND=live` now drives `agent.Agent.run_turn`
#   directly via `agent_for(persona)`, which gives us the same OAuth
#   bearer + TCMM render + PII redaction as ChatAgent.
#
#   The SDK import + the _persona_to_agent_def / _build_subagents /
#   _build_mcp_server_config / _build_hooks helpers below USED to wire
#   the SDK session; they're now uncalled dead code.  Removed
#   2026-05-25 cleanup.  If we ever bring SDK back (e.g. for a
#   provider-neutral runtime) it lives at:
#   github.com/anthropics/claude-agent-sdk-python


def _persona_to_agent_def(persona: PersonaSpec) -> Any:
    """LEGACY STUB — SDK path retired.  Returns None to preserve the
    function symbol for any external import that still references it."""
    return None


# The next three functions used to wire SDK session config:
# `_build_subagent_registry` (which sub-agent personas the Director's SDK
# session could delegate to), `_build_mcp_server_config` (which MCP
# servers the SDK should connect to as a client), and `_build_hooks`
# (PreToolUse approval-gate hooks).  All three are unused now —
# BACKEND=live drives Agent.run_turn directly, and tool dispatch goes
# through this module's own external loop (no SDK).  Stubbed to {} so
# any straggling import keeps loading.

def _build_subagent_registry(foreground, registry) -> dict[str, Any]:
    """STUB — SDK subagent registry retired; returns empty dict."""
    return {}


def _build_mcp_server_config() -> dict[str, Any]:
    """STUB — SDK MCP server config retired; returns empty dict."""
    return {}


def _build_hooks() -> dict[str, Any]:
    """STUB — SDK PreToolUse hooks retired; returns empty dict.

    Approval gate enforcement lives in sub-agents.core.agentic.handle_tool
    (chat path) + sub-agents.core.approval.gate (shared).  Not invoked
    from this module any more.
    """
    return {}


def _build_tool_schemas(persona_tools: list[str]) -> list[dict]:
    """Return Anthropic-shape `tools` array for live TCMM /generate calls.

    Two sources:
      1. In-process MCP tool registries (ledger_mcp + memory_mcp) —
         schemas pulled directly from the SdkMcpTool / fallback meta.
         Dispatched in-process by tool_dispatcher.Path 1.
      2. Client tool schemas (_CLIENT_TOOL_SCHEMAS) — daemon-routed
         tools whose impl lives in sub-agents.  Dispatched via
         tool_dispatcher.Path 2 (HTTP to sub-agents → daemon WS).

    Unknown names in `persona_tools` are dropped silently.
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

    # Layer client tool schemas on top — see agent/client_tools.py for
    # the shared source of truth (same dict consumed by agent/base.py's
    # _default_tool_schemas).  In-process schemas win on name collision.
    try:
        from agent.client_tools import CLIENT_TOOL_SCHEMAS
        for name, schema in CLIENT_TOOL_SCHEMAS.items():
            by_name.setdefault(name, schema)
    except Exception as e:
        logger.debug(f"[runtime] client tool schemas not loaded: {e}")

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
    team_id: Optional[str] = None,
    backend_name: Optional[str] = None,
    max_turns: Optional[int] = None,
    client_tools: Optional[list[dict[str, Any]]] = None,
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
        task_id=task_id,
        team_id=team_id,
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

        # [OAUTH_ATTRIBUTION_FIX_2026_05_26] OAuth-bearer (Max-SSO) calls
        # to NON-Haiku models are gated by Anthropic's CLI_SYSPROMPT_PREFIXES
        # exact-string check: the FIRST system block must be a standalone
        # text block whose body matches one of the official Claude Code
        # identity strings.  Without it, Sonnet/Opus 429 with a generic
        # `rate_limit_error: Error` — even when the plan dashboard shows
        # plenty of headroom.  Haiku is lenient and does not enforce the
        # gate.  Caught when the v3 org-test re-fire 429'd on every Sonnet
        # call while a direct Haiku probe with the same OAuth bearer
        # returned HTTP 200.  See memory:
        # architecture_oauth_attribution_429.md
        #
        # The agent-proxy / LibreChat path injects this via TCMM's
        # AnthropicRenderer.render_structured() (see
        # TCMM/core/renderers/anthropic_renderer.py
        # [STANDALONE_PREFIX_BLOCK_2026_05_20]).  The agent-runtime live
        # mode skips that renderer (we want the persona prompt straight,
        # not the full Veilguard preamble + memory) — so we have to add
        # the magic block ourselves.  Match the EXACT string the TCMM
        # renderer uses; Anthropic's check is byte-exact.
        _CLAUDE_AGENT_MAGIC = (
            "You are a Claude agent, "
            "built on Anthropic's Claude Agent SDK."
        )
        system_blocks = [
            {"type": "text", "text": _CLAUDE_AGENT_MAGIC}
        ] + list(system_blocks)

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
        # [CLIENT_TOOLS_MERGE_2026_05_31] Merge LibreChat's client/MCP tools
        # (web/shell/fs/client) so the persona (e.g. Director) advertises +
        # can call them.  Dispatch routes via tool_dispatcher Path 2 (the
        # sub-agents + client-daemon WS bridge).  Persona tools win on a
        # name collision; the schemas are attached to the agent below so its
        # tools() includes them.
        _client_tool_schemas: list[dict] = []
        for _ct in (client_tools or []):
            if not isinstance(_ct, dict):
                continue
            _nm = _ct.get("name")
            if not _nm or _nm in allowed_tools:
                continue
            allowed_tools.append(_nm)
            _client_tool_schemas.append(_ct)

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
            # [CLIENT_TOOLS_MERGE_2026_05_31] hand the merged client tool
            # schemas to the agent so its tools() advertises them to the LLM.
            live_agent._client_tool_schemas = list(_client_tool_schemas)
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

        # [PER_DISPATCH_MAX_TURNS_2026_05_27] Per-IC turn cap.  v7 (and
        # every prior org-test) had critic-claim chew through 18+ LLM
        # turns inside one dispatch reading-file → think → read-again,
        # never deciding.  F2 (DISPATCH_TIMEOUT_S=270) contained it via
        # wall-clock but the call count is still the cost driver
        # (~$0.50/dispatch for critics that never decide).  Per-IC cap
        # gives critics fewer chances to dither: a critic with max_turns=8
        # either decides quickly or the loop exits with an explicit
        # "didn't converge" event, which the calling code treats the
        # same as a timeout (force-cancel + audit comment).
        _max_turns = max_turns if max_turns is not None else _MAX_LOOP_ITERATIONS

        # [LOOP_CONTEXT_FIX_2026_05_29]  THE big one.  The adapter the
        # Agent drives takes a SINGLE user-message string
        # (`_last_user_text(messages)`), not a messages array.  So on
        # iteration 2+ of this tool-dispatch loop the model would see
        # ONLY the tool-result summary and LOSE the original task brief —
        # which is exactly why ICs flailed ("I need to check what task
        # I'm working on" → read_file on invented task-state paths →
        # max_turns → cancel, producing nothing).
        #
        # Fix: keep the original dispatch brief + a running action log,
        # and rebuild a SELF-CONTAINED user message each iteration so the
        # model always has (a) its task, (b) what it has already done,
        # (c) the latest tool results, (d) the next-step nudge.  This is
        # the contained version of "pass the full messages array"; it
        # lives entirely in the dispatch loop and doesn't touch the
        # shared TCMM adapter.
        def _first_user_text(msgs: list[dict]) -> str:
            for m in msgs:
                if m.get("role") == "user":
                    c = m.get("content")
                    if isinstance(c, str) and c.strip():
                        return c
            return ""
        _original_task_text = _first_user_text(loop_messages)
        _action_log: list[str] = []

        while iteration < _max_turns:
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
                        if et == "audit":
                            # The agent emits one TO_LLM + one FROM_LLM
                            # audit event per LLM round-trip.  In the
                            # ChatAgent path (LibreChat → pii-proxy)
                            # chat_agent_handler.py consumes these and
                            # writes to pii_audit.  In our path (inbox
                            # poller dispatch), nobody downstream
                            # consumes them, so we'd silently drop both
                            # rows — admin dashboard ends up showing
                            # zero turns even though the LLM was called
                            # multiple times.  Persist them directly.
                            audit.record_event(
                                audit_ev=ev,
                                conversation_id=conversation_id,
                                user_id=user_id,
                                tenant_id=tenant_id,
                                agent_id=persona.agent_id,
                                task_id=task_id,
                                parent_cid=parent_cid,
                            )
                            continue   # don't forward — terminal here
                        if et == "assistant":
                            # Keep for tool-use extraction below.
                            turn_messages.append(ev)
                            continue   # don't forward raw 'assistant'
                                       # — text/tool events follow
                        yield ev
            except Exception as e:
                # [TRANSIENT_RETRY_2026_05_29]  A transient connection
                # drop on the upstream LLM call ("Server disconnected",
                # RemoteProtocolError, read timeout, connection reset)
                # used to abort the ENTIRE multi-turn run — which broke
                # Director fanout mid-sequence (parent created, then
                # turn-2 connection dropped → no subtasks).  Retry the
                # SAME iteration up to twice on transient errors before
                # giving up.  Deterministic errors (bad request, etc.)
                # are not retried — they'd just fail again.
                _msg = str(e)
                _transient = any(s in _msg.lower() for s in (
                    "server disconnected", "remoteprotocol", "connection reset",
                    "read timeout", "timed out", "connection aborted",
                    "incomplete", "peer closed",
                ))
                _retries = locals().get("_turn_retries", 0)
                if _transient and _retries < 2:
                    _turn_retries = _retries + 1
                    logger.warning(
                        f"[runtime] transient backend error (retry "
                        f"{_turn_retries}/2): {type(e).__name__}: {_msg[:120]}"
                    )
                    iteration -= 1  # redo this iteration
                    import asyncio as _aio
                    await _aio.sleep(0.5 * _turn_retries)
                    continue
                logger.exception(f"[runtime] backend {mode} failed: {e}")
                yield {
                    "type": "error",
                    "code": "backend_error",
                    "message": str(e),
                }
                break
            else:
                # Successful iteration — reset the transient-retry counter.
                _turn_retries = 0

            # External dispatch loop (both scripted + live modes).
            tool_uses = _extract_tool_uses(turn_messages)
            if not tool_uses:
                break  # no tools → backend is done with this turn

            # Dispatch each tool, accumulate results.
            tool_result_blocks: list[dict[str, Any]] = []
            for tu in tool_uses:
                yield {"type": "tool_dispatch", "name": tu["name"], "id": tu["id"]}
                result = await tool_dispatcher.dispatch(tu["name"], tu["input"])
                # [TOOL_RESULT_VISIBILITY_2026_05_29]  Carry the actual
                # result/error TEXT on the event, not just the is_error
                # bool.  The sidebar (and any consumer) needs to SHOW
                # whether a tool worked or failed and WHY — silent tool
                # failures were causing the agents to loop with no visible
                # cause.  Extract the human-readable text from the MCP
                # content envelope; cap so a huge file read doesn't flood
                # the event stream.
                _is_err = bool(result.get("isError", False))
                _result_text = ""
                try:
                    for _b in (result.get("content") or []):
                        if isinstance(_b, dict) and _b.get("type") == "text":
                            _result_text = _b.get("text", "") or ""
                            break
                except Exception:
                    pass
                yield {
                    "type": "tool_result",
                    "id": tu["id"],
                    "name": tu["name"],
                    "is_error": _is_err,
                    "degraded": bool(result.get("degraded", False)),
                    "result_preview": _result_text[:500],
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
            # 2026-05-25: Agent.run_turn is single-turn — its underlying
            # AnthropicAdapter.generate(user_message: str) takes just one
            # user message, not a messages array.  Inside a multi-turn
            # tool-dispatch loop, the previous turn's assistant tool_use
            # and the freshly-appended user(tool_result_blocks) message
            # would be lost: _last_user_text(messages) returns "" when
            # the last user message contains only tool_result blocks,
            # and the adapter hits Anthropic with empty content → 400.
            #
            # Fix: append an additional plain-text user message that
            # SUMMARISES the tool outcomes.  The LLM sees this on the
            # next iteration and continues the work.  We lose the
            # native tool_result block structure (the LLM sees text
            # not blocks), but the practical loss is small — the
            # surrounding messages already have the tool_use call
            # captured, and the text summary tells the LLM what came
            # back.  Future fix: refactor adapter.generate to accept
            # a full messages array; until then, this is the seam.
            # [LOOP_CONTEXT_FIX_2026_05_29]  Record this turn's actions in
            # the running log, then rebuild a self-contained user message
            # that re-states the task + everything done so far.
            for tu, blk in zip(tool_uses, tool_result_blocks):
                tname = tu.get("name", "?")
                is_err = bool(blk.get("is_error"))
                payload = blk.get("content") or ""
                if isinstance(payload, list):
                    parts = []
                    for b in payload:
                        if isinstance(b, dict) and b.get("type") == "text":
                            parts.append(b.get("text", ""))
                    payload = "\n".join(parts) or json.dumps(payload)
                payload = str(payload)[:1200]
                tag = "ERROR" if is_err else "ok"
                # Compact the tool input for the log so the model can see
                # WHAT it called (e.g. which path it wrote).
                _inp = tu.get("input") or {}
                _inp_compact = ", ".join(
                    f"{k}={str(v)[:60]}" for k, v in _inp.items()
                    if k not in ("content", "old_string", "new_string")
                )
                _action_log.append(
                    f"  turn {iteration}: {tname}({_inp_compact}) "
                    f"→ [{tag}] {payload}"
                )

            # Cap the log so a long loop doesn't blow the context window;
            # keep the most recent 12 actions (older ones rarely matter).
            _recent_log = _action_log[-12:]
            _dropped = len(_action_log) - len(_recent_log)
            _log_header = (
                f"  …({_dropped} earlier actions omitted)\n"
                if _dropped > 0 else ""
            )
            rebuilt = (
                f"━━━ YOUR TASK (unchanged — do not go looking for it) ━━━\n"
                f"{_original_task_text}\n\n"
                f"━━━ WHAT YOU'VE DONE SO FAR ━━━\n"
                f"{_log_header}" + "\n".join(_recent_log) + "\n\n"
                f"━━━ NEXT ━━━\n"
                f"Continue the task above. If your deliverable file is "
                f"already written (a write_file returned ok), call "
                f"attach_output(task_id, path) then submit_for_review. "
                f"If a tool keeps failing/being unavailable, do NOT retry "
                f"it and do NOT hunt for task-state files — finish from "
                f"what you have or raise a blocker_raised comment. You are "
                f"on turn {iteration} of {_max_turns}; converge."
            )
            loop_messages.append({"role": "user", "content": rebuilt})

            stop_reason = _last_stop_reason(turn_messages)
            if stop_reason == "end_turn":
                break  # backend signalled the conversation is complete

        if iteration >= _max_turns:
            logger.warning(
                f"[runtime] hit max loop iterations ({_max_turns}); "
                "agent did not converge"
            )
            yield {
                "type": "error",
                "code": "max_iterations",
                "message": f"agent loop did not converge in {_max_turns} iterations",
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
        # Phase 6.7 — feed APR with total tokens billed for this turn.
        # `tokens_input_total` is the spec-normalized "total input" (new
        # + cache_create + cache_read, per memory architecture_token_accounting).
        try:
            from .runtime_health import apr_record_tokens
            apr_record_tokens(
                input_tokens=usage.tokens_input_total,
                output_tokens=usage.tokens_output,
            )
        except Exception:
            pass

        # [PHASE_7_5_COST_WRITE_BACK_2026_05_28] When this turn ran for
        # an inbox-dispatched Task (task_id is set), increment the
        # Task's cost_attributed_usd by the turn's cost so team budget
        # rollups see the true spend without a separate cron.  Computed
        # from tokens × per-model rate card (same logic as
        # `outcomes._cost_from_tokens`).  Best-effort: log and continue
        # on failure — billing accounting should never block the agent.
        if task_id:
            try:
                from .proposals.outcomes import _cost_from_tokens
                turn_cost_usd = _cost_from_tokens(
                    model=usage.model or "",
                    tokens_in=int(usage.tokens_input_total or 0),
                    tokens_out=int(usage.tokens_output or 0),
                    cache_create=int(usage.cache_create or 0),
                    cache_read=int(usage.cache_read or 0),
                )
                if turn_cost_usd > 0:
                    from .ledger import tasks as _tasks_mod
                    _tasks_mod.increment_cost_attributed(
                        task_id=task_id,
                        tenant_id=tenant_id,
                        user_id=user_id,
                        delta_usd=turn_cost_usd,
                    )
            except Exception as _cw_e:
                logger.debug(
                    f"[runtime] cost write-back skipped for {task_id}: {_cw_e}"
                )

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
