"""ChatAgent-based handler for LibreChat /anthropic/v1/messages.

This is the NEW code path that replaces `_handle_sso_request` in main.py
once the rollout is complete (PR #5b).  For now it sits behind the
VEILGUARD_USE_CHAT_AGENT=1 feature flag so it can be enabled/disabled
without code changes.

Flow:

    LibreChat → POST /anthropic/v1/messages
    pii-proxy main.py routes (when flag is on) → handle_chat_request
        ↓
    ChatAgent.from_libre_chat_request(body) — extracts model+tools+
      system+side-channel from the request
        ↓
    agent.run_turn(messages, ctx) — the unified 5-step pipeline:
      1. ingest raw user message → TCMM memory
      2. TCMM /render_structured → system blocks (raw)
      3. pii.redactor.redact_blocks (cache-stable bytes)
      4. AnthropicAdapter.generate → api.anthropic.com
      5. pii.redactor.rehydrate (REF tokens → originals)
      6. ingest raw assistant text → TCMM memory
        ↓
    Translate agent events → Anthropic Messages API response shape
    (StreamingResponse SSE if stream=True, JSONResponse otherwise)

What's still in main.py (not moved here yet):
  - Side-channel bypass (LibreChat title-gen / summary)
    [ChatAgent.from_libre_chat_request already detects this]
  - Veilguard preamble pinning to TCMM (/pin/system_prompt)
  - Audit row writes (TO_LLM + FROM_LLM with usage)
  - Shadow-tool interception (tcmm_record_turn)
  These will fold into ChatAgent in PR #5c.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi.responses import JSONResponse, StreamingResponse

# Make `from agent import ...` work from BOTH:
#   - Local dev/tests: agent-proxy/app/chat_agent_handler.py →
#     parents[3] is repo root, where agent/ + llm/ + pii/ live.
#   - Inside docker: /agent, /llm, /pii are bind-mounted siblings of
#     /app (see docker-compose.yml pii-proxy.volumes).  Probe `/` too.
_HERE = Path(__file__).resolve()
_PATH_CANDIDATES: list[Path] = []
# Walk up to the filesystem root collecting each ancestor.  In the
# container that's just /app/app → /app → / (3 levels); on the host
# it's deeper (5+).  Probing each one finds wherever agent/ lives.
for i in range(len(_HERE.parents)):
    _PATH_CANDIDATES.append(_HERE.parents[i])
_PATH_CANDIDATES.append(Path("/"))  # container root for /agent mounts
for candidate in _PATH_CANDIDATES:
    if (candidate / "agent" / "__init__.py").is_file():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

logger = logging.getLogger("pii-proxy.chat_agent")


def _write_audit(
    audit_ev: dict,
    *,
    conversation_id: str,
    user_id: str,
    is_stream: bool,
) -> None:
    """Persist one AUDIT event from Agent.run_turn to the pii_audit
    LanceDB table.  Best-effort — never raises out of the handler.

    The handler owns is_stream because the agent doesn't know whether
    its caller will frame the response as SSE or non-stream JSON.
    """
    try:
        from app import audit_db as _audit_db
        _audit_db.record(
            direction=audit_ev.get("direction", ""),
            conversation_id=conversation_id or "",
            user_id=user_id or "",
            model=audit_ev.get("model") or "",
            stream=bool(is_stream),
            content=audit_ev.get("content") or "",
            tokens_input=audit_ev.get("tokens_input_total") or None,
            tokens_output=audit_ev.get("tokens_output") or None,
            cache_create=audit_ev.get("cache_create") or None,
            cache_read=audit_ev.get("cache_read") or None,
            extra={"path": "chat_agent"},
        )
    except Exception as e:
        logger.warning(f"[chat_agent] audit_db record failed: {e}")


# ── Feature flag ────────────────────────────────────────────────────────


def is_enabled() -> bool:
    return os.environ.get("VEILGUARD_USE_CHAT_AGENT", "").strip().lower() in (
        "1", "true", "yes",
    )


# ── Entry point ─────────────────────────────────────────────────────────


async def handle_chat_request(
    body: dict,
    *,
    conversation_id: str,
    user_id: str,
    tenant_id: str = "",
    is_stream: bool = False,
):
    """Route a LibreChat-shaped request through ChatAgent.

    Returns a FastAPI Response:
      - StreamingResponse (Anthropic SSE events) when is_stream=True
      - JSONResponse (Anthropic Messages API shape) when is_stream=False

    On agent error, returns an Anthropic-shaped error in the matching
    format so LibreChat surfaces the issue instead of spinning.
    """
    # [F9_FAIL_CLOSED_2026_05_26] Refuse unauthenticated traffic.  The
    # previous code fell through to user_id="anonymous" + tenant_id=
    # "default" — those rows then sit in pii_audit invisible to any
    # tenant filter on user_id and could leak across users.  LibreChat
    # is the only legitimate caller; its veilguardClient route injects
    # x-user-id after requireJwtAuth validates the session.  If we got
    # here without one, either upstream auth is broken or someone is
    # hitting the route directly — both are 401 cases.
    if not user_id:
        logger.warning(
            "[chat_agent] missing x-user-id header — refusing request. "
            "If you're testing locally, set x-user-id to your hex-24 "
            "tenant id."
        )
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "type": "unauthenticated",
                    "message": (
                        "x-user-id header required. The chat_agent path "
                        "refuses to write audit rows without a real user "
                        "identifier (preventing pii_audit cross-tenant "
                        "leak via anonymous tagging)."
                    ),
                }
            },
        )

    # ── Workspace state injection ──────────────────────────────────────
    # Fetch the user's connected-daemon working folders + OS hint and
    # inject them into the request body as a final SYSTEM block.  The
    # legacy /handle_sso_request path does this at main.py:3361; the
    # ChatAgent path was missing it, so Claude saw no folders and
    # reported "I can't see your workspace".
    #
    # NEVER blocks the request — if the workspace fetch fails (sub-
    # agents down, daemon offline, network hiccup), we proceed without
    # the block.  The user's chat keeps working; the LLM just won't
    # know which folders are pinned.
    if user_id:
        try:
            from app.main import _fetch_workspace_state, _inject_workspace_state
            ws_state = await _fetch_workspace_state(user_id)
            if ws_state:
                _inject_workspace_state(body, "anthropic", ws_state)
                logger.info(
                    f"[chat_agent] injected workspace state: "
                    f"folders={len(ws_state.get('folders') or [])} "
                    f"client={ws_state.get('client_id', '?')}"
                )
        except Exception as _e:
            logger.debug(f"[chat_agent] workspace fetch failed: {_e}")

    try:
        from agent import ChatAgent, TurnContext
    except ImportError as e:
        logger.error(f"[chat_agent] cannot import veilguard.agent: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "type": "configuration_error",
                    "message": f"agent harness unavailable: {e}",
                }
            },
        )

    # ChatAgent now builds its own preamble via the inherited
    # Agent.preamble() → render_preamble(self.tools()) — no need to
    # pre-render here.
    chat = ChatAgent.from_libre_chat_request(body)
    # [F9_FAIL_CLOSED_2026_05_26] user_id is now guaranteed non-empty
    # by the early-return above.  tenant_id defaults to user_id when
    # not explicitly supplied (single-tenant-per-user is the common
    # case); "default" is no longer a valid fallback.
    ctx = TurnContext(
        conversation_id=conversation_id or f"conv-{uuid.uuid4().hex[:8]}",
        user_id=user_id,
        tenant_id=tenant_id or user_id,
    )

    logger.info(
        f"[chat_agent] model={chat._model} stream={is_stream} "
        f"side_channel={chat._side_channel} tools={len(chat._client_tools)} "
        f"conv={ctx.conversation_id[:14]}"
    )

    if is_stream:
        return StreamingResponse(
            _stream_anthropic_sse(chat, body.get("messages", []), ctx),
            media_type="text/event-stream",
        )
    return await _build_anthropic_json(chat, body.get("messages", []), ctx)


# ── Anthropic JSON response builder ─────────────────────────────────────


async def _build_anthropic_json(chat, messages: list, ctx):
    """Drain the agent and return a non-streaming Messages API response.

    Anthropic Messages API non-stream response shape:
      {
        "id":         "msg_...",
        "type":       "message",
        "role":       "assistant",
        "model":      "...",
        "content":    [{"type":"text","text":"..."}, ...],
        "stop_reason":"end_turn" | "tool_use" | "max_tokens",
        "stop_sequence": null,
        "usage": {"input_tokens": N, "output_tokens": M,
                  "cache_creation_input_tokens": K,
                  "cache_read_input_tokens": L},
      }
    """
    content: list[dict] = []
    usage: dict = {}
    stop_reason: str = "end_turn"
    model: str = chat._model
    error: Optional[dict] = None

    async for ev in chat.run_turn(messages, ctx):
        et = ev.get("type")
        if et == "assistant":
            msg = ev.get("message", {})
            content = msg.get("content") or []
            usage = msg.get("usage") or {}
            stop_reason = msg.get("stop_reason") or stop_reason
        elif et == "audit":
            _write_audit(
                ev,
                conversation_id=ctx.conversation_id,
                user_id=ctx.user_id,
                is_stream=False,
            )
        elif et == "error":
            error = {
                "type": ev.get("code", "api_error"),
                "message": ev.get("message", ""),
            }

    if error:
        return JSONResponse(
            status_code=500,
            content={"type": "error", "error": error},
        )

    return JSONResponse(content={
        "id":            f"msg_{uuid.uuid4().hex}",
        "type":          "message",
        "role":          "assistant",
        "model":         model,
        "content":       content,
        "stop_reason":   stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens":                 usage.get("input_tokens", 0),
            "output_tokens":                usage.get("output_tokens", 0),
            "cache_creation_input_tokens":  usage.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens":      usage.get("cache_read_input_tokens", 0),
        },
    })


# ── Anthropic SSE stream synthesizer ────────────────────────────────────


async def _stream_anthropic_sse(chat, messages: list, ctx) -> AsyncIterator[str]:
    """Drain the agent and synthesize Anthropic SSE events.

    Anthropic's Messages API SSE stream event shape (what LibreChat
    expects):

      event: message_start
      data: {"type":"message_start", "message":{...}}

      event: content_block_start
      data: {"type":"content_block_start","index":0,"content_block":{...}}

      event: content_block_delta     (one per chunk; we emit one big chunk)
      data: {"type":"content_block_delta","index":0,
             "delta":{"type":"text_delta","text":"..."}}

      event: content_block_stop
      data: {"type":"content_block_stop","index":0}

      event: message_delta
      data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},
             "usage":{...}}

      event: message_stop
      data: {"type":"message_stop"}

    Since the agent is unary (single LLM call returns full response),
    we emit ONE content_block_delta per block — no real incremental
    streaming.  LibreChat handles this fine; it just sees the full
    response arrive in one delta.
    """
    msg_id = f"msg_{uuid.uuid4().hex}"

    # Drain the agent first.  We collect all events so we can re-frame
    # them as Anthropic SSE.  Real incremental streaming would require
    # the adapter to expose streaming; that's a future improvement.
    content: list[dict] = []
    usage: dict = {}
    stop_reason: str = "end_turn"
    model: str = chat._model
    error: Optional[dict] = None

    async for ev in chat.run_turn(messages, ctx):
        et = ev.get("type")
        if et == "assistant":
            inner = ev.get("message", {})
            content = inner.get("content") or []
            usage = inner.get("usage") or {}
            stop_reason = inner.get("stop_reason") or stop_reason
        elif et == "audit":
            _write_audit(
                ev,
                conversation_id=ctx.conversation_id,
                user_id=ctx.user_id,
                is_stream=True,
            )
        elif et == "error":
            error = {
                "type": ev.get("code", "api_error"),
                "message": ev.get("message", ""),
            }

    if error:
        # Anthropic SSE error event
        yield (
            "event: error\n"
            f"data: {json.dumps({'type':'error','error':error})}\n\n"
        )
        return

    # message_start
    msg_start_payload = {
        "type": "message_start",
        "message": {
            "id":            msg_id,
            "type":          "message",
            "role":          "assistant",
            "model":         model,
            "content":       [],
            "stop_reason":   None,
            "stop_sequence": None,
            "usage": {
                "input_tokens":                usage.get("input_tokens", 0),
                "output_tokens":               0,
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
                "cache_read_input_tokens":     usage.get("cache_read_input_tokens", 0),
            },
        },
    }
    yield f"event: message_start\ndata: {json.dumps(msg_start_payload)}\n\n"

    # One block_start / block_delta / block_stop per content block.
    for idx, blk in enumerate(content):
        btype = blk.get("type")
        if btype == "text":
            text = blk.get("text", "")
            start = {
                "type": "content_block_start",
                "index": idx,
                "content_block": {"type": "text", "text": ""},
            }
            yield f"event: content_block_start\ndata: {json.dumps(start)}\n\n"
            if text:
                delta = {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "text_delta", "text": text},
                }
                yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"
            stop = {"type": "content_block_stop", "index": idx}
            yield f"event: content_block_stop\ndata: {json.dumps(stop)}\n\n"
        elif btype == "tool_use":
            start = {
                "type": "content_block_start",
                "index": idx,
                "content_block": {
                    "type": "tool_use",
                    "id": blk.get("id", ""),
                    "name": blk.get("name", ""),
                    "input": {},
                },
            }
            yield f"event: content_block_start\ndata: {json.dumps(start)}\n\n"
            # Send the entire input as one input_json_delta chunk.
            inp = blk.get("input", {}) or {}
            delta = {
                "type": "content_block_delta",
                "index": idx,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(inp),
                },
            }
            yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"
            stop = {"type": "content_block_stop", "index": idx}
            yield f"event: content_block_stop\ndata: {json.dumps(stop)}\n\n"
        else:
            # Unknown block type — pass through with minimal framing.
            start = {
                "type": "content_block_start",
                "index": idx,
                "content_block": blk,
            }
            yield f"event: content_block_start\ndata: {json.dumps(start)}\n\n"
            stop = {"type": "content_block_stop", "index": idx}
            yield f"event: content_block_stop\ndata: {json.dumps(stop)}\n\n"

    # message_delta + message_stop
    msg_delta_payload = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {
            "input_tokens":  usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens":     usage.get("cache_read_input_tokens", 0),
        },
    }
    yield f"event: message_delta\ndata: {json.dumps(msg_delta_payload)}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type':'message_stop'})}\n\n"


__all__ = ["handle_chat_request", "is_enabled"]
