"""TCMM memory client — render + ingest only.

This replaces the old `agent-runtime/app/tcmm_client.py` and the
side-channel logic in `agent-proxy/app/main.py:_handle_sso_request`.

TCMM is now pure memory: its only role is `/render_structured` (read)
and `/pre_request` + `/post_response` (write).  All LLM calls happen
in `veilguard.llm.adapter`.

Endpoints used:
  POST /render_structured  → returns {prompt, blocks, layout,
                             tier_summary, stats} for the conv
  POST /pre_request        → ingest user message (RAW, pre-redaction)
  POST /post_response      → ingest assistant message (RAW)

Env vars:
  TCMM_URL              http://localhost:8811 (default)
  TCMM_TIMEOUT          30s (default)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger("veilguard.llm.tcmm_client")


TCMM_URL = os.environ.get("TCMM_URL", "http://localhost:8811").rstrip("/")
TCMM_TIMEOUT = float(os.environ.get("TCMM_TIMEOUT", "30"))


# ── Result struct ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class RenderResult:
    """What TCMM /render_structured returns for one conv."""
    prompt: str                       # full assembled prompt text
    blocks: list[dict]                # SDK-ready text blocks with
                                      #   cache_control markers
    tier_summary: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    layout: dict = field(default_factory=dict)


# ── HTTP client (reused across calls) ──────────────────────────────────


_CLIENT: Optional[httpx.AsyncClient] = None


def _client() -> httpx.AsyncClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.AsyncClient(timeout=TCMM_TIMEOUT)
    return _CLIENT


async def close() -> None:
    """Shut down the httpx client (call on process exit)."""
    global _CLIENT
    if _CLIENT is not None:
        await _CLIENT.aclose()
        _CLIENT = None


# ── Public API ──────────────────────────────────────────────────────────


async def render_structured(
    *,
    conv_id: str,
    user_id: str,
    task_query: str,
    model: str = "anthropic",
    pii_sid: str = "",
) -> RenderResult:
    """Render the conversation's memory into SDK-ready blocks.

    Returns RAW (pre-redaction) blocks with cache_control markers
    placed by TCMM's renderer.  Caller is expected to run the blocks
    through `pii.PIIRedactor.redact_blocks(...)` before sending to the
    LLM adapter — that's the cache-stable boundary.

    [PII_SID_CONTRACT_2026_05_30]  When the server-side render-layer
    redaction hook is enabled, `pii_sid` (= caller's `SessionId.canonical()`)
    tells the server which session to mint REF tokens under — the SAME sid
    the caller will rehydrate with.  Harmless when the hook is off.
    """
    body = {
        "conversation_id": conv_id,
        "user_id": user_id or "",
        "task_query": task_query,
        "model": model,
        "pii_sid": pii_sid or "",
    }
    if _tcmm_unreachable():
        # Skip the round-trip entirely; caller's degraded path (magic
        # prefix + persona inline) handles the empty render.
        raise RuntimeError(
            f"TCMM marked unreachable (recent ConnectError); "
            f"will re-probe after {_TCMM_REPROBE_AFTER_S:.0f}s"
        )
    try:
        resp = await _client().post(f"{TCMM_URL}/render_structured", json=body)
    except httpx.ConnectError as e:
        _mark_tcmm_unreachable()
        raise RuntimeError(
            f"TCMM unreachable at {TCMM_URL}/render_structured: {e}. "
            "Start the local stack (start.bat) or set TCMM_URL."
        ) from e

    if resp.status_code != 200:
        raise RuntimeError(
            f"TCMM /render_structured returned {resp.status_code}: "
            f"{resp.text[:300]}"
        )

    data = resp.json()
    return RenderResult(
        prompt=data.get("prompt") or "",
        blocks=data.get("blocks") or [],
        tier_summary=data.get("tier_summary") or {},
        stats=data.get("stats") or {},
        layout=data.get("layout") or {},
    )


async def pin_system_prompt(
    conv_id: str, user_id: str, content: str, *, kind: str = "veilguard_preamble"
) -> None:
    """Pin a static system block to TCMM's immutable tier.

    Idempotent — re-pinning the same (conv_id, kind, content) is a no-op
    on TCMM's side.  Used by ChatAgent.prepare_session to make sure the
    Veilguard preamble + client_system are part of every render for
    this conv.

    Best-effort: TCMM hiccups don't fail the turn.
    """
    if not conv_id or not content:
        return
    if _tcmm_unreachable():
        return  # short-circuit: TCMM was just down; don't waste 250ms
    try:
        # TCMM's PinSystemPromptBody expects `text`, not `content`.
        # Misnaming the field returns 422 Unprocessable Entity which
        # silently breaks the pin (the pipeline keeps running but the
        # cached prefix loses the preamble).
        await _client().post(
            f"{TCMM_URL}/pin/system_prompt",
            json={
                "conversation_id": conv_id,
                "user_id": user_id or "",
                "text": content,
                "kind": kind,
            },
            timeout=TCMM_TIMEOUT,
        )
    except httpx.ConnectError as e:
        _mark_tcmm_unreachable()
        logger.debug(f"[tcmm] pin_system_prompt({kind}) connect failed: {e}")
    except Exception as e:
        logger.debug(f"[tcmm] pin_system_prompt({kind}) failed: {e}")


_TOOL_DEFS_ENDPOINT_AVAILABLE: Optional[bool] = None

# [TCMM_UNREACHABLE_FAST_PATH_2026_05_29]  Track recent ConnectError so
# we stop hammering TCMM endpoints when the service is known-down.  Each
# agent turn does ~3 pin calls + 1 render + 2 ingest = 6 HTTP round-
# trips.  Even at the "fast" connection-refused rate of 250ms/call
# (httpx connection setup + retry probe), that's 1.5s of wasted wall
# clock per turn — directly observable in `[agent.timing]` logs as
# `prepare_session=984ms`.  This flag turns those into <1ms no-ops
# after a single failure, with a 30s re-probe to recover if TCMM comes
# back online.
_TCMM_LAST_CONNECT_FAIL_TS: float = 0.0
_TCMM_REPROBE_AFTER_S: float = 30.0


def _tcmm_unreachable() -> bool:
    """True if a recent TCMM call hit a ConnectError; re-probe after 30s."""
    import time as _t
    return (_t.time() - _TCMM_LAST_CONNECT_FAIL_TS) < _TCMM_REPROBE_AFTER_S


def _mark_tcmm_unreachable() -> None:
    global _TCMM_LAST_CONNECT_FAIL_TS
    import time as _t
    _TCMM_LAST_CONNECT_FAIL_TS = _t.time()


async def pin_tool_definitions(
    conv_id: str, user_id: str, tools: list[dict],
) -> None:
    """Pin tool schemas to TCMM so the rendered prefix includes them.

    Older TCMM builds 404 on this endpoint.  After the first 404 we
    flip a process-local flag and short-circuit every subsequent call —
    this saves ~15ms per agent turn (1× per agent in a multi-agent
    fanout, repeated for every dispatch) that we were burning round-
    tripping to TCMM only to get 404 back.

    Pinning is best-effort: tools are also sent in the API request body,
    so missing the pin only loses cache-stable bundling.
    """
    global _TOOL_DEFS_ENDPOINT_AVAILABLE
    if _TOOL_DEFS_ENDPOINT_AVAILABLE is False:
        return  # known 404 — don't waste a round-trip
    if not conv_id or not tools:
        return
    if _tcmm_unreachable():
        return
    try:
        resp = await _client().post(
            f"{TCMM_URL}/pin/tool_definitions",
            json={
                "conversation_id": conv_id,
                "user_id": user_id or "",
                "tools": tools,
            },
            timeout=TCMM_TIMEOUT,
        )
        # Latch the result so we don't repeat the round-trip.
        if resp.status_code == 404:
            if _TOOL_DEFS_ENDPOINT_AVAILABLE is not False:
                logger.info(
                    "[tcmm] pin/tool_definitions returned 404 once; "
                    "disabling further calls this process lifetime"
                )
            _TOOL_DEFS_ENDPOINT_AVAILABLE = False
        else:
            _TOOL_DEFS_ENDPOINT_AVAILABLE = True
    except httpx.ConnectError as e:
        _mark_tcmm_unreachable()
        logger.debug(f"[tcmm] pin_tool_definitions connect failed: {e}")
    except Exception as e:
        logger.debug(f"[tcmm] pin_tool_definitions failed: {e}")


async def ingest_user(
    conv_id: str, user_id: str, user_msg: str
) -> None:
    """Ingest a user turn into TCMM memory.  Best-effort; logs and
    swallows transport errors so a TCMM hiccup doesn't fail the whole
    agent turn.

    Sends RAW text — TCMM is inside the trust zone per the PII
    boundary design.
    """
    if not conv_id or not user_msg:
        return
    if _tcmm_unreachable():
        return
    try:
        await _client().post(
            f"{TCMM_URL}/pre_request",
            json={
                "user_message": user_msg,
                "conversation_id": conv_id,
                "user_id": user_id or "",
                "recall_only": False,
                "origin": "user",
            },
            timeout=TCMM_TIMEOUT,
        )
    except httpx.ConnectError as e:
        _mark_tcmm_unreachable()
        logger.debug(f"[tcmm] ingest_user connect failed: {e}")
    except Exception as e:
        logger.debug(f"[tcmm] ingest_user failed (continuing): {e}")


async def ingest_assistant(
    conv_id: str,
    user_id: str,
    assistant_text: str,
    *,
    model: str = "",
    flag_obj: Optional[dict] = None,
) -> None:
    """Ingest the assistant reply into TCMM memory.  Best-effort.

    Sends RAW text — same trust-zone rationale as ingest_user.
    flag_obj carries shadow-tool metadata if the model emitted a
    `tcmm_record_turn` tool_use this turn (see
    architecture_universal_shadow_tool memory).
    """
    if not conv_id or not assistant_text:
        return
    body: dict[str, Any] = {
        "raw_output": assistant_text,
        "conversation_id": conv_id,
        "user_id": user_id or "",
        "origin": "assistant_text",
    }
    if model:
        body["model"] = model
    if flag_obj and isinstance(flag_obj, dict):
        body["flag_obj"] = flag_obj
    if _tcmm_unreachable():
        return
    try:
        await _client().post(
            f"{TCMM_URL}/post_response", json=body, timeout=TCMM_TIMEOUT,
        )
    except httpx.ConnectError as e:
        _mark_tcmm_unreachable()
        logger.debug(f"[tcmm] ingest_assistant connect failed: {e}")
    except Exception as e:
        logger.debug(f"[tcmm] ingest_assistant failed (continuing): {e}")


__all__ = [
    "RenderResult",
    "render_structured",
    "pin_system_prompt",
    "pin_tool_definitions",
    "ingest_user",
    "ingest_assistant",
    "close",
    "TCMM_URL",
]
