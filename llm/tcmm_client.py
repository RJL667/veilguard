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
) -> RenderResult:
    """Render the conversation's memory into SDK-ready blocks.

    Returns RAW (pre-redaction) blocks with cache_control markers
    placed by TCMM's renderer.  Caller is expected to run the blocks
    through `pii.PIIRedactor.redact_blocks(...)` before sending to the
    LLM adapter — that's the cache-stable boundary.
    """
    body = {
        "conversation_id": conv_id,
        "user_id": user_id or "",
        "task_query": task_query,
        "model": model,
    }
    try:
        resp = await _client().post(f"{TCMM_URL}/render_structured", json=body)
    except httpx.ConnectError as e:
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
    except Exception as e:
        logger.debug(f"[tcmm] pin_system_prompt({kind}) failed: {e}")


async def pin_tool_definitions(
    conv_id: str, user_id: str, tools: list[dict],
) -> None:
    """Pin tool schemas to TCMM so the rendered prefix includes them.

    Older TCMM builds may 404 on this endpoint; we log and continue.
    Pinning is best-effort: tools are also sent in the API request
    body, so missing the pin only loses cache-stable bundling.
    """
    if not conv_id or not tools:
        return
    try:
        await _client().post(
            f"{TCMM_URL}/pin/tool_definitions",
            json={
                "conversation_id": conv_id,
                "user_id": user_id or "",
                "tools": tools,
            },
            timeout=TCMM_TIMEOUT,
        )
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
    try:
        await _client().post(
            f"{TCMM_URL}/post_response", json=body, timeout=TCMM_TIMEOUT,
        )
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
