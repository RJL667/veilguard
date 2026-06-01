"""Audit middleware — runtime-specific wrappers around the shared writer.

The core LanceDB writer + schema live at ``llm/audit_db.py`` (shared
with pii-proxy).  This module adds the agent-runtime-specific bits:

  - ``TurnUsage`` — accumulator populated as the agent's event stream
    flows past (token totals, content chunks, tool calls, stop reason).
  - ``tap_sdk_stream`` — async generator that taps every event into
    ``TurnUsage`` and forwards it upstream unchanged.
  - ``record_turn`` — writes one FROM_LLM summary row at end-of-turn
    with the accumulated ``TurnUsage``.
  - ``record_event`` — writes one row per ``audit`` event yielded by
    ``Agent.run_turn`` (one TO_LLM + one FROM_LLM per LLM round-trip),
    so the inbox-poller dispatch path doesn't silently drop them.

Why both ``record_turn`` AND ``record_event``?
  - ``record_event`` captures the rich per-LLM-call data (full envelope
    content with [SYSTEM] / [TOOLS] / [MESSAGES] sections; tokens per
    call) — what the admin dashboard's per-message timeline needs.
  - ``record_turn`` captures the turn-level summary (total tokens,
    cache hit rate, all tool calls in one row) — what cost roll-up
    queries hit.

Together they let the dashboard show "this turn made 4 LLM calls
totalling X tokens" with click-through to each call's content.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

# [AUDIT_CONTENT_CAP_2026-06-01] The per-row audit content was hard-capped at
# 65 000 chars, which TRUNCATED the TO_LLM envelope mid-[TOOLS] for Director
# turns (system preamble ~26KB + TCMM memory + 103 tool schemas ~100KB easily
# exceeds 65K). The [MESSAGES] section — the user's latest prompt — sits at
# the END of the envelope, so it fell past the cap and never landed in
# pii_audit, making it invisible in the admin drill-down (along with the tail
# of TCMM memory). Bump the default to 1 MB so the full envelope
# ([SYSTEM]+memory / [TOOLS] / [MESSAGES]) is captured. Tunable via env for
# deployments that want to bound audit-row size.
_MAX_AUDIT_CONTENT = int(os.environ.get("AUDIT_MAX_CONTENT_CHARS", "1000000"))

# The shared writer lives at /llm in the container (bind-mounted from
# Documents/veilguard/llm).  Both record() and AuditDB.get() come from
# there; this module no longer carries its own copy of the schema or
# writer thread.
from llm.audit_db import AuditDB, record  # noqa: F401 (record re-exported)

logger = logging.getLogger("agent-runtime.middleware.audit")


@dataclass
class TurnUsage:
    """Accumulator for one agent turn's token + cost stats.

    Populated as the SDK / Agent yields messages.  Final flush at
    end-of-turn writes one summary row to pii_audit via ``record_turn``.

    Anthropic's ``usage`` block on the final assistant message has:
      - input_tokens (NEW only — cache_create + cache_read are separate)
      - output_tokens
      - cache_creation_input_tokens
      - cache_read_input_tokens

    Per ``architecture_token_accounting`` memory: tokens_input in
    pii_audit is stored as TOTAL (input + cache_create + cache_read),
    not NEW-only.  We sum at write time to match the existing
    convention.
    """

    tokens_input_new: int = 0
    tokens_output: int = 0
    cache_create: int = 0
    cache_read: int = 0
    content_chunks: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    stop_reason: str = ""

    @property
    def tokens_input_total(self) -> int:
        return self.tokens_input_new + self.cache_create + self.cache_read

    def cache_hit_rate(self) -> float:
        total = self.tokens_input_total
        return (self.cache_read / total) if total > 0 else 0.0

    def content_text(self) -> str:
        return "".join(self.content_chunks)


# ── Public writers ──────────────────────────────────────────────────────


def record_turn(
    *,
    conversation_id: str,
    user_id: str,
    tenant_id: str,
    agent_id: str,
    task_id: Optional[str],
    parent_cid: Optional[str],
    usage: TurnUsage,
    direction: str = "FROM_LLM",
) -> None:
    """Write one summary audit row for an entire agent turn.

    Captures aggregate token usage + the full tool-call list + the
    final assistant text.  Called once at end-of-turn from
    ``runtime.run_agent_query``.  For per-LLM-call detail (one row per
    Anthropic call inside the multi-iteration tool loop), see
    ``record_event``.
    """
    extra = {
        "agent_id": agent_id,
        "tenant_id": tenant_id,
        "task_id": task_id,
        "parent_cid": parent_cid,
        "stop_reason": usage.stop_reason,
        "tool_calls": [
            {"name": tc.get("name"), "id": tc.get("id")}
            for tc in usage.tool_calls
        ],
        "cache_hit_rate": round(usage.cache_hit_rate(), 4),
        "source": "agent-runtime",
        "kind": "turn_summary",
    }
    record(
        direction=direction,
        conversation_id=conversation_id,
        user_id=user_id,
        model=usage.model,
        stream=True,
        content=usage.content_text()[:_MAX_AUDIT_CONTENT],
        tokens_input=usage.tokens_input_total,
        tokens_output=usage.tokens_output,
        cache_create=usage.cache_create,
        cache_read=usage.cache_read,
        # [TASK_ID_2026_05_26] Promote task_id to the typed column so
        # cost-per-task queries can do a Lance scan with filter
        # pushdown instead of JSON-string parse on extras.
        task_id=task_id,
        extra=extra,
    )


def record_event(
    *,
    audit_ev: dict,
    conversation_id: str,
    user_id: str,
    tenant_id: str,
    agent_id: str,
    task_id: Optional[str],
    parent_cid: Optional[str],
) -> None:
    """Persist a single AUDIT event (TO_LLM | FROM_LLM | APPROVAL)
    yielded by ``Agent.run_turn`` into pii_audit.

    Mirrors ``agent-proxy/app/chat_agent_handler._write_audit`` so the
    admin dashboard's per-message timeline gets the same rows whether
    the agent ran via ChatAgent (pii-proxy) or via the inbox poller
    dispatch (agent-runtime).  Best-effort — never raises.
    """
    try:
        extra = {
            "agent_id": agent_id,
            "tenant_id": tenant_id,
            "task_id": task_id,
            "parent_cid": parent_cid,
            "source": "agent-runtime",
            "kind": "agent_event",
        }
        record(
            direction=audit_ev.get("direction", "") or "",
            conversation_id=conversation_id,
            user_id=user_id,
            model=audit_ev.get("model") or "",
            stream=False,
            content=(audit_ev.get("content") or "")[:_MAX_AUDIT_CONTENT],
            tokens_input=audit_ev.get("tokens_input_total"),
            tokens_output=audit_ev.get("tokens_output"),
            cache_create=audit_ev.get("cache_create"),
            cache_read=audit_ev.get("cache_read"),
            task_id=task_id,
            extra=extra,
        )
    except Exception as e:
        logger.warning(f"[audit] record_event failed: {e}")


# ── SDK stream tapping ──────────────────────────────────────────────────


async def tap_sdk_stream(
    stream: AsyncIterator[Any],
    usage: TurnUsage,
) -> AsyncIterator[Any]:
    """Pass-through wrapper that taps the SDK's / Agent's message iterator.

    For each message we forward upstream UNCHANGED, while also updating
    ``usage`` in-place with whatever stats / content we can extract.
    The shape varies by source (SDK assistant messages vs Agent dict
    events), so we handle both defensively via getattr / .get.
    """
    async for msg in stream:
        msg_type = getattr(msg, "type", None) or (
            msg.get("type") if isinstance(msg, dict) else None
        )

        if msg_type == "assistant":
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
                    text = (
                        getattr(block, "text", None)
                        or (block.get("text") if isinstance(block, dict) else "")
                        or ""
                    )
                    usage.content_chunks.append(text)
                elif btype == "tool_use":
                    name = (
                        getattr(block, "name", None)
                        or (block.get("name") if isinstance(block, dict) else "")
                    )
                    tid = (
                        getattr(block, "id", None)
                        or (block.get("id") if isinstance(block, dict) else "")
                    )
                    usage.tool_calls.append({"name": name, "id": tid})

            model = (
                getattr(inner, "model", None)
                or (inner.get("model") if isinstance(inner, dict) else None)
            )
            if model:
                usage.model = model
            u = (
                getattr(inner, "usage", None)
                or (inner.get("usage") if isinstance(inner, dict) else None)
            )
            if u is not None:
                usage.tokens_input_new += int(
                    getattr(u, "input_tokens", None)
                    or (u.get("input_tokens") if isinstance(u, dict) else 0)
                    or 0
                )
                usage.tokens_output += int(
                    getattr(u, "output_tokens", None)
                    or (u.get("output_tokens") if isinstance(u, dict) else 0)
                    or 0
                )
                usage.cache_create += int(
                    getattr(u, "cache_creation_input_tokens", None)
                    or (u.get("cache_creation_input_tokens") if isinstance(u, dict) else 0)
                    or 0
                )
                usage.cache_read += int(
                    getattr(u, "cache_read_input_tokens", None)
                    or (u.get("cache_read_input_tokens") if isinstance(u, dict) else 0)
                    or 0
                )

        elif msg_type == "result":
            stop_reason = (
                getattr(msg, "stop_reason", None)
                or (msg.get("stop_reason") if isinstance(msg, dict) else "")
                or ""
            )
            if stop_reason:
                usage.stop_reason = stop_reason

        yield msg


__all__ = [
    "TurnUsage",
    "record_turn",
    "record_event",
    "tap_sdk_stream",
    # Re-export the shared singleton + write fn so callers can still do
    # ``from app.middleware.audit import record`` for one-off writes.
    "AuditDB",
    "record",
]
