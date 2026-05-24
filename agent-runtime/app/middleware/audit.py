"""Audit middleware — wraps SDK query() to capture tokens + decisions.

The Claude Agent SDK doesn't expose a post-LLM-response hook (only
PreToolUse / PostToolUse).  To capture token usage + the assistant
turn for audit / cost roll-up, we consume the SDK's async-iterator
ourselves and tap each message before forwarding it upstream.

Writes audit rows to the SAME `pii_audit` Lance table the proxy
writes to (per spec §3.10 and existing `audit_db.py` contract).  This
keeps cost queries single-table.

Schema fields we write per call:
  - aid (auto-assigned)
  - user_id, conversation_id (from TenantContext)
  - direction = "FROM_LLM" for the assistant response capture
  - model, stream, content
  - tokens_input, tokens_output, cache_create, cache_read
  - extra (JSON): {agent_id, tenant_id, task_id, parent_cid, ...}

Note: pii-proxy writes the TO_LLM direction; we write FROM_LLM.  Both
land in the same table.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from ..config import AUDIT_DB_PATH

logger = logging.getLogger("agent-runtime.middleware.audit")


@dataclass
class TurnUsage:
    """Accumulator for one agent turn's token + cost stats.

    Populated as the SDK yields messages.  Final flush at end-of-turn
    writes one row to pii_audit.

    Anthropic's `usage` block on the final assistant message has:
      - input_tokens (NEW only — cache_create + cache_read are separate)
      - output_tokens
      - cache_creation_input_tokens
      - cache_read_input_tokens

    Per `architecture_token_accounting` memory: tokens_input in pii_audit
    is stored as TOTAL (input + cache_create + cache_read), not NEW-only.
    We sum at write time to match the existing convention.
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


# ── Audit DB writer (background thread, drops on overflow) ───────────────
# Mirrors agent-proxy/app/audit_db.py shape.  Lazy init so missing Lance
# dir (dev / tests) doesn't crash the service; the queue just no-ops.


class _AuditWriter:
    _instance: Optional["_AuditWriter"] = None
    _init_lock = threading.Lock()

    @classmethod
    def get(cls) -> Optional["_AuditWriter"]:
        if cls._instance is not None:
            return cls._instance
        with cls._init_lock:
            if cls._instance is None:
                try:
                    cls._instance = cls()
                except Exception as e:
                    logger.warning(
                        f"[audit] init failed; audit disabled: {e}"
                    )
                    cls._instance = _NullWriter()
            return cls._instance

    def __init__(self):
        # Defer the heavy imports so a missing lancedb (e.g., in unit
        # tests with mocks) doesn't crash module import.
        import lancedb
        import pyarrow as pa

        self._db_path = str(AUDIT_DB_PATH)
        self._table_name = "pii_audit"
        self._db = lancedb.connect(self._db_path)

        # Schema MUST match agent-proxy/app/audit_db.py _SCHEMA.
        self._schema = pa.schema([
            pa.field("aid", pa.int64(), nullable=False),
            pa.field("user_id", pa.string(), nullable=True),
            pa.field("conversation_id", pa.string(), nullable=True),
            pa.field("direction", pa.string(), nullable=False),
            pa.field("model", pa.string(), nullable=True),
            pa.field("stream", pa.bool_(), nullable=True),
            pa.field("content", pa.string(), nullable=True),
            pa.field("created_at", pa.float64(), nullable=False),
            pa.field("tokens_input", pa.int64(), nullable=True),
            pa.field("tokens_output", pa.int64(), nullable=True),
            pa.field("cache_create", pa.int64(), nullable=True),
            pa.field("cache_read", pa.int64(), nullable=True),
            pa.field("extra", pa.string(), nullable=True),
        ])

        try:
            self._tbl = self._db.open_table(self._table_name)
        except Exception:
            self._tbl = self._db.create_table(self._table_name, schema=self._schema)
            logger.info(
                f"[audit] created table {self._table_name} at {self._db_path}"
            )

        self._aid_lock = threading.Lock()
        self._next_aid = self._compute_next_aid()
        self._q: queue.Queue = queue.Queue(maxsize=10_000)
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="agent-runtime-audit", daemon=True
        )
        self._thread.start()
        logger.info(
            f"[audit] writer ready — {self._db_path}::{self._table_name} "
            f"next_aid={self._next_aid}"
        )

    def _compute_next_aid(self) -> int:
        try:
            arr = self._tbl.to_arrow()
            if arr.num_rows == 0:
                return 1
            return int(max(arr.column("aid").to_pylist())) + 1
        except Exception:
            return int(time.time() * 1_000_000)

    def enqueue(self, row: dict) -> None:
        with self._aid_lock:
            row.setdefault("aid", self._next_aid)
            if row["aid"] >= self._next_aid:
                self._next_aid = row["aid"] + 1
        row.setdefault("created_at", time.time())
        for field_name in self._schema.names:
            row.setdefault(field_name, None)
        try:
            self._q.put_nowait(row)
        except queue.Full:
            logger.warning("[audit] queue full; dropping row")

    def _run(self):
        batch: list[dict] = []
        last_flush = time.time()
        while not self._stop_evt.is_set():
            timeout = max(0.0, 2.0 - (time.time() - last_flush))
            try:
                row = self._q.get(timeout=timeout or 0.1)
                batch.append(row)
            except queue.Empty:
                pass
            now = time.time()
            if len(batch) >= 32 or (batch and now - last_flush >= 2.0):
                try:
                    self._tbl.add(batch)
                except Exception as e:
                    logger.warning(
                        f"[audit] flush failed ({len(batch)} rows): {e}"
                    )
                batch = []
                last_flush = now


class _NullWriter:
    def enqueue(self, row: dict) -> None:
        return None


# ── Public API ───────────────────────────────────────────────────────────


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
    """Write one audit row capturing this agent turn's usage.

    Direction default is FROM_LLM (we're writing the response capture).
    Caller may pass TO_LLM for cases where agent-runtime is also
    responsible for the prompt audit (pii-proxy normally handles that).
    """
    writer = _AuditWriter.get()
    if writer is None:
        return

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
    }

    writer.enqueue({
        "user_id": user_id,
        "conversation_id": conversation_id,
        "direction": direction,
        "model": usage.model,
        "stream": True,
        "content": usage.content_text()[:65_000],  # safety truncation
        "tokens_input": usage.tokens_input_total,
        "tokens_output": usage.tokens_output,
        "cache_create": usage.cache_create,
        "cache_read": usage.cache_read,
        "extra": json.dumps(extra),
    })


async def tap_sdk_stream(
    stream: AsyncIterator[Any],
    usage: TurnUsage,
) -> AsyncIterator[Any]:
    """Pass-through wrapper that taps the SDK's message iterator.

    For each message we forward upstream UNCHANGED, while also updating
    `usage` (in-place) with whatever stats / content we can extract.

    The SDK's message types vary by version; we handle the common
    shapes defensively (getattr-based).
    """
    async for msg in stream:
        # AssistantMessage — capture content chunks + tool calls
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

            # Capture model + usage if present on inner message
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

        # Always forward upstream unchanged.
        yield msg


__all__ = [
    "TurnUsage",
    "record_turn",
    "tap_sdk_stream",
]
