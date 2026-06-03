"""Shared LanceDB-backed PII audit writer.

Single source of truth for the ``pii_audit`` table that lives inside
the TCMM data volume.  Used by:
  - ``agent-proxy/app/audit_db.py``   (thin re-export shim)
  - ``agent-runtime/app/middleware/audit.py``  (runtime-specific
    wrappers — TurnUsage / record_turn / record_event / tap_sdk_stream
    — call into ``record()`` from here)

Why this exists
---------------
Before unification, pii-proxy and agent-runtime each carried their own
copy of the schema + writer thread + aid counter.  Two problems:
  1. Schema drift — adding a column required edits in TWO places, and
     they silently fell out of sync.
  2. AID collision — each process computed ``next_aid = max(existing)
     + 1`` independently, so two processes writing in the same second
     could pick the same aid.  Lance accepted both (no PK enforcement)
     but the dashboard's aid-keyed sort got non-deterministic.

This module fixes both: ONE schema, ONE writer per process (still — we
can't share threads across containers), but with ``time.time_ns()``
seeded ``next_aid`` so cross-process collisions are vanishingly rare
(<1 in a billion per nanosecond of overlap).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from typing import Optional

import pyarrow as pa

logger = logging.getLogger("veilguard.audit_db")

# Backend select (mirrors the TCMM/ledger Postgres switch). When
# VEILGUARD_AUDIT_BACKEND (or TCMM_STORAGE) is postgres, pii_audit is a real
# Postgres table; otherwise the legacy LanceDB table. Same record() API either way.
_AUDIT_BACKEND = os.environ.get(
    "VEILGUARD_AUDIT_BACKEND", os.environ.get("TCMM_STORAGE", "lance")
).lower()
_PG = _AUDIT_BACKEND in ("postgres", "postgresql")
_AUDIT_DSN = os.environ.get(
    "VEILGUARD_AUDIT_DATABASE_URL",
    os.environ.get("TCMM_DATABASE_URL", "postgresql://tcmm:tcmm@localhost:5432/tcmm"),
)
_PG_DDL = """
CREATE TABLE IF NOT EXISTS pii_audit (
    aid             BIGINT PRIMARY KEY,
    user_id         TEXT,
    conversation_id TEXT,
    direction       TEXT NOT NULL,
    model           TEXT,
    stream          BOOLEAN,
    content         TEXT,
    created_at      DOUBLE PRECISION NOT NULL,
    tokens_input    BIGINT,
    tokens_output   BIGINT,
    cache_create    BIGINT,
    cache_read      BIGINT,
    task_id         TEXT,
    extra           TEXT
);
CREATE INDEX IF NOT EXISTS ix_pii_created      ON pii_audit (created_at);
CREATE INDEX IF NOT EXISTS ix_pii_user_created ON pii_audit (user_id, created_at);
CREATE INDEX IF NOT EXISTS ix_pii_conv         ON pii_audit (conversation_id);
"""


# Default points at the TCMM data volume which is bind-mounted into
# both pii-proxy + agent-runtime containers at ``/tcmm-data``.  Override
# via ``VEILGUARD_AUDIT_DB_PATH`` for dev / alternate deployments.
_DB_PATH = os.environ.get(
    "VEILGUARD_AUDIT_DB_PATH",
    "/tcmm-data/veilguard/tcmm.db",
)
_TABLE = "pii_audit"

_QUEUE_MAX = 10_000           # drop if the writer stalls longer than this
_BATCH_SIZE = 32              # flush when N rows accumulate
_FLUSH_INTERVAL_S = 2.0       # or every N seconds, whichever first


# Canonical schema — ANY column add/rename MUST happen here, not in a
# downstream wrapper.  Keep nullable as broad as possible so legacy
# rows from before-this-column added stay readable.
SCHEMA = pa.schema([
    pa.field("aid",              pa.int64(),   nullable=False),
    pa.field("user_id",          pa.string(),  nullable=True),
    pa.field("conversation_id",  pa.string(),  nullable=True),
    pa.field("direction",        pa.string(),  nullable=False),  # TO_LLM | FROM_LLM | APPROVAL
    pa.field("model",            pa.string(),  nullable=True),
    pa.field("stream",           pa.bool_(),   nullable=True),
    pa.field("content",          pa.string(),  nullable=True),   # FULL payload, no truncation
    pa.field("created_at",       pa.float64(), nullable=False),  # unix seconds
    pa.field("tokens_input",     pa.int64(),   nullable=True),
    pa.field("tokens_output",    pa.int64(),   nullable=True),
    pa.field("cache_create",     pa.int64(),   nullable=True),
    pa.field("cache_read",       pa.int64(),   nullable=True),
    # [TASK_ID_2026_05_26] Per Phase 2 §3.10 "Cost roll-up depends on
    # trace_ref linking Tasks to pii_audit rows." Without this column,
    # cost-per-Task is structurally unknowable and the constitution's
    # `cost_ceiling_per_task` constraint cannot be enforced.  Nullable
    # so legacy rows stay readable; populated by the inbox_poller
    # dispatch path (every IC sub-call carries its task_id) and the
    # ledger_mcp tool layer.  ChatAgent / SSO turns that aren't bound
    # to a Task leave it NULL.
    pa.field("task_id",          pa.string(),  nullable=True),
    pa.field("extra",            pa.string(),  nullable=True),   # JSON for forward-compat
])


class AuditDB:
    """Singleton writer.  Lazily initialised; silently no-ops on failure."""

    _instance: Optional["AuditDB"] = None
    _init_lock = threading.Lock()

    @classmethod
    def get(cls) -> Optional["AuditDB"]:
        if cls._instance is not None:
            return cls._instance
        with cls._init_lock:
            if cls._instance is None:
                try:
                    cls._instance = cls()
                except Exception as e:
                    logger.warning(
                        f"[audit_db] init failed; DB audit disabled: {e}"
                    )
                    cls._instance = _NullAuditDB()  # type: ignore[assignment]
            return cls._instance

    def __init__(self):
        self._pg = _PG
        if self._pg:
            import psycopg2
            self._conn = psycopg2.connect(_AUDIT_DSN)
            self._conn.autocommit = True
            with self._conn.cursor() as c:
                c.execute(_PG_DDL)
            logger.info(f"[audit_db] postgres backend — ::{_TABLE}")
        else:
            import lancedb  # lazy: only the Lance backend needs it
            self._db = lancedb.connect(_DB_PATH)
            try:
                self._tbl = self._db.open_table(_TABLE)
                # [TASK_ID_MIGRATION_2026_05_26] Idempotent schema-add for the
                # task_id column.  Lance allows adding nullable columns to an
                # existing table; existing rows get NULL automatically.
                try:
                    existing_cols = set(self._tbl.schema.names)
                    if "task_id" not in existing_cols:
                        self._tbl.add_columns({"task_id": "cast(NULL as string)"})
                        logger.info(f"[audit_db] migrated {_TABLE}: added task_id column")
                except Exception as e:
                    logger.warning(f"[audit_db] task_id column add skipped: {e}")
            except Exception:
                self._tbl = self._db.create_table(_TABLE, schema=SCHEMA)
                logger.info(f"[audit_db] created table {_TABLE} at {_DB_PATH}")

        self._aid_lock = threading.Lock()
        # Microsecond-resolution seed (NOT nanosecond — nanosecond aids
        # overflow JavaScript's Number.MAX_SAFE_INTEGER (2^53-1) and the
        # admin dashboard's JSON parser loses precision, which collapses
        # all rows to the same x-for :key and silently empties the
        # Recent Requests table).  Microsecond aids land at ~1.78e15
        # which is well inside JS's safe int range (~9e15) for ~280
        # years.  Still monotonic + cross-process-collision rare
        # (<1 in a million per μs of seed overlap).
        self._next_aid = int(time.time() * 1_000_000)
        self._q: "queue.Queue[dict]" = queue.Queue(maxsize=_QUEUE_MAX)
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="audit-db-writer", daemon=True
        )
        self._thread.start()
        logger.info(
            f"[audit_db] ready — {_DB_PATH}::{_TABLE} next_aid={self._next_aid}"
        )

    def enqueue(self, row: dict) -> None:
        """Assign ``aid`` + ``created_at`` if missing, push to background queue."""
        with self._aid_lock:
            row.setdefault("aid", self._next_aid)
            if row["aid"] >= self._next_aid:
                self._next_aid = row["aid"] + 1
        row.setdefault("created_at", time.time())
        # Every field must be present for PyArrow's schema-enforced
        # insert.  Missing keys → None is safer than a KeyError.
        for field in SCHEMA.names:
            row.setdefault(field, None)
        try:
            self._q.put_nowait(row)
        except queue.Full:
            logger.warning("[audit_db] queue full; dropping row")

    def _run(self):
        batch: list[dict] = []
        last_flush = time.time()
        while not self._stop_evt.is_set():
            timeout = max(0.0, _FLUSH_INTERVAL_S - (time.time() - last_flush))
            try:
                row = self._q.get(timeout=timeout or 0.1)
                batch.append(row)
            except queue.Empty:
                pass
            now = time.time()
            full_enough = len(batch) >= _BATCH_SIZE
            time_up = batch and (now - last_flush) >= _FLUSH_INTERVAL_S
            if full_enough or time_up:
                self._flush(batch)
                batch = []
                last_flush = now
        if batch:
            self._flush(batch)

    def _flush(self, batch: list[dict]):
        try:
            if self._pg:
                from psycopg2.extras import execute_values
                cols = list(SCHEMA.names)
                vals = [tuple(row.get(c) for c in cols) for row in batch]
                with self._conn.cursor() as c:
                    execute_values(
                        c,
                        f"INSERT INTO {_TABLE} ({', '.join(cols)}) VALUES %s "
                        f"ON CONFLICT (aid) DO NOTHING",
                        vals,
                    )
            else:
                self._tbl.add(batch)
        except Exception as e:
            logger.warning(f"[audit_db] flush failed ({len(batch)} rows): {e}")


class _NullAuditDB:
    """Sentinel used when initialisation failed; enqueue is a no-op."""

    def enqueue(self, row: dict) -> None:  # noqa: D401
        return None


def record(
    *,
    direction: str,
    conversation_id: str,
    content: str,
    user_id: str = "",
    model: Optional[str] = None,
    stream: bool = False,
    tokens_input: Optional[int] = None,
    tokens_output: Optional[int] = None,
    cache_create: Optional[int] = None,
    cache_read: Optional[int] = None,
    task_id: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    """Write one audit row to the LanceDB queue (non-blocking).

    [F9_AUDIT_TAG_GUARD_2026_05_26] Refuses to write rows whose user_id
    is a known sentinel ("anonymous", "default", "") or an email-shape
    string — those landed in pii_audit invisible to tenant filters
    keyed on hex-24 tenant ids and could cross-leak.  Production
    callers must pass a real user/tenant identifier; we'd rather drop
    the row + log loudly than persist an unattributable audit entry.
    Test / dev probe scripts that hit audit_db.record directly should
    set user_id to a hex-24 string (the test tenant) so they survive
    the guard.
    """
    db = AuditDB.get()
    if db is None:
        return
    _uid = (user_id or "").strip()
    if not _uid or _uid in ("anonymous", "default", "system") or "@" in _uid:
        # Don't persist unattributable rows.  Log once with enough
        # detail to find the caller, then drop.
        import logging as _log
        _log.getLogger("audit_db").warning(
            f"[audit_db] refusing to persist row with bad user_id={_uid!r} "
            f"(direction={direction}, conv_id={conversation_id!r}); "
            f"caller must supply a real tenant identifier"
        )
        return
    db.enqueue({
        "user_id":         _uid,
        "conversation_id": conversation_id or "",
        "direction":       direction,
        "model":            model,
        "stream":           stream,
        "content":          content or "",
        "tokens_input":     tokens_input,
        "tokens_output":    tokens_output,
        "cache_create":     cache_create,
        "cache_read":       cache_read,
        "task_id":          task_id or None,
        "extra":            json.dumps(extra) if extra else None,
    })


__all__ = ["SCHEMA", "AuditDB", "record"]
