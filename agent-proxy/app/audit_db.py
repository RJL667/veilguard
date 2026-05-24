"""LanceDB-backed PII audit log.

Writes every TO_LLM / FROM_LLM envelope (full, untruncated) to a
``pii_audit`` table inside the SAME LanceDB directory that TCMM uses.
Schema is multi-tenant — every row carries ``user_id`` +
``conversation_id`` so a single table serves all tenants and queries
filter on those columns.

Design notes
------------
* The pii-proxy and TCMM are separate processes.  LanceDB supports
  multi-process access via file-based table versioning, but we still
  confine ourselves to a DIFFERENT table (``pii_audit``) so our writes
  never collide with TCMM's archive / sparse / dream tables.
* All inserts go through a background daemon thread.  Request handlers
  enqueue and return immediately — LanceDB writes are synchronous and
  would otherwise block FastAPI's event loop (~10-50 ms per row).
* If the audit DB is unreachable (path missing, permission denied,
  Lance on-disk version skew), initialization fails silently and
  ``record()`` becomes a no-op.  The file-based ``audit_log()`` in
  main.py remains as a backstop so we never lose an audit entry
  entirely.
* TCMM owns the tcmm.db directory — this module never reaches into
  TCMM package code.  We speak raw LanceDB.
"""

import json
import logging
import os
import queue
import threading
import time
from typing import Optional

import lancedb
import pyarrow as pa

logger = logging.getLogger("veilguard.audit_db")

# Default points at the TCMM data volume inside the pii-proxy
# container; docker-compose mounts ``./tcmm-data`` there.  Override via
# ``VEILGUARD_AUDIT_DB_PATH`` for dev / alternate deployments.
_DB_PATH = os.environ.get(
    "VEILGUARD_AUDIT_DB_PATH",
    "/tcmm-data/veilguard/tcmm.db",
)
_TABLE = "pii_audit"
_QUEUE_MAX = 10_000           # drop if the writer stalls longer than this
_BATCH_SIZE = 32              # flush when N rows accumulate
_FLUSH_INTERVAL_S = 2.0       # or every N seconds, whichever first

# PyArrow schema.  Plain types, nullable where reasonable.  We keep
# ``aid`` as the primary key to match the TCMM archive convention so
# future joins / tooling feel consistent.
_SCHEMA = pa.schema([
    pa.field("aid",              pa.int64(),   nullable=False),
    pa.field("user_id",          pa.string(),  nullable=True),
    pa.field("conversation_id",  pa.string(),  nullable=True),
    pa.field("direction",        pa.string(),  nullable=False),   # TO_LLM | FROM_LLM
    pa.field("model",            pa.string(),  nullable=True),
    pa.field("stream",           pa.bool_(),   nullable=True),
    pa.field("content",          pa.string(),  nullable=True),    # FULL payload, no truncation
    pa.field("created_at",       pa.float64(), nullable=False),   # unix seconds
    pa.field("tokens_input",     pa.int64(),   nullable=True),
    pa.field("tokens_output",    pa.int64(),   nullable=True),
    pa.field("cache_create",     pa.int64(),   nullable=True),
    pa.field("cache_read",       pa.int64(),   nullable=True),
    pa.field("extra",            pa.string(),  nullable=True),    # JSON for forward-compat
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
                    # Cache a sentinel so we stop retrying.
                    cls._instance = _NullAuditDB()
            return cls._instance

    def __init__(self):
        self._db = lancedb.connect(_DB_PATH)
        try:
            self._tbl = self._db.open_table(_TABLE)
        except Exception:
            # First-run: create the table.
            self._tbl = self._db.create_table(_TABLE, schema=_SCHEMA)
            logger.info(f"[audit_db] created table {_TABLE} at {_DB_PATH}")

        self._aid_lock = threading.Lock()
        self._next_aid = self._compute_next_aid()
        self._q: "queue.Queue[dict]" = queue.Queue(maxsize=_QUEUE_MAX)
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="audit-db-writer", daemon=True
        )
        self._thread.start()
        logger.info(
            f"[audit_db] ready — {_DB_PATH}::{_TABLE} next_aid={self._next_aid}"
        )

    def _compute_next_aid(self) -> int:
        """Pick the next ``aid`` = max(existing) + 1, or 1 on empty table."""
        try:
            arr = self._tbl.to_arrow()
            if arr.num_rows == 0:
                return 1
            return int(max(arr.column("aid").to_pylist())) + 1
        except Exception as e:
            logger.warning(f"[audit_db] aid scan failed; seeding from epoch: {e}")
            # Fallback avoids duplicate aids even if the scan is broken.
            return int(time.time() * 1_000_000)

    def enqueue(self, row: dict) -> None:
        """Assign ``aid`` + ``created_at`` if missing, push to background queue.

        Drops the row (with a warning) if the queue is full — means the
        writer thread is stalled.  Migration runs may push thousands
        rapidly; the queue cap prevents unbounded memory growth.
        """
        with self._aid_lock:
            row.setdefault("aid", self._next_aid)
            if row["aid"] >= self._next_aid:
                self._next_aid = row["aid"] + 1
        row.setdefault("created_at", time.time())
        # Every field must be present for PyArrow's schema-enforced
        # insert.  Missing keys → None is safer than a KeyError.
        for field in _SCHEMA.names:
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
            self._tbl.add(batch)
        except Exception as e:
            logger.warning(
                f"[audit_db] flush failed ({len(batch)} rows): {e}"
            )


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
    extra: Optional[dict] = None,
) -> None:
    """Write one audit row to the LanceDB queue (non-blocking)."""
    db = AuditDB.get()
    if db is None:
        return
    db.enqueue({
        "user_id":         user_id or "",
        "conversation_id": conversation_id or "",
        "direction":       direction,
        "model":            model,
        "stream":           stream,
        "content":          content or "",
        "tokens_input":     tokens_input,
        "tokens_output":    tokens_output,
        "cache_create":     cache_create,
        "cache_read":       cache_read,
        "extra":            json.dumps(extra) if extra else None,
    })
