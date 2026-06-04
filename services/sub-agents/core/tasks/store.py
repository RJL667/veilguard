"""Lance-backed persistence for the unified Task graph.

Single table ``tasks`` in the existing Veilguard Lance DB (sibling to
pii_audit / pii_session_mapping / archive).  BTREE scalar indexes on
the hot lookup columns (user_id, parent_task_id, root_task_id, status)
keep queries sub-millisecond even at millions of rows — same scale
pattern documented in pii_session_store.

Concurrency: singleton per process via ``get()``.  Writes use
``merge_insert(["task_id"])`` so concurrent updates from multiple
workers don't double-row.  Reads use Lance filter pushdown — no
to_arrow() full scans.

Eviction: NOT YET.  We leave terminal rows in place so the LLM /
tray can scroll history.  Phase B-2 will add a TTL sweep + a
``materialized_view`` for "recent N completed".  Current expected
volume (tool_call per chat turn × users) is tractable for years.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

import pyarrow as pa

from .model import Task, TaskStatus

logger = logging.getLogger("veilguard.tasks.store")

# Backend select — Postgres when TCMM_STORAGE / VEILGUARD_AUDIT_BACKEND=postgres,
# else the legacy LanceDB table. Same TaskStore API either way.
_PG = os.environ.get(
    "TCMM_STORAGE", os.environ.get("VEILGUARD_AUDIT_BACKEND", "postgres")
).lower() in ("postgres", "postgresql")
_PG_DSN = os.environ.get(
    "VEILGUARD_TASKS_DATABASE_URL",
    os.environ.get("TCMM_DATABASE_URL", "postgresql://tcmm:tcmm@localhost:5432/tcmm"),
)
_PG_DDL = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id          TEXT PRIMARY KEY,
    user_id          TEXT, tenant_id TEXT, conv_id TEXT,
    parent_task_id   TEXT, root_task_id TEXT,
    depends_on       TEXT[],
    kind             TEXT, executor TEXT, title TEXT, description TEXT,
    payload_json     TEXT, status TEXT,
    created_at       DOUBLE PRECISION, started_at DOUBLE PRECISION, finished_at DOUBLE PRECISION,
    result           TEXT, error TEXT,
    interval_seconds BIGINT, next_run_at DOUBLE PRECISION, extras_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_tasks_user   ON tasks(user_id);
CREATE INDEX IF NOT EXISTS ix_tasks_parent ON tasks(parent_task_id);
CREATE INDEX IF NOT EXISTS ix_tasks_root   ON tasks(root_task_id);
CREATE INDEX IF NOT EXISTS ix_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS ix_tasks_kind   ON tasks(kind);
"""


# ── Schema ───────────────────────────────────────────────────────────────


_TABLE_NAME = "tasks"

# Strings everywhere there's an enum so we don't have to migrate when
# we add a new TaskKind / TaskStatus value.  int64 for ts so Arrow's
# float64 epoch values round-trip cleanly.  Lists stored as Utf8 list.
_SCHEMA = pa.schema([
    ("task_id",          pa.string()),
    ("user_id",          pa.string()),
    ("tenant_id",        pa.string()),
    ("conv_id",          pa.string()),
    ("parent_task_id",   pa.string()),
    ("root_task_id",     pa.string()),
    ("depends_on",       pa.list_(pa.string())),
    ("kind",             pa.string()),
    ("executor",         pa.string()),
    ("title",            pa.string()),
    ("description",      pa.string()),
    ("payload_json",     pa.string()),     # JSON of Task.payload dict
    ("status",           pa.string()),
    ("created_at",       pa.float64()),
    ("started_at",       pa.float64()),
    ("finished_at",      pa.float64()),
    ("result",           pa.string()),
    ("error",            pa.string()),
    ("interval_seconds", pa.int64()),
    ("next_run_at",      pa.float64()),
    ("extras_json",      pa.string()),
])

_INDEXED_COLS = ("user_id", "parent_task_id", "root_task_id", "status",
                 "kind", "task_id")


# ── Row codec ─────────────────────────────────────────────────────────────


def _task_to_row(t: Task) -> dict:
    """Encode a Task into the Lance row layout.  payload becomes JSON
    because Lance's nested-dict story is still messy with merge_insert."""
    return {
        "task_id":         t.task_id,
        "user_id":         t.user_id,
        "tenant_id":       t.tenant_id,
        "conv_id":         t.conv_id,
        "parent_task_id":  t.parent_task_id,
        "root_task_id":    t.root_task_id,
        "depends_on":      list(t.depends_on or []),
        "kind":            t.kind,
        "executor":        t.executor,
        "title":           t.title,
        "description":     t.description,
        "payload_json":    json.dumps(t.payload, ensure_ascii=False) if t.payload else "",
        "status":          t.status,
        "created_at":      float(t.created_at or 0),
        "started_at":      float(t.started_at or 0),
        "finished_at":     float(t.finished_at or 0),
        "result":          t.result or "",
        "error":           t.error or "",
        "interval_seconds": int(t.interval_seconds or 0),
        "next_run_at":     float(t.next_run_at or 0),
        "extras_json":     t.extras_json or "",
    }


def _row_to_task(r: dict) -> Task:
    """Decode a Lance row back to a Task."""
    payload = {}
    if r.get("payload_json"):
        try:
            payload = json.loads(r["payload_json"])
        except Exception:
            payload = {}
    return Task(
        task_id=r.get("task_id", ""),
        user_id=r.get("user_id", ""),
        tenant_id=r.get("tenant_id", ""),
        conv_id=r.get("conv_id", ""),
        parent_task_id=r.get("parent_task_id", ""),
        root_task_id=r.get("root_task_id", ""),
        depends_on=list(r.get("depends_on") or []),
        kind=r.get("kind", ""),
        executor=r.get("executor", ""),
        title=r.get("title", ""),
        description=r.get("description", ""),
        payload=payload,
        status=r.get("status", TaskStatus.PENDING.value),
        created_at=float(r.get("created_at") or 0) or time.time(),
        started_at=float(r.get("started_at") or 0) or None,
        finished_at=float(r.get("finished_at") or 0) or None,
        result=r.get("result", ""),
        error=r.get("error", ""),
        interval_seconds=int(r.get("interval_seconds") or 0),
        next_run_at=float(r.get("next_run_at") or 0) or None,
        extras_json=r.get("extras_json", ""),
    )


# ── Store ─────────────────────────────────────────────────────────────────


class TaskStore:
    """Lance-backed Task persistence + tree queries."""

    _instance: Optional["TaskStore"] = None
    _instance_lock = threading.Lock()

    def __init__(self, db_path: Optional[str] = None):
        self._pg = _PG
        if self._pg:
            import psycopg2
            self._conn = psycopg2.connect(_PG_DSN)
            self._conn.autocommit = True
            with self._conn.cursor() as c:
                c.execute(_PG_DDL)
            logger.info("[tasks.store] postgres backend")
            return
        import lancedb  # lazy: only the Lance backend needs it
        path = (
            db_path
            or os.environ.get("VEILGUARD_TASKS_DB_PATH")
            or os.environ.get("VEILGUARD_PII_DB_PATH")
            or os.environ.get("LEDGER_DB_PATH")
            or "/tcmm-data/veilguard/tcmm.db"
        )
        Path(path).mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(path)
        # list_tables() returns paginated tuple in newer lancedb; flatten
        existing = self._list_table_names()
        if _TABLE_NAME not in existing:
            empty = pa.Table.from_pylist([], schema=_SCHEMA)
            self._tbl = self._db.create_table(_TABLE_NAME, empty, mode="overwrite")
            logger.info(f"[tasks.store] created table {_TABLE_NAME} at {path}")
        else:
            self._tbl = self._db.open_table(_TABLE_NAME)
        self._ensure_indexes()

    @classmethod
    def get(cls) -> "TaskStore":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _list_table_names(self) -> list[str]:
        """Normalise both flat-list (old lancedb) and paginated (new) shapes."""
        out = self._db.list_tables()
        if isinstance(out, list):
            return out
        # newer shape: [('page_token', None), ('tables', [...])]
        try:
            for k, v in out:
                if k == "tables" and isinstance(v, list):
                    return v
        except Exception:
            pass
        return []

    def _pg_upsert(self, rows: list[dict]) -> None:
        cols = list(_SCHEMA.names)
        setc = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c != "task_id")
        ph = ", ".join(["%s"] * len(cols))
        with self._conn.cursor() as cur:
            for row in rows:
                cur.execute(
                    f"INSERT INTO tasks ({', '.join(cols)}) VALUES ({ph}) "
                    f"ON CONFLICT (task_id) DO UPDATE SET {setc}",
                    [row.get(c) for c in cols])

    def _pg_select(self, where: str, params: list, limit: int) -> list[Task]:
        import psycopg2.extras
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"SELECT * FROM tasks WHERE {where} ORDER BY created_at LIMIT %s",
                        list(params) + [int(limit)])
            return [_row_to_task(dict(r)) for r in cur.fetchall()]

    def _ensure_indexes(self) -> None:
        """Idempotent BTREE creation, same pattern as pii_session_store."""
        if self._pg:
            return  # Postgres manages its own indexes (DDL in __init__)
        try:
            existing = {idx.columns[0] for idx in self._tbl.list_indices()
                        if idx.columns}
        except Exception as e:
            logger.warning(f"[tasks.store] list_indices failed: {e}")
            existing = set()
        for col in _INDEXED_COLS:
            if col in existing:
                continue
            try:
                self._tbl.create_scalar_index(
                    col, index_type="BTREE", replace=True,
                )
                logger.info(f"[tasks.store] created BTREE on {_TABLE_NAME}.{col}")
            except Exception as e:
                logger.warning(f"[tasks.store] create_scalar_index({col}) failed: {e}")

    # ── Writes ─────────────────────────────────────────────────────────

    def upsert(self, task: Task) -> None:
        """Insert or update a task by primary key (task_id)."""
        row = _task_to_row(task)
        if self._pg:
            self._pg_upsert([row]); return
        try:
            (
                self._tbl.merge_insert("task_id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute([row])
            )
        except Exception as e:
            logger.error(f"[tasks.store] upsert failed for {task.task_id}: {e}")
            raise

    def upsert_many(self, tasks: Iterable[Task]) -> int:
        """Batch upsert.  Returns rows written."""
        rows = [_task_to_row(t) for t in tasks]
        if not rows:
            return 0
        if self._pg:
            self._pg_upsert(rows); return len(rows)
        try:
            (
                self._tbl.merge_insert("task_id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(rows)
            )
            return len(rows)
        except Exception as e:
            logger.error(f"[tasks.store] upsert_many failed ({len(rows)}): {e}")
            raise

    # ── Reads ──────────────────────────────────────────────────────────

    def get_task(self, task_id: str) -> Optional[Task]:
        if not task_id:
            return None
        if self._pg:
            r = self._pg_select("task_id = %s", [task_id], 1)
            return r[0] if r else None
        arr = (
            self._tbl.search()
            .where(f"task_id = '{_esc(task_id)}'")
            .limit(1)
            .to_arrow()
        )
        if arr.num_rows == 0:
            return None
        return _row_to_task(_first_row_to_dict(arr))

    def list_user(
        self, user_id: str, *,
        statuses: Optional[Iterable[str]] = None,
        kinds: Optional[Iterable[str]] = None,
        since_ts: float = 0,
        limit: int = 200,
    ) -> list[Task]:
        """List a user's tasks, optionally filtered by status/kind/since."""
        if not user_id:
            return []
        if self._pg:
            where = "user_id = %s"; params: list = [user_id]
            if statuses:
                where += " AND status = ANY(%s)"; params.append([str(s) for s in statuses])
            if kinds:
                where += " AND kind = ANY(%s)"; params.append([str(k) for k in kinds])
            if since_ts > 0:
                where += " AND created_at >= %s"; params.append(float(since_ts))
            return self._pg_select(where, params, limit)
        clauses = [f"user_id = '{_esc(user_id)}'"]
        if statuses:
            ss = ",".join(f"'{_esc(s)}'" for s in statuses)
            clauses.append(f"status IN ({ss})")
        if kinds:
            ks = ",".join(f"'{_esc(k)}'" for k in kinds)
            clauses.append(f"kind IN ({ks})")
        if since_ts > 0:
            clauses.append(f"created_at >= {float(since_ts)}")
        where = " AND ".join(clauses)
        arr = (
            self._tbl.search()
            .where(where)
            .limit(limit)
            .to_arrow()
        )
        return _arrow_to_tasks(arr)

    def list_subtree(self, root_task_id: str, *, limit: int = 500) -> list[Task]:
        """All tasks (root + descendants) sharing a root_task_id."""
        if not root_task_id:
            return []
        if self._pg:
            return self._pg_select("root_task_id = %s", [root_task_id], limit)
        arr = (
            self._tbl.search()
            .where(f"root_task_id = '{_esc(root_task_id)}'")
            .limit(limit)
            .to_arrow()
        )
        return _arrow_to_tasks(arr)

    def list_children(self, parent_task_id: str, *, limit: int = 500) -> list[Task]:
        """Direct children only (one level deep)."""
        if not parent_task_id:
            return []
        if self._pg:
            return self._pg_select("parent_task_id = %s", [parent_task_id], limit)
        arr = (
            self._tbl.search()
            .where(f"parent_task_id = '{_esc(parent_task_id)}'")
            .limit(limit)
            .to_arrow()
        )
        return _arrow_to_tasks(arr)

    # ── Atomic status flip ─────────────────────────────────────────────

    def mark_status(
        self, task_id: str, status: TaskStatus | str,
        *, error: str = "", result: str = "",
    ) -> bool:
        """Read-modify-write of just the status field + optional result/error.

        Returns True on success, False when the task doesn't exist.
        Single-shot; concurrent flippers can race but the last writer
        wins which is fine for our use case (terminal states stick).
        """
        t = self.get_task(task_id)
        if t is None:
            return False
        if t.is_terminal:
            return False
        t.status = status.value if isinstance(status, TaskStatus) else str(status)
        now = time.time()
        if t.started_at is None and t.status == TaskStatus.RUNNING.value:
            t.started_at = now
        if t.status in TaskStatus.terminal():
            t.finished_at = now
        if error:
            t.error = error[:8000]
        if result:
            t.result = result[:8000]
        self.upsert(t)
        return True


# ── Helpers ─────────────────────────────────────────────────────────────


def _esc(s: str) -> str:
    """SQL-string escape (single-quote doubling) for use in Lance where."""
    return (s or "").replace("'", "''")


def _arrow_to_tasks(arr) -> list[Task]:
    out: list[Task] = []
    if arr is None or arr.num_rows == 0:
        return out
    cols = {name: arr.column(name).to_pylist() for name in arr.schema.names}
    for i in range(arr.num_rows):
        row = {k: v[i] for k, v in cols.items()}
        out.append(_row_to_task(row))
    return out


def _first_row_to_dict(arr) -> dict:
    return {name: arr.column(name)[0].as_py() for name in arr.schema.names}


__all__ = ["TaskStore"]
