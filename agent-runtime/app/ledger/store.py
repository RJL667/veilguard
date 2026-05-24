"""Lance store — connection, table init, tenant-isolated filter helpers.

One connection per process; tables auto-created on first access.  Per
spec §0.1 + memory `architecture_lance_index_perms`: this MUST run as
the same OS user that owns the Lance dir (typically `rudol` on prod).
Mixing root + non-root writes silently corrupts `_indices/` and recall
starts returning 0 rows.  We assert at startup.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .schemas import TABLE_SCHEMAS

logger = logging.getLogger("agent-runtime.ledger.store")


class LanceStoreError(RuntimeError):
    pass


class LedgerStore:
    """Singleton wrapper around lancedb.connect() + table init.

    Lazy import of lancedb so unit tests (which don't install lancedb)
    can mock individual CRUD calls without dragging the import.
    """

    _instance: Optional["LedgerStore"] = None
    _init_lock = threading.Lock()

    @classmethod
    def get(cls, db_path: str | Path | None = None) -> "LedgerStore":
        if cls._instance is not None:
            return cls._instance
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls(db_path)
            return cls._instance

    def __init__(self, db_path: str | Path | None = None):
        import lancedb  # local import

        if db_path is None:
            from ..config import LEDGER_DB_PATH
            db_path = LEDGER_DB_PATH

        self._db_path = str(db_path)
        self._db = lancedb.connect(self._db_path)
        self._tables: dict[str, Any] = {}
        logger.info(f"[ledger] connected to {self._db_path}")

        # Permission sanity (memory: architecture_lance_index_perms).
        # We don't fail hard on POSIX-only checks because dev runs on
        # Windows; we just log if the dir's owner mismatches our euid.
        try:
            st = os.stat(self._db_path)
            if hasattr(os, "geteuid"):
                if st.st_uid != os.geteuid():
                    logger.warning(
                        f"[ledger] WARNING — Lance dir owned by uid={st.st_uid} "
                        f"but process running as uid={os.geteuid()}. "
                        "This is the 2026-05-21 jnb-migration failure pattern. "
                        "Recall may silently return 0 rows."
                    )
        except FileNotFoundError:
            # New deployment; dir will be created when first table opens.
            pass

    def _open_or_create(self, name: str) -> Any:
        """Return the Lance table; create with schema if missing."""
        if name in self._tables:
            return self._tables[name]

        if name not in TABLE_SCHEMAS:
            raise LanceStoreError(f"unknown ledger table: {name}")

        try:
            tbl = self._db.open_table(name)
        except Exception:
            schema = TABLE_SCHEMAS[name]()
            tbl = self._db.create_table(name, schema=schema)
            logger.info(f"[ledger] created table {name}")
        self._tables[name] = tbl
        return tbl

    def table(self, name: str) -> Any:
        """Public accessor — same as _open_or_create but explicitly named."""
        return self._open_or_create(name)


# ── Filter helpers ───────────────────────────────────────────────────────


def ns_filter(tenant_id: str, user_id: str) -> str:
    """Standard tenant + user filter clause.

    Mirrors TCMM's `_ns_filter`/`_user_filter` discipline (per existing
    memory architecture_tcmm_api).  Every read against ledger tables
    MUST AND-clause this filter to avoid cross-tenant leak.
    """
    # SQL injection defense: escape single quotes.
    t = tenant_id.replace("'", "''")
    u = user_id.replace("'", "''")
    return f"tenant_id = '{t}' AND user_id = '{u}'"


def agent_filter(agent_id: str) -> str:
    """Filter rows owned/authored by a specific agent."""
    a = agent_id.replace("'", "''")
    return f"created_by_agent_id = '{a}'"


def task_inbox_filter(
    tenant_id: str,
    user_id: str,
    owner_id: str,
    statuses: list[str] | None = None,
) -> str:
    """`agent_tasks` inbox view: open + active + blocked + review for owner."""
    if statuses is None:
        statuses = ["open", "accepted", "in_progress", "blocked", "review"]
    status_in = ", ".join(f"'{s}'" for s in statuses)
    return (
        f"{ns_filter(tenant_id, user_id)} "
        f"AND owner_id = '{owner_id.replace(chr(39), chr(39)+chr(39))}' "
        f"AND status IN ({status_in})"
    )


def proposal_queue_filter(
    tenant_id: str,
    user_id: str,
    statuses: list[str] | None = None,
) -> str:
    """`task_proposals` queue view: pending proposals for sidebar."""
    if statuses is None:
        statuses = ["pending", "deferred"]
    status_in = ", ".join(f"'{s}'" for s in statuses)
    return f"{ns_filter(tenant_id, user_id)} AND status IN ({status_in})"


def lessons_active_filter(tenant_id: str, user_id: str) -> str:
    """`org_memory` active filter: not expired/retired."""
    return (
        f"{ns_filter(tenant_id, user_id)} "
        f"AND status IN ('accepted', 'proposed') "
        f"AND expires_at > {time.time()}"
    )


__all__ = [
    "LedgerStore",
    "LanceStoreError",
    "ns_filter",
    "agent_filter",
    "task_inbox_filter",
    "proposal_queue_filter",
    "lessons_active_filter",
]
