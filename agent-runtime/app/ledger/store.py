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


def open_ledger_db(db_path=None):
    """Backend-aware ledger DB handle for modules that connect directly.
    With LEDGER_BACKEND/TCMM_STORAGE=postgres -> the PgLedgerStore proxy (its
    .open_table() serves ledger tables AND the TCMM archive/dream CTI views);
    otherwise a raw lancedb connection. Keeps proposals/lessons backend-agnostic."""
    if os.environ.get("LEDGER_BACKEND", os.environ.get("TCMM_STORAGE", "")).lower() in ("postgres", "postgresql"):
        from .pg_store import PgLedgerStore
        return PgLedgerStore.get()._db
    import lancedb
    return lancedb.connect(db_path)


class LedgerStore:
    """Singleton wrapper around lancedb.connect() + table init.

    Lazy import of lancedb so unit tests (which don't install lancedb)
    can mock individual CRUD calls without dragging the import.
    """

    _instance: Optional["LedgerStore"] = None
    _init_lock = threading.Lock()

    @classmethod
    def get(cls, db_path: str | Path | None = None):
        # Postgres backend (W4): LEDGER_BACKEND=postgres (or TCMM_STORAGE=postgres)
        # routes the whole ledger to PgLedgerStore — a Lance-API-compatible shim
        # over Postgres, so every consumer works unchanged.
        _be = os.environ.get("LEDGER_BACKEND", os.environ.get("TCMM_STORAGE", "")).lower()
        if _be in ("postgres", "postgresql"):
            from .pg_store import PgLedgerStore
            return PgLedgerStore.get()
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
        """Return a FRESH Lance table handle; create with schema if missing.

        IMPORTANT: we deliberately do NOT cache the table handle.  Lance
        table handles hold a fixed dataset version — rows written by
        OTHER processes (e.g. the inbox_poller in a sibling container,
        or our test-injection docker exec) are invisible on a cached
        handle until it's re-opened.  Re-opening on every access is
        cheap (microseconds: Lance just re-reads the manifest) and
        eliminates the entire class of stale-read bugs.

        We still TRACK which tables we've seen in `self._tables` so the
        creation-if-missing branch only runs once per name — but the
        actual handle we return is always fresh from
        `self._db.open_table(name)`.
        """
        if name not in TABLE_SCHEMAS:
            raise LanceStoreError(f"unknown ledger table: {name}")

        # Ensure the table exists once.  Subsequent calls go straight
        # to open_table without trying create_table.
        if name not in self._tables:
            try:
                self._db.open_table(name)
                self._tables[name] = True  # sentinel — we know it exists
                # Run Phase 6.0 schema migrations (idempotent, log on apply).
                self._migrate_phase_6_0(name)
            except Exception:
                schema = TABLE_SCHEMAS[name]()
                self._db.create_table(name, schema=schema)
                self._tables[name] = True
                logger.info(f"[ledger] created table {name}")

        # Always return a fresh handle.
        return self._db.open_table(name)

    def _migrate_phase_6_0(self, name: str) -> None:
        """Idempotent schema migrations for Phase 6.0 + Phase 7.1.

        Two stages:
        1. `agent_tasks` first runs the struct-aware migrator for the
           typed `acceptance_criteria` column (the SQL-DEFAULT cast
           generics don't parse on current Lance, so recreate-merge
           handles it specifically).
        2. EVERY table — including agent_tasks after stage 1 — runs
           the generic missing-nullable-column sync that diffs the
           live schema against `TABLE_SCHEMAS[name]()` and adds
           anything missing as NULL.  Catches Phase 3 emergency_lane,
           Phase 7 tcmm_obs_id, Phase 7.5 depends_on, and any future
           additive change in one place.  Drops are still a manual
           operation (we never auto-remove columns).

        Safe to call on every startup.  No-op when schemas match.
        """
        if name == "agent_tasks":
            self._migrate_agent_tasks(name)
        # All tables (including agent_tasks after the struct stage)
        # get the generic additive sync so future additive changes
        # land without touching this dispatcher.
        self._migrate_add_missing_nullable_columns(name)

    def _migrate_agent_tasks(self, name: str) -> None:
        try:
            tbl = self._db.open_table(name)
            existing_fields = {f.name for f in tbl.schema}
            if "acceptance_criteria" in existing_fields:
                return  # already migrated

            # Lance `add_columns` with a SQL DEFAULT — produces an empty
            # list for every existing row.  Lance 0.13+ supports this.
            # Fallback: re-create the table with new schema + merge data.
            try:
                tbl.add_columns({
                    "acceptance_criteria": "cast([] as list<struct<"
                    "id string, statement string, check_kind string, "
                    "check_args string, required boolean, "
                    "rationale string>>)"
                })
                logger.info(
                    f"[ledger] migrated {name}: added acceptance_criteria column "
                    f"(backfilled {tbl.count_rows()} rows with [])"
                )
            except Exception as e:
                # Newer Lance rejects the SQL DEFAULT generics syntax;
                # fall through to manual recreate.
                logger.warning(
                    f"[ledger] add_columns failed ({e}); doing recreate-merge"
                )
                self._recreate_with_acceptance_criteria(name)
        except Exception as e:
            # Migration MUST NOT block table open.  Log loudly; ops will
            # see the warning + AC-1 in CI will catch a missing manifest.
            logger.exception(
                f"[ledger] Phase 6.0 migration failed for {name}: {e}"
            )

    def _migrate_add_missing_nullable_columns(self, name: str) -> None:
        """Generic additive-schema-sync — bring `name` up to current
        `TABLE_SCHEMAS[name]()` by adding ANY missing column as NULL.

        Catches every additive change in one place: Phase 3 emergency_lane,
        Phase 7 tcmm_obs_id, and future additions.  Uses recreate-merge
        (proven path on current Lance) to avoid the SQL-DEFAULT generics
        parse bug that breaks `add_columns` for non-trivial types.
        """
        try:
            tbl = self._db.open_table(name)
            existing_fields = {f.name for f in tbl.schema}
            target = TABLE_SCHEMAS[name]()
            expected_fields = {f.name for f in target}
            missing = expected_fields - existing_fields
            if not missing:
                return  # already in sync
            logger.info(
                f"[ledger] schema drift on {name}: missing columns "
                f"{sorted(missing)} — running recreate-merge to add as NULL"
            )
            self._recreate_with_added_column(name, ",".join(sorted(missing)))
        except Exception as e:
            logger.exception(
                f"[ledger] additive-sync migration failed for {name}: {e}"
            )

    def _recreate_with_added_column(self, name: str, col_name: str) -> None:
        """Generic recreate-merge: rebuild table with new schema, copying
        every existing column and adding the new one as NULL.

        Used for Phase 7.1 nullable-string column adds where Lance's
        SQL-DEFAULT parser chokes on the cast syntax.
        """
        import pyarrow as pa
        old = self._db.open_table(name).to_arrow()
        new_schema = TABLE_SCHEMAS[name]()
        n = old.num_rows
        # Existing columns first
        cols: list[Any] = [old.column(c) for c in old.column_names]
        names = list(old.column_names)
        # Add any missing columns as NULL of the right type.
        for fld in new_schema:
            if fld.name in names:
                continue
            cols.append(pa.array([None] * n, type=fld.type))
            names.append(fld.name)
        new_table = pa.Table.from_arrays(cols, names=names)
        # Reorder to match schema.
        new_table = new_table.select([f.name for f in new_schema])
        self._db.drop_table(name)
        self._db.create_table(name, data=new_table, schema=new_schema)
        logger.info(
            f"[ledger] migrated {name}: recreated with added column "
            f"{col_name!r} ({n} rows backfilled with NULL)"
        )

    def _recreate_with_acceptance_criteria(self, name: str) -> None:
        """Fallback migration: recreate `agent_tasks` with new schema, merge data.

        Reads existing rows, builds the new schema, populates
        `acceptance_criteria=[]` for every row, writes a new table with
        the new schema, drops the old.  Cost: O(N) data copy.  Run-once.
        """
        import pyarrow as pa
        old = self._db.open_table(name).to_arrow()
        new_schema = TABLE_SCHEMAS[name]()
        # Build the new column: empty list per row.
        n = old.num_rows
        empty_acs = pa.array(
            [[] for _ in range(n)],
            type=new_schema.field("acceptance_criteria").type,
        )
        # Combine: drop nothing from old, append the new column.
        cols = [old.column(c) for c in old.column_names] + [empty_acs]
        names = list(old.column_names) + ["acceptance_criteria"]
        new_table = pa.Table.from_arrays(cols, names=names)
        # Reorder to match new schema field order.
        new_table = new_table.select([f.name for f in new_schema])
        # Drop + create.
        self._db.drop_table(name)
        self._db.create_table(name, data=new_table, schema=new_schema)
        logger.info(
            f"[ledger] migrated {name}: recreated with acceptance_criteria "
            f"({n} rows backfilled with [])"
        )

    def table(self, name: str) -> Any:
        """Public accessor — returns a fresh table handle every call.

        See `_open_or_create` for why caching the handle is the wrong
        choice (silent cross-process stale reads).
        """
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
