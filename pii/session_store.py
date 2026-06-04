"""PII session store — append-only, byte-stable, Lance-backed.

This replaces the in-memory PIISessionStore that lived in pii-proxy.
The store maps `(tenant_id, conversation_id, entity_type, original)` to
a stable `REF_<TYPE>_<N>` token.  Same input + same session always
produces the same token, so the cached prefix Anthropic sees is
byte-identical across turns.

Why Lance:
  - Survives restarts (cache stability across deploys)
  - Shared across processes (pii-proxy + agent-runtime both see the
    same mapping for the same conv → identical redacted bytes)
  - Same maintenance discipline (`optimize_indices()`) as the rest of
    Veilguard's storage
  - Tenant-isolated by structural namespace filter (see ns_filter)

Invariants enforced here:
  - APPEND-ONLY: once `(tenant_id, conv_id, entity_type, original_lc)`
    has a token, it never changes.  New PII in later turns gets new
    tokens with higher counters.  This is the cache-stability property.
  - CASE-INSENSITIVE PERSON dedup: "Alice" and "ALICE" share a token.
    First-seen casing wins for rehydration display.
  - PARENT-CID RESOLUTION: sub-cids like `sub-<parent>-<agent>` resolve
    to the parent's mapping (TCMM does the same for memory).  So
    Director's redaction and Researcher's redaction in the same conv
    share REF tokens.

Concurrency:
  - Multiple processes calling `add_mapping()` for the same NEW PII
    simultaneously is rare but possible (e.g. pii-proxy handling
    LibreChat + agent-runtime running a daemon at the same time).
  - We handle it by: (1) re-querying after insert, (2) trusting whichever
    row landed first.  Lance writes are atomic per-row; the second
    writer's row is dropped on `_upsert_token` retry.

Performance:
  - Table is tiny (one row per unique PII per conv).  Even for a
    thousand-turn conv with hundreds of distinct PII entities, this is
    well under a megabyte.  Lookups are filter-scan; sub-millisecond.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pyarrow as pa

logger = logging.getLogger("veilguard.pii.session_store")

# Backend select (mirrors the TCMM/ledger/audit switch). Postgres when
# VEILGUARD_AUDIT_BACKEND / TCMM_STORAGE = postgres; else the legacy LanceDB
# table. Same SessionId / add_mapping / rehydrate API either way.
_PG = os.environ.get(
    "VEILGUARD_AUDIT_BACKEND", os.environ.get("TCMM_STORAGE", "postgres")
).lower() in ("postgres", "postgresql")
_PG_DSN = os.environ.get(
    "VEILGUARD_AUDIT_DATABASE_URL",
    os.environ.get("TCMM_DATABASE_URL", "postgresql://tcmm:tcmm@localhost:5432/tcmm"),
)
_PG_DDL = """
CREATE TABLE IF NOT EXISTS pii_session_mapping (
    tenant_id   TEXT NOT NULL,
    conv_id     TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    original_lc TEXT NOT NULL,
    original    TEXT,
    ref_token   TEXT NOT NULL,
    counter     BIGINT,
    created_ts  DOUBLE PRECISION,
    PRIMARY KEY (tenant_id, conv_id, entity_type, original_lc)
);
CREATE INDEX IF NOT EXISTS ix_pii_sess_conv   ON pii_session_mapping (tenant_id, conv_id);
CREATE INDEX IF NOT EXISTS ix_pii_sess_reftok ON pii_session_mapping (ref_token);
"""


# ── Schema ──────────────────────────────────────────────────────────────


_TABLE_NAME = "pii_session_mapping"

_SCHEMA = pa.schema([
    ("tenant_id",    pa.string()),
    ("conv_id",      pa.string()),     # ALWAYS root conv (sub-cids resolved)
    ("entity_type",  pa.string()),     # PERSON, EMAIL, PHONE, ORG, ...
    ("original_lc",  pa.string()),     # lowercase form for lookup
    ("original",     pa.string()),     # first-seen casing for rehydration
    ("ref_token",    pa.string()),     # REF_PERSON_1 etc.
    ("counter",      pa.int64()),      # the N in REF_TYPE_N
    ("created_ts",   pa.float64()),
])


# Whole-token matcher for rehydration.  `\d+` matches GREEDILY plus the
# `\b` word boundaries on both sides — together they ensure
# `REF_PERSON_1` and `REF_PERSON_15` are disjoint matches.  See the
# 23-Apr-2026 incident captured in agent-proxy/app/session.py docstring.
_REF_TOKEN_RE = re.compile(
    r"\bREF_(?:PERSON|EMAIL|PHONE|IP|LOCATION|URL|CREDIT_CARD|"
    r"ID|IBAN_CODE|IBAN|ORG|DATE|API_KEY|CARD|BANK_ACCOUNT|"
    r"SA_ID|SA_PHONE)_\d+\b"
)


# ── Session id helper ───────────────────────────────────────────────────


@dataclass(frozen=True)
class SessionId:
    """Cache-stability key for a redaction session.

    All callers (pii-proxy, agent-runtime, agents) must construct the
    SAME SessionId for the SAME conversation, or the mapping won't
    align.  This dataclass exists to make the key construction explicit.
    """
    tenant_id: str
    conv_id: str

    def root(self) -> "SessionId":
        """Resolve a sub-cid back to its parent.

        Sub-cids look like `sub-<parent_cid>-<agent_id>` per the
        TCMM/Veilguard convention.  We strip down to the parent so all
        agents working in the same conv share a mapping.
        """
        cid = self.conv_id
        if cid.startswith("sub-"):
            # `sub-<parent>-<agent>` — parent is between the first and
            # second `-` after the `sub-` prefix.
            parts = cid.split("-", 2)
            if len(parts) >= 3:
                # parts[1] is "<parent>"; parts[2] is "<rest>"
                return SessionId(self.tenant_id, parts[1])
        return self

    # [PII_SID_CONTRACT_2026_05_30]  Unambiguous wire form so the render
    # layer (which mints REF tokens) and the boundary (which rehydrates
    # them) key on the IDENTICAL SessionId.  The boundary passes
    # `sid.canonical()` in the /render request; the server does
    # `set_redact_sid(SessionId.parse(...))`; the boundary rehydrates with
    # the same `sid`.  No re-derivation from (user_id, conv_id) → no drift.
    _CANON_SEP = "\x1f"     # unit separator — never appears in ids

    def canonical(self) -> str:
        return f"{self.tenant_id}{self._CANON_SEP}{self.conv_id}"

    @classmethod
    def parse(cls, s: str) -> "SessionId":
        """Inverse of canonical().  A bare string with no separator is
        treated as a conv_id under the legacy `_proxy` tenant (matches
        PIIRedactor._coerce_sid), so older callers still resolve."""
        if s and cls._CANON_SEP in s:
            tenant, conv = s.split(cls._CANON_SEP, 1)
            return cls(tenant, conv)
        return cls("_proxy", s or "pii-default")


# ── Lance store ─────────────────────────────────────────────────────────


# Process-level lock for counter assignment.  Cross-process races are
# rare in practice; we re-query after insert to handle them.
_WRITE_LOCK = threading.Lock()


class PIISessionStore:
    """Lance-backed mapping store.  Append-only.  Singleton per process."""

    _instance: Optional["PIISessionStore"] = None
    _instance_lock = threading.Lock()

    def __init__(self, db_path: Optional[str] = None):
        # In-memory state shared by both backends (memo + write batch + counters).
        self._memo: dict[tuple, str] = {}
        self._pending_rows: list[dict] = []
        self._counters: dict[tuple, int] = {}
        self._warmed_sids: set = set()   # (tenant,conv) whose full map is memo-loaded
        self._pg = _PG
        if self._pg:
            import psycopg2
            self._conn = psycopg2.connect(_PG_DSN)
            self._conn.autocommit = True
            with self._conn.cursor() as c:
                # redaction map is a derived cache — defer the WAL fsync per write
                c.execute("SET synchronous_commit = off")
                c.execute(_PG_DDL)
            self._tbl = None  # Lance-only handle; unused on Postgres
            logger.info("[pii] session store on postgres")
            return
        import lancedb  # lazy: only the Lance backend needs it
        # Reuse the agent-runtime ledger DB by default — same Lance dir,
        # sibling table.  Override via VEILGUARD_PII_DB_PATH (or share
        # with agent-runtime's LEDGER_DB_PATH).
        path = (
            db_path
            or os.environ.get("VEILGUARD_PII_DB_PATH")
            or os.environ.get("LEDGER_DB_PATH")
            or "/tcmm-data/veilguard/tcmm.db"
        )
        Path(path).mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(path)
        # [PII_PERSIST_FIX_2026_05_31]  Do NOT pre-create an EMPTY table —
        # create the table lazily WITH data on the first flush() so it
        # commits real rows to disk; here we just OPEN it if a prior process
        # already persisted it (else None → reads return empty until the
        # first write).
        #
        # ROOT CAUSE of the original cross-process rehydration miss: the old
        # code gated existence on `_TABLE_NAME in self._db.table_names()`.
        # Existence is instead decided by TRYING to open the table, NEVER via
        # open the table, NEVER via `_TABLE_NAME in self._db.table_names()`.
        # lancedb 0.30.x `table_names()` defaults to limit=10 and returns
        # names alphabetically — with >10 sibling tables in this DB,
        # `pii_session_mapping` sorts PAST the first page and is invisible to
        # a membership check even though it exists on disk.  That false
        # "missing" verdict is the whole bug: reads returned empty and the
        # write path overwrote real rows.  open_table is authoritative.
        self._tbl = self._open_existing()
        # [PII_MEMO_2026_05_29]  Process-level memo for (sid, entity, value)
        # → ref_token.  add_mapping previously ran a Lance .search() on
        # EVERY PII span — including the "already mapped?" fast-path check.
        # A memory prefix that mentions the same person 100× did 100 Lance
        # queries (~33ms each → 3s+ for one block; 10s+ for a full 10k-tok
        # render).  Mappings are append-only + immutable once created, so a
        # plain dict is safe: hit → O(1), miss → one Lance query then cache.
        # Reduces redact of a repetitive 10k-token prefix from ~10s to
        # <100ms (matches the proxy-side benchmark).
        self._memo: dict[tuple, str] = {}
        # [PII_BATCH_WRITE_2026_05_29]  Buffer new mappings and flush them
        # in ONE Lance add() per redaction instead of one add() (= one
        # table version) PER PII span.  A fat user-scoped memory prefix
        # has dozens of distinct PII entities; one-version-per-span
        # re-bloated the session table to 100+ versions within a few
        # turns, and each subsequent write walks the whole manifest chain
        # (1.5s on the bind mount) → redaction ballooned to 6-9s.  With
        # batching, version growth is ~1 per redacted block, not per PII.
        # `_counters` tracks the per-(tenant,conv,entity) sequence in
        # memory so we don't re-query Lance (which wouldn't see buffered
        # rows anyway) for each new token.
        self._pending_rows: list[dict] = []
        self._counters: dict[tuple, int] = {}
        # Ensure scalar BTREE indexes exist on the hot lookup columns.
        # Without these every redaction triggers a full-fragment column
        # scan; lookups go from sub-ms to multi-second past ~10M rows.
        # Idempotent: `replace=True` no-ops cheaply when the index is
        # already present and freshens it when row count has grown.
        # Cost is ~200ms per column on a small table, scaling with
        # row count — well under the 2s/turn budget even at 100M rows.
        if self._tbl is not None:
            self._ensure_indexes()

    # ── Index maintenance ─────────────────────────────────────────────────

    _INDEXED_COLS = ("conv_id", "tenant_id", "original_lc")

    def _ensure_indexes(self) -> None:
        """Idempotently create BTREE indexes on the hot lookup columns.

        Called from __init__ AND from `optimize()` (the Lance maintenance
        cron entry point).  Safe to call any time — Lance freshens an
        existing index in place when the table has grown.

        Best-effort: a failure here MUST NOT block startup, because the
        redactor still works without indexes (just slower).  We log the
        warning loudly so it shows up in dashboards.
        """
        if self._pg:
            return  # Postgres manages its own indexes (DDL in __init__)
        try:
            existing = {idx.columns[0] for idx in self._tbl.list_indices()
                        if idx.columns}
        except Exception as e:
            logger.warning(f"[pii] list_indices failed: {e}")
            existing = set()
        for col in self._INDEXED_COLS:
            if col in existing:
                continue
            try:
                self._tbl.create_scalar_index(
                    col, index_type="BTREE", replace=True,
                )
                logger.info(f"[pii] created BTREE index on {_TABLE_NAME}.{col}")
            except Exception as e:
                logger.warning(
                    f"[pii] create_scalar_index({col}) failed: {e}"
                )

    def optimize(self) -> dict:
        """Lance maintenance entry point.  Call from the weekly cron.

        Refreshes scalar indexes (so they cover newly-written rows) and
        compacts data fragments.  Both operations are MVCC-safe — they
        run live without blocking concurrent redactor calls.

        Returns a stats dict for logging / dashboard display.
        """
        if self._pg:
            return {"backend": "postgres", "skipped": "pg_autovacuum"}
        stats: dict = {}
        if self._get_tbl() is None:
            # Table not created yet (no PII ever redacted) → nothing to do.
            stats["skipped"] = "no_table"
            return stats
        t0 = time.time()
        self._ensure_indexes()
        stats["index_refresh_ms"] = int((time.time() - t0) * 1000)
        t0 = time.time()
        try:
            self._tbl.optimize()
        except Exception as e:
            logger.warning(f"[pii] optimize() failed: {e}")
            stats["optimize_error"] = str(e)
        stats["optimize_ms"] = int((time.time() - t0) * 1000)
        return stats

    @classmethod
    def get(cls) -> "PIISessionStore":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ── Lookup ────────────────────────────────────────────────────────

    def _open_existing(self):
        """Open the mapping table if it exists, else None.

        AUTHORITATIVE existence check: try to open and treat any failure as
        'not present'.  Must NOT use `_TABLE_NAME in self._db.table_names()`
        — lancedb 0.30.x table_names() is paginated (default limit=10) and
        alphabetical, so a table sorting past the first page is silently
        absent from the membership test even though it exists on disk.  That
        false-missing verdict is the root cause of the cross-process
        rehydration miss AND of an overwrite that wiped real mappings.
        """
        try:
            return self._db.open_table(_TABLE_NAME)
        except Exception:
            return None

    def _get_tbl(self):
        """[PII_PERSIST_FIX_2026_05_31]  Return the live table handle,
        REFRESHED so writes from OTHER processes (e.g. the host TCMM
        render-side redactor) are visible to this reader.  Returns None ONLY
        when the table genuinely does not exist yet.  Without the
        checkout_latest()/re-open a reader's handle is pinned to the version
        it opened and never sees cross-process tokens.
        """
        try:
            if self._tbl is None:
                self._tbl = self._open_existing()
                if self._tbl is None:
                    return None
                try:
                    self._ensure_indexes()
                except Exception:
                    pass
                return self._tbl
            try:
                self._tbl.checkout_latest()
            except Exception:
                reopened = self._open_existing()
                if reopened is not None:
                    self._tbl = reopened
            return self._tbl
        except Exception:
            return None

    def _query_token(
        self, sid: SessionId, entity_type: str, original_lc: str
    ) -> Optional[str]:
        """Return existing ref_token for this PII, or None."""
        sid_root = sid.root()
        if self._pg:
            with self._conn.cursor() as c:
                c.execute("SELECT ref_token FROM pii_session_mapping WHERE tenant_id=%s "
                          "AND conv_id=%s AND entity_type=%s AND original_lc=%s LIMIT 1",
                          [sid_root.tenant_id, sid_root.conv_id, entity_type, original_lc])
                row = c.fetchone()
            return row[0] if row else None
        tbl = self._get_tbl()
        if tbl is None:
            return None
        # Lance where supports == with string literals; escape single quotes.
        oq = original_lc.replace("'", "''")
        arr = (
            tbl.search()
            .where(
                f"tenant_id = '{sid_root.tenant_id}' "
                f"AND conv_id = '{sid_root.conv_id}' "
                f"AND entity_type = '{entity_type}' "
                f"AND original_lc = '{oq}'"
            )
            .to_arrow()
        )
        if arr.num_rows == 0:
            return None
        return arr.column("ref_token")[0].as_py()

    def _next_counter(self, sid: SessionId, entity_type: str) -> int:
        """Return the next counter for this (sid, entity_type)."""
        sid_root = sid.root()
        if self._pg:
            with self._conn.cursor() as c:
                c.execute("SELECT COALESCE(MAX(counter),0)+1 FROM pii_session_mapping WHERE "
                          "tenant_id=%s AND conv_id=%s AND entity_type=%s",
                          [sid_root.tenant_id, sid_root.conv_id, entity_type])
                return int(c.fetchone()[0])
        tbl = self._get_tbl()
        if tbl is None:
            return 1
        arr = (
            tbl.search()
            .where(
                f"tenant_id = '{sid_root.tenant_id}' "
                f"AND conv_id = '{sid_root.conv_id}' "
                f"AND entity_type = '{entity_type}'"
            )
            .to_arrow()
        )
        if arr.num_rows == 0:
            return 1
        counters = arr.column("counter").to_pylist()
        return max(counters) + 1

    def _rehydrate_map(self, sid: SessionId) -> dict[str, str]:
        """All token→original mappings for this session."""
        sid_root = sid.root()
        if self._pg:
            with self._conn.cursor() as c:
                c.execute("SELECT ref_token, original FROM pii_session_mapping WHERE "
                          "tenant_id=%s AND conv_id=%s", [sid_root.tenant_id, sid_root.conv_id])
                return {tok: orig for tok, orig in c.fetchall()}
        tbl = self._get_tbl()
        if tbl is None:
            return {}
        arr = (
            tbl.search()
            .where(
                f"tenant_id = '{sid_root.tenant_id}' "
                f"AND conv_id = '{sid_root.conv_id}'"
            )
            .to_arrow()
        )
        if arr.num_rows == 0:
            return {}
        tokens = arr.column("ref_token").to_pylist()
        originals = arr.column("original").to_pylist()
        return dict(zip(tokens, originals))

    # ── Write path ────────────────────────────────────────────────────

    def _warm_session_memo(self, sid_root: "SessionId") -> None:
        """[PG] Load a session's ENTIRE existing map into the memo + counters in
        ONE query, once per (tenant,conv). After this, every add_mapping span is
        memo-only (no per-span PG round-trip) — restoring the in-process speed
        the batched Lance design had. A memo miss post-warm = genuinely new."""
        key = (sid_root.tenant_id, sid_root.conv_id)
        if key in self._warmed_sids:
            return
        try:
            with self._conn.cursor() as c:
                c.execute("SELECT entity_type, original_lc, ref_token, counter FROM "
                          "pii_session_mapping WHERE tenant_id=%s AND conv_id=%s",
                          [sid_root.tenant_id, sid_root.conv_id])
                for et, olc, tok, cnt in c.fetchall():
                    self._memo[(sid_root.tenant_id, sid_root.conv_id, et, olc)] = tok
                    ck = (sid_root.tenant_id, sid_root.conv_id, et)
                    if cnt is not None and cnt > self._counters.get(ck, 0):
                        self._counters[ck] = cnt
        except Exception as e:
            logger.warning(f"[pii] session memo warm failed: {e}")
        self._warmed_sids.add(key)

    def add_mapping(
        self,
        sid: SessionId,
        entity_type: str,
        original: str,
    ) -> str:
        """Insert (or return existing) ref_token for this PII.

        Append-only: if (sid, entity_type, original_lc) already exists,
        return the existing token unchanged.  Case-insensitive lookup
        for PERSON; case-sensitive for everything else.
        """
        sid_root = sid.root()
        lookup_key = original.lower() if entity_type == "PERSON" else original

        # [PG] Prefetch the whole session map ONCE → per-span add is memo-only.
        if self._pg:
            self._warm_session_memo(sid_root)

        # O(1) fast path — repeated PII (or anything the warm prefetch loaded).
        memo_key = (sid_root.tenant_id, sid_root.conv_id, entity_type, lookup_key)
        cached = self._memo.get(memo_key)
        if cached is not None:
            return cached

        # Lance fast path: per-span query (PG skips this — the warm covers it).
        if not self._pg:
            existing = self._query_token(sid_root, entity_type, lookup_key)
            if existing:
                self._memo[memo_key] = existing
                return existing

        # Slow path: insert under write lock.  Re-check after acquiring
        # the lock to handle the case where two threads in this process
        # raced to insert.
        with _WRITE_LOCK:
            cached = self._memo.get(memo_key)
            if cached is not None:
                return cached
            if not self._pg:
                existing = self._query_token(sid_root, entity_type, lookup_key)
                if existing:
                    self._memo[memo_key] = existing
                    return existing

            # Allocate the counter from the in-memory sequence (PG: seeded by
            # _warm_session_memo; Lance: seeded from _next_counter) — no
            # per-token query. BUFFER the row; flush() writes the whole batch.
            ckey = (sid_root.tenant_id, sid_root.conv_id, entity_type)
            if ckey not in self._counters:
                self._counters[ckey] = (0 if self._pg
                                        else self._next_counter(sid_root, entity_type) - 1)
            self._counters[ckey] += 1
            counter = self._counters[ckey]
            short_type = (
                entity_type
                .replace("_ADDRESS", "")
                .replace("_NUMBER", "")
                .replace("SA_", "")
            )
            ref_token = f"REF_{short_type}_{counter}"

            self._pending_rows.append({
                "tenant_id":   sid_root.tenant_id,
                "conv_id":     sid_root.conv_id,
                "entity_type": entity_type,
                "original_lc": lookup_key,
                "original":    original,
                "ref_token":   ref_token,
                "counter":     counter,
                "created_ts":  time.time(),
            })
            self._memo[memo_key] = ref_token
            return ref_token

    def flush(self) -> int:
        """[PII_BATCH_WRITE_2026_05_29]  Persist buffered mappings in ONE
        Lance add().  Called by the redactor at the end of a redaction so
        all of a turn's new tokens land in a single table version.  Safe
        to call when empty (no-op).  Returns rows flushed.

        Rehydration runs in a LATER request (after the LLM call), by which
        time this flush has committed — so rehydrate reads complete data
        from Lance with no in-memory-buffer awareness needed.
        """
        with _WRITE_LOCK:
            if not self._pending_rows:
                return 0
            rows = self._pending_rows
            self._pending_rows = []
            if self._pg:
                try:
                    from psycopg2.extras import execute_values
                    vals = [(r["tenant_id"], r["conv_id"], r["entity_type"], r["original_lc"],
                             r["original"], r["ref_token"], r["counter"], r["created_ts"])
                            for r in rows]
                    with self._conn.cursor() as c:
                        execute_values(
                            c,
                            "INSERT INTO pii_session_mapping (tenant_id,conv_id,entity_type,"
                            "original_lc,original,ref_token,counter,created_ts) VALUES %s "
                            "ON CONFLICT (tenant_id,conv_id,entity_type,original_lc) DO NOTHING",
                            vals)
                    return len(rows)
                except Exception as e:
                    logger.error(f"[pii] session-store flush failed ({len(rows)} rows): {e}")
                    return 0
            try:
                arrow = pa.Table.from_pylist(rows, schema=_SCHEMA)
                # [PII_PERSIST_FIX_2026_05_31]  Create the table WITH data the
                # FIRST time so it durably commits to disk and becomes visible
                # to other processes (an empty create_table never persists —
                # see __init__).  Append on every later flush.
                #
                # Resolve the handle through _get_tbl() (open_table-based, so
                # it sees the table even when our own self._tbl is still None
                # or table_names() can't page to it).  Create only when the
                # table genuinely cannot be opened.
                #
                # mode="create" (NOT "overwrite"): if existence detection is
                # ever wrong (a race, a stale handle), create raises
                # "already exists" and we fall back to append.  We must NEVER
                # overwrite — that destroys every prior session's REF
                # mappings, which is exactly what the table_names() bug did.
                tbl = self._get_tbl()
                if tbl is None:
                    try:
                        self._tbl = self._db.create_table(_TABLE_NAME, arrow)
                    except Exception:
                        # Lost the create race / mis-detected: append instead.
                        self._tbl = self._open_existing()
                        if self._tbl is None:
                            raise
                        self._tbl.add(arrow)
                    try:
                        self._ensure_indexes()
                    except Exception:
                        pass
                else:
                    tbl.add(arrow)
                return len(rows)
            except Exception as e:
                logger.error(f"[pii] session-store flush failed ({len(rows)} rows): {e}")
                return 0

    # ── Rehydration ───────────────────────────────────────────────────

    def rehydrate(self, sid: SessionId, text: str) -> str:
        """Swap REF tokens in `text` back to originals.

        Tokens with no mapping in this session are left as-is (stale or
        cross-session leakage).  Uses the same `\b`-bounded regex as the
        old in-memory store so REF_PERSON_1 can't substring-match inside
        REF_PERSON_15.
        """
        if not text or "REF_" not in text:
            return text
        mapping = self._rehydrate_map(sid)
        if not mapping:
            return text
        return _REF_TOKEN_RE.sub(
            lambda m: mapping.get(m.group(0), m.group(0)),
            text,
        )

    def rehydrate_any(self, text: str) -> str:
        """Best-effort, session-LESS rehydration for the proxy /rehydrate
        endpoint (sub-agent scratchpad display), which has no SessionId.

        Only the tokens PRESENT in `text` are looked up — the query is
        bounded by token count, not table size, so this stays cheap even
        on a large table.  Cross-session ambiguity is inherent (the same
        REF_PERSON_1 string maps to different people in different convs);
        most-recently-created wins, matching the old in-memory store's
        last-writer behavior.
        """
        if not text or "REF_" not in text:
            return text
        tokens = sorted(set(_REF_TOKEN_RE.findall(text)))
        if not tokens:
            return text
        if self._pg:
            try:
                with self._conn.cursor() as c:
                    c.execute("SELECT ref_token, original, created_ts FROM pii_session_mapping "
                              "WHERE ref_token = ANY(%s)", [tokens])
                    rows = c.fetchall()
            except Exception as e:
                logger.warning(f"[pii] rehydrate_any query failed: {e}")
                return text
            best: dict[str, tuple[float, str]] = {}
            for tk, og, t in rows:
                if tk not in best or (t or 0) >= best[tk][0]:
                    best[tk] = ((t or 0), og)
            mapping = {tk: v[1] for tk, v in best.items()}
            return _REF_TOKEN_RE.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)
        in_list = ", ".join("'" + t.replace("'", "''") + "'" for t in tokens)
        try:
            arr = (
                self._tbl.search()
                .where(f"ref_token IN ({in_list})")
                .to_arrow()
            )
        except Exception as e:
            logger.warning(f"[pii] rehydrate_any query failed: {e}")
            return text
        if arr.num_rows == 0:
            return text
        toks = arr.column("ref_token").to_pylist()
        origs = arr.column("original").to_pylist()
        ts = arr.column("created_ts").to_pylist()
        # most-recent wins on duplicate token across sessions
        best: dict[str, tuple[float, str]] = {}
        for tk, og, t in zip(toks, origs, ts):
            if tk not in best or t >= best[tk][0]:
                best[tk] = (t, og)
        mapping = {tk: v[1] for tk, v in best.items()}
        return _REF_TOKEN_RE.sub(
            lambda m: mapping.get(m.group(0), m.group(0)),
            text,
        )


# ── Module-level convenience accessors ──────────────────────────────────


def get_store() -> PIISessionStore:
    return PIISessionStore.get()


__all__ = [
    "SessionId",
    "PIISessionStore",
    "get_store",
]
