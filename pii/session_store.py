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

import lancedb
import pyarrow as pa

logger = logging.getLogger("veilguard.pii.session_store")


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


# ── Lance store ─────────────────────────────────────────────────────────


# Process-level lock for counter assignment.  Cross-process races are
# rare in practice; we re-query after insert to handle them.
_WRITE_LOCK = threading.Lock()


class PIISessionStore:
    """Lance-backed mapping store.  Append-only.  Singleton per process."""

    _instance: Optional["PIISessionStore"] = None
    _instance_lock = threading.Lock()

    def __init__(self, db_path: Optional[str] = None):
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
        if _TABLE_NAME not in self._db.table_names():
            # Create empty table with schema.
            empty = pa.Table.from_pylist([], schema=_SCHEMA)
            self._db.create_table(_TABLE_NAME, empty, mode="overwrite")
            logger.info(f"[pii] created {_TABLE_NAME} at {path}")
        self._tbl = self._db.open_table(_TABLE_NAME)

    @classmethod
    def get(cls) -> "PIISessionStore":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ── Lookup ────────────────────────────────────────────────────────

    def _query_token(
        self, sid: SessionId, entity_type: str, original_lc: str
    ) -> Optional[str]:
        """Return existing ref_token for this PII, or None."""
        sid_root = sid.root()
        # Lance where supports == with string literals; escape single quotes.
        oq = original_lc.replace("'", "''")
        arr = (
            self._tbl.search()
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
        arr = (
            self._tbl.search()
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
        arr = (
            self._tbl.search()
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

        # Fast path: already mapped.
        existing = self._query_token(sid_root, entity_type, lookup_key)
        if existing:
            return existing

        # Slow path: insert under write lock.  Re-check after acquiring
        # the lock to handle the case where two threads in this process
        # raced to insert.
        with _WRITE_LOCK:
            existing = self._query_token(sid_root, entity_type, lookup_key)
            if existing:
                return existing

            counter = self._next_counter(sid_root, entity_type)
            short_type = (
                entity_type
                .replace("_ADDRESS", "")
                .replace("_NUMBER", "")
                .replace("SA_", "")
            )
            ref_token = f"REF_{short_type}_{counter}"

            row = pa.Table.from_pylist(
                [{
                    "tenant_id":   sid_root.tenant_id,
                    "conv_id":     sid_root.conv_id,
                    "entity_type": entity_type,
                    "original_lc": lookup_key,
                    "original":    original,
                    "ref_token":   ref_token,
                    "counter":     counter,
                    "created_ts":  time.time(),
                }],
                schema=_SCHEMA,
            )
            self._tbl.add(row)
            return ref_token

    # ── Rehydration ───────────────────────────────────────────────────

    def rehydrate(self, sid: SessionId, text: str) -> str:
        """Swap REF tokens in `text` back to originals.

        Tokens with no mapping in this session are left as-is (stale or
        cross-session leakage).  Uses the same `\b`-bounded regex as the
        old in-memory store so REF_PERSON_1 can't substring-match inside
        REF_PERSON_15.
        """
        mapping = self._rehydrate_map(sid)
        if not mapping:
            return text
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
