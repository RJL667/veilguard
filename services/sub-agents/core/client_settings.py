"""User & conversation permission-level settings (Lance-backed).

Resolves the "should this tool call need approval?" question by combining:

  1. The static per-tool policy matrix (`client_tool_policy.py`)
     — what risk class is this tool? ALLOW / APPROVE / DENY?
  2. The user's permission_level — how strict do they want to be?
  3. An optional per-conversation override — tighten or loosen for one chat.

Effective level resolution order (first match wins):

    conv_override(user_id, conv_id)
      → user_default(user_id)
      → "confirm"     (system default)

Levels and what each does to a policy decision:

    auto    — ALLOW + APPROVE both run silently; DENY still blocks
    confirm — ALLOW runs silently; APPROVE pops toast; DENY blocks
    strict  — every CLIENT_TOOL pops toast regardless; DENY still blocks

Storage:
  - Two sibling tables in the existing Veilguard Lance dir.
  - `client_settings`        — one row per user_id (the default)
  - `client_settings_conv`   — sparse, one row per active conv override
  - Lookups go through BTREE scalar indexes on user_id (+ conv_id for
    the override table) so 100M-row scale stays sub-ms (same pattern
    as pii_session_mapping).

Concurrency:
  - Singleton per process via `get()`.
  - Writes use merge_insert on the unique key so concurrent updates
    from multiple workers don't duplicate rows.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import pyarrow as pa

logger = logging.getLogger("veilguard.client_settings")

# Backend select — Postgres when TCMM_STORAGE / VEILGUARD_AUDIT_BACKEND=postgres.
_PG = os.environ.get(
    "TCMM_STORAGE", os.environ.get("VEILGUARD_AUDIT_BACKEND", "lance")
).lower() in ("postgres", "postgresql")
_PG_DSN = os.environ.get(
    "VEILGUARD_CLIENT_SETTINGS_DATABASE_URL",
    os.environ.get("TCMM_DATABASE_URL", "postgresql://tcmm:tcmm@localhost:5432/tcmm"),
)
_PG_DDL = """
CREATE TABLE IF NOT EXISTS client_settings (
    user_id           TEXT PRIMARY KEY,
    permission_level  TEXT,
    default_timeout_s BIGINT,
    created_ts        DOUBLE PRECISION,
    updated_ts        DOUBLE PRECISION
);
CREATE TABLE IF NOT EXISTS client_settings_conv (
    user_id          TEXT,
    conv_id          TEXT,
    permission_level TEXT,
    expires_at       DOUBLE PRECISION,
    created_ts       DOUBLE PRECISION,
    PRIMARY KEY (user_id, conv_id)
);
"""


# ── Constants ────────────────────────────────────────────────────────────


PERMISSION_LEVELS = ("auto", "confirm", "strict")
DEFAULT_LEVEL = "confirm"
DEFAULT_TIMEOUT_S = 60   # how long the approval gate waits for the user

_TABLE_USER = "client_settings"
_TABLE_CONV = "client_settings_conv"

_INDEXED_USER_COLS = ("user_id",)
_INDEXED_CONV_COLS = ("user_id", "conv_id")


_SCHEMA_USER = pa.schema([
    ("user_id",          pa.string()),
    ("permission_level", pa.string()),
    ("default_timeout_s", pa.int64()),
    ("created_ts",       pa.float64()),
    ("updated_ts",       pa.float64()),
])

_SCHEMA_CONV = pa.schema([
    ("user_id",          pa.string()),
    ("conv_id",          pa.string()),
    ("permission_level", pa.string()),
    ("expires_at",       pa.float64()),   # 0 = never expires
    ("created_ts",       pa.float64()),
])


# ── Singleton store ──────────────────────────────────────────────────────


class ClientSettingsStore:
    """Lance-backed per-user / per-conversation permission-level store."""

    _instance: Optional["ClientSettingsStore"] = None
    _instance_lock = threading.Lock()

    def __init__(self, db_path: Optional[str] = None):
        self._pg = _PG
        if self._pg:
            import psycopg2
            self._conn = psycopg2.connect(_PG_DSN)
            self._conn.autocommit = True
            with self._conn.cursor() as c:
                c.execute(_PG_DDL)
            logger.info("[client_settings] postgres backend")
            return
        import lancedb  # lazy: only the Lance backend needs it
        path = (
            db_path
            or os.environ.get("VEILGUARD_CLIENT_SETTINGS_DB_PATH")
            or os.environ.get("VEILGUARD_PII_DB_PATH")
            or os.environ.get("LEDGER_DB_PATH")
            or "/tcmm-data/veilguard/tcmm.db"
        )
        Path(path).mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(path)
        # Ensure tables exist (once).  We DO NOT cache the table handle
        # past this point — every read calls _tbl(name) which re-opens.
        # Lance table handles are pinned to a snapshot version at
        # open_table() time; another process writing the row we want
        # is invisible to a cached handle until it re-opens.  This bit
        # the LedgerStore earlier; same fix here.
        self._open_or_create(_TABLE_USER, _SCHEMA_USER)
        self._open_or_create(_TABLE_CONV, _SCHEMA_CONV)
        self._ensure_indexes(self._tbl(_TABLE_USER), _INDEXED_USER_COLS)
        self._ensure_indexes(self._tbl(_TABLE_CONV), _INDEXED_CONV_COLS)

    def _open_or_create(self, name: str, schema: pa.Schema):
        # IMPORTANT: lancedb >=0.x returns a paginated result object
        # from list_tables(), NOT a plain list.  ``name in
        # self._db.list_tables()`` silently evaluates False even when
        # the table exists, which triggered the create_table(mode=
        # "overwrite") branch on every sub-agents restart — wiping the
        # user's saved permission_level back to default.  Try-open is
        # the contract-safe pattern: if the table exists, open it; if
        # not, the open raises and we create.  NEVER use mode=
        # "overwrite" here.
        try:
            return self._db.open_table(name)
        except Exception:
            empty = pa.Table.from_pylist([], schema=schema)
            tbl = self._db.create_table(name, empty)
            logger.info(f"[client_settings] created table {name}")
            return tbl

    def _tbl(self, name: str):
        """Always return a fresh table handle.  See __init__ for why
        caching this handle is wrong (Lance snapshot pinning blinds us
        to cross-process writes the daemon tray's set_user_level makes)."""
        return self._db.open_table(name)

    @property
    def _tbl_user(self):
        return self._tbl(_TABLE_USER)

    @property
    def _tbl_conv(self):
        return self._tbl(_TABLE_CONV)

    def _ensure_indexes(self, tbl, cols: tuple[str, ...]) -> None:
        """Idempotent BTREE creation; same pattern as pii_session_store."""
        try:
            existing = {idx.columns[0] for idx in tbl.list_indices()
                        if idx.columns}
        except Exception as e:
            logger.warning(f"[client_settings] list_indices failed: {e}")
            existing = set()
        for col in cols:
            if col in existing:
                continue
            try:
                tbl.create_scalar_index(col, index_type="BTREE", replace=True)
                logger.info(f"[client_settings] created BTREE on {col}")
            except Exception as e:
                logger.warning(
                    f"[client_settings] create_scalar_index({col}) failed: {e}"
                )

    @classmethod
    def get(cls) -> "ClientSettingsStore":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ── User-default API ─────────────────────────────────────────────────

    def get_user_level(self, user_id: str) -> tuple[str, int]:
        """Return (permission_level, default_timeout_s) for `user_id`.

        Unknown users get `(DEFAULT_LEVEL, DEFAULT_TIMEOUT_S)` without
        a DB write — opt-in: a row only appears once the user touches
        their setting from the tray / chat command.
        """
        if not user_id:
            return DEFAULT_LEVEL, DEFAULT_TIMEOUT_S
        if self._pg:
            with self._conn.cursor() as c:
                c.execute("SELECT permission_level, default_timeout_s FROM client_settings "
                          "WHERE user_id=%s", [user_id])
                r = c.fetchone()
            if not r:
                return DEFAULT_LEVEL, DEFAULT_TIMEOUT_S
            return (r[0] or DEFAULT_LEVEL), int(r[1] or DEFAULT_TIMEOUT_S)
        try:
            arr = (
                self._tbl_user.search()
                .where(f"user_id = '{user_id}'")
                .to_arrow()
            )
        except Exception as e:
            logger.warning(f"[client_settings] read failed for {user_id}: {e}")
            return DEFAULT_LEVEL, DEFAULT_TIMEOUT_S
        if arr.num_rows == 0:
            return DEFAULT_LEVEL, DEFAULT_TIMEOUT_S
        level = arr.column("permission_level")[0].as_py() or DEFAULT_LEVEL
        timeout = arr.column("default_timeout_s")[0].as_py() or DEFAULT_TIMEOUT_S
        return level, int(timeout)

    def set_user_level(
        self, user_id: str, level: str,
        *, default_timeout_s: Optional[int] = None,
    ) -> None:
        """Upsert the user's default level.  Validates against PERMISSION_LEVELS."""
        if not user_id:
            raise ValueError("user_id required")
        if level not in PERMISSION_LEVELS:
            raise ValueError(
                f"invalid level {level!r}; expected one of {PERMISSION_LEVELS}"
            )
        now = time.time()
        # Preserve existing created_ts if the row already exists.
        prev = self.get_user_level(user_id)
        prev_timeout = default_timeout_s if default_timeout_s is not None else prev[1]
        row = {
            "user_id":          user_id,
            "permission_level": level,
            "default_timeout_s": int(prev_timeout),
            "created_ts":       now,   # overwritten by merge_insert if row exists
            "updated_ts":       now,
        }
        if self._pg:
            with self._conn.cursor() as c:
                c.execute("INSERT INTO client_settings (user_id,permission_level,default_timeout_s,"
                          "created_ts,updated_ts) VALUES (%s,%s,%s,%s,%s) ON CONFLICT (user_id) DO "
                          "UPDATE SET permission_level=EXCLUDED.permission_level, "
                          "default_timeout_s=EXCLUDED.default_timeout_s, updated_ts=EXCLUDED.updated_ts",
                          [user_id, level, int(prev_timeout), now, now])
            logger.info(f"[client_settings] user={user_id[:8]} level={level} timeout={prev_timeout}s")
            return
        try:
            (
                self._tbl_user.merge_insert("user_id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute([row])
            )
            logger.info(
                f"[client_settings] user={user_id[:8]} level={level} "
                f"timeout={prev_timeout}s"
            )
        except Exception as e:
            logger.error(f"[client_settings] set_user_level failed: {e}")
            raise

    # ── Per-conversation override API ────────────────────────────────────

    def get_conv_override(self, user_id: str, conv_id: str) -> Optional[str]:
        """Return the conv override level, or None if absent / expired."""
        if not (user_id and conv_id):
            return None
        if self._pg:
            with self._conn.cursor() as c:
                c.execute("SELECT permission_level, expires_at FROM client_settings_conv "
                          "WHERE user_id=%s AND conv_id=%s", [user_id, conv_id])
                r = c.fetchone()
            if not r:
                return None
            level, expires = r[0], (r[1] or 0.0)
            if expires and expires < time.time():
                return None
            return level if level in PERMISSION_LEVELS else None
        try:
            arr = (
                self._tbl_conv.search()
                .where(
                    f"user_id = '{user_id}' AND conv_id = '{conv_id}'"
                )
                .to_arrow()
            )
        except Exception as e:
            logger.warning(f"[client_settings] conv read failed: {e}")
            return None
        if arr.num_rows == 0:
            return None
        expires = arr.column("expires_at")[0].as_py() or 0.0
        if expires and expires < time.time():
            return None
        level = arr.column("permission_level")[0].as_py()
        return level if level in PERMISSION_LEVELS else None

    def set_conv_override(
        self, user_id: str, conv_id: str, level: str,
        *, ttl_seconds: int = 0,
    ) -> None:
        """Set / refresh a conv override.  `ttl_seconds=0` = never expires."""
        if not (user_id and conv_id):
            raise ValueError("user_id and conv_id required")
        if level not in PERMISSION_LEVELS:
            raise ValueError(
                f"invalid level {level!r}; expected one of {PERMISSION_LEVELS}"
            )
        now = time.time()
        expires = (now + ttl_seconds) if ttl_seconds > 0 else 0.0
        row = {
            "user_id":          user_id,
            "conv_id":          conv_id,
            "permission_level": level,
            "expires_at":       expires,
            "created_ts":       now,
        }
        if self._pg:
            with self._conn.cursor() as c:
                c.execute("INSERT INTO client_settings_conv (user_id,conv_id,permission_level,"
                          "expires_at,created_ts) VALUES (%s,%s,%s,%s,%s) ON CONFLICT (user_id,conv_id) "
                          "DO UPDATE SET permission_level=EXCLUDED.permission_level, "
                          "expires_at=EXCLUDED.expires_at",
                          [user_id, conv_id, level, expires, now])
            return
        try:
            (
                self._tbl_conv.merge_insert(["user_id", "conv_id"])
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute([row])
            )
            logger.info(
                f"[client_settings] override user={user_id[:8]} "
                f"conv={conv_id[:8]} level={level} ttl={ttl_seconds}s"
            )
        except Exception as e:
            logger.error(f"[client_settings] set_conv_override failed: {e}")
            raise

    def clear_conv_override(self, user_id: str, conv_id: str) -> None:
        """Remove a conv override (e.g. user clicked 'reset' in tray)."""
        if not (user_id and conv_id):
            return
        if self._pg:
            with self._conn.cursor() as c:
                c.execute("DELETE FROM client_settings_conv WHERE user_id=%s AND conv_id=%s",
                          [user_id, conv_id])
            return
        try:
            self._tbl_conv.delete(
                f"user_id = '{user_id}' AND conv_id = '{conv_id}'"
            )
        except Exception as e:
            logger.warning(f"[client_settings] clear override failed: {e}")

    # ── Combined resolution ──────────────────────────────────────────────

    def get_effective(
        self, user_id: str, conv_id: str = "",
    ) -> tuple[str, int]:
        """Resolve (level, timeout) for one tool call.

        Resolution: conv override → user default → DEFAULT_LEVEL.
        Timeout always comes from user default (overrides don't carry one).
        """
        user_level, timeout = self.get_user_level(user_id)
        override = self.get_conv_override(user_id, conv_id) if conv_id else None
        return (override or user_level, timeout)


__all__ = [
    "ClientSettingsStore",
    "PERMISSION_LEVELS",
    "DEFAULT_LEVEL",
    "DEFAULT_TIMEOUT_S",
]
