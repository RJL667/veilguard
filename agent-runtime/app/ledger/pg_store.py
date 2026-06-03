"""Postgres-backed LedgerStore (W4 port) — a thin Lance-table-API-compatible
shim over Postgres, so the ~15 ledger consumers (inbox_poller, a2a, proposals/*,
outcomes) keep working unchanged.

Consumers use the Lance fluent API:
    tbl.add([row])
    tbl.search().where(filt).limit(k).to_arrow().to_pylist()
    tbl.merge_insert(keys).when_matched_update_all().when_not_matched_insert_all().execute([row])
    tbl.count_rows(filt) ; tbl.delete(filt) ; tbl.to_arrow() ; tbl.schema ; tbl.update(where, values)

PgTable emulates that surface with SQL. Scalars -> typed columns (from the
pyarrow TABLE_SCHEMAS); list/struct columns -> jsonb. Lance `.where(...)`
filters are already SQL-ish (`user_id = 'x' AND status = 'open'`) so they pass
through to Postgres WHERE. Transactional by construction — no merge_insert
clobber, no lease races (the agentic ledger is `mutable_transactional`, exactly
what Lance was worst at).
"""
import os
import json
import logging
import threading

import psycopg2
import psycopg2.extras
from psycopg2.extras import Json, RealDictCursor

import pyarrow as pa

from .schemas import TABLE_SCHEMAS

logger = logging.getLogger(__name__)


def _sql_type(field: "pa.Field") -> str:
    t = field.type
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "TEXT"
    if pa.types.is_boolean(t):
        return "BOOLEAN"
    if pa.types.is_floating(t):
        return "DOUBLE PRECISION"
    if pa.types.is_integer(t):
        return "BIGINT"
    if pa.types.is_timestamp(t):
        return "TIMESTAMPTZ"
    # list / struct / map / anything else -> jsonb
    return "JSONB"


def _jsonb_cols(schema: "pa.Schema") -> set:
    out = set()
    for f in schema:
        t = f.type
        if not (pa.types.is_string(t) or pa.types.is_boolean(t) or pa.types.is_floating(t)
                or pa.types.is_integer(t) or pa.types.is_timestamp(t)):
            out.add(f.name)
    return out


class _Rows:
    """Result wrapper supporting the few terminal calls consumers make."""
    def __init__(self, rows):
        self._rows = rows
    def to_pylist(self):
        return self._rows
    def to_pandas(self):
        import pandas as pd
        return pd.DataFrame(self._rows)
    def __iter__(self):
        return iter(self._rows)
    def __len__(self):
        return len(self._rows)


def _to_arrow_table(rows):
    """Build a REAL pyarrow.Table from psycopg2 dict rows, so consumers can use
    the full Lance/pyarrow column API (num_rows, column_names, slice, and
    column(c)[i].as_py()) exactly as they did against Lance's .to_arrow().
    Empty -> a 0x0 table (every consumer guards on num_rows==0 before column
    access)."""
    if not rows:
        return pa.table({})
    try:
        return pa.Table.from_pylist(rows)
    except Exception as e:  # irregular jsonb shapes can defeat type inference
        logger.warning("[pg_store] arrow inference fell back to json-encode: %s", e)
        safe = [{k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                 for k, v in r.items()} for r in rows]
        return pa.Table.from_pylist(safe)


class _Query:
    def __init__(self, table):
        self._t = table
        self._where = None
        self._limit = None
        self._select = None
    def where(self, filt):
        self._where = filt
        return self
    def limit(self, k):
        self._limit = int(k)
        return self
    def select(self, cols):
        self._select = list(cols)
        return self
    def _run(self):
        cols = ", ".join(self._select) if self._select else "*"
        sql = f"SELECT {cols} FROM {self._t._name}"
        if self._where:
            sql += f" WHERE {self._where}"
        if self._limit:
            sql += f" LIMIT {self._limit}"
        with self._t._conn.cursor(cursor_factory=RealDictCursor) as c:
            c.execute(sql)
            return [dict(r) for r in c.fetchall()]
    def to_arrow(self):
        return _to_arrow_table(self._run())
    def to_pylist(self):
        return self._run()
    def to_pandas(self):
        return _Rows(self._run()).to_pandas()


class _Merge:
    def __init__(self, table, keys):
        self._t = table
        self._keys = keys if isinstance(keys, (list, tuple)) else [keys]
        self._update = False
        self._insert = False
    def when_matched_update_all(self):
        self._update = True
        return self
    def when_not_matched_insert_all(self):
        self._insert = True
        return self
    def execute(self, rows):
        rows = list(rows.to_pylist()) if hasattr(rows, "to_pylist") else list(rows)
        for row in rows:
            self._t._upsert(row, self._keys, do_update=self._update)


class PgTable:
    """Lance-table-API-compatible shim over one Postgres table."""

    def __init__(self, conn, name, schema):
        self._conn = conn
        self._name = name
        self._schema = schema
        self._cols = [f.name for f in schema]
        self._jsonb = _jsonb_cols(schema)

    @property
    def schema(self):
        return self._schema

    def _enc(self, row):
        """Encode a row dict to (cols, vals) for SQL, jsonb-wrapping complex cols."""
        cols, vals = [], []
        for k, v in row.items():
            if k not in self._cols:
                continue
            cols.append(k)
            vals.append(Json(v) if (k in self._jsonb and v is not None) else v)
        return cols, vals

    def add(self, rows):
        rows = list(rows.to_pylist()) if hasattr(rows, "to_pylist") else list(rows)
        with self._conn.cursor() as c:
            for row in rows:
                cols, vals = self._enc(row)
                ph = ", ".join(["%s"] * len(cols))
                c.execute(f"INSERT INTO {self._name} ({', '.join(cols)}) VALUES ({ph})", vals)

    def _upsert(self, row, keys, do_update=True):
        cols, vals = self._enc(row)
        ph = ", ".join(["%s"] * len(cols))
        conflict = ", ".join(keys)
        if do_update:
            setc = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c not in keys)
            tail = f"DO UPDATE SET {setc}" if setc else "DO NOTHING"
        else:
            tail = "DO NOTHING"
        with self._conn.cursor() as c:
            c.execute(f"INSERT INTO {self._name} ({', '.join(cols)}) VALUES ({ph}) "
                      f"ON CONFLICT ({conflict}) {tail}", vals)

    def merge_insert(self, keys):
        return _Merge(self, keys)

    def search(self, *a, **k):
        return _Query(self)

    def count_rows(self, filter=None):
        sql = f"SELECT COUNT(*) FROM {self._name}"
        if filter:
            sql += f" WHERE {filter}"
        with self._conn.cursor() as c:
            c.execute(sql)
            return c.fetchone()[0]

    def delete(self, filter):
        with self._conn.cursor() as c:
            c.execute(f"DELETE FROM {self._name} WHERE {filter}")

    def update(self, where=None, values=None, **kw):
        values = values or kw.get("values") or {}
        where = where or kw.get("where")
        if not values:
            return
        setc = ", ".join(f"{k}=%s" for k in values)
        vals = [Json(v) if (k in self._jsonb and v is not None) else v for k, v in values.items()]
        sql = f"UPDATE {self._name} SET {setc}"
        if where:
            sql += f" WHERE {where}"
        with self._conn.cursor() as c:
            c.execute(sql, vals)

    def to_arrow(self):
        return _Query(self).to_arrow()

    def to_pandas(self):
        return _Query(self).to_pandas()


class _OpenTableProxy:
    """Lets consumers' `store._db.open_table(name)` route to the right Postgres
    table: ledger tables via the store; the TCMM `archive`/`dream_archive` tables
    via their CTI views (proposals/lessons read those directly off the shared DB)."""
    _TCMM_VIEWS = {"archive": "v_archive", "dream_archive": "v_dream"}
    def __init__(self, store):
        self._store = store
    def open_table(self, name):
        if name in self._TCMM_VIEWS:
            import pyarrow as _pa
            return PgTable(self._store._conn, self._TCMM_VIEWS[name], _pa.schema([]))
        return self._store.table(name)
    def table_names(self):
        return list(TABLE_SCHEMAS.keys()) + list(self._TCMM_VIEWS.keys())


class PgLedgerStore:
    """Postgres LedgerStore — drop-in for the LanceDB LedgerStore singleton."""

    _instance = None
    _init_lock = threading.Lock()

    @classmethod
    def get(cls, dsn=None):
        if cls._instance is not None:
            return cls._instance
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls(dsn)
            return cls._instance

    def __init__(self, dsn=None):
        self._dsn = (dsn or os.environ.get("LEDGER_DATABASE_URL")
                     or os.environ.get("TCMM_DATABASE_URL")
                     or "postgresql://tcmm:tcmm@localhost:5432/tcmm")
        self._conn = psycopg2.connect(self._dsn)
        self._conn.autocommit = True
        # consumers do `store._db.open_table(name)` — proxy it to .table().
        self._db = _OpenTableProxy(self)
        self._created = set()
        logger.info(f"[ledger] connected to Postgres {self._dsn.split('@')[-1]}")
        # create all ledger tables up front (idempotent)
        for name in TABLE_SCHEMAS:
            self._open_or_create(name)

    def _ddl(self, name, schema):
        defs = []
        for f in schema:
            null = "" if f.nullable else " NOT NULL"
            defs.append(f"{f.name} {_sql_type(f)}{null}")
        # 'id' is the natural PK for every ledger table.
        pk = "id" if "id" in [f.name for f in schema] else None
        if pk:
            defs.append(f"PRIMARY KEY ({pk})")
        return f"CREATE TABLE IF NOT EXISTS {name} ({', '.join(defs)})"

    def _open_or_create(self, name):
        if name not in TABLE_SCHEMAS:
            raise ValueError(f"unknown ledger table: {name}")
        schema = TABLE_SCHEMAS[name]()
        if name not in self._created:
            with self._conn.cursor() as c:
                c.execute(self._ddl(name, schema))
                # additive sync: add any missing columns (mirrors the Lance migrator)
                c.execute("SELECT column_name FROM information_schema.columns WHERE table_name=%s", [name])
                have = {r[0] for r in c.fetchall()}
                for f in schema:
                    if f.name not in have:
                        c.execute(f"ALTER TABLE {name} ADD COLUMN IF NOT EXISTS {f.name} {_sql_type(f)}")
            self._created.add(name)
        return PgTable(self._conn, name, schema)

    def table(self, name):
        return self._open_or_create(name)

    @property
    def _db_path(self):
        return self._dsn
