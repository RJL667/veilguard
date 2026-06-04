"""Direct Lance queries for archive size, fragmentation, FTS index status.

Reads the same Lance directory the TCMM service uses
(``${TCMM_DATA_DIR}/veilguard/tcmm.db``). All queries are read-only.
We avoid touching anything that could compete with the live TCMM
process for write locks.

2026-05-21: rewrite — all per-row statistics now use
``count_rows(filter=...)`` with DataFusion predicate pushdown instead of
``to_arrow()`` + Python aggregation. Avoids materialising the whole
archive table (and especially the 768-d ``vector`` column) every 10s.
A 25s TTL cache wraps the three dashboard-facing functions so the
auto-refresh cycle no longer re-runs them on every tick.
"""
from __future__ import annotations

import os
import time as _time
from pathlib import Path
from typing import Any, Callable


LANCE_DIR = os.environ.get(
    "ADMIN_LANCE_DIR",
    "/home/rudol/veilguard/tcmm-data/veilguard/tcmm.db",
)

# Backend select — when TCMM moved to Postgres these stats read the CTI tables
# (base_node / archive / dream) instead of the Lance dir.
_TCMM_BACKEND = os.environ.get("TCMM_STORAGE", "postgres").lower()
_PG = _TCMM_BACKEND in ("postgres", "postgresql")
_TCMM_DSN = os.environ.get("TCMM_DATABASE_URL", "postgresql://tcmm:tcmm@localhost:5432/tcmm")


def _pg_conn():
    import psycopg2
    conn = psycopg2.connect(_TCMM_DSN)
    conn.autocommit = True   # each stat query independent; a bad predicate can't poison the rest
    return conn

# Tables we care about for the dashboard. archive is the hot one;
# pii_audit feeds the redaction view; the others are tracked for
# fragmentation / size only.
_HOT_TABLES = ("archive", "pii_audit", "sparse", "embeddings", "dream_archive")


# ── TTL cache ────────────────────────────────────────────────────────
#
# Dashboard auto-refreshes every 10s and fires ``/api/health``,
# ``/api/lance``, ``/api/nlp-coverage`` in parallel — each of which
# calls into this module. Table sizes and coverage percentages change
# on the order of minutes (NLP workers ingest rows in batches), so
# 25s freshness is plenty and load drops to ~1 scan per 25s instead
# of 3 scans per 10s.
_TTL_CACHE: dict = {}
_TTL_SECONDS = 25.0
_MAX_KEYS = 16


def _cached(fn: Callable) -> Callable:
    """Decorate a function with a (fn_name, args)-keyed TTL cache."""
    def wrapper(*args, **kwargs):
        key = (fn.__name__, args, tuple(sorted(kwargs.items())))
        now = _time.monotonic()
        entry = _TTL_CACHE.get(key)
        if entry is not None and (now - entry[0]) < _TTL_SECONDS:
            return entry[1]
        result = fn(*args, **kwargs)
        _TTL_CACHE[key] = (now, result)
        if len(_TTL_CACHE) > _MAX_KEYS:
            oldest = min(_TTL_CACHE, key=lambda k: _TTL_CACHE[k][0])
            del _TTL_CACHE[oldest]
        return result
    wrapper.__name__ = fn.__name__
    wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    return wrapper


def _connect():
    import lancedb  # lazy: only the Lance backend needs it
    return lancedb.connect(LANCE_DIR)


def _quote(s: str) -> str:
    """Escape a single-quoted SQL literal."""
    return s.replace("'", "''")


# ── Postgres backends for the three dashboard stat functions ──────────
# Stage coverage predicates per column type: topics/entities/claims are
# bigint[]/text[] arrays (cardinality); the dicts + link maps are jsonb.
_PG_STAGE_PRED = {
    "topics":           "cardinality(topics) > 0",
    "topic_dicts":      "jsonb_typeof(topic_dicts) = 'array' AND jsonb_array_length(topic_dicts) > 0",
    "entities":         "cardinality(entities) > 0",
    "entity_dicts":     "jsonb_typeof(entity_dicts) = 'array' AND jsonb_array_length(entity_dicts) > 0",
    "claims":           "cardinality(claims) > 0",
    "semantic_links":   "semantic_links   IS NOT NULL AND semantic_links   <> '{}'::jsonb",
    "entity_links":     "entity_links     IS NOT NULL AND entity_links     <> '{}'::jsonb",
    "topic_links":      "topic_links      IS NOT NULL AND topic_links      <> '{}'::jsonb",
    "contextual_links": "contextual_links IS NOT NULL AND contextual_links <> '{}'::jsonb",
}


def _pg_overview() -> dict[str, Any]:
    out: dict[str, Any] = {"path": "postgres", "backend": "postgres", "tables": []}
    try:
        conn = _pg_conn()
    except Exception as e:
        out["error"] = f"open failed: {e}"
        return out
    try:
        cur = conn.cursor()
        # List EVERY user table (memory + agentic ledger + pii + embeddings),
        # not a hardcoded subset — so the dashboard shows the whole Postgres
        # surface (agent_tasks, task_comments, block_vectors, etc.).
        cur.execute("SELECT relname FROM pg_stat_user_tables ORDER BY relname")
        names = [r[0] for r in cur.fetchall()]
        for name in names:
            rec: dict[str, Any] = {"name": name}
            try:
                cur.execute(f'SELECT count(*) FROM "{name}"')
                rec["rows"] = cur.fetchone()[0]
                cur.execute("SELECT pg_total_relation_size(%s)", [name])
                rec["disk_bytes"] = cur.fetchone()[0]
                rec["fragments"] = 0  # Postgres: no Lance fragmentation (autovacuum)
            except Exception as e:
                rec["error"] = str(e)[:160]
            out["tables"].append(rec)
        return out
    finally:
        conn.close()


def _pg_fts_status() -> dict[str, Any]:
    try:
        conn = _pg_conn()
    except Exception as e:
        return {"built": False, "error": str(e)[:160]}
    try:
        cur = conn.cursor()
        cur.execute("SELECT indexname FROM pg_indexes WHERE indexname IN "
                    "('ix_base_fts','ix_base_vec','ix_base_srcblocks')")
        idx = {r[0] for r in cur.fetchall()}
        return {"built": "ix_base_fts" in idx, "backend": "postgres",
                "fts_index": "ix_base_fts" in idx, "vector_index": "ix_base_vec" in idx,
                "path": "postgres:base_node"}
    finally:
        conn.close()


def _pg_nlp_coverage(user_id: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"archive_rows": 0, "user_id": user_id, "stages": {},
                           "embeddings_by_type": {}}
    try:
        conn = _pg_conn()
    except Exception as e:
        out["error"] = f"open failed: {e}"
        return out
    try:
        cur = conn.cursor()

        def cnt(extra: str | None = None) -> int:
            q = "SELECT count(*) FROM v_archive WHERE TRUE"
            p: list = []
            if user_id:
                q += " AND user_id = %s"; p.append(user_id)
            if extra:
                q += f" AND ({extra})"
            try:
                cur.execute(q, p); return cur.fetchone()[0]
            except Exception:
                return 0

        total = cnt()
        out["archive_rows"] = total
        if total == 0:
            return out

        def stage_map(rec_only: bool):
            m: dict[str, dict] = {}
            pre = "recallable = TRUE AND " if rec_only else ""
            denom = rec_total if rec_only else total
            for stage, pred in _PG_STAGE_PRED.items():
                n = cnt(pre + f"({pred})")
                m[stage] = {"covered": n, "missing": denom - n, "percent": _pct(n, denom)}
            nc = cnt(pre + "block_class IS NOT NULL AND block_class <> ''")
            m["block_class"] = {"covered": nc, "missing": denom - nc, "percent": _pct(nc, denom)}
            return m

        out["stages"] = stage_map(False)
        rec_total = cnt("recallable = TRUE")
        out["recallable_archive_rows"] = rec_total
        out["recallable_total"] = rec_total
        out["not_recallable_total"] = total - rec_total
        out["stages_recallable"] = stage_map(True)

        vec = cnt("vector IS NOT NULL")
        out["vector_coverage"] = {"covered": vec, "missing": total - vec, "percent": _pct(vec, total)}

        # class × recallable cross-tab via GROUP BY
        try:
            q = ("SELECT COALESCE(NULLIF(block_class,''),'UNCLASSIFIED'), recallable, count(*) "
                 "FROM v_archive WHERE TRUE")
            p = []
            if user_id:
                q += " AND user_id = %s"; p.append(user_id)
            q += " GROUP BY 1,2"
            cur.execute(q, p)
            breakdown: dict[str, dict[str, int]] = {}
            for cls, rec, n in cur.fetchall():
                row = breakdown.setdefault(cls or "UNCLASSIFIED",
                                           {"recallable": 0, "not_recallable": 0, "total": 0})
                row["recallable" if rec is True else "not_recallable"] += n
                row["total"] += n
            out["class_breakdown"] = dict(sorted(breakdown.items(), key=lambda kv: -kv[1]["total"]))
        except Exception as e:
            out["class_breakdown_error"] = str(e)[:160]

        out["link_totals"] = {s: out["stages"].get(s, {}).get("covered", 0)
                              for s in ("semantic_links", "entity_links",
                                        "topic_links", "contextual_links")}

        # embeddings: base_node.vector (archive type) + block_vectors by emb_type
        bytype: dict[str, int] = {}
        try:
            q = "SELECT count(*) FROM base_node WHERE vector IS NOT NULL"
            p = []
            if user_id:
                q += " AND user_id = %s"; p.append(user_id)
            cur.execute(q, p); bytype["archive"] = cur.fetchone()[0]
        except Exception:
            pass
        try:
            q = "SELECT emb_type, count(*) FROM block_vectors WHERE TRUE"
            p = []
            if user_id:
                q += " AND user_id = %s"; p.append(user_id)
            q += " GROUP BY 1"
            cur.execute(q, p)
            for et, n in cur.fetchall():
                bytype[et] = n
        except Exception:
            pass
        out["embeddings_total"] = sum(bytype.values())
        out["embeddings_by_type"] = dict(sorted(bytype.items()))
        return out
    finally:
        conn.close()


# ── overview ─────────────────────────────────────────────────────────


@_cached
def overview() -> dict[str, Any]:
    """Row counts + on-disk size per table + dataset version + fragment count.

    All errors are caught per-table — a missing or corrupt table can't
    take the dashboard down.
    """
    if _PG:
        return _pg_overview()
    out: dict[str, Any] = {"path": LANCE_DIR, "tables": []}
    try:
        db = _connect()
    except Exception as e:
        out["error"] = f"open failed: {e}"
        return out

    try:
        names = list(db.table_names(limit=100_000))  # default limit=10 truncates the list
    except Exception:
        try:
            names = list(db.list_tables())
        except Exception as e:
            out["error"] = f"list_tables failed: {e}"
            return out

    for name in sorted(names):
        rec: dict[str, Any] = {"name": name}
        try:
            t = db.open_table(name)
            rec["rows"] = t.count_rows()
            ds = getattr(t, "to_lance", None)
            if callable(ds):
                try:
                    lance_ds = t.to_lance()
                    rec["version"] = getattr(lance_ds, "version", None)
                    fragments = lance_ds.get_fragments()
                    rec["fragments"] = len(fragments)
                    if rec["fragments"] > 0 and rec["rows"]:
                        rec["rows_per_fragment"] = round(
                            rec["rows"] / rec["fragments"], 1
                        )
                except Exception:
                    # pylance not installed in this container — skip
                    # fragmentation reporting silently.
                    pass
        except Exception as e:
            rec["error"] = str(e)[:160]
        try:
            tdir = Path(LANCE_DIR) / f"{name}.lance"
            if tdir.exists():
                size = sum(
                    f.stat().st_size for f in tdir.rglob("*") if f.is_file()
                )
                rec["disk_bytes"] = size
        except Exception:
            pass
        out["tables"].append(rec)
    return out


# ── nlp_coverage ─────────────────────────────────────────────────────


_LIST_STAGES = (
    "topics", "topic_dicts", "entities", "entity_dicts", "claims",
    "semantic_links", "entity_links", "topic_links", "contextual_links",
)


def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 1) if total else 0.0


@_cached
def nlp_coverage(user_id: str | None = None) -> dict[str, Any]:
    """Per-row NLP-work coverage on the ``archive`` table.

    Stage coverage is computed entirely via ``count_rows(filter=...)``
    with DataFusion predicate pushdown — no row materialisation, no
    PyArrow buffers held in memory. Vector coverage uses
    ``vector[1] != 0.0`` pushdown (1-based fixed-size-list indexing),
    eliminating the previous ~18 MB scan + per-row Python loop.

    Two small projection scans remain:
      * ``block_class`` + ``recallable`` (2 cols) for the class × recallable
        cross-tab, which needs ``GROUP BY`` semantics ``count_rows`` can't
        express.
      * ``emb_type`` from the ``embeddings`` table for the emb-type
        breakdown.
    Both use ``search().select([...]).to_arrow()`` so Lance only reads
    the requested columns (~50–100 KB total instead of a multi-MB scan).
    """
    if _PG:
        return _pg_nlp_coverage(user_id)
    out: dict[str, Any] = {
        "archive_rows": 0,
        "user_id": user_id,
        "stages": {},
        "embeddings_by_type": {},
    }
    try:
        db = _connect()
        archive = db.open_table("archive")
    except Exception as e:
        out["error"] = f"open archive failed: {e}"
        return out

    base = f"user_id = '{_quote(user_id)}'" if user_id else None

    def cnt(extra: str | None = None) -> int:
        if base and extra:
            f: str | None = f"({base}) AND ({extra})"
        else:
            f = base or extra
        try:
            return archive.count_rows(filter=f) if f else archive.count_rows()
        except Exception:
            return 0

    total = cnt()
    out["archive_rows"] = total
    if total == 0:
        return out

    # ── Headline stages ──────────────────────────────────────────────
    # NOTE: DataFusion's ``length()`` on list columns returns the
    # underlying byte length (or fails on List<Struct>), NOT the
    # element count. Use ``cardinality()`` which is the canonical
    # SQL/array function for list size and works on List<Utf8>,
    # List<Struct>, and List<Float>.
    stages: dict[str, dict[str, int | float]] = {}
    for stage in _LIST_STAGES:
        n = cnt(f"cardinality({stage}) > 0")
        stages[stage] = {
            "covered": n,
            "missing": total - n,
            "percent": _pct(n, total),
        }
    n_class = cnt("block_class IS NOT NULL AND block_class != ''")
    stages["block_class"] = {
        "covered": n_class,
        "missing": total - n_class,
        "percent": _pct(n_class, total),
    }
    out["stages"] = stages

    # ── Recallable subset ────────────────────────────────────────────
    rec_total = cnt("recallable = TRUE")
    out["recallable_archive_rows"] = rec_total
    out["recallable_total"] = rec_total
    out["not_recallable_total"] = total - rec_total

    stages_rec: dict[str, dict[str, int | float]] = {}
    for stage in _LIST_STAGES:
        n = cnt(f"recallable = TRUE AND cardinality({stage}) > 0")
        stages_rec[stage] = {
            "covered": n,
            "missing": rec_total - n,
            "percent": _pct(n, rec_total),
        }
    n_class_rec = cnt(
        "recallable = TRUE AND block_class IS NOT NULL AND block_class != ''"
    )
    stages_rec["block_class"] = {
        "covered": n_class_rec,
        "missing": rec_total - n_class_rec,
        "percent": _pct(n_class_rec, rec_total),
    }
    out["stages_recallable"] = stages_rec

    # ── Vector coverage ──────────────────────────────────────────────
    # DataFusion uses 1-based indexing into fixed_size_list. A row with
    # an all-zero placeholder vector returns ``vector[1] = 0.0``; once
    # the embedder writes a real vector, vector[1] is non-zero in
    # practice (768-d unit vectors are dense).
    vec_nz = cnt("vector[1] != 0.0")
    out["vector_coverage"] = {
        "covered": vec_nz,
        "missing": total - vec_nz,
        "percent": _pct(vec_nz, total),
    }

    # ── Class breakdown × recallable (small projection scan) ─────────
    try:
        q = archive.search().select(["block_class", "recallable"])
        if base:
            q = q.where(base)
        bc_arr = q.limit(total + 10).to_arrow()
        bc_col = bc_arr.column("block_class").to_pylist()
        rec_col = bc_arr.column("recallable").to_pylist()
        breakdown: dict[str, dict[str, int]] = {}
        for cls, rec in zip(bc_col, rec_col):
            label = (cls or "").strip() or "UNCLASSIFIED"
            row = breakdown.setdefault(
                label, {"recallable": 0, "not_recallable": 0, "total": 0}
            )
            if rec is True:
                row["recallable"] += 1
            else:
                row["not_recallable"] += 1
            row["total"] += 1
        out["class_breakdown"] = dict(
            sorted(breakdown.items(), key=lambda kv: -kv[1]["total"])
        )
    except Exception as e:
        out["class_breakdown_error"] = str(e)[:160]

    # ── Link totals (sum of list lengths across rows) ────────────────
    # count_rows can't express SUM; we'd need GROUP BY-like semantics
    # that DataFusion supports but lancedb.Table.count_rows doesn't
    # expose. Approximate with a count of "has at least one link" per
    # type — already computed above in ``stages`` — and surface that
    # under ``link_totals`` for backwards compat with the dashboard.
    # If the operator needs precise edge counts they can drop into
    # the TCMM service; the dashboard panel only needs an
    # at-a-glance "is the graph being built" signal.
    out["link_totals"] = {
        stage: stages.get(stage, {}).get("covered", 0)
        for stage in ("semantic_links", "entity_links",
                      "topic_links", "contextual_links")
    }

    # ── Embeddings table breakdown ───────────────────────────────────
    try:
        emb = db.open_table("embeddings")
        emb_total = (
            emb.count_rows(filter=base) if base else emb.count_rows()
        )
        out["embeddings_total"] = emb_total
        if emb_total:
            q = emb.search().select(["emb_type"])
            if base:
                q = q.where(base)
            emb_arr = q.limit(emb_total + 10).to_arrow()
            types = emb_arr.column("emb_type").to_pylist()
            out["embeddings_by_type"] = dict(
                sorted({t: types.count(t) for t in set(types)}.items())
            )
    except Exception as e:
        out["embeddings_error"] = str(e)[:160]

    return out


# ── fts_index_status ─────────────────────────────────────────────────


@_cached
def fts_index_status() -> dict[str, Any]:
    """Whether the sparse table has its FTS index built.

    Lance stores indices under ``<table>.lance/_indices/``. We just
    check that path exists and report mtime so we know when it last
    rebuilt.
    """
    if _PG:
        return _pg_fts_status()
    sparse_dir = Path(LANCE_DIR) / "sparse_archive.lance" / "_indices"
    if not sparse_dir.exists():
        return {"built": False, "path": str(sparse_dir)}
    indices = list(sparse_dir.iterdir())
    return {
        "built": bool(indices),
        "count": len(indices),
        "path": str(sparse_dir),
        "last_mtime": max((p.stat().st_mtime for p in indices), default=0),
    }
