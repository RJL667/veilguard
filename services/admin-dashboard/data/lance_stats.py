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

import lancedb


LANCE_DIR = os.environ.get(
    "ADMIN_LANCE_DIR",
    "/home/rudol/veilguard/tcmm-data/veilguard/tcmm.db",
)

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
    return lancedb.connect(LANCE_DIR)


def _quote(s: str) -> str:
    """Escape a single-quoted SQL literal."""
    return s.replace("'", "''")


# ── overview ─────────────────────────────────────────────────────────


@_cached
def overview() -> dict[str, Any]:
    """Row counts + on-disk size per table + dataset version + fragment count.

    All errors are caught per-table — a missing or corrupt table can't
    take the dashboard down.
    """
    out: dict[str, Any] = {"path": LANCE_DIR, "tables": []}
    try:
        db = _connect()
    except Exception as e:
        out["error"] = f"open failed: {e}"
        return out

    try:
        names = list(db.table_names())
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
