"""Pull live counters from the TCMM MCP service (port 8811).

The TCMM service exposes /debug_workers, /debug_persist, /api/sessions,
/api/memory_heatmap, /api/token_stats. We fan out over them with a
short timeout so a single slow endpoint can't stall the dashboard.
"""
from __future__ import annotations

import os
from typing import Any

import httpx


TCMM_BASE = os.environ.get("TCMM_URL", "http://127.0.0.1:8811").rstrip("/")
TIMEOUT = 3.0

# When TCMM is on Postgres, the NLP/embedding-progress counts come from the CTI
# `archive` stage flags instead of a Lance scan.
_PG = os.environ.get("TCMM_STORAGE", "postgres").lower() in ("postgres", "postgresql")
_TCMM_DSN = os.environ.get("TCMM_DATABASE_URL", "postgresql://tcmm:tcmm@localhost:5432/tcmm")


def _pg_nlp_progress() -> dict[str, int]:
    """archive total + semantic_done/embedding_done counts from Postgres."""
    out = {"archive_total": 0, "nlp_processed": 0, "nlp_pending": 0,
           "embed_processed": 0, "embed_pending": 0}
    try:
        import psycopg2
        conn = psycopg2.connect(_TCMM_DSN)
        conn.autocommit = True   # read-only stats: don't sit in an open transaction
        try:
            with conn.cursor() as c:
                c.execute("SELECT count(*), count(*) FILTER (WHERE semantic_done), "
                          "count(*) FILTER (WHERE embedding_done) FROM archive")
                total, sem, emb = c.fetchone()
        finally:
            conn.close()
        out["archive_total"] = total or 0
        out["nlp_processed"] = sem or 0
        out["embed_processed"] = emb or 0
        out["nlp_pending"] = out["archive_total"] - out["nlp_processed"]
        out["embed_pending"] = out["archive_total"] - out["embed_processed"]
    except Exception:
        pass
    return out


async def _get(path: str) -> dict[str, Any] | None:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get(f"{TCMM_BASE}{path}")
            if r.status_code != 200:
                return None
            return r.json()
    except Exception:
        return None


async def health() -> dict[str, Any]:
    h = await _get("/health")
    if h is None:
        return {"status": "down", "service": "tcmm-memory"}
    return h


def _lance_nlp_progress() -> dict[str, int]:
    """Count NLP/embedding completion straight from the archive table.

    The ``/debug_workers`` endpoint reports per-session in-memory
    counters (``semantic.enqueue``, ``semantic.dequeue`` etc.), but real
    NLP processing happens on the **process-level shared polling
    worker** that iterates Lance directly via ``WHERE semantic_done=
    false``. That shared worker doesn't update any TCMM instance's
    per-session metrics, so the per-session counters always look zero
    for ``processed`` / ``success`` even when the workers have caught
    up on thousands of rows. The dashboard tiles bound to those
    counters then permanently show "—" / 0, which is what made the
    user think the workers were stuck (PROCESSED column blank).

    Reading Lance is the only honest answer to "how much of the
    archive has been enriched". This is a cold scan of two boolean
    columns over the archive table — single-digit ms after compaction.
    Returns zeros if the archive is unreachable so the frontend's
    placeholders render rather than crashing.
    """
    if _PG:
        return _pg_nlp_progress()
    out = {
        "archive_total": 0,
        "nlp_processed": 0,
        "nlp_pending": 0,
        "embed_processed": 0,
        "embed_pending": 0,
    }
    try:
        # Use lancedb (already an admin-container dep — same package
        # pii_audit_stats uses) instead of the lower-level ``lance``
        # package which isn't installed here. The .to_arrow() result
        # is the same pyarrow Table either way, so the column-sum math
        # below is identical to a direct lance.dataset() call.
        import lancedb
        import pyarrow.compute as pc
        db_dir = os.environ.get(
            "ADMIN_LANCE_DIR",
            "/tcmm-data/veilguard/tcmm.db",
        )
        db = lancedb.connect(db_dir)
        tbl = db.open_table("archive")
        arr = tbl.to_arrow().select(["semantic_done", "embedding_done"])
        out["archive_total"] = len(arr)
        if len(arr):
            sem_int = pc.cast(arr.column("semantic_done"), "int32")
            emb_int = pc.cast(arr.column("embedding_done"), "int32")
            out["nlp_processed"] = pc.sum(sem_int).as_py() or 0
            out["embed_processed"] = pc.sum(emb_int).as_py() or 0
            out["nlp_pending"] = out["archive_total"] - out["nlp_processed"]
            out["embed_pending"] = out["archive_total"] - out["embed_processed"]
    except Exception:
        # Lance unreachable / table missing — return zeros and let the
        # frontend render placeholders. Worker liveness is the
        # authoritative health check elsewhere; this is just for the
        # progress display.
        pass
    return out


async def workers() -> dict[str, Any]:
    """Return: threads liveness, queue depths, NLP progress + adapter type.

    Mixes two data sources because each tells you something different:

    * ``/debug_workers`` (TCMM service): thread liveness + per-session
      in-memory queue depths + adapter type. Authoritative for "are
      the worker threads alive".

    * Lance archive table (direct read): authoritative source for
      "what fraction of archive blocks are enriched" — counts
      ``semantic_done=true`` / ``embedding_done=true`` directly. This
      is what the dashboard's PROCESSED / PENDING tiles display.

    When TCMM is unreachable ``_get`` returns ``None`` and we surface
    ``nlp_adapter = None`` so the frontend's ``!!recall.nlp_adapter``
    health-pill check goes red. Returning ``"?"`` here would falsely
    paint the pill green because any non-empty string is truthy in JS.
    """
    raw = await _get("/debug_workers")
    d = raw or {}
    threads = d.get("threads", {})

    # Lance source-of-truth pulls a small archive scan (~ms after
    # compaction). Run in a thread so it doesn't block the event loop.
    import asyncio as _asyncio
    progress = await _asyncio.to_thread(_lance_nlp_progress)

    return {
        "alive": all(v.get("alive") for v in threads.values()) if threads else False,
        "thread_states": {k: bool(v.get("alive")) for k, v in threads.items()},
        "queues": d.get("queues", {}),
        "metrics": d.get("metrics", {}),
        "nlp_adapter": (d.get("nlp") or {}).get("type") if raw else None,
        # Source-of-truth NLP/embedding progress from Lance.
        **progress,
    }


async def sessions() -> dict[str, Any]:
    """Active session count + recent stats."""
    d = await _get("/api/sessions") or {}
    return d


async def token_stats() -> dict[str, Any]:
    """Token cost / cache stats."""
    d = await _get("/api/token_stats") or {}
    return d


async def memory_heatmap() -> dict[str, Any]:
    """Heat distribution across archive."""
    d = await _get("/api/memory_heatmap") or {}
    return d
