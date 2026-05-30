"""Phase 3 — the two missing generative signal emitters (§3.7.1).

Spec: "2 require new emitter implementation in dream itself (Phase
3 task):
  * low_stability_cluster   (clusters that fail to stabilise)
  * stale_supersession_chain (chains untouched too long)

Until these emitters ship, proposal generation runs on 5 signal
types, not 7."

Rather than surgery on `dream_engine.py` (multi-hundred LOC each,
high regression risk), we emit them agent-runtime-side from
`archive.lance` directly.  Same target table (`dream_archive`)
the DreamScanner reads, same downstream pipeline.  When dream-engine
gains typed emitters, this scanner can swap to read those instead
without changing anything below.

Detection heuristics:

  low_stability_cluster:
    A cluster of archive rows sharing a topic where the avg
    confidence (per claim weight / density_score proxy) is below
    floor AND the cluster has > N claims attempted.  Surfaced as
    a `block_class=LOW_STABILITY_CLUSTER` row injected into
    `dream_archive` so DreamScanner picks it up on next cycle.

  stale_supersession_chain:
    A topic whose claim chain has gone untouched for > N days AND
    has > M claims linked by `lineage` supersession.  Surfaced as
    `block_class=STALE_SUPERSESSION_CHAIN`.

Both have v1 thresholds tuned conservatively — better to under-emit
than spam the queue.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections import defaultdict
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Tunables (env-overridable).

LOW_STABILITY_MIN_CLAIMS = int(os.environ.get(
    "VEILGUARD_LOW_STABILITY_MIN_CLAIMS", "5"
))
LOW_STABILITY_DENSITY_FLOOR = float(os.environ.get(
    "VEILGUARD_LOW_STABILITY_DENSITY_FLOOR", "0.30"
))

STALE_CHAIN_MIN_CLAIMS = int(os.environ.get(
    "VEILGUARD_STALE_CHAIN_MIN_CLAIMS", "3"
))
STALE_CHAIN_AGE_DAYS = float(os.environ.get(
    "VEILGUARD_STALE_CHAIN_AGE_DAYS", "60.0"
))


def _now() -> float:
    return time.time()


def _scan_topics(
    *,
    tbl: Any,
    limit: int = 5000,
) -> dict[str, list[dict[str, Any]]]:
    """Group archive rows by their primary topic.

    Returns {topic: [row, ...]} for the last `limit` rows.  Empty
    topics map to "_uncategorised" so the caller can ignore them
    cleanly.
    """
    # [EMPTY_ARCHIVE_GUARD_2026_05_28]  In local dev the `archive`
    # table is an empty placeholder (0 rows, 0 schema fields). Lance's
    # `.select([...])` then raises `SchemaError: No field named aid`
    # because the projection list references fields that don't exist
    # in the empty schema. The worker logged the same error every cycle
    # for the past day — noisy and misleading (the schema is FINE on
    # prod where archive has real rows). Short-circuit on empty tables
    # so log noise is proportional to real bugs.
    try:
        if tbl.count_rows() == 0:
            return {}
        sch_fields = {f.name for f in tbl.schema}
        if "aid" not in sch_fields or "topics" not in sch_fields:
            # Schema not initialised — archive hasn't been written yet
            # this lifetime.
            return {}
    except Exception:
        # If even count_rows fails, fall through to the original try/
        # except so the caller still sees a structured warning.
        pass
    try:
        arr = (
            tbl.search()
            .select([
                "aid", "topics", "claims", "density_score",
                "timestamp", "user_id", "namespace", "block_class",
                "source_block_ids",
            ])
            .limit(limit)
            .to_arrow()
        )
    except Exception as e:
        logger.warning(f"[signal_emitters] archive scan failed: {e}")
        return {}
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i in range(arr.num_rows):
        topics = arr.column("topics")[i].as_py() or []
        if not topics:
            continue
        primary = str(topics[0]) if topics else "_uncategorised"
        row = {c: arr.column(c)[i].as_py() for c in arr.column_names}
        out[primary].append(row)
    return out


def _inject_synthetic_dream_with_ts(
    *,
    dream_tbl: Any,
    user_id: str,
    block_class: str,
    text: str,
    topics: list[str],
    claims: list[str],
    source_block_ids: list[int],
    density_score: float,
    heat: float,
    timestamp: float,
    origin: str = "INFERRED",
) -> Optional[int]:
    """Same as _inject_synthetic_dream but lets caller specify the
    `timestamp` field — needed by stale_supersession_chain so the
    scanner's age_days derivation reflects cluster age, not emit time."""
    return _inject_synthetic_dream_inner(
        dream_tbl=dream_tbl, user_id=user_id, block_class=block_class,
        text=text, topics=topics, claims=claims,
        source_block_ids=source_block_ids,
        density_score=density_score, heat=heat, origin=origin,
        override_timestamp=timestamp,
    )


def _inject_synthetic_dream(
    *,
    dream_tbl: Any,
    user_id: str,
    block_class: str,
    text: str,
    topics: list[str],
    claims: list[str],
    source_block_ids: list[int],
    density_score: float,
    heat: float,
    origin: str = "INFERRED",
) -> Optional[int]:
    """Write a synthetic dream_archive row that DreamScanner will
    pick up on its next cycle.  Returns the new aid or None on
    failure.

    Schema-tolerant: fills required fields with defaults, ignores
    any fields the destination schema doesn't have.
    """
    return _inject_synthetic_dream_inner(
        dream_tbl=dream_tbl, user_id=user_id, block_class=block_class,
        text=text, topics=topics, claims=claims,
        source_block_ids=source_block_ids,
        density_score=density_score, heat=heat, origin=origin,
        override_timestamp=None,
    )


def _inject_synthetic_dream_inner(
    *,
    dream_tbl: Any,
    user_id: str,
    block_class: str,
    text: str,
    topics: list[str],
    claims: list[str],
    source_block_ids: list[int],
    density_score: float,
    heat: float,
    origin: str = "INFERRED",
    override_timestamp: Optional[float] = None,
) -> Optional[int]:
    try:
        schema_fields = {f.name for f in dream_tbl.schema}
    except Exception:
        return None
    now = _now()
    use_ts = override_timestamp if override_timestamp is not None else now
    aid = int(now * 1000) % (2**31)
    row: dict[str, Any] = {
        "aid":              aid,
        "namespace":        f"agent/signal-emitter/{block_class.lower()}",
        "user_id":          user_id,
        "id":               aid,
        "text":             text,
        "origin":           origin,
        "block_class":      block_class,
        "topics":           list(topics),
        "claims":           list(claims),
        "source_block_ids": list(source_block_ids),
        "density_score":    float(density_score),
        "heat":             float(heat),
        "timestamp":        float(use_ts),
        "created_ts":       int(use_ts),
        "session_id":       f"emit-{block_class[:10].lower()}",
        "session_date":     time.strftime("%Y-%m-%d", time.gmtime()),
        "recallable":       True,
        "vector":           [0.0] * 384,
        "knowledge_class":  "knowledge",
        "temporal":         {"prev_aid": 0, "next_aid": 0,
                             "prev_weight": 0.0, "next_weight": 0.0},
        "archive_stats":    {"attempts": 0, "used": 0},
        "lineage":          {"root": 0, "parents": []},
        "extras_json":      "{}",
        "extracted_by":     "agent:signal-emitter",
        "behavioural_json": "{}",
        "suppresses_json":  "{}",
        "semantic_json":    "{}",
        "entropy_static_json": "{}",
        "semantic_text":    text,
        "semantic_text_tail": "",
        "fingerprint":      0,
    }
    # Drop fields not in schema; fill required-not-null fields with defaults
    row = {k: v for k, v in row.items() if k in schema_fields}
    for f in dream_tbl.schema:
        if not f.nullable and row.get(f.name) is None:
            t = str(f.type)
            if "int" in t:           row[f.name] = 0
            elif "float" in t or "double" in t: row[f.name] = 0.0
            elif "bool" in t:        row[f.name] = False
            elif "string" in t:      row[f.name] = ""
            elif "list" in t:        row[f.name] = []
            elif "struct" in t:      row[f.name] = {}
    try:
        dream_tbl.add([row])
        return aid
    except Exception as e:
        logger.warning(f"[signal_emitters] inject failed: {e}")
        return None


def emit_low_stability_clusters(
    *,
    archive_tbl: Any,
    dream_tbl: Any,
    min_claims: int = LOW_STABILITY_MIN_CLAIMS,
    density_floor: float = LOW_STABILITY_DENSITY_FLOOR,
) -> dict[str, Any]:
    """Scan archive for low-stability clusters; emit one synthetic
    dream node per cluster.  Returns aggregate stats.
    """
    grouped = _scan_topics(tbl=archive_tbl)
    emitted = []
    skipped = 0
    for topic, rows in grouped.items():
        if topic == "_uncategorised" or len(rows) < min_claims:
            skipped += 1
            continue
        densities = [
            float(r.get("density_score") or 0.0)
            for r in rows
            if r.get("density_score") is not None
        ]
        if not densities:
            skipped += 1
            continue
        avg_density = sum(densities) / len(densities)
        if avg_density >= density_floor:
            skipped += 1
            continue
        # Have ourselves a low-stability cluster.  Don't re-emit if a
        # synthetic node with the same topic already exists in
        # dream_archive (idempotency).
        try:
            existing = (
                dream_tbl.search()
                .where(
                    f"block_class = 'LOW_STABILITY_CLUSTER' "
                    f"AND user_id = '{rows[0].get('user_id') or ''}'"
                )
                .limit(50)
                .to_arrow()
            )
            already = False
            for i in range(existing.num_rows):
                ex_topics = existing.column("topics")[i].as_py() or []
                if topic in ex_topics:
                    already = True
                    break
            if already:
                continue
        except Exception:
            pass
        all_claims = []
        for r in rows[:10]:
            cl = r.get("claims") or []
            all_claims.extend(c for c in cl if isinstance(c, str))
        source_ids = [int(r.get("aid") or 0) for r in rows[:10]]
        text = (
            f"Low-stability cluster detected on topic '{topic}': "
            f"avg density {avg_density:.2f} across {len(rows)} archive "
            f"rows (floor {density_floor:.2f}).  Re-investigate the "
            f"cluster or retire its claims with supersession notes."
        )
        new_aid = _inject_synthetic_dream(
            dream_tbl=dream_tbl,
            user_id=rows[0].get("user_id") or "",
            block_class="LOW_STABILITY_CLUSTER",
            text=text,
            topics=[topic],
            claims=all_claims[:5],
            source_block_ids=source_ids,
            density_score=avg_density,
            heat=1.0 - avg_density,  # inverse — instability = high signal
        )
        if new_aid:
            emitted.append({"topic": topic, "aid": new_aid, "rows": len(rows)})
    out = {
        "topics_scanned":   len(grouped),
        "emitted":          len(emitted),
        "skipped":          skipped,
        "rows":             emitted,
    }
    if emitted:
        logger.info(f"[signal_emitters.low_stability] cycle: {out}")
    return out


def emit_stale_supersession_chains(
    *,
    archive_tbl: Any,
    dream_tbl: Any,
    min_claims: int = STALE_CHAIN_MIN_CLAIMS,
    age_days: float = STALE_CHAIN_AGE_DAYS,
) -> dict[str, Any]:
    """Scan for topics whose claim chain hasn't been touched in >age_days.

    v1 proxy: group by topic, look at max(timestamp) — if it's older
    than the age cutoff AND the topic has ≥min_claims rows, emit.
    """
    cutoff = _now() - age_days * 86400.0
    grouped = _scan_topics(tbl=archive_tbl)
    emitted = []
    skipped = 0
    for topic, rows in grouped.items():
        if topic == "_uncategorised" or len(rows) < min_claims:
            skipped += 1
            continue
        timestamps = [float(r.get("timestamp") or 0.0) for r in rows]
        if not timestamps:
            skipped += 1
            continue
        latest = max(timestamps)
        if latest > cutoff:
            skipped += 1
            continue
        # Idempotency
        try:
            existing = (
                dream_tbl.search()
                .where(
                    f"block_class = 'STALE_SUPERSESSION_CHAIN' "
                    f"AND user_id = '{rows[0].get('user_id') or ''}'"
                )
                .limit(50)
                .to_arrow()
            )
            already = False
            for i in range(existing.num_rows):
                ex_topics = existing.column("topics")[i].as_py() or []
                if topic in ex_topics:
                    already = True
                    break
            if already:
                continue
        except Exception:
            pass
        age = (_now() - latest) / 86400.0
        source_ids = [int(r.get("aid") or 0) for r in rows[:10]]
        text = (
            f"Stale supersession chain on topic '{topic}': last touched "
            f"{age:.0f} days ago across {len(rows)} archive rows.  "
            f"Refresh-or-retire decision per claim recommended."
        )
        # Inject with the cluster's actual last-touched timestamp so
        # downstream scoring's `age_days` derivation reflects the
        # cluster age, not the emit time.
        new_aid = _inject_synthetic_dream_with_ts(
            dream_tbl=dream_tbl,
            user_id=rows[0].get("user_id") or "",
            block_class="STALE_SUPERSESSION_CHAIN",
            text=text,
            topics=[topic],
            claims=[],
            source_block_ids=source_ids,
            density_score=0.5,
            heat=min(1.0, age / 365.0),  # older → hotter (proxy)
            timestamp=latest,             # ← cluster's actual age
        )
        if new_aid:
            emitted.append({"topic": topic, "aid": new_aid,
                            "age_days": age, "rows": len(rows)})
    out = {
        "topics_scanned":  len(grouped),
        "emitted":         len(emitted),
        "skipped":         skipped,
        "rows":            emitted,
    }
    if emitted:
        logger.info(f"[signal_emitters.stale_chain] cycle: {out}")
    return out


def run_one_cycle(
    *, db_path: str = "/tcmm-data/veilguard/tcmm.db",
) -> dict[str, Any]:
    """One pass — run both emitters.  Daily cadence is fine."""
    try:
        import lancedb
        db = lancedb.connect(db_path)
        archive_tbl = db.open_table("archive")
        dream_tbl = db.open_table("dream_archive")
    except Exception as e:
        return {"error": f"open: {e}"}
    return {
        "low_stability":  emit_low_stability_clusters(
            archive_tbl=archive_tbl, dream_tbl=dream_tbl,
        ),
        "stale_chain":    emit_stale_supersession_chains(
            archive_tbl=archive_tbl, dream_tbl=dream_tbl,
        ),
    }


class SignalEmitterWorker:
    """Daily scan for the 2 missing dream signal types."""

    def __init__(
        self,
        *,
        interval_seconds: float = 24 * 3600.0,
        db_path: str = "/tcmm-data/veilguard/tcmm.db",
    ):
        self.interval_seconds = float(interval_seconds)
        self.db_path = db_path
        self._stop_evt = asyncio.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    async def run(self) -> None:
        logger.info(
            f"[signal_emitters] worker starting; "
            f"interval={self.interval_seconds:.0f}s "
            f"low_stab_floor={LOW_STABILITY_DENSITY_FLOOR} "
            f"stale_age_days={STALE_CHAIN_AGE_DAYS}"
        )
        try:
            while not self._stop_evt.is_set():
                try:
                    run_one_cycle(db_path=self.db_path)
                except Exception as e:
                    logger.exception(f"[signal_emitters] cycle error: {e}")
                try:
                    await asyncio.wait_for(
                        self._stop_evt.wait(), timeout=self.interval_seconds,
                    )
                except asyncio.TimeoutError:
                    pass
        finally:
            logger.info("[signal_emitters] worker stopped")


__all__ = [
    "LOW_STABILITY_MIN_CLAIMS",
    "LOW_STABILITY_DENSITY_FLOOR",
    "STALE_CHAIN_MIN_CLAIMS",
    "STALE_CHAIN_AGE_DAYS",
    "emit_low_stability_clusters",
    "emit_stale_supersession_chains",
    "run_one_cycle",
    "SignalEmitterWorker",
]
