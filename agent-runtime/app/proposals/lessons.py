"""Phase 3 — org_memory lesson promotion (§3.9.2 + §3.10).

Lessons are RULES about how the organization operates — distinct from
skills (how-tos at `agents/skills/`) and from blackboard (factual
knowledge in dream).  They live in the `org_memory` Lance table.

Promotion path:
  reflective_heuristic dream node
    │
    ▼  candidate: trigger + rule extracted from the node
    │
    ▼  apply promotion criteria:
       - high confidence (≥ promotion threshold, default 0.75)
       - reinforced ≥ N times by ≥2 distinct extracted_by sources
    │
    ▼  write org_memory row with expires_at/review_after stamps
    │
    ▼  Phase 4 surface: "Lessons due for review" queue +
       constitution-amendment proposals for ≥5×-reinforced + 0.75 conf

This module owns:
  - `extract_lesson_from_dream_row()` — pure-function shape transform
  - `evaluate_promotion()`   — pure-function criteria checker
  - `write_lesson()`         — Lance insert with the safety defaults
  - `LessonPromotionWorker`  — periodic scan over dream_archive

Cadence default 24h.  Idempotent: lessons keyed on `promoted_from`
(reflective_heuristic node id); re-running with no new heuristic
nodes is a no-op.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Promotion thresholds — defaults from spec §3.9.2.

PROMOTION_CONFIDENCE_MIN = 0.50
"""Minimum heuristic confidence to be eligible for promotion."""

PROMOTION_REINFORCEMENT_MIN = 2
"""Minimum reinforcement count (how many times the pattern surfaced)."""

PROMOTION_DISTINCT_AGENTS_MIN = 2
"""Spec §3.8.5: reinforcement requires evidence from ≥2 distinct
``extracted_by`` values to cross the tenant-trust boundary."""

# Constitution-amendment-eligibility thresholds (spec §3.10).
CONST_AMENDMENT_CONFIDENCE_MIN = 0.75
CONST_AMENDMENT_REINFORCEMENT_MIN = 5

# Lifetime defaults.
DEFAULT_EXPIRES_DAYS = 180
DEFAULT_REVIEW_AFTER_DAYS = 90
DEFAULT_CONFIDENCE_DECAY_PER_WEEK = 0.02


def _now() -> float:
    return time.time()


def _new_id() -> str:
    return f"lsn-{uuid.uuid4().hex[:12]}"


def extract_lesson_from_dream_row(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Pull (trigger, rule) out of a dream_archive row that has
    ``block_class=REFLECTIVE_HEURISTIC`` (or its variants).

    The dream layer doesn't currently emit a strict
    `{trigger, rule}` shape — it emits text + claims.  v1 heuristic:
      * `trigger` = first claim if present, else first sentence of text
      * `rule`    = remaining claims joined, else second sentence + on
      * Both fall back to empty strings; caller should treat empty
        trigger/rule as "couldn't extract — skip promotion"

    Returns ``None`` when the row isn't a reflective_heuristic-style
    candidate at all.
    """
    bc = (row.get("block_class") or "").upper()
    if bc not in ("REFLECTIVE_HEURISTIC", "RECURRING_RITUAL"):
        return None
    text = (row.get("text") or "").strip()
    claims = row.get("claims") or []
    if not claims and not text:
        return None
    if claims:
        trigger = (claims[0] if isinstance(claims[0], str) else str(claims[0])).strip()
        rule = " ".join(
            str(c).strip()
            for c in claims[1:]
            if isinstance(c, str) and c.strip()
        ) or text[:240]
    else:
        # No claims — split text by first '.'/'?' so we have both halves
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        trigger = sentences[0] if sentences else text[:120]
        rule = (". ".join(sentences[1:])).strip() if len(sentences) > 1 else trigger
    return {
        "trigger":          trigger,
        "rule":             rule,
        "promoted_from":    f"dream-aid-{row.get('aid') or row.get('id') or 'unknown'}",
    }


def evaluate_promotion(
    *,
    confidence: float,
    reinforcement_count: int,
    distinct_agents: int,
    confidence_min: float = PROMOTION_CONFIDENCE_MIN,
    reinforcement_min: int = PROMOTION_REINFORCEMENT_MIN,
    distinct_agents_min: int = PROMOTION_DISTINCT_AGENTS_MIN,
) -> dict[str, Any]:
    """Decide if a candidate is eligible for org_memory promotion.

    Returns:
        {
          "promote":              bool,
          "amendment_eligible":   bool,   # ≥5 reinforcements + ≥0.75 confidence
          "reasons":              [str, ...]  # why or why not
        }
    """
    reasons: list[str] = []
    promote = True
    if confidence < confidence_min:
        promote = False
        reasons.append(f"confidence {confidence:.2f} < min {confidence_min:.2f}")
    if reinforcement_count < reinforcement_min:
        promote = False
        reasons.append(
            f"reinforcement_count {reinforcement_count} < min {reinforcement_min}"
        )
    if distinct_agents < distinct_agents_min:
        promote = False
        reasons.append(
            f"distinct_agents {distinct_agents} < min {distinct_agents_min} "
            f"(spec §3.8.5 cross-agent reinforcement)"
        )
    amendment = (
        confidence >= CONST_AMENDMENT_CONFIDENCE_MIN
        and reinforcement_count >= CONST_AMENDMENT_REINFORCEMENT_MIN
    )
    return {
        "promote":            promote,
        "amendment_eligible": bool(amendment),
        "reasons":            reasons,
    }


def write_lesson(
    *,
    lessons_tbl: Any,
    tenant_id: str,
    user_id: str,
    trigger: str,
    rule: str,
    confidence: float,
    promoted_from: str,
    reinforcement_count: int = 1,
    reinforced_by_agent_ids: Optional[list[str]] = None,
    evidence_task_ids: Optional[list[str]] = None,
    constitution_version: Optional[int] = None,
    expires_at: Optional[float] = None,
    review_after: Optional[float] = None,
    decay_per_week: float = DEFAULT_CONFIDENCE_DECAY_PER_WEEK,
) -> str:
    """Insert one org_memory row.  Returns the new lesson id.

    Idempotency is at the caller's discretion — by convention we don't
    re-write a row that already has the same ``promoted_from`` value;
    promote_one() handles that check.
    """
    now = _now()
    return _write_lesson_row(
        lessons_tbl=lessons_tbl,
        row={
            "id":                          _new_id(),
            "kind":                        "lesson",
            "status":                      "active",
            "parent_id":                   None,
            "lineage_chain":               [],
            "tenant_id":                   tenant_id,
            "user_id":                     user_id,
            "created_by_agent_id":         "lesson-promotion",
            "created_ts":                  now,
            "updated_ts":                  now,
            "cost_attributed_usd":         0.0,
            "trigger":                     trigger,
            "rule":                        rule,
            "confidence":                  float(confidence),
            "evidence_task_ids":           evidence_task_ids or [],
            "promoted_from":               promoted_from,
            "reinforcement_count":         int(reinforcement_count),
            "reinforced_by_agent_ids":     reinforced_by_agent_ids or [],
            "last_reinforced_ts":          now,
            "expires_at":                  expires_at if expires_at is not None else now + DEFAULT_EXPIRES_DAYS * 86400.0,
            "review_after":                review_after if review_after is not None else now + DEFAULT_REVIEW_AFTER_DAYS * 86400.0,
            "confidence_decay_per_week":   float(decay_per_week),
            "reviews_json":                "[]",
            "imported_from_tenant":        None,
            "imported_from_lesson_id":     None,
            "constitution_version":        constitution_version,
            "extras_json":                 None,
        },
    )


def _write_lesson_row(*, lessons_tbl: Any, row: dict[str, Any]) -> str:
    lessons_tbl.add([row])
    return row["id"]


def _existing_promoted_from(
    *,
    lessons_tbl: Any,
    tenant_id: str,
    user_id: str,
    promoted_from: str,
) -> bool:
    """Idempotency check: did we already promote this dream node?

    Post-M1 cutover: queries TCMM `archive` instead of org_memory.
    `lessons_tbl` is now an unused vestigial param kept for back-compat
    with test fixtures.  The check looks for any existing `[lesson]`
    observation with the same promoted_from in extracted_by.
    """
    try:
        from ..memory.lessons_reader import list_lessons
        for L in list_lessons(
            tenant_id=tenant_id, user_id=user_id, status="all", limit=500,
        ):
            # promoted_from in the lessons_reader output mirrors the
            # original aid / dream-node id; check exact match.
            if L.get("promoted_from") == promoted_from:
                return True
        return False
    except Exception:
        return False


async def promote_one(
    *,
    lessons_tbl: Any,
    tenant_id: str,
    user_id: str,
    dream_row: dict[str, Any],
    confidence: float,
    reinforcement_count: int = 1,
    distinct_agents: int = 1,
    reinforced_by_agent_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Evaluate + (maybe) write one lesson.  Returns a summary dict.

    {
      "promoted":            bool,
      "amendment_eligible":  bool,
      "lesson_id":           str | None,
      "reasons":             [str, ...]   # gate failures, empty when promoted
    }
    """
    extracted = extract_lesson_from_dream_row(dream_row)
    if extracted is None:
        return {
            "promoted": False, "amendment_eligible": False,
            "lesson_id": None, "reasons": ["row is not a reflective_heuristic"],
        }
    if not extracted["trigger"] or not extracted["rule"]:
        return {
            "promoted": False, "amendment_eligible": False,
            "lesson_id": None,
            "reasons": ["empty trigger or rule after extraction"],
        }
    decision = evaluate_promotion(
        confidence=confidence,
        reinforcement_count=reinforcement_count,
        distinct_agents=distinct_agents,
    )
    if not decision["promote"]:
        return {
            "promoted":           False,
            "amendment_eligible": decision["amendment_eligible"],
            "lesson_id":          None,
            "reasons":            decision["reasons"],
        }
    if _existing_promoted_from(
        lessons_tbl=lessons_tbl,
        tenant_id=tenant_id, user_id=user_id,
        promoted_from=extracted["promoted_from"],
    ):
        return {
            "promoted":           False,
            "amendment_eligible": decision["amendment_eligible"],
            "lesson_id":          None,
            "reasons":            ["already promoted (idempotency hit)"],
        }
    # Post-M1 cutover: TCMM is the sole destination.  The Phase 7 M1
    # split-writer observes to TCMM `team/knowledge/<team_id>` (which
    # the middleware routes to `agent/<critic_id>/observations/<uid>`
    # in practice — see [F4c_LESSON_NAMESPACE_2026_05_28]).  Legacy
    # `write_lesson` call removed.
    try:
        from ..memory.phase_7_writers import promote_lesson_to_team_knowledge
        ok = await promote_lesson_to_team_knowledge(
            tenant_id=tenant_id,
            user_id=user_id,
            team_id=tenant_id,  # v1: team scope = tenant scope
            trigger=extracted["trigger"],
            rule=extracted["rule"],
            confidence=confidence,
            critic_id="critic-prose",
            promoted_from=extracted["promoted_from"],
        )
        if not ok:
            return {
                "promoted":           False,
                "amendment_eligible": decision["amendment_eligible"],
                "lesson_id":          None,
                "reasons":            ["TCMM observation failed"],
            }
    except Exception as e:
        return {
            "promoted":           False,
            "amendment_eligible": decision["amendment_eligible"],
            "lesson_id":          None,
            "reasons":            [f"writer error: {e}"],
        }
    # Synthetic lesson_id from the promotion source (TCMM doesn't echo
    # back its archive aid in the response shape today).
    lid = f"lsn-from-{extracted['promoted_from']}"
    return {
        "promoted":           True,
        "amendment_eligible": decision["amendment_eligible"],
        "lesson_id":          lid,
        "reasons":            [],
    }


async def run_one_cycle(
    *,
    db_path: str = "/tcmm-data/veilguard/tcmm.db",
    confidence_field: str = "density_score",  # v1 proxy for heuristic confidence
) -> dict[str, Any]:
    """Scan dream_archive for reflective_heuristic / recurring_ritual
    rows and promote eligible ones to org_memory.

    Returns aggregate stats per cycle.  Idempotent.

    `confidence_field` — which dream-archive column to use as the
    heuristic confidence.  Dream doesn't emit a typed `confidence`
    field today; `density_score` is a defensible v1 proxy (clusters
    that are dense are more trustworthy).  Bump to a typed field when
    dream-engine grows one.
    """
    try:
        import lancedb
    except ImportError:
        return {"scanned": 0, "promoted": 0, "error": "lancedb missing"}
    try:
        from ..ledger.store import open_ledger_db; db = open_ledger_db(db_path)
        dream_tbl = db.open_table("dream_archive")
        # [M1_CUTOVER_2026_05_28] `org_memory` is gone post-cutover —
        # promote_one no longer needs a Lance table handle; it writes
        # via the Phase 7 split-writer.  We pass None as the
        # `lessons_tbl` placeholder so the signature stays stable.
        lessons_tbl = None
    except Exception as e:
        return {"scanned": 0, "promoted": 0, "error": f"open: {e}"}
    try:
        arr = (
            dream_tbl.search()
            .limit(2000)
            .to_arrow()
        )
    except Exception as e:
        return {"scanned": 0, "promoted": 0, "error": f"scan: {e}"}
    scanned = arr.num_rows
    promoted = 0
    skipped = 0
    amendment_candidates = 0
    reasons_counter: dict[str, int] = {}
    for i in range(arr.num_rows):
        row = {c: arr.column(c)[i].as_py() for c in arr.column_names}
        bc = (row.get("block_class") or "").upper()
        if bc not in ("REFLECTIVE_HEURISTIC", "RECURRING_RITUAL"):
            continue
        tenant_id = row.get("user_id") or ""
        user_id = tenant_id
        confidence = float(row.get(confidence_field) or 0.0)
        # v1 proxies for reinforcement: count of source_block_ids,
        # which approximates "how many underlying observations rolled
        # into this dream node".  distinct_agents comes from
        # extracted_by — fall back to 1 when absent.
        source_ids = row.get("source_block_ids") or []
        reinforcement = max(1, len(source_ids))
        eb_str = row.get("extracted_by") or ""
        agents = [a for a in str(eb_str).split(",") if a.strip()]
        distinct_agents = max(1, len(set(agents)))
        result = await promote_one(
            lessons_tbl=lessons_tbl,
            tenant_id=tenant_id, user_id=user_id,
            dream_row=row,
            confidence=confidence,
            reinforcement_count=reinforcement,
            distinct_agents=distinct_agents,
            reinforced_by_agent_ids=list(set(agents)),
        )
        if result["promoted"]:
            promoted += 1
        else:
            skipped += 1
            for r in result["reasons"]:
                reasons_counter[r] = reasons_counter.get(r, 0) + 1
        if result["amendment_eligible"]:
            amendment_candidates += 1
    out = {
        "scanned":               scanned,
        "promoted":              promoted,
        "skipped":               skipped,
        "amendment_candidates":  amendment_candidates,
        "skip_reasons":          reasons_counter,
    }
    if promoted or amendment_candidates:
        logger.info(f"[lessons] cycle: {out}")
    return out


# ── Lifecycle management — DEPRECATED post-M1-cutover (2026-05-28) ─────
#
# These functions all operated on the dropped  Lance table.
# Post-M1 cutover, lessons live in TCMM  (read via
# ), and lifecycle is observation-marker
# based ( / ) rather than in-place
# Lance row mutation.
#
# Stubs are kept so callers (LessonPromotionWorker maintenance loop,
# any external scripts) don't crash on import — they log a warning
# once and return a zero summary.  When the maintenance-as-observations
# pattern lands properly, these become real functions again.

import warnings as _warnings

_DEPRECATION_LOGGED: set[str] = set()


def _warn_once(name: str) -> None:
    if name in _DEPRECATION_LOGGED:
        return
    _DEPRECATION_LOGGED.add(name)
    logger.warning(
        f"[lessons] {name} is deprecated post-M1 cutover; "
        f"see decision log 2026-05-28 — returning no-op result"
    )


def find_lessons_due_for_review(
    *, lessons_tbl: Any = None, tenant_id: str, user_id: str,
    now_ts: Optional[float] = None, limit: int = 200,
) -> list[dict[str, Any]]:
    """DEPRECATED — use ."""
    _warn_once("find_lessons_due_for_review")
    return []


def expire_stale_lessons(
    *, lessons_tbl: Any = None, tenant_id: str, user_id: str,
    now_ts: Optional[float] = None,
) -> int:
    """DEPRECATED — observation-marker pattern () replaces it."""
    _warn_once("expire_stale_lessons")
    return 0


def apply_confidence_decay(
    *, lessons_tbl: Any = None, tenant_id: str, user_id: str,
    decay_per_week: float = DEFAULT_CONFIDENCE_DECAY_PER_WEEK,
    now_ts: Optional[float] = None,
) -> int:
    """DEPRECATED — decay-via-observation not yet implemented; returns 0."""
    _warn_once("apply_confidence_decay")
    return 0


def reinforce_lesson(
    *, lessons_tbl: Any = None, tenant_id: str, user_id: str,
    promoted_from: str, reinforcing_agent_id: Optional[str] = None,
) -> bool:
    """DEPRECATED — reinforcement = additional TCMM observation by a new author."""
    _warn_once("reinforce_lesson")
    return False


def run_maintenance_cycle(
    *, db_path: str = "/tcmm-data/veilguard/tcmm.db",
) -> dict[str, Any]:
    """DEPRECATED — no-op summary so the worker loop doesn't crash."""
    _warn_once("run_maintenance_cycle")
    return {"tenants": 0, "expired": 0, "decayed": 0, "deprecated": True}




# ── Background worker ──────────────────────────────────────────────────


class LessonPromotionWorker:
    """Daily scan for promotable reflective_heuristic dream nodes."""

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
            f"[lessons] worker starting; interval={self.interval_seconds:.0f}s "
            f"conf_min={PROMOTION_CONFIDENCE_MIN} "
            f"reinf_min={PROMOTION_REINFORCEMENT_MIN} "
            f"distinct_agents_min={PROMOTION_DISTINCT_AGENTS_MIN}"
        )
        try:
            while not self._stop_evt.is_set():
                # Promotion cycle — maintenance was deprecated in the
                # 2026-05-28 M1 cutover (org_memory Lance table dropped;
                # lessons live in TCMM; expiry/decay re-derived on read
                # from observation count + age in `lessons_reader`).
                # The old `run_maintenance_cycle()` no-op call was
                # emitting a deprecation WARNING every cycle for no
                # signal — removed 2026-05-28.
                try:
                    await run_one_cycle(db_path=self.db_path)
                except Exception as e:
                    logger.exception(f"[lessons] promotion cycle error: {e}")
                try:
                    await asyncio.wait_for(
                        self._stop_evt.wait(), timeout=self.interval_seconds,
                    )
                except asyncio.TimeoutError:
                    pass
        finally:
            logger.info("[lessons] worker stopped")


__all__ = [
    "PROMOTION_CONFIDENCE_MIN",
    "PROMOTION_REINFORCEMENT_MIN",
    "PROMOTION_DISTINCT_AGENTS_MIN",
    "CONST_AMENDMENT_CONFIDENCE_MIN",
    "CONST_AMENDMENT_REINFORCEMENT_MIN",
    "extract_lesson_from_dream_row",
    "evaluate_promotion",
    "promote_one",
    "run_one_cycle",
    "run_maintenance_cycle",
    "find_lessons_due_for_review",
    "expire_stale_lessons",
    "apply_confidence_decay",
    "reinforce_lesson",
    "LessonPromotionWorker",
]
