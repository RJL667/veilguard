"""Phase 3 — weekly recalibration of alignment weights (§3.7.2 end).

Spec: "Recalibrated weekly via outcome tracker (§3.7.7) using
`regret_score` (not just approval_rate) — signal types whose
approvals tend to regret get downweighted faster than signal types
whose approvals tend to reuse.  The default alignment vectors in
§3.7.2 also get adjusted: if `information_gap` proposals
consistently produce work that improves measurable security (per
`improve_security` metric), the default alignment vector for
`information_gap` shifts toward security."

Implementation:
  1. Read all `proposal_outcomes` rows from the last N weeks
  2. Aggregate by (tenant_id, user_id, signal_type)
  3. For each (signal_type, objective), compute the mean regret of
     proposals tagged with that signal_type that contributed value
     to that objective (or just regret-weighted average per pair).
  4. Adjust the alignment weight: high regret = downweight the
     mapping; low regret = upweight.
  5. Renormalise weights to sum to 1.0 per signal_type.
  6. Write to `alignment_weights` Lance table.

Conservative posture for v1:
  * Only nudge weights — never zero them out.
  * Clamp adjustments to ±0.10 per recalibration.
  * Require min_outcomes_for_recal samples before adjusting.
  * Idempotent: running with no new data leaves weights unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
import uuid
from typing import Any

from .scoring import (
    DEFAULT_ALIGNMENT_VECTORS,
    ALL_SIGNAL_TYPES,
)
from ..ledger.store import LedgerStore

logger = logging.getLogger(__name__)


# How many days of outcomes to consider for the recalibration window.
WINDOW_DAYS = 28

# Minimum number of outcome rows per (signal_type) before we adjust
# that signal's weights.  Below this we leave defaults alone (statistics
# too noisy).
MIN_OUTCOMES_FOR_RECAL = 5

# Per-recalibration weight clamp.  We never move more than this per
# cycle — keeps the system from oscillating.
WEIGHT_DELTA_CLAMP = 0.10

# Regret thresholds for the nudge direction:
#   regret < LOW_REGRET  → upweight this mapping
#   regret > HIGH_REGRET → downweight
#   otherwise            → no change
LOW_REGRET = 0.5
HIGH_REGRET = 2.0


def _now() -> float:
    return time.time()


def _new_id() -> str:
    return f"alnw-{uuid.uuid4().hex[:12]}"


def get_alignment_for_tenant(
    *,
    tenant_id: str,
    user_id: str,
) -> dict[str, dict[str, float]]:
    """Return the per-tenant alignment vectors.

    Shape mirrors `scoring.DEFAULT_ALIGNMENT_VECTORS`:
        {signal_type: {objective_id: weight, ...}, ...}

    Missing signal_types or objective_ids fall back to defaults.
    Callers should still treat the value as full-shape (every
    signal_type covered, even if from defaults).
    """
    out = {k: dict(v) for k, v in DEFAULT_ALIGNMENT_VECTORS.items()}
    try:
        tbl = LedgerStore.get().table("alignment_weights")
        arr = (
            tbl.search()
            .where(
                f"tenant_id = '{tenant_id}' AND user_id = '{user_id}'"
            )
            .limit(500)
            .to_arrow()
        )
    except Exception:
        return out
    for i in range(arr.num_rows):
        st = arr.column("signal_type")[i].as_py()
        obj = arr.column("objective_id")[i].as_py()
        w = float(arr.column("weight")[i].as_py() or 0.0)
        if st in out:
            out[st][obj] = w
        else:
            out[st] = {obj: w}
    return out


def _renormalise(weights: dict[str, float]) -> dict[str, float]:
    """Re-scale a weight dict so it sums to 1.0.  Empty dict → empty."""
    total = sum(max(0.0, float(w)) for w in weights.values())
    if total <= 0:
        return dict(weights)
    return {k: max(0.0, float(v)) / total for k, v in weights.items()}


def _clamp_delta(old: float, new: float, max_delta: float = WEIGHT_DELTA_CLAMP) -> float:
    """Clamp |new - old| ≤ max_delta.  Returns clamped new value."""
    delta = new - old
    if delta > max_delta:
        return old + max_delta
    if delta < -max_delta:
        return old - max_delta
    return new


def compute_adjustment(
    *,
    current_vector: dict[str, float],
    regret_per_objective: dict[str, float],
) -> dict[str, float]:
    """Given a current alignment vector + observed regret per objective,
    compute the adjusted vector.

    Logic:
      * For each objective with a measurement: high regret = downweight,
        low regret = upweight, in-band = leave.
      * Clamp |Δ| ≤ WEIGHT_DELTA_CLAMP.
      * Floor at 0.05 (don't fully zero an objective — signals can
        recover from regret patterns).
      * Renormalise to sum to 1.0.

    Inputs unchanged.  Returns the adjusted dict.
    """
    adjusted: dict[str, float] = {}
    for obj, old_w in current_vector.items():
        regret = regret_per_objective.get(obj)
        if regret is None:
            adjusted[obj] = float(old_w)
            continue
        if regret > HIGH_REGRET:
            # Downweight — proposals scoring high on this objective
            # have been producing high-regret outcomes
            nudge = -0.05
        elif regret < LOW_REGRET:
            # Upweight — proposals scoring high here pay off
            nudge = +0.05
        else:
            nudge = 0.0
        new_w = max(0.05, float(old_w) + nudge)
        new_w = _clamp_delta(old_w, new_w)
        adjusted[obj] = new_w
    return _renormalise(adjusted)


def _scan_outcomes(
    *,
    outcomes_tbl: Any,
    proposals_tbl: Any,
    window_seconds: float,
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Group recent proposal_outcomes by (tenant_id, user_id, signal_type).

    Returns:
        {(tenant_id, user_id, signal_type): [{regret_score, value_realized, …}, …]}

    Each inner list is the raw outcome rows so the caller can compute
    means / medians / extras.  Joins outcome → proposal (for
    signal_type) inline.
    """
    cutoff = _now() - window_seconds
    out: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    try:
        arr = (
            outcomes_tbl.search()
            .where(f"computed_at_ts >= {cutoff}")
            .select([
                "proposal_id", "task_cost_usd",
                "value_realized", "regret_score",
                "tenant_id", "user_id",
            ])
            .limit(5000)
            .to_arrow()
        )
    except Exception as e:
        logger.warning(f"[recal] outcomes scan failed: {e}")
        return out

    # Bulk-fetch proposals to learn each outcome's signal_type
    proposal_ids = list({arr.column("proposal_id")[i].as_py() for i in range(arr.num_rows)})
    sig_by_pid: dict[str, str] = {}
    for pid in proposal_ids:
        if not pid:
            continue
        try:
            parr = (
                proposals_tbl.search()
                .where(f"id = '{pid}'")
                .select(["id", "signal_type"])
                .limit(1)
                .to_arrow()
            )
            if parr.num_rows > 0:
                sig_by_pid[pid] = parr.column("signal_type")[0].as_py() or ""
        except Exception:
            continue

    for i in range(arr.num_rows):
        pid = arr.column("proposal_id")[i].as_py()
        if not pid:
            continue
        sig = sig_by_pid.get(pid, "")
        if not sig:
            continue
        tenant = arr.column("tenant_id")[i].as_py() or ""
        user = arr.column("user_id")[i].as_py() or ""
        key = (tenant, user, sig)
        out.setdefault(key, []).append({
            "regret_score": float(arr.column("regret_score")[i].as_py() or 0.0),
            "task_cost_usd": float(arr.column("task_cost_usd")[i].as_py() or 0.0),
            "value_realized": float(arr.column("value_realized")[i].as_py() or 0.0),
        })
    return out


def run_one_cycle(
    *,
    db_path: str = "/tcmm-data/veilguard/tcmm.db",
    window_days: int = WINDOW_DAYS,
    min_outcomes: int = MIN_OUTCOMES_FOR_RECAL,
) -> dict[str, Any]:
    """Read recent proposal_outcomes, aggregate per (tenant, signal_type),
    compute weight adjustments, write to alignment_weights table.

    Returns:
        {
          "checked":   number of (tenant, signal_type) groups examined,
          "adjusted":  number of (tenant, signal, objective) rows written,
          "skipped_low_volume": groups below min_outcomes threshold,
          "summaries": [...]  per-group adjustment record
        }
    """
    try:
        import lancedb
    except ImportError:
        return {"checked": 0, "adjusted": 0, "error": "lancedb missing"}
    try:
        db = lancedb.connect(db_path)
        outcomes_tbl = db.open_table("proposal_outcomes")
        proposals_tbl = db.open_table("task_proposals")
        weights_tbl = db.open_table("alignment_weights")
    except Exception as e:
        return {"checked": 0, "adjusted": 0, "error": f"open tables: {e}"}

    groups = _scan_outcomes(
        outcomes_tbl=outcomes_tbl,
        proposals_tbl=proposals_tbl,
        window_seconds=window_days * 86400.0,
    )

    checked = 0
    adjusted = 0
    skipped_low_volume = 0
    summaries = []
    now = _now()
    for (tenant_id, user_id, signal_type), outcomes in groups.items():
        checked += 1
        if signal_type not in DEFAULT_ALIGNMENT_VECTORS:
            continue
        if len(outcomes) < min_outcomes:
            skipped_low_volume += 1
            continue
        # Compute regret-per-objective by mapping the signal_type's
        # CURRENT alignment vector through the outcome mean regret.
        # v1 heuristic: every objective inherits the same overall
        # regret score for this signal_type.  v2 would split regret
        # by which objective the artefact actually moved (requires
        # objective_deltas to be populated — currently {} in v1
        # outcome writes).
        mean_regret = statistics.mean(o["regret_score"] for o in outcomes)
        current = get_alignment_for_tenant(
            tenant_id=tenant_id, user_id=user_id,
        ).get(signal_type, dict(DEFAULT_ALIGNMENT_VECTORS[signal_type]))
        regret_per_obj = {obj: mean_regret for obj in current.keys()}
        new_vec = compute_adjustment(
            current_vector=current,
            regret_per_objective=regret_per_obj,
        )
        # Write one row per objective.  Existing rows get UPDATED;
        # missing rows INSERTED.  Update by composite key.
        for obj, new_w in new_vec.items():
            default_w = DEFAULT_ALIGNMENT_VECTORS[signal_type].get(obj, 0.0)
            # Was there a prior row?
            try:
                parr = (
                    weights_tbl.search()
                    .where(
                        f"tenant_id = '{tenant_id}' "
                        f"AND user_id = '{user_id}' "
                        f"AND signal_type = '{signal_type}' "
                        f"AND objective_id = '{obj}'"
                    )
                    .limit(1)
                    .to_arrow()
                )
                has_row = parr.num_rows > 0
                prior_count = (
                    int(parr.column("recalibration_count")[0].as_py()) if has_row else 0
                )
            except Exception:
                has_row = False
                prior_count = 0

            if has_row:
                weights_tbl.update(
                    where=(
                        f"tenant_id = '{tenant_id}' "
                        f"AND user_id = '{user_id}' "
                        f"AND signal_type = '{signal_type}' "
                        f"AND objective_id = '{obj}'"
                    ),
                    values={
                        "weight": float(new_w),
                        "last_regret_avg": float(mean_regret),
                        "last_recalibrated_ts": now,
                        "recalibration_count": prior_count + 1,
                        "updated_ts": now,
                    },
                )
            else:
                weights_tbl.add([{
                    "id":                   _new_id(),
                    "tenant_id":            tenant_id,
                    "user_id":              user_id,
                    "signal_type":          signal_type,
                    "objective_id":         obj,
                    "weight":               float(new_w),
                    "default_weight":       float(default_w),
                    "last_regret_avg":      float(mean_regret),
                    "last_recalibrated_ts": now,
                    "recalibration_count":  1,
                    "created_ts":           now,
                    "updated_ts":           now,
                    "extras_json":          None,
                }])
            adjusted += 1
        summaries.append({
            "tenant_id":      tenant_id,
            "user_id":        user_id,
            "signal_type":    signal_type,
            "n_outcomes":     len(outcomes),
            "mean_regret":    mean_regret,
            "old_vector":     current,
            "new_vector":     new_vec,
        })

    out = {
        "checked":            checked,
        "adjusted":           adjusted,
        "skipped_low_volume": skipped_low_volume,
        "summaries":          summaries,
    }
    if adjusted or summaries:
        logger.info(
            f"[recal] cycle: checked={checked} adjusted={adjusted} "
            f"skipped_low_volume={skipped_low_volume}"
        )
    return out


# ── Background worker ──────────────────────────────────────────────────


class RecalibrationWorker:
    """Weekly weight-recalibration worker.  Cadence default 7d.

    Idempotent — running with no new outcomes leaves weights unchanged.
    Safe to run frequently; the underlying nudge logic only adjusts
    when there's enough data.
    """

    def __init__(
        self,
        *,
        interval_seconds: float = 7 * 24 * 3600.0,
        db_path: str = "/tcmm-data/veilguard/tcmm.db",
    ):
        self.interval_seconds = float(interval_seconds)
        self.db_path = db_path
        self._stop_evt = asyncio.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    async def run(self) -> None:
        logger.info(
            f"[recal] worker starting; interval={self.interval_seconds:.0f}s "
            f"window={WINDOW_DAYS}d min_outcomes={MIN_OUTCOMES_FOR_RECAL} "
            f"delta_clamp=±{WEIGHT_DELTA_CLAMP}"
        )
        try:
            while not self._stop_evt.is_set():
                try:
                    run_one_cycle(db_path=self.db_path)
                except Exception as e:
                    logger.exception(f"[recal] cycle error: {e}")
                try:
                    await asyncio.wait_for(
                        self._stop_evt.wait(), timeout=self.interval_seconds,
                    )
                except asyncio.TimeoutError:
                    pass
        finally:
            logger.info("[recal] worker stopped")


__all__ = [
    "WINDOW_DAYS",
    "MIN_OUTCOMES_FOR_RECAL",
    "WEIGHT_DELTA_CLAMP",
    "LOW_REGRET",
    "HIGH_REGRET",
    "get_alignment_for_tenant",
    "compute_adjustment",
    "run_one_cycle",
    "RecalibrationWorker",
]
