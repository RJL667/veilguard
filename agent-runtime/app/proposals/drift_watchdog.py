"""Phase 3 — auto-pause on signal-quality drift (§3.7.3).

Spec: "if approved+pending+shelved proposals/day exceeds 3× the
trailing-30d median for that tenant, recalibration job emits
`signal_quality_drift` alert and auto-pauses the proactive stream
pending user review.  Prevents misconfigured constitution weights
from flooding the queue silently."

This module:
  1. Per tenant, computes daily proposal volume across last 30d
     (rolling window).
  2. Computes the trailing-30d median.
  3. Compares today's volume to 3× median; trips the gate when over.
  4. Calls `proactive_config.pause(reason="signal_quality_drift")`.

Idempotent: if the tenant is already paused, skip.  Daily cadence
(same worker schedule as outcomes).
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from typing import Any

from ..ledger import proactive_config
from ..ledger.store import LedgerStore

logger = logging.getLogger(__name__)


# Volume window: how many days back we look for the median.
WINDOW_DAYS = 30

# Trip threshold — today's count must exceed this multiplier of the
# trailing-30d median to auto-pause.  Per spec: 3×.
DRIFT_MULTIPLIER = 3.0

# Daily volume floor — if median is below this we ignore (low-volume
# tenants would otherwise auto-pause on a single noisy day).
DRIFT_MIN_MEDIAN = 1.0


def _daily_volume(
    *,
    tbl: Any,
    tenant_id: str,
    user_id: str,
    start_ts: float,
    end_ts: float,
) -> int:
    """Count proposals created for this tenant in [start_ts, end_ts).

    Counts ALL proposals (pending/approved/deferred/shelved/expired)
    so the watchdog sees the surface-side volume, not the
    converted-to-Task subset.
    """
    try:
        arr = (
            tbl.search()
            .where(
                f"tenant_id = '{tenant_id}' "
                f"AND user_id = '{user_id}' "
                f"AND created_ts >= {start_ts} "
                f"AND created_ts < {end_ts}"
            )
            .select(["id"])
            .limit(5000)
            .to_arrow()
        )
        return arr.num_rows
    except Exception as e:
        logger.warning(
            f"[drift] volume scan failed for {tenant_id}/{user_id}: {e}"
        )
        return 0


def evaluate_tenant(
    *,
    tbl: Any,
    tenant_id: str,
    user_id: str,
    now_ts: float | None = None,
    window_days: int = WINDOW_DAYS,
    multiplier: float = DRIFT_MULTIPLIER,
    min_median: float = DRIFT_MIN_MEDIAN,
) -> dict[str, Any]:
    """Compute drift state for one tenant.

    Returns a dict the caller logs and uses to decide auto-pause:
      {
        "tenant_id", "user_id",
        "daily_counts": [int per day, oldest..today],
        "median":       float,
        "today_count":  int,
        "threshold":    float,
        "drifted":      bool,
      }
    """
    now_ts = now_ts or time.time()
    day_seconds = 86400.0
    # Build per-day buckets covering [now - window_days, now]
    daily = []
    for d in range(window_days):
        start = now_ts - (window_days - d) * day_seconds
        end   = start + day_seconds
        daily.append(_daily_volume(
            tbl=tbl, tenant_id=tenant_id, user_id=user_id,
            start_ts=start, end_ts=end,
        ))
    today_count = daily[-1] if daily else 0
    trailing = daily[:-1] or [0]
    try:
        median = float(statistics.median(trailing))
    except statistics.StatisticsError:
        median = 0.0
    threshold = median * multiplier
    drifted = (
        median >= min_median
        and today_count > threshold
    )
    return {
        "tenant_id":    tenant_id,
        "user_id":      user_id,
        "daily_counts": daily,
        "median":       median,
        "today_count":  today_count,
        "threshold":    threshold,
        "drifted":      drifted,
    }


def discover_active_pairs(tbl: Any) -> list[tuple[str, str]]:
    """Find all (tenant_id, user_id) pairs that have proposals."""
    try:
        arr = (
            tbl.search()
            .select(["tenant_id", "user_id"])
            .limit(5000)
            .to_arrow()
        )
    except Exception as e:
        logger.warning(f"[drift] discover pairs failed: {e}")
        return []
    pairs = set()
    for i in range(arr.num_rows):
        t = arr.column("tenant_id")[i].as_py() or ""
        u = arr.column("user_id")[i].as_py() or ""
        if t and u:
            pairs.add((t, u))
    return sorted(pairs)


def run_one_cycle(
    *,
    db_path: str = "/tcmm-data/veilguard/tcmm.db",
    window_days: int = WINDOW_DAYS,
    multiplier: float = DRIFT_MULTIPLIER,
    min_median: float = DRIFT_MIN_MEDIAN,
) -> dict[str, Any]:
    """One scan + auto-pause pass.  Returns per-tenant summaries.

    For each tenant: evaluate drift, and if drifted AND not already
    paused, call `proactive_config.pause()`.  Idempotent — already
    paused tenants don't get re-stamped.
    """
    try:
        import lancedb
    except ImportError:
        return {"checked": 0, "paused": 0, "error": "lancedb missing"}
    try:
        from ..ledger.store import open_ledger_db; db = open_ledger_db(db_path)
        tbl = db.open_table("task_proposals")
    except Exception as e:
        return {"checked": 0, "paused": 0, "error": f"open: {e}"}

    pairs = discover_active_pairs(tbl)
    paused_now = 0
    results = []
    for tenant_id, user_id in pairs:
        ev = evaluate_tenant(
            tbl=tbl, tenant_id=tenant_id, user_id=user_id,
            window_days=window_days, multiplier=multiplier,
            min_median=min_median,
        )
        results.append(ev)
        if not ev["drifted"]:
            continue
        # Drifted — auto-pause if not already paused
        try:
            current = proactive_config.get_or_default(tenant_id, user_id)
            if current.paused:
                logger.debug(
                    f"[drift] {tenant_id}/{user_id} already paused, skip"
                )
                continue
            proactive_config.pause(
                tenant_id=tenant_id,
                user_id=user_id,
                reason=(
                    f"signal_quality_drift: today={ev['today_count']} "
                    f"median={ev['median']:.1f} threshold={ev['threshold']:.1f} "
                    f"(>{multiplier:.0f}× trailing 30d median)"
                ),
            )
            paused_now += 1
            logger.warning(
                f"[drift] AUTO-PAUSED {tenant_id}/{user_id}: "
                f"today={ev['today_count']} median={ev['median']:.1f} "
                f"threshold={ev['threshold']:.1f}"
            )
        except Exception as e:
            logger.exception(
                f"[drift] auto-pause failed for {tenant_id}/{user_id}: {e}"
            )
    summary = {
        "checked":  len(pairs),
        "paused":   paused_now,
        "results":  results,
    }
    if paused_now or any(r["drifted"] for r in results):
        logger.info(f"[drift] cycle: {summary}")
    return summary


# ── Background worker ──────────────────────────────────────────────────


class DriftWatchdog:
    """Daily auto-pause watchdog.  Shares cadence with OutcomesWorker."""

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
            f"[drift] watchdog starting; interval={self.interval_seconds:.0f}s "
            f"window={WINDOW_DAYS}d multiplier={DRIFT_MULTIPLIER}× "
            f"min_median={DRIFT_MIN_MEDIAN}"
        )
        try:
            while not self._stop_evt.is_set():
                try:
                    run_one_cycle(db_path=self.db_path)
                except Exception as e:
                    logger.exception(f"[drift] cycle error: {e}")
                try:
                    await asyncio.wait_for(
                        self._stop_evt.wait(), timeout=self.interval_seconds,
                    )
                except asyncio.TimeoutError:
                    pass
        finally:
            logger.info("[drift] watchdog stopped")


__all__ = [
    "WINDOW_DAYS",
    "DRIFT_MULTIPLIER",
    "DRIFT_MIN_MEDIAN",
    "evaluate_tenant",
    "discover_active_pairs",
    "run_one_cycle",
    "DriftWatchdog",
]
