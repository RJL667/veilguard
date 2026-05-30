"""Phase 3 — proposal lifecycle worker.

Per spec §3.7.4: deferred proposals decay 0.9× per cycle until
re-triggered or below floor (auto-shelve as ``aged_unaddressed``).
TTL is 7d (auto-shelve as ``expired_ttl``).  Recurrence escalation
at 5+ surfaces → spec says present as forced choice; this module
exposes ``escalated_proposals()`` to surface them for the UI.

The worker is dependency-tolerant: if Lance isn't available we no-op
and log instead of crashing the host process — same posture as the
inbox poller's startup sweep.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..ledger import proposals as _props
from ..ledger.store import LedgerStore, proposal_queue_filter

logger = logging.getLogger(__name__)


# Per-cycle decay factor (§3.7.4).  0.9× per cycle until a candidate
# either re-triggers (impact recomputed fresh) or drops below the floor.
DECAY_FACTOR = 0.9

# Decay floor below which a deferred proposal is auto-shelved with
# reason ``aged_unaddressed``.  Tuneable; 0.05 is "below 5% of the
# original impact" — i.e. ~28 cycles of decay if never re-triggered.
DECAY_FLOOR = 0.05

# TTL — proposals older than this auto-expire regardless of decay.
TTL_SECONDS = 7 * 24 * 3600.0

# Recurrence escalation threshold (§3.7.4 — "5+ times without approval").
RECURRENCE_ESCALATION_THRESHOLD = 5


def auto_shelf_below_floor(*, tenant_id: str, user_id: str) -> int:
    """Auto-shelf deferred proposals whose decay_score < DECAY_FLOOR.

    Pairs with the existing `proposals.apply_decay()` which DOES the
    decay; this worker harvests proposals that have decayed past the
    point of being worth re-surfacing.  Stamps shelf_reason
    `aged_unaddressed` per spec §3.7.4.

    Returns number of proposals shelved.
    """
    try:
        tbl = LedgerStore.get().table("task_proposals")
    except Exception as e:
        logger.warning(f"[proposals.lifecycle] auto_shelf: ledger unavailable: {e}")
        return 0
    where = (
        f"tenant_id = '{tenant_id}' AND user_id = '{user_id}' "
        f"AND status = 'deferred' "
        f"AND decay_score < {DECAY_FLOOR}"
    )
    try:
        arr = tbl.search().where(where).to_arrow()
    except Exception as e:
        logger.warning(f"[proposals.lifecycle] auto_shelf scan failed: {e}")
        return 0
    n = 0
    for i in range(arr.num_rows):
        pid = arr.column("id")[i].as_py()
        try:
            tbl.update(where=f"id = '{pid}'", values={
                "status": "shelved",
                "shelf_reason": "aged_unaddressed",
                "updated_ts": time.time(),
            })
            n += 1
        except Exception as e:
            logger.exception(
                f"[proposals.lifecycle] failed to shelf {pid}: {e}"
            )
    if n:
        logger.info(
            f"[proposals.lifecycle] auto-shelved {n} aged_unaddressed "
            f"proposal(s) for {tenant_id}/{user_id}"
        )
    return n


def escalated_proposals(
    *, tenant_id: str, user_id: str, limit: int = 50,
) -> list[dict[str, Any]]:
    """Return proposals whose `recurrence_count` ≥ threshold and which
    are still pending/deferred.  Phase 3 sidebar surfaces these with
    visual emphasis per §3.7.4: "approve, shelve-with-reason, or
    recalibrate scoring".  This is a read; no mutation.
    """
    try:
        tbl = LedgerStore.get().table("task_proposals")
    except Exception as e:
        logger.warning(f"[proposals.lifecycle] escalated: ledger unavailable: {e}")
        return []
    where = (
        f"tenant_id = '{tenant_id}' AND user_id = '{user_id}' "
        f"AND status IN ('pending', 'deferred') "
        f"AND recurrence_count >= {RECURRENCE_ESCALATION_THRESHOLD}"
    )
    try:
        arr = tbl.search().where(where).limit(limit).to_arrow()
    except Exception as e:
        logger.warning(f"[proposals.lifecycle] escalated scan failed: {e}")
        return []
    rows = [
        {col: arr.column(col)[i].as_py() for col in arr.column_names}
        for i in range(arr.num_rows)
    ]
    rows.sort(key=lambda r: r.get("recurrence_count", 0), reverse=True)
    return rows


def run_one_cycle(*, tenant_id: str, user_id: str) -> dict[str, int]:
    """Run a single lifecycle pass: decay → auto-shelf → expire.

    Per §3.7.4 the spec calls for this on each dream cycle; until the
    dream-cycle hook is wired we expose it as a stand-alone function
    a scheduler / cron / admin endpoint can drive.

    Returns:
        {
            "decayed":   number of deferred proposals whose decay_score updated,
            "shelved":   number auto-shelved as aged_unaddressed,
            "expired":   number expired_ttl past the 7d window,
            "escalated": number currently above the recurrence threshold,
        }
    """
    decayed  = _props.apply_decay(tenant_id=tenant_id, user_id=user_id, decay_factor=DECAY_FACTOR)
    shelved  = auto_shelf_below_floor(tenant_id=tenant_id, user_id=user_id)
    expired  = _props.expire_stale(tenant_id=tenant_id, user_id=user_id, ttl_seconds=TTL_SECONDS)
    escalated = len(escalated_proposals(tenant_id=tenant_id, user_id=user_id))
    out = {
        "decayed":   decayed,
        "shelved":   shelved,
        "expired":   expired,
        "escalated": escalated,
    }
    logger.info(f"[proposals.lifecycle] cycle for {tenant_id}/{user_id}: {out}")
    return out


# ── async background worker ─────────────────────────────────────────────


class LifecycleWorker:
    """Periodic per-tenant lifecycle sweep.

    Intended cadence: once per dream cycle.  Today's wiring runs on a
    simple sleep loop (default 1h between sweeps) because the dream
    scheduler doesn't exist yet — when it ships, the worker can
    subscribe to dream-cycle-end events instead.

    Tenant discovery: walks `task_proposals` for any (tenant_id,
    user_id) pair that has at least one non-terminal row.  No external
    registry needed.
    """
    def __init__(self, interval_seconds: float = 3600.0):
        self.interval_seconds = float(interval_seconds)
        self._stop_evt = asyncio.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    def discover_active_pairs(self) -> list[tuple[str, str]]:
        """Find (tenant_id, user_id) tuples with at least one
        pending/deferred proposal.  Returns deduped, sorted list."""
        try:
            tbl = LedgerStore.get().table("task_proposals")
            arr = (
                tbl.search()
                .where("status IN ('pending', 'deferred')")
                .select(["tenant_id", "user_id"])
                .limit(5000)
                .to_arrow()
            )
        except Exception as e:
            logger.warning(f"[proposals.lifecycle] discover pairs failed: {e}")
            return []
        pairs = set()
        for i in range(arr.num_rows):
            t = arr.column("tenant_id")[i].as_py() or ""
            u = arr.column("user_id")[i].as_py() or ""
            if t and u:
                pairs.add((t, u))
        return sorted(pairs)

    async def run(self) -> None:
        logger.info(
            f"[proposals.lifecycle] worker starting; "
            f"interval={self.interval_seconds:.0f}s"
        )
        try:
            while not self._stop_evt.is_set():
                try:
                    pairs = self.discover_active_pairs()
                    for tenant_id, user_id in pairs:
                        try:
                            run_one_cycle(tenant_id=tenant_id, user_id=user_id)
                        except Exception as e:
                            logger.exception(
                                f"[proposals.lifecycle] cycle failed for "
                                f"{tenant_id}/{user_id}: {e}"
                            )
                except Exception as e:
                    logger.exception(f"[proposals.lifecycle] sweep error: {e}")
                try:
                    await asyncio.wait_for(
                        self._stop_evt.wait(),
                        timeout=self.interval_seconds,
                    )
                except asyncio.TimeoutError:
                    pass  # normal — keep sweeping
        finally:
            logger.info("[proposals.lifecycle] worker stopped")


__all__ = [
    "DECAY_FACTOR",
    "DECAY_FLOOR",
    "TTL_SECONDS",
    "RECURRENCE_ESCALATION_THRESHOLD",
    "auto_shelf_below_floor",
    "escalated_proposals",
    "run_one_cycle",
    "LifecycleWorker",
]
