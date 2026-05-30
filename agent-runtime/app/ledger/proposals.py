"""task_proposals CRUD — dream-as-scheduler queue.

Per spec §3.7.6.  Most proposals never become Tasks; lifecycle is:
  pending → (approved → done) | deferred → ... | shelved | expired

Daily worker handles decay (0.9× per cycle), TTL expiry (7d), recurrence
escalation at 5+ surfaces.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from .store import LedgerStore, ns_filter, proposal_queue_filter


def _now() -> float:
    return time.time()


def _new_id() -> str:
    return f"prop-{uuid.uuid4().hex[:12]}"


def create_proposal(
    *,
    tenant_id: str,
    user_id: str,
    signal_type: str,
    signal_node_ids: list[int],
    impact_score: float,
    proposed_brief: str,
    proposed_assignee: str,
    proposed_deliverable_spec: Optional[str] = None,
    rationale: Optional[str] = None,
    objective_alignment: Optional[float] = None,
    constraint_violations: Optional[list[str]] = None,
    constitution_version: Optional[int] = None,
) -> str:
    """Insert a new proposal — or bump an existing duplicate's recurrence.

    [PROPOSAL_DEDUP_2026_05_28]  When DreamScanner re-encounters the same
    `(tenant_id, user_id, signal_type, sorted(signal_node_ids))` tuple
    while a prior proposal is still pending or deferred, we MUST NOT
    write a duplicate row.  Instead bump the existing row's
    `recurrence_count` and refresh `last_surfaced_ts`.  Closes the TODO
    at `dream_scanner.py:22`.

    The recurrence count drives:
      * `lifecycle.escalated_proposals()` — rows with recurrence ≥ 5 get
        visual emphasis in the sidebar emergency lane.
      * recalibration outcome attribution — repeated proposals on the
        same node cluster signal "still relevant; the user hasn't acted".

    Returns the proposal id (NEW or EXISTING — caller treats both the
    same).
    """
    tbl = LedgerStore.get().table("task_proposals")
    ts = _now()

    # --- Dedup check: same (tenant, user, signal_type, node_ids) ---
    # node_ids order doesn't matter — store/match on sorted tuple.
    try:
        sorted_ids = sorted(int(n) for n in (signal_node_ids or []))
    except (TypeError, ValueError):
        sorted_ids = []
    if sorted_ids:
        # Lance .where() can't compare list-of-ints inline, so we scan
        # the small "still surfaceable" subset and dedup in Python.
        # That subset is bounded (~10-50 rows per user) so the cost is
        # negligible compared to writing a duplicate.
        try:
            where = (
                f"tenant_id = '{tenant_id}' "
                f"AND user_id = '{user_id}' "
                f"AND signal_type = '{signal_type}' "
                f"AND status IN ('pending', 'deferred')"
            )
            arr = tbl.search().where(where).limit(200).to_arrow()
            for i in range(arr.num_rows):
                row_ids = arr.column("signal_node_ids")[i].as_py() or []
                if sorted(int(n) for n in row_ids) != sorted_ids:
                    continue
                # Match — bump recurrence + last_surfaced_ts + decay_score
                # back up to current impact_score so the sidebar re-
                # surfaces.  (Decay accumulates between sightings; a
                # fresh emission means the signal is still warm.)
                existing_id = arr.column("id")[i].as_py()
                existing_rc = arr.column("recurrence_count")[i].as_py() or 1
                new_rc = int(existing_rc) + 1
                tbl.update(
                    where=f"id = '{existing_id}'",
                    values={
                        "recurrence_count":  new_rc,
                        "last_surfaced_ts":  ts,
                        "updated_ts":        ts,
                        "decay_score":       max(
                            float(arr.column("decay_score")[i].as_py() or 0.0),
                            float(impact_score),
                        ),
                    },
                )
                # Broadcast a status-change-style event so the sidebar
                # bumps the row without a full re-fetch.
                try:
                    from ..events import broadcast
                    broadcast({
                        "type":             "proposal_recurrence_bumped",
                        "tenant_id":        tenant_id,
                        "user_id":          user_id,
                        "id":               existing_id,
                        "recurrence_count": new_rc,
                        "signal_type":      signal_type,
                    })
                except Exception:
                    pass
                return existing_id
        except Exception:
            # On any scan error, fall through to the original create
            # path — better to occasionally double-emit than to drop a
            # proposal because the dedup query glitched.
            pass

    pid = _new_id()
    row = {
        "id": pid,
        "kind": "proposal",
        "status": "pending",
        "parent_id": None,
        "lineage_chain": [],
        "tenant_id": tenant_id,
        "user_id": user_id,
        "created_by_agent_id": "dream-cycle",
        "created_ts": ts,
        "updated_ts": ts,
        "cost_attributed_usd": 0.0,
        "signal_type": signal_type,
        "signal_node_ids": signal_node_ids,
        "impact_score": impact_score,
        "decay_score": impact_score,  # starts == impact; decays from here
        "objective_alignment": objective_alignment,
        "constraint_violations": constraint_violations or [],
        "proposed_brief": proposed_brief,
        "proposed_assignee": proposed_assignee,
        "proposed_deliverable_spec": proposed_deliverable_spec,
        "rationale": rationale,
        "recurrence_count": 1,
        "first_surfaced_ts": ts,
        "last_surfaced_ts": ts,
        "director_decision_ts": None,
        "shelf_reason": None,
        "resulting_task_id": None,
        "constitution_version": constitution_version,
        "extras_json": None,
    }
    tbl.add([row])
    try:
        from ..events import broadcast
        broadcast({
            "type": "proposal_created",
            "tenant_id": tenant_id, "user_id": user_id,
            "id": pid, "signal_type": signal_type,
            "impact_score": impact_score, "assignee": proposed_assignee,
            "brief": proposed_brief[:200],
        })
    except Exception:
        pass
    try:
        from ..runtime_health import apr_record_artifact
        apr_record_artifact()  # Phase 6.7
    except Exception:
        pass
    return pid


def get_proposal(
    proposal_id: str,
    tenant_id: str,
    user_id: str,
) -> dict[str, Any] | None:
    tbl = LedgerStore.get().table("task_proposals")
    where = f"{ns_filter(tenant_id, user_id)} AND id = '{proposal_id}'"
    arr = tbl.search().where(where).limit(1).to_arrow()
    if arr.num_rows == 0:
        return None
    return {col: arr.column(col)[0].as_py() for col in arr.column_names}


def approve_proposal(
    *,
    proposal_id: str,
    tenant_id: str,
    user_id: str,
    resulting_task_id: str,
) -> None:
    """User approved → Task created → link back + status=approved."""
    tbl = LedgerStore.get().table("task_proposals")
    where = f"{ns_filter(tenant_id, user_id)} AND id = '{proposal_id}'"
    tbl.update(where=where, values={
        "status": "approved",
        "resulting_task_id": resulting_task_id,
        "director_decision_ts": _now(),
        "updated_ts": _now(),
    })
    try:
        from ..events import broadcast
        broadcast({
            "type": "proposal_status_changed",
            "tenant_id": tenant_id, "user_id": user_id,
            "id": proposal_id, "status": "approved",
            "resulting_task_id": resulting_task_id,
        })
    except Exception:
        pass


def shelve_proposal(
    *,
    proposal_id: str,
    tenant_id: str,
    user_id: str,
    reason: str,
) -> None:
    """User shelved → status=shelved, reason recorded."""
    tbl = LedgerStore.get().table("task_proposals")
    where = f"{ns_filter(tenant_id, user_id)} AND id = '{proposal_id}'"
    tbl.update(where=where, values={
        "status": "shelved",
        "shelf_reason": reason,
        "director_decision_ts": _now(),
        "updated_ts": _now(),
    })
    try:
        from ..events import broadcast
        broadcast({
            "type": "proposal_status_changed",
            "tenant_id": tenant_id, "user_id": user_id,
            "id": proposal_id, "status": "shelved",
        })
    except Exception:
        pass


def defer_proposal(
    *,
    proposal_id: str,
    tenant_id: str,
    user_id: str,
) -> None:
    """User deferred → status=deferred (eligible for re-surface next cycle)."""
    tbl = LedgerStore.get().table("task_proposals")
    where = f"{ns_filter(tenant_id, user_id)} AND id = '{proposal_id}'"
    tbl.update(where=where, values={
        "status": "deferred",
        "director_decision_ts": _now(),
        "updated_ts": _now(),
    })
    try:
        from ..events import broadcast
        broadcast({
            "type": "proposal_status_changed",
            "tenant_id": tenant_id, "user_id": user_id,
            "id": proposal_id, "status": "deferred",
        })
    except Exception:
        pass


def queue(
    *,
    tenant_id: str,
    user_id: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return pending + deferred proposals sorted by decay_score desc."""
    tbl = LedgerStore.get().table("task_proposals")
    where = proposal_queue_filter(tenant_id, user_id)
    arr = tbl.search().where(where).limit(limit).to_arrow()
    rows = [
        {col: arr.column(col)[i].as_py() for col in arr.column_names}
        for i in range(arr.num_rows)
    ]
    rows.sort(key=lambda r: r.get("decay_score", 0.0), reverse=True)
    return rows


def apply_decay(
    *,
    tenant_id: str,
    user_id: str,
    decay_factor: float = 0.9,
) -> int:
    """Per-cycle decay (spec §3.7.4): deferred proposals decay 0.9×.

    Returns number of proposals decayed.
    """
    tbl = LedgerStore.get().table("task_proposals")
    where = (
        f"{ns_filter(tenant_id, user_id)} "
        f"AND status = 'deferred'"
    )
    arr = tbl.search().where(where).to_arrow()
    n = 0
    for i in range(arr.num_rows):
        pid = arr.column("id")[i].as_py()
        current = arr.column("decay_score")[i].as_py() or 0.0
        new = current * decay_factor
        tbl.update(
            where=f"id = '{pid}'",
            values={"decay_score": new, "updated_ts": _now()},
        )
        n += 1
    return n


def expire_stale(
    *,
    tenant_id: str,
    user_id: str,
    ttl_seconds: float = 7 * 24 * 3600.0,
) -> int:
    """Auto-shelf proposals older than ttl_seconds (spec §3.7.4 TTL)."""
    tbl = LedgerStore.get().table("task_proposals")
    cutoff = _now() - ttl_seconds
    where = (
        f"{ns_filter(tenant_id, user_id)} "
        f"AND status IN ('pending', 'deferred') "
        f"AND first_surfaced_ts < {cutoff}"
    )
    arr = tbl.search().where(where).to_arrow()
    n = 0
    for i in range(arr.num_rows):
        pid = arr.column("id")[i].as_py()
        tbl.update(
            where=f"id = '{pid}'",
            values={
                "status": "expired",
                "shelf_reason": "expired_ttl",
                "updated_ts": _now(),
            },
        )
        n += 1
    return n


__all__ = [
    "create_proposal",
    "get_proposal",
    "approve_proposal",
    "shelve_proposal",
    "defer_proposal",
    "queue",
    "apply_decay",
    "expire_stale",
]
