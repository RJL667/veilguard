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
    """Insert a new proposal.  Called by the dream-cycle hook."""
    tbl = LedgerStore.get().table("task_proposals")
    pid = _new_id()
    ts = _now()
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
