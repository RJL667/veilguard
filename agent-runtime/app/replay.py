"""Phase 4 — Replay v1 (§Phase 4).

Spec: "Replay v1 — counterfactual path reconstruction using
bitemporal + lineage_chain.  Shows alternate decision paths, not
byte-exact LLM outputs."

Replay v1 surfaces the decision-tree around a task:
  * the task itself
  * its lineage_chain ancestors (Director → Task → subtasks)
  * its peer subtasks (siblings of any ancestor)
  * the comment chain (review_decision pivots — these are the
    branching decision points)
  * the task_proposals row that spawned it (if any)
  * any proposal_outcomes row (regret + value_realized)
  * alternative proposals from the SAME signal_node_ids cluster that
    weren't approved — these ARE the counterfactual paths

We render a compact JSON the UI / agent can walk.  Pure-read; no
mutations.  No LLM calls — pure structural walk over the existing
ledger tables.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _lookup_task(tbl: Any, task_id: str, tenant_id: str, user_id: str) -> Optional[dict[str, Any]]:
    try:
        arr = (
            tbl.search()
            .where(
                f"tenant_id = '{tenant_id}' AND user_id = '{user_id}' "
                f"AND id = '{task_id}'"
            )
            .limit(1)
            .to_arrow()
        )
        if arr.num_rows == 0:
            return None
        return {c: arr.column(c)[0].as_py() for c in arr.column_names}
    except Exception:
        return None


def _lookup_proposal(
    tbl: Any, *, tenant_id: str, user_id: str, task_id: str,
) -> Optional[dict[str, Any]]:
    """Find the proposal whose resulting_task_id == task_id."""
    try:
        arr = (
            tbl.search()
            .where(
                f"tenant_id = '{tenant_id}' AND user_id = '{user_id}' "
                f"AND resulting_task_id = '{task_id}'"
            )
            .limit(1)
            .to_arrow()
        )
        if arr.num_rows == 0:
            return None
        return {c: arr.column(c)[0].as_py() for c in arr.column_names}
    except Exception:
        return None


def _lookup_outcome(
    tbl: Any, *, proposal_id: str,
) -> Optional[dict[str, Any]]:
    try:
        arr = (
            tbl.search()
            .where(f"proposal_id = '{proposal_id}'")
            .limit(1)
            .to_arrow()
        )
        if arr.num_rows == 0:
            return None
        return {c: arr.column(c)[0].as_py() for c in arr.column_names}
    except Exception:
        return None


def _peer_proposals_same_signal(
    tbl: Any, *, tenant_id: str, user_id: str,
    signal_type: str, signal_node_ids: list[int], exclude_id: str,
) -> list[dict[str, Any]]:
    """Find proposals with the same signal_type that share at least
    one signal_node_id — these are alternative emissions from the
    same dream cluster.  Counterfactual paths.
    """
    if not signal_type or not signal_node_ids:
        return []
    try:
        arr = (
            tbl.search()
            .where(
                f"tenant_id = '{tenant_id}' AND user_id = '{user_id}' "
                f"AND signal_type = '{signal_type}'"
            )
            .limit(200)
            .to_arrow()
        )
    except Exception:
        return []
    needle = set(int(n) for n in signal_node_ids if isinstance(n, (int, float)))
    out: list[dict[str, Any]] = []
    for i in range(arr.num_rows):
        pid = arr.column("id")[i].as_py()
        if pid == exclude_id:
            continue
        peer_ids = arr.column("signal_node_ids")[i].as_py() or []
        peer_set = set(int(n) for n in peer_ids if isinstance(n, (int, float)))
        if needle & peer_set:
            out.append({c: arr.column(c)[i].as_py() for c in arr.column_names})
    return out


def replay_task(
    *,
    task_id: str,
    tenant_id: str,
    user_id: str,
) -> dict[str, Any]:
    """Return the full decision-path payload for a task.

    Shape:
      {
        "task":             {agent_tasks row},
        "ancestors":        [tasks up the lineage_chain],
        "siblings":         [tasks sharing parent_id with the task or any ancestor],
        "comments":         [task_comments, oldest first],
        "spawned_by":       {task_proposals row} | None,
        "outcome":          {proposal_outcomes row} | None,
        "counterfactuals":  [peer proposals from same signal cluster],
        "decision_points":  [review_decision comments — branch pivots]
      }
    """
    from .ledger.store import LedgerStore
    from .ledger import comments as _cm
    store = LedgerStore.get()

    tasks_tbl = store.table("agent_tasks")
    proposals_tbl = store.table("task_proposals")
    outcomes_tbl = store.table("proposal_outcomes")

    task = _lookup_task(tasks_tbl, task_id, tenant_id, user_id)
    if task is None:
        return {
            "task": None, "error": f"task {task_id} not found",
        }
    lineage = task.get("lineage_chain") or []
    ancestors = []
    for aid in lineage:
        a = _lookup_task(tasks_tbl, aid, tenant_id, user_id)
        if a:
            ancestors.append(a)

    # Siblings = tasks with the same parent_id (if task has a parent)
    siblings: list[dict[str, Any]] = []
    parent_id = task.get("parent_id")
    if parent_id:
        try:
            arr = (
                tasks_tbl.search()
                .where(
                    f"tenant_id = '{tenant_id}' AND user_id = '{user_id}' "
                    f"AND parent_id = '{parent_id}'"
                )
                .limit(50)
                .to_arrow()
            )
            for i in range(arr.num_rows):
                row = {c: arr.column(c)[i].as_py() for c in arr.column_names}
                if row.get("id") != task_id:
                    siblings.append(row)
        except Exception:
            pass

    # Full comment chain
    try:
        comments = _cm.list_comments(
            task_id=task_id, tenant_id=tenant_id, user_id=user_id,
        )
    except Exception:
        comments = []
    decision_points = [
        c for c in comments if c.get("kind") == "review_decision"
    ]

    # Proposal it came from (if any)
    spawned_by = _lookup_proposal(
        proposals_tbl, tenant_id=tenant_id, user_id=user_id, task_id=task_id,
    )
    outcome = None
    counterfactuals: list[dict[str, Any]] = []
    if spawned_by:
        outcome = _lookup_outcome(outcomes_tbl, proposal_id=spawned_by["id"])
        counterfactuals = _peer_proposals_same_signal(
            proposals_tbl,
            tenant_id=tenant_id, user_id=user_id,
            signal_type=spawned_by.get("signal_type") or "",
            signal_node_ids=spawned_by.get("signal_node_ids") or [],
            exclude_id=spawned_by["id"],
        )

    return {
        "task":             task,
        "ancestors":        ancestors,
        "siblings":         siblings,
        "comments":         comments,
        "decision_points":  decision_points,
        "spawned_by":       spawned_by,
        "outcome":          outcome,
        "counterfactuals":  counterfactuals,
        "summary": {
            "lineage_depth":     len(ancestors),
            "sibling_count":     len(siblings),
            "comment_count":     len(comments),
            "decision_count":    len(decision_points),
            "had_proposal":      spawned_by is not None,
            "had_outcome":       outcome is not None,
            "alt_paths":         len(counterfactuals),
        },
    }


__all__ = ["replay_task"]
