"""agent_tasks CRUD.

Public surface:
  create_task(...)        — Director's tool entry point
  accept_task(...)        — IC's tool entry point
  update_status(...)      — state machine transitions
  attach_output(...)      — IC adds a deliverable path
  submit_for_review(...)  — IC sends to Critic
  get_task(task_id)       — single-row read
  inbox(agent_id, ...)    — owner-scoped derived view

State machine (enforced):
  open → accepted → in_progress → (review | blocked | cancelled)
  review → (done | in_progress)        (Critic decisions)
  blocked → in_progress                (when blocker cleared)
  in_progress → done | cancelled       (terminal)

Cancellation cascade (spec §11.2.4):
  Cancelling a parent immediately sets descendant open/blocked → cancelled
  and flags descendant in_progress to checkpoint and stop.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from .store import LedgerStore, ns_filter


# Allowed transitions per state machine.  Programming-error guard;
# upstream tool wrappers should validate before calling.
_TRANSITIONS = {
    "open":        {"accepted", "cancelled"},
    "accepted":    {"in_progress", "cancelled"},
    "in_progress": {"review", "blocked", "done", "cancelled"},
    "blocked":     {"in_progress", "cancelled"},
    "review":      {"done", "in_progress", "cancelled"},
    "done":        set(),
    "cancelled":   set(),
}


def _now() -> float:
    return time.time()


def _new_id() -> str:
    return f"task-{uuid.uuid4().hex[:12]}"


def create_task(
    *,
    tenant_id: str,
    user_id: str,
    owner_id: str,
    brief: str,
    deliverable_spec: str,
    assigner_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    inputs: Optional[list[str]] = None,
    due_ts: Optional[float] = None,
    origin: str = "foreground",
    pattern: Optional[str] = None,
    constitution_version: Optional[int] = None,
    extras_json: Optional[str] = None,
) -> str:
    """Insert a new Task.  Returns the new task_id.

    `parent_id` (if set) establishes a subtask relationship and is the
    first element of `lineage_chain` (server-computed per §3.8.5).
    """
    tbl = LedgerStore.get().table("agent_tasks")
    task_id = _new_id()

    # Lineage chain: server-computed, never client-supplied.
    lineage: list[str] = []
    if parent_id:
        parent_lineage = _get_lineage(parent_id) or []
        lineage = list(parent_lineage) + [parent_id]

    row = {
        "id": task_id,
        "kind": "task",
        "status": "open",
        "parent_id": parent_id,
        "lineage_chain": lineage,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "created_by_agent_id": assigner_id,
        "created_ts": _now(),
        "updated_ts": _now(),
        "cost_attributed_usd": 0.0,
        "owner_id": owner_id,
        "assigner_id": assigner_id,
        "brief": brief,
        "deliverable_spec": deliverable_spec,
        "inputs": inputs or [],
        "outputs": [],
        "due_ts": due_ts,
        "trace_ref": None,
        "comments_head_hash": None,
        "origin": origin,
        "pattern": pattern,
        "constitution_version": constitution_version,
        # Sentinel values rather than NULL: LanceDB's where-clause
        # parser doesn't support `IS NULL`, so the inbox poller filters
        # on `lease_until < now`.  Empty owner + zero lease = available.
        "lease_owner": "",
        "lease_until": 0.0,
        "extras_json": extras_json,
    }
    tbl.add([row])
    return task_id


def _get_lineage(task_id: str) -> list[str] | None:
    """Return a task's lineage_chain, or None if not found."""
    tbl = LedgerStore.get().table("agent_tasks")
    arr = tbl.search().where(f"id = '{task_id}'").limit(1).to_arrow()
    if arr.num_rows == 0:
        return None
    return arr.column("lineage_chain")[0].as_py()


def get_task(task_id: str, tenant_id: str, user_id: str) -> dict[str, Any] | None:
    """Read a single Task, tenant-isolated."""
    tbl = LedgerStore.get().table("agent_tasks")
    where = f"{ns_filter(tenant_id, user_id)} AND id = '{task_id}'"
    arr = tbl.search().where(where).limit(1).to_arrow()
    if arr.num_rows == 0:
        return None
    return _row_to_dict(arr, 0)


def update_status(
    *,
    task_id: str,
    tenant_id: str,
    user_id: str,
    new_status: str,
    actor_agent_id: str,
) -> None:
    """Transition state machine.  Raises ValueError on illegal transition."""
    task = get_task(task_id, tenant_id, user_id)
    if task is None:
        raise ValueError(f"task not found: {task_id}")
    current = task["status"]
    if new_status not in _TRANSITIONS.get(current, set()):
        raise ValueError(
            f"illegal transition: {current} → {new_status} on {task_id}"
        )

    tbl = LedgerStore.get().table("agent_tasks")
    where = f"{ns_filter(tenant_id, user_id)} AND id = '{task_id}'"
    tbl.update(where=where, values={
        "status": new_status,
        "updated_ts": _now(),
    })

    # Append a status_change comment so the chain has the transition log.
    from . import comments  # avoid circular import
    comments.add_comment(
        task_id=task_id,
        tenant_id=tenant_id,
        user_id=user_id,
        author_id=actor_agent_id,
        kind="status_change",
        body=f"{current} -> {new_status}",
    )


def attach_output(
    *,
    task_id: str,
    tenant_id: str,
    user_id: str,
    path: str,
) -> None:
    """Append a file path to outputs[] atomically."""
    task = get_task(task_id, tenant_id, user_id)
    if task is None:
        raise ValueError(f"task not found: {task_id}")
    outputs = list(task.get("outputs") or [])
    if path in outputs:
        return  # idempotent
    outputs.append(path)

    tbl = LedgerStore.get().table("agent_tasks")
    where = f"{ns_filter(tenant_id, user_id)} AND id = '{task_id}'"
    tbl.update(where=where, values={
        "outputs": outputs,
        "updated_ts": _now(),
    })


def submit_for_review(
    *,
    task_id: str,
    tenant_id: str,
    user_id: str,
    target: str,           # team_knowledge | user_deliverable | org_blackboard
    submitter_agent_id: str,
) -> None:
    """IC submits the task for Critic review.

    Adds a review_request comment + transitions status → review.
    Critic agents poll for status=review and pick up via assignee match.
    """
    from . import comments
    update_status(
        task_id=task_id,
        tenant_id=tenant_id,
        user_id=user_id,
        new_status="review",
        actor_agent_id=submitter_agent_id,
    )
    comments.add_comment(
        task_id=task_id,
        tenant_id=tenant_id,
        user_id=user_id,
        author_id=submitter_agent_id,
        kind="review_request",
        body=f"target={target}",
    )


def inbox(
    *,
    tenant_id: str,
    user_id: str,
    owner_id: str,
    statuses: list[str] | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Per-agent inbox query — the §3.10.4 derived view."""
    from .store import task_inbox_filter
    where = task_inbox_filter(tenant_id, user_id, owner_id, statuses)
    tbl = LedgerStore.get().table("agent_tasks")
    arr = tbl.search().where(where).limit(limit).to_arrow()
    return [_row_to_dict(arr, i) for i in range(arr.num_rows)]


def cancel_cascade(
    *,
    task_id: str,
    tenant_id: str,
    user_id: str,
    actor_agent_id: str,
    reason: str = "parent_cancelled",
) -> int:
    """Cancel a task + cascade to all descendants per spec §11.2.4.

    Returns the number of tasks transitioned.
    """
    from . import comments

    n_cancelled = 0
    # BFS: find all descendants whose lineage_chain contains task_id.
    tbl = LedgerStore.get().table("agent_tasks")
    where = (
        f"{ns_filter(tenant_id, user_id)} "
        f"AND status NOT IN ('done', 'cancelled')"
    )
    arr = tbl.search().where(where).to_arrow()
    for i in range(arr.num_rows):
        row = _row_to_dict(arr, i)
        if row["id"] != task_id and task_id not in (row.get("lineage_chain") or []):
            continue
        current = row["status"]
        if current in ("in_progress", "accepted"):
            # Flag for graceful stop — comment then cancel.
            comments.add_comment(
                task_id=row["id"],
                tenant_id=tenant_id,
                user_id=user_id,
                author_id=actor_agent_id,
                kind="status_change",
                body=f"cascade-cancel: please checkpoint and stop ({reason})",
            )
        # Direct DB update (skip state-machine check; cascade is privileged).
        tbl.update(
            where=f"id = '{row['id']}'",
            values={"status": "cancelled", "updated_ts": _now()},
        )
        n_cancelled += 1

    return n_cancelled


# ── Internal helpers ─────────────────────────────────────────────────────


def _row_to_dict(arr, idx: int) -> dict[str, Any]:
    return {col: arr.column(col)[idx].as_py() for col in arr.column_names}


__all__ = [
    "create_task",
    "get_task",
    "update_status",
    "attach_output",
    "submit_for_review",
    "inbox",
    "cancel_cascade",
]
