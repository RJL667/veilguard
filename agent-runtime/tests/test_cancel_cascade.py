"""§11.2.4 — subtask cancellation cascade (edge-case catalog MUST-HAVE #5).

Cancelling a parent must cascade to the WHOLE subtree (every task whose
`lineage_chain` contains the parent id) and leave unrelated tasks alone.
`in_progress` / `accepted` descendants get a "checkpoint and stop" comment
before they're cancelled (the graceful-stop signal per spec §11.2.4), and
already-terminal descendants are left untouched.

Exercises `app.ledger.tasks.cancel_cascade`, which had NO direct test
before this file (audited 2026-06-01 during the UAT loop — the only
prior "cascade" coverage was cross-tenant dedup and orphan-LEASE reclaim,
neither of which touches the parent-cancel cascade path).

Uses a real temp Lance store (the cascade does a BFS over lineage_chain
plus a privileged direct `tbl.update`, so mocking the table chain would
be brittle and low-value).
"""

from __future__ import annotations

import time

import pytest

TEN, USR = "t-cascade", "u-cascade"


@pytest.fixture
def store(tmp_path):
    """Fresh, isolated LedgerStore pointed at a per-test temp Lance dir."""
    from app.ledger.store import LedgerStore
    LedgerStore._instance = None
    inst = LedgerStore.get(db_path=str(tmp_path / "ledger.db"))
    inst.table("agent_tasks")  # force table creation
    yield inst
    LedgerStore._instance = None


def _mk(owner: str, brief: str, parent: str | None = None) -> str:
    """Create a task via the real API (AC contract bypassed — this suite
    is about cascade, not acceptance criteria)."""
    from app.ledger import tasks as T
    return T.create_task(
        tenant_id=TEN, user_id=USR, owner_id=owner,
        brief=brief, deliverable_spec="n/a",
        parent_id=parent, _phase_6_legacy_exempt=True,
    )


def _status(store, task_id: str) -> str:
    from app.ledger.store import ns_filter
    arr = (
        store.table("agent_tasks").search()
        .where(f"{ns_filter(TEN, USR)} AND id = '{task_id}'")
        .to_arrow()
    )
    assert arr.num_rows == 1, f"task {task_id} not found"
    return arr.column("status")[0].as_py()


def _set_status(store, task_id: str, status: str) -> None:
    """Test-setup shortcut: force a status directly (mirrors how
    cancel_cascade itself writes — privileged, skips the state machine)."""
    store.table("agent_tasks").update(
        where=f"id = '{task_id}'",
        values={"status": status, "updated_ts": time.time()},
    )


def test_cancel_cascade_cancels_subtree_not_siblings(store):
    from app.ledger import tasks as T

    parent = _mk("director", "coordinator")
    child_a = _mk("researcher", "child A", parent=parent)
    child_b = _mk("builder", "child B", parent=parent)
    grandchild = _mk("researcher", "grandchild", parent=child_b)
    unrelated = _mk("researcher", "unrelated top-level")  # no parent

    # One descendant mid-flight to exercise the checkpoint-comment branch.
    _set_status(store, child_b, "in_progress")

    n = T.cancel_cascade(
        task_id=parent, tenant_id=TEN, user_id=USR,
        actor_agent_id="director", reason="parent_cancelled",
    )

    assert n == 4  # parent + 3 descendants
    for tid in (parent, child_a, child_b, grandchild):
        assert _status(store, tid) == "cancelled", f"{tid} should be cancelled"
    # The unrelated top-level task is NOT in the parent's lineage → untouched.
    assert _status(store, unrelated) == "open"


def test_cancel_cascade_skips_already_terminal_descendants(store):
    from app.ledger import tasks as T

    parent = _mk("director", "coordinator")
    done_child = _mk("researcher", "already done", parent=parent)
    open_child = _mk("builder", "still open", parent=parent)
    _set_status(store, done_child, "done")

    n = T.cancel_cascade(
        task_id=parent, tenant_id=TEN, user_id=USR, actor_agent_id="director",
    )

    # parent + open_child only; the done child is excluded by the
    # `status NOT IN ('done','cancelled')` filter and stays done.
    assert n == 2
    assert _status(store, done_child) == "done"
    assert _status(store, open_child) == "cancelled"
    assert _status(store, parent) == "cancelled"


def test_cancel_cascade_checkpoint_comment_for_in_progress(store):
    from app.ledger import tasks as T
    from app.ledger import comments

    parent = _mk("director", "coordinator")
    busy = _mk("builder", "mid-flight", parent=parent)
    _set_status(store, busy, "in_progress")

    T.cancel_cascade(
        task_id=parent, tenant_id=TEN, user_id=USR,
        actor_agent_id="director", reason="parent_cancelled",
    )

    bodies = [
        c.get("body", "")
        for c in comments.list_comments(task_id=busy, tenant_id=TEN, user_id=USR)
    ]
    assert any("checkpoint and stop" in b for b in bodies), (
        "an in_progress descendant must receive a graceful-stop comment "
        "before being cancelled (spec §11.2.4)"
    )


def test_cancel_cascade_is_tenant_scoped(store):
    """A cascade in one tenant must never touch an identically-shaped tree
    in another tenant (cross-tenant isolation, §11.7.1)."""
    from app.ledger import tasks as T
    from app.ledger.store import ns_filter

    parent = _mk("director", "coordinator")
    _mk("researcher", "child", parent=parent)

    # Same-shaped tree in a DIFFERENT tenant.
    other_parent = T.create_task(
        tenant_id="other-tenant", user_id="other-user", owner_id="director",
        brief="other coordinator", deliverable_spec="n/a",
        _phase_6_legacy_exempt=True,
    )

    T.cancel_cascade(
        task_id=parent, tenant_id=TEN, user_id=USR, actor_agent_id="director",
    )

    arr = (
        store.table("agent_tasks").search()
        .where(f"{ns_filter('other-tenant', 'other-user')} AND id = '{other_parent}'")
        .to_arrow()
    )
    assert arr.column("status")[0].as_py() == "open", (
        "cascade leaked across the tenant boundary"
    )
