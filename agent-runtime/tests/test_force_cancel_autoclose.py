"""F11 — force-cancel (turn-cap / timeout) must propagate parent autoclose.

Both `_force_cancel_on_turn_cap` and `_force_cancel_on_timeout` raw-write
`cancelled` to Lance, bypassing `update_status`'s `[PARENT_AUTOCLOSE]` hook.
Before the 2026-06-02 fix, force-cancelling the LAST open child of a
coordinator left the coordinator stuck `open` forever (orphan). Caught live
in a Pattern-C fanout: the Builder hit the 25-turn cap and was force-cancelled,
the Researcher had completed, yet the coordinator never closed.
"""

from __future__ import annotations

import time

import pytest

TEN, USR = "t-f11", "u-f11"


@pytest.fixture
def store(tmp_path):
    from app.ledger.store import LedgerStore
    LedgerStore._instance = None
    inst = LedgerStore.get(db_path=str(tmp_path / "ledger.db"))
    inst.table("agent_tasks")
    yield inst
    LedgerStore._instance = None


def _mk(owner: str, brief: str, parent: str | None = None) -> str:
    from app.ledger import tasks as T
    return T.create_task(
        tenant_id=TEN, user_id=USR, owner_id=owner, brief=brief,
        deliverable_spec="n/a", parent_id=parent, _phase_6_legacy_exempt=True,
    )


def _status(store, tid: str) -> str:
    from app.ledger.store import ns_filter
    arr = store.table("agent_tasks").search().where(
        f"{ns_filter(TEN, USR)} AND id = '{tid}'"
    ).to_arrow()
    return arr.column("status")[0].as_py()


def _set_status(store, tid: str, st: str) -> None:
    store.table("agent_tasks").update(
        where=f"id = '{tid}'", values={"status": st, "updated_ts": time.time()},
    )


def _row(store, tid: str) -> dict:
    arr = store.table("agent_tasks").search().where(f"id = '{tid}'").to_arrow()
    return {c: arr.column(c)[0].as_py() for c in arr.column_names}


def _poller():
    from app.workers.inbox_poller import InboxPoller
    from app.personas.loader import PersonaRegistry
    return InboxPoller(PersonaRegistry({}))


def test_turn_cap_force_cancel_autocloses_parent(store):
    coord = _mk("team-lead", "coordinator")
    child = _mk("builder", "child", parent=coord)
    _set_status(store, child, "in_progress")

    _poller()._force_cancel_on_turn_cap(_row(store, child), 25)

    assert _status(store, child) == "cancelled"
    # F11: the parent must now be terminal, NOT stuck `open`.
    assert _status(store, coord) != "open"


def test_timeout_force_cancel_autocloses_parent(store):
    coord = _mk("team-lead", "coordinator")
    child = _mk("researcher", "child", parent=coord)
    _set_status(store, child, "in_progress")

    _poller()._force_cancel_on_timeout(_row(store, child))

    assert _status(store, child) == "cancelled"
    assert _status(store, coord) != "open"


def test_force_cancel_top_level_task_is_safe(store):
    """A force-cancel of a parentless task must not error in the autoclose
    propagation (no parent to close)."""
    t = _mk("builder", "top-level")
    _set_status(store, t, "in_progress")

    _poller()._force_cancel_on_turn_cap(_row(store, t), 25)

    assert _status(store, t) == "cancelled"
