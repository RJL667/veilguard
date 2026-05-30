"""AC-P7.4 runtime probe — the operational hot path must stay COLD on TCMM.

The static check in `tests/memory/test_phase_7_acceptance.py` covers one
status-index query; this probe fires the hot endpoints under load and
asserts TCMM observe is never reached.  If it fires under simulated
traffic, future code drift (someone adding a recall to /proposals's
sort path, say) would be caught immediately.

The list of "hot paths" must stay synchronized with the AC-P7.4 spec
clause; add new operational GETs here as they ship.
"""

from __future__ import annotations

import asyncio
import unittest.mock as mock

import pytest


def _mock_lance_table_with_empty_arrow():
    """Build a Lance-table-shaped mock whose .search().where().limit().to_arrow()
    chain returns an empty pa.Table — fast, no I/O, no file locks."""
    import pyarrow as pa
    arr = pa.Table.from_pylist([{}]).slice(0, 0)
    q = mock.MagicMock()
    q.where.return_value = q
    q.limit.return_value = q
    q.select.return_value = q
    q.to_arrow.return_value = arr
    tbl = mock.MagicMock()
    tbl.search.return_value = q
    return tbl


@pytest.mark.asyncio
async def test_ac_p7_4_proposals_queue_cold_under_repeated_calls():
    """100 reads from the proposal status-index path must make zero TCMM
    observe calls.  Catches regression where someone wires recall into
    the sort/rank path of the operational hot endpoint.

    Patches LedgerStore.get() to a stubbed handle so we never hit a
    real Lance file — keeps the test fast and isolated from container
    file-lock contention.
    """
    call_counter = {"observe": 0}

    async def _spy(*a, **kw):
        call_counter["observe"] += 1
        return False

    fake_store = mock.MagicMock()
    fake_store.table.return_value = _mock_lance_table_with_empty_arrow()

    with mock.patch("app.middleware.tcmm.observe_agent_output", _spy), \
         mock.patch("app.ledger.proposals.LedgerStore.get", return_value=fake_store):
        from app.ledger import proposals as _p
        for _ in range(100):
            try:
                _p.queue(tenant_id="ac-p7-4-probe", user_id="ac-p7-4-probe", limit=20)
            except Exception:
                pass

    assert call_counter["observe"] == 0, (
        f"AC-P7.4 violation: 100 reads of proposals.queue made "
        f"{call_counter['observe']} TCMM observe calls; operational hot "
        f"path must stay cold on TCMM."
    )


@pytest.mark.asyncio
async def test_ac_p7_4_get_task_cold():
    """Task lookups (get_task) must also stay cold on TCMM."""
    call_counter = {"observe": 0}

    async def _spy(*a, **kw):
        call_counter["observe"] += 1
        return False

    fake_store = mock.MagicMock()
    fake_store.table.return_value = _mock_lance_table_with_empty_arrow()

    with mock.patch("app.middleware.tcmm.observe_agent_output", _spy), \
         mock.patch("app.ledger.tasks.LedgerStore.get", return_value=fake_store):
        from app.ledger import tasks as _t
        for _ in range(50):
            try:
                _t.get_task("task-nonexistent-probe", "probe", "probe")
            except Exception:
                pass

    assert call_counter["observe"] == 0, (
        f"AC-P7.4 violation: get_task made {call_counter['observe']} "
        f"TCMM observe calls; task lookups must stay cold."
    )


@pytest.mark.asyncio
async def test_ac_p7_4_list_comments_cold():
    """Comment-list reads must stay cold (M4 split keeps SHA chain in
    the ledger, no TCMM detour for state-machine comments)."""
    call_counter = {"observe": 0}

    async def _spy(*a, **kw):
        call_counter["observe"] += 1
        return False

    fake_store = mock.MagicMock()
    fake_store.table.return_value = _mock_lance_table_with_empty_arrow()

    with mock.patch("app.middleware.tcmm.observe_agent_output", _spy), \
         mock.patch("app.ledger.comments.LedgerStore.get", return_value=fake_store):
        from app.ledger import comments as _c
        for _ in range(50):
            try:
                _c.list_comments(
                    task_id="task-nonexistent-probe",
                    tenant_id="probe", user_id="probe",
                )
            except Exception:
                pass

    assert call_counter["observe"] == 0, (
        f"AC-P7.4 violation: list_comments made {call_counter['observe']} "
        f"TCMM observe calls; comment reads must stay cold."
    )
