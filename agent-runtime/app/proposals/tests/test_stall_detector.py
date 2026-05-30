"""Phase 7.5 stall detector — unit tests.

Cover the three trigger conditions (lease expired / heartbeat silent /
progress frozen), the idempotency check, and the worker discovery
loop.  All mocked Lance; no live DB needed.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest


def _mock_table(rows: list[dict]) -> MagicMock:
    """Build a fake Lance table whose .search().where().limit().to_arrow()
    returns a pa.Table with the requested rows."""
    tbl = MagicMock()
    arr = pa.Table.from_pylist(rows) if rows else pa.Table.from_pylist([{}]).slice(0, 0)
    query = MagicMock()
    query.where.return_value = query
    query.limit.return_value = query
    query.select.return_value = query
    query.to_arrow.return_value = arr
    query.to_pylist.return_value = rows
    # Some code paths call .to_arrow().to_pylist() — chain it.
    arr_mock = MagicMock()
    arr_mock.to_pylist.return_value = rows
    arr_mock.num_rows = len(rows)
    # Make column access work on the real pa.Table.
    arr_mock.column = arr.column
    arr_mock.column_names = arr.column_names
    query.to_arrow.return_value = arr_mock
    tbl.search.return_value = query
    return tbl


# ── detect_stalls ───────────────────────────────────────────────────────


def test_detect_stall_lease_expired_past_grace():
    from app.proposals.stall_detector import (
        detect_stalls, STALL_LEASE_GRACE_S, TRIGGER_LEASE_EXPIRED,
    )
    now = 10_000_000.0
    # Lease expired 10 min ago — well past the grace window.
    task = {
        "id": "task-stuck-1", "kind": "task", "status": "in_progress",
        "tenant_id": "t1", "user_id": "u1",
        "lease_owner": "worker-x",
        "lease_until": now - 600.0,
        "extras_json": None, "owner_id": "researcher",
    }
    fake_tasks_tbl = _mock_table([task])
    fake_store = MagicMock()
    fake_store.table.return_value = fake_tasks_tbl
    with patch("app.proposals.stall_detector.LedgerStore.get", return_value=fake_store):
        findings = detect_stalls(tenant_id="t1", user_id="u1", now=now)
    assert len(findings) == 1
    assert findings[0]["trigger"] == TRIGGER_LEASE_EXPIRED
    assert "task-stuck-1" == findings[0]["task_id"]


def test_no_stall_when_within_grace():
    """Lease expired but within grace — inbox-poller will reclaim, no
    stall comment needed."""
    from app.proposals.stall_detector import detect_stalls, STALL_LEASE_GRACE_S
    now = 10_000_000.0
    task = {
        "id": "task-ok", "kind": "task", "status": "in_progress",
        "tenant_id": "t1", "user_id": "u1",
        "lease_owner": "worker-x",
        "lease_until": now - (STALL_LEASE_GRACE_S / 2),  # just expired
        "extras_json": None, "owner_id": "researcher",
    }
    fake_store = MagicMock()
    fake_store.table.return_value = _mock_table([task])
    # Stub heartbeat lookup to return None so trigger #2 is skipped
    # (otherwise the shared MagicMock leaks the task rows into the
    # heartbeat scan and falsely fires heartbeat_silent on time=0).
    with patch("app.proposals.stall_detector.LedgerStore.get", return_value=fake_store), \
         patch("app.proposals.stall_detector._last_heartbeat_for_task", return_value=None):
        findings = detect_stalls(tenant_id="t1", user_id="u1", now=now)
    assert findings == []


def test_detect_stall_progress_frozen():
    from app.proposals.stall_detector import (
        detect_stalls, STALL_PROGRESS_FROZEN_S, TRIGGER_PROGRESS_FROZEN,
    )
    now = 10_000_000.0
    # Lease in future so trigger #1 doesn't fire.
    # No heartbeat → trigger #2 short-circuits (returns None).
    # progress_ledger latest step way too old → trigger #3 should fire.
    last_step = now - (STALL_PROGRESS_FROZEN_S * 2)
    extras = {"progress_ledger": [
        {"step": 0, "ts": now - 10000, "observation": "started"},
        {"step": 1, "ts": last_step, "observation": "stuck"},
    ]}
    task = {
        "id": "task-frozen", "kind": "task", "status": "in_progress",
        "tenant_id": "t1", "user_id": "u1",
        "lease_owner": "worker-x",
        "lease_until": now + 3600,  # future — trigger 1 won't fire
        "extras_json": json.dumps(extras),
    }
    fake_store = MagicMock()
    fake_store.table.return_value = _mock_table([task])
    with patch("app.proposals.stall_detector.LedgerStore.get", return_value=fake_store), \
         patch("app.proposals.stall_detector._last_heartbeat_for_task", return_value=None):
        findings = detect_stalls(tenant_id="t1", user_id="u1", now=now)
    assert len(findings) == 1
    assert findings[0]["trigger"] == TRIGGER_PROGRESS_FROZEN
    assert findings[0]["evidence"]["progress_steps"] == 2


def test_detect_stall_heartbeat_silent():
    from app.proposals.stall_detector import (
        detect_stalls, STALL_HEARTBEAT_SILENT_S, TRIGGER_HEARTBEAT_SILENT,
    )
    now = 10_000_000.0
    last_beat = now - (STALL_HEARTBEAT_SILENT_S * 2)
    task = {
        "id": "task-silent", "kind": "task", "status": "in_progress",
        "tenant_id": "t1", "user_id": "u1",
        "lease_owner": "worker-x",
        "lease_until": now + 3600,
        "extras_json": None,
    }
    fake_store = MagicMock()
    fake_store.table.return_value = _mock_table([task])
    with patch("app.proposals.stall_detector.LedgerStore.get", return_value=fake_store), \
         patch("app.proposals.stall_detector._last_heartbeat_for_task", return_value=last_beat):
        findings = detect_stalls(tenant_id="t1", user_id="u1", now=now)
    assert len(findings) == 1
    assert findings[0]["trigger"] == TRIGGER_HEARTBEAT_SILENT


def test_detect_stalls_skips_terminal_tasks():
    """Status != in_progress → never flagged.  The DB filter handles
    this at scan time, but defensive: the where-clause is the contract."""
    from app.proposals.stall_detector import detect_stalls
    # Empty table simulates "no in_progress rows" — the where clause did its job.
    fake_store = MagicMock()
    fake_store.table.return_value = _mock_table([])
    with patch("app.proposals.stall_detector.LedgerStore.get", return_value=fake_store):
        findings = detect_stalls(tenant_id="t1", user_id="u1", now=time.time())
    assert findings == []


def test_lease_expired_short_circuits_progress_check():
    """A task that triggers lease_expired must NOT also be added under
    progress_frozen — avoid double-counting in by_trigger."""
    from app.proposals.stall_detector import (
        detect_stalls, STALL_LEASE_GRACE_S, STALL_PROGRESS_FROZEN_S,
    )
    now = 10_000_000.0
    extras = {"progress_ledger": [
        {"step": 0, "ts": now - (STALL_PROGRESS_FROZEN_S * 2)},
    ]}
    task = {
        "id": "task-both", "kind": "task", "status": "in_progress",
        "tenant_id": "t1", "user_id": "u1",
        "lease_owner": "worker-x",
        "lease_until": now - (STALL_LEASE_GRACE_S * 2),  # expired
        "extras_json": json.dumps(extras),
    }
    fake_store = MagicMock()
    fake_store.table.return_value = _mock_table([task])
    with patch("app.proposals.stall_detector.LedgerStore.get", return_value=fake_store):
        findings = detect_stalls(tenant_id="t1", user_id="u1", now=now)
    assert len(findings) == 1, "should flag exactly once"


# ── emit_stall_comments ─────────────────────────────────────────────────


def test_emit_stall_writes_blocker_raised_comment():
    from app.proposals.stall_detector import emit_stall_comments
    findings = [{
        "task_id": "task-x", "trigger": "lease_expired",
        "reason": "lease expired 1000s ago",
        "evidence": {"lease_until": 0, "lease_owner": "w-1"},
    }]
    written_kinds: list[str] = []
    written_bodies: list[str] = []

    def _fake_add(*, task_id, tenant_id, user_id, author_id, kind, body, extras_json=None):
        written_kinds.append(kind)
        written_bodies.append(body)
        return "cmt-x"

    with patch("app.proposals.stall_detector._existing_open_stall_comment", return_value=False), \
         patch("app.ledger.comments.add_comment", side_effect=_fake_add):
        n = emit_stall_comments(
            tenant_id="t1", user_id="u1", findings=findings,
        )
    assert n == 1
    assert written_kinds == ["blocker_raised"]
    assert written_bodies[0].startswith("[task_stalled] trigger=lease_expired")
    assert "evidence=" in written_bodies[0]


def test_emit_stall_idempotent_when_already_open():
    from app.proposals.stall_detector import emit_stall_comments
    findings = [{
        "task_id": "task-x", "trigger": "lease_expired",
        "reason": "...", "evidence": {},
    }]
    call_count = {"n": 0}

    def _fake_add(*a, **kw):
        call_count["n"] += 1
        return "cmt-x"

    with patch("app.proposals.stall_detector._existing_open_stall_comment", return_value=True), \
         patch("app.ledger.comments.add_comment", side_effect=_fake_add):
        n = emit_stall_comments(
            tenant_id="t1", user_id="u1", findings=findings,
        )
    assert n == 0
    assert call_count["n"] == 0


# ── _existing_open_stall_comment ────────────────────────────────────────


def test_existing_stall_finds_open_marker():
    from app.proposals.stall_detector import _existing_open_stall_comment
    rows = [
        {"body": "user comment, unrelated", "ts": 100.0},
        {"body": "[task_stalled] trigger=lease_expired ...", "ts": 200.0},
    ]
    with patch("app.ledger.comments.list_comments", return_value=rows):
        assert _existing_open_stall_comment("task-1", "t1", "u1") is True


def test_existing_stall_finds_cleared_marker():
    """If a `stall_cleared` follows the `task_stalled`, the stall was
    resolved — allow re-flagging."""
    from app.proposals.stall_detector import _existing_open_stall_comment
    rows = [
        {"body": "[task_stalled] trigger=lease_expired", "ts": 100.0},
        {"body": "[stall_cleared] worker resumed", "ts": 200.0},
    ]
    with patch("app.ledger.comments.list_comments", return_value=rows):
        assert _existing_open_stall_comment("task-1", "t1", "u1") is False


def test_existing_stall_returns_false_when_no_marker():
    from app.proposals.stall_detector import _existing_open_stall_comment
    rows = [
        {"body": "user comment", "ts": 100.0},
        {"body": "another comment", "ts": 200.0},
    ]
    with patch("app.ledger.comments.list_comments", return_value=rows):
        assert _existing_open_stall_comment("task-1", "t1", "u1") is False


# ── Worker scaffolding ──────────────────────────────────────────────────


def test_worker_init_defaults_match_spec():
    from app.proposals.stall_detector import StallDetectorWorker
    w = StallDetectorWorker()
    assert w.interval_seconds == 300.0  # 5 min default


def test_worker_discover_pairs_returns_unique_sorted():
    from app.proposals.stall_detector import StallDetectorWorker
    rows = [
        {"tenant_id": "t1", "user_id": "u1"},
        {"tenant_id": "t1", "user_id": "u1"},   # dup
        {"tenant_id": "t2", "user_id": "u2"},
    ]
    fake_store = MagicMock()
    fake_store.table.return_value = _mock_table(rows)
    with patch("app.proposals.stall_detector.LedgerStore.get", return_value=fake_store):
        w = StallDetectorWorker()
        pairs = w.discover_active_pairs()
    assert pairs == [("t1", "u1"), ("t2", "u2")]


def test_run_one_cycle_returns_summary_with_by_trigger():
    from app.proposals.stall_detector import run_one_cycle, TRIGGER_LEASE_EXPIRED
    findings = [
        {"task_id": "t1", "trigger": TRIGGER_LEASE_EXPIRED, "reason": "x", "evidence": {}},
        {"task_id": "t2", "trigger": TRIGGER_LEASE_EXPIRED, "reason": "x", "evidence": {}},
    ]
    with patch("app.proposals.stall_detector.detect_stalls", return_value=findings), \
         patch("app.proposals.stall_detector.emit_stall_comments", return_value=2):
        summary = run_one_cycle(tenant_id="t1", user_id="u1")
    assert summary["findings"] == 2
    assert summary["emitted"] == 2
    assert summary["by_trigger"] == {TRIGGER_LEASE_EXPIRED: 2}


# ── Exports surface ─────────────────────────────────────────────────────


def test_exported_constants_present():
    from app.proposals import stall_detector as sd
    for name in (
        "STALL_LEASE_GRACE_S", "STALL_PROGRESS_FROZEN_S",
        "STALL_HEARTBEAT_SILENT_S",
        "TRIGGER_LEASE_EXPIRED", "TRIGGER_HEARTBEAT_SILENT",
        "TRIGGER_PROGRESS_FROZEN",
    ):
        assert hasattr(sd, name), f"stall_detector missing export {name}"
