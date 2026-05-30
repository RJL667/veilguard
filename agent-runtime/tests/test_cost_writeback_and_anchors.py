"""Phase 7.5 finishers — cost write-back + snapshot-anchor wiring.

#2: When the LLM turn completes for a Task, runtime.py increments
    `agent_tasks.cost_attributed_usd` by the turn's cost.  Closes the
    team budget loop without a sweeper.

#3: Director's route / synthesize / propose now emit SnapshotAnchors
    at each decision point so the future replay harness can detect
    "same substrate state" via the anchor_hash.
"""

from __future__ import annotations

import asyncio
import unittest.mock as mock
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


# ── increment_cost_attributed (#2) ──────────────────────────────────────


def test_increment_cost_attributed_zero_delta_returns_false():
    from app.ledger.tasks import increment_cost_attributed
    ok = increment_cost_attributed(
        task_id="task-x", tenant_id="t1", user_id="u1", delta_usd=0.0,
    )
    assert ok is False


def test_increment_cost_attributed_negative_delta_returns_false():
    from app.ledger.tasks import increment_cost_attributed
    ok = increment_cost_attributed(
        task_id="task-x", tenant_id="t1", user_id="u1", delta_usd=-1.0,
    )
    assert ok is False


def test_increment_cost_attributed_reads_then_writes():
    """Mock the LedgerStore — verify the function reads current cost,
    adds the delta, and writes the new total."""
    from app.ledger import tasks as _t
    import pyarrow as pa

    # Mocked Lance table that returns cost_attributed_usd=10.0 on read.
    arr = pa.Table.from_pylist([{"cost_attributed_usd": 10.0}])
    arr_mock = mock.MagicMock()
    arr_mock.num_rows = 1
    arr_mock.column = arr.column

    q = mock.MagicMock()
    q.where.return_value = q
    q.select.return_value = q
    q.limit.return_value = q
    q.to_arrow.return_value = arr_mock

    tbl = mock.MagicMock()
    tbl.search.return_value = q

    fake_store = mock.MagicMock()
    fake_store.table.return_value = tbl

    with mock.patch("app.ledger.tasks.LedgerStore.get", return_value=fake_store):
        ok = _t.increment_cost_attributed(
            task_id="task-1", tenant_id="t1", user_id="u1",
            delta_usd=2.5,
        )
    assert ok is True
    # The update payload must carry cost_attributed_usd = 12.5.
    update_call = tbl.update.call_args
    assert update_call is not None
    values = update_call.kwargs.get("values") or {}
    assert values.get("cost_attributed_usd") == pytest.approx(12.5)


def test_increment_cost_attributed_missing_task_returns_false():
    from app.ledger import tasks as _t
    import pyarrow as pa
    arr = pa.Table.from_pylist([{"cost_attributed_usd": 0.0}]).slice(0, 0)
    arr_mock = mock.MagicMock()
    arr_mock.num_rows = 0
    arr_mock.column = arr.column

    q = mock.MagicMock()
    q.where.return_value = q
    q.select.return_value = q
    q.limit.return_value = q
    q.to_arrow.return_value = arr_mock

    tbl = mock.MagicMock()
    tbl.search.return_value = q

    fake_store = mock.MagicMock()
    fake_store.table.return_value = tbl

    with mock.patch("app.ledger.tasks.LedgerStore.get", return_value=fake_store):
        ok = _t.increment_cost_attributed(
            task_id="task-phantom", tenant_id="t1", user_id="u1",
            delta_usd=1.0,
        )
    assert ok is False
    tbl.update.assert_not_called()


# ── runtime.py cost write-back wiring (#2) ──────────────────────────────


def test_runtime_calls_increment_cost_attributed_on_turn_end():
    """Source-grep: runtime.py must call increment_cost_attributed only
    when task_id is set (background dispatch path) so unrelated /agent/query
    calls don't crash."""
    src = (REPO_ROOT / "app" / "runtime.py").read_text(encoding="utf-8")
    assert "increment_cost_attributed" in src
    # The call must be guarded by `if task_id:` so non-task /agent/query
    # turns don't try to attribute cost.
    idx = src.find("increment_cost_attributed")
    pre = src[max(0, idx - 600):idx]
    assert "if task_id" in pre, (
        "cost write-back must be guarded by `if task_id:` so "
        "non-Task turns aren't billed against a phantom row"
    )


# ── Director snapshot anchor wiring (#3) ────────────────────────────────


def test_director_route_emits_snapshot_anchor():
    """When `route` is called, a SnapshotAnchor must be computed and
    `record_anchor` invoked (best-effort)."""
    from agent.director import DirectorAgent

    class _Persona:
        agent_id = "director-test"
        system_prompt = ""
        model_for = staticmethod(lambda kind: "claude-sonnet-4-5")

    agent = DirectorAgent.__new__(DirectorAgent)
    agent.persona = _Persona()  # type: ignore[attr-defined]

    # Stub tenant context so _emit_anchor proceeds past its guard.
    from app.middleware import tenant
    with tenant.set_tenant_context(
        conversation_id="conv-1", user_id="u1", tenant_id="t1",
        agent_id="director",
    ):
        with mock.patch("app.replay.record_anchor", return_value=True) as recorded:
            asyncio.run(agent.route({"brief": "test", "signal_type": "x"}))
        assert recorded.call_count == 1


def test_director_synthesize_emits_snapshot_anchor_with_task_id():
    from agent.director import DirectorAgent
    from app.middleware import tenant

    class _Persona:
        agent_id = "director-test"
        system_prompt = ""
        model_for = staticmethod(lambda kind: "claude-sonnet-4-5")

    agent = DirectorAgent.__new__(DirectorAgent)
    agent.persona = _Persona()  # type: ignore[attr-defined]

    with tenant.set_tenant_context(
        conversation_id="conv-1", user_id="u1", tenant_id="t1",
        agent_id="director",
    ):
        with mock.patch("app.replay.record_anchor", return_value=True) as recorded:
            asyncio.run(agent.synthesize("task-abc", [{"output": "x"}]))
        # Verify the record_anchor call carries the task_id.
        assert recorded.call_count == 1
        call_kwargs = recorded.call_args.kwargs
        assert call_kwargs.get("task_id") == "task-abc"


def test_director_propose_emits_snapshot_anchor():
    from agent.director import DirectorAgent
    from app.middleware import tenant

    class _Persona:
        agent_id = "director-test"
        system_prompt = ""
        model_for = staticmethod(lambda kind: "claude-sonnet-4-5")

    agent = DirectorAgent.__new__(DirectorAgent)
    agent.persona = _Persona()  # type: ignore[attr-defined]

    with tenant.set_tenant_context(
        conversation_id="conv-1", user_id="u1", tenant_id="t1",
        agent_id="director",
    ):
        with mock.patch("app.replay.record_anchor", return_value=True) as recorded:
            asyncio.run(agent.propose([{"id": "p-1"}, {"id": "p-2"}]))
        assert recorded.call_count == 1


def test_director_methods_dont_crash_without_tenant_context():
    """Director's anchor emission must be best-effort — if no tenant
    context is set, the methods still complete successfully."""
    from agent.director import DirectorAgent

    class _Persona:
        agent_id = "director-test"
        system_prompt = ""
        model_for = staticmethod(lambda kind: "claude-sonnet-4-5")

    agent = DirectorAgent.__new__(DirectorAgent)
    agent.persona = _Persona()  # type: ignore[attr-defined]

    # No tenant context — anchor emission should silently skip.
    result = asyncio.run(agent.route({"brief": "test"}))
    assert result["target_persona"] == "researcher"


def test_director_anchor_decision_points_match_telemetry_keys():
    """The 3 anchor decision points (director.route, director.synthesize,
    director.propose) must line up with the 3 telemetry buckets so a
    replay match also has a corresponding latency measurement."""
    from agent.director import DIRECTOR_METHOD_LATENCY_MS
    expected_methods = {"route", "synthesize", "propose"}
    assert set(DIRECTOR_METHOD_LATENCY_MS.keys()) == expected_methods
