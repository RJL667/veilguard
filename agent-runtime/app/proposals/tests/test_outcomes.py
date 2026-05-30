"""Unit tests for proposals/outcomes.py — regret writer.

No real Lance; all tables mocked.  Validates:
  * Eligibility gate (status=approved + 30d-old terminal task + idempotent)
  * Cost roll-up from pii_audit
  * value_realized proxy via dream overlap
  * regret_score = cost / max(value, ε)
  * Idempotency — running twice doesn't double-write
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from app.proposals import outcomes as O


# ── Test fixtures: tiny fake Lance objects ──────────────────────────────


class FakeColumn:
    def __init__(self, values):
        self._v = values
    def __getitem__(self, i):
        v = self._v[i]
        return FakeCell(v)


class FakeCell:
    def __init__(self, v): self._v = v
    def as_py(self): return self._v


class FakeArrow:
    def __init__(self, rows: list[dict]):
        self._rows = rows
        self.num_rows = len(rows)
        keys = set()
        for r in rows: keys.update(r.keys())
        self.column_names = sorted(keys)
    def column(self, name):
        return FakeColumn([r.get(name) for r in self._rows])


class FakeTable:
    """Minimal table that supports search().where().select().limit().to_arrow()
    and .add().  Stores added rows in `.added`; .where filters by ==."""
    def __init__(self, rows: list[dict] | None = None):
        self._rows = list(rows or [])
        self.added: list[dict] = []
        self._pending_filter = None
    def search(self):
        self._pending_filter = None
        return self
    def where(self, expr: str):
        # parse simple "col = 'val'" or "col = N" filter
        self._pending_filter = expr
        return self
    def select(self, *_args, **_kw): return self
    def limit(self, n): self._n = n; return self
    def to_arrow(self):
        rows = self._rows
        if self._pending_filter:
            # crude filter parser: supports `col = 'val'` and `col = N`
            expr = self._pending_filter
            # tolerate `col IN (a, b)` lazily — not used in tests
            if " = " in expr:
                key, val = expr.split(" = ", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                rows = [r for r in rows if str(r.get(key)) == val]
        rows = rows[: getattr(self, "_n", 5000)]
        self._pending_filter = None
        return FakeArrow(rows)
    def add(self, new_rows: list[dict]):
        self.added.extend(new_rows)
        self._rows.extend(new_rows)


# ── Helpers ────────────────────────────────────────────────────────────


def _proposal(pid="prop-1", task_id="task-1", status="approved",
              signal_type="information_gap", tenant="t1", user="u1",
              impact=2.5):
    return {
        "id": pid, "status": status, "resulting_task_id": task_id,
        "signal_type": signal_type,
        "tenant_id": tenant, "user_id": user,
        "proposed_assignee": "researcher",
        "impact_score": impact,
    }


def _task(tid="task-1", status="done", outputs=None, updated_age_days=45):
    return {
        "id": tid, "status": status,
        "outputs": outputs or ["team/drafts/popia-card.md"],
        "updated_ts": time.time() - updated_age_days * 86400,
    }


def _audit_row(task_id="task-1", cost=0.05, tokens_in=2000, tokens_out=500,
               model="claude-sonnet-4-6", cache_create=0, cache_read=100):
    """Audit-row fixture.  Note: as of 2026-05-28 _compute_task_cost
    derives `cost` from tokens × per-model rate card (memory
    `architecture_token_accounting.md`); the `cost` kwarg here is
    retained for back-compat with old test signatures but is IGNORED
    by the production path.  Pass `tokens_in`/`tokens_out` to drive
    expected dollar amounts.
    """
    return {
        "task_id": task_id, "model": model,
        # `cost_usd` is still in the fixture for back-compat with any
        # test that wants to introspect it, but the production code
        # no longer reads this column.
        "cost_usd": cost,
        "tokens_input": tokens_in, "tokens_output": tokens_out,
        "cache_create": cache_create, "cache_read": cache_read,
    }


# Derived rate-card constants (Sonnet 4.6 — matches fixture default).
# Used to compute expected dollar values in tests without hardcoding.
_SONNET_4_6_IN_PER_M  = 3.0      # USD/M input tokens
_SONNET_4_6_OUT_PER_M = 15.0     # USD/M output tokens
_SONNET_4_6_CR_PER_M  = 0.30     # USD/M cache_read tokens
_SONNET_4_6_CW_PER_M  = 6.0      # USD/M cache_create tokens


def _expected_cost_sonnet(tokens_in: int, tokens_out: int,
                          cache_create: int = 0, cache_read: int = 100) -> float:
    """Compute expected USD cost for an audit row with Sonnet 4.6 model."""
    return (
        tokens_in    * _SONNET_4_6_IN_PER_M  / 1_000_000.0 +
        tokens_out   * _SONNET_4_6_OUT_PER_M / 1_000_000.0 +
        cache_create * _SONNET_4_6_CW_PER_M  / 1_000_000.0 +
        cache_read   * _SONNET_4_6_CR_PER_M  / 1_000_000.0
    )


# ── find_eligible_proposals ─────────────────────────────────────────────


def test_eligible_basic_done_task_45d_old():
    proposals = FakeTable([_proposal()])
    tasks = FakeTable([_task(updated_age_days=45)])
    outcomes = FakeTable([])  # nothing computed yet
    eligible = O.find_eligible_proposals(
        proposals_tbl=proposals, tasks_tbl=tasks, outcomes_tbl=outcomes,
    )
    assert len(eligible) == 1
    assert eligible[0]["proposal_id"] == "prop-1"
    assert eligible[0]["task_status"] == "done"


def test_eligible_skips_proposals_with_existing_outcome():
    proposals = FakeTable([_proposal()])
    tasks = FakeTable([_task(updated_age_days=45)])
    outcomes = FakeTable([{"id": "outc-x", "proposal_id": "prop-1"}])
    eligible = O.find_eligible_proposals(
        proposals_tbl=proposals, tasks_tbl=tasks, outcomes_tbl=outcomes,
    )
    assert eligible == []


def test_eligible_skips_pending_proposals():
    """status != approved → not eligible."""
    proposals = FakeTable([_proposal(status="pending")])
    tasks = FakeTable([_task()])
    outcomes = FakeTable([])
    eligible = O.find_eligible_proposals(
        proposals_tbl=proposals, tasks_tbl=tasks, outcomes_tbl=outcomes,
    )
    assert eligible == []


def test_eligible_skips_recent_tasks():
    """Task only 5d old (< 30d threshold) → not eligible yet."""
    proposals = FakeTable([_proposal()])
    tasks = FakeTable([_task(updated_age_days=5)])
    outcomes = FakeTable([])
    eligible = O.find_eligible_proposals(
        proposals_tbl=proposals, tasks_tbl=tasks, outcomes_tbl=outcomes,
    )
    assert eligible == []


def test_eligible_skips_in_progress_tasks():
    """Task not in terminal state → not eligible."""
    proposals = FakeTable([_proposal()])
    tasks = FakeTable([_task(status="in_progress")])
    outcomes = FakeTable([])
    eligible = O.find_eligible_proposals(
        proposals_tbl=proposals, tasks_tbl=tasks, outcomes_tbl=outcomes,
    )
    assert eligible == []


def test_eligible_handles_missing_resulting_task():
    """Proposal references a task_id that doesn't exist in agent_tasks → skip."""
    proposals = FakeTable([_proposal()])
    tasks = FakeTable([])   # task gone
    outcomes = FakeTable([])
    eligible = O.find_eligible_proposals(
        proposals_tbl=proposals, tasks_tbl=tasks, outcomes_tbl=outcomes,
    )
    assert eligible == []


def test_eligible_passes_through_cancelled_tasks():
    """Cancelled tasks ARE eligible — their regret score reflects sunk cost."""
    proposals = FakeTable([_proposal()])
    tasks = FakeTable([_task(status="cancelled")])
    outcomes = FakeTable([])
    eligible = O.find_eligible_proposals(
        proposals_tbl=proposals, tasks_tbl=tasks, outcomes_tbl=outcomes,
    )
    assert len(eligible) == 1
    assert eligible[0]["task_status"] == "cancelled"


# ── _compute_task_cost ──────────────────────────────────────────────────


def test_task_cost_sums_audit_rows():
    audit = FakeTable([
        _audit_row(cost=0.05),
        _audit_row(cost=0.12),
        _audit_row(cost=0.03),
    ])
    cost, tokens = O._compute_task_cost(audit_tbl=audit, task_id="task-1")
    # Each _audit_row defaults to 2000 in, 500 out, 0 cc, 100 cr on Sonnet 4.6.
    expected = 3 * _expected_cost_sonnet(2000, 500)
    assert cost == pytest.approx(expected)
    assert tokens["tokens_input"] == 6000
    assert tokens["tokens_output"] == 1500


def test_task_cost_zero_when_no_audit_rows():
    audit = FakeTable([])
    cost, tokens = O._compute_task_cost(audit_tbl=audit, task_id="task-1")
    assert cost == 0.0
    assert tokens == {
        "tokens_input": 0, "tokens_output": 0,
        "cache_create": 0, "cache_read": 0,
    }


# ── compute_one + regret math ──────────────────────────────────────────


# Phase 7 wire-up: compute_one now writes via the M2 split-writer
# instead of directly into the FakeTable.  We patch the writer and
# capture its kwargs to assert on (and mirror into the FakeTable so
# downstream idempotency lookups still find the row).
def _patch_writer(outcomes_tbl: "FakeTable") -> "Any":
    captured: list[dict] = []

    async def _stub(**kwargs):
        captured.append(kwargs)
        # Mirror into the FakeTable so find_eligible_proposals's
        # idempotency lookup ("does an outcome row already exist?")
        # behaves the same as before.
        import json as _json
        oid = f"out-test-{len(captured)}"
        outcomes_tbl.added.append({
            "id":                  oid,
            "proposal_id":         kwargs["proposal_id"],
            "resulting_task_id":   kwargs.get("resulting_task_id"),
            "task_status":         kwargs["task_status"],
            "task_cost_usd":       kwargs["task_cost_usd"],
            "value_realized":      kwargs["value_realized"],
            "regret_score":        kwargs["regret_score"],
            "tenant_id":           kwargs["tenant_id"],
            "user_id":             kwargs["user_id"],
            "objective_deltas_json": _json.dumps(kwargs.get("objective_deltas") or {}),
            "extras_json":         _json.dumps({"heuristic_ver": "v1"}),
        })
        # Also push into the FakeTable's row store so later searches see it
        outcomes_tbl._rows.append(outcomes_tbl.added[-1])
        return (oid, None)

    return _stub, captured


async def test_compute_one_writes_outcome_with_regret():
    proposals = FakeTable([_proposal()])
    tasks = FakeTable([_task(updated_age_days=45)])
    audit = FakeTable([_audit_row(cost=0.20)])
    outcomes = FakeTable([])
    dream = FakeTable([])  # no recalls → value_realized=0 → ε

    eligible = O.find_eligible_proposals(
        proposals_tbl=proposals, tasks_tbl=tasks, outcomes_tbl=outcomes,
    )[0]
    stub, captured = _patch_writer(outcomes)
    with patch("app.memory.phase_7_writers.record_outcome_with_narrative", side_effect=stub):
        summary = await O.compute_one(
            proposals_tbl=proposals, tasks_tbl=tasks, outcomes_tbl=outcomes,
            audit_tbl=audit, dream_tbl=dream, eligible=eligible,
        )
    # cost derived from tokens × rate card (Sonnet 4.6 default).
    expected_cost = _expected_cost_sonnet(2000, 500)
    assert summary["task_cost_usd"] == pytest.approx(expected_cost)
    assert summary["value_realized"] == 0.0
    # regret = expected_cost / max(0, ε=0.1)
    assert summary["regret_score"] == pytest.approx(expected_cost / 0.1)
    assert len(captured) == 1
    row = captured[0]
    assert row["proposal_id"] == "prop-1"
    assert row["task_status"] == "done"
    assert row["regret_score"] == pytest.approx(expected_cost / 0.1)


async def test_compute_one_value_realized_from_dream_overlap():
    """value_realized > 0 when dream rows mention the task output."""
    proposals = FakeTable([_proposal()])
    tasks = FakeTable([_task(
        outputs=["team/drafts/popia-card.md"], updated_age_days=45,
    )])
    audit = FakeTable([_audit_row(cost=0.10)])
    outcomes = FakeTable([])
    dream = FakeTable([
        {"text": "Reusing popia-card.md insights",
         "claims": [], "source_block_ids": [], "namespace": "other"},
        {"text": "Refers to team/drafts/popia-card.md",
         "claims": [], "source_block_ids": [], "namespace": "other"},
        {"text": "unrelated dream node",
         "claims": [], "source_block_ids": [], "namespace": "other"},
    ])
    eligible = O.find_eligible_proposals(
        proposals_tbl=proposals, tasks_tbl=tasks, outcomes_tbl=outcomes,
    )[0]
    stub, _ = _patch_writer(outcomes)
    with patch("app.memory.phase_7_writers.record_outcome_with_narrative", side_effect=stub):
        summary = await O.compute_one(
            proposals_tbl=proposals, tasks_tbl=tasks, outcomes_tbl=outcomes,
            audit_tbl=audit, dream_tbl=dream, eligible=eligible,
        )
    assert summary["value_realized"] == 2.0
    # cost from tokens × Sonnet rates; regret = cost / value
    expected_cost = _expected_cost_sonnet(2000, 500)
    assert summary["regret_score"] == pytest.approx(expected_cost / 2.0)


async def test_compute_one_excludes_self_recalls():
    """dream rows in the task's own namespace are NOT counted (gaming guard)."""
    proposals = FakeTable([_proposal()])
    tasks = FakeTable([_task(
        outputs=["team/drafts/popia-card.md"], updated_age_days=45,
    )])
    audit = FakeTable([])
    outcomes = FakeTable([])
    dream = FakeTable([
        {"text": "Reusing popia-card.md",
         "claims": [], "source_block_ids": [], "namespace": "task-1"},
        {"text": "Reusing popia-card.md",
         "claims": [], "source_block_ids": [], "namespace": "other-ns"},
    ])
    eligible = O.find_eligible_proposals(
        proposals_tbl=proposals, tasks_tbl=tasks, outcomes_tbl=outcomes,
    )[0]
    stub, _ = _patch_writer(outcomes)
    with patch("app.memory.phase_7_writers.record_outcome_with_narrative", side_effect=stub):
        summary = await O.compute_one(
            proposals_tbl=proposals, tasks_tbl=tasks, outcomes_tbl=outcomes,
            audit_tbl=audit, dream_tbl=dream, eligible=eligible,
        )
    # self-recall in "task-1" excluded; only "other-ns" counts
    assert summary["value_realized"] == 1.0


# ── Idempotency ────────────────────────────────────────────────────────


async def test_outcome_already_exists_blocks_double_write():
    proposals = FakeTable([_proposal()])
    tasks = FakeTable([_task(updated_age_days=45)])
    outcomes = FakeTable([])
    audit = FakeTable([])
    dream = FakeTable([])

    # First run — write the outcome
    eligible = O.find_eligible_proposals(
        proposals_tbl=proposals, tasks_tbl=tasks, outcomes_tbl=outcomes,
    )
    assert len(eligible) == 1
    stub, _ = _patch_writer(outcomes)
    with patch("app.memory.phase_7_writers.record_outcome_with_narrative", side_effect=stub):
        await O.compute_one(
            proposals_tbl=proposals, tasks_tbl=tasks, outcomes_tbl=outcomes,
            audit_tbl=audit, dream_tbl=dream, eligible=eligible[0],
        )
    # Second run — should find nothing (outcome already exists)
    eligible = O.find_eligible_proposals(
        proposals_tbl=proposals, tasks_tbl=tasks, outcomes_tbl=outcomes,
    )
    assert eligible == []
    # Still only 1 row written
    assert len(outcomes.added) == 1
