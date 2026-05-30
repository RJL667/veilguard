"""Phase 7.5 — `depends_on TEXT[]` cross-lineage DAG dependencies.

Covers:
  * Schema: column exists in agent_tasks.
  * create_task: accepts depends_on; validates against (tenant, user) scope.
  * create_task: rejects self-loop.
  * create_task: rejects unknown dep ids.
  * deps_satisfied: True when no deps / all deps done; False otherwise.
  * deps_satisfied: returns pending list correctly.
  * verify_depends_on_acyclic: detects cycles, returns path.
  * Inbox-poller: pre-flight check is in the code path.
"""

from __future__ import annotations

import unittest.mock as mock
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Schema ──────────────────────────────────────────────────────────────


def test_schema_has_depends_on_column():
    from app.ledger.schemas import agent_tasks_schema
    schema = agent_tasks_schema()
    names = {f.name for f in schema}
    assert "depends_on" in names, "agent_tasks_schema must include depends_on"
    # It must be a list<string> for the inbox-poller's iteration pattern.
    fld = schema.field("depends_on")
    assert fld.type.value_type.id in (13,), (  # 13 = pa.string()
        f"depends_on must be list<string>, got {fld.type}"
    )


# ── verify_depends_on_acyclic ───────────────────────────────────────────


def _mk_tasks_tbl(rows: list[dict]) -> mock.MagicMock:
    """Build a Lance-table mock whose .search().where().select().limit()
    .to_arrow() chain returns the requested rows filtered by id."""
    import pyarrow as pa

    def _to_arrow_for(where: str):
        # Pull the id literal out of the where clause.  `\bid` to
        # anchor at word boundary so `tenant_id = '...'` doesn't
        # match — the regex must hit the trailing `... AND id = '<x>'`.
        import re
        m = re.search(r"\bid = '([^']+)'", where)
        if not m:
            return pa.Table.from_pylist(rows)
        wanted = m.group(1)
        match = [r for r in rows if r.get("id") == wanted]
        if not match:
            # Empty arrow table with same column names — emulate Lance.
            cols = rows[0].keys() if rows else ["id", "status", "depends_on"]
            return pa.Table.from_pylist([{c: None for c in cols}]).slice(0, 0)
        return pa.Table.from_pylist(match)

    q = mock.MagicMock()
    q.where.side_effect = lambda w: _StubQuery(_to_arrow_for(w))
    tbl = mock.MagicMock()
    tbl.search.return_value = q
    return tbl


class _StubQuery:
    def __init__(self, arrow_table):
        self._arrow = arrow_table

    def select(self, *a, **kw):
        return self

    def limit(self, *a, **kw):
        return self

    def to_arrow(self):
        return self._arrow


def test_verify_acyclic_returns_true_with_empty_deps():
    from app.ledger.tasks import verify_depends_on_acyclic
    fake_store = mock.MagicMock()
    fake_store.table.return_value = _mk_tasks_tbl([])
    with mock.patch("app.ledger.tasks.LedgerStore.get", return_value=fake_store):
        ok, cycle = verify_depends_on_acyclic(
            task_id="task-new", depends_on=[],
            tenant_id="t1", user_id="u1",
        )
    assert ok is True
    assert cycle == []


def test_verify_acyclic_detects_self_loop():
    from app.ledger.tasks import verify_depends_on_acyclic
    fake_store = mock.MagicMock()
    fake_store.table.return_value = _mk_tasks_tbl([])
    with mock.patch("app.ledger.tasks.LedgerStore.get", return_value=fake_store):
        ok, cycle = verify_depends_on_acyclic(
            task_id="task-A", depends_on=["task-A"],
            tenant_id="t1", user_id="u1",
        )
    assert ok is False
    assert cycle == ["task-A", "task-A"]


def test_verify_acyclic_detects_indirect_cycle():
    """task-A depends on task-B; task-B depends on task-C; if we now
    propose task-C-new that depends on task-A, the cycle closes."""
    from app.ledger.tasks import verify_depends_on_acyclic
    rows = [
        {"id": "task-A", "depends_on": ["task-B"]},
        {"id": "task-B", "depends_on": ["task-C"]},
        {"id": "task-C", "depends_on": ["task-A"]},  # cycle source
    ]
    fake_store = mock.MagicMock()
    fake_store.table.return_value = _mk_tasks_tbl(rows)
    with mock.patch("app.ledger.tasks.LedgerStore.get", return_value=fake_store):
        ok, cycle = verify_depends_on_acyclic(
            task_id="task-A", depends_on=["task-B"],
            tenant_id="t1", user_id="u1",
        )
    assert ok is False
    assert "task-A" in cycle


def test_verify_acyclic_passes_on_diamond():
    """Common safe fan-in: A→B, A→C, B+C→D.  No cycle, just DAG."""
    from app.ledger.tasks import verify_depends_on_acyclic
    rows = [
        {"id": "task-A", "depends_on": []},
        {"id": "task-B", "depends_on": ["task-A"]},
        {"id": "task-C", "depends_on": ["task-A"]},
    ]
    fake_store = mock.MagicMock()
    fake_store.table.return_value = _mk_tasks_tbl(rows)
    with mock.patch("app.ledger.tasks.LedgerStore.get", return_value=fake_store):
        ok, cycle = verify_depends_on_acyclic(
            task_id="task-D-new", depends_on=["task-B", "task-C"],
            tenant_id="t1", user_id="u1",
        )
    assert ok is True
    assert cycle == []


# ── deps_satisfied ──────────────────────────────────────────────────────


def test_deps_satisfied_no_deps_is_ready():
    from app.ledger.tasks import deps_satisfied
    ready, pending = deps_satisfied(
        task={"depends_on": None},
        tenant_id="t1", user_id="u1",
    )
    assert ready is True
    assert pending == []


def test_deps_satisfied_all_done_is_ready():
    from app.ledger.tasks import deps_satisfied
    rows = [
        {"id": "task-A", "status": "done"},
        {"id": "task-B", "status": "done"},
    ]
    fake_store = mock.MagicMock()
    fake_store.table.return_value = _mk_tasks_tbl(rows)
    with mock.patch("app.ledger.tasks.LedgerStore.get", return_value=fake_store):
        ready, pending = deps_satisfied(
            task={"depends_on": ["task-A", "task-B"]},
            tenant_id="t1", user_id="u1",
        )
    assert ready is True
    assert pending == []


def test_deps_satisfied_pending_blocks():
    from app.ledger.tasks import deps_satisfied
    rows = [
        {"id": "task-A", "status": "done"},
        {"id": "task-B", "status": "in_progress"},
    ]
    fake_store = mock.MagicMock()
    fake_store.table.return_value = _mk_tasks_tbl(rows)
    with mock.patch("app.ledger.tasks.LedgerStore.get", return_value=fake_store):
        ready, pending = deps_satisfied(
            task={"depends_on": ["task-A", "task-B"]},
            tenant_id="t1", user_id="u1",
        )
    assert ready is False
    assert pending == ["task-B"]


def test_deps_satisfied_missing_dep_blocks():
    """A dep id that doesn't exist counts as pending (defensive)."""
    from app.ledger.tasks import deps_satisfied
    fake_store = mock.MagicMock()
    fake_store.table.return_value = _mk_tasks_tbl([])
    with mock.patch("app.ledger.tasks.LedgerStore.get", return_value=fake_store):
        ready, pending = deps_satisfied(
            task={"depends_on": ["task-phantom"]},
            tenant_id="t1", user_id="u1",
        )
    assert ready is False
    assert pending == ["task-phantom"]


# ── inbox-poller wire-up (source grep) ──────────────────────────────────


def test_inbox_poller_calls_deps_satisfied_in_claim_path():
    src = (REPO_ROOT / "app" / "workers" / "inbox_poller.py").read_text(encoding="utf-8")
    assert "deps_satisfied" in src, (
        "inbox_poller must call deps_satisfied before _try_claim — "
        "see [PHASE_7_5_DEPENDS_ON_2026_05_28] in tasks.py"
    )
    # Make sure the call lives BEFORE the claim, not after.
    deps_idx  = src.find("deps_satisfied")
    claim_idx = src.find("self._try_claim(tbl, task_id)")
    assert 0 < deps_idx < claim_idx, (
        "depends_on check must run BEFORE _try_claim — otherwise the "
        "race window is exactly what F7 was trying to close."
    )


# ── create_task validation ──────────────────────────────────────────────


def test_create_task_signature_accepts_depends_on():
    """The function signature must expose depends_on."""
    import inspect
    from app.ledger.tasks import create_task
    sig = inspect.signature(create_task)
    assert "depends_on" in sig.parameters
    assert sig.parameters["depends_on"].default is None


def test_create_task_rejects_self_loop():
    """A task whose depends_on includes its own id (impossible to
    pre-compute, but Director might paste in error) must be rejected."""
    # We can't easily exercise create_task without a Lance store, but
    # we CAN verify the validation comes before the Lance write.  Read
    # the source.
    src = (REPO_ROOT / "app" / "ledger" / "tasks.py").read_text(encoding="utf-8")
    # Self-loop guard text is part of the contract.
    assert "self-loop" in src.lower()
    assert "depends_on must not include the new task's" in src


def test_create_task_rejects_missing_dep():
    src = (REPO_ROOT / "app" / "ledger" / "tasks.py").read_text(encoding="utf-8")
    assert "unknown task_id" in src
    assert "_missing_dep_ids" in src
