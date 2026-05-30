"""Phase 7.5 — team_id propagation through dispatch.

When a task carries a `team_id`, the inbox-poller dispatches it with
that team_id set in the TenantContext so downstream tool calls (cost
rollup, team_cost_report, team-scoped recall) know which team to bill.

Covers:
  * TenantContext fields: task_id + team_id present.
  * set_tenant_context accepts and persists them.
  * Source-grep: inbox_poller dispatch reads task_row['team_id'] and
    threads it to run_agent_query.
  * Source-grep: run_agent_query forwards team_id to set_tenant_context.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_tenant_context_has_team_id_field():
    from app.middleware.tenant import TenantContext
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(TenantContext)}
    assert "team_id" in field_names
    assert "task_id" in field_names


def test_set_tenant_context_accepts_team_id():
    from app.middleware.tenant import set_tenant_context, current
    with set_tenant_context(
        conversation_id="conv-1",
        user_id="u1",
        tenant_id="t1",
        agent_id="researcher",
        task_id="task-abc",
        team_id="team-xyz",
    ) as ctx:
        assert ctx.task_id == "task-abc"
        assert ctx.team_id == "team-xyz"
        live = current()
        assert live is ctx


def test_set_tenant_context_resets_after_exit():
    from app.middleware.tenant import set_tenant_context, current
    with set_tenant_context(
        conversation_id="conv-1", user_id="u1", tenant_id="t1",
        agent_id="researcher", task_id="task-A", team_id="team-A",
    ):
        pass
    # After exit, no live context (test isolation).
    assert current() is None


def test_inbox_poller_reads_team_id_from_task_row():
    src = (REPO_ROOT / "app" / "workers" / "inbox_poller.py").read_text(encoding="utf-8")
    # Wire-up grep — the dispatch must read team_id from the task_row
    # and thread it to run_agent_query.
    assert "task_row.get(\"team_id\")" in src or "task_row.get('team_id')" in src
    assert "team_id=_team_id_for_dispatch" in src


def test_run_agent_query_forwards_team_id():
    src = (REPO_ROOT / "app" / "runtime.py").read_text(encoding="utf-8")
    # Signature accepts team_id and passes it to set_tenant_context.
    assert "team_id: Optional[str] = None" in src
    assert "team_id=team_id" in src
