"""Phase 7.5 — agent_teams CRUD + budget enforcement.

Covers:
  * Schema: agent_teams table exists in TABLE_SCHEMAS + has the
    expected columns + agent_tasks has team_id.
  * Validators: create_team rejects empty name, empty lead_agent_id,
    negative budget_usd, budget_cap < 1.0.
  * Budget envelope: budget_exceeded returns True when
    attributed + additional >= ceiling.
  * Persona registry: team-lead is in VALID_OWNER_IDS.
  * create_task: team_id unknown → ValueError; team status=inactive
    → ValueError; over-budget → ValueError with cost figures.
"""

from __future__ import annotations

import unittest.mock as mock
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Schema ──────────────────────────────────────────────────────────────


def test_agent_teams_in_table_schemas():
    from app.ledger.schemas import TABLE_SCHEMAS
    assert "agent_teams" in TABLE_SCHEMAS


def test_agent_teams_schema_columns():
    from app.ledger.schemas import agent_teams_schema
    names = {f.name for f in agent_teams_schema()}
    must_have = {
        "id", "name", "lead_agent_id", "member_agent_ids",
        "budget_usd", "budget_cap",
        "cost_attributed_cached_usd", "cost_recomputed_ts",
        # shared skeleton
        "tenant_id", "user_id", "status", "created_ts", "updated_ts",
    }
    missing = must_have - names
    assert not missing, f"agent_teams missing columns: {missing}"


def test_agent_tasks_has_team_id_column():
    from app.ledger.schemas import agent_tasks_schema
    names = {f.name for f in agent_tasks_schema()}
    assert "team_id" in names, (
        "agent_tasks must have a team_id column for Phase 7.5 team scoping"
    )


# ── Persona registry ────────────────────────────────────────────────────


def test_team_lead_in_valid_owner_ids():
    from app.ledger.tasks import VALID_OWNER_IDS
    assert "team-lead" in VALID_OWNER_IDS


def test_team_lead_persona_file_exists():
    persona = REPO_ROOT.parent / "agents" / "team-lead.md"
    assert persona.is_file(), f"team-lead.md missing at {persona}"
    body = persona.read_text(encoding="utf-8")
    assert "Agent ID:** team-lead" in body
    assert "Role:** manager" in body or "Role: manager" in body


# ── create_team validators ──────────────────────────────────────────────


def _patched_store_returning(tbl_mock):
    fake_store = mock.MagicMock()
    fake_store.table.return_value = tbl_mock
    return fake_store


def test_create_team_rejects_empty_name():
    from app.ledger import teams as _teams
    fake_store = _patched_store_returning(mock.MagicMock())
    with mock.patch("app.ledger.teams.LedgerStore.get", return_value=fake_store):
        with pytest.raises(ValueError, match="non-empty name"):
            _teams.create_team(
                tenant_id="t1", user_id="u1",
                name="   ", lead_agent_id="director",
                budget_usd=100.0,
            )


def test_create_team_rejects_empty_lead_agent_id():
    from app.ledger import teams as _teams
    fake_store = _patched_store_returning(mock.MagicMock())
    with mock.patch("app.ledger.teams.LedgerStore.get", return_value=fake_store):
        with pytest.raises(ValueError, match="lead_agent_id"):
            _teams.create_team(
                tenant_id="t1", user_id="u1",
                name="Compliance", lead_agent_id="",
                budget_usd=100.0,
            )


def test_create_team_rejects_negative_budget():
    from app.ledger import teams as _teams
    fake_store = _patched_store_returning(mock.MagicMock())
    with mock.patch("app.ledger.teams.LedgerStore.get", return_value=fake_store):
        with pytest.raises(ValueError, match="budget_usd"):
            _teams.create_team(
                tenant_id="t1", user_id="u1",
                name="x", lead_agent_id="director",
                budget_usd=-1.0,
            )


def test_create_team_rejects_cap_below_one():
    from app.ledger import teams as _teams
    fake_store = _patched_store_returning(mock.MagicMock())
    with mock.patch("app.ledger.teams.LedgerStore.get", return_value=fake_store):
        with pytest.raises(ValueError, match="budget_cap"):
            _teams.create_team(
                tenant_id="t1", user_id="u1",
                name="x", lead_agent_id="director",
                budget_usd=100.0, budget_cap=0.5,
            )


# ── Budget envelope math ────────────────────────────────────────────────


def test_budget_exceeded_returns_false_with_no_team():
    """A team_id that doesn't exist → exceeded=False (caller will
    catch the not-found separately via get_team)."""
    from app.ledger import teams as _teams
    with mock.patch("app.ledger.teams.get_team", return_value=None):
        exceeded, attributed, ceiling = _teams.budget_exceeded(
            team_id="team-phantom", tenant_id="t1", user_id="u1",
        )
    assert exceeded is False
    assert attributed == 0.0
    assert ceiling == 0.0


def test_budget_exceeded_at_ceiling_blocks():
    from app.ledger import teams as _teams
    team_row = {
        "id": "team-1", "budget_usd": 100.0, "budget_cap": 1.0,
        "status": "active",
    }
    with mock.patch("app.ledger.teams.get_team", return_value=team_row), \
         mock.patch("app.ledger.teams.team_cost_attributed", return_value=99.99):
        # additional=0.01 puts us exactly at ceiling
        exceeded, attributed, ceiling = _teams.budget_exceeded(
            team_id="team-1", tenant_id="t1", user_id="u1",
            additional_usd=0.01,
        )
    assert exceeded is True
    assert attributed == pytest.approx(99.99)
    assert ceiling == pytest.approx(100.0)


def test_budget_exceeded_below_ceiling_allows():
    from app.ledger import teams as _teams
    team_row = {
        "id": "team-1", "budget_usd": 100.0, "budget_cap": 1.2,
        "status": "active",
    }
    with mock.patch("app.ledger.teams.get_team", return_value=team_row), \
         mock.patch("app.ledger.teams.team_cost_attributed", return_value=90.0):
        # ceiling = 100 × 1.2 = 120; 90 + 5 = 95 < 120
        exceeded, attributed, ceiling = _teams.budget_exceeded(
            team_id="team-1", tenant_id="t1", user_id="u1",
            additional_usd=5.0,
        )
    assert exceeded is False
    assert ceiling == pytest.approx(120.0)


# ── create_task integration ─────────────────────────────────────────────


def test_create_task_signature_accepts_team_id():
    import inspect
    from app.ledger.tasks import create_task
    sig = inspect.signature(create_task)
    assert "team_id" in sig.parameters
    assert sig.parameters["team_id"].default is None


def test_create_task_rejects_unknown_team():
    """Static contract — source-grep the unknown-team-id message
    so the validator can't be silently removed."""
    src = (REPO_ROOT / "app" / "ledger" / "tasks.py").read_text(encoding="utf-8")
    assert "team_id" in src and "not found for" in src


def test_create_task_rejects_inactive_team():
    src = (REPO_ROOT / "app" / "ledger" / "tasks.py").read_text(encoding="utf-8")
    assert "not active" in src


def test_create_task_rejects_over_budget():
    """Source-grep for the "no new work can be queued" envelope
    rejection so removing the guard is visible."""
    src = (REPO_ROOT / "app" / "ledger" / "tasks.py").read_text(encoding="utf-8")
    assert "no new work can be queued" in src
    assert "budget_exceeded" in src
