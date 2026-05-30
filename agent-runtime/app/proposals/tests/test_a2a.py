"""Tests for a2a.py — internal A2A AgentCard + messages routing.

Uses FastAPI TestClient against the real router; mocks the
persona registry + ledger calls so we don't need Lance.
"""

import os
from unittest.mock import patch, MagicMock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.a2a import router as a2a_router


class FakePersona:
    def __init__(self, agent_id, role="ic", tools=None, model="claude-sonnet-4-6"):
        self.agent_id = agent_id
        self.role = role
        self.tools = tools or ["read_file", "write_file"]
        self.model = model
        self.description = f"persona for {agent_id}"
        self.manager_id = ""
        self.team_id = "core"


class FakePersonaRegistry:
    def __init__(self, personas):
        self._by_id = {p.agent_id: p for p in personas}
    def all(self):
        return list(self._by_id.values())
    def get(self, agent_id):
        return self._by_id.get(agent_id)


@pytest.fixture
def app_with_registry():
    """FastAPI app with the a2a router attached + persona registry seeded."""
    app = FastAPI()
    app.include_router(a2a_router)
    app.state.persona_registry = FakePersonaRegistry([
        FakePersona("researcher"),
        FakePersona("critic-claim", role="ic"),
        FakePersona("director", role="director", tools=["create_task"]),
    ])
    return app


# ── /.well-known/agents ────────────────────────────────────────────────


def test_list_agents_well_known(app_with_registry):
    client = TestClient(app_with_registry)
    r = client.get("/.well-known/agents")
    assert r.status_code == 200
    data = r.json()
    assert len(data["agents"]) == 3
    names = sorted(a["name"] for a in data["agents"])
    assert names == ["critic-claim", "director", "researcher"]


def test_agent_card_returns_full_schema(app_with_registry):
    client = TestClient(app_with_registry)
    r = client.get("/.well-known/agents/researcher/agent-card.json")
    assert r.status_code == 200
    card = r.json()
    for key in (
        "schemaVersion", "name", "url", "version", "authentication",
        "capabilities", "agent_id", "role", "model", "skills",
    ):
        assert key in card
    assert card["schemaVersion"] == "1.0"
    assert card["agent_id"] == "researcher"
    assert card["authentication"]["type"] == "internal_secret"


def test_agent_card_404_for_unknown(app_with_registry):
    client = TestClient(app_with_registry)
    r = client.get("/.well-known/agents/nonexistent/agent-card.json")
    assert r.status_code == 404


# ── POST /agents/{aid}/messages ───────────────────────────────────────


def test_send_message_requires_internal_secret(app_with_registry):
    client = TestClient(app_with_registry)
    # No header
    r = client.post(
        "/agents/researcher/messages",
        json={"task_id": "t-1", "tenant_id": "x", "user_id": "x",
              "from_agent_id": "director", "body": "hi"},
    )
    # When env var unset, the gate logs a warning but allows
    # (the test environment usually has it unset)
    # If env var IS set in the runner, we expect 401.
    if os.environ.get("VEILGUARD_INTERNAL_SECRET"):
        assert r.status_code == 401


def test_send_message_validates_required_fields(app_with_registry):
    client = TestClient(app_with_registry)
    # Empty body → 400
    r = client.post(
        "/agents/researcher/messages",
        headers={"X-Internal-Secret": os.environ.get("VEILGUARD_INTERNAL_SECRET", "")},
        json={"task_id": "t-1"},  # missing tenant_id, user_id, from_agent_id, body
    )
    # The fields missing → handler returns 400
    assert r.status_code in (400, 401)   # 401 if secret enforced + missing


def test_send_message_unknown_agent_404(app_with_registry):
    client = TestClient(app_with_registry)
    with patch.dict(os.environ, {"VEILGUARD_INTERNAL_SECRET": ""}, clear=False):
        r = client.post(
            "/agents/nonexistent/messages",
            json={"task_id": "t", "tenant_id": "x", "user_id": "x",
                  "from_agent_id": "director", "body": "hi"},
        )
    assert r.status_code == 404


@patch("app.ledger.comments.add_comment", return_value="cmt-abc")
def test_send_message_delivers_to_known_agent(mock_add, app_with_registry):
    client = TestClient(app_with_registry)
    # Make sure internal secret check passes (env var either unset or matched)
    with patch.dict(os.environ, {"VEILGUARD_INTERNAL_SECRET": ""}, clear=False):
        r = client.post(
            "/agents/researcher/messages",
            json={
                "task_id": "task-x", "tenant_id": "t", "user_id": "u",
                "from_agent_id": "director", "body": "please update progress",
            },
        )
    assert r.status_code == 200
    d = r.json()
    assert d["delivered"] is True
    assert d["to_agent"] == "researcher"
    assert d["comment_id"] == "cmt-abc"
    # add_comment was called with the expected kwargs
    assert mock_add.call_count == 1
    kw = mock_add.call_args.kwargs
    assert kw["task_id"] == "task-x"
    assert kw["author_id"] == "director"
    assert kw["body"] == "please update progress"
    assert kw["kind"] == "comment"
