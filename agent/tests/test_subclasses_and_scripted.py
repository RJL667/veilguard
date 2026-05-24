"""Verify subclass dispatch + ScriptedAdapter through Agent.run_turn.

This is the bridge between the unified Agent pipeline (PR #3) and the
existing 4 scripted demos (Pattern A/B/C + Critic-iterate).  When
agent-runtime gets rewired in PR #4's runtime.py change, the demos
will instantiate DirectorAgent / ICAgent / etc. with ScriptedAdapter
and run through Agent.run_turn — same pipeline as production.

Tests prove:
  - agent_for(persona) picks the right subclass for each role
  - critic-claim and critic-prose get their dedicated classes (not ICAgent)
  - ScriptedAdapter injected into Agent.run_turn produces the canned
    output through the full pipeline (TCMM render + redact + adapter +
    rehydrate + ingest)
  - Turn counter increments per persona across multiple invocations
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


# Reuse fixtures from test_chat_agent
pytest_plugins = []


@pytest.fixture(autouse=True)
def isolated_pii_db(tmp_path, monkeypatch):
    db_dir = tmp_path / "pii_db"
    db_dir.mkdir()
    monkeypatch.setenv("VEILGUARD_PII_DB_PATH", str(db_dir))

    from pii import session_store, redactor as redactor_mod
    session_store.PIISessionStore._instance = None
    redactor_mod.PIIRedactor._instance = None
    yield
    session_store.PIISessionStore._instance = None
    redactor_mod.PIIRedactor._instance = None


@pytest.fixture
def stub_tcmm_http(monkeypatch):
    """Mock TCMM HTTP calls so demos don't need a running tcmm-service."""
    import llm.tcmm_client as tcmm_mod
    import httpx

    state = {
        "render_response": {
            "status": "ok",
            "format": "anthropic-structured",
            "prompt": "<sys>",
            "blocks": [
                {"type": "text", "text": "Static preamble.",
                 "cache_control": {"type": "ephemeral"}},
            ],
            "tier_summary": {},
            "stats": {},
            "layout": {},
        },
        "calls": [],
    }

    def handler(req: httpx.Request) -> httpx.Response:
        import json
        body_bytes = req.read()
        state["calls"].append({
            "url": str(req.url),
            "body": json.loads(body_bytes.decode()) if body_bytes else {},
        })
        path = req.url.path
        if path.endswith("/render_structured"):
            return httpx.Response(200, json=state["render_response"])
        if path.endswith("/pre_request") or path.endswith("/post_response"):
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, timeout=10.0)
    tcmm_mod._CLIENT = client

    yield state

    asyncio.run(client.aclose())
    tcmm_mod._CLIENT = None


# ── Helpers ─────────────────────────────────────────────────────────────


def _mk_persona(agent_id, role, model="claude-sonnet-4-5", tools=None):
    from agent.persona import PersonaSpec
    return PersonaSpec(
        agent_id=agent_id,
        role=role,
        manager_id=None,
        team_id=None,
        model=model,
        model_map={"reactive": model},
        tools=tools or [],
        system_prompt=f"You are {agent_id}.",
    )


# ── Subclass dispatch ───────────────────────────────────────────────────


def test_agent_for_director():
    from agent import agent_for, DirectorAgent
    persona = _mk_persona("director", "director")
    assert agent_for(persona) is DirectorAgent


def test_agent_for_researcher_is_ic():
    from agent import agent_for, ICAgent
    persona = _mk_persona("researcher", "ic")
    assert agent_for(persona) is ICAgent


def test_agent_for_builder_is_ic():
    from agent import agent_for, ICAgent
    persona = _mk_persona("builder", "ic")
    assert agent_for(persona) is ICAgent


def test_agent_for_critic_claim_is_critic_claim():
    """critic-claim's persona role is 'ic' but persona_id overrides it."""
    from agent import agent_for, CriticClaimAgent
    persona = _mk_persona("critic-claim", "ic", model="claude-haiku-4-5")
    assert agent_for(persona) is CriticClaimAgent


def test_agent_for_critic_prose_is_critic_prose():
    from agent import agent_for, CriticProseAgent
    persona = _mk_persona("critic-prose", "ic")
    assert agent_for(persona) is CriticProseAgent


def test_agent_for_consultant():
    from agent import agent_for, ConsultantAgent
    persona = _mk_persona("phishing-analyst", "consultant")
    assert agent_for(persona) is ConsultantAgent


def test_agent_for_unknown_role_falls_back_to_ic():
    from agent import agent_for, ICAgent
    persona = _mk_persona("mystery", "wat")
    assert agent_for(persona) is ICAgent


# ── ScriptedAdapter through the pipeline ────────────────────────────────


def test_scripted_adapter_drives_director_pipeline(stub_tcmm_http):
    """DirectorAgent with ScriptedAdapter runs the full pipeline."""
    from agent import DirectorAgent, TurnContext
    from llm import ScriptedAdapter, Turn, ToolCall, set_script, clear_script

    set_script({
        "director": [
            Turn(
                text="Acknowledged.  Creating a task.",
                tool_calls=[
                    ToolCall(name="create_task", input={
                        "owner_id": "researcher",
                        "brief": "investigate X",
                        "deliverable_spec": "memo.md",
                    }),
                ],
            ),
        ],
    })

    try:
        persona = _mk_persona("director", "director",
                               tools=["create_task"])
        director = DirectorAgent(persona, adapter_cls=ScriptedAdapter)
        ctx = TurnContext(
            conversation_id="conv-1", user_id="u-1", tenant_id="t-1",
        )

        async def run():
            events = []
            async for ev in director.run_turn(
                [{"role": "user", "content": "Take this on."}], ctx,
            ):
                events.append(ev)
            return events

        events = asyncio.run(run())

        # Should have run the full pipeline
        types = [e["type"] for e in events]
        assert "run_start" in types
        assert "assistant" in types
        assert "tool_call" in types
        assert "final_result" in types
        assert "run_end" in types

        # Tool call surfaces the scripted input
        tool_calls = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "create_task"
        assert tool_calls[0]["input"]["owner_id"] == "researcher"

        # Pipeline hit TCMM (render + ingest)
        paths = [c["url"].rsplit("/", 1)[-1] for c in stub_tcmm_http["calls"]]
        assert "render_structured" in paths
        assert "pre_request" in paths
        assert "post_response" in paths
    finally:
        clear_script()


def test_scripted_adapter_advances_turn_counter(stub_tcmm_http):
    """Each invocation of the SAME persona advances its turn index."""
    from agent import DirectorAgent, TurnContext
    from llm import (
        ScriptedAdapter, Turn, set_script, clear_script, get_turn_counter,
    )

    set_script({
        "director": [
            Turn(text="Turn 1 response.", stop_reason="end_turn"),
            Turn(text="Turn 2 response.", stop_reason="end_turn"),
        ],
    })

    try:
        persona = _mk_persona("director", "director")
        ctx = TurnContext(
            conversation_id="conv-2", user_id="u-1", tenant_id="t-1",
        )

        async def run_once(msg):
            director = DirectorAgent(persona, adapter_cls=ScriptedAdapter)
            async for _ in director.run_turn(
                [{"role": "user", "content": msg}], ctx,
            ):
                pass

        assert get_turn_counter("director") == 0
        asyncio.run(run_once("first"))
        assert get_turn_counter("director") == 1
        asyncio.run(run_once("second"))
        assert get_turn_counter("director") == 2
    finally:
        clear_script()


def test_scripted_runs_end_to_end_pii_safe(stub_tcmm_http):
    """Even scripted demos go through PII redaction.

    Doesn't matter that the LLM is fake — if PII appears in the user
    message, the rendered prefix to the (fake) adapter must still be
    redacted.  Tests that the pipeline doesn't short-circuit redaction
    when ScriptedAdapter is injected.
    """
    from agent import ICAgent, TurnContext
    from llm import ScriptedAdapter, Turn, set_script, clear_script
    from llm import scripted_adapter as sa_mod

    # Set the TCMM render to inject PII into the system blocks
    stub_tcmm_http["render_response"]["blocks"] = [
        {"type": "text",
         "text": "Past memory: Alice Johnson is the CFO.",
         "cache_control": {"type": "ephemeral"}},
    ]

    set_script({
        "researcher": [Turn(text="Done.", stop_reason="end_turn")],
    })

    # Patch ScriptedAdapter to capture what the system_blocks looked like
    captured: dict = {}
    real_init = ScriptedAdapter.__init__

    def capturing_init(self, **kw):
        captured["system_blocks"] = kw.get("system_blocks")
        real_init(self, **kw)

    try:
        ScriptedAdapter.__init__ = capturing_init  # type: ignore

        persona = _mk_persona("researcher", "ic")
        agent = ICAgent(persona, adapter_cls=ScriptedAdapter)
        ctx = TurnContext(
            conversation_id="conv-3", user_id="u-1", tenant_id="t-1",
        )

        async def run():
            async for _ in agent.run_turn(
                [{"role": "user", "content": "What about Bob Smith?"}], ctx,
            ):
                pass

        asyncio.run(run())

        # Verify the redactor processed the rendered blocks before
        # ScriptedAdapter saw them (just like AnthropicAdapter would).
        blocks = captured.get("system_blocks") or []
        text = "\n".join(b.get("text", "") for b in blocks)
        assert "Alice Johnson" not in text, (
            "PII leaked into scripted adapter — pipeline short-circuited "
            "redaction"
        )
        assert "REF_PERSON_" in text, (
            "Expected REF token in redacted system_blocks; got: "
            f"{text!r}"
        )
    finally:
        ScriptedAdapter.__init__ = real_init  # type: ignore
        clear_script()
