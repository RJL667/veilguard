"""Tests for ChatAgent.prepare_session — preamble + tool pinning.

These prove the pin hook fires when expected, skips when expected
(side-channel turns), and threads the right content to TCMM.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


# ── Fixtures ────────────────────────────────────────────────────────────


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
def fake_tcmm_adapter(monkeypatch, tmp_path):
    """Stub the TCMM AnthropicGenerationAdapter import."""
    fake_root = tmp_path / "fake_tcmm"
    fake_root.mkdir()
    (fake_root / "adapters").mkdir()
    (fake_root / "adapters" / "__init__.py").write_text("")
    (fake_root / "adapters" / "anthropic_adapter.py").write_text(
        """
class AnthropicGenerationAdapter:
    def __init__(self, **kw):
        self.last_usage = {'input_tokens': 10, 'output_tokens': 5,
                           'cache_creation_input_tokens': 0,
                           'cache_read_input_tokens': 0}
        self.last_response_blocks = [{'type': 'text', 'text': 'ok'}]
        self.last_stop_reason = 'end_turn'
        self._mode = 'Fake'
    def generate(self, prompt, label=None):
        return 'ok'
"""
    )
    monkeypatch.setenv("TCMM_ROOT", str(fake_root))
    import llm.adapter as adapter_mod
    adapter_mod._ADAPTER_CLS = None
    yield fake_root
    fake_root_str = str(fake_root)
    if fake_root_str in sys.path:
        sys.path.remove(fake_root_str)
    adapter_mod._ADAPTER_CLS = None
    for k in [k for k in list(sys.modules) if k.startswith("adapters")]:
        del sys.modules[k]


@pytest.fixture
def stub_tcmm_http(monkeypatch):
    """Capture all TCMM HTTP calls so we can assert pin invocations."""
    import llm.tcmm_client as tcmm_mod
    import httpx

    state = {"calls": []}

    def handler(req: httpx.Request) -> httpx.Response:
        import json
        body_bytes = req.read()
        state["calls"].append({
            "url": str(req.url),
            "path": req.url.path,
            "body": json.loads(body_bytes.decode()) if body_bytes else {},
        })
        path = req.url.path
        if path.endswith("/render_structured"):
            return httpx.Response(200, json={
                "status": "ok",
                "prompt": "<sys>",
                "blocks": [{"type": "text", "text": "memory"}],
                "tier_summary": {},
                "stats": {},
                "layout": {},
            })
        if path.endswith("/pin/system_prompt"):
            return httpx.Response(200, json={"status": "ok"})
        if path.endswith("/pin/tool_definitions"):
            return httpx.Response(200, json={"status": "ok"})
        if path.endswith("/pre_request") or path.endswith("/post_response"):
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, timeout=10.0)
    tcmm_mod._CLIENT = client

    yield state

    asyncio.run(client.aclose())
    tcmm_mod._CLIENT = None


# ── Tests ───────────────────────────────────────────────────────────────


def test_prepare_session_pins_preamble(fake_tcmm_adapter, stub_tcmm_http):
    """ChatAgent with veilguard_preamble pins it via TCMM."""
    from agent import ChatAgent, TurnContext

    chat = ChatAgent(
        model="claude-sonnet-4-5",
        veilguard_preamble="You are a Claude agent...\n[Veilguard preamble]",
    )
    ctx = TurnContext(
        conversation_id="conv-pin-1", user_id="u-1", tenant_id="t-1",
    )

    asyncio.run(_drain(chat, [{"role": "user", "content": "Hi"}], ctx))

    pin_calls = [
        c for c in stub_tcmm_http["calls"]
        if c["path"].endswith("/pin/system_prompt")
    ]
    assert len(pin_calls) >= 1
    # Verify the preamble made it through
    bodies = [c["body"].get("content", "") for c in pin_calls]
    assert any("Veilguard preamble" in b for b in bodies)
    # And the kind tag
    kinds = [c["body"].get("kind") for c in pin_calls]
    assert "veilguard_preamble" in kinds


def test_prepare_session_pins_client_system_and_tools(
    fake_tcmm_adapter, stub_tcmm_http,
):
    from agent import ChatAgent, TurnContext

    tools = [
        {"name": "spawn_subagent", "description": "...",
         "input_schema": {"type": "object"}},
    ]
    chat = ChatAgent(
        model="claude-sonnet-4-5",
        client_tools=tools,
        veilguard_preamble="VG preamble",
        client_system_text="You are a helpful assistant.",
    )
    ctx = TurnContext(
        conversation_id="conv-pin-2", user_id="u-1", tenant_id="t-1",
    )

    asyncio.run(_drain(chat, [{"role": "user", "content": "Hi"}], ctx))

    # Two system_prompt pins (preamble + client_system)
    sys_pins = [
        c for c in stub_tcmm_http["calls"]
        if c["path"].endswith("/pin/system_prompt")
    ]
    assert len(sys_pins) == 2
    kinds = sorted(c["body"].get("kind") for c in sys_pins)
    assert kinds == ["client_system", "veilguard_preamble"]

    # One tool_definitions pin with the tool list
    tool_pins = [
        c for c in stub_tcmm_http["calls"]
        if c["path"].endswith("/pin/tool_definitions")
    ]
    assert len(tool_pins) == 1
    assert tool_pins[0]["body"]["tools"] == tools


def test_prepare_session_skipped_on_side_channel(
    fake_tcmm_adapter, stub_tcmm_http,
):
    """Side-channel turns (title-gen) skip TCMM entirely incl. pinning."""
    from agent import ChatAgent, TurnContext

    chat = ChatAgent(
        model="claude-haiku-4-5",
        veilguard_preamble="VG preamble",
        side_channel=True,
    )
    ctx = TurnContext(
        conversation_id="conv-title-gen", user_id="u-1", tenant_id="t-1",
    )

    asyncio.run(_drain(chat, [{"role": "user",
                                "content": "Provide a 5-word title..."}], ctx))

    pin_calls = [
        c for c in stub_tcmm_http["calls"]
        if "/pin/" in c["path"]
    ]
    assert len(pin_calls) == 0, (
        f"side-channel turn made {len(pin_calls)} pin calls, expected 0"
    )


def test_base_agent_prepare_session_is_noop(fake_tcmm_adapter, stub_tcmm_http):
    """Non-ChatAgent subclasses don't pin (no override of prepare_session)."""
    from agent import DirectorAgent, TurnContext
    from agent.persona import PersonaSpec

    persona = PersonaSpec(
        agent_id="director", role="director",
        manager_id=None, team_id=None,
        model="claude-sonnet-4-5",
        model_map={"reactive": "claude-sonnet-4-5"},
        tools=[],
        system_prompt="You are Director.",
    )
    director = DirectorAgent(persona)
    ctx = TurnContext(
        conversation_id="conv-dir", user_id="u-1", tenant_id="t-1",
    )

    asyncio.run(_drain(director, [{"role": "user", "content": "Plan."}], ctx))

    pin_calls = [
        c for c in stub_tcmm_http["calls"]
        if "/pin/" in c["path"]
    ]
    assert len(pin_calls) == 0


def test_from_libre_chat_request_threads_preamble_through(fake_tcmm_adapter):
    """from_libre_chat_request passes veilguard_preamble to constructor."""
    from agent import ChatAgent

    body = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    chat = ChatAgent.from_libre_chat_request(
        body, veilguard_preamble="VG preamble text",
    )
    assert chat._veilguard_preamble == "VG preamble text"


# ── Helpers ─────────────────────────────────────────────────────────────


async def _drain(agent, messages, ctx):
    async for _ in agent.run_turn(messages, ctx):
        pass
