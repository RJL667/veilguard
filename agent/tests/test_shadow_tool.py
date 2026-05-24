"""Tests for the tcmm_record_turn shadow tool.

Verifies:
  - schema is well-formed (Anthropic input_schema shape)
  - inject_into_tools is idempotent and prepends to user tools
  - intercept_response strips the shadow tool, returns flag_obj
  - stop_reason downgrades "tool_use" → "end_turn" when shadow was
    the only tool emitted
  - ChatAgent.prepare_tools + intercept_response wire it into the
    Agent pipeline (the model "sees" the shadow tool, the user does not)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


# ── Fixtures (shared with other agent tests) ────────────────────────────


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
    fake_root = tmp_path / "fake_tcmm"
    fake_root.mkdir()
    (fake_root / "adapters").mkdir()
    (fake_root / "adapters" / "__init__.py").write_text("")
    (fake_root / "adapters" / "anthropic_adapter.py").write_text(
        """
_CAPTURED = {}

class AnthropicGenerationAdapter:
    def __init__(self, *, api_key='', model_name='', system_prompt=None,
                 system_blocks=None, tools=None):
        _CAPTURED['init'] = {'tools': tools}
        # Default: model emits text + the shadow tool_use as last block
        self.last_usage = {'input_tokens': 10, 'output_tokens': 5,
                           'cache_creation_input_tokens': 0,
                           'cache_read_input_tokens': 0}
        self.last_response_blocks = [
            {'type': 'text', 'text': 'Answer.'},
            {'type': 'tool_use', 'id': 'tu_shadow', 'name': 'tcmm_record_turn',
             'input': {
                 'knowledge_class': 'derived',
                 'used': {'7': 0.8, '12': 0.5},
                 'emit_class': 'factoid',
             }},
        ]
        self.last_stop_reason = 'tool_use'
        self._mode = 'Fake'

    def generate(self, prompt, label=None):
        return 'Answer.'

def get_captured():
    return _CAPTURED
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
    import llm.tcmm_client as tcmm_mod
    import httpx

    state = {"calls": []}

    def handler(req: httpx.Request) -> httpx.Response:
        import json
        body_bytes = req.read()
        state["calls"].append({
            "path": req.url.path,
            "body": json.loads(body_bytes.decode()) if body_bytes else {},
        })
        path = req.url.path
        if path.endswith("/render_structured"):
            return httpx.Response(200, json={
                "status": "ok", "prompt": "<sys>",
                "blocks": [{"type": "text", "text": "memory"}],
                "tier_summary": {}, "stats": {}, "layout": {},
            })
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, timeout=10.0)
    tcmm_mod._CLIENT = client
    yield state
    asyncio.run(client.aclose())
    tcmm_mod._CLIENT = None


# ── Schema + helpers ────────────────────────────────────────────────────


def test_schema_shape():
    from agent.shadow_tool import TCMM_RECORD_TURN_TOOL as T
    assert T["name"] == "tcmm_record_turn"
    schema = T["input_schema"]
    assert schema["type"] == "object"
    props = schema["properties"]
    assert "knowledge_class" in props
    assert "used" in props
    assert "epoch_complete" in props
    assert "emit_class" in props
    assert set(schema["required"]) == {"knowledge_class", "used", "emit_class"}


def test_inject_prepends_to_tools():
    from agent.shadow_tool import inject_into_tools
    user_tools = [{"name": "search", "description": "x", "input_schema": {}}]
    out = inject_into_tools(user_tools)
    assert len(out) == 2
    assert out[0]["name"] == "tcmm_record_turn"
    assert out[1]["name"] == "search"


def test_inject_handles_empty_or_none():
    from agent.shadow_tool import inject_into_tools
    out = inject_into_tools([])
    assert len(out) == 1 and out[0]["name"] == "tcmm_record_turn"
    out = inject_into_tools(None)
    assert len(out) == 1 and out[0]["name"] == "tcmm_record_turn"


def test_inject_is_idempotent():
    from agent.shadow_tool import inject_into_tools, TCMM_RECORD_TURN_TOOL
    once = inject_into_tools([{"name": "x", "description": "", "input_schema": {}}])
    twice = inject_into_tools(once)
    assert once == twice  # no duplicate


# ── intercept_response ──────────────────────────────────────────────────


def test_intercept_strips_shadow_block_and_captures_flag():
    from agent.shadow_tool import intercept_response
    blocks = [
        {"type": "text", "text": "Hello."},
        {"type": "tool_use", "id": "tu_1", "name": "tcmm_record_turn",
         "input": {"knowledge_class": "derived", "used": {},
                   "emit_class": "small_talk"}},
    ]
    cleaned, flag_obj, stop = intercept_response(blocks, "tool_use")
    # Shadow block removed
    assert len(cleaned) == 1
    assert cleaned[0]["type"] == "text"
    # Flag obj captured
    assert flag_obj["knowledge_class"] == "derived"
    assert flag_obj["emit_class"] == "small_talk"
    # Stop reason downgraded — shadow was the only tool_use
    assert stop == "end_turn"


def test_intercept_keeps_real_tool_uses():
    """When real tools and shadow co-exist, real tools survive and
    stop_reason stays 'tool_use'."""
    from agent.shadow_tool import intercept_response
    blocks = [
        {"type": "text", "text": "Searching..."},
        {"type": "tool_use", "id": "tu_real", "name": "search",
         "input": {"q": "x"}},
        {"type": "tool_use", "id": "tu_shadow", "name": "tcmm_record_turn",
         "input": {"knowledge_class": "mixed", "used": {"3": 0.5},
                   "emit_class": "decision"}},
    ]
    cleaned, flag_obj, stop = intercept_response(blocks, "tool_use")
    assert len(cleaned) == 2
    assert any(b.get("name") == "search" for b in cleaned)
    assert not any(b.get("name") == "tcmm_record_turn" for b in cleaned)
    assert flag_obj["emit_class"] == "decision"
    # Real tool_use remains → stop stays
    assert stop == "tool_use"


def test_intercept_no_shadow_returns_empty_flag():
    from agent.shadow_tool import intercept_response
    blocks = [{"type": "text", "text": "Just text."}]
    cleaned, flag_obj, stop = intercept_response(blocks, "end_turn")
    assert cleaned == blocks
    assert flag_obj == {}
    assert stop == "end_turn"


# ── End-to-end through ChatAgent.run_turn ───────────────────────────────


def test_chat_agent_injects_and_intercepts(
    fake_tcmm_adapter, stub_tcmm_http,
):
    """ChatAgent.prepare_tools injects shadow tool; intercept_response
    strips it before user sees the response."""
    from agent import ChatAgent, TurnContext

    chat = ChatAgent(model="claude-sonnet-4-5", client_tools=[])
    ctx = TurnContext(conversation_id="c", user_id="u", tenant_id="t")

    async def run():
        events = []
        async for ev in chat.run_turn(
            [{"role": "user", "content": "Hi"}], ctx,
        ):
            events.append(ev)
        return events

    events = asyncio.run(run())

    # Adapter received the shadow tool in its tools list
    import importlib
    mod = importlib.import_module("adapters.anthropic_adapter")
    init_tools = mod.get_captured()["init"]["tools"] or []
    assert any(t.get("name") == "tcmm_record_turn" for t in init_tools), (
        "shadow tool not injected into adapter call"
    )

    # User-facing assistant message DOES NOT contain the shadow tool
    assistant_evts = [e for e in events if e["type"] == "assistant"]
    assert assistant_evts
    content = assistant_evts[0]["message"]["content"]
    assert not any(
        b.get("name") == "tcmm_record_turn" for b in content
    ), "shadow tool leaked into user-facing response"
    # Text survives
    assert any(b.get("type") == "text" for b in content)

    # stop_reason in the user-facing event was downgraded
    assert assistant_evts[0]["message"]["stop_reason"] == "end_turn"


def test_chat_agent_passes_flag_obj_to_ingest(
    fake_tcmm_adapter, stub_tcmm_http,
):
    """The captured shadow-tool input gets forwarded to TCMM
    /post_response as flag_obj — that's how block_class lands in
    the archive."""
    from agent import ChatAgent, TurnContext

    chat = ChatAgent(model="claude-sonnet-4-5", client_tools=[])
    ctx = TurnContext(conversation_id="c2", user_id="u", tenant_id="t")

    async def run():
        async for _ in chat.run_turn(
            [{"role": "user", "content": "Hi"}], ctx,
        ):
            pass

    asyncio.run(run())

    post_response_calls = [
        c for c in stub_tcmm_http["calls"]
        if c["path"].endswith("/post_response")
    ]
    assert post_response_calls, "no /post_response call made"
    flag = post_response_calls[0]["body"].get("flag_obj")
    assert flag is not None, "flag_obj not forwarded to TCMM"
    assert flag["knowledge_class"] == "derived"
    assert flag["used"] == {"7": 0.8, "12": 0.5}
    assert flag["emit_class"] == "factoid"


def test_director_does_not_inject_shadow(fake_tcmm_adapter, stub_tcmm_http):
    """Non-chat agents (Director/IC/etc.) don't use the shadow tool —
    their tool_use blocks are real and go through tool_dispatcher.
    """
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
    ctx = TurnContext(conversation_id="cd", user_id="u", tenant_id="t")

    async def run():
        async for _ in director.run_turn(
            [{"role": "user", "content": "Plan."}], ctx,
        ):
            pass

    asyncio.run(run())

    import importlib
    mod = importlib.import_module("adapters.anthropic_adapter")
    init_tools = mod.get_captured()["init"]["tools"] or []
    assert not any(
        t.get("name") == "tcmm_record_turn" for t in init_tools
    ), "shadow tool incorrectly injected for Director"
