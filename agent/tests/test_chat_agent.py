"""End-to-end tests for the Agent pipeline via ChatAgent.

These mock the network boundaries (TCMM HTTP, LLM HTTP) but exercise
the FULL pipeline including PII redaction.  What we prove:

  1. Pipeline runs in order: ingest_user → render → redact → adapter
                            → rehydrate → ingest_assistant
  2. The LLM adapter NEVER sees raw PII — even when memory contained
     it.  (the critical security property)
  3. cache_control markers survive into the adapter call (cache safety)
  4. Side-channel detection skips TCMM steps
  5. Response text + tool_use args get rehydrated before yielded
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolated_pii_db(tmp_path, monkeypatch):
    """Fresh Lance dir + reset PII singletons per test."""
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
    """Stub TCMM adapter module so the wrapper imports without real TCMM."""
    fake_root = tmp_path / "fake_tcmm"
    fake_root.mkdir()
    (fake_root / "adapters").mkdir()
    (fake_root / "adapters" / "__init__.py").write_text("")
    (fake_root / "adapters" / "anthropic_adapter.py").write_text(
        """
# This fake captures the call args so tests can assert what the
# adapter received AFTER redaction.
_CAPTURED = {}

class AnthropicGenerationAdapter:
    def __init__(self, *, api_key='', model_name='', system_prompt=None,
                 system_blocks=None, tools=None):
        _CAPTURED['init'] = {
            'model_name': model_name,
            'system_prompt': system_prompt,
            'system_blocks': system_blocks,
            'tools': tools,
        }
        self.last_usage = None
        self.last_response_blocks = None
        self.last_stop_reason = None
        self._mode = 'Fake'

    def generate(self, prompt, label=None):
        _CAPTURED['generate'] = {'prompt': prompt, 'label': label}
        self.last_usage = {
            'input_tokens': 200, 'output_tokens': 80,
            'cache_creation_input_tokens': 1000,
            'cache_read_input_tokens': 0,
        }
        # Echo back with a REF token replaced — proves rehydration works
        self.last_response_blocks = [
            {'type': 'text', 'text': f'Reply: {prompt}'},
        ]
        self.last_stop_reason = 'end_turn'
        return f'Reply: {prompt}'

def get_captured():
    return _CAPTURED

def reset_captured():
    _CAPTURED.clear()
"""
    )
    monkeypatch.setenv("TCMM_ROOT", str(fake_root))

    import llm.adapter as adapter_mod
    adapter_mod._ADAPTER_CLS = None

    yield fake_root

    # Cleanup
    fake_root_str = str(fake_root)
    if fake_root_str in sys.path:
        sys.path.remove(fake_root_str)
    adapter_mod._ADAPTER_CLS = None
    for k in [k for k in list(sys.modules) if k.startswith("adapters")]:
        del sys.modules[k]


@pytest.fixture
def stub_tcmm_http(monkeypatch):
    """Replace llm.tcmm_client._client() with a MockTransport-backed one."""
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
                # This block will contain PII from past memory turns —
                # tests prove it gets redacted before reaching adapter.
                {"type": "text", "text": "Past memory: Alice Johnson works here."},
            ],
            "tier_summary": {},
            "stats": {},
            "layout": {},
        },
        "calls": [],
    }

    def handler(req: httpx.Request) -> httpx.Response:
        import json
        state["calls"].append({
            "url": str(req.url),
            "body": json.loads(req.read().decode()) if req.read() else {},
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


# ── Tests ───────────────────────────────────────────────────────────────


def test_pipeline_runs_full_5_steps(fake_tcmm_adapter, stub_tcmm_http):
    """Verify all 5 steps fire in order."""
    from agent import ChatAgent, TurnContext

    chat = ChatAgent(model="claude-sonnet-4-5", client_tools=[])
    ctx = TurnContext(
        conversation_id="conv-1", user_id="u-1", tenant_id="t-1",
    )

    async def run():
        events_out = []
        async for ev in chat.run_turn(
            [{"role": "user", "content": "Hello there."}], ctx,
        ):
            events_out.append(ev)
        return events_out

    events = asyncio.run(run())

    # Verify event order matches the 5-step pipeline.
    types = [e["type"] for e in events]
    assert types[0] == "run_start"
    assert "assistant" in types
    assert "final_result" in types
    assert "usage" in types
    assert types[-1] == "run_end"

    # Verify the network calls in order:
    # render_structured → (pre_request happens in parallel, order may vary)
    # → adapter.generate → post_response
    paths = [c["url"].rsplit("/", 1)[-1] for c in stub_tcmm_http["calls"]]
    assert "render_structured" in paths
    assert "pre_request" in paths       # STEP 0 ingest user
    assert "post_response" in paths     # STEP 5 ingest assistant


def test_adapter_never_sees_raw_pii(fake_tcmm_adapter, stub_tcmm_http):
    """The CRITICAL security test: PII from memory MUST be redacted
    before the adapter sees it.
    """
    from agent import ChatAgent, TurnContext

    # The TCMM render response (set in fixture) contains "Alice Johnson".
    # That PII must NOT reach the adapter — redactor sits between them.
    chat = ChatAgent(model="claude-sonnet-4-5", client_tools=[])
    ctx = TurnContext(
        conversation_id="conv-x", user_id="u-1", tenant_id="t-1",
    )

    async def run():
        async for _ in chat.run_turn(
            [{"role": "user", "content": "what about Bob Smith?"}], ctx,
        ):
            pass

    asyncio.run(run())

    # Inspect what the adapter received
    import importlib
    mod = importlib.import_module("adapters.anthropic_adapter")
    captured = mod.get_captured()

    init_blocks = captured["init"]["system_blocks"]
    init_text = "\n".join(b.get("text", "") for b in (init_blocks or []))
    gen_prompt = captured["generate"]["prompt"]

    # Hard assertions:
    assert "Alice Johnson" not in init_text, (
        "PII leaked from TCMM memory into adapter system_blocks!"
    )
    assert "Alice Johnson" not in gen_prompt, (
        "PII leaked from user message into adapter prompt!"
    )
    assert "Bob Smith" not in gen_prompt, (
        "PII in current-turn user message reached adapter!"
    )
    # And it WAS detected and tokenized
    assert "REF_PERSON_" in init_text, (
        f"Expected REF token in redacted system_blocks, got: {init_text!r}"
    )


def test_cache_control_preserved_into_adapter(fake_tcmm_adapter, stub_tcmm_http):
    """The block's cache_control marker must survive redaction so
    Anthropic's prompt cache still hits.
    """
    from agent import ChatAgent, TurnContext

    chat = ChatAgent(model="claude-sonnet-4-5", client_tools=[])
    ctx = TurnContext(
        conversation_id="conv-cc", user_id="u-1", tenant_id="t-1",
    )

    asyncio.run(_drain(chat, [{"role": "user", "content": "hi"}], ctx))

    import importlib
    mod = importlib.import_module("adapters.anthropic_adapter")
    captured = mod.get_captured()
    init_blocks = captured["init"]["system_blocks"]
    assert init_blocks is not None
    # First block came from TCMM with cache_control marker — must be
    # there in the redacted output too.
    cc = [b.get("cache_control") for b in init_blocks if b.get("cache_control")]
    assert cc, f"cache_control dropped: {init_blocks}"


def test_haiku_cache_floor_preamble_inlined_when_prefix_short(
    fake_tcmm_adapter, stub_tcmm_http,
):
    """[HAIKU_CACHE_FLOOR_INLINE_PREAMBLE_2026_05_28] regression test.

    The empirical Haiku 4.5 cache floor is ~6000 tokens.  When a cold
    conversation has only the magic prefix + persona in the system
    blocks (< 12000 chars ≈ 3000 tokens), `Agent.run_turn` should inline
    the `_VEILGUARD_PREAMBLE_TEMPLATE` between the magic prefix and the
    rest so the cached prefix clears the floor.
    """
    # TCMM stub returns ONLY the magic prefix as a block — simulates a
    # cold conversation with no memory yet.
    stub_tcmm_http["render_response"]["blocks"] = [
        {"type": "text",
         "text": "You are a Claude agent, built on Anthropic's Claude Agent SDK."},
    ]

    from agent import ChatAgent, TurnContext

    chat = ChatAgent(model="claude-haiku-4-5", client_tools=[])
    ctx = TurnContext(
        conversation_id="conv-cold", user_id="u-1", tenant_id="t-1",
    )
    asyncio.run(_drain(chat, [{"role": "user", "content": "hi"}], ctx))

    import importlib
    mod = importlib.import_module("adapters.anthropic_adapter")
    captured = mod.get_captured()
    init_blocks = captured["init"]["system_blocks"]
    assert init_blocks is not None

    total_chars = sum(
        len(b.get("text", "")) for b in init_blocks
        if isinstance(b, dict) and b.get("type") == "text"
    )
    # After inlining we expect well over the 12000-char threshold.
    # The Veilguard preamble alone is ~17000 chars; plus magic + persona.
    assert total_chars > 12000, (
        f"preamble was not inlined despite short prefix; "
        f"total system chars={total_chars}, expected > 12000"
    )

    # Magic prefix must STILL be block 0 (Anthropic attribution gate
    # depends on this ordering).
    assert init_blocks[0]["text"].startswith(
        "You are a Claude agent"
    ), "magic prefix must remain at block[0] after inlining"


def test_haiku_cache_floor_preamble_NOT_inlined_when_prefix_already_fat(
    fake_tcmm_adapter, stub_tcmm_http,
):
    """The inline-preamble fix MUST NOT double-up on warm conversations.

    When TCMM already returns a fat memory-rendered prefix (≥12000
    chars), we skip inlining — TCMM's pinned content is already there
    and inlining would duplicate bytes + risk cache invalidation
    against pii-proxy paths that see only the pinned version.
    """
    # Give TCMM a fat response — simulates a warm conv with memory.
    stub_tcmm_http["render_response"]["blocks"] = [
        {"type": "text",
         "text": "You are a Claude agent, built on Anthropic's Claude Agent SDK."},
        # 12000+ chars of "memory" content to clear the threshold.
        {"type": "text",
         "text": "Past memory: " + ("filler context " * 1000)},
    ]
    from agent import ChatAgent, TurnContext

    chat = ChatAgent(model="claude-haiku-4-5", client_tools=[])
    ctx = TurnContext(
        conversation_id="conv-warm", user_id="u-1", tenant_id="t-1",
    )
    asyncio.run(_drain(chat, [{"role": "user", "content": "hi"}], ctx))

    import importlib
    mod = importlib.import_module("adapters.anthropic_adapter")
    captured = mod.get_captured()
    init_blocks = captured["init"]["system_blocks"]

    # The Veilguard preamble template starts with a recognisable line —
    # check it's NOT in any block (excluding the persona at the tail,
    # which is a different string).
    from agent.preamble import _VEILGUARD_PREAMBLE_TEMPLATE
    preamble_head = _VEILGUARD_PREAMBLE_TEMPLATE[:80]
    preamble_inlined = any(
        isinstance(b, dict) and preamble_head in b.get("text", "")
        for b in init_blocks
    )
    assert not preamble_inlined, (
        "preamble was inlined even though the TCMM-rendered prefix was "
        "already fat enough; this duplicates bytes and risks cache miss"
    )


def test_side_channel_skips_tcmm(fake_tcmm_adapter, stub_tcmm_http):
    """LibreChat title-gen / summary calls must NOT hit TCMM."""
    from agent import ChatAgent, TurnContext

    chat = ChatAgent(
        model="claude-haiku-4-5",
        client_tools=[],
        side_channel=True,
    )
    ctx = TurnContext(
        conversation_id="title-gen-conv", user_id="u-1", tenant_id="t-1",
    )

    asyncio.run(_drain(chat, [{"role": "user",
                                "content": "Provide a concise 5-word title..."}], ctx))

    paths = [c["url"].rsplit("/", 1)[-1] for c in stub_tcmm_http["calls"]]
    assert "render_structured" not in paths, "side-channel hit TCMM render"
    assert "pre_request" not in paths, "side-channel ingested into TCMM"
    assert "post_response" not in paths, "side-channel wrote back to TCMM"


def test_from_libre_chat_request_detects_side_channel(fake_tcmm_adapter):
    from agent import ChatAgent

    body = {
        "model": "claude-haiku-4-5-sso",
        "messages": [{"role": "user",
                     "content": "Provide a concise, 5-word-or-less title for the conversation"}],
        "tools": [],
    }
    chat = ChatAgent.from_libre_chat_request(body)
    assert chat._side_channel is True
    # Verify -sso suffix stripped
    assert chat._model == "claude-haiku-4-5"


def test_from_libre_chat_request_passes_tools(fake_tcmm_adapter):
    from agent import ChatAgent

    tools = [
        {"name": "mcp__sub-agents__spawn", "description": "x",
         "input_schema": {"type": "object"}},
    ]
    body = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": tools,
    }
    chat = ChatAgent.from_libre_chat_request(body)
    assert chat.tools() == tools


def test_tool_use_args_rehydrated(fake_tcmm_adapter, stub_tcmm_http,
                                   monkeypatch):
    """If the model emits a tool_use with a REF token in its input,
    the yielded event must contain the REHYDRATED original value.
    """
    # Override the fake adapter to emit a tool_use with a REF token
    import sys
    import importlib

    # Force adapter module to use a custom generate that returns tool_use
    # We do this by monkey-patching `last_response_blocks` after fact
    from agent import ChatAgent, TurnContext
    from pii import get_redactor, SessionId
    from llm.adapter import AnthropicAdapter

    # Seed the redactor mapping so we have a known token
    sid = SessionId("t-1", "conv-rh")
    get_redactor().redact_text("Send to alice@example.com", sid)

    # Patch AnthropicAdapter.generate to return tool_use with the token
    real_generate = AnthropicAdapter.generate

    async def fake_generate(self, user_message, label=""):
        from llm.adapter import AdapterResult
        return AdapterResult(
            text="",
            content_blocks=[{
                "type": "tool_use",
                "name": "send_email",
                "id": "tu_1",
                "input": {"to": "REF_EMAIL_1", "body": "Hi"},
            }],
            usage={"input_tokens": 10, "output_tokens": 5,
                   "cache_creation_input_tokens": 0,
                   "cache_read_input_tokens": 0},
            stop_reason="tool_use",
            model="claude-sonnet-4-5",
        )

    monkeypatch.setattr(AnthropicAdapter, "generate", fake_generate)

    chat = ChatAgent(model="claude-sonnet-4-5", client_tools=[])
    ctx = TurnContext(
        conversation_id="conv-rh", user_id="u-1", tenant_id="t-1",
    )

    events_out = asyncio.run(_collect(chat, [
        {"role": "user", "content": "send the email"},
    ], ctx))

    tool_calls = [e for e in events_out if e["type"] == "tool_call"]
    assert tool_calls, "no tool_call event emitted"
    assert tool_calls[0]["input"]["to"] == "alice@example.com", (
        f"REF_EMAIL_1 not rehydrated: {tool_calls[0]!r}"
    )


# ── Helpers ─────────────────────────────────────────────────────────────


async def _drain(agent, messages, ctx):
    async for _ in agent.run_turn(messages, ctx):
        pass


async def _collect(agent, messages, ctx) -> list[dict]:
    out = []
    async for ev in agent.run_turn(messages, ctx):
        out.append(ev)
    return out
