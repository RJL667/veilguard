"""Tests for the new chat_agent_handler (PR #5a).

These don't need pii-proxy's full stack — just the handler module + a
mocked ChatAgent.  We assert:

  - Feature flag is_enabled() reads env correctly
  - JSON response shape matches Anthropic Messages API
  - SSE stream emits the right event sequence
  - Errors produce Anthropic-shaped error responses
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest


# Put repo root + pii-proxy on sys.path so the handler's `from agent
# import ...` works AND `from app.chat_agent_handler import ...` works
# regardless of whether pytest was launched from repo root or from
# agent-proxy/.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PROXY_ROOT = Path(__file__).resolve().parent.parent
for p in (_REPO_ROOT, _PROXY_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ── is_enabled() ────────────────────────────────────────────────────────


def test_is_enabled_off_by_default(monkeypatch):
    monkeypatch.delenv("VEILGUARD_USE_CHAT_AGENT", raising=False)
    from _chat_agent_handler_loader import is_enabled
    assert is_enabled() is False


def test_is_enabled_on_with_1(monkeypatch):
    monkeypatch.setenv("VEILGUARD_USE_CHAT_AGENT", "1")
    from _chat_agent_handler_loader import is_enabled
    assert is_enabled() is True


def test_is_enabled_accepts_true_yes(monkeypatch):
    from _chat_agent_handler_loader import is_enabled
    for val in ("true", "TRUE", "yes", "True"):
        monkeypatch.setenv("VEILGUARD_USE_CHAT_AGENT", val)
        assert is_enabled() is True


def test_is_enabled_off_with_zero(monkeypatch):
    monkeypatch.setenv("VEILGUARD_USE_CHAT_AGENT", "0")
    from _chat_agent_handler_loader import is_enabled
    assert is_enabled() is False


# ── Mock ChatAgent ──────────────────────────────────────────────────────


class _FakeChatAgent:
    """Stand-in for agent.ChatAgent that yields canned events."""
    def __init__(self, *, model="claude-sonnet-4-5", side_channel=False,
                 client_tools=None, error=None,
                 content=None, usage=None, stop_reason="end_turn"):
        self._model = model
        self._side_channel = side_channel
        self._client_tools = client_tools or []
        self._error = error
        self._content = content or [{"type": "text", "text": "Hi there."}]
        self._usage = usage or {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        self._stop_reason = stop_reason

    @classmethod
    def from_libre_chat_request(cls, body, **kw):
        return cls(model=body.get("model", "claude-sonnet-4-5"))

    async def run_turn(self, messages, ctx):
        if self._error:
            yield {"type": "error", "code": "api_error",
                   "message": self._error}
            return
        yield {"type": "run_start", "agent_id": "chat",
               "model": self._model, "backend": "test"}
        yield {
            "type": "assistant",
            "message": {
                "content": self._content,
                "usage": self._usage,
                "stop_reason": self._stop_reason,
            },
        }
        yield {"type": "final_result", "result": "text",
               "stop_reason": self._stop_reason}
        yield {"type": "run_end", "stop_reason": self._stop_reason}


@pytest.fixture
def patch_chat_agent(monkeypatch):
    """Replace `agent.ChatAgent` with our fake.  Yields a configurator."""
    import agent
    saved = agent.ChatAgent
    # Default: simple text response
    monkeypatch.setattr(agent, "ChatAgent", _FakeChatAgent)
    return _FakeChatAgent


# ── JSON response shape ─────────────────────────────────────────────────


def test_json_response_anthropic_shape(patch_chat_agent):
    from _chat_agent_handler_loader import handle_chat_request

    async def run():
        return await handle_chat_request(
            {"model": "claude-sonnet-4-5",
             "messages": [{"role": "user", "content": "Hi"}]},
            conversation_id="conv-1", user_id="u-1", tenant_id="t-1",
            is_stream=False,
        )

    resp = asyncio.run(run())
    body = json.loads(resp.body.decode())

    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "claude-sonnet-4-5"
    assert body["id"].startswith("msg_")
    assert body["stop_reason"] == "end_turn"
    assert body["stop_sequence"] is None
    assert body["content"] == [{"type": "text", "text": "Hi there."}]
    assert body["usage"]["input_tokens"] == 100
    assert body["usage"]["output_tokens"] == 20


def _make_fake_class(**defaults):
    """Build a fake ChatAgent class whose from_libre_chat_request returns
    a configured _FakeChatAgent instance."""
    class _Fake(_FakeChatAgent):
        @classmethod
        def from_libre_chat_request(cls, body, **kw):
            return _FakeChatAgent(**defaults)
    return _Fake


def test_json_response_propagates_tool_use(monkeypatch):
    import agent
    monkeypatch.setattr(agent, "ChatAgent", _make_fake_class(
        model="claude-sonnet-4-5",
        content=[
            {"type": "text", "text": "Creating a task."},
            {"type": "tool_use", "id": "tu_1", "name": "create_task",
             "input": {"owner_id": "researcher"}},
        ],
        stop_reason="tool_use",
    ))
    from _chat_agent_handler_loader import handle_chat_request

    async def run():
        return await handle_chat_request(
            {"model": "claude-sonnet-4-5",
             "messages": [{"role": "user", "content": "Take this on"}]},
            conversation_id="c", user_id="u", tenant_id="t",
            is_stream=False,
        )

    resp = asyncio.run(run())
    body = json.loads(resp.body.decode())
    assert body["stop_reason"] == "tool_use"
    assert len(body["content"]) == 2
    assert body["content"][1]["type"] == "tool_use"
    assert body["content"][1]["input"] == {"owner_id": "researcher"}


def test_json_response_error_is_anthropic_shaped(monkeypatch):
    import agent
    monkeypatch.setattr(agent, "ChatAgent", _make_fake_class(
        error="OAuth credentials expired",
    ))
    from _chat_agent_handler_loader import handle_chat_request

    async def run():
        return await handle_chat_request(
            {"model": "claude-sonnet-4-5",
             "messages": [{"role": "user", "content": "Hi"}]},
            conversation_id="c", user_id="u", tenant_id="t",
            is_stream=False,
        )

    resp = asyncio.run(run())
    assert resp.status_code == 500
    body = json.loads(resp.body.decode())
    assert body["type"] == "error"
    assert body["error"]["type"] == "api_error"
    assert "OAuth credentials expired" in body["error"]["message"]


# ── SSE stream shape ────────────────────────────────────────────────────


def test_sse_stream_emits_anthropic_events(patch_chat_agent):
    from _chat_agent_handler_loader import handle_chat_request

    async def run():
        resp = await handle_chat_request(
            {"model": "claude-sonnet-4-5",
             "messages": [{"role": "user", "content": "Hi"}],
             "stream": True},
            conversation_id="c", user_id="u", tenant_id="t",
            is_stream=True,
        )
        # StreamingResponse.body_iterator is the async generator
        chunks = []
        async for chunk in resp.body_iterator:
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")
            chunks.append(chunk)
        return "".join(chunks)

    raw = asyncio.run(run())

    # Verify the Anthropic SSE event sequence
    assert "event: message_start" in raw
    assert "event: content_block_start" in raw
    assert "event: content_block_delta" in raw
    assert "event: content_block_stop" in raw
    assert "event: message_delta" in raw
    assert "event: message_stop" in raw

    # The actual text reaches the client
    assert "Hi there." in raw


def test_sse_stream_error_emits_error_event(monkeypatch):
    import agent
    monkeypatch.setattr(agent, "ChatAgent", _make_fake_class(
        error="rate_limit_error",
    ))
    from _chat_agent_handler_loader import handle_chat_request

    async def run():
        resp = await handle_chat_request(
            {"model": "claude-sonnet-4-5",
             "messages": [{"role": "user", "content": "Hi"}]},
            conversation_id="c", user_id="u", tenant_id="t",
            is_stream=True,
        )
        chunks = []
        async for chunk in resp.body_iterator:
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")
            chunks.append(chunk)
        return "".join(chunks)

    raw = asyncio.run(run())
    assert "event: error" in raw
    assert "rate_limit_error" in raw
    # No message_start when we error early
    assert "event: message_start" not in raw


def test_sse_stream_tool_use_uses_input_json_delta(monkeypatch):
    """tool_use blocks need input_json_delta, not text_delta."""
    import agent
    monkeypatch.setattr(agent, "ChatAgent", _make_fake_class(
        content=[
            {"type": "tool_use", "id": "tu_1", "name": "create_task",
             "input": {"owner_id": "researcher"}},
        ],
        stop_reason="tool_use",
    ))
    from _chat_agent_handler_loader import handle_chat_request

    async def run():
        resp = await handle_chat_request(
            {"model": "claude-sonnet-4-5",
             "messages": [{"role": "user", "content": "Take this on"}]},
            conversation_id="c", user_id="u", tenant_id="t",
            is_stream=True,
        )
        chunks = []
        async for chunk in resp.body_iterator:
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")
            chunks.append(chunk)
        return "".join(chunks)

    raw = asyncio.run(run())
    assert "input_json_delta" in raw
    assert "owner_id" in raw
    # stop_reason in message_delta is tool_use, not end_turn
    assert '"stop_reason": "tool_use"' in raw or '"stop_reason":"tool_use"' in raw
