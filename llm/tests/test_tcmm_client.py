"""Tests for veilguard.llm.tcmm_client — render + ingest only.

Uses respx to mock TCMM HTTP responses; no real :8811 needed.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest


# Use httpx's built-in MockTransport to avoid an extra dep.

def _mock_transport(handler):
    """Wrap a sync handler into an httpx MockTransport."""
    return httpx.MockTransport(handler)


@pytest.fixture
def stub_tcmm(monkeypatch):
    """Replace llm.tcmm_client._client() with one backed by MockTransport.

    The handler is set per-test via _set_handler.
    """
    import llm.tcmm_client as tcmm_mod

    state = {"handler": None}

    def _handle(request: httpx.Request) -> httpx.Response:
        if state["handler"] is None:
            return httpx.Response(500, json={"error": "no handler set"})
        return state["handler"](request)

    transport = httpx.MockTransport(_handle)
    client = httpx.AsyncClient(transport=transport, timeout=10.0)

    # Inject our client and force teardown of any cached one.
    tcmm_mod._CLIENT = client

    def set_handler(fn):
        state["handler"] = fn

    yield set_handler

    # Cleanup — close the test client and clear the module's cache so
    # next test gets a fresh one.
    asyncio.run(client.aclose())
    tcmm_mod._CLIENT = None


# ── render_structured ───────────────────────────────────────────────────


def test_render_structured_happy_path(stub_tcmm):
    from llm.tcmm_client import render_structured

    def handler(req):
        # Verify the request shape
        body = req.read().decode()
        import json
        body_json = json.loads(body)
        assert body_json["conversation_id"] == "conv-1"
        assert body_json["user_id"] == "u-1"
        assert body_json["task_query"] == "what's up?"
        assert body_json["model"] == "anthropic"
        return httpx.Response(200, json={
            "status": "ok",
            "format": "anthropic-structured",
            "prompt": "<rendered>",
            "blocks": [
                {"type": "text", "text": "preamble",
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "memory blob"},
            ],
            "tier_summary": {"immutable": {"chars": 1000}},
            "stats": {"prompt_chars": 1234},
            "layout": {"immutable": {"start_byte": 0, "end_byte": 1000}},
        })

    stub_tcmm(handler)

    res = asyncio.run(render_structured(
        conv_id="conv-1", user_id="u-1", task_query="what's up?",
    ))
    assert res.prompt == "<rendered>"
    assert len(res.blocks) == 2
    assert res.blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert res.tier_summary == {"immutable": {"chars": 1000}}
    assert res.stats == {"prompt_chars": 1234}


def test_render_structured_connection_error_raises(stub_tcmm):
    """When TCMM is unreachable, a clear RuntimeError fires."""
    from llm.tcmm_client import render_structured

    def handler(req):
        raise httpx.ConnectError("test: connection refused")

    stub_tcmm(handler)

    with pytest.raises(RuntimeError, match="TCMM unreachable"):
        asyncio.run(render_structured(
            conv_id="c", user_id="u", task_query="q",
        ))


def test_render_structured_non_200_raises(stub_tcmm):
    from llm.tcmm_client import render_structured

    def handler(req):
        return httpx.Response(500, text="boom")

    stub_tcmm(handler)

    with pytest.raises(RuntimeError, match="returned 500"):
        asyncio.run(render_structured(
            conv_id="c", user_id="u", task_query="q",
        ))


# ── ingest_user / ingest_assistant ──────────────────────────────────────


def test_ingest_user_swallows_errors(stub_tcmm):
    """ingest_user is best-effort — should not raise even on TCMM error."""
    from llm.tcmm_client import ingest_user

    def handler(req):
        return httpx.Response(500, text="oops")

    stub_tcmm(handler)
    # Should not raise.
    asyncio.run(ingest_user("conv-1", "u-1", "hello there"))


def test_ingest_user_skips_empty_message(stub_tcmm):
    from llm.tcmm_client import ingest_user

    called = {"n": 0}
    def handler(req):
        called["n"] += 1
        return httpx.Response(200)

    stub_tcmm(handler)
    asyncio.run(ingest_user("conv-1", "u-1", ""))
    asyncio.run(ingest_user("", "u-1", "hi"))
    assert called["n"] == 0


def test_ingest_user_sends_raw_text(stub_tcmm):
    """Verify RAW (pre-redaction) text is what reaches TCMM."""
    from llm.tcmm_client import ingest_user

    captured = {}
    def handler(req):
        import json
        captured["body"] = json.loads(req.read().decode())
        return httpx.Response(200, json={"status": "ok"})

    stub_tcmm(handler)
    raw = "Email Alice Johnson at alice@acme.com."
    asyncio.run(ingest_user("conv-1", "u-1", raw))
    assert captured["body"]["user_message"] == raw
    assert captured["body"]["origin"] == "user"
    assert captured["body"]["recall_only"] is False


def test_ingest_assistant_includes_model_and_flag(stub_tcmm):
    from llm.tcmm_client import ingest_assistant

    captured = {}
    def handler(req):
        import json
        captured["body"] = json.loads(req.read().decode())
        return httpx.Response(200, json={"status": "ok"})

    stub_tcmm(handler)
    asyncio.run(ingest_assistant(
        "conv-1", "u-1", "Sure thing.",
        model="claude-sonnet-4-5",
        flag_obj={"used": True, "knowledge_class": "factoid"},
    ))
    assert captured["body"]["model"] == "claude-sonnet-4-5"
    assert captured["body"]["flag_obj"] == {
        "used": True, "knowledge_class": "factoid",
    }
    assert captured["body"]["raw_output"] == "Sure thing."
    assert captured["body"]["origin"] == "assistant_text"
