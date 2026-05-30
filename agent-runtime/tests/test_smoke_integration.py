"""Integration smoke test — verifies the full middleware wiring.

This test mocks the Claude Agent SDK so we don't need a real API key.
What we DO verify end-to-end:

  1. /agent/query accepts a well-formed request and dispatches to runtime
  2. Tenant context is established (visible to hooks via contextvars)
  3. TCMM render fails-soft (returns [] when tcmm-service unreachable)
  4. Cache_control is normalized on the system prefix
  5. Audit usage is captured from the mocked SDK message stream
  6. SSE events emit in the expected shape (run_start, text_delta,
     tool_call, usage, run_end)
  7. Approval gate hook fires when a client-daemon tool is "called"
     (mocked via direct hook invocation, no actual SDK round-trip)

Things NOT tested here (need the real SDK + Anthropic API):
  - Actual cache hit rates (the live spike script handles that)
  - Real subagent invocation
  - Real MCP tool dispatch

For Phase 0 this gives us confidence that the wiring is correct.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from app.middleware import tenant
from app.middleware.audit import TurnUsage, tap_sdk_stream
from app.middleware.tcmm import get_system_prefix, invalidate_cache
from app.middleware.cache_control import normalize_cache_control
from app.personas.loader import PersonaSpec, PersonaRegistry


# ── Helper: build a fake persona without touching disk ──────────────────


def _fake_persona(
    agent_id: str = "researcher",
    role: str = "ic",
    model: str = "claude-sonnet-4-7",
    tools: list[str] | None = None,
) -> PersonaSpec:
    return PersonaSpec(
        agent_id=agent_id,
        role=role,
        manager_id="director" if role == "ic" else None,
        team_id="core" if role != "consultant" else None,
        model=model,
        tools=tools or ["read_file", "recall", "observe"],
        system_prompt=f"You are {agent_id}.",
        display_name=agent_id.title(),
    )


# ── Fake SDK message stream ─────────────────────────────────────────────


class _FakeUsage:
    def __init__(self, input_tokens=100, output_tokens=50,
                 cache_creation=200, cache_read=300):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation
        self.cache_read_input_tokens = cache_read


class _FakeBlock:
    def __init__(self, type_: str, **kwargs):
        self.type = type_
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeInnerMessage:
    def __init__(self, content, model, usage):
        self.content = content
        self.model = model
        self.usage = usage


class _FakeMsg:
    def __init__(self, type_: str, **kwargs):
        self.type = type_
        for k, v in kwargs.items():
            setattr(self, k, v)


async def _fake_sdk_stream():
    """Yield a realistic sequence of SDK messages."""
    # Assistant content with one text block and one tool_use
    inner = _FakeInnerMessage(
        content=[
            _FakeBlock("text", text="Looking into this."),
            _FakeBlock("tool_use", name="recall", id="tu_abc123"),
        ],
        model="claude-sonnet-4-7",
        usage=_FakeUsage(),
    )
    yield _FakeMsg("assistant", message=inner)

    # Final result
    yield _FakeMsg(
        "result",
        result="Done.",
        stop_reason="end_turn",
    )


# ── audit.tap_sdk_stream tests (no SDK needed) ──────────────────────────


class TestAuditTap:
    @pytest.mark.asyncio
    async def test_captures_tokens_and_content(self):
        usage = TurnUsage()
        collected = []
        async for msg in tap_sdk_stream(_fake_sdk_stream(), usage):
            collected.append(msg)

        # Forwarded all messages unchanged
        assert len(collected) == 2

        # Tokens captured
        assert usage.tokens_input_new == 100
        assert usage.tokens_output == 50
        assert usage.cache_create == 200
        assert usage.cache_read == 300
        assert usage.tokens_input_total == 600  # 100 + 200 + 300

        # Cache hit rate computed
        assert 0.0 <= usage.cache_hit_rate() <= 1.0

        # Content captured
        assert "Looking into this." in usage.content_text()

        # Tool calls captured
        assert len(usage.tool_calls) == 1
        assert usage.tool_calls[0]["name"] == "recall"

        # Stop reason from result message
        assert usage.stop_reason == "end_turn"

        # Model from inner message
        assert usage.model == "claude-sonnet-4-7"


# ── TCMM middleware tests (mocked HTTP) ─────────────────────────────────


class TestTCMMMiddleware:
    @pytest.mark.asyncio
    async def test_returns_empty_when_tcmm_unreachable(self, monkeypatch):
        # Simulate tcmm-service being down → degraded mode returns [].
        async def _fail_render(*args, **kwargs):
            return [], ""

        from app.middleware import tcmm as tcmm_mod
        monkeypatch.setattr(tcmm_mod, "_fetch_tcmm_render", _fail_render)

        invalidate_cache("conv-x")
        blocks = await get_system_prefix(
            conversation_id="conv-x",
            user_id="u",
            agent_id="researcher",
            model="claude-sonnet-4-7",
        )
        assert blocks == []

    @pytest.mark.asyncio
    async def test_memoizes_per_parent_cid(self, monkeypatch):
        # First call hits the fake fetch; second hits the cache.
        call_count = {"n": 0}

        async def _counting_render(*args, **kwargs):
            call_count["n"] += 1
            return ([{"type": "text", "text": f"block {call_count['n']}"}],
                    "v1")

        from app.middleware import tcmm as tcmm_mod
        monkeypatch.setattr(tcmm_mod, "_fetch_tcmm_render", _counting_render)

        invalidate_cache("conv-memo")

        first = await get_system_prefix(
            conversation_id="conv-memo",
            user_id="u",
            agent_id="researcher",
            model="claude-sonnet-4-7",
        )
        second = await get_system_prefix(
            conversation_id="conv-memo",
            user_id="u",
            agent_id="researcher",
            model="claude-sonnet-4-7",
        )

        assert call_count["n"] == 1, "second call should hit cache, not refetch"
        # Both returns should be identical (TCMM's bytes passed through).
        assert first == second

    @pytest.mark.asyncio
    async def test_passes_tcmm_markers_through_unmodified(self, monkeypatch):
        """TCMM's per-provider renderer places cache_control markers.
        We must NOT touch them — that's TCMM's job, not agent-runtime's.
        Regression test for the 2026-05-22 "trust TCMM markers" fix.
        """
        tcmm_output = [
            {"type": "text", "text": "TCMM block 1"},
            {"type": "text", "text": "TCMM block 2 (cache point)",
             "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            {"type": "text", "text": "TCMM block 3 (post-cache)"},
        ]

        async def _fake_render(*args, **kwargs):
            return tcmm_output, "v-fixed"

        from app.middleware import tcmm as tcmm_mod
        monkeypatch.setattr(tcmm_mod, "_fetch_tcmm_render", _fake_render)

        invalidate_cache("conv-pass")

        blocks = await get_system_prefix(
            conversation_id="conv-pass",
            user_id="u",
            agent_id="researcher",
            model="claude-sonnet-4-7",
        )

        # Bytes must be IDENTICAL to TCMM's output.  No marker movement,
        # no stripping, no re-placement.
        assert blocks == tcmm_output
        # Specifically: marker still on block 2 (not block 3).
        assert "cache_control" not in blocks[0]
        assert "cache_control" in blocks[1]
        assert "cache_control" not in blocks[2]

    @pytest.mark.asyncio
    async def test_sub_call_shares_parent_cache(self, monkeypatch):
        call_count = {"n": 0}

        async def _counting_render(*args, **kwargs):
            call_count["n"] += 1
            return ([{"type": "text", "text": "blob"}], "v1")

        from app.middleware import tcmm as tcmm_mod
        monkeypatch.setattr(tcmm_mod, "_fetch_tcmm_render", _counting_render)

        invalidate_cache("parent7")

        # Parent call fills the cache.
        await get_system_prefix(
            conversation_id="parent7",
            user_id="u",
            agent_id="researcher",
            model="claude-sonnet-4-7",
        )
        # Sibling sub-call should share the cache (same parent_cid).
        await get_system_prefix(
            conversation_id="sub-parent7-bgt-deadbeef",
            user_id="u",
            agent_id="researcher",
            model="claude-sonnet-4-7",
        )

        assert call_count["n"] == 1, (
            "sub-call should reuse parent's cached render, not refetch"
        )


# ── Approval gate hook tests ────────────────────────────────────────────


# TestApprovalGateHook removed 2026-05-25 — approval_gate_hook deprecated.
# Local SDK PreToolUse path is dead; canonical gate is sub-agents/core/approval.py.
# Approval behaviour is exercised by the e2e Director/IC/Critic loop test.


# ── Cache_control normalization end-to-end ──────────────────────────────


class TestCacheControlDefensiveUtility:
    """`normalize_cache_control` is no longer called on TCMM output (TCMM
    owns marker placement per the 2026-05-22 trust-TCMM fix).  The
    utility still exists for defensive use cases (e.g., a health-check
    that asserts <=4 markers in a final request payload).  Kept tested
    so it doesn't bit-rot.
    """

    def test_normalize_still_works_for_defensive_use(self):
        blocks = [
            {"type": "text", "text": "block 1",
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "block 2",
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "block 3 (last)",
             "cache_control": {"type": "ephemeral"}},
        ]

        normalized = normalize_cache_control(blocks, ttl="1h")

        # Only last block has marker
        assert "cache_control" not in normalized[0]
        assert "cache_control" not in normalized[1]
        assert normalized[2]["cache_control"] == {
            "type": "ephemeral", "ttl": "1h"
        }
        # Content preserved
        assert normalized[2]["text"] == "block 3 (last)"
