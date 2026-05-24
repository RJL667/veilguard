"""Tests for `gather_hints` — parallel recall fan-out.

Covers the deadline-bounded fan-out, score calibration, error
isolation, and the circuit breaker that protects the recall pipeline
from a misbehaving connector.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
import time

import pytest

_CONNECTORS_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CONNECTORS_DIR))

from _base import (  # noqa: E402
    Capability,
    Chunk,
    Connector,
    Content,
    Ref,
    Snippet,
    UserContext,
    gather_hints,
)
from _base import recall as _recall_mod  # noqa: E402


def _ctx() -> UserContext:
    return UserContext(tenant_id="t", user_id="u", principals=[])


def _run(coro):
    return asyncio.run(coro)


# Reset breaker state between tests so they don't leak into each other.
@pytest.fixture(autouse=True)
def _reset_breaker():
    _recall_mod._breaker_state.clear()
    yield
    _recall_mod._breaker_state.clear()


# ─── helpers ──────────────────────────────────────────────────────────


class _BaseFake(Connector):
    """Base fake that satisfies Connector ABC and tracks call count."""

    def __init__(self, name: str):
        self.name = name
        self.version = "0.0.0"
        self.capabilities = {Capability.HINT}
        self.call_count = 0

    def chunk(self, content: Content) -> list[Chunk]:
        return []


def _snip(score: float, source: str = "fake", text: str = "x") -> Snippet:
    return Snippet(
        content=text,
        score=score,
        ref=Ref(source, "read", {"id": "1"}),
        acl=[],
        source=source,
    )


# ─── fan-out behavior ─────────────────────────────────────────────────


class _FastFake(_BaseFake):
    async def hint(self, prompt, user_ctx, deadline_ms, top_k=10):
        self.call_count += 1
        return [_snip(0.5, source=self.name)]


class TestFanOut:
    def test_empty_connectors_returns_empty(self):
        out = _run(gather_hints([], "anything", _ctx()))
        assert out == []

    def test_empty_prompt_returns_empty(self):
        c = _FastFake("a")
        out = _run(gather_hints([c], "", _ctx()))
        assert out == []
        assert c.call_count == 0

    def test_single_connector_returns_snippets(self):
        c = _FastFake("a")
        out = _run(gather_hints([c], "q", _ctx()))
        assert len(out) == 1
        assert out[0].source == "a"
        assert c.call_count == 1

    def test_multiple_connectors_fan_out(self):
        a = _FastFake("a")
        b = _FastFake("b")
        c = _FastFake("c")
        out = _run(gather_hints([a, b, c], "q", _ctx()))
        assert len(out) == 3
        assert {s.source for s in out} == {"a", "b", "c"}
        assert a.call_count == b.call_count == c.call_count == 1

    def test_results_sorted_descending_by_score(self):
        class _ScoredFake(_BaseFake):
            def __init__(self, name, score):
                super().__init__(name)
                self._score = score

            async def hint(self, prompt, user_ctx, deadline_ms, top_k=10):
                return [_snip(self._score, source=self.name)]

        connectors = [
            _ScoredFake("low", 0.1),
            _ScoredFake("high", 0.9),
            _ScoredFake("mid", 0.5),
        ]
        out = _run(gather_hints(connectors, "q", _ctx()))
        scores = [s.score for s in out]
        assert scores == sorted(scores, reverse=True)


# ─── deadline ─────────────────────────────────────────────────────────


class _SlowFake(_BaseFake):
    async def hint(self, prompt, user_ctx, deadline_ms, top_k=10):
        await asyncio.sleep(2.0)  # well past any reasonable deadline
        return [_snip(0.5, source=self.name)]


class TestDeadline:
    def test_slow_connector_dropped(self):
        slow = _SlowFake("slow")
        fast = _FastFake("fast")
        t0 = time.monotonic()
        out = _run(gather_hints([slow, fast], "q", _ctx(), deadline_ms=100))
        elapsed = time.monotonic() - t0
        # Should complete near deadline, not 2s
        assert elapsed < 1.0
        assert {s.source for s in out} == {"fast"}

    def test_slow_connector_no_breaker_trip(self):
        # Timeouts are expected and should NOT trip the breaker.
        slow = _SlowFake("slow")
        for _ in range(5):
            _run(gather_hints([slow], "q", _ctx(), deadline_ms=50))
        assert not _run(_recall_mod._is_breaker_open("slow"))


# ─── error isolation ──────────────────────────────────────────────────


class _BoomFake(_BaseFake):
    async def hint(self, prompt, user_ctx, deadline_ms, top_k=10):
        raise RuntimeError("simulated upstream failure")


class TestErrorIsolation:
    def test_one_failure_does_not_kill_others(self):
        boom = _BoomFake("boom")
        ok = _FastFake("ok")
        out = _run(gather_hints([boom, ok], "q", _ctx()))
        assert {s.source for s in out} == {"ok"}

    def test_not_implemented_silently_returns_empty(self):
        class _NotImpl(_BaseFake):
            async def hint(self, prompt, user_ctx, deadline_ms, top_k=10):
                raise NotImplementedError

        c = _NotImpl("x")
        out = _run(gather_hints([c], "q", _ctx()))
        assert out == []


# ─── circuit breaker ──────────────────────────────────────────────────


class TestCircuitBreaker:
    def test_three_failures_open_breaker(self):
        boom = _BoomFake("boom")
        for _ in range(3):
            _run(gather_hints([boom], "q", _ctx()))
        assert _run(_recall_mod._is_breaker_open("boom"))

    def test_open_breaker_skips_hint_calls(self):
        # Trip the breaker first
        boom = _BoomFake("boom")
        for _ in range(3):
            _run(gather_hints([boom], "q", _ctx()))

        # Now use a connector with the same name that would succeed —
        # the breaker should still be open and skip calling it. We use
        # the breaker module directly rather than swapping the
        # connector class, since the breaker is keyed by name.
        class _Counted(_BaseFake):
            async def hint(self, prompt, user_ctx, deadline_ms, top_k=10):
                self.call_count += 1
                return [_snip(0.5, source=self.name)]

        counted = _Counted("boom")  # same name as the failing connector
        _run(gather_hints([counted], "q", _ctx()))
        assert counted.call_count == 0

    def test_success_clears_breaker(self):
        # No failures yet → breaker closed
        ok = _FastFake("ok")
        _run(gather_hints([ok], "q", _ctx()))
        assert not _run(_recall_mod._is_breaker_open("ok"))


# ─── score calibration ────────────────────────────────────────────────


class TestCalibration:
    def test_calibrate_applied(self):
        class _RawScores(_BaseFake):
            def calibrate(self, raw):
                # Halve raw score
                return raw / 2.0

            async def hint(self, prompt, user_ctx, deadline_ms, top_k=10):
                return [_snip(1.0, source=self.name)]

        c = _RawScores("rs")
        out = _run(gather_hints([c], "q", _ctx()))
        assert out[0].score == 0.5

    def test_calibrate_default_clamps_to_unit_interval(self):
        class _OutOfBounds(_BaseFake):
            async def hint(self, prompt, user_ctx, deadline_ms, top_k=10):
                return [_snip(2.0, source=self.name), _snip(-0.5, source=self.name)]

        c = _OutOfBounds("oob")
        out = _run(gather_hints([c], "q", _ctx()))
        assert all(0.0 <= s.score <= 1.0 for s in out)
