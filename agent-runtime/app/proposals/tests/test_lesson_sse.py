"""Tests for the lesson SSE wiring added 2026-05-28.

Verifies:
  1. `promote_lesson_to_team_knowledge` broadcasts `lesson_created`
     when the TCMM observation persisted.
  2. It does NOT broadcast when the TCMM write returned False.
  3. The broadcast carries the expected fields (id, tenant_id, user_id,
     team_id, trigger, confidence, critic_id).
  4. A failed broadcast does NOT fail the lesson write.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch, AsyncMock

import pytest

from app.memory import phase_7_writers as W
from app import events as E


@pytest.fixture(autouse=True)
def fresh_main_loop():
    """events.broadcast needs a registered loop to schedule fan-out.
    Each test gets a fresh one; cleared after.
    """
    loop = asyncio.new_event_loop()
    E.attach_main_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def subscribe():
    """Helper that gives back a (start, get) pair.

    start(tenant, user): subscribe to the (tenant, user) namespace
    get(timeout=0.2): wait up to `timeout` for the next event after
    the `ready` frame.
    """
    state: dict = {"events": [], "task": None, "queue": None}

    async def _runner(tenant: str, user: str):
        gen = E.event_stream(tenant, user)
        async for chunk in gen:
            state["events"].append(chunk)
            # Hard stop after we see 1 non-ready event so the test
            # doesn't have to time out the heartbeat.
            data_events = [c for c in state["events"] if c.strip().startswith("data: ") and "ready" not in c]
            if data_events:
                return

    def start(tenant: str, user: str):
        # Start the subscriber coroutine on the loop the broadcast
        # also fans out on.
        loop = asyncio.get_event_loop()
        state["task"] = loop.create_task(_runner(tenant, user))

    async def get(timeout: float = 0.5):
        try:
            await asyncio.wait_for(state["task"], timeout=timeout)
        except asyncio.TimeoutError:
            pass
        # Extract event JSONs (skip the `ready` connection frame).
        import json as _j
        out = []
        for c in state["events"]:
            s = c.strip()
            if not s.startswith("data: "):
                continue
            try:
                evt = _j.loads(s[len("data: "):])
            except Exception:
                continue
            if evt.get("type") == "ready":
                continue
            out.append(evt)
        return out

    return start, get


def test_promote_lesson_broadcasts_lesson_created(subscribe):
    """Happy path: TCMM observe persisted → lesson_created event fans out."""
    start, get = subscribe
    tenant, user = "t-promote-ok", "u-promote-ok"

    async def _go():
        start(tenant, user)
        # Mock the TCMM write to succeed.
        with patch(
            "app.middleware.tcmm.observe_agent_output",
            new_callable=AsyncMock,
            return_value=True,
        ):
            # Fire after the subscription is established.
            await asyncio.sleep(0.05)
            ok = await W.promote_lesson_to_team_knowledge(
                tenant_id=tenant, user_id=user, team_id="team-A",
                trigger="trig",
                rule="rule",
                confidence=0.9,
                critic_id="critic-prose",
            )
            assert ok is True
        return await get(timeout=0.5)

    events = asyncio.get_event_loop().run_until_complete(_go())
    types = [e.get("type") for e in events]
    assert "lesson_created" in types, f"got types={types}"
    created = next(e for e in events if e["type"] == "lesson_created")
    assert created["tenant_id"] == tenant
    assert created["user_id"] == user
    assert created["team_id"] == "team-A"
    assert created["trigger"] == "trig"
    assert created["critic_id"] == "critic-prose"
    assert created["confidence"] == pytest.approx(0.9)
    # id should be derived as `lesson-<hex12>` for sidebar deep-link.
    assert created["id"].startswith("lesson-")


def test_promote_lesson_does_NOT_broadcast_on_tcmm_failure(subscribe):
    """If TCMM observe returned False, we MUST NOT emit a phantom event.

    A broadcast for a write that didn't land would mislead the sidebar
    into re-fetching and finding nothing new (or worse, double-rendering
    on later success).
    """
    start, get = subscribe
    tenant, user = "t-promote-fail", "u-promote-fail"

    async def _go():
        start(tenant, user)
        with patch(
            "app.middleware.tcmm.observe_agent_output",
            new_callable=AsyncMock,
            return_value=False,
        ):
            await asyncio.sleep(0.05)
            ok = await W.promote_lesson_to_team_knowledge(
                tenant_id=tenant, user_id=user, team_id="team-B",
                trigger="trig", rule="r", confidence=0.4,
            )
            assert ok is False
        return await get(timeout=0.3)

    events = asyncio.get_event_loop().run_until_complete(_go())
    assert events == [], (
        f"got phantom events from a failed TCMM write: {events}"
    )


def test_promote_lesson_survives_broadcast_failure():
    """A broken broadcast MUST NOT fail the lesson write.

    The sidebar's 10s polling fallback will catch missed events; failing
    the write because the SSE pipe hiccupped would be a regression.
    """
    async def _go():
        with patch(
            "app.middleware.tcmm.observe_agent_output",
            new_callable=AsyncMock,
            return_value=True,
        ), patch(
            "app.events.broadcast",
            side_effect=RuntimeError("simulated SSE failure"),
        ):
            ok = await W.promote_lesson_to_team_knowledge(
                tenant_id="t", user_id="u", team_id="team-x",
                trigger="t", rule="r", confidence=0.7,
            )
            return ok

    assert asyncio.get_event_loop().run_until_complete(_go()) is True
