"""Unit tests for app.workers.inbox_poller.

Without a real Lance backend, we test:
  - InboxPoller constructs cleanly
  - _make_sub_cid produces the expected format
  - ELIGIBLE_OWNERS matches the personas it should
  - stop() flips the event so run() exits on the next tick
"""

from __future__ import annotations

import asyncio
import pytest

from app.personas.loader import PersonaRegistry
from app.workers.inbox_poller import (
    InboxPoller,
    ELIGIBLE_OWNERS,
)


class TestEligibility:
    def test_eligible_owners_is_the_four_ic_agents(self):
        assert ELIGIBLE_OWNERS == frozenset({
            "researcher",
            "builder",
            "critic-claim",
            "critic-prose",
        })
        # Director not in there — Director runs reactive only
        assert "director" not in ELIGIBLE_OWNERS


class TestSubCidFormat:
    def test_make_sub_cid_format(self):
        cid = InboxPoller._make_sub_cid("parent7", "ibx")
        # Format: sub-<parent7>-<kind3>-<uuid8>
        parts = cid.split("-")
        assert parts[0] == "sub"
        assert parts[1] == "parent7"
        assert parts[2] == "ibx"
        assert len(parts[3]) == 8  # uuid hex chars

    def test_make_sub_cid_truncates_long_parent(self):
        # `parent_cid[:7]` truncates to 7 chars including hyphens; for
        # "conv-very-long-parent-id" that yields "conv-ve".  Verify the
        # truncation happened (substring "conv-ve" appears) rather than
        # naive splitting (which breaks when the parent itself contains
        # hyphens).
        cid = InboxPoller._make_sub_cid("conv-very-long-parent-id", "bgt")
        assert cid.startswith("sub-conv-ve-bgt-")
        # Tail uuid still 8 chars
        assert len(cid.rsplit("-", 1)[1]) == 8

    def test_make_sub_cid_truncates_long_kind(self):
        cid = InboxPoller._make_sub_cid("parent7", "longkind")
        parts = cid.split("-")
        assert len(parts[2]) == 3


class TestPollerLifecycle:
    @pytest.mark.asyncio
    async def test_stop_flips_event(self):
        registry = PersonaRegistry({})
        poller = InboxPoller(registry)
        assert not poller._stop_evt.is_set()
        poller.stop()
        assert poller._stop_evt.is_set()

    @pytest.mark.asyncio
    async def test_run_exits_on_stop(self):
        """Run the poller for a short window, then stop.

        With no ledger backend the poll loop will catch the error in
        _poll_once and continue.  When we set stop_evt the loop should
        exit on the next interval.
        """
        registry = PersonaRegistry({})
        poller = InboxPoller(registry)
        # Patch interval down so the test doesn't take forever.
        import app.workers.inbox_poller as mod
        orig = mod.POLL_INTERVAL_S
        mod.POLL_INTERVAL_S = 0.05
        try:
            runner = asyncio.create_task(poller.run())
            await asyncio.sleep(0.15)  # let it poll a couple times
            poller.stop()
            await asyncio.wait_for(runner, timeout=2.0)
        finally:
            mod.POLL_INTERVAL_S = orig


class TestPollerInit:
    def test_worker_id_unique_per_instance(self):
        registry = PersonaRegistry({})
        a = InboxPoller(registry)
        b = InboxPoller(registry)
        assert a._worker_id != b._worker_id
        assert a._worker_id.startswith("worker-")

    def test_semaphore_respects_total_concurrent(self):
        # Phase 6.2 — global cap replaced by per-persona caps; total
        # semaphore == sum of per-persona caps.
        from app.workers.inbox_poller import (
            PERSONA_CAPS, _TOTAL_CONCURRENT_DISPATCHES,
        )
        registry = PersonaRegistry({})
        poller = InboxPoller(registry)
        assert _TOTAL_CONCURRENT_DISPATCHES == sum(PERSONA_CAPS.values())
        assert poller._semaphore._value == _TOTAL_CONCURRENT_DISPATCHES
