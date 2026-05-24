"""Unit tests for middleware.tenant — per-request context propagation.

Verifies:
  - contextvars-based propagation across async boundaries
  - nested set_tenant_context (e.g., Director → Researcher sub-call) works
  - is_subagent_cid / parent_cid_from_sub helpers
"""

import asyncio

import pytest

from app.middleware import tenant


class TestSetTenantContext:
    def test_basic(self):
        with tenant.set_tenant_context(
            conversation_id="c-1",
            user_id="u-1",
            tenant_id="t-1",
            agent_id="director",
        ) as ctx:
            assert ctx.conversation_id == "c-1"
            current = tenant.current()
            assert current is ctx

    def test_resets_on_exit(self):
        assert tenant.current() is None
        with tenant.set_tenant_context(
            conversation_id="c-1", user_id="u-1",
            tenant_id="t-1", agent_id="d",
        ):
            assert tenant.current() is not None
        assert tenant.current() is None

    def test_nested_inner_overrides(self):
        with tenant.set_tenant_context(
            conversation_id="c-1", user_id="u-1",
            tenant_id="t-1", agent_id="director",
        ):
            with tenant.set_tenant_context(
                conversation_id="sub-c-1xx-bgt-deadbeef",
                user_id="u-1", tenant_id="t-1",
                agent_id="researcher",
                parent_cid="c-1",
                is_background=True,
            ):
                assert tenant.current().agent_id == "researcher"
                assert tenant.current().is_background
            assert tenant.current().agent_id == "director"

    def test_require_raises_when_unset(self):
        with pytest.raises(RuntimeError):
            tenant.require()


class TestSubagentCidHelpers:
    def test_is_subagent_cid_true(self):
        assert tenant.is_subagent_cid("sub-abc1234-bgt-deadbeef")

    def test_is_subagent_cid_false(self):
        assert not tenant.is_subagent_cid("conv-1234")
        assert not tenant.is_subagent_cid("")

    def test_parent_cid_from_sub(self):
        assert tenant.parent_cid_from_sub("sub-abc1234-bgt-deadbeef") == "abc1234"

    def test_parent_cid_from_non_sub_is_none(self):
        assert tenant.parent_cid_from_sub("conv-1234") is None

    def test_parent_cid_from_malformed_is_none(self):
        assert tenant.parent_cid_from_sub("sub-abc") is None


class TestAsyncPropagation:
    @pytest.mark.asyncio
    async def test_context_propagates_to_awaited_coro(self):
        captured = []

        async def child():
            captured.append(tenant.current())

        with tenant.set_tenant_context(
            conversation_id="c", user_id="u",
            tenant_id="t", agent_id="x",
        ):
            await child()

        assert captured[0] is not None
        assert captured[0].agent_id == "x"

    @pytest.mark.asyncio
    async def test_context_isolated_between_concurrent_tasks(self):
        # Each task sets its own context; we verify they don't bleed.
        results = []

        async def with_agent(agent_id: str):
            with tenant.set_tenant_context(
                conversation_id=f"c-{agent_id}",
                user_id=f"u-{agent_id}",
                tenant_id="t",
                agent_id=agent_id,
            ):
                await asyncio.sleep(0.01)
                results.append((agent_id, tenant.current().agent_id))

        await asyncio.gather(
            with_agent("director"),
            with_agent("researcher"),
            with_agent("builder"),
        )

        # All three should have their own context, undisturbed.
        for expected_a, actual_a in results:
            assert expected_a == actual_a
