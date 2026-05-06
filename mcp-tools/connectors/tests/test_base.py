"""Contract tests for the Connector framework SDK.

Pure tests over the type contract — no TCMM, no MCP, no I/O, no async test
plugins. Run with::

    pip install pytest
    cd mcp-tools/connectors && python -m pytest tests/

These tests pin the public shapes that connectors and Veilguard's recall
step rely on. Breaking changes here mean a coordinated rollout across all
connectors and the recall pipeline.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
from datetime import datetime

import pytest

# Add connectors/ to sys.path so we can import _base as a package without
# requiring an installable distribution.
_CONNECTORS_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CONNECTORS_DIR))

from _base import (  # noqa: E402
    Capability,
    Chunk,
    Connector,
    Content,
    RecallHit,
    Ref,
    Snippet,
    UserContext,
)


# ─── Ref ──────────────────────────────────────────────────────────────


class TestRef:
    def test_is_frozen(self):
        r = Ref(connector="x", tool="y", args={})
        with pytest.raises(Exception):
            r.connector = "z"  # type: ignore[misc]

    def test_to_tool_call_renders_kwargs(self):
        r = Ref(
            "sharepoint",
            "read_file",
            {"item_id": "01ABC", "site_id": "X"},
        )
        s = r.to_tool_call()
        assert s.startswith("sharepoint.read_file(")
        assert "item_id='01ABC'" in s
        assert "site_id='X'" in s

    def test_to_tool_call_no_args(self):
        r = Ref("slack", "list_channels", {})
        assert r.to_tool_call() == "slack.list_channels()"


# ─── Snippet ──────────────────────────────────────────────────────────


class TestSnippet:
    def test_required_fields_only(self):
        s = Snippet(
            content="hi",
            score=0.5,
            ref=Ref("x", "y", {}),
            acl=["g1"],
            source="x",
        )
        assert s.title is None
        assert s.last_modified is None
        assert s.etag is None

    def test_optional_metadata_round_trips(self):
        ts = datetime(2026, 5, 4)
        s = Snippet(
            content="hi",
            score=0.5,
            ref=Ref("x", "y", {}),
            acl=["g1"],
            source="x",
            title="Doc",
            last_modified=ts,
            etag="v1",
        )
        assert s.title == "Doc"
        assert s.last_modified == ts
        assert s.etag == "v1"


# ─── UserContext ──────────────────────────────────────────────────────


class TestUserContext:
    def test_default_principals_is_empty_list(self):
        u = UserContext(tenant_id="t", user_id="u")
        assert u.principals == []

    def test_principals_not_shared_across_instances(self):
        a = UserContext(tenant_id="t", user_id="u1")
        b = UserContext(tenant_id="t", user_id="u2")
        a.principals.append("g1")
        assert b.principals == []  # default_factory, not class-level mutable


# ─── Content & Chunk ──────────────────────────────────────────────────


class TestContent:
    def test_minimal_content(self):
        c = Content(
            text="body",
            ref=Ref("x", "read", {"id": "1"}),
            acl=["g1"],
        )
        assert c.text == "body"
        assert c.title is None
        assert c.etag is None
        assert c.extra == {}


class TestChunk:
    def test_chunk_keeps_position(self):
        c = Chunk(text="part 1", position=0)
        assert c.position == 0
        assert c.extra == {}


# ─── RecallHit ────────────────────────────────────────────────────────


class TestRecallHit:
    def test_is_live_hint_defaults_false(self):
        h = RecallHit(content="hi", score=0.7, source="tcmm", tool_ref=None)
        assert h.is_live_hint is False

    def test_live_hint_carries_tool_ref(self):
        # Convention (not enforced by type): a live connector hit always
        # carries tool_ref so the LLM can dereference. tcmm-only memory
        # hits set tool_ref=None.
        h = RecallHit(
            content="hi",
            score=0.9,
            source="sharepoint",
            tool_ref=Ref("sharepoint", "read_file", {"id": "X"}),
            is_live_hint=True,
        )
        assert h.tool_ref is not None
        assert h.is_live_hint is True


# ─── Capability ───────────────────────────────────────────────────────


class TestCapability:
    def test_string_enum_value(self):
        assert Capability.HINT.value == "hint"
        assert Capability.READ.value == "read"

    def test_set_membership(self):
        caps = {Capability.HINT, Capability.READ}
        assert Capability.HINT in caps
        assert Capability.WRITE not in caps


# ─── Connector ABC ────────────────────────────────────────────────────


class _MinimalConnector(Connector):
    """Concrete subclass for tests — implements only the abstract method."""

    name = "minimal"
    version = "1.0.0"
    capabilities = set()

    def chunk(self, content: Content) -> list[Chunk]:
        return [Chunk(text=content.text, position=0)]


class TestConnectorABC:
    def test_cannot_instantiate_without_chunk(self):
        class NoChunk(Connector):
            name = "broken"

        with pytest.raises(TypeError):
            NoChunk()  # type: ignore[abstract]

    def test_minimal_subclass_instantiates(self):
        c = _MinimalConnector()
        assert c.name == "minimal"
        assert c.version == "1.0.0"
        assert c.capabilities == set()

    def test_calibrate_clamps_to_unit_interval(self):
        c = _MinimalConnector()
        assert c.calibrate(-0.5) == 0.0
        assert c.calibrate(1.5) == 1.0
        assert c.calibrate(0.7) == 0.7


class TestConnectorAsyncDefaults:
    """Async defaults are exercised via asyncio.run() to avoid pulling in
    pytest-asyncio just for two test cases."""

    def test_unimplemented_capabilities_raise(self):
        async def check():
            c = _MinimalConnector()
            ctx = UserContext(tenant_id="t", user_id="u")

            with pytest.raises(NotImplementedError):
                await c.hint("p", ctx, deadline_ms=300)
            with pytest.raises(NotImplementedError):
                await c.read(Ref("x", "y", {}), ctx)
            with pytest.raises(NotImplementedError):
                await c.search("q", ctx)
            with pytest.raises(NotImplementedError):
                await c.list(ctx)
            with pytest.raises(NotImplementedError):
                await c.get_permissions(Ref("x", "y", {}), ctx)

        asyncio.run(check())

    def test_healthcheck_default(self):
        async def check():
            c = _MinimalConnector()
            h = await c.healthcheck()
            assert h["name"] == "minimal"
            assert h["version"] == "1.0.0"
            assert h["ok"] is True

        asyncio.run(check())
