"""Tests for SharePointConnector — the integration of credentials,
Graph client, and the connector contract.

The Graph client is replaced with a fake that returns canned data,
so these tests run without httpx or any network. The credential
resolver is StaticCredentialResolver — production deployments use
HttpCredentialResolver against LibreChat.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

import pytest

_CONNECTORS_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CONNECTORS_DIR))

from _base import (  # noqa: E402
    Capability,
    Content,
    OAuthToken,
    ReauthenticationRequiredError,
    Ref,
    StaticCredentialResolver,
    UserContext,
)
from sharepoint.connector import SharePointConnector  # noqa: E402
from sharepoint.graph import (  # noqa: E402
    FileContent,
    FolderItem,
    GraphError,
    SearchHit,
)


def _ctx(user_id: str = "u1") -> UserContext:
    return UserContext(tenant_id="t", user_id=user_id, principals=[])


def _run(coro):
    return asyncio.run(coro)


def _resolver_with(user_id: str = "u1") -> StaticCredentialResolver:
    r = StaticCredentialResolver()
    r.set(user_id, "sharepoint", OAuthToken("AT"))
    return r


# ─── fake Graph client ────────────────────────────────────────────────


class _FakeGraph:
    """Stub Graph client that returns canned results / records calls."""

    def __init__(self):
        self.search_calls: list[tuple[str, str, int]] = []
        self.search_returns: list[SearchHit] = []
        self.search_raise: Exception | None = None

        self.download_calls: list[tuple[str, str, str]] = []
        self.download_return: FileContent | None = None
        self.download_raise: Exception | None = None

        self.permissions_calls: list[tuple[str, str, str]] = []
        self.permissions_return: list[str] = []
        self.permissions_raise: Exception | None = None

        self.list_calls: list[tuple[str, str, str | None]] = []
        self.list_returns: list[FolderItem] = []

    async def search(self, access_token, query, *, top=10):
        self.search_calls.append((access_token, query, top))
        if self.search_raise:
            raise self.search_raise
        return list(self.search_returns)

    async def download(self, access_token, *, drive_id, item_id):
        self.download_calls.append((access_token, drive_id, item_id))
        if self.download_raise:
            raise self.download_raise
        return self.download_return

    async def get_permissions(self, access_token, *, drive_id, item_id):
        self.permissions_calls.append((access_token, drive_id, item_id))
        if self.permissions_raise:
            raise self.permissions_raise
        return list(self.permissions_return)

    async def list_folder(self, access_token, *, drive_id, item_id=None):
        self.list_calls.append((access_token, drive_id, item_id))
        return list(self.list_returns)


# ─── construction ─────────────────────────────────────────────────────


class TestConstruction:
    def test_advertises_full_capability_set(self):
        c = SharePointConnector(credentials=StaticCredentialResolver())
        assert Capability.HINT in c.capabilities
        assert Capability.SEARCH in c.capabilities
        assert Capability.READ in c.capabilities
        assert Capability.LIST in c.capabilities
        assert Capability.PERMISSIONS in c.capabilities
        assert Capability.WRITE not in c.capabilities

    def test_name_and_version(self):
        c = SharePointConnector(credentials=StaticCredentialResolver())
        assert c.name == "sharepoint"
        assert c.version == "0.1.0"


# ─── hint / search ────────────────────────────────────────────────────


class TestHint:
    def test_returns_snippets_with_provenance(self):
        graph = _FakeGraph()
        graph.search_returns = [
            SearchHit(
                item_id="01ABC",
                drive_id="D1",
                name="Q1 Status.docx",
                summary="Quarterly status update",
                score=1.0,
                last_modified="2026-04-15T10:00:00Z",
                web_url=None,
            )
        ]
        c = SharePointConnector(credentials=_resolver_with(), graph=graph)
        snippets = _run(c.hint("status", _ctx(), deadline_ms=1000))
        assert len(snippets) == 1
        s = snippets[0]
        assert s.source == "sharepoint"
        assert s.ref.connector == "sharepoint"
        assert s.ref.tool == "read"
        assert s.ref.args["item_id"] == "01ABC"
        assert s.ref.args["drive_id"] == "D1"
        assert s.title == "Q1 Status.docx"
        assert s.content == "Quarterly status update"
        # rank=1 → score 1.0 after calibrate
        assert s.score == 1.0

    def test_passes_token_to_graph(self):
        graph = _FakeGraph()
        c = SharePointConnector(credentials=_resolver_with(), graph=graph)
        _run(c.hint("q", _ctx(), deadline_ms=1000))
        assert graph.search_calls == [("AT", "q", 10)]

    def test_empty_query_returns_empty(self):
        graph = _FakeGraph()
        c = SharePointConnector(credentials=_resolver_with(), graph=graph)
        out = _run(c.hint("   ", _ctx(), deadline_ms=1000))
        assert out == []
        assert graph.search_calls == []  # no Graph call

    def test_missing_user_id_raises_reauth(self):
        c = SharePointConnector(credentials=_resolver_with(), graph=_FakeGraph())
        with pytest.raises(ReauthenticationRequiredError):
            _run(c.hint("q", _ctx(user_id=""), deadline_ms=1000))

    def test_no_token_raises_reauth(self):
        # Resolver is empty — no token for this user
        graph = _FakeGraph()
        c = SharePointConnector(credentials=StaticCredentialResolver(), graph=graph)
        with pytest.raises(ReauthenticationRequiredError):
            _run(c.hint("q", _ctx(), deadline_ms=1000))

    def test_graph_error_propagates(self):
        graph = _FakeGraph()
        graph.search_raise = GraphError(500, "boom")
        c = SharePointConnector(credentials=_resolver_with(), graph=graph)
        with pytest.raises(GraphError):
            _run(c.hint("q", _ctx(), deadline_ms=1000))


class TestSearch:
    def test_returns_refs_only(self):
        graph = _FakeGraph()
        graph.search_returns = [
            SearchHit(item_id="X", drive_id="D", name="x.txt",
                      summary="", score=1.0, last_modified=None, web_url=None),
            SearchHit(item_id="Y", drive_id="D", name="y.txt",
                      summary="", score=2.0, last_modified=None, web_url=None),
        ]
        c = SharePointConnector(credentials=_resolver_with(), graph=graph)
        refs = _run(c.search("q", _ctx()))
        assert len(refs) == 2
        assert all(r.connector == "sharepoint" for r in refs)
        assert all(r.tool == "read" for r in refs)
        assert {r.args["item_id"] for r in refs} == {"X", "Y"}


# ─── read ─────────────────────────────────────────────────────────────


class TestRead:
    def test_returns_content_with_provenance(self):
        graph = _FakeGraph()
        graph.download_return = FileContent(
            raw=b"hello world",
            name="notes.txt",
            mime_type="text/plain",
            etag="v3",
            last_modified="2026-04-15T10:00:00Z",
            size=11,
        )
        graph.permissions_return = ["group:engineering"]
        c = SharePointConnector(credentials=_resolver_with(), graph=graph)
        ref = Ref("sharepoint", "read", {"item_id": "01ABC", "drive_id": "D1"})
        content = _run(c.read(ref, _ctx()))
        assert content.text == "hello world"
        assert content.title == "notes.txt"
        assert content.etag == "v3"
        assert content.acl == ["group:engineering"]
        assert content.mime_type == "text/plain"
        assert content.extra["size"] == 11
        assert content.ref == ref

    def test_acl_failure_falls_back_to_empty(self):
        graph = _FakeGraph()
        graph.download_return = FileContent(
            raw=b"x", name="a.txt", mime_type=None,
            etag=None, last_modified=None, size=1,
        )
        graph.permissions_raise = GraphError(500, "boom")
        c = SharePointConnector(credentials=_resolver_with(), graph=graph)
        ref = Ref("sharepoint", "read", {"item_id": "01ABC", "drive_id": "D1"})
        content = _run(c.read(ref, _ctx()))
        assert content.acl == []  # graceful degradation

    def test_missing_item_id_raises_value_error(self):
        c = SharePointConnector(credentials=_resolver_with(), graph=_FakeGraph())
        ref = Ref("sharepoint", "read", {"drive_id": "D"})  # no item_id
        with pytest.raises(ValueError):
            _run(c.read(ref, _ctx()))

    def test_wrong_connector_raises_value_error(self):
        c = SharePointConnector(credentials=_resolver_with(), graph=_FakeGraph())
        ref = Ref("slack", "read", {"item_id": "X", "drive_id": "D"})
        with pytest.raises(ValueError):
            _run(c.read(ref, _ctx()))

    def test_site_id_synonym_for_drive_id(self):
        graph = _FakeGraph()
        graph.download_return = FileContent(
            raw=b"x", name="a.txt", mime_type=None,
            etag=None, last_modified=None, size=1,
        )
        graph.permissions_return = []
        c = SharePointConnector(credentials=_resolver_with(), graph=graph)
        ref = Ref("sharepoint", "read", {"item_id": "X", "site_id": "S1"})
        # Should not raise — site_id maps to drive_id
        _run(c.read(ref, _ctx()))
        assert graph.download_calls == [("AT", "S1", "X")]


# ─── list ─────────────────────────────────────────────────────────────


class TestList:
    def test_empty_path_returns_empty(self):
        c = SharePointConnector(credentials=_resolver_with(), graph=_FakeGraph())
        assert _run(c.list(_ctx())) == []

    def test_lists_files_only(self):
        graph = _FakeGraph()
        graph.list_returns = [
            FolderItem(item_id="F1", drive_id="D", name="folder", is_folder=True, size=0),
            FolderItem(item_id="X", drive_id="D", name="x.txt", is_folder=False, size=10),
            FolderItem(item_id="Y", drive_id="D", name="y.docx", is_folder=False, size=20),
        ]
        c = SharePointConnector(credentials=_resolver_with(), graph=graph)
        refs = _run(c.list(_ctx(), path="D"))
        assert len(refs) == 2
        assert {r.args["item_id"] for r in refs} == {"X", "Y"}


# ─── permissions ──────────────────────────────────────────────────────


class TestGetPermissions:
    def test_returns_principals(self):
        graph = _FakeGraph()
        graph.permissions_return = ["user:u1", "group:g1"]
        c = SharePointConnector(credentials=_resolver_with(), graph=graph)
        ref = Ref("sharepoint", "read", {"item_id": "X", "drive_id": "D"})
        out = _run(c.get_permissions(ref, _ctx()))
        assert "user:u1" in out
        assert "group:g1" in out


# ─── chunk ────────────────────────────────────────────────────────────


class TestChunk:
    """chunk() delegates to _base.parsing.chunk_text (LlamaIndex
    SentenceSplitter when available, paragraph-split fallback when
    not). These tests assert connector-level behavior — sequential
    positions, empty handling — without pinning the exact splitting
    strategy."""

    def test_empty_content_returns_empty(self):
        c = SharePointConnector(credentials=StaticCredentialResolver())
        content = Content(
            text="",
            ref=Ref("sharepoint", "read", {"item_id": "x", "drive_id": "d"}),
            acl=[],
        )
        assert c.chunk(content) == []

    def test_whitespace_content_returns_empty(self):
        c = SharePointConnector(credentials=StaticCredentialResolver())
        content = Content(
            text="   \n\n   ",
            ref=Ref("sharepoint", "read", {"item_id": "x", "drive_id": "d"}),
            acl=[],
        )
        assert c.chunk(content) == []

    def test_short_text_yields_at_least_one_chunk(self):
        c = SharePointConnector(credentials=StaticCredentialResolver())
        content = Content(
            text="A short document with one sentence.",
            ref=Ref("sharepoint", "read", {"item_id": "x", "drive_id": "d"}),
            acl=[],
        )
        chunks = c.chunk(content)
        assert len(chunks) >= 1
        assert chunks[0].position == 0
        assert "short document" in chunks[0].text

    def test_long_text_produces_multiple_chunks(self):
        # ~10000 chars — should split into multiple chunks under
        # the 512-token default
        sentences = [f"This is sentence number {i} with some content." for i in range(300)]
        text = " ".join(sentences)
        c = SharePointConnector(credentials=StaticCredentialResolver())
        content = Content(
            text=text,
            ref=Ref("sharepoint", "read", {"item_id": "x", "drive_id": "d"}),
            acl=[],
        )
        chunks = c.chunk(content)
        assert len(chunks) > 1

    def test_chunk_positions_sequential(self):
        sentences = [f"Sentence {i}." for i in range(100)]
        text = " ".join(sentences)
        c = SharePointConnector(credentials=StaticCredentialResolver())
        content = Content(
            text=text,
            ref=Ref("sharepoint", "read", {"item_id": "x", "drive_id": "d"}),
            acl=[],
        )
        chunks = c.chunk(content)
        assert [ch.position for ch in chunks] == list(range(len(chunks)))


# ─── calibrate ────────────────────────────────────────────────────────


class TestCalibrate:
    def test_zero_or_negative_returns_zero(self):
        c = SharePointConnector(credentials=StaticCredentialResolver())
        assert c.calibrate(0) == 0.0
        assert c.calibrate(-5) == 0.0

    def test_rank_ordinal_inverse(self):
        c = SharePointConnector(credentials=StaticCredentialResolver())
        assert c.calibrate(1) == 1.0
        assert c.calibrate(2) == 0.5
        assert c.calibrate(10) == 0.1

    def test_subunit_passes_through(self):
        c = SharePointConnector(credentials=StaticCredentialResolver())
        assert c.calibrate(0.5) == 0.5
        assert c.calibrate(0.99) == 0.99
