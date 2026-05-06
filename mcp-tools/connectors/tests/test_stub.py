"""End-to-end smoke tests for the StubConnector.

Exercises every capability of the framework against canned data, no
I/O. If these pass, the SDK contract is wired correctly and any new
real connector just has to mirror the StubConnector pattern.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

import pytest

_CONNECTORS_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CONNECTORS_DIR))

from _base import Capability, Ref, UserContext  # noqa: E402
from _stub.connector import StubConnector  # noqa: E402


def _ctx() -> UserContext:
    return UserContext(
        tenant_id="t1",
        user_id="u1",
        principals=["group:engineering"],
    )


def _run(coro):
    return asyncio.run(coro)


class TestStubCapabilities:
    def test_advertises_full_capability_set(self):
        c = StubConnector()
        for cap in (
            Capability.HINT,
            Capability.SEARCH,
            Capability.READ,
            Capability.LIST,
            Capability.PERMISSIONS,
        ):
            assert cap in c.capabilities
        assert Capability.WRITE not in c.capabilities


class TestStubHint:
    def test_returns_snippets_for_matching_prompt(self):
        c = StubConnector()
        hits = _run(c.hint("project status alpha", _ctx(), deadline_ms=300))
        assert len(hits) >= 1
        top = hits[0]
        assert top.source == "stub"
        assert top.ref.connector == "stub"
        assert top.ref.tool == "read"
        assert top.score > 0.1
        assert top.title is not None
        assert top.etag is not None

    def test_results_descending_by_score(self):
        c = StubConnector()
        hits = _run(c.hint("incident credential leak", _ctx(), deadline_ms=300))
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_empty_for_unrelated_prompt(self):
        c = StubConnector()
        hits = _run(c.hint("the weather forecast tomorrow", _ctx(), deadline_ms=300))
        assert hits == []

    def test_top_k_caps_results(self):
        c = StubConnector()
        # Prompt that touches multiple corpus entries
        hits = _run(c.hint(
            "project onboarding incident credential",
            _ctx(),
            deadline_ms=300,
            top_k=2,
        ))
        assert len(hits) <= 2

    def test_snippet_carries_acl_and_etag(self):
        c = StubConnector()
        hits = _run(c.hint("incident credential", _ctx(), deadline_ms=300))
        assert hits, "expected at least one hit"
        s = hits[0]
        assert "group:security" in s.acl
        assert s.etag == "v7"


class TestStubRead:
    def test_returns_canned_content(self):
        c = StubConnector()
        ref = Ref("stub", "read", {"doc_id": "doc_alpha"})
        content = _run(c.read(ref, _ctx()))
        assert "Alpha" in content.text
        assert content.title == "Q1 Project Status"
        assert content.etag == "v3"
        assert "group:engineering" in content.acl
        assert content.ref == ref

    def test_unknown_doc_raises(self):
        c = StubConnector()
        ref = Ref("stub", "read", {"doc_id": "does_not_exist"})
        with pytest.raises(KeyError):
            _run(c.read(ref, _ctx()))


class TestStubSearch:
    def test_search_returns_refs_only(self):
        c = StubConnector()
        refs = _run(c.search("incident credential", _ctx()))
        assert len(refs) >= 1
        for r in refs:
            assert r.connector == "stub"
            assert r.tool == "read"
            assert "doc_id" in r.args


class TestStubList:
    def test_list_returns_full_corpus(self):
        c = StubConnector()
        refs = _run(c.list(_ctx()))
        assert len(refs) == 3
        doc_ids = {r.args["doc_id"] for r in refs}
        assert doc_ids == {"doc_alpha", "doc_beta", "doc_gamma"}


class TestStubPermissions:
    def test_returns_doc_acl(self):
        c = StubConnector()
        ref = Ref("stub", "read", {"doc_id": "doc_alpha"})
        acl = _run(c.get_permissions(ref, _ctx()))
        assert "group:engineering" in acl
        assert "group:execs" in acl

    def test_unknown_doc_returns_empty_acl(self):
        c = StubConnector()
        ref = Ref("stub", "read", {"doc_id": "missing"})
        acl = _run(c.get_permissions(ref, _ctx()))
        assert acl == []


class TestStubChunk:
    def test_chunks_have_sequential_positions(self):
        c = StubConnector()
        ref = Ref("stub", "read", {"doc_id": "doc_alpha"})
        content = _run(c.read(ref, _ctx()))
        chunks = c.chunk(content)
        assert len(chunks) >= 2
        positions = [ch.position for ch in chunks]
        assert positions == list(range(len(chunks)))


class TestStubFullLifecycle:
    """End-to-end: hint → dereference via ref → read → chunk.

    Mirrors what Veilguard's recall + auto-ingest path will do for any
    real connector.
    """

    def test_hint_to_read_to_chunk(self):
        c = StubConnector()
        ctx = _ctx()

        # 1. hint() produces candidates with refs
        hits = _run(c.hint("project status", ctx, deadline_ms=300))
        assert hits

        # 2. LLM (simulated) picks top hit and dereferences via its ref
        chosen_ref = hits[0].ref
        content = _run(c.read(chosen_ref, ctx))
        assert content.etag == hits[0].etag  # provenance preserved

        # 3. Connector's chunker prepares for ingest
        chunks = c.chunk(content)
        assert all(ch.text for ch in chunks)
        assert chunks[0].position == 0
