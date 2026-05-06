"""Stub connector — canned data for framework smoke tests.

Used by the contract test suite and by anyone bringing up a new
deployment to validate the recall pipeline end-to-end without touching
a real data source. Capabilities: HINT, SEARCH, READ, LIST, PERMISSIONS.

The corpus is three tiny "documents" with hard-coded ACLs and etags so
tests can assert the full provenance flow (ref → read → ingest → recall).
"""
from __future__ import annotations

import pathlib
import sys
from datetime import datetime, timezone

# Add connectors/ to sys.path so we can import _base. Mirrors the
# pattern used by mcp-tools/_shared.
_THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))

from _base import (  # noqa: E402
    Capability,
    Chunk,
    Connector,
    Content,
    Ref,
    Snippet,
    UserContext,
)


# ─── canned corpus ────────────────────────────────────────────────────


_CORPUS: dict[str, dict] = {
    "doc_alpha": {
        "title": "Q1 Project Status",
        "text": (
            "The Alpha project is on track for a March release. "
            "Outstanding risks: vendor delivery on signing service. "
            "Owner: J. Smith. Last update: 2026-04-15."
        ),
        "acl": ["group:engineering", "group:execs"],
        "etag": "v3",
        "keywords": ["project", "status", "alpha", "release", "risk"],
    },
    "doc_beta": {
        "title": "Onboarding Runbook",
        "text": (
            "New employee onboarding: provision laptop, AD account, "
            "VPN cert, and seat assignment within 48h of start date."
        ),
        "acl": ["group:hr", "group:it"],
        "etag": "v1",
        "keywords": ["onboarding", "employee", "laptop", "vpn"],
    },
    "doc_gamma": {
        "title": "Incident Response Notes",
        "text": (
            "On a credential leak, rotate within 30 minutes. "
            "Revoke all OAuth tokens. Page the SRE oncall."
        ),
        "acl": ["group:security"],
        "etag": "v7",
        "keywords": ["incident", "credential", "leak", "oauth", "rotate"],
    },
}

_CORPUS_TIMESTAMP = datetime(2026, 4, 15, tzinfo=timezone.utc)


def _score_for(prompt: str, keywords: list[str]) -> float:
    """Toy scoring: count keyword matches in prompt, normalize by total."""
    p = prompt.lower()
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw in p)
    return min(1.0, hits / max(1, len(keywords)) + 0.1) if hits else 0.0


# ─── connector ────────────────────────────────────────────────────────


class StubConnector(Connector):
    name = "stub"
    version = "0.1.0"
    capabilities = {
        Capability.HINT,
        Capability.SEARCH,
        Capability.READ,
        Capability.LIST,
        Capability.PERMISSIONS,
    }

    async def hint(
        self,
        prompt: str,
        user_ctx: UserContext,
        deadline_ms: int,
        top_k: int = 10,
    ) -> list[Snippet]:
        results: list[Snippet] = []
        for doc_id, doc in _CORPUS.items():
            score = _score_for(prompt, doc["keywords"])
            if score <= 0.1:
                continue
            results.append(
                Snippet(
                    content=doc["text"][:200],
                    score=score,
                    ref=Ref(self.name, "read", {"doc_id": doc_id}),
                    acl=list(doc["acl"]),
                    source=self.name,
                    title=doc["title"],
                    last_modified=_CORPUS_TIMESTAMP,
                    etag=doc["etag"],
                )
            )
        results.sort(key=lambda s: s.score, reverse=True)
        return results[:top_k]

    async def search(
        self,
        query: str,
        user_ctx: UserContext,
        **filters,
    ) -> list[Ref]:
        return [
            Ref(self.name, "read", {"doc_id": doc_id})
            for doc_id, doc in _CORPUS.items()
            if _score_for(query, doc["keywords"]) > 0.1
        ]

    async def read(self, ref: Ref, user_ctx: UserContext) -> Content:
        doc_id = ref.args.get("doc_id")
        if doc_id not in _CORPUS:
            raise KeyError(f"Stub corpus has no doc_id={doc_id!r}")
        doc = _CORPUS[doc_id]
        return Content(
            text=doc["text"],
            ref=ref,
            acl=list(doc["acl"]),
            title=doc["title"],
            last_modified=_CORPUS_TIMESTAMP,
            etag=doc["etag"],
            mime_type="text/plain",
        )

    async def list(
        self,
        user_ctx: UserContext,
        path: str | None = None,
    ) -> list[Ref]:
        return [Ref(self.name, "read", {"doc_id": k}) for k in _CORPUS]

    async def get_permissions(
        self,
        ref: Ref,
        user_ctx: UserContext,
    ) -> list[str]:
        doc_id = ref.args.get("doc_id")
        if doc_id not in _CORPUS:
            return []
        return list(_CORPUS[doc_id]["acl"])

    def chunk(self, content: Content) -> list[Chunk]:
        sentences = [s.strip() for s in content.text.split(". ") if s.strip()]
        return [
            Chunk(
                text=s + ("." if not s.endswith(".") else ""),
                position=i,
            )
            for i, s in enumerate(sentences)
        ]
