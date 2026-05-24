"""Connector abstract base class.

A Connector is an integration with an external data source (SharePoint,
Slack, Jira, ...) exposing two roles:

  1. Server-side ``hint(prompt)`` for recall-time candidate generation.
  2. LLM-facing tools (``search``/``read``/``list``/``write``) registered as
     MCP tools by the connector's FastMCP server.

Subclasses implement only the methods listed in their ``capabilities``. Any
method not in capabilities raises NotImplementedError. The recall fan-out
inspects ``capabilities`` to decide whether to call ``hint()`` at all.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from .types import (
    Capability,
    Chunk,
    Content,
    Ref,
    Snippet,
    UserContext,
)


class Connector(ABC):
    name: str = ""
    version: str = "0.0.0"
    capabilities: set[Capability] = set()

    # ─── recall-time ──────────────────────────────────────────────────
    async def hint(
        self,
        prompt: str,
        user_ctx: UserContext,
        deadline_ms: int,
        top_k: int = 10,
    ) -> list[Snippet]:
        """Return ranked hint snippets for ``prompt`` constrained to the
        user's permissions in the source system. MUST respect
        ``deadline_ms`` — return partial results rather than block."""
        raise NotImplementedError(f"{self.name}: HINT not supported")

    # ─── LLM-facing tools ─────────────────────────────────────────────
    async def search(
        self,
        query: str,
        user_ctx: UserContext,
        **filters,
    ) -> list[Ref]:
        raise NotImplementedError(f"{self.name}: SEARCH not supported")

    async def read(
        self,
        ref: Ref,
        user_ctx: UserContext,
    ) -> Content:
        raise NotImplementedError(f"{self.name}: READ not supported")

    async def list(
        self,
        user_ctx: UserContext,
        path: str | None = None,
    ) -> list[Ref]:
        raise NotImplementedError(f"{self.name}: LIST not supported")

    # ─── permissions / ACL ────────────────────────────────────────────
    async def get_permissions(
        self,
        ref: Ref,
        user_ctx: UserContext,
    ) -> list[str]:
        """Return principal IDs (group/user IDs) allowed to view this
        resource. Used at ingest to tag the TCMM entry's acl, and at recall
        as defense-in-depth against ACL drift."""
        raise NotImplementedError(f"{self.name}: PERMISSIONS not supported")

    # ─── ingest helpers ───────────────────────────────────────────────
    @abstractmethod
    def chunk(self, content: Content) -> list[Chunk]:
        """Split ``content`` for embedding. Each connector picks chunk
        boundaries that match its content type — one message per Slack
        chunk, semantic split for SharePoint docs, row-grouped for tabular
        sources, etc."""

    def calibrate(self, raw_score: float) -> float:
        """Map a connector-native score to [0, 1]. Default: clamp + identity.
        Override when the source returns a non-uniform scale (BM25, vendor
        scores, etc.) so the reranker fuses fairly across connectors."""
        return max(0.0, min(1.0, raw_score))

    # ─── ops ──────────────────────────────────────────────────────────
    async def healthcheck(self) -> dict:
        return {"name": self.name, "version": self.version, "ok": True}
