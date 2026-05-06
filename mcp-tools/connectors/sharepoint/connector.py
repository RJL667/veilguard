"""SharePointConnector — Veilguard's first real connector.

Implements the connector framework's `Connector` ABC against
Microsoft Graph + SharePoint. Per-user OAuth tokens come from a
:class:`CredentialResolver` (production: HttpCredentialResolver
calling LibreChat's internal token endpoint). The MS Graph client is
injected so tests can mock it without touching the real Graph API.

Capabilities (V1):
  * HINT, SEARCH, READ, LIST, PERMISSIONS

WRITE is intentionally absent for V1 — read-only connector. Adding
upload / update is a Graph endpoint addition + a write tool on the
MCP server, both small once the read path is solid.
"""
from __future__ import annotations

import logging
import pathlib
import sys
from typing import Any

_THIS_DIR = pathlib.Path(__file__).resolve().parent
# Add both `connectors/` (so ``_base`` resolves as a top-level package)
# and `sharepoint/` (so sibling modules like ``graph`` resolve as
# top-level). server.py runs us as a script (cwd=sharepoint) per its
# module docstring, so bare imports must work without a parent package.
sys.path.insert(0, str(_THIS_DIR.parent))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _base import (  # noqa: E402
    Capability,
    Chunk,
    Connector,
    Content,
    CredentialResolver,
    Ref,
    Snippet,
    UserContext,
)
from _base.parsing import chunk_text, parse_to_text  # noqa: E402
from graph import (  # noqa: E402
    GraphError,
    GraphNotFoundError,
    SearchHit,
    SharePointGraphClient,
)


logger = logging.getLogger("connectors.sharepoint")


class SharePointConnector(Connector):
    """SharePoint connector — see module docstring for the design."""

    name = "sharepoint"
    version = "0.1.0"
    capabilities = {
        Capability.HINT,
        Capability.SEARCH,
        Capability.READ,
        Capability.LIST,
        Capability.PERMISSIONS,
    }

    def __init__(
        self,
        *,
        credentials: CredentialResolver,
        graph: SharePointGraphClient | None = None,
    ):
        self._credentials = credentials
        self._graph = graph or SharePointGraphClient()

    # ─── helpers ──────────────────────────────────────────────────

    async def _token_for(self, user_ctx: UserContext) -> str:
        """Resolve the user's OAuth access token for SharePoint."""
        if not user_ctx.user_id:
            # Without a user we have no per-user OAuth token. Surface
            # via the same error path so the connector emits a clear
            # signal rather than a generic "no results".
            from _base import ReauthenticationRequiredError
            raise ReauthenticationRequiredError(
                self.name, "user_id missing on UserContext"
            )
        token = await self._credentials.get_oauth_token(
            user_ctx.user_id, self.name
        )
        return token.access_token

    def _hit_to_snippet(self, hit: SearchHit) -> Snippet:
        return Snippet(
            content=hit.summary or hit.name,
            score=self.calibrate(hit.score),
            ref=Ref(
                connector=self.name,
                tool="read",
                args={"item_id": hit.item_id, "drive_id": hit.drive_id or ""},
            ),
            acl=[],  # V1: ACLs resolved on-demand via get_permissions(),
                     # not at hint time. Hint cost is one Graph call;
                     # ACL resolution is one extra Graph call per hit
                     # which we defer until ingest.
            source=self.name,
            title=hit.name or None,
            last_modified=_parse_iso(hit.last_modified),
            etag=None,  # Graph search doesn't return etag; populated on read()
        )

    # ─── recall-time ──────────────────────────────────────────────

    async def hint(
        self,
        prompt: str,
        user_ctx: UserContext,
        deadline_ms: int,
        top_k: int = 10,
    ) -> list[Snippet]:
        if not prompt.strip():
            return []
        access_token = await self._token_for(user_ctx)
        try:
            hits = await self._graph.search(access_token, prompt, top=top_k)
        except GraphError as e:
            # Hint failures are non-fatal — recall fan-out logs and
            # drops this connector for the turn (see gather_hints).
            # Re-raise GraphError so the breaker tracks it; a transient
            # rate-limit shouldn't take SharePoint out for a minute,
            # so we let recall.py decide the policy.
            logger.info(
                f"[sharepoint] hint() Graph call failed "
                f"(status={e.status}): {e}"
            )
            raise
        return [self._hit_to_snippet(h) for h in hits]

    # ─── LLM-facing tools ─────────────────────────────────────────

    async def search(
        self,
        query: str,
        user_ctx: UserContext,
        **filters,
    ) -> list[Ref]:
        access_token = await self._token_for(user_ctx)
        top = int(filters.get("top", 10))
        hits = await self._graph.search(access_token, query, top=top)
        return [
            Ref(
                connector=self.name,
                tool="read",
                args={"item_id": h.item_id, "drive_id": h.drive_id or ""},
            )
            for h in hits
        ]

    async def read(self, ref: Ref, user_ctx: UserContext) -> Content:
        item_id, drive_id = self._extract_ref(ref)
        access_token = await self._token_for(user_ctx)

        file = await self._graph.download(
            access_token, drive_id=drive_id, item_id=item_id,
        )

        text = parse_to_text(file.raw, file.name, mime_type=file.mime_type)

        # Resolve ACL at read time. If the call fails (permissions API
        # quirks, scope issues), surface an empty ACL — the recall-side
        # filter treats empty ACL as "no per-user constraint", which is
        # safe because the user already had to be authorized to download
        # the file in the first place.
        try:
            acl = await self._graph.get_permissions(
                access_token, drive_id=drive_id, item_id=item_id,
            )
        except GraphError as e:
            logger.warning(
                f"[sharepoint] permissions() failed for item={item_id}: "
                f"{type(e).__name__} status={e.status}"
            )
            acl = []

        return Content(
            text=text,
            ref=ref,
            acl=acl,
            title=file.name or None,
            last_modified=_parse_iso(file.last_modified),
            etag=file.etag,
            mime_type=file.mime_type,
            extra={
                "size": file.size,
            },
        )

    async def list(
        self,
        user_ctx: UserContext,
        path: str | None = None,
    ) -> list[Ref]:
        # ``path`` is encoded as ``drive_id`` or ``drive_id/item_id`` —
        # Graph's drive paths need both pieces. For the minimum viable
        # interface, treat the path as a drive_id and list its root.
        if not path:
            return []
        access_token = await self._token_for(user_ctx)
        drive_id, _, item_id = path.partition("/")
        items = await self._graph.list_folder(
            access_token,
            drive_id=drive_id,
            item_id=item_id or None,
        )
        return [
            Ref(
                connector=self.name,
                tool="read",
                args={"item_id": i.item_id, "drive_id": i.drive_id or drive_id},
            )
            for i in items
            if not i.is_folder
        ]

    async def get_permissions(
        self,
        ref: Ref,
        user_ctx: UserContext,
    ) -> list[str]:
        item_id, drive_id = self._extract_ref(ref)
        access_token = await self._token_for(user_ctx)
        try:
            return await self._graph.get_permissions(
                access_token, drive_id=drive_id, item_id=item_id,
            )
        except GraphNotFoundError:
            return []

    # ─── ingest helpers ───────────────────────────────────────────

    def chunk(self, content: Content) -> list[Chunk]:
        """Sentence-aware, token-bounded chunking via shared LlamaIndex
        wrapper (``_base.parsing.chunk_text``).

        Defaults: 512-token chunks with 64-token overlap — matches the
        training distribution of common embedding models (BGE, MiniLM,
        mpnet). Empty input returns []. Falls back to a paragraph
        split if LlamaIndex isn't installed.
        """
        text = content.text or ""
        chunks_text = chunk_text(text)
        return [
            Chunk(text=t, position=i) for i, t in enumerate(chunks_text)
        ]

    def calibrate(self, raw_score: float) -> float:
        """Normalize Graph search rank to [0, 1].

        Graph's `rank` field is a 1-based ordinal (1 = best match) for
        most search responses. Higher-quality APIs return a `score`
        value (0..1ish or arbitrary). We clamp to [0, 1] and treat
        rank-as-ordinal by flipping (1 → 1.0, 10 → 0.1).
        """
        if raw_score <= 0:
            return 0.0
        # Heuristic: values > 1 are likely 1-based ranks; map to 1/rank
        if raw_score >= 1:
            return min(1.0, 1.0 / raw_score)
        return min(1.0, max(0.0, raw_score))

    # ─── internal ─────────────────────────────────────────────────

    @staticmethod
    def _extract_ref(ref: Ref) -> tuple[str, str]:
        """Pull (item_id, drive_id) out of a Ref. Both are required for
        SharePoint operations; we accept ``site_id`` as a synonym for
        ``drive_id`` since some search responses surface it that way."""
        if ref.connector != "sharepoint":
            raise ValueError(
                f"SharePointConnector cannot handle ref for {ref.connector!r}"
            )
        args = ref.args or {}
        item_id = args.get("item_id") or ""
        drive_id = args.get("drive_id") or args.get("site_id") or ""
        if not item_id or not drive_id:
            raise ValueError(
                f"sharepoint ref requires item_id and drive_id, got args={args!r}"
            )
        return item_id, drive_id


def _parse_iso(s: str | None):
    """Best-effort ISO-8601 parse. Returns None on failure rather than
    raising — connector outputs should not break on a missing or
    weird timestamp."""
    if not s:
        return None
    try:
        from datetime import datetime
        # Graph returns ...Z; fromisoformat handles +00:00 not Z
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None
