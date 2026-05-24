"""Microsoft Graph API client for SharePoint operations.

Thin async wrapper over the small subset of Graph endpoints the
SharePoint connector uses. Single source of truth for HTTP shape,
error mapping, and Graph-specific quirks (token-vs-driveItem search,
hostname/site URL parsing, permissions response format).

Designed to be mocked in tests via the ``client_factory`` argument
on :class:`SharePointGraphClient` — test fixtures inject an httpx
MockTransport without re-implementing this class.

Error mapping:

  * 401 / 403  → :class:`ReauthenticationRequiredError`
                 (token missing/expired or scope insufficient)
  * 404        → :class:`GraphNotFoundError`
                 (item does not exist or is hidden from this user)
  * 429        → :class:`GraphRateLimitedError`
                 (caller decides whether to retry / drop)
  * 5xx        → :class:`GraphServerError`
  * Network    → :class:`GraphTransportError`
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import sys
import pathlib

# Make the framework importable for ReauthenticationRequiredError.
_THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))

from _base import ReauthenticationRequiredError  # noqa: E402


logger = logging.getLogger("connectors.sharepoint.graph")


GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"


# ─── exceptions ───────────────────────────────────────────────────────


class GraphError(Exception):
    """Base class for SharePoint Graph errors."""

    def __init__(self, status: int | None, message: str):
        self.status = status
        super().__init__(message)


class GraphNotFoundError(GraphError):
    """Item not found (or not visible to this user)."""


class GraphRateLimitedError(GraphError):
    """Too many requests — caller decides whether to retry."""


class GraphServerError(GraphError):
    """5xx from Graph — typically transient."""


class GraphTransportError(GraphError):
    """Network-level failure — connection refused, DNS, timeout."""


# ─── result types ─────────────────────────────────────────────────────


@dataclass
class SearchHit:
    """One driveItem result from Graph search."""

    item_id: str
    drive_id: str | None
    name: str
    summary: str
    score: float                      # Graph's rank score
    last_modified: str | None         # ISO 8601, kept opaque
    web_url: str | None


@dataclass
class FileContent:
    """Result of a file download."""

    raw: bytes
    name: str
    mime_type: str | None
    etag: str | None
    last_modified: str | None
    size: int


@dataclass
class FolderItem:
    """One item in a folder listing."""

    item_id: str
    drive_id: str | None
    name: str
    is_folder: bool
    size: int


# ─── client ───────────────────────────────────────────────────────────


class SharePointGraphClient:
    """Async client for the SharePoint endpoints we care about.

    A single instance can be reused across many calls — it doesn't
    hold connections; httpx clients are created per request to avoid
    leaking sockets across connector readers/listers.

    For tests, pass a ``client_factory`` that returns a configured
    httpx.AsyncClient (typically with a MockTransport). Defaults to
    a real client.
    """

    def __init__(
        self,
        *,
        base_url: str = GRAPH_BASE_URL,
        request_timeout_seconds: float = 15.0,
        client_factory: Callable | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = request_timeout_seconds
        self._client_factory = client_factory

    # ── auth ──────────────────────────────────────────────────────

    @staticmethod
    def _bearer(access_token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {access_token}"}

    # ── error mapping ─────────────────────────────────────────────

    @staticmethod
    def _raise_for_status(status: int, body: str, server_name: str) -> None:
        if status == 200 or status == 201 or status == 204:
            return
        if status in (401, 403):
            # Token missing scope, revoked, or expired beyond refresh.
            # Surface as a reauth signal so the connector can return a
            # clear "reconnect SharePoint" error to the LLM.
            raise ReauthenticationRequiredError(
                server_name,
                f"Graph returned {status}: {body[:200]}",
            )
        if status == 404:
            raise GraphNotFoundError(status, f"Graph 404: {body[:200]}")
        if status == 429:
            raise GraphRateLimitedError(status, f"Graph 429: {body[:200]}")
        if 500 <= status < 600:
            raise GraphServerError(status, f"Graph {status}: {body[:200]}")
        raise GraphError(status, f"Graph unexpected status {status}: {body[:200]}")

    # ── HTTP plumbing ─────────────────────────────────────────────

    async def _request(
        self,
        method: str,
        url: str,
        *,
        access_token: str,
        json_body: dict | None = None,
        accept_binary: bool = False,
    ):
        try:
            import httpx
        except ImportError as e:
            raise RuntimeError(
                "httpx required for SharePointGraphClient. "
                "Install via the sharepoint/requirements.txt."
            ) from e

        headers = dict(self._bearer(access_token))
        if not accept_binary:
            headers["Accept"] = "application/json"

        if self._client_factory is not None:
            client_cm = self._client_factory()
        else:
            client_cm = httpx.AsyncClient(timeout=self._timeout)

        try:
            async with client_cm as client:
                resp = await client.request(
                    method, url, headers=headers, json=json_body
                )
        except httpx.HTTPError as e:
            raise GraphTransportError(
                None, f"network error calling Graph: {type(e).__name__}: {e}"
            ) from e

        self._raise_for_status(
            resp.status_code,
            "" if accept_binary else resp.text,
            "sharepoint",
        )
        return resp

    # ── operations ────────────────────────────────────────────────

    async def search(
        self,
        access_token: str,
        query: str,
        *,
        top: int = 10,
    ) -> list[SearchHit]:
        """POST /search/query with entityTypes=[driveItem]."""
        if not query.strip():
            return []

        body = {
            "requests": [
                {
                    "entityTypes": ["driveItem"],
                    "query": {"queryString": query},
                    "from": 0,
                    "size": top,
                }
            ]
        }
        url = f"{self._base_url}/search/query"
        resp = await self._request("POST", url, access_token=access_token, json_body=body)
        return self._parse_search_response(resp.json())

    @staticmethod
    def _parse_search_response(payload: dict) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for response in payload.get("value", []):
            for container in response.get("hitsContainers", []):
                for hit in container.get("hits", []):
                    resource = hit.get("resource") or {}
                    if resource.get("@odata.type") and "driveItem" not in resource["@odata.type"]:
                        continue
                    parent = resource.get("parentReference") or {}
                    drive_id = parent.get("driveId")
                    summary = hit.get("summary") or ""
                    score_raw = hit.get("rank") or hit.get("score") or 0
                    try:
                        score = float(score_raw)
                    except (TypeError, ValueError):
                        score = 0.0
                    item_id = resource.get("id") or ""
                    if not item_id:
                        continue
                    hits.append(
                        SearchHit(
                            item_id=item_id,
                            drive_id=drive_id,
                            name=resource.get("name") or "",
                            summary=summary,
                            score=score,
                            last_modified=resource.get("lastModifiedDateTime"),
                            web_url=resource.get("webUrl"),
                        )
                    )
        return hits

    async def download(
        self,
        access_token: str,
        *,
        drive_id: str,
        item_id: str,
    ) -> FileContent:
        """Fetch file content + metadata.

        Two-step: GET item metadata, then GET item content. Single-step
        download via /content would skip the metadata, but we need
        name + etag + size for the connector return value.
        """
        meta_url = f"{self._base_url}/drives/{drive_id}/items/{item_id}"
        meta_resp = await self._request("GET", meta_url, access_token=access_token)
        meta = meta_resp.json()

        content_url = f"{self._base_url}/drives/{drive_id}/items/{item_id}/content"
        content_resp = await self._request(
            "GET", content_url,
            access_token=access_token,
            accept_binary=True,
        )

        return FileContent(
            raw=content_resp.content,
            name=meta.get("name", ""),
            mime_type=(meta.get("file") or {}).get("mimeType"),
            etag=meta.get("eTag") or meta.get("cTag"),
            last_modified=meta.get("lastModifiedDateTime"),
            size=int(meta.get("size") or 0),
        )

    async def list_folder(
        self,
        access_token: str,
        *,
        drive_id: str,
        item_id: str | None = None,
    ) -> list[FolderItem]:
        """List children of a folder. ``item_id=None`` means drive root."""
        if item_id:
            url = f"{self._base_url}/drives/{drive_id}/items/{item_id}/children"
        else:
            url = f"{self._base_url}/drives/{drive_id}/root/children"
        resp = await self._request("GET", url, access_token=access_token)
        payload = resp.json()
        out: list[FolderItem] = []
        for entry in payload.get("value", []):
            out.append(
                FolderItem(
                    item_id=entry.get("id", ""),
                    drive_id=drive_id,
                    name=entry.get("name", ""),
                    is_folder="folder" in entry,
                    size=int(entry.get("size") or 0),
                )
            )
        return out

    async def get_permissions(
        self,
        access_token: str,
        *,
        drive_id: str,
        item_id: str,
    ) -> list[str]:
        """Return a list of principal IDs that can read this item.

        Graph's permissions endpoint returns mixed types: user IDs,
        group IDs, sharing-link tokens. We canonicalize to the
        ``"<kind>:<id>"`` form used elsewhere in Veilguard ACLs.
        """
        url = (
            f"{self._base_url}/drives/{drive_id}/items/{item_id}/permissions"
        )
        resp = await self._request("GET", url, access_token=access_token)
        payload = resp.json()
        principals: list[str] = []
        for perm in payload.get("value", []):
            granted = perm.get("grantedToV2") or perm.get("grantedTo") or {}
            user = granted.get("user") if isinstance(granted, dict) else None
            group = granted.get("group") if isinstance(granted, dict) else None
            if isinstance(user, dict) and user.get("id"):
                principals.append(f"user:{user['id']}")
            if isinstance(group, dict) and group.get("id"):
                principals.append(f"group:{group['id']}")
            for granted_id in perm.get("grantedToIdentitiesV2") or []:
                if not isinstance(granted_id, dict):
                    continue
                u = granted_id.get("user") if isinstance(granted_id, dict) else None
                g = granted_id.get("group") if isinstance(granted_id, dict) else None
                if isinstance(u, dict) and u.get("id"):
                    principals.append(f"user:{u['id']}")
                if isinstance(g, dict) and g.get("id"):
                    principals.append(f"group:{g['id']}")
        return principals
