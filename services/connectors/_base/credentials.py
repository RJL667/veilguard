"""Per-user OAuth credential resolution for connectors.

Connectors talk to external systems (SharePoint, Slack, Jira, ...) on
behalf of the current user. Each user's OAuth token lives in
LibreChat's MongoDB via ``MCPTokenStorage`` (encrypted with
``CREDS_KEY`` using AES-256-CTR). Python MCP tools cannot read that
storage directly — encryption keys and Mongo connection details are
Node.js-private — so we resolve tokens via an internal HTTP endpoint
exposed by LibreChat on the local/private network.

LibreChat-side endpoint contract (to be implemented in
``librechat-src/api/server/routes/mcp.js``)::

    GET  /api/mcp/internal/oauth/token/:userId/:serverName
    Auth: localhost / private-network bind only
          (service-to-service trust; the network IS the boundary)

    200 -> {"access_token": str,
            "expires_at": int|null,   # epoch seconds
            "scopes":     list[str]}
    404 -> {"error": "no token for user/server"}
    410 -> {"error": "token expired, reauthentication required"}

The Python side here is the consumer of that contract. Until the
endpoint exists, connectors can use :class:`StaticCredentialResolver`
in tests / bring-up — production deployments must use
:class:`HttpCredentialResolver` with the real endpoint.

Cache: ``HttpCredentialResolver`` caches tokens in-process per
(user_id, server_name) for ``cache_ttl_seconds`` (default 5 min).
Cache is invalidated automatically when a token's ``expires_at``
passes (with a short leeway so tokens about to expire force a refetch
rather than failing the next API call).
"""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class ReauthenticationRequiredError(Exception):
    """Raised when a user's OAuth token is missing, revoked, or expired
    beyond automatic refresh.

    The connector should surface a clear "reconnect <connector>"
    instruction to the LLM (and ultimately the user). Never silently
    fail — silent failure looks like the connector returned no
    matches, which is indistinguishable from a real empty result and
    leaves the user wondering why their data isn't surfacing.
    """

    def __init__(self, server_name: str, reason: str):
        self.server_name = server_name
        self.reason = reason
        super().__init__(f"reauth required for {server_name!r}: {reason}")


@dataclass
class OAuthToken:
    access_token: str
    expires_at: float | None = None  # epoch seconds; None = no expiry known
    scopes: list[str] = field(default_factory=list)

    def is_expired(self, leeway_seconds: float = 30.0) -> bool:
        """A token is "expired" when ``expires_at`` is in the past, or
        within ``leeway_seconds`` of the current time. Leeway forces a
        refetch slightly before actual expiry to avoid races where the
        cached token is fresh enough at cache-check time but stale by
        the time the API call lands."""
        if self.expires_at is None:
            return False
        return time.time() >= self.expires_at - leeway_seconds


class CredentialResolver(ABC):
    """Strategy interface for fetching per-user OAuth tokens.

    Connectors take a resolver in their constructor (or read the
    process default). Tests inject :class:`StaticCredentialResolver`
    with canned tokens; production uses :class:`HttpCredentialResolver`.
    """

    @abstractmethod
    async def get_oauth_token(self, user_id: str, server_name: str) -> OAuthToken:
        """Resolve the user's token for the named connector/server.

        Raises:
            ReauthenticationRequiredError: token missing, revoked, or
                expired beyond the resolver's refresh ability.
        """


class StaticCredentialResolver(CredentialResolver):
    """In-memory resolver. For unit tests and pre-LibreChat-endpoint bring-up.

    Not safe for production: there is no encryption, no rotation, no
    audit trail. Useful when developing a connector locally with a
    single hand-issued token.
    """

    def __init__(self, tokens: dict[tuple[str, str], OAuthToken] | None = None):
        self._tokens: dict[tuple[str, str], OAuthToken] = dict(tokens or {})

    def set(self, user_id: str, server_name: str, token: OAuthToken) -> None:
        self._tokens[(user_id, server_name)] = token

    def remove(self, user_id: str, server_name: str) -> None:
        self._tokens.pop((user_id, server_name), None)

    async def get_oauth_token(self, user_id: str, server_name: str) -> OAuthToken:
        token = self._tokens.get((user_id, server_name))
        if token is None:
            raise ReauthenticationRequiredError(
                server_name, "no static token configured"
            )
        if token.is_expired():
            raise ReauthenticationRequiredError(
                server_name, "static token expired (set a fresh one)"
            )
        return token


class HttpCredentialResolver(CredentialResolver):
    """Fetches OAuth tokens from LibreChat's internal endpoint.

    Tokens are cached in-process per (user_id, server_name) for
    ``cache_ttl_seconds`` (default 5 minutes) — short enough to pick
    up rotation quickly, long enough to avoid hammering LibreChat on
    every recall fan-out. The cache is per-MCP-server-process; each
    server has its own.

    Concurrent fetches for the same key are coalesced via
    :class:`asyncio.Lock` so a thundering herd of concurrent recall
    calls produces at most one HTTP request per unique key.
    """

    def __init__(
        self,
        librechat_base_url: str,
        *,
        cache_ttl_seconds: float = 300.0,
        request_timeout_seconds: float = 5.0,
    ):
        if not librechat_base_url:
            raise ValueError("librechat_base_url must be non-empty")
        self._base_url = librechat_base_url.rstrip("/")
        self._cache_ttl = cache_ttl_seconds
        self._timeout = request_timeout_seconds
        # Cache: (user_id, server_name) -> (token, fetched_at_epoch)
        self._cache: dict[tuple[str, str], tuple[OAuthToken, float]] = {}
        self._lock = asyncio.Lock()

    @property
    def base_url(self) -> str:
        return self._base_url

    async def get_oauth_token(self, user_id: str, server_name: str) -> OAuthToken:
        if not user_id:
            raise ReauthenticationRequiredError(server_name, "user_id is empty")

        key = (user_id, server_name)

        # Fast-path cache check without holding the lock — racy reads
        # are safe; worst case is one extra HTTP call.
        cached = self._cache.get(key)
        if cached and self._is_cache_fresh(cached):
            return cached[0]

        # Coalesce concurrent fetches for the same key.
        async with self._lock:
            cached = self._cache.get(key)
            if cached and self._is_cache_fresh(cached):
                return cached[0]
            token = await self._fetch_from_librechat(user_id, server_name)
            self._cache[key] = (token, time.time())
            return token

    def _is_cache_fresh(self, entry: tuple[OAuthToken, float]) -> bool:
        token, fetched_at = entry
        if time.time() - fetched_at >= self._cache_ttl:
            return False
        if token.is_expired():
            return False
        return True

    async def _fetch_from_librechat(
        self, user_id: str, server_name: str
    ) -> OAuthToken:
        # Lazy import: httpx is only required when production HTTP is used,
        # not for tests that drive resolution through StaticCredentialResolver.
        try:
            import httpx
        except ImportError as e:
            raise RuntimeError(
                "httpx is required for HttpCredentialResolver. "
                "Install it (`pip install httpx`) or use "
                "StaticCredentialResolver in tests."
            ) from e

        url = (
            f"{self._base_url}/api/mcp/internal/oauth/token/"
            f"{user_id}/{server_name}"
        )

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url)
        except httpx.HTTPError as e:
            # Network-level failure (DNS, connection refused, timeout).
            # Treat as a reauth signal so the connector emits a clear
            # error rather than swallowing — operator likely needs to
            # check that LibreChat is reachable from this process.
            raise ReauthenticationRequiredError(
                server_name,
                f"credential endpoint unreachable: {type(e).__name__}",
            ) from e

        return self._parse_response(server_name, resp.status_code, resp)

    def _parse_response(self, server_name: str, status: int, resp) -> OAuthToken:
        """Decode an httpx response. Split out so tests can drive it
        with hand-crafted response objects."""
        if status == 404:
            raise ReauthenticationRequiredError(
                server_name,
                "user has not connected this source",
            )
        if status == 410:
            raise ReauthenticationRequiredError(
                server_name,
                "token expired and refresh failed — user must reconnect",
            )
        if status != 200:
            raise ReauthenticationRequiredError(
                server_name,
                f"credential endpoint returned HTTP {status}",
            )

        try:
            data = resp.json()
        except Exception as e:
            raise ReauthenticationRequiredError(
                server_name,
                f"credential endpoint returned non-JSON body: {type(e).__name__}",
            ) from e

        if not isinstance(data, dict):
            raise ReauthenticationRequiredError(
                server_name, "credential endpoint returned non-object body"
            )

        access_token = data.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise ReauthenticationRequiredError(
                server_name, "credential endpoint response missing access_token"
            )

        expires_at = data.get("expires_at")
        if expires_at is not None:
            try:
                expires_at = float(expires_at)
            except (TypeError, ValueError):
                expires_at = None

        scopes_raw = data.get("scopes") or []
        if isinstance(scopes_raw, list):
            scopes = [str(s) for s in scopes_raw]
        else:
            scopes = []

        return OAuthToken(
            access_token=access_token,
            expires_at=expires_at,
            scopes=scopes,
        )

    def invalidate(self, user_id: str, server_name: str) -> None:
        """Drop a cache entry — call after a connector receives a 401
        from the upstream API to force a refetch on next call."""
        self._cache.pop((user_id, server_name), None)

    def clear_cache(self) -> None:
        self._cache.clear()
