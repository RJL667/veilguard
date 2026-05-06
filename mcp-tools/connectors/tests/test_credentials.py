"""Tests for the per-user credential resolver framework.

Covers:
  * Token expiry semantics (incl. leeway).
  * StaticCredentialResolver — happy path + missing/expired errors.
  * HttpCredentialResolver caching, lock coalescing, invalidation.
  * Response decoding for the LibreChat endpoint contract (200, 404,
    410, malformed body).

The HttpCredentialResolver tests stub `_fetch_from_librechat` or
`_parse_response` directly — no real httpx round trip. The actual
network/JSON path is exercised via the response-decoding tests, which
construct synthetic response objects.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
import time

import pytest

_CONNECTORS_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CONNECTORS_DIR))

from _base import (  # noqa: E402
    HttpCredentialResolver,
    OAuthToken,
    ReauthenticationRequiredError,
    StaticCredentialResolver,
)


def _run(coro):
    return asyncio.run(coro)


# ─── OAuthToken expiry ────────────────────────────────────────────────


class TestOAuthTokenExpiry:
    def test_no_expiry_never_expired(self):
        assert not OAuthToken("a").is_expired()

    def test_future_expiry_not_expired(self):
        assert not OAuthToken("a", expires_at=time.time() + 3600).is_expired()

    def test_past_expiry_expired(self):
        assert OAuthToken("a", expires_at=time.time() - 100).is_expired()

    def test_leeway_treats_near_expiry_as_expired(self):
        t = OAuthToken("a", expires_at=time.time() + 5)
        assert t.is_expired(leeway_seconds=30)
        assert not t.is_expired(leeway_seconds=1)


# ─── StaticCredentialResolver ─────────────────────────────────────────


class TestStaticCredentialResolver:
    def test_returns_token(self):
        r = StaticCredentialResolver()
        r.set("u1", "sharepoint", OAuthToken("tok"))
        result = _run(r.get_oauth_token("u1", "sharepoint"))
        assert result.access_token == "tok"

    def test_missing_token_raises(self):
        r = StaticCredentialResolver()
        with pytest.raises(ReauthenticationRequiredError) as excinfo:
            _run(r.get_oauth_token("u1", "sharepoint"))
        assert excinfo.value.server_name == "sharepoint"

    def test_expired_token_raises(self):
        r = StaticCredentialResolver()
        r.set("u1", "sharepoint", OAuthToken("tok", expires_at=time.time() - 100))
        with pytest.raises(ReauthenticationRequiredError):
            _run(r.get_oauth_token("u1", "sharepoint"))

    def test_remove_drops_token(self):
        r = StaticCredentialResolver()
        r.set("u1", "sp", OAuthToken("a"))
        r.remove("u1", "sp")
        with pytest.raises(ReauthenticationRequiredError):
            _run(r.get_oauth_token("u1", "sp"))

    def test_isolated_per_user(self):
        r = StaticCredentialResolver()
        r.set("u1", "sp", OAuthToken("u1tok"))
        r.set("u2", "sp", OAuthToken("u2tok"))
        assert _run(r.get_oauth_token("u1", "sp")).access_token == "u1tok"
        assert _run(r.get_oauth_token("u2", "sp")).access_token == "u2tok"


# ─── HttpCredentialResolver — cache + locking ─────────────────────────


class TestHttpCredentialResolverCache:
    def test_constructor_rejects_empty_url(self):
        with pytest.raises(ValueError):
            HttpCredentialResolver("")

    def test_strips_trailing_slash_from_url(self):
        r = HttpCredentialResolver("http://lc/")
        assert r.base_url == "http://lc"

    def test_caches_within_ttl(self):
        r = HttpCredentialResolver("http://lc")
        calls = []

        async def fake_fetch(user_id, server_name):
            calls.append((user_id, server_name))
            return OAuthToken("tok")

        r._fetch_from_librechat = fake_fetch  # type: ignore[method-assign]

        async def main():
            t1 = await r.get_oauth_token("u1", "sp")
            t2 = await r.get_oauth_token("u1", "sp")
            t3 = await r.get_oauth_token("u1", "sp")
            return t1, t2, t3

        results = _run(main())
        assert all(t.access_token == "tok" for t in results)
        assert len(calls) == 1  # only one fetch, others cached

    def test_invalidate_forces_refetch(self):
        r = HttpCredentialResolver("http://lc")
        counter = {"n": 0}

        async def fake_fetch(user_id, server_name):
            counter["n"] += 1
            return OAuthToken(f"tok{counter['n']}")

        r._fetch_from_librechat = fake_fetch  # type: ignore[method-assign]

        async def main():
            t1 = await r.get_oauth_token("u1", "sp")
            r.invalidate("u1", "sp")
            t2 = await r.get_oauth_token("u1", "sp")
            return t1, t2

        t1, t2 = _run(main())
        assert t1.access_token == "tok1"
        assert t2.access_token == "tok2"

    def test_expired_token_in_cache_refetched(self):
        r = HttpCredentialResolver("http://lc")
        # Hand-seed cache with an already-expired token
        r._cache[("u1", "sp")] = (
            OAuthToken("expired", expires_at=time.time() - 100),
            time.time(),  # cache TTL is fresh, but token is stale
        )

        async def fake_fetch(user_id, server_name):
            return OAuthToken("fresh", expires_at=time.time() + 3600)

        r._fetch_from_librechat = fake_fetch  # type: ignore[method-assign]

        result = _run(r.get_oauth_token("u1", "sp"))
        assert result.access_token == "fresh"

    def test_ttl_expiry_refetches(self):
        r = HttpCredentialResolver("http://lc", cache_ttl_seconds=0.05)
        counter = {"n": 0}

        async def fake_fetch(user_id, server_name):
            counter["n"] += 1
            return OAuthToken(f"tok{counter['n']}")

        r._fetch_from_librechat = fake_fetch  # type: ignore[method-assign]

        async def main():
            await r.get_oauth_token("u1", "sp")
            await asyncio.sleep(0.1)
            await r.get_oauth_token("u1", "sp")

        _run(main())
        assert counter["n"] == 2

    def test_concurrent_fetches_coalesce(self):
        r = HttpCredentialResolver("http://lc")
        counter = {"n": 0}

        async def fake_fetch(user_id, server_name):
            counter["n"] += 1
            await asyncio.sleep(0.05)  # simulate latency
            return OAuthToken("tok")

        r._fetch_from_librechat = fake_fetch  # type: ignore[method-assign]

        async def main():
            # Five concurrent calls for same key — should produce 1 fetch
            results = await asyncio.gather(
                *[r.get_oauth_token("u1", "sp") for _ in range(5)]
            )
            return results

        results = _run(main())
        assert all(t.access_token == "tok" for t in results)
        assert counter["n"] == 1

    def test_empty_user_id_raises_immediately(self):
        r = HttpCredentialResolver("http://lc")
        with pytest.raises(ReauthenticationRequiredError):
            _run(r.get_oauth_token("", "sp"))


# ─── HttpCredentialResolver — response decoding ───────────────────────


class _FakeResp:
    def __init__(self, status_code: int, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        if callable(self._body):
            return self._body()
        return self._body


class TestHttpCredentialResolverResponse:
    def test_200_with_full_payload(self):
        r = HttpCredentialResolver("http://lc")
        resp = _FakeResp(200, {
            "access_token": "AT",
            "expires_at": time.time() + 3600,
            "scopes": ["read", "write"],
        })
        token = r._parse_response("sp", 200, resp)
        assert token.access_token == "AT"
        assert token.expires_at is not None
        assert token.scopes == ["read", "write"]

    def test_200_without_expires_or_scopes(self):
        r = HttpCredentialResolver("http://lc")
        resp = _FakeResp(200, {"access_token": "AT"})
        token = r._parse_response("sp", 200, resp)
        assert token.access_token == "AT"
        assert token.expires_at is None
        assert token.scopes == []

    def test_404_raises_no_connection(self):
        r = HttpCredentialResolver("http://lc")
        resp = _FakeResp(404, {"error": "not connected"})
        with pytest.raises(ReauthenticationRequiredError) as excinfo:
            r._parse_response("sp", 404, resp)
        assert "not connected" in excinfo.value.reason.lower()

    def test_410_raises_expired(self):
        r = HttpCredentialResolver("http://lc")
        resp = _FakeResp(410, {"error": "expired"})
        with pytest.raises(ReauthenticationRequiredError) as excinfo:
            r._parse_response("sp", 410, resp)
        assert "expired" in excinfo.value.reason.lower()

    def test_unexpected_status_raises(self):
        r = HttpCredentialResolver("http://lc")
        resp = _FakeResp(500, {"error": "boom"})
        with pytest.raises(ReauthenticationRequiredError) as excinfo:
            r._parse_response("sp", 500, resp)
        assert "500" in excinfo.value.reason

    def test_invalid_json_raises(self):
        r = HttpCredentialResolver("http://lc")

        def boom():
            raise ValueError("not json")

        resp = _FakeResp(200, boom)
        with pytest.raises(ReauthenticationRequiredError):
            r._parse_response("sp", 200, resp)

    def test_missing_access_token_raises(self):
        r = HttpCredentialResolver("http://lc")
        resp = _FakeResp(200, {"expires_at": 0})
        with pytest.raises(ReauthenticationRequiredError) as excinfo:
            r._parse_response("sp", 200, resp)
        assert "access_token" in excinfo.value.reason

    def test_non_object_body_raises(self):
        r = HttpCredentialResolver("http://lc")
        resp = _FakeResp(200, ["not", "an", "object"])
        with pytest.raises(ReauthenticationRequiredError):
            r._parse_response("sp", 200, resp)

    def test_invalid_expires_at_falls_back_to_none(self):
        r = HttpCredentialResolver("http://lc")
        resp = _FakeResp(200, {
            "access_token": "AT",
            "expires_at": "not-a-number",
        })
        token = r._parse_response("sp", 200, resp)
        assert token.access_token == "AT"
        assert token.expires_at is None
