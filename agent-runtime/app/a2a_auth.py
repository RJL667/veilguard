"""Phase 5 — OAuth + mTLS auth for /external/a2a/* (§Phase 5).

Spec: "API-key + (later) OAuth/mTLS."

This module adds two AUTH paths on top of the existing API-key gate
in `a2a_external.py`:

  1. OAuth/JWT — `Authorization: Bearer <jwt>`.
     Verified against a configured JWKS URL (PUBLIC keys only —
     server NEVER holds signing keys).  JWT `aud` must match
     `VEILGUARD_A2A_JWT_AUD` and `iss` must match
     `VEILGUARD_A2A_JWT_ISS`.  The JWT subject (`sub` or a
     configurable claim) maps to a tenant via the existing
     `a2a_external_keys` table — a JWT "binds" to a tenant the same
     way a key does.

  2. mTLS — `X-Client-Cert-Subject` header (set by the TLS-terminating
     reverse proxy, e.g. Caddy/nginx).  Maps the certificate CN to a
     tenant_id via a config table.  No JWT signature verification
     needed — the TLS layer already verified the chain.

Auth precedence at request time:
  1. `Authorization: Bearer ...` → JWT
  2. `X-Client-Cert-Subject: CN=…` → mTLS
  3. `X-API-Key: …` → existing API-key (a2a_external_keys lookup)

Returns the same authenticated row shape as the API-key path so
downstream call-sites (`/messages`, `/tasks`, `/inbox`) don't need
to know which auth method was used.

Defensive defaults — auth is OPT-IN per gate:
  * If `VEILGUARD_A2A_JWT_JWKS_URL` is unset → JWT path returns
    "unconfigured" and falls through to the next gate.
  * If `VEILGUARD_A2A_MTLS_ENABLED` is unset → mTLS path same.

This keeps the existing API-key flow working unchanged while
making OAuth/mTLS available when the operator turns them on.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional
from urllib.request import urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)


# ── Env-config ─────────────────────────────────────────────────────────


def _jwt_enabled() -> bool:
    return bool(os.environ.get("VEILGUARD_A2A_JWT_JWKS_URL", "").strip())


def _mtls_enabled() -> bool:
    return os.environ.get("VEILGUARD_A2A_MTLS_ENABLED", "").lower() in ("1", "true", "yes")


def _expected_aud() -> str:
    return os.environ.get("VEILGUARD_A2A_JWT_AUD", "").strip()


def _expected_iss() -> str:
    return os.environ.get("VEILGUARD_A2A_JWT_ISS", "").strip()


def _tenant_claim_name() -> str:
    """Which JWT claim carries the tenant_id.  Default 'sub'."""
    return os.environ.get("VEILGUARD_A2A_JWT_TENANT_CLAIM", "sub")


# ── JWKS cache ─────────────────────────────────────────────────────────


_JWKS_CACHE: dict[str, Any] = {"keys": None, "fetched_ts": 0.0, "ttl_s": 300}


def _fetch_jwks() -> Optional[dict[str, Any]]:
    """Fetch the JWKS document.  Cached for ~5 min so we don't hit
    the identity provider on every request.

    Returns None on failure — caller falls through to next auth path.
    """
    url = os.environ.get("VEILGUARD_A2A_JWT_JWKS_URL", "").strip()
    if not url:
        return None
    now = time.time()
    if (
        _JWKS_CACHE["keys"] is not None
        and (now - _JWKS_CACHE["fetched_ts"]) < _JWKS_CACHE["ttl_s"]
    ):
        return _JWKS_CACHE["keys"]
    try:
        with urlopen(url, timeout=5) as resp:
            payload = json.loads(resp.read())
        if not isinstance(payload, dict) or "keys" not in payload:
            return None
        _JWKS_CACHE["keys"] = payload
        _JWKS_CACHE["fetched_ts"] = now
        return payload
    except (URLError, json.JSONDecodeError, OSError) as e:
        logger.warning(f"[a2a_auth] JWKS fetch failed from {url}: {e}")
        return None


# ── JWT verification ───────────────────────────────────────────────────


def verify_jwt(token: str) -> Optional[dict[str, Any]]:
    """Verify a JWT against the configured JWKS, audience, and issuer.

    Returns the decoded payload on success, None on any failure.
    Uses PyJWT if installed; falls back to a header+payload decode
    without signature check ONLY when an env override
    `VEILGUARD_A2A_JWT_INSECURE_UNSIGNED=1` is set (dev/test only —
    NEVER do this in prod).
    """
    if not _jwt_enabled():
        return None
    if not token:
        return None
    try:
        import jwt as _pyjwt   # PyJWT
    except ImportError:
        _pyjwt = None

    if _pyjwt is None:
        if os.environ.get("VEILGUARD_A2A_JWT_INSECURE_UNSIGNED") == "1":
            # Dev fallback — decode header.payload without sig check.
            try:
                parts = token.split(".")
                if len(parts) < 2:
                    return None
                import base64 as _b64
                pad = "=" * (-len(parts[1]) % 4)
                payload_raw = _b64.urlsafe_b64decode(parts[1] + pad)
                payload = json.loads(payload_raw)
                logger.warning(
                    "[a2a_auth] INSECURE — accepting JWT without signature "
                    "verification (VEILGUARD_A2A_JWT_INSECURE_UNSIGNED=1)"
                )
                return payload
            except Exception as e:
                logger.warning(f"[a2a_auth] insecure JWT decode failed: {e}")
                return None
        logger.warning(
            "[a2a_auth] JWT auth requested but PyJWT not installed; "
            "set VEILGUARD_A2A_JWT_INSECURE_UNSIGNED=1 for dev or "
            "`pip install pyjwt[crypto]` for prod."
        )
        return None

    jwks = _fetch_jwks()
    if not jwks:
        return None
    aud = _expected_aud() or None
    iss = _expected_iss() or None
    options = {"verify_signature": True, "verify_aud": bool(aud), "verify_iss": bool(iss)}
    try:
        # PyJWT >= 2 handles JWKS lookup via PyJWKClient
        try:
            from jwt import PyJWKClient
            url = os.environ.get("VEILGUARD_A2A_JWT_JWKS_URL", "").strip()
            jwks_client = PyJWKClient(url, cache_keys=True)
            signing_key = jwks_client.get_signing_key_from_jwt(token).key
        except Exception:
            # Older PyJWT — pass keys dict directly (less optimal)
            signing_key = jwks
        payload = _pyjwt.decode(
            token, signing_key,
            algorithms=["RS256", "ES256", "PS256"],
            audience=aud, issuer=iss,
            options=options,
        )
        return payload
    except Exception as e:
        logger.info(f"[a2a_auth] JWT verify failed: {type(e).__name__}: {e}")
        return None


# ── mTLS subject extraction ────────────────────────────────────────────


def parse_mtls_subject(subject_header: str) -> Optional[dict[str, str]]:
    """Parse a `CN=foo,O=bar` style subject header into a dict.

    The TLS terminator (Caddy/nginx) sets a header like
    ``X-Client-Cert-Subject: CN=acme-corp,O=Acme,C=ZA``.
    """
    if not subject_header:
        return None
    out = {}
    for part in subject_header.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip().upper()] = v.strip()
    return out or None


def mtls_to_tenant(subject_header: str) -> Optional[dict[str, Any]]:
    """Map an mTLS cert subject to a tenant row.

    Reads `a2a_external_keys` for a row whose `label` matches the
    CN (operator convention: when issuing mTLS-bound access, set
    label=CN).  Returns the same row shape as the API-key path.
    """
    if not _mtls_enabled():
        return None
    parsed = parse_mtls_subject(subject_header)
    if not parsed or not parsed.get("CN"):
        return None
    cn = parsed["CN"]
    try:
        from .ledger.store import LedgerStore
        tbl = LedgerStore.get().table("a2a_external_keys")
        arr = (
            tbl.search()
            .where(f"label = '{cn}' AND status = 'active'")
            .limit(1)
            .to_arrow()
        )
        if arr.num_rows == 0:
            return None
        return {c: arr.column(c)[0].as_py() for c in arr.column_names}
    except Exception as e:
        logger.warning(f"[a2a_auth] mTLS tenant lookup failed: {e}")
        return None


# ── Combined authenticate() ────────────────────────────────────────────


def authenticate(
    *,
    bearer_token: Optional[str],
    mtls_subject: Optional[str],
    api_key: Optional[str],
    target_agent_id: str,
) -> Optional[dict[str, Any]]:
    """Try each auth method in spec precedence: JWT → mTLS → API-key.

    Returns the authenticated tenant-context row, or None if none of
    the methods produced a valid identity.  Caller still has to
    enforce allow-list + rate-limit on the row that's returned.

    The bound `tenant_id` always comes from a key registry row (so
    rate-limits + allow-lists work uniformly).  JWT carries a claim
    that selects WHICH key registry row to use.
    """
    # 1. JWT
    if bearer_token:
        payload = verify_jwt(bearer_token)
        if payload is not None:
            claim_name = _tenant_claim_name()
            tenant_id = str(payload.get(claim_name) or "").strip()
            if tenant_id:
                # Find the registry row whose tenant_id matches; the
                # row carries the rate-limit + allow-list config.
                try:
                    from .ledger.store import LedgerStore
                    tbl = LedgerStore.get().table("a2a_external_keys")
                    arr = (
                        tbl.search()
                        .where(
                            f"tenant_id = '{tenant_id}' AND status = 'active'"
                        )
                        .limit(1)
                        .to_arrow()
                    )
                    if arr.num_rows > 0:
                        row = {c: arr.column(c)[0].as_py() for c in arr.column_names}
                        row["_auth_method"] = "jwt"
                        row["_jwt_payload"] = payload
                        return row
                except Exception as e:
                    logger.warning(f"[a2a_auth] JWT→tenant lookup failed: {e}")
    # 2. mTLS
    if mtls_subject:
        row = mtls_to_tenant(mtls_subject)
        if row is not None:
            row["_auth_method"] = "mtls"
            return row
    # 3. API-key (fall through to existing path)
    if api_key:
        from .a2a_external import _resolve_key as _resolve
        row = _resolve(api_key)
        if row is not None:
            row["_auth_method"] = "api_key"
            return row
    return None


__all__ = [
    "verify_jwt",
    "parse_mtls_subject",
    "mtls_to_tenant",
    "authenticate",
]
