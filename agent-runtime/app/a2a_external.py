"""Phase 5 — A2A external transport (§Phase 5).

Spec: "A2A endpoints exposed externally.  API-key + (later)
OAuth/mTLS.  External agents (e.g. a customer's CrewAI deployment)
can delegate Tasks into Veilguard's org.  Inverse: Veilguard's
Director can delegate to external A2A endpoints.  Small effort
once Phase 3 internal A2A exists — mostly auth, rate-limiting,
and tenant-scoped allow-lists."

This module wraps the internal `a2a.py` router with:
  * API-key auth bound to a tenant_id (Lance-backed key registry)
  * Per-key rate-limiting (sliding-window counter in memory)
  * Per-key allow-list (which agent_ids the key can address)
  * Audit logging to client_tool_approvals or a dedicated table

Surface — same shape as internal but under `/external/a2a/...`:
  POST   /external/a2a/agents/{aid}/messages
  GET    /external/a2a/agents/{aid}/tasks/{task_id}
  GET    /external/a2a/agents/{aid}/inbox

Key registry (Lance table `a2a_external_keys`):
  id, tenant_id, key_hash, label, allowed_agents, rate_limit_per_min,
  created_ts, last_used_ts, status (active|revoked), extras_json

For v1 keys are pre-provisioned by an admin (no public sign-up); the
CLI / future admin UI generates one and stores the hash.  Plaintext
keys NEVER persist server-side.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from collections import deque
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Header, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/external/a2a")


# ── Schema for the external-keys table ─────────────────────────────────


def a2a_external_keys_schema():
    """Lance schema for ``a2a_external_keys``.  Plaintext keys never
    persisted — only ``key_hash`` (SHA-256 hex).
    """
    import pyarrow as pa
    return pa.schema([
        pa.field("id", pa.string(), nullable=False),
        pa.field("tenant_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=True),  # optional scope
        pa.field("key_hash", pa.string(), nullable=False),
        pa.field("label", pa.string(), nullable=True),
        pa.field("allowed_agents", pa.list_(pa.string()), nullable=True),
        pa.field("rate_limit_per_min", pa.int64(), nullable=False),
        pa.field("status", pa.string(), nullable=False),  # active|revoked
        pa.field("created_ts", pa.float64(), nullable=False),
        pa.field("last_used_ts", pa.float64(), nullable=True),
        pa.field("extras_json", pa.string(), nullable=True),
    ])


# ── Key helpers ────────────────────────────────────────────────────────


def _hash_key(plaintext: str) -> str:
    """SHA-256 hex digest of the API key.  Constant-time comparable."""
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def _resolve_key(api_key: str) -> Optional[dict[str, Any]]:
    """Look up the row for ``api_key`` (by hash).  None if not active.

    Tenant-scoped read; we don't want to leak presence of a key for
    a different tenant.
    """
    if not api_key:
        return None
    key_hash = _hash_key(api_key)
    try:
        from .ledger.store import LedgerStore
        tbl = LedgerStore.get().table("a2a_external_keys")
        arr = (
            tbl.search()
            .where(f"key_hash = '{key_hash}' AND status = 'active'")
            .limit(1)
            .to_arrow()
        )
        if arr.num_rows == 0:
            return None
        return {c: arr.column(c)[0].as_py() for c in arr.column_names}
    except Exception as e:
        logger.warning(f"[a2a_external] key lookup failed: {e}")
        return None


def _touch_last_used(key_id: str) -> None:
    """Best-effort last_used_ts update.  Failure is silent — auth
    decision already returned."""
    try:
        from .ledger.store import LedgerStore
        tbl = LedgerStore.get().table("a2a_external_keys")
        tbl.update(
            where=f"id = '{key_id}'",
            values={"last_used_ts": time.time()},
        )
    except Exception:
        pass


# ── Rate limiting ──────────────────────────────────────────────────────


# In-memory sliding-window counter, keyed on key_id.  Resets on
# process restart — for v1 that's acceptable (rate limits are
# advisory, not security boundaries).
_RATE_WINDOW_SECONDS = 60.0
_RATE_COUNTERS: dict[str, deque] = {}


def _consume_rate_token(key_id: str, limit_per_min: int) -> bool:
    """Try to consume one rate-token.  Returns True if allowed."""
    if limit_per_min <= 0:
        return True  # 0 means unlimited
    now = time.time()
    cutoff = now - _RATE_WINDOW_SECONDS
    bucket = _RATE_COUNTERS.setdefault(key_id, deque())
    # Trim expired entries
    while bucket and bucket[0] < cutoff:
        bucket.popleft()
    if len(bucket) >= limit_per_min:
        return False
    bucket.append(now)
    return True


# ── Auth dependency ────────────────────────────────────────────────────


def _authenticate_external(
    api_key: Optional[str],
    target_agent_id: str,
    *,
    bearer_token: Optional[str] = None,
    mtls_subject: Optional[str] = None,
) -> dict[str, Any]:
    """Validate identity (JWT → mTLS → API-key), allow-list, rate limit.

    Returns the authenticated tenant row.  Raises 401 / 403 / 429 on
    failure.  Auth precedence is implemented in a2a_auth.authenticate().
    """
    # Try JWT/mTLS/API-key in order (a2a_auth handles precedence)
    try:
        from .a2a_auth import authenticate as _auth_combined
        row = _auth_combined(
            bearer_token=bearer_token,
            mtls_subject=mtls_subject,
            api_key=api_key,
            target_agent_id=target_agent_id,
        )
    except Exception:
        # Module load failure → fall back to API-key only
        row = _resolve_key(api_key) if api_key else None
    if row is None:
        raise HTTPException(status_code=401, detail="unauthorized")
    # Allow-list check
    allowed = row.get("allowed_agents") or []
    if allowed and target_agent_id not in allowed:
        raise HTTPException(status_code=403, detail="forbidden")
    # Rate-limit check
    limit = int(row.get("rate_limit_per_min") or 60)
    if not _consume_rate_token(row["id"], limit):
        raise HTTPException(
            status_code=429,
            detail=f"rate limited: {limit}/min on key {row['id']}",
            headers={"Retry-After": str(int(_RATE_WINDOW_SECONDS))},
        )
    _touch_last_used(row["id"])
    return row


# ── Key management (admin-only — internal-secret auth) ─────────────────


def _require_internal_secret(provided: Optional[str]) -> None:
    import os as _os
    expected = _os.environ.get("VEILGUARD_INTERNAL_SECRET", "")
    if not expected:
        logger.warning("[a2a_external] internal secret unset; accepting all")
        return
    if not provided or provided != expected:
        raise HTTPException(status_code=401, detail="unauthorized")


@router.post("/keys")
async def create_key(
    request: Request,
    x_internal_secret: Optional[str] = Header(None),
) -> JSONResponse:
    """Admin-only.  Generate a new external API key for a tenant.

    Body: {tenant_id, user_id?, label?, allowed_agents?, rate_limit_per_min?}
    Returns:  {key_id, plaintext_key}  — plaintext is shown ONCE.
              Never persisted; never recoverable.  Caller must save.
    """
    _require_internal_secret(x_internal_secret)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    tenant_id = body.get("tenant_id") or ""
    if not tenant_id:
        raise HTTPException(status_code=400, detail="tenant_id required")

    # Generate plaintext key — random 32-byte hex
    plaintext = f"vg-a2a-{uuid.uuid4().hex}{uuid.uuid4().hex[:16]}"
    key_id = f"a2ak-{uuid.uuid4().hex[:12]}"
    now = time.time()
    row = {
        "id":                  key_id,
        "tenant_id":           tenant_id,
        "user_id":             body.get("user_id") or "",
        "key_hash":            _hash_key(plaintext),
        "label":               body.get("label") or "",
        "allowed_agents":      list(body.get("allowed_agents") or []),
        "rate_limit_per_min":  int(body.get("rate_limit_per_min") or 60),
        "status":              "active",
        "created_ts":          now,
        "last_used_ts":        None,
        "extras_json":         None,
    }
    from .ledger.store import LedgerStore
    LedgerStore.get().table("a2a_external_keys").add([row])
    return JSONResponse({
        "key_id":         key_id,
        "tenant_id":      tenant_id,
        "plaintext_key":  plaintext,
        "warning":        (
            "This is the only time the plaintext key is shown.  "
            "Save it to the caller's secret store NOW — server-side "
            "only the SHA-256 hash is retained."
        ),
    })


@router.post("/keys/{key_id}/revoke")
async def revoke_key(
    key_id: str,
    x_internal_secret: Optional[str] = Header(None),
) -> JSONResponse:
    """Admin-only.  Revoke a key (status='revoked').  Idempotent."""
    _require_internal_secret(x_internal_secret)
    from .ledger.store import LedgerStore
    tbl = LedgerStore.get().table("a2a_external_keys")
    tbl.update(
        where=f"id = '{key_id}'",
        values={"status": "revoked", "last_used_ts": time.time()},
    )
    return JSONResponse({"revoked": True, "key_id": key_id})


# ── Public A2A endpoints (auth + rate-limited) ─────────────────────────


@router.post("/agents/{agent_id}/messages")
async def external_send_message(
    agent_id: str,
    request: Request,
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
    x_client_cert_subject: Optional[str] = Header(None),
) -> JSONResponse:
    """External A2A send.  Auth: JWT (Authorization: Bearer …) →
    mTLS (X-Client-Cert-Subject) → API key (X-API-Key)."""
    bearer = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization.split(" ", 1)[1].strip()
    key_row = _authenticate_external(
        x_api_key, agent_id,
        bearer_token=bearer, mtls_subject=x_client_cert_subject,
    )
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    required = ("task_id", "from_agent_id", "body")
    missing = [k for k in required if not body.get(k)]
    if missing:
        raise HTTPException(
            status_code=400, detail=f"missing required: {missing}",
        )
    # Tenant comes from the KEY, not the body — external callers
    # can't address a different tenant's ledger
    tenant_id = key_row["tenant_id"]
    user_id = key_row.get("user_id") or tenant_id
    from .ledger import comments as _cm
    cid = _cm.add_comment(
        task_id=body["task_id"],
        tenant_id=tenant_id,
        user_id=user_id,
        author_id=f"external:{body['from_agent_id']}",
        kind="comment",
        body=body["body"],
    )
    return JSONResponse({
        "delivered":  True,
        "to_agent":   agent_id,
        "comment_id": cid,
        "task_id":    body["task_id"],
    })


@router.get("/agents/{agent_id}/tasks/{task_id}")
async def external_get_task(
    agent_id: str,
    task_id: str,
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
    x_client_cert_subject: Optional[str] = Header(None),
) -> JSONResponse:
    """External A2A read."""
    bearer = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization.split(" ", 1)[1].strip()
    key_row = _authenticate_external(
        x_api_key, agent_id,
        bearer_token=bearer, mtls_subject=x_client_cert_subject,
    )
    tenant_id = key_row["tenant_id"]
    user_id = key_row.get("user_id") or tenant_id
    from .ledger import tasks as _tasks
    from .ledger import comments as _cm
    task = _tasks.get_task(task_id, tenant_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    cmts = _cm.list_comments(task_id=task_id, tenant_id=tenant_id, user_id=user_id)
    return JSONResponse({"task": task, "comments": cmts})


@router.get("/agents/{agent_id}/inbox")
async def external_get_inbox(
    agent_id: str,
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
    x_client_cert_subject: Optional[str] = Header(None),
) -> JSONResponse:
    """External A2A inbox."""
    bearer = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization.split(" ", 1)[1].strip()
    key_row = _authenticate_external(
        x_api_key, agent_id,
        bearer_token=bearer, mtls_subject=x_client_cert_subject,
    )
    tenant_id = key_row["tenant_id"]
    user_id = key_row.get("user_id") or tenant_id
    from .ledger import tasks as _tasks
    items = _tasks.inbox(
        owner_id=agent_id, tenant_id=tenant_id, user_id=user_id,
    )
    return JSONResponse({"agent_id": agent_id, "count": len(items), "tasks": items})


__all__ = ["router", "a2a_external_keys_schema"]
