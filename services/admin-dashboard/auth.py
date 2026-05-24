"""LibreChat session auth + ADMIN role gate.

Every admin-dashboard request is protected by ``require_admin``.
Lifecycle:

1. Pull the LibreChat JWT from cookies (``token``) or the
   ``Authorization: Bearer ...`` header.
2. Verify with ``JWT_SECRET`` from the LibreChat .env (HS256).
3. Look up the user in the MongoDB ``users`` collection by ``_id``.
4. Allow only if ``role == "ADMIN"``. Anything else → 401/403.

Cookie-name and signing-secret defaults match LibreChat's. Override via
env if a deployment changes them. The MongoDB connection is async
(motor) so it composes with FastAPI's async handlers without thread
hops.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import jwt
from bson import ObjectId
from fastapi import Cookie, Depends, Header, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorClient


JWT_SECRET = os.environ.get("JWT_SECRET", "")
JWT_REFRESH_SECRET = os.environ.get("JWT_REFRESH_SECRET", "")
JWT_ALG = os.environ.get("JWT_ALG", "HS256")
# LibreChat sets only `refreshToken` (httpOnly, JWT_REFRESH_SECRET-signed)
# and `token_provider` cookies; the access token is returned in the
# response body and held in JS memory, never as a cookie. So for a
# browser-driven dashboard the only credential available is the
# refreshToken cookie. The Bearer-header path (JWT_SECRET) stays
# available for curl/scripts.
COOKIE_NAME = os.environ.get("ADMIN_COOKIE_NAME", "refreshToken")
MONGO_URI = os.environ.get(
    "MONGO_URI", "mongodb://veilguard-mongodb-1:27017/LibreChat"
)
MONGO_DB = os.environ.get("MONGO_DB", "LibreChat")


_mongo_client: Optional[AsyncIOMotorClient] = None


def get_mongo_db():
    """Lazy singleton — one client per process."""
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = AsyncIOMotorClient(MONGO_URI)
    return _mongo_client[MONGO_DB]


async def _decode_jwt(token: str, *, secret: str, kind: str) -> dict[str, Any]:
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{kind} secret not configured",
        )
    try:
        return jwt.decode(token, secret, algorithms=[JWT_ALG])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"{kind} expired — sign back into LibreChat",
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid {kind}: {e}",
        )


async def require_admin(
    refresh_cookie: Optional[str] = Cookie(None, alias=COOKIE_NAME),
    authorization: Optional[str] = Header(None),
) -> dict[str, Any]:
    """FastAPI dependency. Resolves to the user dict on success.

    Two acceptable credentials:
    1. Browser flow — ``refreshToken`` cookie set by LibreChat, signed
       with ``JWT_REFRESH_SECRET`` (the access-token cookie does not
       exist in LibreChat's flow).
    2. Script flow — ``Authorization: Bearer <accessToken>``, signed
       with ``JWT_SECRET``. Useful for curl, monitoring probes, etc.
    """
    bearer: Optional[str] = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()

    if refresh_cookie:
        payload = await _decode_jwt(
            refresh_cookie, secret=JWT_REFRESH_SECRET, kind="refreshToken"
        )
    elif bearer:
        payload = await _decode_jwt(
            bearer, secret=JWT_SECRET, kind="access token"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No auth token (refreshToken cookie or Bearer header)",
        )

    user_id = payload.get("id") or payload.get("_id") or payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing user id",
        )

    db = get_mongo_db()
    try:
        oid = ObjectId(str(user_id))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token user id is not a valid ObjectId",
        )
    user = await db.users.find_one(
        {"_id": oid},
        {"_id": 1, "name": 1, "username": 1, "email": 1, "role": 1},
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User from token not found",
        )

    role = (user.get("role") or "").upper()
    if role != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Requires ADMIN role (you are {role or 'unset'})",
        )

    user["_id"] = str(user["_id"])
    return user


AdminUser = Depends(require_admin)
