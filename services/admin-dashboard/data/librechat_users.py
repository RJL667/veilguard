"""User + usage queries against the LibreChat MongoDB.

Read-only. We surface counts, last-active times, and per-user
redaction overlay (joined client-side with pii_audit_stats.per_user).

Token / message counts come from the standard LibreChat collections:
``users``, ``messages``, ``conversations``. We aggregate cheaply with
``$group`` rather than streaming whole collections.
"""
from __future__ import annotations

from typing import Any

from auth import get_mongo_db


async def users_by_id() -> dict[str, dict[str, Any]]:
    """Return ``{user_id: {name, email, role}}``.

    Used to enrich pii_audit rows (which carry only ``user_id``) with
    something a human can read. Cheap — the users collection is small.
    Caching is left to the FastAPI layer; right now the dashboard hits
    Mongo on every refresh which is fine for ~dozens of users.
    """
    db = get_mongo_db()
    out: dict[str, dict[str, Any]] = {}
    async for u in db.users.find({}, {"_id": 1, "name": 1, "email": 1, "role": 1}):
        out[str(u["_id"])] = {
            "name": u.get("name"),
            "email": u.get("email"),
            "role": u.get("role"),
        }
    return out


async def all_users() -> list[dict[str, Any]]:
    db = get_mongo_db()
    rows = []
    async for u in db.users.find(
        {},
        {
            "_id": 1,
            "name": 1,
            "username": 1,
            "email": 1,
            "role": 1,
            "createdAt": 1,
            "lastLogin": 1,
        },
    ):
        u["_id"] = str(u["_id"])
        rows.append(u)
    rows.sort(key=lambda r: r.get("name") or r.get("email") or "")
    return rows


async def usage_summary() -> dict[str, Any]:
    """Cheap aggregate counts: total conversations, messages, tokens."""
    db = get_mongo_db()
    out = {
        "users_total": await db.users.count_documents({}),
        "users_admin": await db.users.count_documents({"role": "ADMIN"}),
        "conversations_total": 0,
        "messages_total": 0,
    }
    try:
        out["conversations_total"] = await db.conversations.count_documents({})
    except Exception:
        pass
    try:
        out["messages_total"] = await db.messages.count_documents({})
    except Exception:
        pass
    return out


async def per_user_messages(top_n: int = 20) -> list[dict[str, Any]]:
    """Aggregate message counts + last activity per user.

    LibreChat messages have ``user`` (ObjectId), ``createdAt``, and
    sometimes ``tokenCount``. The pipeline groups by user and joins
    against ``users`` for the display name.
    """
    db = get_mongo_db()
    pipeline = [
        {
            "$group": {
                "_id": "$user",
                "messages": {"$sum": 1},
                "last_message": {"$max": "$createdAt"},
                "tokens": {"$sum": {"$ifNull": ["$tokenCount", 0]}},
            }
        },
        {"$sort": {"messages": -1}},
        {"$limit": top_n},
        {
            "$lookup": {
                "from": "users",
                "localField": "_id",
                "foreignField": "_id",
                "as": "user",
            }
        },
        {"$unwind": {"path": "$user", "preserveNullAndEmptyArrays": True}},
        {
            "$project": {
                "_id": 0,
                "user_id": {"$toString": "$_id"},
                "name": "$user.name",
                "email": "$user.email",
                "role": "$user.role",
                "messages": 1,
                "tokens": 1,
                "last_message": 1,
            }
        },
    ]
    try:
        return [doc async for doc in db.messages.aggregate(pipeline)]
    except Exception:
        return []
