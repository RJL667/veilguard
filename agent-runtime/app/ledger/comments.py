"""task_comments append-only chain.

Per spec §3.8.5: comments are stored in a separate table with
`prev_hash` chaining so mutation of any prior comment breaks the chain
(detectable via walk).  The Task row's `comments_head_hash` tracks the
latest comment; auditing detects tampering by comparing the recomputed
chain hash against the stored head.

Comments are NEVER updated in place.  Even "corrections" are new
comments referencing the prior one.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, Optional

from .store import LedgerStore, ns_filter


_ALLOWED_KINDS = {
    "comment",
    "status_change",
    "review_request",
    "review_decision",
    "blocker_raised",
    "blocker_cleared",
}


def _now() -> float:
    return time.time()


def _new_id() -> str:
    return f"cmt-{uuid.uuid4().hex[:12]}"


def _hash_comment(
    *,
    id: str,
    task_id: str,
    author_id: str,
    kind: str,
    body: str,
    ts: float,
    prev_hash: Optional[str],
) -> str:
    """Canonical SHA-256 of a comment's content + prev_hash.

    Order matters; we use sorted JSON for determinism.
    """
    canon = json.dumps({
        "id": id,
        "task_id": task_id,
        "author_id": author_id,
        "kind": kind,
        "body": body,
        "ts": round(ts, 6),
        "prev_hash": prev_hash or "",
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def add_comment(
    *,
    task_id: str,
    tenant_id: str,
    user_id: str,
    author_id: str,
    kind: str,
    body: str,
    extras_json: Optional[str] = None,
) -> str:
    """Append a comment to the chain.  Returns the new comment_id.

    Atomicity: the chain head update on the Task row + the new comment
    insert should ideally be one transaction.  LanceDB doesn't support
    cross-table transactions; we do it sequentially with append first,
    then head update.  If the head update fails, the comment is orphaned
    but detectable on next chain walk (the head doesn't match).
    """
    if kind not in _ALLOWED_KINDS:
        raise ValueError(f"unknown comment kind: {kind}")

    # Find current head.
    head_hash = _current_head(task_id, tenant_id, user_id)

    cid = _new_id()
    ts = _now()
    self_hash = _hash_comment(
        id=cid,
        task_id=task_id,
        author_id=author_id,
        kind=kind,
        body=body,
        ts=ts,
        prev_hash=head_hash,
    )

    tbl = LedgerStore.get().table("task_comments")
    tbl.add([{
        "id": cid,
        "task_id": task_id,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "author_id": author_id,
        "kind": kind,
        "body": body,
        "ts": ts,
        "prev_hash": head_hash,
        "self_hash": self_hash,
        "extras_json": extras_json,
    }])

    # Update Task's comments_head_hash so chain walk validates.
    tasks_tbl = LedgerStore.get().table("agent_tasks")
    tasks_tbl.update(
        where=f"{ns_filter(tenant_id, user_id)} AND id = '{task_id}'",
        values={"comments_head_hash": self_hash, "updated_ts": ts},
    )

    try:
        from ..events import broadcast
        broadcast({
            "type": "task_comment_added",
            "tenant_id": tenant_id, "user_id": user_id,
            "task_id": task_id, "comment_id": cid,
            "author_id": author_id, "kind": kind,
            "body": body[:240],
        })
    except Exception:
        pass
    # Phase 6.7 — comments are state mutations.  Discussion/note kinds
    # are pure narration (don't count); state-machine kinds count.
    try:
        if kind in ("status_change", "review_decision",
                    "blocker_raised", "blocker_cleared",
                    "review_request", "criterion_check"):
            from ..runtime_health import apr_record_artifact
            apr_record_artifact()
    except Exception:
        pass

    return cid


def list_comments(
    *,
    task_id: str,
    tenant_id: str,
    user_id: str,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Return all comments for a task, oldest first."""
    tbl = LedgerStore.get().table("task_comments")
    where = f"{ns_filter(tenant_id, user_id)} AND task_id = '{task_id}'"
    arr = tbl.search().where(where).limit(limit).to_arrow()
    rows = [
        {col: arr.column(col)[i].as_py() for col in arr.column_names}
        for i in range(arr.num_rows)
    ]
    rows.sort(key=lambda r: r.get("ts", 0))
    return rows


def _current_head(task_id: str, tenant_id: str, user_id: str) -> Optional[str]:
    """Get the Task's comments_head_hash, or None."""
    tbl = LedgerStore.get().table("agent_tasks")
    where = f"{ns_filter(tenant_id, user_id)} AND id = '{task_id}'"
    arr = tbl.search().where(where).limit(1).to_arrow()
    if arr.num_rows == 0:
        return None
    return arr.column("comments_head_hash")[0].as_py()


def verify_chain(
    *,
    task_id: str,
    tenant_id: str,
    user_id: str,
) -> tuple[bool, list[str]]:
    """Walk the chain from oldest to newest, recomputing hashes.

    Returns (chain_valid, broken_comment_ids).  Used by audit jobs to
    detect tampering.
    """
    rows = list_comments(task_id=task_id, tenant_id=tenant_id, user_id=user_id)
    broken: list[str] = []
    prev_hash: Optional[str] = None

    for row in rows:
        expected = _hash_comment(
            id=row["id"],
            task_id=row["task_id"],
            author_id=row["author_id"],
            kind=row["kind"],
            body=row["body"],
            ts=row["ts"],
            prev_hash=prev_hash,
        )
        if row.get("self_hash") != expected:
            broken.append(row["id"])
            # Continue walking to find all breaks
        if row.get("prev_hash") != prev_hash:
            broken.append(row["id"])
        prev_hash = row.get("self_hash")

    # Check head pointer matches the last comment's self_hash
    head = _current_head(task_id, tenant_id, user_id)
    last_hash = rows[-1]["self_hash"] if rows else None
    if head != last_hash:
        broken.append("HEAD_MISMATCH")

    return (len(broken) == 0, broken)


__all__ = [
    "add_comment",
    "list_comments",
    "verify_chain",
]
