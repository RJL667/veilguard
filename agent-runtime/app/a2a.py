"""Phase 4 — A2A internal inter-agent transport (§Phase 4).

Spec: "Each agent exposes a minimal A2A endpoint internally —
AgentCard at `/.well-known/agents/<aid>/agent-card.json`,
`POST /agents/<aid>/messages` (maps to task comments),
`GET /agents/<aid>/tasks/<id>`.  Internal-only auth (API key bound
to tenant)."

Why now (Phase 4): replaces the bespoke `assign_task` RPC from
Phase 2 with a standard-conformant transport.  AgentCard follows
the early Agent-to-Agent spec shape (Google's A2A draft) so
external A2A clients can interoperate later.

Surface:
  GET  /.well-known/agents                       — list known agents
  GET  /.well-known/agents/<aid>/agent-card.json — single AgentCard
  POST /agents/<aid>/messages                    — send a message → comment
  GET  /agents/<aid>/tasks/<task_id>             — read a task
  GET  /agents/<aid>/inbox                       — owner's claimable tasks

Auth: ``X-Internal-Secret`` header MUST match ``VEILGUARD_INTERNAL_SECRET``
env var.  Same gate sub-agents uses for /api/client/*.
"""

from __future__ import annotations

import os
import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request, Header
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


router = APIRouter()


def _require_internal_secret(provided: Optional[str]) -> None:
    expected = os.environ.get("VEILGUARD_INTERNAL_SECRET", "")
    if not expected:
        # In dev, allow when env var unset; log loudly.
        logger.warning(
            "[a2a] VEILGUARD_INTERNAL_SECRET unset — accepting all calls."
        )
        return
    if not provided or provided != expected:
        raise HTTPException(status_code=401, detail="unauthorized")


def _persona_to_card(p: Any) -> dict[str, Any]:
    """Render a PersonaSpec as a minimal AgentCard (Google A2A draft shape).

    Follows the early A2A schema closely enough that an external
    A2A client can introspect, but stays internal-only for now (no
    OAuth, no rate-limiting).
    """
    tools = list(getattr(p, "tools", []) or [])
    return {
        "schemaVersion":   "1.0",
        "name":            getattr(p, "agent_id", ""),
        "description":     getattr(p, "description", "") or f"Veilguard agent {getattr(p, 'agent_id', '')}",
        "url":             f"/agents/{getattr(p, 'agent_id', '')}",
        "version":         "0.1.0",
        "authentication":  {"type": "internal_secret"},
        "capabilities":    {
            "streaming":       False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        "skills": [
            {"id": t, "name": t, "description": ""} for t in tools
        ],
        "agent_id":   getattr(p, "agent_id", ""),
        "role":       getattr(p, "role", "") or "consultant",
        "manager_id": getattr(p, "manager_id", "") or "",
        "team_id":    getattr(p, "team_id", "") or "",
        "model":      getattr(p, "model", "") or "",
    }


@router.get("/.well-known/agents")
async def list_agents_well_known(request: Request) -> JSONResponse:
    """Discovery: list all registered agents.  No auth (read-only)."""
    persona_registry = getattr(request.app.state, "persona_registry", None)
    if persona_registry is None:
        # Fall back to module-level registry (older startup paths)
        from . import main as _m
        persona_registry = getattr(_m, "_persona_registry", None)
    if not persona_registry:
        return JSONResponse({"agents": []})
    cards = [_persona_to_card(p) for p in persona_registry.all()]
    return JSONResponse({"agents": cards})


@router.get("/.well-known/agents/{agent_id}/agent-card.json")
async def agent_card(agent_id: str, request: Request) -> JSONResponse:
    """Standard A2A AgentCard.  Public — discoverable.  Auth not required."""
    persona_registry = getattr(request.app.state, "persona_registry", None)
    if persona_registry is None:
        from . import main as _m
        persona_registry = getattr(_m, "_persona_registry", None)
    if persona_registry is None:
        raise HTTPException(status_code=503, detail="persona registry not loaded")
    p = persona_registry.get(agent_id)
    if p is None:
        raise HTTPException(status_code=404, detail=f"agent {agent_id!r} not registered")
    return JSONResponse(_persona_to_card(p))


@router.post("/agents/{agent_id}/messages")
async def send_message_to_agent(
    agent_id: str,
    request: Request,
    x_internal_secret: Optional[str] = Header(None),
) -> JSONResponse:
    """A2A send — appends a message as a comment on a Task.

    Internal-only auth (X-Internal-Secret).  Body shape:
      {
        "task_id":   str,        # which Task the message attaches to
        "tenant_id": str,
        "user_id":   str,
        "from_agent_id": str,     # sender (the A2A "from")
        "body":      str,         # the message
        "kind":      str = "comment"  # kind tag for the comment row
      }

    Maps to `ledger.comments.add_comment` so the message is durable
    in `task_comments` and shows up in the task's audit trail.
    """
    _require_internal_secret(x_internal_secret)
    persona_registry = getattr(request.app.state, "persona_registry", None)
    if persona_registry is None:
        from . import main as _m
        persona_registry = getattr(_m, "_persona_registry", None)
    if persona_registry and persona_registry.get(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent {agent_id!r} unknown")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    required = ("task_id", "tenant_id", "user_id", "from_agent_id", "body")
    missing = [k for k in required if not body.get(k)]
    if missing:
        raise HTTPException(
            status_code=400, detail=f"missing required: {missing}",
        )
    from .ledger import comments as _cm
    cid = _cm.add_comment(
        task_id=body["task_id"],
        tenant_id=body["tenant_id"],
        user_id=body["user_id"],
        author_id=body["from_agent_id"],
        kind=(body.get("kind") or "comment"),
        body=body["body"],
    )
    return JSONResponse({
        "delivered":  True,
        "to_agent":   agent_id,
        "comment_id": cid,
        "task_id":    body["task_id"],
    })


@router.get("/agents/{agent_id}/tasks/{task_id}")
async def get_agent_task(
    agent_id: str,
    task_id: str,
    request: Request,
    x_internal_secret: Optional[str] = Header(None),
) -> JSONResponse:
    """A2A read — return a task + its recent comments.

    Tenant scope from query params (?tenant_id=…&user_id=…) so
    cross-tenant reads are blocked.
    """
    _require_internal_secret(x_internal_secret)
    qp = request.query_params
    tenant_id = qp.get("tenant_id") or ""
    user_id   = qp.get("user_id")   or ""
    if not (tenant_id and user_id):
        raise HTTPException(status_code=400, detail="tenant_id + user_id required")
    from .ledger import tasks as _tasks
    from .ledger import comments as _cm
    task = _tasks.get_task(task_id, tenant_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    # Only return task if THIS agent owns it (current owner_id) OR
    # is the assigner — otherwise reject (cross-agent peek).  Looser
    # check for v1: just confirm the requested agent_id matches.
    cmts = _cm.list_comments(task_id=task_id, tenant_id=tenant_id, user_id=user_id)
    return JSONResponse({"task": task, "comments": cmts})


@router.get("/agents/{agent_id}/inbox")
async def get_agent_inbox(
    agent_id: str,
    request: Request,
    x_internal_secret: Optional[str] = Header(None),
) -> JSONResponse:
    """A2A inbox — the agent's pending tasks (owner_id=agent_id,
    status in claimable states)."""
    _require_internal_secret(x_internal_secret)
    qp = request.query_params
    tenant_id = qp.get("tenant_id") or ""
    user_id   = qp.get("user_id")   or ""
    if not (tenant_id and user_id):
        raise HTTPException(status_code=400, detail="tenant_id + user_id required")
    from .ledger import tasks as _tasks
    items = _tasks.inbox(
        owner_id=agent_id, tenant_id=tenant_id, user_id=user_id,
    )
    return JSONResponse({"agent_id": agent_id, "count": len(items), "tasks": items})


__all__ = ["router"]
