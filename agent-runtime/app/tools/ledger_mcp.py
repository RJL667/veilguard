"""In-process MCP server exposing the decision-ledger CRUD as tools.

The Claude Agent SDK calls these tools when an agent's LLM decides to.
Each tool reads TenantContext from contextvars (the SDK doesn't propagate
per-request data through tool calls), so the runtime MUST establish
context before invoking query().

Tool naming convention: `ledger_*` so they don't collide with the
client-daemon's `mcp__client_daemon__*` namespace.  The SDK exposes them
to the LLM as `mcp__veilguard_ledger__ledger_*`.

Why in-process rather than a separate MCP server process:
  - Lower latency (no HTTP round-trip for every tool call)
  - Shares the LedgerStore singleton (one Lance connection)
  - Same audit + tenant-context propagation
  - The SDK's `create_sdk_mcp_server` is designed for exactly this case

Trust boundary: agents can only see tools their persona's allow_list
includes.  Director sees create_task/assign_task/etc.; ICs see
accept_task/add_comment/submit_for_review; consultants see less still.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from ..middleware import tenant
from ..ledger import tasks, comments, proposals

logger = logging.getLogger("agent-runtime.tools.ledger_mcp")


# ── SDK import (optional / defensive) ────────────────────────────────────


_SDK_AVAILABLE = False

try:
    from claude_agent_sdk import (  # type: ignore
        create_sdk_mcp_server,
        tool,
    )
    _SDK_AVAILABLE = True
except Exception as e:  # pragma: no cover
    logger.warning(
        f"[ledger_mcp] claude_agent_sdk import failed: {e!r}; "
        "ledger tools won't be registered with the SDK"
    )

    # Provide a no-op decorator so module-level @tool annotations don't
    # crash during import in test environments.
    def tool(name: str, description: str, input_schema: dict) -> Any:
        def _decorator(fn):
            fn._mcp_tool_meta = {
                "name": name,
                "description": description,
                "input_schema": input_schema,
            }
            return fn
        return _decorator

    def create_sdk_mcp_server(name: str, version: str, tools: list) -> None:
        return None


# ── Helpers: tenant context extraction ───────────────────────────────────


def _ctx_or_error() -> tuple[Optional[tenant.TenantContext], Optional[dict]]:
    """Return (TenantContext, None) on success, or (None, error_dict) on
    missing context.  Tools call this first; if context is missing, they
    return the error_dict instead of executing — fail-closed pattern.
    """
    ctx = tenant.current()
    if ctx is None:
        return None, {
            "content": [{
                "type": "text",
                "text": (
                    "ERROR: ledger tool called without TenantContext; "
                    "this is a programmer error in agent-runtime (runtime "
                    "must establish context via set_tenant_context before "
                    "invoking query)."
                ),
            }],
            "isError": True,
        }
    return ctx, None


def _ok(data: dict) -> dict:
    """Wrap a successful tool response in the MCP content envelope."""
    return {
        "content": [{
            "type": "text",
            "text": json.dumps(data, default=str, indent=2),
        }],
    }


def _err(msg: str) -> dict:
    return {
        "content": [{"type": "text", "text": f"ERROR: {msg}"}],
        "isError": True,
    }


# ── Director tools ──────────────────────────────────────────────────────


@tool(
    name="create_task",
    description=(
        "Create a new Task assigned to an IC or consultant. Director-only. "
        "Use deliverable_spec to give a concrete file path + format + "
        "section structure. 'Write a memo' is not a spec; 'Write 400 "
        "words at team/drafts/foo.md with sections TLDR/Risks/Recs' is."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "owner_id": {
                "type": "string",
                "description": "agent_id of the IC or consultant to assign",
            },
            "brief": {
                "type": "string",
                "description": "What needs to be done",
            },
            "deliverable_spec": {
                "type": "string",
                "description": "Concrete definition of 'done' (path, format, length)",
            },
            "parent_id": {
                "type": ["string", "null"],
                "description": "Parent task_id for subtask decomposition; null for root tasks",
            },
            "inputs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Upstream task_ids or artifact paths the assignee should read",
            },
            "due_ts": {
                "type": ["number", "null"],
                "description": "Optional Unix timestamp deadline",
            },
        },
        "required": ["owner_id", "brief", "deliverable_spec"],
    },
)
async def create_task_tool(args: dict[str, Any]) -> dict[str, Any]:
    ctx, err = _ctx_or_error()
    if err:
        return err
    try:
        task_id = tasks.create_task(
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            owner_id=args["owner_id"],
            brief=args["brief"],
            deliverable_spec=args["deliverable_spec"],
            assigner_id=ctx.agent_id,
            parent_id=args.get("parent_id"),
            inputs=args.get("inputs") or [],
            due_ts=args.get("due_ts"),
            origin="background" if ctx.is_background else "foreground",
        )
        return _ok({
            "task_id": task_id,
            "owner_id": args["owner_id"],
            "status": "open",
        })
    except Exception as e:
        logger.exception(f"[ledger_mcp] create_task failed: {e}")
        return _err(f"create_task failed: {e}")


@tool(
    name="assign_task",
    description=(
        "Reassign an existing Task to a different owner. Director-only. "
        "Use when an IC raises blocker_raised and a different specialist "
        "is better suited."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "new_owner_id": {"type": "string"},
        },
        "required": ["task_id", "new_owner_id"],
    },
)
async def assign_task_tool(args: dict[str, Any]) -> dict[str, Any]:
    ctx, err = _ctx_or_error()
    if err:
        return err
    try:
        from ..ledger.store import LedgerStore, ns_filter
        tbl = LedgerStore.get().table("agent_tasks")
        where = (
            f"{ns_filter(ctx.tenant_id, ctx.user_id)} "
            f"AND id = '{args['task_id']}'"
        )
        import time
        tbl.update(where=where, values={
            "owner_id": args["new_owner_id"],
            "updated_ts": time.time(),
        })
        comments.add_comment(
            task_id=args["task_id"],
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            author_id=ctx.agent_id,
            kind="comment",
            body=f"reassigned to {args['new_owner_id']} by {ctx.agent_id}",
        )
        return _ok({"task_id": args["task_id"], "new_owner_id": args["new_owner_id"]})
    except Exception as e:
        return _err(f"assign_task failed: {e}")


@tool(
    name="convert_proposal",
    description=(
        "After the user approves a proactive proposal in the sidebar, "
        "Director calls this to materialize it as a Task. Director-only. "
        "Required step — proactive proposals NEVER auto-become Tasks."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "proposal_id": {"type": "string"},
        },
        "required": ["proposal_id"],
    },
)
async def convert_proposal_tool(args: dict[str, Any]) -> dict[str, Any]:
    ctx, err = _ctx_or_error()
    if err:
        return err
    try:
        prop = proposals.get_proposal(
            args["proposal_id"], ctx.tenant_id, ctx.user_id
        )
        if prop is None:
            return _err(f"proposal not found: {args['proposal_id']}")
        if prop.get("status") not in ("pending", "deferred"):
            return _err(
                f"proposal {args['proposal_id']} has status "
                f"{prop.get('status')!r}; only pending/deferred can be converted"
            )
        task_id = tasks.create_task(
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            owner_id=prop["proposed_assignee"],
            brief=prop["proposed_brief"],
            deliverable_spec=prop.get("proposed_deliverable_spec") or "",
            assigner_id=ctx.agent_id,
            origin="background",  # proactive stream is always background
            constitution_version=prop.get("constitution_version"),
            extras_json=json.dumps({
                "from_proposal_id": args["proposal_id"],
                "signal_type": prop.get("signal_type"),
            }),
        )
        proposals.approve_proposal(
            proposal_id=args["proposal_id"],
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            resulting_task_id=task_id,
        )
        return _ok({"task_id": task_id, "from_proposal_id": args["proposal_id"]})
    except Exception as e:
        return _err(f"convert_proposal failed: {e}")


@tool(
    name="shelve_proposal",
    description="Shelve a proactive proposal with a reason. Director-only.",
    input_schema={
        "type": "object",
        "properties": {
            "proposal_id": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["proposal_id", "reason"],
    },
)
async def shelve_proposal_tool(args: dict[str, Any]) -> dict[str, Any]:
    ctx, err = _ctx_or_error()
    if err:
        return err
    try:
        proposals.shelve_proposal(
            proposal_id=args["proposal_id"],
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            reason=args["reason"],
        )
        return _ok({"proposal_id": args["proposal_id"], "status": "shelved"})
    except Exception as e:
        return _err(f"shelve_proposal failed: {e}")


# ── IC tools (accept, status, comment, output, submit) ──────────────────


@tool(
    name="accept_task",
    description=(
        "IC accepts an assigned Task; transitions status open → in_progress. "
        "Call this AFTER reading brief + deliverable_spec, BEFORE doing work."
    ),
    input_schema={
        "type": "object",
        "properties": {"task_id": {"type": "string"}},
        "required": ["task_id"],
    },
)
async def accept_task_tool(args: dict[str, Any]) -> dict[str, Any]:
    ctx, err = _ctx_or_error()
    if err:
        return err
    try:
        # Two-step: open → accepted → in_progress (matches state machine).
        tasks.update_status(
            task_id=args["task_id"],
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            new_status="accepted",
            actor_agent_id=ctx.agent_id,
        )
        tasks.update_status(
            task_id=args["task_id"],
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            new_status="in_progress",
            actor_agent_id=ctx.agent_id,
        )
        return _ok({"task_id": args["task_id"], "status": "in_progress"})
    except Exception as e:
        return _err(f"accept_task failed: {e}")


@tool(
    name="add_comment",
    description=(
        "Append a comment to a Task's audit chain. ICs use this for "
        "status updates, blocker_raised, peer-DM-style notes. "
        "kind ∈ {comment, blocker_raised, blocker_cleared}."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "kind": {
                "type": "string",
                "enum": ["comment", "blocker_raised", "blocker_cleared"],
            },
            "body": {"type": "string"},
        },
        "required": ["task_id", "kind", "body"],
    },
)
async def add_comment_tool(args: dict[str, Any]) -> dict[str, Any]:
    ctx, err = _ctx_or_error()
    if err:
        return err
    try:
        cid = comments.add_comment(
            task_id=args["task_id"],
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            author_id=ctx.agent_id,
            kind=args["kind"],
            body=args["body"],
        )
        return _ok({"comment_id": cid, "task_id": args["task_id"]})
    except Exception as e:
        return _err(f"add_comment failed: {e}")


@tool(
    name="attach_output",
    description=(
        "Append a workspace file path to the Task's outputs[]. Atomic; "
        "idempotent. Call AFTER the file is fully written."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "path": {"type": "string"},
        },
        "required": ["task_id", "path"],
    },
)
async def attach_output_tool(args: dict[str, Any]) -> dict[str, Any]:
    ctx, err = _ctx_or_error()
    if err:
        return err
    try:
        tasks.attach_output(
            task_id=args["task_id"],
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            path=args["path"],
        )
        return _ok({"task_id": args["task_id"], "path": args["path"]})
    except Exception as e:
        return _err(f"attach_output failed: {e}")


@tool(
    name="submit_for_review",
    description=(
        "IC submits Task to Critic. target ∈ {team_knowledge, user_deliverable, "
        "org_blackboard}. Pick the LOWEST-IMPACT target that satisfies the "
        "brief — don't over-promote one-off work."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "target": {
                "type": "string",
                "enum": ["team_knowledge", "user_deliverable", "org_blackboard"],
            },
        },
        "required": ["task_id", "target"],
    },
)
async def submit_for_review_tool(args: dict[str, Any]) -> dict[str, Any]:
    ctx, err = _ctx_or_error()
    if err:
        return err
    try:
        tasks.submit_for_review(
            task_id=args["task_id"],
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            target=args["target"],
            submitter_agent_id=ctx.agent_id,
        )
        return _ok({"task_id": args["task_id"], "status": "review", "target": args["target"]})
    except Exception as e:
        return _err(f"submit_for_review failed: {e}")


@tool(
    name="get_task",
    description=(
        "Read a single Task by id. Returns brief, deliverable_spec, "
        "status, inputs, outputs, recent comments. Use to load context "
        "after accept_task."
    ),
    input_schema={
        "type": "object",
        "properties": {"task_id": {"type": "string"}},
        "required": ["task_id"],
    },
)
async def get_task_tool(args: dict[str, Any]) -> dict[str, Any]:
    ctx, err = _ctx_or_error()
    if err:
        return err
    try:
        task = tasks.get_task(args["task_id"], ctx.tenant_id, ctx.user_id)
        if task is None:
            return _err(f"task not found: {args['task_id']}")
        cmts = comments.list_comments(
            task_id=args["task_id"],
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
        )
        return _ok({"task": task, "comments": cmts})
    except Exception as e:
        return _err(f"get_task failed: {e}")


@tool(
    name="review_decision",
    description=(
        "Critic-only: register a review decision on a Task. "
        "decision ∈ {approved, changes_requested, declined}. "
        "Atomically adds a review_decision comment to the chain AND "
        "transitions task status: approved → done, changes_requested → "
        "in_progress (so the IC can iterate), declined → cancelled."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "decision": {
                "type": "string",
                "enum": ["approved", "changes_requested", "declined"],
            },
            "notes": {
                "type": "string",
                "description": "Specific reason for the decision (~1 sentence)",
            },
        },
        "required": ["task_id", "decision", "notes"],
    },
)
async def review_decision_tool(args: dict[str, Any]) -> dict[str, Any]:
    ctx, err = _ctx_or_error()
    if err:
        return err
    try:
        decision = args["decision"]
        # Add review_decision comment with the full body.
        body = f"{decision}: {args['notes']}"
        comments.add_comment(
            task_id=args["task_id"],
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            author_id=ctx.agent_id,
            kind="review_decision",
            body=body,
        )
        # Transition task status to match decision.
        target_status = {
            "approved": "done",
            "changes_requested": "in_progress",
            "declined": "cancelled",
        }[decision]
        tasks.update_status(
            task_id=args["task_id"],
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            new_status=target_status,
            actor_agent_id=ctx.agent_id,
        )
        return _ok({
            "task_id": args["task_id"],
            "decision": decision,
            "new_status": target_status,
        })
    except Exception as e:
        return _err(f"review_decision failed: {e}")


@tool(
    name="inbox",
    description=(
        "Return the calling agent's open + in_progress + blocked + review "
        "Tasks. The agent's personal queue."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "default": 20},
        },
    },
)
async def inbox_tool(args: dict[str, Any]) -> dict[str, Any]:
    ctx, err = _ctx_or_error()
    if err:
        return err
    try:
        rows = tasks.inbox(
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            owner_id=ctx.agent_id,
            limit=args.get("limit", 20),
        )
        return _ok({"tasks": rows, "count": len(rows)})
    except Exception as e:
        return _err(f"inbox failed: {e}")


# ── Server factory ──────────────────────────────────────────────────────


_ALL_TOOLS = [
    # Director
    create_task_tool,
    assign_task_tool,
    convert_proposal_tool,
    shelve_proposal_tool,
    # IC
    accept_task_tool,
    add_comment_tool,
    attach_output_tool,
    submit_for_review_tool,
    get_task_tool,
    inbox_tool,
    # Critic
    review_decision_tool,
]


def build_ledger_mcp_server():
    """Construct the in-process MCP server with all ledger tools.

    Returns None if SDK isn't available (tests / dev without
    claude_agent_sdk).  Production must have the SDK.
    """
    if not _SDK_AVAILABLE:
        return None
    return create_sdk_mcp_server(
        name="veilguard_ledger",
        version="0.1.0",
        tools=_ALL_TOOLS,
    )


__all__ = [
    "build_ledger_mcp_server",
    "_ALL_TOOLS",  # exported for unit tests
]
