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
        "words at team/drafts/foo.md with sections TLDR/Risks/Recs' is. "
        "OPTIONAL: pass acceptance_criteria with concrete mechanical "
        "checks (output_path_exists, output_path_matches_regex, etc.) — "
        "if you don't, a default `output_path_exists` AC is synthesized "
        "from the deliverable_spec so the Critic can still gate `done`. "
        "COORDINATORS: a parent fan-out task owned by you (director) or a "
        "team-lead carries NO file deliverable and gets ZERO acceptance "
        "criteria — it auto-closes when all its children finish. Put any "
        "synthesis/rollup in its OWN subtask owned by a researcher/builder "
        "with depends_on=[the analysis subtask ids] so it runs only after "
        "they complete."
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
            "depends_on": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional task_ids that must reach status=done before "
                    "this task is dispatched. Use for a synthesis/rollup "
                    "subtask that must read sibling outputs: pass the "
                    "sibling analysis task_ids here so the poller holds "
                    "this task until they finish."
                ),
            },
            "acceptance_criteria": {
                "type": "array",
                "description": (
                    "Optional list of AC dicts {id, statement, check_kind, "
                    "check_args, required, rationale}. If omitted, a "
                    "default `output_path_exists` AC is synthesized."
                ),
                "items": {"type": "object"},
            },
            "team_id": {
                "type": ["string", "null"],
                "description": "Optional team_id to scope this task to a Team",
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
        # [COORDINATOR_NO_FILE_AC_2026_05_29]  Owners the inbox poller
        # NEVER dispatches — director, team-lead — are COORDINATORS, not
        # workers.  They group a fan-out; they don't personally write a
        # file.  The old code synthesized an `output_path_exists` AC from
        # their deliverable_spec, but since they're never dispatched to
        # WRITE that file the required AC could never pass and the
        # coordinator hung OPEN forever (caught live 2026-05-29: a
        # team-lead parent stuck on a phantom AC for a synthesis file
        # nothing wrote).  A coordinator's acceptance is structural — "all
        # children done" — enforced by the parent-autoclose hook in
        # tasks.update_status, NOT a file check.  So for these owners we
        # create the task with ZERO required ACs (Phase 6.0.2 exempt).
        # Real synthesis, when needed, must be its OWN subtask owned by a
        # dispatchable IC with depends_on the analyses (see the Director
        # persona Pattern-C guidance).
        #
        # [DIRECTOR_DEFAULT_AC_2026_05_28] For WORKER owners that didn't
        # supply acceptance_criteria, synthesize a minimal
        # `output_path_exists` check from the deliverable_spec.  Phase
        # 6.0.2 requires ≥1 mechanical required AC at create_task time;
        # without this the worker delegation fails with a confusing
        # back-end error.  The synthesized AC is deliberately weak (file
        # exists, non-empty) — the caller should override with sharper
        # checks (matches_regex, jsonschema, etc.) when the deliverable
        # shape is known.
        _COORDINATOR_OWNERS = {"director", "team-lead"}
        acs = args.get("acceptance_criteria")
        _is_coordinator = (not acs) and (
            args.get("owner_id") in _COORDINATOR_OWNERS
        )
        if _is_coordinator:
            acs = []  # closed by the children-done autoclose hook
        elif not acs:
            # Worker leaf, no explicit ACs → synthesize from deliverable_spec.
            # Shared helper (also used by the /proposals/convert path) so the
            # two create paths can't drift — see F13 (2026-06-02).
            acs = tasks.synthesize_default_acceptance_criteria(
                args.get("owner_id"), args["deliverable_spec"] or "",
            )
        # [PARENT_ID_GUARD_2026_05_29]  Reject a parent_id that doesn't
        # resolve to a real task in this tenant.  The Director, when
        # fanning out (Pattern C), tends to create the parent + all
        # subtasks in ONE turn and passes a PLACEHOLDER string
        # (literal "PARENT_TASK_ID") for the children's parent_id —
        # because it hasn't seen the parent's real id yet.  Silently
        # accepting that produced orphaned subtasks pointing at a
        # non-existent parent, broke consolidation, and made the whole
        # decomposition fail.  Failing loudly here forces the Director
        # to do it correctly: create the parent FIRST, read its task_id
        # from the result, THEN create each child with that id.
        _pid = args.get("parent_id")
        if _pid:
            from ..ledger.tasks import get_task as _get_task
            if _get_task(_pid, ctx.tenant_id, ctx.user_id) is None:
                return _err(
                    f"parent_id {_pid!r} does not exist. For a parallel "
                    f"fanout: FIRST call create_task for the PARENT (no "
                    f"parent_id) and read its task_id from the result, "
                    f"THEN call create_task for each subtask passing that "
                    f"real task_id as parent_id. Do NOT invent a "
                    f"placeholder like 'PARENT_TASK_ID'."
                )

        # [SUBTASK_OWNER_GUARD_2026_05_29]  A SUBTASK (parent_id set) must
        # be owned by a dispatchable IC — never the Director.  The inbox
        # poller only dispatches IC-owned tasks; a subtask assigned to
        # 'director' (the model self-assigns despite the persona saying
        # "never assign yourself") sits OPEN forever, never executes, and
        # hangs the whole fanout.  Caught live 2026-05-29: 2 of 4
        # subtasks were owner=director and hung indefinitely.  The PARENT
        # (parent_id=None) may be director-owned — it's a coordinator,
        # not dispatched.
        _DISPATCHABLE_ICS = {
            "researcher", "builder", "critic-claim", "critic-prose",
            "phishing-analyst", "threat-analyst", "report-writer", "team-lead",
        }
        _owner = args.get("owner_id")
        if _pid and _owner not in _DISPATCHABLE_ICS:
            return _err(
                f"subtask owner_id={_owner!r} is not a dispatchable IC. "
                f"Subtasks must be assigned to one of {sorted(_DISPATCHABLE_ICS)} "
                f"— NEVER to 'director' (yourself): the platform will never "
                f"execute a director-owned subtask and it will hang OPEN "
                f"forever. Pick the IC whose role fits the work "
                f"(researcher for analysis, builder for code/files)."
            )
        task_id = tasks.create_task(
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            owner_id=args["owner_id"],
            brief=args["brief"],
            deliverable_spec=args["deliverable_spec"],
            acceptance_criteria=acs,
            assigner_id=ctx.agent_id,
            parent_id=_pid,
            inputs=args.get("inputs") or [],
            due_ts=args.get("due_ts"),
            depends_on=args.get("depends_on") or None,
            team_id=args.get("team_id"),
            origin="background" if ctx.is_background else "foreground",
            # Coordinators (director/team-lead) carry zero required ACs;
            # they're closed structurally by the children-done autoclose
            # hook, so bypass the Phase 6.0.2 ≥1-mechanical-AC contract.
            _phase_6_legacy_exempt=_is_coordinator,
        )
        return _ok({
            "task_id": task_id,
            "owner_id": args["owner_id"],
            "status": "open",
            "is_coordinator": _is_coordinator,
            "synthesized_acs": (
                args.get("acceptance_criteria") is None and not _is_coordinator
            ),
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


@tool(
    name="defer_proposal",
    description=(
        "Defer a proactive proposal — keeps it in the queue but decays "
        "its score 0.9× per cycle until re-triggered or expired. "
        "Director-only; use when you want to revisit later but not now."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "proposal_id": {"type": "string"},
        },
        "required": ["proposal_id"],
    },
)
async def defer_proposal_tool(args: dict[str, Any]) -> dict[str, Any]:
    ctx, err = _ctx_or_error()
    if err:
        return err
    try:
        proposals.defer_proposal(
            proposal_id=args["proposal_id"],
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
        )
        return _ok({"proposal_id": args["proposal_id"], "status": "deferred"})
    except Exception as e:
        return _err(f"defer_proposal failed: {e}")


@tool(
    name="list_proposals",
    description=(
        "List proactive proposals in this user's queue.  Returns "
        "pending + deferred sorted by decay_score DESC.  Director uses "
        "this before deciding to convert/defer/shelve."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Max rows to return (default 20)",
            },
        },
    },
)
async def list_proposals_tool(args: dict[str, Any]) -> dict[str, Any]:
    ctx, err = _ctx_or_error()
    if err:
        return err
    try:
        limit = int(args.get("limit") or 20)
        rows = proposals.queue(
            tenant_id=ctx.tenant_id, user_id=ctx.user_id, limit=limit,
        )
        compact = [
            {
                "id":               r.get("id"),
                "signal_type":      r.get("signal_type"),
                "impact_score":     r.get("impact_score"),
                "decay_score":      r.get("decay_score"),
                "proposed_brief":   (r.get("proposed_brief") or "")[:200],
                "proposed_assignee": r.get("proposed_assignee"),
                "status":           r.get("status"),
                "rationale":        r.get("rationale"),
            } for r in rows
        ]
        return _ok({"count": len(compact), "proposals": compact})
    except Exception as e:
        return _err(f"list_proposals failed: {e}")


@tool(
    name="list_lessons_for_review",
    description=(
        "List active org_memory lessons whose review_after has passed. "
        "Director or critic-prose calls this to surface lessons the "
        "user should review (keep / amend / retire)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "limit": {"type": "integer"},
        },
    },
)
async def list_lessons_for_review_tool(args: dict[str, Any]) -> dict[str, Any]:
    ctx, err = _ctx_or_error()
    if err:
        return err
    try:
        from ..ledger.store import LedgerStore
        from ..proposals.lessons import find_lessons_due_for_review
        tbl = LedgerStore.get().table("org_memory")
        rows = find_lessons_due_for_review(
            lessons_tbl=tbl,
            tenant_id=ctx.tenant_id, user_id=ctx.user_id,
            limit=int(args.get("limit") or 50),
        )
        compact = [
            {
                "id":               r.get("id"),
                "trigger":          r.get("trigger"),
                "rule":             r.get("rule"),
                "confidence":       r.get("confidence"),
                "reinforcement_count": r.get("reinforcement_count"),
                "review_after":     r.get("review_after"),
                "expires_at":       r.get("expires_at"),
            } for r in rows
        ]
        return _ok({"count": len(compact), "lessons": compact})
    except Exception as e:
        return _err(f"list_lessons_for_review failed: {e}")


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
    path = args.get("path", "")
    if not path:
        return _err("attach_output: path is required")
    task_id = args.get("task_id", "")
    if not task_id:
        return _err("attach_output: task_id is required")

    # [F5_STAT_BEFORE_ATTACH_2026_05_26] Verify the file actually
    # exists on the daemon's host before recording it as a deliverable.
    # Previously, agents could `attach_output("team/drafts/x.md")` WITHOUT
    # ever having called write_file — critic-claim/prose then went into
    # multi-minute file-not-found retry loops (v2, v4) before F2 timed
    # them out.  Pre-flight via the daemon's read_file (1-line probe)
    # surfaces the missing file at the source — the IC sees a clear
    # error and either writes the file or doesn't attach.
    try:
        from ..tool_dispatcher import dispatch as _dispatch
        stat_result = await _dispatch(
            "read_file",
            {"path": path, "offset": 0, "limit": 1},
        )
        # dispatch_remote_tool returns MCP-shape: {content:[{text:...}], isError}
        is_err = bool(stat_result.get("isError"))
        text = ""
        for blk in stat_result.get("content") or []:
            if blk.get("type") == "text":
                text = blk.get("text", "")
                break
        if is_err or text.lower().startswith("error:") or "not found" in text.lower():
            return _err(
                f"attach_output refused — file does not exist at {path!r}. "
                f"Write the file first (write_file), then attach. "
                f"Daemon said: {text[:200]}"
            )
    except ImportError:
        # In-process attach (e.g. running tests without the dispatcher).
        # Fall through and skip the stat check.
        pass
    except Exception as e:
        # Daemon unreachable / dispatch failure — DON'T block attach
        # (we'd rather record the path and let the critic flag it later
        # than hard-fail when the daemon is briefly offline).  Log and
        # proceed.
        logger.warning(f"[attach_output] stat probe failed (proceeding): {e}")

    try:
        tasks.attach_output(
            task_id=task_id,
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            path=path,
        )
        return _ok({"task_id": task_id, "path": path})
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


def _run_required_ac_checks(
    *, task_id: str, tenant_id: str, user_id: str,
) -> tuple[bool, dict[str, str], list[str]]:
    """[AC_RESULTS_WIRING_2026_05_29]  Run the mechanical acceptance-
    criterion executors server-side and persist the verdicts to
    `extras.ac_results`.

    THE missing link in Phase 6.0: the hard-gate in `tasks.update_status`
    reads `extras.ac_results[ac_id] == 'pass'` for every required AC,
    but until now NOTHING ever wrote `ac_results`.  So every required AC
    was permanently `None` → `review_decision(approved)` could never
    transition a task to `done` → critics looped to max_turns on every
    single review.  Mechanical checks (output_path_exists,
    output_path_matches_regex, …) are deterministic and belong on the
    server, not asserted by the LLM.

    NB: this is a MODULE-LEVEL helper deliberately placed BEFORE the
    `@tool(name="review_decision")` decorator — putting it between the
    decorator and review_decision_tool makes the decorator wrap THIS
    function instead, un-registering the real tool (caught live
    2026-05-29: review_decision fell through to the daemon path and the
    critic looped).

    Returns (all_required_pass, {ac_id: status}, human_detail_lines).
    """
    import os as _os
    import json as _json
    from ..acceptance.executors import run_check
    from ..ledger.store import LedgerStore, ns_filter

    tbl = LedgerStore.get().table("agent_tasks")
    where = f"{ns_filter(tenant_id, user_id)} AND id = '{task_id}'"
    arr = tbl.search().where(where).limit(1).to_arrow()
    if arr.num_rows == 0:
        return False, {}, [f"task {task_id!r} not found for AC check"]
    row = {c: arr.column(c)[0].as_py() for c in arr.column_names}

    acs = row.get("acceptance_criteria") or []
    outputs = row.get("outputs") or []
    try:
        extras = _json.loads(row.get("extras_json") or "{}")
    except Exception:
        extras = {}
    results: dict[str, str] = dict(extras.get("ac_results") or {})

    ctx_for_check = {
        "workspace_root": _os.environ.get("VEILGUARD_WORKSPACE_ROOT", "/workspace"),
        "outputs": outputs,
        "task_id": task_id,
    }
    detail: list[str] = []
    all_required_pass = True
    for ac in acs:
        ac_id = ac.get("id") or "?"
        check_kind = ac.get("check_kind") or ""
        check_args = ac.get("check_args") or {}
        required = ac.get("required", True)
        # If the AC's check_args has no path but the task has exactly one
        # output, default to that output (covers the synthesized
        # AC-default when the IC wrote to the spec'd path).
        if check_kind in ("output_path_exists", "output_path_matches_regex"):
            if isinstance(check_args, dict) and not check_args.get("path") and outputs:
                check_args = {**check_args, "path": outputs[0]}
        res = run_check(check_kind, check_args, ctx_for_check)
        status = getattr(res, "status", "error")
        results[ac_id] = status
        detail.append(
            f"{ac_id} [{check_kind}] → {status}"
            + (f" ({getattr(res, 'reason', '')})" if status != "pass" else "")
        )
        if required and status != "pass":
            all_required_pass = False

    # Persist ac_results back to extras_json so the hard-gate can read it.
    extras["ac_results"] = results
    import time as _t
    tbl.update(where=where, values={
        "extras_json": _json.dumps(extras),
        "updated_ts":  _t.time(),
    })
    return all_required_pass, results, detail


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
        # [AC_RESULTS_WIRING_2026_05_29]  Before an `approved` decision
        # tries to transition the task to `done`, RUN the mechanical AC
        # checks and persist ac_results — otherwise the hard-gate in
        # update_status sees ac_results=None for every required AC and
        # raises IllegalTransition, which the critic cannot fix by
        # retrying (it loops to max_turns).  If a required mechanical
        # check fails (e.g. the artifact file doesn't actually exist),
        # auto-downgrade the decision to changes_requested with the
        # failure detail so the IC fixes it instead of the critic
        # rubber-stamping a missing deliverable.
        if decision == "approved":
            ok_checks, ac_results, ac_detail = _run_required_ac_checks(
                task_id=args["task_id"],
                tenant_id=ctx.tenant_id,
                user_id=ctx.user_id,
            )
            if not ok_checks:
                decision = "changes_requested"
                args = {
                    **args,
                    "notes": (
                        f"[auto-downgraded from approved: mechanical "
                        f"acceptance checks failed] "
                        f"{'; '.join(ac_detail)}. Original critic note: "
                        f"{args.get('notes', '')}"
                    ),
                }
        # [REVISION_ROUND_CAP_2026_06_01] Bound the critic↔IC revision loop.
        # changes_requested bounces the task review→in_progress and
        # re-dispatches the IC; with NO cap, an UNSATISFIABLE brief (e.g.
        # "Review firewall logs" with no actual logs) loops FOREVER — each
        # round burns ~25 IC turns + ~10 critic turns. Measured 2026-06-01:
        # one such task ran 40 LLM turns / 982k input tokens in 30 min and
        # would never stop (the artifact can't satisfy the critic, and the
        # poller re-claims review/in_progress endlessly). After the cap-th
        # rejection, STOP bouncing: CANCEL the task (legal from `review`,
        # and the inbox poller only claims open/review/in_progress so a
        # cancelled task is NEVER re-dispatched). Tunable via env; default 2
        # = the IC gets exactly one revision attempt before escalate-by-
        # cancel. A human re-scopes + re-creates if the work still matters.
        import os as _os
        _MAX_REVISION_ROUNDS = max(
            1, int(_os.environ.get("VEILGUARD_MAX_REVISION_ROUNDS", "2"))
        )
        _cap_hit = False
        _prior_rejections = 0
        if decision == "changes_requested":
            try:
                for _c in comments.list_comments(
                    task_id=args["task_id"],
                    tenant_id=ctx.tenant_id,
                    user_id=ctx.user_id,
                ):
                    if (_c.get("kind") == "review_decision"
                            and (_c.get("body") or "").startswith(
                                "changes_requested")):
                        _prior_rejections += 1
            except Exception:
                _prior_rejections = 0
            # This decision would be the (_prior_rejections + 1)-th rejection.
            if _prior_rejections + 1 >= _MAX_REVISION_ROUNDS:
                _cap_hit = True
                logger.warning(
                    f"[review_decision] task {args['task_id']} hit revision-"
                    f"round cap ({_MAX_REVISION_ROUNDS}) on rejection "
                    f"#{_prior_rejections + 1}; CANCELLING instead of bouncing "
                    f"to in_progress (IC↔critic could not converge)."
                )

        # Add review_decision comment with the full body.
        if _cap_hit:
            body = (
                f"changes_requested: {args['notes']}  "
                f"[REVISION-CAP HIT] rejection #{_prior_rejections + 1} ≥ cap "
                f"{_MAX_REVISION_ROUNDS} → task auto-CANCELLED to stop the "
                f"IC↔critic loop. Re-scope the brief and re-create the task "
                f"if the work is still needed."
            )
        else:
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
        if _cap_hit:
            target_status = "cancelled"   # terminal: kill the revision loop
        tasks.update_status(
            task_id=args["task_id"],
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            new_status=target_status,
            actor_agent_id=ctx.agent_id,
        )

        # On changes_requested, restore owner_id to the original IC so
        # they can iterate.  submit_for_review stashed the original
        # owner in the review_request comment body as
        # `original_owner=<id>`.  Without this, the task would stay
        # assigned to the critic and the inbox poller would re-dispatch
        # to the critic in an infinite loop.
        # [REVISION_ROUND_CAP_2026_06_01] Skip the bounce when the cap was
        # hit — the task is now `cancelled`; restoring the IC owner + reset
        # lease would re-arm the very loop we just stopped.
        if decision == "changes_requested" and not _cap_hit:
            try:
                cmts = comments.list_comments(
                    task_id=args["task_id"],
                    tenant_id=ctx.tenant_id,
                    user_id=ctx.user_id,
                )
                original_owner = None
                # Most recent review_request comment wins (in case of
                # multi-round critique).
                for c in reversed(cmts):
                    if c.get("kind") != "review_request":
                        continue
                    body = c.get("body") or ""
                    for tok in body.split():
                        if tok.startswith("original_owner="):
                            original_owner = tok.split("=", 1)[1]
                            break
                    if original_owner:
                        break
                if original_owner:
                    from ..ledger.store import LedgerStore, ns_filter
                    import time as _time
                    import json as _json
                    tbl = LedgerStore.get().table("agent_tasks")
                    where = (
                        f"{ns_filter(ctx.tenant_id, ctx.user_id)} "
                        f"AND id = '{args['task_id']}'"
                    )
                    # Phase 6.4 — set extras_json.is_revision=True so the
                    # revision-priority lane sorter (runtime_health.
                    # revision_lane) claims this task before fresh work
                    # from the same persona.
                    cur_arr = tbl.search().where(where).limit(1).to_arrow()
                    extras_str = "{}"
                    if cur_arr.num_rows > 0:
                        cur_extras = cur_arr.column("extras_json")[0].as_py()
                        try:
                            cur_extras = _json.loads(cur_extras or "{}")
                        except Exception:
                            cur_extras = {}
                        cur_extras["is_revision"] = True
                        extras_str = _json.dumps(cur_extras)
                    tbl.update(where=where, values={
                        "owner_id":    original_owner,
                        "lease_owner": "",
                        "lease_until": 0.0,
                        "updated_ts":  _time.time(),
                        "extras_json": extras_str,
                    })
            except Exception:
                # Non-fatal: the task is still in_progress, just owned by
                # the critic.  Director can re-route manually.
                pass

        # [PHASE_2_TYPED_TEAM_CHANNELS_2026_05_26] On approve, publish
        # the deliverable into the appropriate typed channel so other
        # team members can recall it.  Per spec §3.4.1:
        #   target=team_knowledge   → team/<tid>/knowledge/<user_id>
        #   target=org_blackboard   → blackboard/<user_id>
        #   target=user_deliverable → no channel (it's for the user;
        #                              they see it in the chat)
        # The submission target was stamped on the review_request
        # comment by submit_for_review ("target=<X>"); we re-read it
        # here rather than threading another arg.  team_id is read
        # from the persona registry (falls back to "core" — the only
        # team today).
        if decision == "approved":
            try:
                await _publish_to_typed_channel(
                    task_id=args["task_id"],
                    tenant_id=ctx.tenant_id,
                    user_id=ctx.user_id,
                    reviewer_agent_id=ctx.agent_id,
                )
            except Exception as e:
                # Non-fatal: approval still stands, just no team-channel
                # publish.  Log for follow-up.
                logger.warning(
                    f"[review_decision] team-channel publish failed for "
                    f"{args['task_id']}: {e}"
                )

        return _ok({
            "task_id": args["task_id"],
            "decision": decision,
            "new_status": target_status,
            "revision_cap_hit": _cap_hit,
        })
    except Exception as e:
        return _err(f"review_decision failed: {e}")


async def _publish_to_typed_channel(
    *,
    task_id: str,
    tenant_id: str,
    user_id: str,
    reviewer_agent_id: str,
) -> None:
    """Write a team_knowledge / blackboard channel record for an approved
    task.  Best-effort; never raises (caller wraps in try/except).

    The record is a STUB (task brief + outputs[], not the raw file
    bytes).  Reasoning: deliverables are arbitrary files on the daemon's
    host (binary, large, sometimes images).  Reading them via daemon
    round-trip + injecting full contents into TCMM is a separate design
    decision per spec §3.5.  A stub gives any team member's recall
    enough breadcrumb to fetch the artifact themselves.
    """
    # Re-read task to find target + outputs.  submit_for_review stamps
    # the target on the review_request comment as "target=<X>".
    task_row = tasks.get_task(task_id, tenant_id, user_id)
    if task_row is None:
        return
    cmts = comments.list_comments(
        task_id=task_id, tenant_id=tenant_id, user_id=user_id,
    )
    target = ""
    for c in reversed(cmts):
        if c.get("kind") != "review_request":
            continue
        for tok in (c.get("body") or "").split():
            if tok.startswith("target="):
                target = tok.split("=", 1)[1]
                break
        if target:
            break
    if not target:
        return  # No target stamped → nothing to publish.

    # user_deliverable → no team channel; user sees the artifact in chat.
    if target == "user_deliverable":
        return

    outputs = task_row.get("outputs") or []
    brief = task_row.get("brief") or ""
    deliverable_spec = task_row.get("deliverable_spec") or ""
    owner_id = task_row.get("owner_id") or "?"

    # Resolve team_id.  Today the only team is "core"; the registry
    # lookup is the right architecture for when more teams exist.
    team_id = "core"
    try:
        from .. import persona_registry as _reg  # may not exist on first import
        # fall through to the default if no helper is wired yet
    except Exception:
        pass

    # [F14_FS_SAFE_CHANNEL_2026_05_27] Use slash-free namespace ids
    # because TCMM derives a file path from the session_id (`session_<sid_prefix>.json`)
    # and Windows can't open a filename containing `/`.  Slashes in the
    # logical channel name break TCMM's own live-block persistence —
    # observed 2026-05-27 07:01:23 with "[TCMM] Failed to persist live
    # blocks: [Errno 2] No such file or directory: ...session_team/core/kn.json".
    # The underscore form lines up with how TCMM normalizes namespaces
    # internally anyway (slashes → underscores in `_normalize_id`).
    if target == "team_knowledge":
        channel = f"team_{team_id}_knowledge_{user_id}"
    elif target == "org_blackboard":
        channel = f"blackboard_{user_id}"
    else:
        return  # Unknown target — skip.

    record_text = (
        f"[approved {task_id}] {brief}\n"
        f"deliverable_spec: {deliverable_spec}\n"
        f"outputs: {', '.join(repr(p) for p in outputs) if outputs else '(none)'}\n"
        f"reviewed_by: {reviewer_agent_id}; original_owner: {owner_id}"
    )

    payload = {
        "conversation_id": channel,
        "user_id": user_id,
        "items": [
            {
                "text": record_text,
                "origin": "observation",
                "role": "assistant",
                # Provenance: the CRITIC published this — they're the
                # authority that approved it.
                "extracted_by": f"agent:{reviewer_agent_id}",
                "source_kind": "TOOL_RESULT",
                "metadata": {
                    "task_id": task_id,
                    "target": target,
                    "team_id": team_id,
                },
            }
        ],
    }

    # POST to TCMM /ingest_turn.  Same pattern as observe_agent_output.
    import httpx
    from ..middleware.tcmm import (
        TCMM_URL, TCMM_ENABLED, TCMM_RENDER_TIMEOUT_S, VEILGUARD_INTERNAL_SECRET,
    )
    if not TCMM_ENABLED:
        return
    headers = {}
    if VEILGUARD_INTERNAL_SECRET:
        headers["x-veilguard-internal-secret"] = VEILGUARD_INTERNAL_SECRET
    async with httpx.AsyncClient(timeout=TCMM_RENDER_TIMEOUT_S) as client:
        r = await client.post(
            f"{TCMM_URL.rstrip('/')}/ingest_turn",
            json=payload, headers=headers,
        )
    if r.status_code >= 400:
        logger.warning(
            f"[typed_channel_publish] ingest_turn {channel!r} returned "
            f"{r.status_code}: {r.text[:200]}"
        )
    else:
        logger.info(
            f"[typed_channel_publish] task={task_id} target={target} "
            f"→ {channel!r} OK"
        )


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


# ── Phase 7.5 — agent_teams MCP surface ────────────────────────────────


@tool(
    name="create_team",
    description=(
        "Director-only: form a new Team — a named bundle of agents with "
        "a shared lead, budget, and cost ceiling.  Tasks assigned to "
        "the team (via create_task's team_id arg) consume the budget; "
        "create_task fails when the team has crossed budget_usd × "
        "budget_cap.  Use this when delegating a multi-task project to "
        "a sub-organisation (Compliance team, Threat-research team)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "lead_agent_id": {
                "type": "string",
                "description": "agent_id of the team lead — usually 'team-lead'",
            },
            "member_agent_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "ICs and consultants in the team",
            },
            "budget_usd": {
                "type": "number",
                "description": "Soft budget envelope; 0 = no spend allowed",
            },
            "budget_cap": {
                "type": ["number", "null"],
                "description": "Multiplier on budget_usd; default 1.0 (hard ceiling), 1.2 = 20% slack",
            },
        },
        "required": ["name", "lead_agent_id", "budget_usd"],
    },
)
async def create_team_tool(args: dict[str, Any]) -> dict[str, Any]:
    ctx, err = _ctx_or_error()
    if err:
        return err
    try:
        from ..ledger import teams as _teams
        team_id = _teams.create_team(
            tenant_id=ctx.tenant_id, user_id=ctx.user_id,
            name=args["name"],
            lead_agent_id=args["lead_agent_id"],
            member_agent_ids=args.get("member_agent_ids") or [],
            budget_usd=float(args["budget_usd"]),
            budget_cap=args.get("budget_cap"),
            created_by_agent_id=ctx.agent_id,
        )
        return _ok({"team_id": team_id})
    except ValueError as e:
        return _err(f"create_team rejected: {e}")
    except Exception as e:
        return _err(f"create_team failed: {e}")


@tool(
    name="list_teams",
    description=(
        "Return all teams for the current tenant.  Useful for Director "
        "before deciding whether to route work to an existing team or "
        "create a new one."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "default": 200},
        },
    },
)
async def list_teams_tool(args: dict[str, Any]) -> dict[str, Any]:
    ctx, err = _ctx_or_error()
    if err:
        return err
    try:
        from ..ledger import teams as _teams
        rows = _teams.list_teams(
            tenant_id=ctx.tenant_id, user_id=ctx.user_id,
            limit=int(args.get("limit", 200)),
        )
        return _ok({"teams": rows, "count": len(rows)})
    except Exception as e:
        return _err(f"list_teams failed: {e}")


@tool(
    name="team_cost_report",
    description=(
        "Cost rollup for a team — sums every team-tagged task's "
        "cost_attributed_usd live and returns (attributed, ceiling, "
        "exceeded).  Refreshes the cached snapshot on agent_teams."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "team_id": {"type": "string"},
        },
        "required": ["team_id"],
    },
)
async def team_cost_report_tool(args: dict[str, Any]) -> dict[str, Any]:
    ctx, err = _ctx_or_error()
    if err:
        return err
    try:
        from ..ledger import teams as _teams
        team = _teams.get_team(args["team_id"], ctx.tenant_id, ctx.user_id)
        if team is None:
            return _err(f"team_id {args['team_id']!r} not found")
        attributed = _teams.team_cost_attributed(
            team_id=args["team_id"],
            tenant_id=ctx.tenant_id, user_id=ctx.user_id,
            refresh_cache=True,
        )
        exceeded, _, ceiling = _teams.budget_exceeded(
            team_id=args["team_id"],
            tenant_id=ctx.tenant_id, user_id=ctx.user_id,
        )
        return _ok({
            "team_id":          args["team_id"],
            "name":             team.get("name"),
            "attributed_usd":   round(attributed, 4),
            "budget_usd":       team.get("budget_usd"),
            "budget_cap":       team.get("budget_cap"),
            "ceiling_usd":      round(ceiling, 4),
            "exceeded":         exceeded,
            "pct_consumed":     (
                round(100.0 * attributed / ceiling, 1) if ceiling > 0 else None
            ),
        })
    except Exception as e:
        return _err(f"team_cost_report failed: {e}")


# ── Server factory ──────────────────────────────────────────────────────


_ALL_TOOLS = [
    # Director
    create_task_tool,
    assign_task_tool,
    convert_proposal_tool,
    shelve_proposal_tool,
    defer_proposal_tool,
    list_proposals_tool,
    list_lessons_for_review_tool,
    # IC
    accept_task_tool,
    add_comment_tool,
    attach_output_tool,
    submit_for_review_tool,
    get_task_tool,
    inbox_tool,
    # Critic
    review_decision_tool,
    # Phase 7.5 — Director team management
    create_team_tool,
    list_teams_tool,
    team_cost_report_tool,
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
