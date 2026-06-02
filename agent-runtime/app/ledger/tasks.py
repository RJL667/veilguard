"""agent_tasks CRUD.

Public surface:
  create_task(...)        — Director's tool entry point
  accept_task(...)        — IC's tool entry point
  update_status(...)      — state machine transitions
  attach_output(...)      — IC adds a deliverable path
  submit_for_review(...)  — IC sends to Critic
  get_task(task_id)       — single-row read
  inbox(agent_id, ...)    — owner-scoped derived view

State machine (enforced):
  open → accepted → in_progress → (review | blocked | cancelled)
  review → (done | in_progress)        (Critic decisions)
  blocked → in_progress                (when blocker cleared)
  in_progress → done | cancelled       (terminal)

Cancellation cascade (spec §11.2.4):
  Cancelling a parent immediately sets descendant open/blocked → cancelled
  and flags descendant in_progress to checkpoint and stop.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional

from .store import LedgerStore, ns_filter

logger = logging.getLogger("agent-runtime.ledger.tasks")


# Allowed transitions per state machine.  Programming-error guard;
# upstream tool wrappers should validate before calling.
_TRANSITIONS = {
    "open":        {"accepted", "cancelled"},
    "accepted":    {"in_progress", "cancelled"},
    "in_progress": {"review", "blocked", "done", "cancelled"},
    "blocked":     {"in_progress", "cancelled"},
    "review":      {"done", "in_progress", "cancelled"},
    "done":        set(),
    "cancelled":   set(),
}


# Valid owner_ids — must match a persona file under services/agent-runtime/
# agents/*.md.  Caught a real Director planning bug 2026-05-25 where
# Director tried to assign a task to owner_id="critic" (no such persona;
# valid names are critic-claim and critic-prose) — the task just sat in
# `open` forever because the inbox_poller's ELIGIBLE_OWNERS allowlist
# rejected the bare name silently.  Now create_task raises ValueError
# immediately so Director gets feedback on the same turn and can re-plan
# with a correct persona name.
#
# Keep this in sync with the agents/*.md files.  When adding a new
# persona, add it here too — Lance schema doesn't enforce, so this is
# our gate.
VALID_OWNER_IDS = frozenset({
    "director",
    "researcher",
    "builder",
    "critic-claim",
    "critic-prose",
    "phishing-analyst",
    "threat-analyst",
    "report-writer",
    # Phase 7.5 — team-lead persona (mini-Director scoped to one team).
    "team-lead",
})

# Owners that coordinate a fan-out (group children) rather than personally
# produce a file deliverable. They carry ZERO acceptance criteria and close
# structurally via the children-done autoclose hook.
_COORDINATOR_OWNERS = frozenset({"director", "team-lead"})


def synthesize_default_acceptance_criteria(
    owner_id: str, deliverable_spec: str,
) -> list[dict]:
    """Synthesize a minimal AC when a caller supplied none.

    Phase 6.0.2 requires >=1 required mechanical AC at create_task time.
    Callers that don't pass one (the Director's MCP create_task tool, the
    proactive `/proposals/convert` flow) use this so the Critic can still
    gate `done`.

    - Coordinator owners (director / team-lead) → `[]` (no file deliverable;
      closed by the children-done autoclose hook — caller must pass
      `_phase_6_legacy_exempt=True`).
    - Worker owners → a single `output_path_exists` AC derived from the path
      mentioned in `deliverable_spec` (or `deliverable.md` if none).

    Shared by both create paths SO THEY CAN'T DRIFT — drift was UAT finding
    F13 (2026-06-02): `/proposals/convert` called create_task with no ACs and
    every proposal approval failed with a Phase 6.0.2 400.
    """
    if owner_id in _COORDINATOR_OWNERS:
        return []
    import re as _re
    m = _re.search(
        r"([a-zA-Z0-9_./\-]+\.(?:md|txt|json|yaml|yml|html|csv|pdf|py|js|ts|tsx))",
        deliverable_spec or "",
    )
    target_path = m.group(1) if m else "deliverable.md"
    return [{
        "id":         "AC-default",
        "statement":  f"Output file {target_path!r} exists and is non-empty",
        "check_kind": "output_path_exists",
        "check_args": {"path": target_path, "min_bytes": 1},
        "required":   True,
        "rationale":  "synthesized default: no acceptance_criteria supplied for a worker task",
    }]


def _now() -> float:
    return time.time()


def _new_id() -> str:
    return f"task-{uuid.uuid4().hex[:12]}"


def create_task(
    *,
    tenant_id: str,
    user_id: str,
    owner_id: str,
    brief: str,
    deliverable_spec: str,
    acceptance_criteria: Optional[list[dict]] = None,
    assigner_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    inputs: Optional[list[str]] = None,
    due_ts: Optional[float] = None,
    origin: str = "foreground",
    pattern: Optional[str] = None,
    constitution_version: Optional[int] = None,
    depends_on: Optional[list[str]] = None,
    team_id: Optional[str] = None,
    extras_json: Optional[str] = None,
    _phase_6_legacy_exempt: bool = False,
) -> str:
    """Insert a new Task.  Returns the new task_id.

    `parent_id` (if set) establishes a subtask relationship and is the
    first element of `lineage_chain` (server-computed per §3.8.5).

    `acceptance_criteria` (Phase 6.0.2 Director-side contract) — a list
    of AC dicts (id, statement, check_kind, check_args, required,
    rationale).  By default the call REQUIRES at least one required AC
    of a mechanical kind (i.e. `check_kind ∉ {'manual_user'}`).  Set
    `_phase_6_legacy_exempt=True` ONLY for migrations / backfills /
    tests that need to bypass the Director contract; production callers
    must always supply real ACs.

    Raises ValueError if:
      - `owner_id` is not a known persona — Director's most common
        planning mistake is shortening a persona name (e.g. "critic"
        instead of "critic-claim").
      - `acceptance_criteria` is empty (and not legacy-exempt).
      - No `required=True` AC has a mechanical `check_kind`.
    """
    if owner_id not in VALID_OWNER_IDS:
        raise ValueError(
            f"unknown owner_id {owner_id!r}; valid persona names are "
            f"{sorted(VALID_OWNER_IDS)}"
        )

    # Phase 6.0.2 Director-side acceptance contract.
    acs = list(acceptance_criteria or [])
    if not _phase_6_legacy_exempt:
        if not acs:
            raise ValueError(
                "create_task requires non-empty acceptance_criteria "
                "(Phase 6.0.2 Director contract). Use _phase_6_legacy_exempt=True "
                "ONLY for migrations / backfills / tests."
            )
        # Iron rule: at least one required AC must be mechanical.
        from ..acceptance.executors import MECHANICAL_CHECK_KINDS
        required_mechanical = [
            ac for ac in acs
            if ac.get("required", True)
            and ac.get("check_kind") in MECHANICAL_CHECK_KINDS
        ]
        if not required_mechanical:
            raise ValueError(
                "acceptance_criteria must include at least one required AC "
                "with a mechanical check_kind. Got kinds: "
                f"{[ac.get('check_kind') for ac in acs]!r}. "
                f"Mechanical kinds: {sorted(MECHANICAL_CHECK_KINDS)}."
            )

    # Normalise AC rows for Lance (check_args must be JSON string).
    import json as _json
    normalized_acs: list[dict] = []
    for ac in acs:
        normalized_acs.append({
            "id":         str(ac.get("id") or f"AC-{len(normalized_acs)+1}"),
            "statement":  str(ac["statement"]),
            "check_kind": str(ac["check_kind"]),
            "check_args": (
                ac["check_args"] if isinstance(ac.get("check_args"), str)
                else _json.dumps(ac.get("check_args") or {})
            ),
            "required":   bool(ac.get("required", True)),
            "rationale":  ac.get("rationale"),
        })

    tbl = LedgerStore.get().table("agent_tasks")
    task_id = _new_id()

    # Lineage chain: server-computed, never client-supplied.
    lineage: list[str] = []
    if parent_id:
        parent_lineage = _get_lineage(parent_id) or []
        lineage = list(parent_lineage) + [parent_id]

    # [PHASE_7_5_DEPENDS_ON_2026_05_28] Validate cross-lineage deps.
    # Director is responsible for cycle prevention at planning time —
    # we validate that every dep exists in the same (tenant, user)
    # scope and reject obvious self-loops here.  Full cycle detection
    # happens via `verify_depends_on_acyclic` at the Director-tool
    # boundary and is exposed for tests.
    deps = list(depends_on or [])
    if deps:
        if task_id in deps:
            raise ValueError(
                f"create_task: depends_on must not include the new task's "
                f"own id ({task_id!r}) — self-loop"
            )
        # Each dep id must exist for this tenant/user.  Reject
        # gracefully so Director sees a clear error rather than a
        # task that's permanently un-claimable.
        missing = _missing_dep_ids(
            tenant_id=tenant_id, user_id=user_id, dep_ids=deps,
        )
        if missing:
            raise ValueError(
                f"create_task: depends_on includes unknown task_id(s) for "
                f"({tenant_id}/{user_id}): {sorted(missing)}"
            )

    # [PHASE_7_5_TEAM_ID_2026_05_28] Team membership + budget guard.
    # Director's mini-plan-time check: if the assigned team is already
    # over its budget × cap envelope, reject before any work is queued.
    # Recurrence: the budget rolls over to the next operator-funded
    # period; this only blocks at the boundary, doesn't auto-shelve.
    if team_id:
        from . import teams as _teams
        team_row = _teams.get_team(team_id, tenant_id, user_id)
        if team_row is None:
            raise ValueError(
                f"create_task: team_id {team_id!r} not found for "
                f"({tenant_id}/{user_id})"
            )
        if (team_row.get("status") or "active") != "active":
            raise ValueError(
                f"create_task: team {team_id!r} is "
                f"status={team_row.get('status')!r}, not active"
            )
        exceeded, attributed, ceiling = _teams.budget_exceeded(
            team_id=team_id, tenant_id=tenant_id, user_id=user_id,
        )
        if exceeded:
            raise ValueError(
                f"create_task: team {team_id!r} has already crossed its "
                f"budget envelope (attributed=${attributed:.2f} "
                f">= ceiling=${ceiling:.2f}); no new work can be queued"
            )

    row = {
        "id": task_id,
        "kind": "task",
        "status": "open",
        "parent_id": parent_id,
        "lineage_chain": lineage,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "created_by_agent_id": assigner_id,
        "created_ts": _now(),
        "updated_ts": _now(),
        "cost_attributed_usd": 0.0,
        "owner_id": owner_id,
        "assigner_id": assigner_id,
        "brief": brief,
        "deliverable_spec": deliverable_spec,
        "inputs": inputs or [],
        "outputs": [],
        "due_ts": due_ts,
        "trace_ref": None,
        "comments_head_hash": None,
        "origin": origin,
        "pattern": pattern,
        "constitution_version": constitution_version,
        # Sentinel values rather than NULL: LanceDB's where-clause
        # parser doesn't support `IS NULL`, so the inbox poller filters
        # on `lease_until < now`.  Empty owner + zero lease = available.
        "lease_owner": "",
        "lease_until": 0.0,
        "acceptance_criteria": normalized_acs,
        "team_id": team_id,
        "depends_on": deps,
        "extras_json": extras_json,
    }
    tbl.add([row])
    try:
        from ..events import broadcast
        broadcast({
            "type": "task_created",
            "tenant_id": tenant_id, "user_id": user_id,
            "id": task_id, "owner_id": owner_id,
            "brief": brief[:200], "status": "open",
            "parent_id": parent_id,
        })
    except Exception:
        pass
    # Phase 6.7 — count state mutations toward APR.
    try:
        from ..runtime_health import apr_record_artifact
        apr_record_artifact()
    except Exception:
        pass
    return task_id


def _get_lineage(task_id: str) -> list[str] | None:
    """Return a task's lineage_chain, or None if not found."""
    tbl = LedgerStore.get().table("agent_tasks")
    arr = tbl.search().where(f"id = '{task_id}'").limit(1).to_arrow()
    if arr.num_rows == 0:
        return None
    return arr.column("lineage_chain")[0].as_py()


def get_task(task_id: str, tenant_id: str, user_id: str) -> dict[str, Any] | None:
    """Read a single Task, tenant-isolated."""
    tbl = LedgerStore.get().table("agent_tasks")
    where = f"{ns_filter(tenant_id, user_id)} AND id = '{task_id}'"
    arr = tbl.search().where(where).limit(1).to_arrow()
    if arr.num_rows == 0:
        return None
    return _row_to_dict(arr, 0)


class IllegalTransition(ValueError):
    """Raised when `update_status` rejects a transition.

    Distinct from generic ValueError so Phase 6.0.1 hard-gate tests can
    assert specifically on the gate firing (AC-9, AC-10).
    """
    pass


def _phase_6_hard_gate(task: dict, new_status: str) -> None:
    """Phase 6.0.1 hard-gate: enforce `state='done'` invariants.

    A task transitions to `done` only if:
      (1) the last review_decision (in task_comments) was 'accepted', AND
      (2) the task has no unresolved constraint_violations, AND
      (3) every required AC has been evaluated and reports `status='pass'`.

    Raises IllegalTransition if any condition fails.  Other transitions
    (open → accepted, accepted → in_progress, etc.) pass through.

    Per spec §6.0.1, this gate must fire at BOTH write paths:
      - via `update_status` (this function — the normal path)
      - via raw Lance writes in `proposals.py`, etc. — those callers
        should invoke `_phase_6_hard_gate(...)` before their direct
        `tbl.update(values={'status': 'done', ...})` (AC-10).
    """
    if new_status != "done":
        return  # gate only applies to terminal `done`

    # (1) Review decision must be 'accepted'.
    review_decision = (task.get("extras_json") or "")
    # The review_decision is recorded as a task_comments row of kind=
    # 'review_decision' with body starting "accepted: ..." | "declined: ..."
    # — read it from task_comments rather than relying on task.extras.
    review_decision = _read_last_review_decision(
        task["id"], task["tenant_id"], task["user_id"],
    )
    if review_decision != "accepted":
        raise IllegalTransition(
            f"Phase 6.0.1 hard-gate: cannot transition task {task['id']!r} "
            f"to 'done' — last review_decision is "
            f"{review_decision!r}, expected 'accepted'."
        )

    # (2) Constraint violations must be empty.  Stored in extras_json
    #     for now; promoted to a dedicated column in Phase 7.
    import json as _json
    extras_raw = task.get("extras_json") or "{}"
    try:
        extras = _json.loads(extras_raw)
    except Exception:
        extras = {}
    violations = extras.get("constraint_violations") or []
    if violations:
        raise IllegalTransition(
            f"Phase 6.0.1 hard-gate: cannot transition task {task['id']!r} "
            f"to 'done' — constraint_violations are non-empty: {violations!r}."
        )

    # (3) All required ACs must report pass.  An AC without a recorded
    #     check_result is treated as a fail (we cannot prove pass).
    #     The check_result lives under extras.ac_results = {ac_id: status}.
    ac_results = extras.get("ac_results") or {}
    acs = task.get("acceptance_criteria") or []
    for ac in acs:
        if not ac.get("required", True):
            continue
        ac_id = ac.get("id")
        status = ac_results.get(ac_id)
        if status != "pass":
            raise IllegalTransition(
                f"Phase 6.0.1 hard-gate: cannot transition task "
                f"{task['id']!r} to 'done' — required AC "
                f"{ac_id!r} ({ac.get('check_kind')}) is "
                f"{status!r}, expected 'pass'."
            )


def _read_last_review_decision(
    task_id: str, tenant_id: str, user_id: str,
) -> str | None:
    """Return the verdict ('accepted'|'declined'|'changes_requested') of
    the most recent review_decision comment on this task, or None.
    """
    try:
        ctbl = LedgerStore.get().table("task_comments")
        where = (
            f"{ns_filter(tenant_id, user_id)} "
            f"AND task_id = '{task_id}' "
            f"AND kind = 'review_decision'"
        )
        arr = ctbl.search().where(where).to_arrow()
        if arr.num_rows == 0:
            return None
        # Most recent by ts.
        ts_col = arr.column("ts").to_pylist()
        body_col = arr.column("body").to_pylist()
        latest_idx = max(range(arr.num_rows), key=lambda i: ts_col[i] or 0)
        body = body_col[latest_idx] or ""
        # body convention: "accepted: ..." | "declined: ..." | "changes_requested: ..."
        verdict = body.split(":", 1)[0].strip().lower()
        if verdict in ("accepted", "approved"):  # legacy alias
            return "accepted"
        if verdict in ("declined", "rejected", "changes_requested"):
            return verdict
        return None
    except Exception:
        return None


def update_status(
    *,
    task_id: str,
    tenant_id: str,
    user_id: str,
    new_status: str,
    actor_agent_id: str,
) -> None:
    """Transition state machine.  Raises IllegalTransition on rejection.

    Phase 6.0.1 hard-gate: transitions to `done` go through
    `_phase_6_hard_gate(...)` which verifies (review accepted, no
    constraint violations, all required ACs pass).  Other transitions
    are validated only against the basic `_TRANSITIONS` table.
    """
    task = get_task(task_id, tenant_id, user_id)
    if task is None:
        raise IllegalTransition(f"task not found: {task_id}")
    current = task["status"]
    if new_status not in _TRANSITIONS.get(current, set()):
        raise IllegalTransition(
            f"illegal transition: {current} → {new_status} on {task_id}"
        )

    # Phase 6.0.1 hard-gate — raises IllegalTransition if `done` blocked.
    _phase_6_hard_gate(task, new_status)

    tbl = LedgerStore.get().table("agent_tasks")
    where = f"{ns_filter(tenant_id, user_id)} AND id = '{task_id}'"
    tbl.update(where=where, values={
        "status": new_status,
        "updated_ts": _now(),
    })
    try:
        from ..events import broadcast
        broadcast({
            "type": "task_status_changed",
            "tenant_id": tenant_id, "user_id": user_id,
            "id": task_id, "status": new_status,
            "owner_id": task.get("owner_id", ""),
            "actor_agent_id": actor_agent_id,
        })
    except Exception:
        pass
    try:
        from ..runtime_health import apr_record_artifact
        apr_record_artifact()  # Phase 6.7
    except Exception:
        pass

    # Append a status_change comment so the chain has the transition log.
    from . import comments  # avoid circular import
    comments.add_comment(
        task_id=task_id,
        tenant_id=tenant_id,
        user_id=user_id,
        author_id=actor_agent_id,
        kind="status_change",
        body=f"{current} -> {new_status}",
    )

    # [PARENT_AUTOCLOSE_2026_05_29] A terminal child may now complete its
    # coordinator parent.  Pattern-C fan-out roots are director/team-lead
    # owned and the inbox poller never dispatches director-owned tasks, so
    # without this hook they hang OPEN forever in the Work Queue even after
    # every sub-task finishes — which is exactly the stuck state the user
    # sees in the sidebar.
    if new_status in ("done", "cancelled"):
        try:
            _maybe_autoclose_parents(
                child_task_id=task_id,
                tenant_id=tenant_id,
                user_id=user_id,
                actor_agent_id=actor_agent_id,
            )
        except Exception:
            logger.exception(
                "[autoclose] parent check failed for child %s", task_id
            )


def _maybe_autoclose_parents(
    *,
    child_task_id: str,
    tenant_id: str,
    user_id: str,
    actor_agent_id: str,
    _depth: int = 0,
) -> list[str]:
    """[PARENT_AUTOCLOSE_2026_05_29] Close a coordinator parent once all
    of its children are terminal.

    A "coordinator" is a fan-out root (Pattern C): it has children and is
    typically director/team-lead owned, so the inbox poller never picks it
    up.  When every child reaches a terminal state (done|cancelled) AND at
    least one child is 'done', the coordinator's job is finished and we
    transition it to 'done'.  Recurses upward so a grandparent closes once
    its final branch lands.

    This is a PRIVILEGED direct write: a coordinator never receives a
    review_decision, so it can't pass `_phase_6_hard_gate`.  We instead
    require the parent to have no unmet required ACs of its own and no
    constraint_violations — if it does, it isn't a pure coordinator and we
    leave it to the normal review→done flow.

    Returns the list of parent ids closed (possibly empty).
    """
    if _depth > 8:
        return []
    child = get_task(child_task_id, tenant_id, user_id)
    if child is None:
        return []
    parent_id = child.get("parent_id")
    if not parent_id:
        return []
    parent = get_task(parent_id, tenant_id, user_id)
    if parent is None or parent.get("status") in ("done", "cancelled"):
        return []

    tbl = LedgerStore.get().table("agent_tasks")
    where = f"{ns_filter(tenant_id, user_id)} AND parent_id = '{parent_id}'"
    arr = tbl.search().where(where).select(["status"]).to_arrow()
    if arr.num_rows == 0:
        return []
    statuses = arr.column("status").to_pylist()
    if not all(s in ("done", "cancelled") for s in statuses):
        return []                       # work still outstanding
    if "done" not in statuses:
        # [COORDINATOR_DRAIN_2026_05_29] Every child is cancelled (none
        # done) → the initiative failed wholesale. Cancel the coordinator
        # too so it drains from the queue instead of hanging OPEN with no
        # path to completion.
        pwhere = f"{ns_filter(tenant_id, user_id)} AND id = '{parent_id}'"
        prev = parent.get("status")
        tbl.update(
            where=pwhere,
            values={"status": "cancelled", "updated_ts": _now()},
        )
        logger.info(
            "[autoclose] cancelled coordinator %s (%s -> cancelled): "
            "all %d children cancelled",
            parent_id, prev, arr.num_rows,
        )
        try:
            from ..events import broadcast
            broadcast({
                "type": "task_status_changed",
                "tenant_id": tenant_id, "user_id": user_id,
                "id": parent_id, "status": "cancelled",
                "owner_id": parent.get("owner_id", ""),
                "actor_agent_id": "system:autoclose",
            })
        except Exception:
            pass
        try:
            from . import comments
            comments.add_comment(
                task_id=parent_id, tenant_id=tenant_id, user_id=user_id,
                author_id="system:autoclose", kind="status_change",
                body=f"{prev} -> cancelled (auto: all {arr.num_rows} children cancelled)",
            )
        except Exception:
            pass
        return [parent_id] + _maybe_autoclose_parents(
            child_task_id=parent_id, tenant_id=tenant_id, user_id=user_id,
            actor_agent_id=actor_agent_id, _depth=_depth + 1,
        )

    # The parent must not carry unmet required ACs / violations.  A
    # coordinator is never dispatched to a critic, so its own required ACs
    # would otherwise stay unevaluated forever.  Evaluate them HERE with
    # the same mechanical executors the critic uses (output_path_exists,
    # …) and close only if they genuinely pass — the children have already
    # written the spec'd files, so this verifies the deliverable rather
    # than rubber-stamping it.  (run_check lives in acceptance.executors,
    # not ledger_mcp, so no circular import.)
    import json as _json
    import os as _os
    try:
        pextras = _json.loads(parent.get("extras_json") or "{}")
    except Exception:
        pextras = {}
    if pextras.get("constraint_violations"):
        return []

    parent_extras_json = None
    req_acs = [
        a for a in (parent.get("acceptance_criteria") or [])
        if a.get("required", True)
    ]
    if req_acs:
        from ..acceptance.executors import run_check
        ctx_for_check = {
            "workspace_root": _os.environ.get(
                "VEILGUARD_WORKSPACE_ROOT", "/workspace"
            ),
            "outputs": parent.get("outputs") or [],
            "task_id": parent_id,
        }
        ac_results = dict(pextras.get("ac_results") or {})
        all_pass = True
        for ac in req_acs:
            ck = ac.get("check_kind") or ""
            cargs = ac.get("check_args") or {}
            if (
                ck in ("output_path_exists", "output_path_matches_regex")
                and isinstance(cargs, dict)
                and not cargs.get("path")
                and ctx_for_check["outputs"]
            ):
                cargs = {**cargs, "path": ctx_for_check["outputs"][0]}
            res = run_check(ck, cargs, ctx_for_check)
            st = getattr(res, "status", "error")
            ac_results[ac.get("id") or "?"] = st
            if st != "pass":
                all_pass = False
        if not all_pass:
            logger.info(
                "[autoclose] parent %s left OPEN: required AC unmet %s",
                parent_id, ac_results,
            )
            return []
        pextras["ac_results"] = ac_results
        parent_extras_json = _json.dumps(pextras)

    # Privileged close.
    pwhere = f"{ns_filter(tenant_id, user_id)} AND id = '{parent_id}'"
    prev_status = parent.get("status")
    _values = {"status": "done", "updated_ts": _now()}
    if parent_extras_json is not None:
        _values["extras_json"] = parent_extras_json
    tbl.update(where=pwhere, values=_values)
    closed = [parent_id]
    logger.info(
        "[autoclose] closed coordinator %s (%s -> done): all %d children terminal",
        parent_id, prev_status, arr.num_rows,
    )
    try:
        from ..events import broadcast
        broadcast({
            "type": "task_status_changed",
            "tenant_id": tenant_id, "user_id": user_id,
            "id": parent_id, "status": "done",
            "owner_id": parent.get("owner_id", ""),
            "actor_agent_id": "system:autoclose",
        })
    except Exception:
        pass
    try:
        from . import comments
        comments.add_comment(
            task_id=parent_id,
            tenant_id=tenant_id,
            user_id=user_id,
            author_id="system:autoclose",
            kind="status_change",
            body=f"{prev_status} -> done (auto: all {arr.num_rows} children terminal)",
        )
    except Exception:
        pass

    # Recurse: the parent we just closed may itself complete a grandparent.
    closed += _maybe_autoclose_parents(
        child_task_id=parent_id,
        tenant_id=tenant_id,
        user_id=user_id,
        actor_agent_id=actor_agent_id,
        _depth=_depth + 1,
    )
    return closed


def attach_output(
    *,
    task_id: str,
    tenant_id: str,
    user_id: str,
    path: str,
) -> None:
    """Append a file path to outputs[] atomically."""
    task = get_task(task_id, tenant_id, user_id)
    if task is None:
        raise ValueError(f"task not found: {task_id}")
    outputs = list(task.get("outputs") or [])
    if path in outputs:
        return  # idempotent
    outputs.append(path)

    tbl = LedgerStore.get().table("agent_tasks")
    where = f"{ns_filter(tenant_id, user_id)} AND id = '{task_id}'"
    tbl.update(where=where, values={
        "outputs": outputs,
        "updated_ts": _now(),
    })


# Default critic routing.  team_knowledge gets the structural/claim
# reviewer; user_deliverable / org_blackboard get the semantic prose
# reviewer.  Eventually we'd chain claim → prose for the heavyweight
# targets, but for the MVP one critic per submission keeps the loop
# debuggable.
_DEFAULT_CRITIC_BY_TARGET = {
    "team_knowledge":   "critic-claim",
    "user_deliverable": "critic-prose",
    "org_blackboard":   "critic-prose",
}


def submit_for_review(
    *,
    task_id: str,
    tenant_id: str,
    user_id: str,
    target: str,           # team_knowledge | user_deliverable | org_blackboard
    submitter_agent_id: str,
    critic_id: Optional[str] = None,
) -> None:
    """IC submits the task for Critic review.

    Adds a review_request comment + transitions status → review +
    REASSIGNS owner_id to the chosen critic and resets the lease so the
    inbox poller picks the critic up on its next cycle.

    The original submitter is preserved in the review_request comment's
    author_id; on review_decision(changes_requested) the original
    submitter is restored as owner_id by the review_decision tool.
    """
    from . import comments
    if critic_id is None:
        critic_id = _DEFAULT_CRITIC_BY_TARGET.get(target, "critic-prose")

    # [SUBMIT_IDEMPOTENT_2026_06_01] If the task is ALREADY in `review`,
    # this is a no-op — NOT an error.  The IC submits on one turn, but its
    # agent loop can keep running and call submit_for_review AGAIN on a
    # later turn (observed live: researcher re-submitted turns 14-16 after a
    # successful turn-13 submit).  Without this guard update_status raises
    # IllegalTransition(review→review), the IC sees a tool FAILURE, and
    # burns its remaining turns "retrying" a submit it already completed.
    # Return cleanly so the dispatch can end this turn cycle.
    _cur = get_task(task_id, tenant_id, user_id)
    if _cur and _cur.get("status") == "review":
        return

    # Status: in_progress → review (state machine enforced).
    update_status(
        task_id=task_id,
        tenant_id=tenant_id,
        user_id=user_id,
        new_status="review",
        actor_agent_id=submitter_agent_id,
    )

    # Reassign owner_id to the critic + reset lease (otherwise the IC's
    # expired lease would still match the inbox poller's `lease_until <
    # now` predicate, but the dispatched persona would be the critic's
    # — slightly racy on the wrong row).  Direct table update, not a
    # status transition, so it's privileged.
    tbl = LedgerStore.get().table("agent_tasks")
    where = f"{ns_filter(tenant_id, user_id)} AND id = '{task_id}'"
    tbl.update(where=where, values={
        "owner_id":    critic_id,
        "lease_owner": "",
        "lease_until": 0.0,
        "updated_ts":  _now(),
    })

    comments.add_comment(
        task_id=task_id,
        tenant_id=tenant_id,
        user_id=user_id,
        author_id=submitter_agent_id,
        kind="review_request",
        body=f"target={target} reassigned_to={critic_id} original_owner={submitter_agent_id}",
    )


def inbox(
    *,
    tenant_id: str,
    user_id: str,
    owner_id: str,
    statuses: list[str] | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Per-agent inbox query — the §3.10.4 derived view."""
    from .store import task_inbox_filter
    where = task_inbox_filter(tenant_id, user_id, owner_id, statuses)
    tbl = LedgerStore.get().table("agent_tasks")
    arr = tbl.search().where(where).limit(limit).to_arrow()
    return [_row_to_dict(arr, i) for i in range(arr.num_rows)]


def cancel_cascade(
    *,
    task_id: str,
    tenant_id: str,
    user_id: str,
    actor_agent_id: str,
    reason: str = "parent_cancelled",
) -> int:
    """Cancel a task + cascade to all descendants per spec §11.2.4.

    Returns the number of tasks transitioned.
    """
    from . import comments

    n_cancelled = 0
    # BFS: find all descendants whose lineage_chain contains task_id.
    tbl = LedgerStore.get().table("agent_tasks")
    where = (
        f"{ns_filter(tenant_id, user_id)} "
        f"AND status NOT IN ('done', 'cancelled')"
    )
    arr = tbl.search().where(where).to_arrow()
    for i in range(arr.num_rows):
        row = _row_to_dict(arr, i)
        if row["id"] != task_id and task_id not in (row.get("lineage_chain") or []):
            continue
        current = row["status"]
        if current in ("in_progress", "accepted"):
            # Flag for graceful stop — comment then cancel.
            comments.add_comment(
                task_id=row["id"],
                tenant_id=tenant_id,
                user_id=user_id,
                author_id=actor_agent_id,
                kind="status_change",
                body=f"cascade-cancel: please checkpoint and stop ({reason})",
            )
        # Direct DB update (skip state-machine check; cascade is privileged).
        tbl.update(
            where=f"id = '{row['id']}'",
            values={"status": "cancelled", "updated_ts": _now()},
        )
        n_cancelled += 1

    return n_cancelled


# ── Internal helpers ─────────────────────────────────────────────────────


def _row_to_dict(arr, idx: int) -> dict[str, Any]:
    return {col: arr.column(col)[idx].as_py() for col in arr.column_names}


# ── Phase 7.5 — depends_on cross-lineage helpers ────────────────────────


def _missing_dep_ids(
    *, tenant_id: str, user_id: str, dep_ids: list[str],
) -> set[str]:
    """Return the subset of `dep_ids` that don't exist for this
    (tenant, user).  Empty input → empty set.  Used by `create_task`
    to reject unknown deps at creation time."""
    if not dep_ids:
        return set()
    tbl = LedgerStore.get().table("agent_tasks")
    found: set[str] = set()
    for tid in dep_ids:
        try:
            arr = (
                tbl.search()
                .where(
                    f"{ns_filter(tenant_id, user_id)} AND id = '{tid}'"
                )
                .limit(1)
                .to_arrow()
            )
            if arr.num_rows > 0:
                found.add(tid)
        except Exception:
            # If the lookup itself errors we conservatively treat
            # the dep as missing — Director gets a clear failure.
            pass
    return set(dep_ids) - found


def deps_satisfied(
    *, task: dict[str, Any], tenant_id: str, user_id: str,
) -> tuple[bool, list[str]]:
    """Phase 7.5 — return (ready, pending_dep_ids).

    `ready=True` iff every id in `task['depends_on']` exists AND has
    `status='done'`.  Tasks with no deps are always ready.  Used by
    `workers.inbox_poller` to skip tasks whose preconditions haven't
    fired.

    `pending_dep_ids` enumerates the deps that aren't yet done — empty
    when ready, sized N when N deps still pending.  Lets the sidebar
    surface "blocked on X, Y" without an extra query.
    """
    deps = task.get("depends_on") or []
    if not deps:
        return True, []
    tbl = LedgerStore.get().table("agent_tasks")
    pending: list[str] = []
    for dep_id in deps:
        try:
            arr = (
                tbl.search()
                .where(
                    f"{ns_filter(tenant_id, user_id)} AND id = '{dep_id}'"
                )
                .select(["status"])
                .limit(1)
                .to_arrow()
            )
            if arr.num_rows == 0:
                # Missing dep → treat as un-satisfied (defensive).
                pending.append(dep_id)
                continue
            status = arr.column("status")[0].as_py() or ""
            if status != "done":
                pending.append(dep_id)
        except Exception:
            pending.append(dep_id)
    return (len(pending) == 0, pending)


def deps_dead(
    *, task: dict[str, Any], tenant_id: str, user_id: str,
) -> list[str]:
    """[DEPS_DEAD_CASCADE_2026_05_29] Return the subset of `depends_on`
    that are CANCELLED — i.e. deps that can never reach `done`, so the
    dependent can never become ready.

    A synthesis/rollup subtask `depends_on` its analysis siblings; if an
    analysis is cancelled (e.g. force-cancelled on dispatch timeout, or
    rejected), the dependent would otherwise wait FOREVER (deps_satisfied
    needs `done`, and `cancelled != done`), stranding the dependent AND
    its coordinator parent OPEN. The poller uses this to cascade-cancel
    the dependent so the subtree drains instead of hanging.
    """
    deps = task.get("depends_on") or []
    if not deps:
        return []
    tbl = LedgerStore.get().table("agent_tasks")
    dead: list[str] = []
    for dep_id in deps:
        try:
            arr = (
                tbl.search()
                .where(f"{ns_filter(tenant_id, user_id)} AND id = '{dep_id}'")
                .select(["status"])
                .limit(1)
                .to_arrow()
            )
            if arr.num_rows and (arr.column("status")[0].as_py() or "") == "cancelled":
                dead.append(dep_id)
        except Exception:
            pass
    return dead


def increment_cost_attributed(
    *,
    task_id: str,
    tenant_id: str,
    user_id: str,
    delta_usd: float,
) -> bool:
    """Phase 7.5 — atomically add `delta_usd` to the Task's
    `cost_attributed_usd`.  Used by the LLM turn-end hook in
    `runtime.run_agent_query` so team budget rollups see the true
    spend without a separate sweeper.

    Idempotency: we read-then-write.  In the rare contention case
    (two workers writing to the same task at the same moment) one
    update wins per Lance's last-write semantics.  That's acceptable
    for cost attribution — budgets are soft envelopes, not financial
    ledgers; the next read recomputes the rollup from authoritative
    per-row data.

    Returns True on success, False on failure (logs the error rather
    than raising — billing must never block the agent).
    """
    if delta_usd <= 0:
        return False
    try:
        tbl = LedgerStore.get().table("agent_tasks")
        arr = (
            tbl.search()
            .where(
                f"{ns_filter(tenant_id, user_id)} AND id = '{task_id}'"
            )
            .select(["cost_attributed_usd"])
            .limit(1)
            .to_arrow()
        )
        if arr.num_rows == 0:
            return False
        current_cost = float(arr.column("cost_attributed_usd")[0].as_py() or 0.0)
        new_cost = current_cost + float(delta_usd)
        tbl.update(
            where=(
                f"{ns_filter(tenant_id, user_id)} AND id = '{task_id}'"
            ),
            values={
                "cost_attributed_usd": new_cost,
                "updated_ts":          _now(),
            },
        )
        return True
    except Exception as e:
        logger.warning(
            f"[tasks] increment_cost_attributed({task_id}) failed: {e}"
        )
        return False


def verify_depends_on_acyclic(
    *, task_id: str, depends_on: list[str], tenant_id: str, user_id: str,
) -> tuple[bool, list[str]]:
    """Director-tool helper — walk the dep graph from the proposed
    new task's deps and confirm `task_id` doesn't reappear.  Returns
    (ok, cycle_path) where cycle_path is the chain that closed the
    loop (for diagnostic) and `ok=True` means the DAG is acyclic.

    This is O(N) over the transitive closure of the deps.  We bound
    visited size at 1000 nodes to prevent runaway scans on adversarial
    inputs.
    """
    if not depends_on:
        return True, []
    if task_id in depends_on:
        return False, [task_id, task_id]

    tbl = LedgerStore.get().table("agent_tasks")
    visited: set[str] = set()
    stack: list[tuple[str, list[str]]] = [(d, [d]) for d in depends_on]
    while stack:
        current, path = stack.pop()
        if current == task_id:
            return False, path + [task_id]
        if current in visited:
            continue
        visited.add(current)
        if len(visited) > 1000:
            # Guardrail — Director's DAG should never need this many.
            return False, ["scan_limit_exceeded"]
        try:
            arr = (
                tbl.search()
                .where(
                    f"{ns_filter(tenant_id, user_id)} AND id = '{current}'"
                )
                .select(["depends_on"])
                .limit(1)
                .to_arrow()
            )
            if arr.num_rows == 0:
                continue
            parent_deps = arr.column("depends_on")[0].as_py() or []
            for pd in parent_deps:
                if pd in visited:
                    continue
                stack.append((pd, path + [pd]))
        except Exception:
            continue
    return True, []


__all__ = [
    "create_task",
    "get_task",
    "update_status",
    "attach_output",
    "submit_for_review",
    "inbox",
    "cancel_cascade",
    # Phase 7.5 depends_on
    "deps_satisfied",
    "verify_depends_on_acyclic",
    # Phase 7.5 cost write-back
    "increment_cost_attributed",
]
