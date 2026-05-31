"""Phase 6.8 — Typed memory writer functions.

Implements the §3.11 memory write-path discipline.  Every memory
mutation in agent-runtime flows through one of these functions; the
linter in `tests/memory/test_write_path_lint.py` (AC-36) rejects any
other module that imports Lance handles / TCMM clients / workspace fs
writers directly.

The auto-generated docs (`agent-runtime/docs/MEMORY_WRITE_PATHS.md`,
AC-37) are produced by `generate_docs()` at the bottom of this file,
called from the test runner so the doc cannot drift from the writer
signatures.

Phase 7 will migrate `update_org_memory` to write TCMM `team/knowledge/`
instead of the Lance org_memory table.  For now both paths exist; the
boundary is enforced by the destination map.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("agent-runtime.memory.writers")


class WriterError(RuntimeError):
    """Raised when a writer cannot complete (TCMM down, Lance error,
    invalid args).  Distinct from the wrapped tool errors so callers
    can distinguish a memory-layer failure from a logic-layer failure.
    """
    pass


@dataclass(frozen=True)
class _WriterDest:
    """Documentation entry for the auto-generated write-path map.

    Captures (writer × destination × trigger × transformation_rule)
    per AC-37.  Auto-rebuilt from this list whenever the module loads.
    """
    writer:       str
    destination:  str
    trigger:      str
    transformation: str
    # [CHANNEL_2026_05_29] spec §6.4 / AC-37 — the TCMM channel the write
    # lands in ("n/a" for ledger-only writers). Default keeps this frozen
    # dataclass non-breaking for the existing kw-constructed entries.
    channel:      str = "n/a"


WRITER_DESTINATIONS: list[_WriterDest] = [
    _WriterDest(
        writer="record_episode",
        destination="TCMM agent/<agent_id>/observations/<user_id>/",
        trigger="every agent observation during a turn",
        transformation="text → tcmm.observe(source_kind=AGENT_*, extracted_by=agent:<aid>)",
        channel="agent_private",
    ),
    _WriterDest(
        writer="record_heartbeat",
        destination="ledger agent_task_heartbeats (Phase 6.3)",
        trigger="every IC turn boundary; inbox-poller sweep uses these",
        transformation="(task_id, worker_id, now) → heartbeat row",
    ),
    _WriterDest(
        writer="promote_to_semantic",
        destination="TCMM team/knowledge/<team_id>/ (Critic-gated)",
        trigger="Critic accepts a claim destined for team_knowledge",
        transformation="claim → tcmm.observe with typed_claim shape + provenance",
        channel="team_knowledge",
    ),
    _WriterDest(
        writer="log_decision",
        destination="ledger agent_tasks (status) + task_comments (SHA-chained)",
        trigger="state transition or review_decision",
        transformation="atomic write: tasks.update_status + comments.add_comment",
    ),
    _WriterDest(
        writer="update_org_memory",
        destination="DEPRECATED post-M1-cutover 2026-05-28; use promote_lesson_to_team_knowledge",
        trigger="back-compat shim only; new callers must use the Phase 7 split-writer",
        transformation="(no-op surface; org_memory Lance table removed)",
    ),
    _WriterDest(
        writer="enqueue_dream_input",
        destination="TCMM dream_archive (via TCMM /ingest_turn)",
        trigger="event crosses TCMM importance threshold",
        transformation="serialized event → tcmm.observe(source_kind=EVENT)",
        channel="team_events",
    ),
    _WriterDest(
        writer="record_approval",
        destination="ledger client_tool_approvals (append-only audit)",
        trigger="approval-gate decision (allow/deny/timeout)",
        transformation="approval row with TOCTOU token + immutable timestamps",
    ),
    _WriterDest(
        writer="attach_artifact",
        destination="ledger agent_tasks.outputs[] + workspace fs (task-owned dir)",
        trigger="Builder produces a deliverable file",
        transformation="path validation + outputs[].append + fs write",
    ),
    # Phase 7 split-writers — sanctioned cross-ref sites only.
    _WriterDest(
        writer="record_outcome_with_narrative",
        destination="ledger proposal_outcomes (numeric) + TCMM outcome_narrative observation (cross-ref via tcmm_obs_id ↔ outcome_id)",
        trigger="Phase 7 M2: outcome with regret/lesson narrative",
        transformation="ledger row first (numeric exact) → TCMM observe → write tcmm_obs_id back",
        channel="team_events",
    ),
    _WriterDest(
        writer="record_proposal_with_content",
        destination="ledger task_proposals (status index) + TCMM proposal observation (cross-ref via tcmm_obs_id ↔ proposal_id)",
        trigger="Phase 7 M3: proposal creation with content + rationale",
        transformation="ledger create_proposal → TCMM observe → write tcmm_obs_id back",
        channel="team_events",
    ),
    _WriterDest(
        writer="promote_lesson_to_team_knowledge",
        destination="TCMM team/knowledge/<team_id>/ (canonical; post-M1-cutover 2026-05-28)",
        trigger="Phase 7 M1: Critic-validated lesson promotion",
        transformation="single-destination TCMM observation; legacy org_memory dual-write removed",
        channel="team_knowledge",
    ),
    _WriterDest(
        writer="record_discussion_comment",
        destination="TCMM agent/<author_id>/observations/ (source_kind=discussion|agent_observation)",
        trigger="Phase 7 M4: non-state-machine task comment",
        transformation="kind=comment/note → TCMM observation; state-machine kinds still go through log_decision",
        channel="agent_private",
    ),
]


# ── Writers ────────────────────────────────────────────────────────────


def record_heartbeat(
    *,
    task_id: str,
    worker_id: str,
    tenant_id: str,
    user_id: str,
    lease_ttl_s: float | None = None,
) -> str:
    """Phase 6.3 — write one heartbeat row.

    Workers call this at every turn boundary (per N turns, configurable).
    The inbox-poller sweep walks `agent_task_heartbeats`, picks the
    most-recent row per task_id, and auto-reclaims tasks whose
    `now - last_beat_at > lease_ttl_s` with an audit comment.

    [HEARTBEAT_TTL_TRACKS_LEASE_2026_05_29]  The staleness threshold MUST
    track the dispatch lease (VEILGUARD_LEASE_DURATION_S).  Previously it
    was a hardcoded 300s while the lease was env-bumped to 1200s on slow
    envs — so a dispatch whose cold first turn (model load + cold
    redaction + a contended LLM call) ran >300s between turn-events got
    reclaimed mid-work, cancelling live IC tasks (caught live 2026-05-29:
    C1 analyses force-cancelled at ~300s despite a 1200s lease).  Default
    now reads the same env so the two stay consistent.

    Returns the heartbeat row id.
    """
    if not (task_id and worker_id and tenant_id and user_id):
        raise WriterError(
            "record_heartbeat requires task_id/worker_id/tenant_id/user_id"
        )
    if lease_ttl_s is None:
        import os
        lease_ttl_s = float(
            os.environ.get("VEILGUARD_LEASE_DURATION_S", "300")
        )
    from ..ledger.store import LedgerStore  # sanctioned
    import uuid
    hb_id = f"hb-{uuid.uuid4().hex[:12]}"
    now = time.time()
    tbl = LedgerStore.get().table("agent_task_heartbeats")
    tbl.add([{
        "id":           hb_id,
        "task_id":      task_id,
        "worker_id":    worker_id,
        "tenant_id":    tenant_id,
        "user_id":      user_id,
        "last_beat_at": now,
        "lease_ttl_s":  lease_ttl_s,
        "created_ts":   now,
        "extras_json":  None,
    }])
    return hb_id


async def record_episode(
    *,
    tenant_id: str,
    user_id: str,
    agent_id: str,
    conversation_id: str,
    text: str,
    source: str = "agent",
) -> bool:
    """Append a raw observation to TCMM.

    Thin wrapper around `middleware/tcmm.py:observe_agent_output` —
    centralised so the linter has a single sanctioned site for TCMM
    observation writes.  Returns True if persisted, False if TCMM was
    unreachable (callers should surface the failure to the LLM per the
    [LOOP_FIX_2026_05_27] convention).
    """
    if not (tenant_id and user_id and agent_id):
        raise WriterError("record_episode requires tenant_id/user_id/agent_id")
    if not text or not text.strip():
        return False  # silently skip empty observations
    from ..middleware.tcmm import observe_agent_output  # sanctioned import
    return await observe_agent_output(
        conversation_id=conversation_id,
        user_id=user_id,
        agent_id=agent_id,
        text=text,
        source=source,
        channel="agent_private",  # [CHANNEL_2026_05_29] spec §6.4
    )


async def promote_to_semantic(
    *,
    tenant_id: str,
    user_id: str,
    team_id: str,
    critic_id: str,
    claim_text: str,
    typed_claims: Optional[list[dict[str, Any]]] = None,
    source_task_id: Optional[str] = None,
) -> bool:
    """Critic-gated promotion of a claim to team_knowledge.

    Phase 6.8: writes to the same TCMM observation channel as
    record_episode but with `source_kind=TEAM_KNOWLEDGE` and a
    `promoted_by_critic_id` provenance tag.  Phase 7 M1 will migrate
    this to TCMM `team/knowledge/<team_id>/` lane=`org_critic` with
    hard-include floor.

    Returns True if persisted; False if TCMM was unreachable.
    """
    if not (tenant_id and user_id and team_id and critic_id):
        raise WriterError("promote_to_semantic requires tenant/user/team/critic ids")
    if not claim_text.strip():
        raise WriterError("claim_text cannot be empty")
    from ..middleware.tcmm import observe_agent_output  # sanctioned import
    # Format the claim payload with provenance.  We do NOT inline the
    # typed_claims into the text body — they go in a TCMM extras blob
    # the next phase of work will plumb.  For now we serialize them
    # alongside the body so recall + dream cycle can both see them.
    import json as _json
    body = claim_text
    if typed_claims:
        body = (
            f"{claim_text}\n\n"
            f"[typed_claims={_json.dumps(typed_claims, ensure_ascii=False)}]"
        )
    return await observe_agent_output(
        conversation_id=f"team/knowledge/{team_id}",
        user_id=user_id,
        agent_id=critic_id,
        text=body,
        source="critic_promote",
        channel="team_knowledge",  # [CHANNEL_2026_05_29] spec §6.4
    )


def log_decision(
    *,
    task_id: str,
    tenant_id: str,
    user_id: str,
    actor_agent_id: str,
    new_status: Optional[str] = None,
    comment_kind: Optional[str] = None,
    comment_body: Optional[str] = None,
) -> str | None:
    """State-machine write: status transition AND/OR comment.

    The single sanctioned site that touches both agent_tasks and
    task_comments.  Either or both of new_status / comment_kind may be
    provided.  Returns the comment_id if a comment was written.

    Raises IllegalTransition (from tasks.update_status) if the
    transition violates the state machine OR the Phase 6.0.1 hard-gate.
    """
    from ..ledger import tasks as _t, comments as _c  # sanctioned imports
    comment_id: str | None = None
    if comment_kind:
        if not comment_body:
            raise WriterError("comment_body required when comment_kind set")
        comment_id = _c.add_comment(
            task_id=task_id,
            tenant_id=tenant_id,
            user_id=user_id,
            author_id=actor_agent_id,
            kind=comment_kind,
            body=comment_body,
        )
    if new_status:
        # tasks.update_status enforces the hard-gate; let IllegalTransition propagate.
        _t.update_status(
            task_id=task_id,
            tenant_id=tenant_id,
            user_id=user_id,
            new_status=new_status,
            actor_agent_id=actor_agent_id,
        )
    return comment_id


def update_org_memory(
    *,
    tenant_id: str,
    user_id: str,
    trigger: str,
    rule: str,
    confidence: float,
    promoted_from: Optional[str] = None,
    confidence_decay_per_week: float = 0.05,
    review_after_days: int = 30,
) -> str:
    """**DEPRECATED post-M1-cutover (2026-05-28).** Back-compat shim.

    Callers should migrate to `promote_lesson_to_team_knowledge`
    (Phase 7 split-writer) which is the canonical lesson-write path.
    The legacy `org_memory` Lance table has been dropped; this shim
    now routes through the Phase 7 writer via asyncio.run so existing
    sync callers don't break.

    Returns a synthetic lesson_id since TCMM doesn't echo back an aid.
    """
    if not (tenant_id and user_id and trigger and rule):
        raise WriterError("update_org_memory requires tenant/user/trigger/rule")
    if not 0.0 <= confidence <= 1.0:
        raise WriterError(f"confidence must be in [0,1], got {confidence}")
    import asyncio
    from .phase_7_writers import promote_lesson_to_team_knowledge
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Sync caller inside an async context — can't asyncio.run.
            # Tell them to migrate (or use ensure_future).
            raise WriterError(
                "update_org_memory called from async context — migrate "
                "the caller to `await promote_lesson_to_team_knowledge(...)`"
            )
    except RuntimeError:
        # No running loop — safe to use asyncio.run below.
        pass
    ok = asyncio.run(promote_lesson_to_team_knowledge(
        tenant_id=tenant_id, user_id=user_id, team_id=tenant_id,
        trigger=trigger, rule=rule, confidence=confidence,
        promoted_from=promoted_from or "manual-update_org_memory",
    ))
    if not ok:
        raise WriterError("update_org_memory: TCMM observation failed")
    return f"lsn-{trigger[:24]}-{int(time.time())}"


async def enqueue_dream_input(
    *,
    tenant_id: str,
    user_id: str,
    agent_id: str,
    event_text: str,
    importance_score: float = 0.5,
) -> bool:
    """High-importance event → TCMM dream cycle input.

    Currently just an observation with `source_kind=EVENT`; dream cycle
    picks it up on the next pass based on importance.  Future work
    might bypass `observe` and write directly to a "hot" dream input
    queue (Phase 7 tiered micro-dreams, Q22).
    """
    if not (tenant_id and user_id and agent_id):
        raise WriterError("enqueue_dream_input requires tenant/user/agent ids")
    if not event_text.strip():
        return False
    if not 0.0 <= importance_score <= 1.0:
        raise WriterError(f"importance_score must be in [0,1], got {importance_score}")
    from ..middleware.tcmm import observe_agent_output
    body = f"[importance={importance_score:.2f}] {event_text}"
    return await observe_agent_output(
        conversation_id=f"events/{tenant_id}",
        user_id=user_id,
        agent_id=agent_id,
        text=body,
        source="event",
        channel="team_events",  # [CHANNEL_2026_05_29] spec §6.4
    )


def record_approval(
    *,
    tenant_id: str,
    user_id: str,
    tool_name: str,
    tool_args_hash: str,
    decision: str,
    approval_token: str,
    expires_at: Optional[float] = None,
    daemon_version: Optional[str] = None,
    granted_scope: Optional[list[str]] = None,
) -> str:
    """Append-only audit row for an approval-gate decision.

    NEVER mutated post-write — supports SOC2 / GDPR audit replay.  The
    `approval_token` is the TOCTOU defense: subsequent tool dispatches
    must echo it, and the gate re-verifies the token hasn't been
    revoked.  Token is opaque to agents.
    """
    if decision not in ("allow", "deny", "timeout", "auto_foreground", "revoked"):
        raise WriterError(f"unknown decision {decision!r}")
    if not approval_token:
        raise WriterError("approval_token required (TOCTOU defense)")
    from ..ledger.store import LedgerStore  # sanctioned import
    import uuid
    approval_id = f"appr-{uuid.uuid4().hex[:12]}"
    tbl = LedgerStore.get().table("client_tool_approvals")
    tbl.add([{
        "id":               approval_id,
        "kind":             "approval",
        "status":           decision,
        "parent_id":        None,
        "lineage_chain":    [],
        "tenant_id":        tenant_id,
        "user_id":          user_id,
        "created_by_agent_id": None,
        "created_ts":       time.time(),
        "updated_ts":       time.time(),
        "cost_attributed_usd": 0.0,
        "tool_name":        tool_name,
        "tool_args_hash":   tool_args_hash,
        "approval_token":   approval_token,
        "expires_at":       expires_at,
        "daemon_version":   daemon_version,
        "granted_scope":    granted_scope or [],
    }])
    return approval_id


def attach_artifact(
    *,
    task_id: str,
    tenant_id: str,
    user_id: str,
    output_path: str,
    actor_agent_id: str,
) -> None:
    """Builder output: append to agent_tasks.outputs[].

    Does NOT write the file content itself — that's the Builder's
    responsibility via the approval gate.  This is purely the ledger
    side: record that path `output_path` belongs to this task's
    deliverable set.

    Workspace isolation (Phase 7+) will additionally verify the path
    is under the task-owned dir; for now we just trust the caller.
    """
    if not output_path:
        raise WriterError("output_path required")
    from ..ledger import tasks as _t  # sanctioned import
    task = _t.get_task(task_id, tenant_id, user_id)
    if task is None:
        raise WriterError(f"task not found: {task_id}")
    outputs = list(task.get("outputs") or [])
    if output_path in outputs:
        return  # idempotent — already attached
    outputs.append(output_path)
    from ..ledger.store import LedgerStore, ns_filter
    tbl = LedgerStore.get().table("agent_tasks")
    where = f"{ns_filter(tenant_id, user_id)} AND id = '{task_id}'"
    tbl.update(where=where, values={
        "outputs":     outputs,
        "updated_ts":  time.time(),
    })


# ── Auto-generated docs (AC-37) ────────────────────────────────────────


def generate_docs() -> str:
    """Build the MEMORY_WRITE_PATHS.md content from WRITER_DESTINATIONS.

    Called from the test suite (test_write_path_lint.py) to keep the
    doc in sync with the writer signatures.  AC-37 asserts the doc
    exists + has the auto-generated header.
    """
    lines = [
        "# Memory write-path discipline — auto-generated map",
        "",
        "> **DO NOT EDIT BY HAND.** This file is regenerated by",
        "> `app/memory/writers.py:generate_docs()` on every test run.",
        "> See `MULTI_AGENT_PLATFORM.md` §3.11 for the discipline rule.",
        "",
        "| writer | channel | destination | trigger | transformation |",
        "|---|---|---|---|---|",
    ]
    for d in WRITER_DESTINATIONS:
        lines.append(
            f"| `{d.writer}` | {d.channel} | {d.destination} | {d.trigger} | {d.transformation} |"
        )
    lines.append("")
    return "\n".join(lines)
