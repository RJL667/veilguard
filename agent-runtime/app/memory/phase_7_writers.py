"""Phase 7 split-writers — the sanctioned cross-ref sites.

Implements the M2/M3 cross-ref pattern: ledger row first → get the PK
→ post TCMM observation with `extras.entity_id=<pk>` → update ledger
row with `tcmm_obs_id`.  Both directions reference each other; neither
mirrors the other's fields.  Per the §2 design principle (TCMM is the
knowledge-graph substrate; the ledger is the state machine; observations
may reference ledger PKs but never mirror ledger fields), these are the
ONLY functions allowed to write to both substrates in one call.

The Phase 6.8 memory-write-path lint allow-lists this module
(`app/memory/phase_7_writers.py`) as a sanctioned dual-write site.
"""

from __future__ import annotations

import json as _json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger("agent-runtime.memory.phase_7_writers")


# # boundary: split-writer
# Phase 7 — Outcome narrative split-writer.
async def record_outcome_with_narrative(
    *,
    proposal_id: str,
    resulting_task_id: Optional[str],
    tenant_id: str,
    user_id: str,
    # Ledger-side (numeric):
    task_status: str,
    task_cost_usd: float,
    value_realized: float,
    regret_score: float,
    objective_deltas: Optional[dict] = None,
    # TCMM-side (narrative):
    regret_text: Optional[str] = None,
    what_went_wrong: Optional[str] = None,
    lessons_learned: Optional[str] = None,
    critic_id: str = "critic-prose",
) -> tuple[str, Optional[str]]:
    """Phase 7.1 M2 split-writer.

    Writes ledger row first (numeric metrics, exact), then posts TCMM
    observation with the narrative + `extras.outcome_id` back-ref.
    Updates ledger row with `tcmm_obs_id`.

    Returns (outcome_id, tcmm_obs_id-or-None).  TCMM observation may
    fail (offline); ledger row still commits.  This is intentional —
    numeric outcome accounting must not be blocked on TCMM availability.
    """
    from ..ledger.store import LedgerStore, ns_filter  # sanctioned
    import uuid
    outcome_id = f"out-{uuid.uuid4().hex[:12]}"
    now = time.time()
    deltas_str = _json.dumps(objective_deltas or {})
    tbl = LedgerStore.get().table("proposal_outcomes")
    tbl.add([{
        "id":                  outcome_id,
        "kind":                "outcome",
        "status":              task_status,
        "parent_id":           proposal_id,
        "lineage_chain":       [],
        "tenant_id":           tenant_id,
        "user_id":             user_id,
        "created_by_agent_id": critic_id,
        "created_ts":          now,
        "updated_ts":          now,
        "cost_attributed_usd": task_cost_usd,
        "proposal_id":         proposal_id,
        "resulting_task_id":   resulting_task_id,
        "task_status":         task_status,
        "task_cost_usd":       task_cost_usd,
        "value_realized":      value_realized,
        "regret_score":        regret_score,
        "objective_deltas_json": deltas_str,
        "computed_at_ts":      now,
        "tcmm_obs_id":         None,  # filled in below if TCMM accepts
        "extras_json":         None,
    }])

    # Narrative half → TCMM (best-effort).
    narrative_parts = [p for p in (regret_text, what_went_wrong, lessons_learned) if p]
    if not narrative_parts:
        return (outcome_id, None)
    narrative_body = (
        f"[outcome:{outcome_id}] proposal_id={proposal_id} status={task_status}\n"
        f"regret_score={regret_score:.3f} cost_usd={task_cost_usd:.2f} "
        f"value_realized={value_realized:.2f}\n\n"
        + "\n\n".join(narrative_parts)
    )
    try:
        from ..middleware.tcmm import observe_agent_output  # sanctioned
        persisted = await observe_agent_output(
            conversation_id=f"outcomes/{tenant_id}",
            user_id=user_id,
            agent_id=critic_id,
            text=narrative_body,
            source="outcome_narrative",  # AC-P7.6 allow-list
            channel="team_events",  # [CHANNEL_2026_05_29] spec §6.4
        )
        if not persisted:
            return (outcome_id, None)
    except Exception as e:
        logger.warning(f"[phase_7_writers] outcome narrative TCMM write failed: {e}")
        return (outcome_id, None)

    # TCMM doesn't return an ID today; we synthesize a content-hash
    # ref so the cross-ref column has a stable pointer.
    import hashlib
    tcmm_obs_id = f"tcmm-{hashlib.sha256(narrative_body.encode()).hexdigest()[:24]}"
    # Update ledger row with the back-ref.
    try:
        where = (
            f"{ns_filter(tenant_id, user_id)} "
            f"AND id = '{outcome_id}'"
        )
        tbl.update(where=where, values={
            "tcmm_obs_id": tcmm_obs_id,
            "updated_ts":  time.time(),
        })
    except Exception as e:
        logger.warning(
            f"[phase_7_writers] couldn't write tcmm_obs_id back-ref for "
            f"outcome {outcome_id}: {e}"
        )
    return (outcome_id, tcmm_obs_id)


# # boundary: split-writer
# Phase 7 — Proposal content split-writer.
async def record_proposal_with_content(
    *,
    tenant_id: str,
    user_id: str,
    signal_type: str,
    signal_node_ids: list[int],
    impact_score: float,
    proposed_assignee: str,
    proposed_brief: str,
    rationale: Optional[str] = None,
    proposed_deliverable_spec: Optional[str] = None,
    objective_alignment: Optional[float] = None,
    constraint_violations: Optional[list[str]] = None,
    constitution_version: Optional[int] = None,
    emergency_lane: bool = False,
) -> tuple[str, Optional[str]]:
    """Phase 7.1 M3 split-writer.

    Ledger keeps operational fields (status, scores, signal metadata).
    TCMM gets the content (brief, rationale).  Cross-ref via
    `tcmm_obs_id ↔ proposal_id`.

    Returns (proposal_id, tcmm_obs_id-or-None).
    """
    from ..ledger import proposals as _proposals  # sanctioned
    # Ledger row first.  We pass the brief + rationale through so
    # legacy queries don't break (Phase 7 transitional — operational
    # callers still expect proposed_brief column populated).  Final
    # state may strip these once all readers are migrated.
    proposal_id = _proposals.create_proposal(
        tenant_id=tenant_id,
        user_id=user_id,
        signal_type=signal_type,
        signal_node_ids=signal_node_ids,
        impact_score=impact_score,
        proposed_brief=proposed_brief,
        proposed_assignee=proposed_assignee,
        proposed_deliverable_spec=proposed_deliverable_spec,
        rationale=rationale,
        objective_alignment=objective_alignment,
        constraint_violations=constraint_violations,
        constitution_version=constitution_version,
    )

    # TCMM-side observation.
    content_parts = [f"PROPOSAL_BRIEF: {proposed_brief}"]
    if rationale:
        content_parts.append(f"RATIONALE: {rationale}")
    narrative_body = (
        f"[proposal:{proposal_id}] signal_type={signal_type} "
        f"assignee={proposed_assignee} score={impact_score:.3f}\n\n"
        + "\n\n".join(content_parts)
    )
    try:
        from ..middleware.tcmm import observe_agent_output  # sanctioned
        persisted = await observe_agent_output(
            conversation_id=f"proposals/{tenant_id}",
            user_id=user_id,
            agent_id="dream-cycle",
            text=narrative_body,
            source="proposal",  # AC-P7.6 allow-list
            channel="team_events",  # [CHANNEL_2026_05_29] spec §6.4
        )
        if not persisted:
            return (proposal_id, None)
    except Exception as e:
        logger.warning(f"[phase_7_writers] proposal TCMM write failed: {e}")
        return (proposal_id, None)

    import hashlib
    tcmm_obs_id = f"tcmm-{hashlib.sha256(narrative_body.encode()).hexdigest()[:24]}"
    try:
        from ..ledger.store import LedgerStore, ns_filter
        tbl = LedgerStore.get().table("task_proposals")
        where = (
            f"{ns_filter(tenant_id, user_id)} "
            f"AND id = '{proposal_id}'"
        )
        tbl.update(where=where, values={
            "tcmm_obs_id": tcmm_obs_id,
            "updated_ts":  time.time(),
        })
    except Exception as e:
        logger.warning(
            f"[phase_7_writers] couldn't write tcmm_obs_id back-ref for "
            f"proposal {proposal_id}: {e}"
        )
    return (proposal_id, tcmm_obs_id)


# # boundary: split-writer
# Phase 7 — Lesson promotion (M1: full migration to TCMM).
async def promote_lesson_to_team_knowledge(
    *,
    tenant_id: str,
    user_id: str,
    team_id: str,
    trigger: str,
    rule: str,
    confidence: float,
    critic_id: str = "critic-prose",
    promoted_from: Optional[str] = None,
    confidence_decay_per_week: float = 0.05,
) -> bool:
    """Phase 7.1 M1 — promote a Critic-validated lesson to TCMM
    team/knowledge/<team_id>.

    Post-M1-cutover (2026-05-28): TCMM is the sole destination.  The
    legacy `org_memory` Lance write has been removed; reads go through
    `app.memory.lessons_reader` which scans the TCMM archive.  Spec
    §7.3 AC-P7.2 (`org_memory` Lance table absent) now passes.

    Returns True iff the TCMM observation persisted.
    """
    try:
        from ..middleware.tcmm import observe_agent_output  # sanctioned
        body = (
            f"[lesson] trigger={trigger}\n"
            f"rule={rule}\n"
            f"confidence={confidence:.3f}"
        )
        ok = await observe_agent_output(
            conversation_id=f"team/knowledge/{team_id}",
            user_id=user_id,
            agent_id=critic_id,
            text=body,
            source="lesson",  # AC-P7.6 allow-list
            channel="team_knowledge",  # [CHANNEL_2026_05_29] spec §6.4
        )
        # [LESSON_SSE_2026_05_28]  Surface lesson promotion to the
        # sidebar Lessons tab.  Without this, the tab falls back to
        # 10s polling — the §3.10.5 spec TODO.  Broadcast AFTER the
        # TCMM write to avoid emitting an event for an observation
        # that didn't actually persist (would mislead the sidebar
        # into re-fetching and finding nothing new).
        #
        # **Note on `id`:** the lesson_id here is a STABLE TRIGGER
        # HASH (sha256("{tenant}:{user}:{trigger}")[:12]).  It does
        # NOT match the `lsn-tcmm-<aid>` format that
        # `lessons_reader.find_lessons_due_for_review` returns
        # (because `observe_agent_output` doesn't surface the TCMM
        # aid back to us at promotion time).  The id is therefore
        # only useful for client-side DEDUP of repeated SSE events
        # for the same trigger; it is NOT a correlation key into the
        # sidebar's row state.  Sidebar should treat lesson_created
        # as "re-fetch /lessons/review_queue" — same pattern as
        # task_created.
        if ok:
            try:
                from ..events import broadcast
                import hashlib as _hashlib
                lesson_id = "lesson-" + _hashlib.sha256(
                    f"{tenant_id}:{user_id}:{trigger}".encode()
                ).hexdigest()[:12]
                broadcast({
                    "type":       "lesson_created",
                    "tenant_id":  tenant_id,
                    "user_id":    user_id,
                    "id":         lesson_id,
                    "team_id":    team_id,
                    "trigger":    trigger[:200],
                    "confidence": confidence,
                    "critic_id":  critic_id,
                })
            except Exception as _bcast_err:
                # Never fail the lesson write because the sidebar
                # broadcast hiccupped — the sidebar's 10s polling
                # fallback will catch up.
                logger.debug(
                    f"[phase_7_writers] lesson_created broadcast failed: "
                    f"{_bcast_err}"
                )
        return ok
    except Exception as e:
        logger.warning(f"[phase_7_writers] lesson TCMM write failed: {e}")
        return False


# # boundary: split-writer
# Phase 7 — task_comments split-by-kind (M4).
async def record_discussion_comment(
    *,
    task_id: str,
    tenant_id: str,
    user_id: str,
    author_id: str,
    body: str,
    note: bool = False,
) -> bool:
    """Phase 7.1 M4 split-writer for `task_comments[kind=discussion|note]`.

    State-machine kinds (`status_change`, `review_decision`,
    `blocker_raised`, `blocker_cleared`, `review_request`) stay in the
    SHA-chained `task_comments` Lance table and are written via
    `log_decision` / `add_comment`.  This writer is for the
    *non-state-machine* kinds — discussion and operator notes — which
    are derivative narrative and belong in TCMM `agent/<aid>/observations/`.

    No back-ref column is needed (one-directional: the SHA-chain in
    ledger is the immutable source of truth for state; TCMM is the
    derivative narrative store).

    Returns True on persist success; False if TCMM unreachable (the
    caller may decide to fall back to a legacy `add_comment(kind="comment")`
    transitional write while M4 cutover is in progress, but the
    canonical destination going forward is TCMM).
    """
    if not (task_id and tenant_id and user_id and author_id):
        raise ValueError("record_discussion_comment requires task/tenant/user/author ids")
    if not body or not body.strip():
        return False
    source_kind = "agent_observation" if note else "discussion"
    narrative_body = (
        f"[task:{task_id}] author={author_id}\n\n{body.strip()}"
    )
    try:
        from ..middleware.tcmm import observe_agent_output  # sanctioned
        return await observe_agent_output(
            conversation_id=f"agent/{author_id}/observations",
            user_id=user_id,
            agent_id=author_id,
            text=narrative_body,
            source=source_kind,
            channel="agent_private",  # [CHANNEL_2026_05_29] spec §6.4
        )
    except Exception as e:
        logger.warning(
            f"[phase_7_writers] discussion comment TCMM write failed: {e}"
        )
        return False
