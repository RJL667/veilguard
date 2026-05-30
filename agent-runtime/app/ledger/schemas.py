"""PyArrow schemas for the decision-ledger Lance tables.

Per spec §3.3, §3.7.6, §3.8, §3.9.2, §3.10 — every entity (Task,
proposal, lesson, outcome, approval) carries the shared lifecycle
skeleton plus its kind-specific columns.

Tables (all live in `/tcmm-data/veilguard/tcmm.db/` alongside TCMM's
archive/embeddings/sparse_archive and pii-proxy's pii_audit):

  agent_tasks                — work-in-progress entities (Tasks)
  task_comments              — append-only chain with prev_hash
  task_proposals             — candidate Tasks from dream-as-scheduler
  proposal_outcomes          — completion + regret tracking
  org_memory                 — institutional lessons with expiry
  client_tool_approvals      — audit log for approval gate decisions
  client_tool_bypass         — user-defined "always allow" rules

Shared columns on every table:
  - tenant_id (utf8, NOT NULL)
  - user_id (utf8, NOT NULL)
  - lineage_chain (list<utf8>)
  - created_ts (float64)
  - updated_ts (float64)

These enable the §3.10 `work_items_v` UNION view: every table has the
same skeleton columns, so cross-kind queries work via UNION ALL.
"""

import pyarrow as pa


# ── Shared lifecycle status enum (string column; not enforced by pa) ────
# Per spec §3.10.1:
#   proposed | accepted | in_progress | blocked | review |
#   done | cancelled | retired | institutionalized | expired

# ── Shared skeleton (extracted as a list of fields) ──────────────────────
# Used in every kind-specific schema to keep the UNION view shape stable.

def _shared_skeleton_fields() -> list[pa.Field]:
    return [
        pa.field("id", pa.string(), nullable=False),
        pa.field("kind", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("parent_id", pa.string(), nullable=True),
        pa.field("lineage_chain", pa.list_(pa.string()), nullable=True),
        pa.field("tenant_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("created_by_agent_id", pa.string(), nullable=True),
        pa.field("created_ts", pa.float64(), nullable=False),
        pa.field("updated_ts", pa.float64(), nullable=False),
        pa.field("cost_attributed_usd", pa.float64(), nullable=True),
    ]


# ── agent_tasks ──────────────────────────────────────────────────────────


# ── Phase 6.0 — Acceptance-criteria struct ──────────────────────────────
#
# Per spec §6.0, every task carries a list of AcceptanceCriterion structs.
# Critics walk this list one-by-one; a task transitions to `state='done'`
# only if all required ACs report `status='pass'`.  Stored as a strongly
# typed Lance column (NOT json) so the post-deploy invariant scan
# (AC-12 / AC-P7.3) can filter on it directly.
#
# `check_args` carries the executor's parameters — kind-specific schema
# nested inside; encoded as JSON string for forward-compatibility.  See
# `agent-runtime/app/acceptance/executors.py` for the executor contract.


def _acceptance_criterion_struct() -> pa.StructType:
    return pa.struct([
        pa.field("id",         pa.string(), nullable=False),  # "AC-1", monotonic per task
        pa.field("statement",  pa.string(), nullable=False),
        pa.field("check_kind", pa.string(), nullable=False),  # see executors.py CHECK_KINDS
        pa.field("check_args", pa.string(), nullable=False),  # JSON-encoded args dict
        pa.field("required",   pa.bool_(),  nullable=False),
        pa.field("rationale",  pa.string(), nullable=True),
    ])


def agent_tasks_schema() -> pa.Schema:
    """The Task entity — work items assigned to agents.

    Comments live in `task_comments` (separate table, append-only chain).
    Outputs are file paths in workspace; outputs[] is a list of strings.
    Inputs reference upstream task_ids or artifact paths.

    Phase 6.0 added `acceptance_criteria` — the structured AC list the
    Critic walks before allowing `state='done'`.  Phase 6.0.1 + 6.0.2
    add the hard-gate guard + Director-side validation respectively.
    """
    return pa.schema(_shared_skeleton_fields() + [
        pa.field("owner_id", pa.string(), nullable=False),     # agent_id
        pa.field("assigner_id", pa.string(), nullable=True),    # agent_id; None = self
        pa.field("brief", pa.string(), nullable=False),
        pa.field("deliverable_spec", pa.string(), nullable=False),
        pa.field("inputs", pa.list_(pa.string()), nullable=True),
        pa.field("outputs", pa.list_(pa.string()), nullable=True),
        pa.field("due_ts", pa.float64(), nullable=True),
        pa.field("trace_ref", pa.string(), nullable=True),       # TCMM trace cid
        pa.field("comments_head_hash", pa.string(), nullable=True),
        # Comments live in task_comments; we store the head hash here
        # so a mutation to that table is detectable (per spec §3.8.5).
        pa.field("origin", pa.string(), nullable=True),          # foreground|background
        pa.field("pattern", pa.string(), nullable=True),         # A|B|C|D
        pa.field("constitution_version", pa.int64(), nullable=True),
        # Inbox-poller lease (workers/inbox_poller.py).  Workers claim
        # a task by atomically writing lease_owner + lease_until.  If
        # lease_until is in the past or null, the task is available.
        pa.field("lease_owner", pa.string(), nullable=True),
        pa.field("lease_until", pa.float64(), nullable=True),
        # Phase 6.0 — structured acceptance criteria.  Non-nullable at the
        # row level (always a list — may be empty for legacy/backfilled
        # rows, but never NULL).  The Director-side validator in
        # `create_task` rejects empty lists for new rows.
        pa.field(
            "acceptance_criteria",
            pa.list_(_acceptance_criterion_struct()),
            nullable=False,
        ),
        # [PHASE_7_5_TEAM_ID_2026_05_28] Optional team membership.
        # When set, `create_task` checks the team's budget envelope
        # before persisting.  `update_status('done')` triggers a
        # rollup of `agent_teams.cost_attributed_cached_usd`.  Null
        # for tasks that aren't team-scoped (legacy + ad-hoc Director
        # work outside any team).
        pa.field("team_id", pa.string(), nullable=True),
        # [PHASE_7_5_DEPENDS_ON_2026_05_28] Cross-lineage DAG dependencies.
        # `parent_id` + `lineage_chain` only model single-edge subtask
        # relationships (the producer-of-X tree); `depends_on` lets a
        # Task block on N sibling/cousin tasks completing first — fan-in
        # patterns (Researcher A's draft must merge with Researcher B's
        # findings before Builder can start).  Inbox-poller treats a task
        # as un-claimable until every task_id in this list reports
        # status='done'.  Cycle prevention is the Director's responsibility
        # at create_task time; the ledger only stores + reads.
        pa.field("depends_on", pa.list_(pa.string()), nullable=True),
        pa.field("extras_json", pa.string(), nullable=True),     # forward-compat
    ])


# ── task_comments (append-only chain) ────────────────────────────────────


def task_comments_schema() -> pa.Schema:
    """Append-only comment chain per spec §3.8.5.

    `prev_hash` chains comments so mutation of any prior comment breaks
    the chain (audit alarm).  `kind` enum drives downstream processing:
      comment              — plain text comment
      status_change        — `status_from → status_to` transition log
      review_request       — IC asked for review
      review_decision      — Critic returned approve/changes/decline
      blocker_raised       — IC raised a blocker
      blocker_cleared      — Director resolved a blocker

    Comments are NEVER updated in place.  The Task row's
    `comments_head_hash` tracks the latest one; any mutation that
    rewrites a prior comment will be detected by walking the chain.
    """
    return pa.schema([
        pa.field("id", pa.string(), nullable=False),             # uuid
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("tenant_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("author_id", pa.string(), nullable=False),      # agent_id or "user"
        pa.field("kind", pa.string(), nullable=False),
        pa.field("body", pa.string(), nullable=False),
        pa.field("ts", pa.float64(), nullable=False),
        pa.field("prev_hash", pa.string(), nullable=True),       # SHA-256 of prior comment
        pa.field("self_hash", pa.string(), nullable=False),      # SHA-256 of this comment's canonical form
        pa.field("extras_json", pa.string(), nullable=True),
    ])


# ── task_proposals (dream-as-scheduler candidates) ───────────────────────


def task_proposals_schema() -> pa.Schema:
    """Proposal entity — candidate Tasks from dream-as-scheduler.

    Per spec §3.7.6.  Most proposals never become Tasks (decay → shelve →
    expire); separate table from agent_tasks so dead-letter rows don't
    bloat the active inbox queries.
    """
    return pa.schema(_shared_skeleton_fields() + [
        # signal that triggered this proposal
        pa.field("signal_type", pa.string(), nullable=False),
        pa.field("signal_node_ids", pa.list_(pa.int64()), nullable=True),
        # scoring
        pa.field("impact_score", pa.float64(), nullable=False),
        pa.field("decay_score", pa.float64(), nullable=False),
        pa.field("objective_alignment", pa.float64(), nullable=True),
        pa.field("constraint_violations", pa.list_(pa.string()), nullable=True),
        # Director's pre-eval output
        pa.field("proposed_brief", pa.string(), nullable=False),
        pa.field("proposed_assignee", pa.string(), nullable=False),
        pa.field("proposed_deliverable_spec", pa.string(), nullable=True),
        pa.field("rationale", pa.string(), nullable=True),
        # lifecycle
        pa.field("recurrence_count", pa.int64(), nullable=False),
        pa.field("first_surfaced_ts", pa.float64(), nullable=False),
        pa.field("last_surfaced_ts", pa.float64(), nullable=False),
        pa.field("director_decision_ts", pa.float64(), nullable=True),
        pa.field("shelf_reason", pa.string(), nullable=True),
        pa.field("resulting_task_id", pa.string(), nullable=True),
        pa.field("constitution_version", pa.int64(), nullable=True),
        # [PHASE_3_EMERGENCY_LANE_COLUMN_2026_05_27]  Per spec §3.7.3
        # USER×USER contradictions bypass per-day caps + Director
        # pre-eval and surface immediately.  Typed column lets the
        # sidebar query split queue/emergency cleanly without the
        # rationale-prefix heuristic the /proposals endpoint used to
        # rely on.  Nullable so older rows just read as False.
        pa.field("emergency_lane", pa.bool_(), nullable=True),
        # Phase 7.1 M3 — cross-ref to TCMM observation carrying the
        # proposal's content (proposed_brief + rationale).  Set when
        # `record_proposal` split-writer fires; null on pre-Phase-7
        # rows.  Ledger keeps the operational fields; TCMM gets the
        # narrative.  Pointer-not-mirror per §2 design principle.
        pa.field("tcmm_obs_id", pa.string(), nullable=True),
        pa.field("extras_json", pa.string(), nullable=True),
    ])


# ── proposal_outcomes (regret + value_realized tracking) ─────────────────


def proposal_outcomes_schema() -> pa.Schema:
    """Per spec §3.7.7.  Written 30d after task completion.

    `regret_score = cost / max(value_realized, ε)` for v1.
    `value_realized` excludes producer-self-recalls (regret gaming
    mitigation per §11.6.2).
    """
    return pa.schema(_shared_skeleton_fields() + [
        pa.field("proposal_id", pa.string(), nullable=False),
        pa.field("resulting_task_id", pa.string(), nullable=True),
        pa.field("task_status", pa.string(), nullable=False),  # done|cancelled|failed
        pa.field("task_cost_usd", pa.float64(), nullable=False),
        pa.field("value_realized", pa.float64(), nullable=False),
        pa.field("regret_score", pa.float64(), nullable=False),
        pa.field("objective_deltas_json", pa.string(), nullable=True),
        pa.field("computed_at_ts", pa.float64(), nullable=False),
        # Phase 7.1 M2 — cross-ref to TCMM observation carrying
        # outcome narrative (regret_text, what_went_wrong,
        # lessons_learned).  Same pointer-not-mirror discipline as M3.
        pa.field("tcmm_obs_id", pa.string(), nullable=True),
        pa.field("extras_json", pa.string(), nullable=True),
    ])


# ── org_memory (institutional lessons) ───────────────────────────────────


def org_memory_schema() -> pa.Schema:
    """Per spec §3.9.2 — institutional lessons with expiry + decay.

    Distinct from skills (how-tos at agents/skills/) and from blackboard
    (factual knowledge in dream).  Lessons are rules about how the
    organization operates.

    Expiry mechanics:
      - `expires_at` default = created_ts + 180d
      - `review_after` default = created_ts + 90d
      - `confidence_decay_per_week` = 0.02 (eroded if not reinforced)
      - reinforcement requires evidence from ≥2 distinct `extracted_by`
        values (cross-agent or USER — spec §3.8.5)
    """
    return pa.schema(_shared_skeleton_fields() + [
        pa.field("trigger", pa.string(), nullable=False),
        pa.field("rule", pa.string(), nullable=False),
        pa.field("confidence", pa.float64(), nullable=False),
        pa.field("evidence_task_ids", pa.list_(pa.string()), nullable=True),
        pa.field("promoted_from", pa.string(), nullable=True),  # reflective_heuristic id
        pa.field("reinforcement_count", pa.int64(), nullable=False),
        pa.field("reinforced_by_agent_ids", pa.list_(pa.string()), nullable=True),
        pa.field("last_reinforced_ts", pa.float64(), nullable=False),
        pa.field("expires_at", pa.float64(), nullable=False),
        pa.field("review_after", pa.float64(), nullable=False),
        pa.field("confidence_decay_per_week", pa.float64(), nullable=False),
        pa.field("reviews_json", pa.string(), nullable=True),  # serialized history
        pa.field("imported_from_tenant", pa.string(), nullable=True),
        pa.field("imported_from_lesson_id", pa.string(), nullable=True),
        pa.field("constitution_version", pa.int64(), nullable=True),
        pa.field("extras_json", pa.string(), nullable=True),
    ])


# ── client_tool_approvals (approval gate audit) ──────────────────────────


def client_tool_approvals_schema() -> pa.Schema:
    """Audit log for every approval gate decision (per spec §3.8).

    Single source of truth for "what was asked, what did the user say,
    when, why."  Includes both APPROVED and DENIED outcomes plus the
    foreground bypasses (decision=auto_foreground) so the log is
    complete.
    """
    return pa.schema([
        pa.field("id", pa.string(), nullable=False),
        pa.field("ts", pa.float64(), nullable=False),       # server-stamped (received_at_vm)
        pa.field("ts_local", pa.float64(), nullable=True),  # daemon-supplied (decided_at_local)
        pa.field("tenant_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("agent_id", pa.string(), nullable=False),
        pa.field("conversation_id", pa.string(), nullable=True),
        pa.field("parent_cid", pa.string(), nullable=True),
        pa.field("tool", pa.string(), nullable=False),
        pa.field("args_sha256", pa.string(), nullable=False),
        pa.field("args_preview", pa.string(), nullable=True),  # PII-redacted
        pa.field("origin", pa.string(), nullable=False),       # foreground|background
        pa.field("decision", pa.string(), nullable=False),     # allow|deny|approve|timeout|auto_foreground
        pa.field("reason", pa.string(), nullable=True),
        pa.field("latency_ms", pa.int64(), nullable=True),
        pa.field("bypass_rule_id", pa.string(), nullable=True),
        pa.field("approval_token", pa.string(), nullable=True),  # for TOCTOU defense per §3.8.5
        pa.field("extras_json", pa.string(), nullable=True),
    ])


# ── client_tool_bypass (user-defined "always allow") ─────────────────────


def client_tool_bypass_schema() -> pa.Schema:
    """User-defined bypass rules per spec §3.8.

    Keyed on (user_id, agent_id, tool, arg_glob).  Optional expires_at
    for time-bounded grants.  Stored in Lance (not TCMM extras_json)
    because security policy shouldn't blend with memory blocks.
    """
    return pa.schema([
        pa.field("id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("agent_id", pa.string(), nullable=True),  # null = applies to any agent
        pa.field("tool", pa.string(), nullable=False),
        pa.field("arg_glob", pa.string(), nullable=False),
        pa.field("created_ts", pa.float64(), nullable=False),
        pa.field("expires_at", pa.float64(), nullable=True),
        pa.field("created_via", pa.string(), nullable=True),  # toast|sidebar|cli
        pa.field("active", pa.bool_(), nullable=False),
        pa.field("extras_json", pa.string(), nullable=True),
    ])


# ── alignment_weights (runtime-tunable scoring weights) ────────────────


def alignment_weights_schema() -> pa.Schema:
    """Per spec §3.7.2 end — weekly recalibration adjusts
    DEFAULT_ALIGNMENT_VECTORS based on proposal_outcomes regret
    aggregates.  Storing them in Lance (instead of hardcoded in
    `scoring.py`) lets the recalibration job persist its decisions.

    One row per (tenant_id, signal_type, objective_id).  Workers read
    all-rows-for-tenant via `get_alignment_for_tenant()` and fall back
    to the static defaults in `scoring.DEFAULT_ALIGNMENT_VECTORS` when
    no row exists (start-state).

    `weight` is the current calibrated value [0, 1].  `last_regret_avg`
    is the rolling 4-week mean regret_score for this (signal_type,
    objective_id) combo — what the recalibration uses to nudge weights.
    """
    return pa.schema([
        pa.field("id", pa.string(), nullable=False),
        pa.field("tenant_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("signal_type", pa.string(), nullable=False),
        pa.field("objective_id", pa.string(), nullable=False),
        pa.field("weight", pa.float64(), nullable=False),
        pa.field("default_weight", pa.float64(), nullable=False),  # for diff/audit
        pa.field("last_regret_avg", pa.float64(), nullable=True),
        pa.field("last_recalibrated_ts", pa.float64(), nullable=False),
        pa.field("recalibration_count", pa.int64(), nullable=False),
        pa.field("created_ts", pa.float64(), nullable=False),
        pa.field("updated_ts", pa.float64(), nullable=False),
        pa.field("extras_json", pa.string(), nullable=True),
    ])


# ── tenant_proactive_config (per-tenant proactive-stream gates) ─────────


def tenant_proactive_config_schema() -> pa.Schema:
    """Per spec §3.7.3 + Phase 3 plan — per-tenant gates on the
    proactive stream.  Promoted from open-question to ship-requirement.

    One row per (tenant_id, user_id).  Missing row → defaults apply
    (proactive stream enabled, 12 cycles/day, 20 approvals/day cap,
    $5/day cost ceiling).
    """
    return pa.schema([
        pa.field("id", pa.string(), nullable=False),
        pa.field("tenant_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        # Master switch — when False the DreamScanner + LifecycleWorker
        # both skip this tenant.  Dream itself still runs (no change to
        # TCMM); proposals just don't surface.
        pa.field("proactive_stream_enabled", pa.bool_(), nullable=False),
        # Cycle cadence — DreamScanner / dream cycle cadence per day.
        # Default 12 (= every 2h).  Tunable per tenant for cost control.
        pa.field("proactive_cycles_per_day", pa.int64(), nullable=False),
        # Approval cap — Director may approve at most this many
        # proposals/day automatically.  User-driven Tasks bypass.
        pa.field("proactive_approval_cap_per_day", pa.int64(), nullable=False),
        # Cost ceiling — aggregate per-day cap from constitution
        # constraint cost_ceiling_per_tenant_per_day; defaults to $5.
        pa.field("cost_ceiling_per_tenant_per_day_usd", pa.float64(), nullable=False),
        # Auto-pause state — set by signal-quality-drift watchdog
        # (deferred Phase 3 work).  True means stream is paused.
        pa.field("paused", pa.bool_(), nullable=False),
        pa.field("paused_reason", pa.string(), nullable=True),
        pa.field("paused_at_ts", pa.float64(), nullable=True),
        pa.field("created_ts", pa.float64(), nullable=False),
        pa.field("updated_ts", pa.float64(), nullable=False),
        pa.field("extras_json", pa.string(), nullable=True),
    ])


# ── agent_task_heartbeats (Phase 6.3 lease TTL + heartbeats) ────────────


def agent_teams_schema() -> pa.Schema:
    """Phase 7.5 — `agent_teams` Lance schema.

    A Team is a named bundle of agents with a shared lead, a budget,
    and a cost ceiling.  Tasks may be assigned to a team via
    `agent_tasks.team_id`; the Director-side `create_task` validator
    rejects new work for teams whose attributed cost has crossed the
    budget ceiling (`team_cost_attributed > budget_usd × budget_cap`).

    Membership is stored inline as `member_agent_ids` — small list
    (<20 typical), no need for a join table at v1 scale.
    """
    return pa.schema(_shared_skeleton_fields() + [
        pa.field("name",             pa.string(),  nullable=False),
        pa.field("lead_agent_id",    pa.string(),  nullable=False),
        pa.field("member_agent_ids", pa.list_(pa.string()), nullable=True),
        # Budget envelope.  `budget_usd` is the soft ceiling; cap=1.0
        # means "block at exactly budget"; cap=1.2 means "20% slack
        # before hard block".  Operator-configurable per team.
        pa.field("budget_usd",       pa.float64(), nullable=False),
        pa.field("budget_cap",       pa.float64(), nullable=True),
        # Cached cost rollup — recomputed by `team_cost_attributed()`
        # at create_task time; this column is a denormalised snapshot
        # for the sidebar / Director's view, NOT the source of truth.
        # Source of truth = sum(agent_tasks.cost_attributed_usd
        # WHERE team_id = <this>).
        pa.field("cost_attributed_cached_usd", pa.float64(), nullable=True),
        pa.field("cost_recomputed_ts",         pa.float64(), nullable=True),
        pa.field("extras_json",                pa.string(),  nullable=True),
    ])


def agent_task_heartbeats_schema() -> pa.Schema:
    """Heartbeat ring per spec §6.3 — workers beat at every turn so the
    inbox-poller can detect dead workers holding stale claims.

    One row per (task_id, worker_id) heartbeat event.  The most-recent
    row for a task_id is the "live" beat; the inbox-poller sweep
    auto-reclaims tasks where `now - max(last_beat_at) > lease_ttl_s`.
    """
    return pa.schema([
        pa.field("id",           pa.string(),  nullable=False),
        pa.field("task_id",      pa.string(),  nullable=False),
        pa.field("worker_id",    pa.string(),  nullable=False),
        pa.field("tenant_id",    pa.string(),  nullable=False),
        pa.field("user_id",      pa.string(),  nullable=False),
        pa.field("last_beat_at", pa.float64(), nullable=False),
        pa.field("lease_ttl_s",  pa.float64(), nullable=False),
        pa.field("created_ts",   pa.float64(), nullable=False),
        pa.field("extras_json",  pa.string(),  nullable=True),
    ])


# ── Registry: name → schema function ─────────────────────────────────────

# [M1_CUTOVER_2026_05_28] `org_memory` removed from the active registry.
# Lessons live in TCMM `archive` (read via `app.memory.lessons_reader`).
# The `org_memory_schema()` function is retained below for callers that
# need to inspect the legacy shape (one-shot migrations, archaeology)
# but isn't registered for table-init / migration.
TABLE_SCHEMAS = {
    "agent_tasks": agent_tasks_schema,
    "task_comments": task_comments_schema,
    "task_proposals": task_proposals_schema,
    "proposal_outcomes": proposal_outcomes_schema,
    "client_tool_approvals": client_tool_approvals_schema,
    "client_tool_bypass": client_tool_bypass_schema,
    "tenant_proactive_config": tenant_proactive_config_schema,
    "alignment_weights": alignment_weights_schema,
    "agent_task_heartbeats": agent_task_heartbeats_schema,
    # Phase 7.5
    "agent_teams": agent_teams_schema,
}


def _register_external_tables() -> None:
    """Late-register tables defined outside this module (Phase 5+).

    Modules that own non-ledger Lance schemas (e.g. `a2a_external.py`)
    register them via this hook so `store._open_or_create` finds
    them.  Call from main.py startup once the modules import.
    """
    try:
        from ..a2a_external import a2a_external_keys_schema
        TABLE_SCHEMAS.setdefault("a2a_external_keys", a2a_external_keys_schema)
    except Exception:
        pass


__all__ = [
    "TABLE_SCHEMAS",
    "agent_tasks_schema",
    "task_comments_schema",
    "task_proposals_schema",
    "proposal_outcomes_schema",
    "org_memory_schema",
    "tenant_proactive_config_schema",
    "agent_teams_schema",
    "client_tool_approvals_schema",
    "client_tool_bypass_schema",
]
