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


def agent_tasks_schema() -> pa.Schema:
    """The Task entity — work items assigned to agents.

    Comments live in `task_comments` (separate table, append-only chain).
    Outputs are file paths in workspace; outputs[] is a list of strings.
    Inputs reference upstream task_ids or artifact paths.
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


# ── Registry: name → schema function ─────────────────────────────────────

TABLE_SCHEMAS = {
    "agent_tasks": agent_tasks_schema,
    "task_comments": task_comments_schema,
    "task_proposals": task_proposals_schema,
    "proposal_outcomes": proposal_outcomes_schema,
    "org_memory": org_memory_schema,
    "client_tool_approvals": client_tool_approvals_schema,
    "client_tool_bypass": client_tool_bypass_schema,
}


__all__ = [
    "TABLE_SCHEMAS",
    "agent_tasks_schema",
    "task_comments_schema",
    "task_proposals_schema",
    "proposal_outcomes_schema",
    "org_memory_schema",
    "client_tool_approvals_schema",
    "client_tool_bypass_schema",
]
