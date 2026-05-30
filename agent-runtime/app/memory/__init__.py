"""Phase 6.8 — Memory writer-function layer.

The 5-7 typed writer functions are the ONLY sanctioned write surface
for memory destinations.  Direct imports of Lance handles, TCMM HTTP
clients, or workspace fs writers from any module outside this package
violate the §3.11 memory write-path discipline and are caught by the
lint test in `tests/memory/test_write_path_lint.py` (AC-36).

Public surface:
  - record_episode       — raw observations during a turn → TCMM agent ns
  - promote_to_semantic  — Critic-gated claim → TCMM team/knowledge
  - log_decision         — status_transition / review_decision → ledger
  - update_org_memory    — institutional lesson → org_memory ledger (Phase 7 moves to TCMM)
  - enqueue_dream_input  — high-importance event → TCMM dream_archive
  - record_approval      — approval-gate decision → ledger client_tool_approvals
  - attach_artifact      — builder output → agent_tasks.outputs[] + workspace
"""

from .writers import (
    record_episode,
    record_heartbeat,
    promote_to_semantic,
    log_decision,
    update_org_memory,    # deprecated post-M1 cutover; kept for back-compat
    enqueue_dream_input,
    record_approval,
    attach_artifact,
    WriterError,
    WRITER_DESTINATIONS,
)
# Phase 7 split-writers — the sanctioned cross-ref sites (ledger PK ↔
# TCMM obs_id).  Imported but kept distinct from the Phase 6.8 single-
# destination writers above; the linter and AC-37 doc-gen include both.
from .phase_7_writers import (
    record_outcome_with_narrative,
    record_proposal_with_content,
    promote_lesson_to_team_knowledge,
    record_discussion_comment,
)

__all__ = [
    "record_episode",
    "record_heartbeat",
    "promote_to_semantic",
    "log_decision",
    "update_org_memory",
    "enqueue_dream_input",
    "record_approval",
    "attach_artifact",
    # Phase 7 split-writers
    "record_outcome_with_narrative",
    "record_proposal_with_content",
    "promote_lesson_to_team_knowledge",
    "record_discussion_comment",
    "WriterError",
    "WRITER_DESTINATIONS",
]
