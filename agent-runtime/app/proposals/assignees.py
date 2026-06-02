"""Phase 3 — signal_type → default agent assignment.

Per spec §3.7.1, every confirmed generative signal has a default
assignee role.  This module is the dispatch table; callers pass the
signal_type and (optionally) signal payload context, get back the
agent_id that should own the proposed Task by default.

The mapping is intentionally simple — Director's LLM rank-pass can
override for ranks 4-10.  Keep one explicit case per signal so future
edits don't accidentally route the wrong work to the wrong role.
"""

from __future__ import annotations

from typing import Mapping, Optional

from .scoring import (
    SIGNAL_INFORMATION_GAP,
    SIGNAL_CONTRADICTION_ARC,
    SIGNAL_REFLECTIVE_HEURISTIC,
    SIGNAL_RECURRING_RITUAL,
    SIGNAL_STANCE_ARC,
    SIGNAL_LOW_STABILITY,
    SIGNAL_STALE_SUPERSESSION,
)


# Default assignee per signal_type.
#
# [PROPOSAL_ASSIGNEE_NO_CRITIC_2026_06_02] A proposal's OWNER must PRODUCE the
# deliverable — write_file / attach_output / submit_for_review.  Critics are
# read-everywhere / write-nowhere (§4.4) and have NONE of those tools, so a
# critic can never own a proposal: it composes the content inline in a comment,
# raises a blocker, and the task dead-ends.  Caught live 2026-06-02 — a
# recurring_ritual proposal routed to critic-prose ("tools required but absent:
# write_file, attach_output, submit_for_review" → NON-CONVERGENCE).
#
# The critic split (§4.4a claim / §4.4b prose) applies at REVIEW time via the
# submit_for_review target, NOT to ownership.  Every current signal produces a
# markdown/research artifact → researcher; builder is reserved for code/shell
# work (no current signal needs it).
_DEFAULT_ASSIGNEES: dict[str, str] = {
    SIGNAL_INFORMATION_GAP:      "researcher",
    SIGNAL_REFLECTIVE_HEURISTIC: "researcher",
    SIGNAL_RECURRING_RITUAL:     "researcher",
    SIGNAL_LOW_STABILITY:        "researcher",
    SIGNAL_STALE_SUPERSESSION:   "researcher",
    SIGNAL_CONTRADICTION_ARC:    "researcher",  # writes the investigation note
    SIGNAL_STANCE_ARC:           "researcher",  # writes the committee verdict
}

# Critics own no producing tools — they must never be a proposal owner.
_CRITIC_ROLES = frozenset({"critic-claim", "critic-prose"})


def default_assignee(
    signal_type: str,
    payload: Optional[Mapping[str, object]] = None,
) -> str:
    """Return the default PRODUCER agent_id for this signal's proposal.

    All current generative signals produce a written deliverable, so the owner
    is always a producer (researcher).  ``payload`` is accepted for API
    compatibility + future per-payload routing (e.g. a code-producing signal →
    builder), but is not consumed today.

    Safety net: if a mapping ever resolves to a critic, coerce to researcher —
    critics review artifacts at submit time, they never own production (§4.4).
    """
    assignee = _DEFAULT_ASSIGNEES.get(signal_type, "researcher")
    if assignee in _CRITIC_ROLES:
        return "researcher"
    return assignee


__all__ = ["default_assignee"]
