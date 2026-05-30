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


# Default assignee per signal_type.  ``critic-claim``/``critic-prose``
# split mirrors §4.4a vs §4.4b: claim-level structural verification vs
# prose-level semantic review.
_DEFAULT_ASSIGNEES: dict[str, str] = {
    SIGNAL_INFORMATION_GAP:      "researcher",
    SIGNAL_REFLECTIVE_HEURISTIC: "critic-prose",
    SIGNAL_RECURRING_RITUAL:     "critic-prose",
    SIGNAL_LOW_STABILITY:        "researcher",
    SIGNAL_STALE_SUPERSESSION:   "researcher",
    # contradiction + stance_arc have payload-conditional logic below
}


def default_assignee(
    signal_type: str,
    payload: Optional[Mapping[str, object]] = None,
) -> str:
    """Return the default agent_id for this signal.

    For most signals the mapping is static.  For ``contradiction_arc``
    the choice depends on the source_kind pair: USER×USER conflicts go
    to critic-claim (structural arbitration), USER×AGENT or
    AGENT×AGENT to researcher (find the right evidence first), and
    INFERRED-only to critic-claim (low-cost cleanup).

    For ``stance_arc``, when two AGENT stances disagree we route to
    critic-claim — committee review is structural.  The spec also calls
    out a relevant-consultant invitation, which the Director can layer
    on at dispatch time.
    """
    static = _DEFAULT_ASSIGNEES.get(signal_type)
    if static:
        return static

    payload = payload or {}

    if signal_type == SIGNAL_CONTRADICTION_ARC:
        a = str(payload.get("source_kind_a") or "").upper()
        b = str(payload.get("source_kind_b") or "").upper()
        if "USER" in (a, b):
            # User claims always need the higher-stakes arbiter
            return "critic-claim"
        if a == "AGENT" and b == "AGENT":
            return "researcher"
        # INFERRED-only or unknown → critic-claim (cheap cleanup)
        return "critic-claim"

    if signal_type == SIGNAL_STANCE_ARC:
        # Multi-reviewer review per §4.4a
        return "critic-claim"

    # Unknown signal_type → researcher (safest default — produces a
    # research artefact the user can decide on)
    return "researcher"


__all__ = ["default_assignee"]
