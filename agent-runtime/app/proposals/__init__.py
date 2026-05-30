"""Phase 3 — dream-as-scheduler scoring + assignment + briefs.

Layered separate from `app.ledger.proposals` (storage) so the pure-
function scoring + brief composition can be unit-tested without any
Lance setup, and the storage layer doesn't grow dependencies on
constitution / signal-emitter modules.

Module map:
    scoring.py    — pure-function impact + objective + constraint scoring
    briefs.py     — template-driven brief composition per signal_type
    assignees.py  — signal_type → default agent_id mapping
    lifecycle.py  — decay/expire workers (tenant scan loop)

Caller flow (Phase 3 §3.7.5 pre-evaluation):
    signal_node → SignalPayload → final_score → top-N → brief → save
"""

from .scoring import (
    CONFIRMED_SIGNAL_TYPES,
    DEFERRED_SIGNAL_TYPES,
    ALL_SIGNAL_TYPES,
    SIGNAL_CONTRADICTION_ARC,
    SignalPayload,
    signal_impact,
    objective_alignment,
    constraint_gate,
    final_score,
)


def is_emergency_lane(signal_type: str, payload: SignalPayload) -> bool:
    """Phase 3 §3.7.3 emergency lane.

    `USER × USER` contradictions skip per-day caps and Director's
    pre-eval — they surface directly to the user as a "needs your
    attention" sidebar item.  Rationale: a contradiction the user
    themselves stated on both sides is a *real* one, not noise; queueing
    it behind low-impact info_gaps would be wrong.

    Other USER-involved contradictions (USER×AGENT, USER×INFERRED) are
    high-impact but still pre-evaluated normally.  Only the symmetric
    USER×USER case is emergency.
    """
    if signal_type != SIGNAL_CONTRADICTION_ARC:
        return False
    a = (getattr(payload, "source_kind_a", "") or "").upper()
    b = (getattr(payload, "source_kind_b", "") or "").upper()
    return a == "USER" and b == "USER"


__all__ = [
    "CONFIRMED_SIGNAL_TYPES",
    "DEFERRED_SIGNAL_TYPES",
    "ALL_SIGNAL_TYPES",
    "SignalPayload",
    "signal_impact",
    "objective_alignment",
    "constraint_gate",
    "final_score",
    "is_emergency_lane",
]
