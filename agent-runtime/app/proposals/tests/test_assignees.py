"""Proposal assignee routing — owners must be PRODUCERS, never critics.

A proposal's owner produces the deliverable (write_file / attach_output /
submit_for_review). Critics have NONE of those tools (§4.4 — read-everywhere,
write-nowhere), so routing a proposal to a critic dead-ends the task. Caught
live 2026-06-02: a `recurring_ritual` proposal was routed to critic-prose,
which logged "tools required but absent: write_file, attach_output,
submit_for_review", composed the skill inline in a comment, raised a blocker,
and the task NON-CONVERGED. Owners must always be producers.
"""

from __future__ import annotations

from app.proposals.assignees import default_assignee, _CRITIC_ROLES
from app.proposals.scoring import (
    ALL_SIGNAL_TYPES,
    SIGNAL_RECURRING_RITUAL,
    SIGNAL_REFLECTIVE_HEURISTIC,
    SIGNAL_CONTRADICTION_ARC,
    SIGNAL_STANCE_ARC,
)


def test_no_signal_routes_to_a_critic():
    for sig in ALL_SIGNAL_TYPES:
        a = default_assignee(sig)
        assert a not in _CRITIC_ROLES, (
            f"{sig} routed to critic {a!r} — critics have no write tools and "
            f"would dead-end the task"
        )


def test_recurring_ritual_goes_to_producer():
    # The exact case that dead-ended live.
    assert default_assignee(SIGNAL_RECURRING_RITUAL) == "researcher"


def test_reflective_heuristic_goes_to_producer():
    assert default_assignee(SIGNAL_REFLECTIVE_HEURISTIC) == "researcher"


def test_contradiction_and_stance_go_to_producer():
    # These previously routed to critic-claim regardless of payload.
    for sig in (SIGNAL_CONTRADICTION_ARC, SIGNAL_STANCE_ARC):
        assert default_assignee(sig) not in _CRITIC_ROLES
    # USER×USER contradiction (the old critic-claim branch) is no exception.
    assert default_assignee(
        SIGNAL_CONTRADICTION_ARC,
        {"source_kind_a": "USER", "source_kind_b": "USER"},
    ) == "researcher"


def test_guard_coerces_a_critic_mapping_to_researcher():
    """Defense-in-depth: even if a future edit maps a signal to a critic, the
    guard must coerce it."""
    import app.proposals.assignees as A
    A._DEFAULT_ASSIGNEES["__test_bad__"] = "critic-prose"
    try:
        assert default_assignee("__test_bad__") == "researcher"
    finally:
        del A._DEFAULT_ASSIGNEES["__test_bad__"]
