"""CriticAgent — review gate before team-knowledge / org-blackboard writes.

Split into two flavors per spec §4.4 (the bundling-forced-shallow-review
decision):

  CriticClaimAgent  (agents/critic-claim.md)
    Haiku, INLINE.  Runs structural validation on every claim destined
    for team_knowledge.  Cheap (~1-3s).  Tools: memory(recall),
    validation(validate_claims, flag).

  CriticProseAgent  (agents/critic-prose.md)
    Sonnet, ASYNC.  Runs semantic review on prose artifacts destined
    for org_blackboard or user_deliverable.  Expensive (minutes).
    Tools: memory(recall), review(score, request_changes, approve,
    decline).

Both inherit the standard 5-step pipeline.  Their review_decision tool
call lands in the agent_tasks ledger via the normal external dispatch
loop; the runtime doesn't need to know they're "critics".
"""

from __future__ import annotations

from .base import Agent


class CriticClaimAgent(Agent):
    """Inline structural validator (Haiku).

    Phase 6.1 — Fresh-context critic.  This persona MUST receive its
    dispatch prompt with ONLY (spec, acceptance_criteria, artifact).
    The producer's per-turn reasoning, status_transition comments, and
    narrative deliberation MUST NOT be inlined — otherwise the critic
    inherits the producer's trajectory blind spots and rubber-stamps.
    Enforced by `inbox_poller._make_critic_user_message` (Phase 6.1
    dispatch path) + AC-21/AC-22/AC-23 acceptance tests.
    """

    _FRESH_CONTEXT: bool = True


class CriticProseAgent(Agent):
    """Async semantic reviewer (Sonnet).  See Phase 6.1 note on
    CriticClaimAgent — identical fresh-context discipline applies.
    """

    _FRESH_CONTEXT: bool = True


__all__ = ["CriticClaimAgent", "CriticProseAgent"]
