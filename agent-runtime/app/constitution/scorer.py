"""Constitution-aware scoring for proposals.

Per spec §3.7.2:

    final_score = signal_impact × objective_alignment × constraint_gate

where:
  - signal_impact ∈ [0, ∞)  ← computed by dream-side scoring per signal type
  - objective_alignment ∈ [0, 1]  ← dot product of proposal's expected
        objective delta vector against constitution.objectives weights
  - constraint_gate ∈ {0, 1}  ← 0 if any constitution constraint would be
        violated by the proposal's execution; 1 otherwise

Multiplicative composition: any weak factor zeroes the candidate.
This module is pure functions; no I/O, easy to unit test.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("agent-runtime.scorer")


# Default alignment vectors per signal type — per spec §3.7.2.
# These are seed values; weekly recalibration updates them based on
# measured objective_deltas from proposal_outcomes.  For top-3
# candidates we use these defaults directly (no LLM call); for ranks
# 4-N the Director's Haiku rank pass refines them per-candidate.
DEFAULT_ALIGNMENT_VECTORS: dict[str, dict[str, float]] = {
    "information_gap": {
        "reduce_toil": 0.5,
        "improve_security": 0.3,
        "preserve_user_agency": 0.2,
    },
    "contradiction_arc": {
        "improve_security": 0.5,
        "preserve_user_agency": 0.4,
        "reduce_toil": 0.1,
    },
    "reflective_heuristic": {
        "reduce_toil": 0.7,
        "improve_security": 0.2,
        "preserve_user_agency": 0.1,
    },
    "recurring_ritual": {
        "reduce_toil": 0.8,
        "improve_security": 0.1,
        "preserve_user_agency": 0.1,
    },
    "stance_arc": {
        "preserve_user_agency": 0.5,
        "improve_security": 0.4,
        "reduce_toil": 0.1,
    },
    "low_stability_cluster": {
        "improve_security": 0.6,
        "preserve_user_agency": 0.3,
        "reduce_toil": 0.1,
    },
    "stale_supersession_chain": {
        "improve_security": 0.5,
        "reduce_toil": 0.4,
        "preserve_user_agency": 0.1,
    },
}


def objective_alignment(
    delta_vector: dict[str, float],
    constitution_objectives: list[dict[str, Any]],
) -> float:
    """Dot product of a proposal's expected objective delta vector
    against the constitution's objective weights.

    delta_vector       — e.g. {"reduce_toil": 0.6, "improve_security": 0.3}
    constitution_objectives — list of {id, weight, description}

    Returns a scalar in roughly [0, 1].  Values >1 are possible if the
    proposal claims to advance multiple objectives strongly; we don't
    clip because the comparator only cares about relative ordering.
    """
    if not delta_vector:
        return 0.0
    score = 0.0
    for obj in constitution_objectives:
        oid = obj.get("id")
        w = obj.get("weight", 0.0)
        if oid and oid in delta_vector:
            score += w * delta_vector[oid]
    return score


def default_alignment_for_signal(
    signal_type: str,
    constitution_objectives: list[dict[str, Any]],
) -> float:
    """Compute alignment from the static signal-type default vector
    (used for top-3 candidates without an LLM call).

    If the signal_type isn't in DEFAULT_ALIGNMENT_VECTORS, returns 0.0
    so the candidate effectively drops to the bottom of the queue —
    correct behavior for unknown signal types.
    """
    delta = DEFAULT_ALIGNMENT_VECTORS.get(signal_type)
    if not delta:
        logger.warning(
            f"[scorer] no default alignment vector for signal_type={signal_type!r}; "
            "candidate will score 0"
        )
        return 0.0
    return objective_alignment(delta, constitution_objectives)


def constraint_gate(
    proposal: dict[str, Any],
    constitution_constraints: list[dict[str, Any]],
) -> tuple[int, list[str]]:
    """Evaluate the constitution's constraints against a proposal.

    Returns:
        (gate_value, violations)

      gate_value=1  → all constraints pass; proposal may proceed
      gate_value=0  → ≥1 constraint violated; proposal auto-declines

    Each constraint's `rule` field is a natural-language string that
    describes the policy.  Mechanical enforcement of every rule isn't
    feasible (some require LLM judgment), so this function implements
    the *mechanically-checkable* ones and returns "violations" for the
    rest as a flag the human reviewer can use.

    Mechanically enforced:
      - no_hidden_automation: rejects if proposed_cost_usd > 0.50 AND
        the proposal lacks user_approved=True
      - cost_ceiling_per_task: rejects if proposed_cost_usd > 5.00
      - cost_ceiling_per_tenant_per_day: rejects if tenant's daily
        accumulated cost + proposed_cost_usd would exceed the limit
        (limit defaults to $5/day; configurable per spec §3.9.1)

    Returns gate_value=0 if any mechanically-checked constraint fails.
    Returns the list of violation IDs in `violations` for audit logging.
    """
    violations: list[str] = []
    cost = float(proposal.get("estimated_cost_usd", 0.0))
    user_approved = bool(proposal.get("user_approved", False))

    for c in constitution_constraints:
        cid = c.get("id", "")

        if cid == "no_hidden_automation":
            if cost > 0.50 and not user_approved:
                violations.append(cid)

        elif cid == "cost_ceiling_per_task":
            if cost > 5.00:
                violations.append(cid)

        elif cid == "cost_ceiling_per_tenant_per_day":
            tenant_daily_cost = float(proposal.get("tenant_daily_cost_usd", 0.0))
            daily_limit = float(c.get("limit_usd", 5.00))
            if tenant_daily_cost + cost > daily_limit:
                violations.append(cid)

        elif cid == "preserve_provenance":
            # Mechanical check: if proposal would publish a claim
            # with source_kind=INFERRED to org_blackboard, decline
            # unless explicitly Critic-promoted.
            if (proposal.get("target") == "org_blackboard"
                    and proposal.get("source_kind") == "INFERRED"
                    and not proposal.get("critic_promoted", False)):
                violations.append(cid)

        elif cid == "no_autonomous_client_daemon_access":
            # Mechanical check: if proposal would invoke client-daemon
            # tools from a background-origin agent without explicit
            # user approval, decline.  This duplicates the runtime
            # approval gate but at the *proposal* layer (cheaper to
            # catch here than after Task creation).
            if (proposal.get("invokes_client_daemon", False)
                    and proposal.get("origin") == "background"
                    and not user_approved):
                violations.append(cid)

        # Other constraints are advisory; logged but don't gate.
        else:
            logger.debug(
                f"[scorer] constraint {cid!r} is advisory (no mechanical check)"
            )

    return (0 if violations else 1, violations)


def final_score(
    signal_impact: float,
    signal_type: str,
    proposal: dict[str, Any],
    constitution: dict[str, Any] | None,
) -> tuple[float, list[str]]:
    """Compose the final proposal score per §3.7.2.

    Returns (score, violated_constraints).  If constitution is None
    (not loaded yet), score = signal_impact and no constraint check
    runs — degraded mode for fresh deployments.
    """
    if constitution is None:
        return signal_impact, []

    objectives = constitution.get("objectives", [])
    constraints = constitution.get("constraints", [])

    alignment = default_alignment_for_signal(signal_type, objectives)
    gate, violations = constraint_gate(proposal, constraints)

    return signal_impact * alignment * gate, violations


__all__ = [
    "DEFAULT_ALIGNMENT_VECTORS",
    "objective_alignment",
    "default_alignment_for_signal",
    "constraint_gate",
    "final_score",
]
