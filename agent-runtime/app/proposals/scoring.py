"""Phase 3 — dream-as-scheduler impact scoring.

Per spec §3.7.2.  Multiplicative composition: any single weak factor
zeroes the candidate.  ALL formulas are pure functions — unit-testable,
no I/O, no side effects.

```
final_score = signal_impact × objective_alignment × constraint_gate

  constraint_gate = 0 if any constitution constraint violated, else 1
  objective_alignment ∈ [0, 1] from §3.7.2's default alignment vectors
  signal_impact     ∈ [0, ∞) — per-signal formula below
```

The default alignment vectors are static seeds; Director's LLM rank-pass
refines per-candidate for ranks 4-10.  Recalibrated weekly via the
outcome tracker (§3.7.7) using regret_score, not just approval_rate.

Implementation discipline:
  - NO LLM calls in this module.
  - NO Lance reads (caller supplies the signal node payload).
  - Floats are double-precision; use `_safe_mul` to handle missing fields
    cleanly (None → 0.0 so the multiplicative composition drops the
    candidate instead of crashing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ── Signal taxonomy (per §3.7.1) ─────────────────────────────────────────

# 5 confirmed signals (emitted by dream today)
SIGNAL_INFORMATION_GAP        = "information_gap"
SIGNAL_CONTRADICTION_ARC      = "contradiction_arc"
SIGNAL_REFLECTIVE_HEURISTIC   = "reflective_heuristic"
SIGNAL_RECURRING_RITUAL       = "recurring_ritual"
SIGNAL_STANCE_ARC             = "stance_arc"

# 2 deferred signals (require new emitter implementation)
SIGNAL_LOW_STABILITY          = "low_stability_cluster"
SIGNAL_STALE_SUPERSESSION     = "stale_supersession_chain"

CONFIRMED_SIGNAL_TYPES = frozenset({
    SIGNAL_INFORMATION_GAP,
    SIGNAL_CONTRADICTION_ARC,
    SIGNAL_REFLECTIVE_HEURISTIC,
    SIGNAL_RECURRING_RITUAL,
    SIGNAL_STANCE_ARC,
})

DEFERRED_SIGNAL_TYPES = frozenset({
    SIGNAL_LOW_STABILITY,
    SIGNAL_STALE_SUPERSESSION,
})

ALL_SIGNAL_TYPES = CONFIRMED_SIGNAL_TYPES | DEFERRED_SIGNAL_TYPES


# ── Default alignment vectors (per §3.7.2) ──────────────────────────────
# Seeds for the deterministic top-3 path so a candidate gets a
# constitution-aware score without an LLM call.  Director's rank-pass
# refines these per-candidate for ranks 4-10.  Weekly recalibration
# (§3.7.7) reweights them based on `regret_score` + `objective_deltas`.

DEFAULT_ALIGNMENT_VECTORS: dict[str, dict[str, float]] = {
    SIGNAL_INFORMATION_GAP: {
        "reduce_toil":          0.5,
        "improve_security":     0.3,
        "preserve_user_agency": 0.2,
    },
    SIGNAL_CONTRADICTION_ARC: {
        "improve_security":     0.5,
        "preserve_user_agency": 0.4,
        "reduce_toil":          0.1,
    },
    SIGNAL_REFLECTIVE_HEURISTIC: {
        "reduce_toil":          0.7,
        "improve_security":     0.2,
        "preserve_user_agency": 0.1,
    },
    SIGNAL_RECURRING_RITUAL: {
        "reduce_toil":          0.8,
        "improve_security":     0.1,
        "preserve_user_agency": 0.1,
    },
    SIGNAL_STANCE_ARC: {
        "preserve_user_agency": 0.5,
        "improve_security":     0.4,
        "reduce_toil":          0.1,
    },
    SIGNAL_LOW_STABILITY: {
        "improve_security":     0.6,
        "preserve_user_agency": 0.3,
        "reduce_toil":          0.1,
    },
    SIGNAL_STALE_SUPERSESSION: {
        "improve_security":     0.5,
        "reduce_toil":          0.4,
        "preserve_user_agency": 0.1,
    },
}


# ── Source severity for contradiction_arc (per §3.7.2) ──────────────────
# USER claims have highest weight; AGENT×INFERRED is noise floor.
# Used by contradiction_impact() AND by the emergency-lane router
# (USER×USER bypasses Director's pre-eval — §3.7.3).
CONTRADICTION_SOURCE_SEVERITY: dict[tuple[str, str], float] = {
    ("USER", "USER"):     10.0,
    ("USER", "INFERRED"):  5.0,
    ("USER", "AGENT"):     5.0,   # symmetric with USER×INFERRED
    ("AGENT", "USER"):     5.0,
    ("AGENT", "AGENT"):    3.0,
    ("AGENT", "INFERRED"): 1.0,
    ("INFERRED", "USER"):  5.0,
    ("INFERRED", "AGENT"): 1.0,
    ("INFERRED", "INFERRED"): 0.5,
}


def _safe_mul(*factors: Optional[float]) -> float:
    """Multiply factors; treat None as 0.0 so the composition drops the
    candidate instead of crashing.

    Used everywhere in this module so an upstream missing field is
    explicit (final_score = 0) rather than a stack trace.
    """
    out = 1.0
    for f in factors:
        if f is None:
            return 0.0
        try:
            out *= float(f)
        except (TypeError, ValueError):
            return 0.0
    return out


# ── Per-signal impact formulas (per §3.7.2) ─────────────────────────────

@dataclass(frozen=True)
class SignalPayload:
    """The fields signal emitters provide to scoring.

    NOT every field applies to every signal — see the per-formula
    docstrings for which subset each consumes.  Callers should pass
    Nones for irrelevant fields; the scoring functions only read what
    they need.
    """
    # information_gap
    gap_breadth: Optional[float] = None
    downstream_pressure: Optional[float] = None
    # contradiction_arc
    source_kind_a: Optional[str] = None
    source_kind_b: Optional[str] = None
    claim_centrality: Optional[float] = None   # alias for bridge_score
    # reflective_heuristic, recurring_ritual
    recurrence: Optional[float] = None
    success_rate: Optional[float] = None
    token_savings_potential: Optional[float] = None
    # stance_arc
    polarity_distance: Optional[float] = None
    claim_stake: Optional[float] = None
    # low_stability_cluster
    failure_count: Optional[float] = None
    cluster_recall_frequency: Optional[float] = None
    # stale_supersession_chain
    age_days: Optional[float] = None
    recall_count: Optional[float] = None
    topic_currency_index: Optional[float] = None


def information_gap_impact(p: SignalPayload) -> float:
    """`gap_breadth × downstream_pressure`.

    gap_breadth ∈ [0, 1]    : fraction of a topic-cluster's claims that
                              are missing or marked low-confidence.
    downstream_pressure ∈ [0, ∞): how many active Tasks have queried
                              this topic in the last 7d (capped at 5).
    """
    return _safe_mul(p.gap_breadth, p.downstream_pressure)


def contradiction_impact(p: SignalPayload) -> float:
    """`source_severity × claim_centrality`.

    source_severity lookup from CONTRADICTION_SOURCE_SEVERITY based on
    (source_kind_a, source_kind_b).  Missing/unknown pair → 0.
    """
    if not p.source_kind_a or not p.source_kind_b:
        return 0.0
    a = str(p.source_kind_a).upper()
    b = str(p.source_kind_b).upper()
    # Try (a, b) first then (b, a) — the table is symmetric but we
    # store one direction; allow callers to pass in either order.
    severity = (
        CONTRADICTION_SOURCE_SEVERITY.get((a, b))
        or CONTRADICTION_SOURCE_SEVERITY.get((b, a))
        or 0.0
    )
    return _safe_mul(severity, p.claim_centrality)


def reflective_heuristic_impact(p: SignalPayload) -> float:
    """`recurrence × success_rate × token_savings_potential`.

    success_rate ∈ [0, 1]; recurrence ≥ 1 (count); token_savings ∈ [0, ∞).
    """
    return _safe_mul(p.recurrence, p.success_rate, p.token_savings_potential)


def recurring_ritual_impact(p: SignalPayload) -> float:
    """Same formula as reflective_heuristic (procedure flavour)."""
    return reflective_heuristic_impact(p)


def stance_arc_impact(p: SignalPayload) -> float:
    """`polarity_distance × claim_stake`.

    polarity_distance ∈ [0, 1] (cosine distance between agent stances).
    claim_stake ∈ [0, ∞) (how much downstream work depends on the claim).
    """
    return _safe_mul(p.polarity_distance, p.claim_stake)


def low_stability_impact(p: SignalPayload) -> float:
    """`failure_count × cluster_recall_frequency`.

    failure_count is the number of attempts to claim/use this cluster
    that returned ambiguous or contradictory results.
    """
    return _safe_mul(p.failure_count, p.cluster_recall_frequency)


def stale_chain_impact(p: SignalPayload) -> float:
    """`age_days × recall_count × topic_currency_index`.

    topic_currency_index ∈ [0, 1] — how time-sensitive the topic is
    (e.g. rapidly-changing tech: 0.9; settled facts: 0.1).
    """
    return _safe_mul(p.age_days, p.recall_count, p.topic_currency_index)


_SIGNAL_IMPACT_FNS: dict[str, callable] = {
    SIGNAL_INFORMATION_GAP:      information_gap_impact,
    SIGNAL_CONTRADICTION_ARC:    contradiction_impact,
    SIGNAL_REFLECTIVE_HEURISTIC: reflective_heuristic_impact,
    SIGNAL_RECURRING_RITUAL:     recurring_ritual_impact,
    SIGNAL_STANCE_ARC:           stance_arc_impact,
    SIGNAL_LOW_STABILITY:        low_stability_impact,
    SIGNAL_STALE_SUPERSESSION:   stale_chain_impact,
}


def signal_impact(signal_type: str, payload: SignalPayload) -> float:
    """Dispatch helper — looks up the right formula by signal_type.

    Unknown signal_type → 0 (zero out the candidate; safer than guessing).
    """
    fn = _SIGNAL_IMPACT_FNS.get(signal_type)
    if fn is None:
        return 0.0
    return fn(payload)


# ── Objective alignment (per §3.7.2) ────────────────────────────────────

def objective_alignment(
    signal_type: str,
    constitution_objectives: dict[str, float],
    *,
    alignment_vectors: Optional[dict[str, dict[str, float]]] = None,
) -> float:
    """`dot(signal_alignment, constitution.objectives)`.

    Both maps key on objective name (e.g. "reduce_toil",
    "improve_security", "preserve_user_agency").  Missing keys in
    either side are treated as 0.0 in the dot product.

    `alignment_vectors` is the per-tenant override map (from the
    recalibration worker at `proposals.recalibration`); when None we
    fall back to `DEFAULT_ALIGNMENT_VECTORS`.  This lets the scoring
    layer stay pure-function while still being tunable per tenant.

    The result is bounded in [0, 1] when the input vectors are; in
    practice both sides should sum to ~1.0 for interpretable scores.
    """
    vectors = alignment_vectors if alignment_vectors is not None else DEFAULT_ALIGNMENT_VECTORS
    seed = vectors.get(signal_type, {})
    if not seed or not constitution_objectives:
        return 0.0
    out = 0.0
    for k, v in seed.items():
        weight = constitution_objectives.get(k, 0.0)
        try:
            out += float(v) * float(weight)
        except (TypeError, ValueError):
            continue
    return out


# ── Constraint gate (per §3.7.2) ────────────────────────────────────────

def constraint_gate(constraint_violations: list[str]) -> int:
    """Returns 0 if any constitution constraint is violated, else 1.

    A non-empty list means at least one constraint failed — the
    candidate gets zeroed out and won't surface.  Empty list means
    no constraints applied to this candidate.
    """
    return 0 if (constraint_violations and len(constraint_violations) > 0) else 1


# ── final_score composition (per §3.7.2) ────────────────────────────────

def final_score(
    *,
    signal_type: str,
    payload: SignalPayload,
    constitution_objectives: dict[str, float],
    constraint_violations: Optional[list[str]] = None,
    alignment_vectors: Optional[dict[str, dict[str, float]]] = None,
) -> tuple[float, dict[str, float]]:
    """Compute final_score plus a breakdown dict for audit/explainability.

    `alignment_vectors` is the per-tenant override (recalibration
    worker); None → use `DEFAULT_ALIGNMENT_VECTORS`.

    Returns:
        (final_score, breakdown) where breakdown contains:
          - signal_impact
          - objective_alignment
          - constraint_gate (0 or 1)
          - final_score
    """
    s_imp = signal_impact(signal_type, payload)
    obj_align = objective_alignment(
        signal_type, constitution_objectives,
        alignment_vectors=alignment_vectors,
    )
    gate = constraint_gate(constraint_violations or [])
    final = s_imp * obj_align * gate
    breakdown = {
        "signal_impact":       s_imp,
        "objective_alignment": obj_align,
        "constraint_gate":     float(gate),
        "final_score":         final,
    }
    return final, breakdown


__all__ = [
    "SIGNAL_INFORMATION_GAP",
    "SIGNAL_CONTRADICTION_ARC",
    "SIGNAL_REFLECTIVE_HEURISTIC",
    "SIGNAL_RECURRING_RITUAL",
    "SIGNAL_STANCE_ARC",
    "SIGNAL_LOW_STABILITY",
    "SIGNAL_STALE_SUPERSESSION",
    "CONFIRMED_SIGNAL_TYPES",
    "DEFERRED_SIGNAL_TYPES",
    "ALL_SIGNAL_TYPES",
    "DEFAULT_ALIGNMENT_VECTORS",
    "CONTRADICTION_SOURCE_SEVERITY",
    "SignalPayload",
    "signal_impact",
    "information_gap_impact",
    "contradiction_impact",
    "reflective_heuristic_impact",
    "recurring_ritual_impact",
    "stance_arc_impact",
    "low_stability_impact",
    "stale_chain_impact",
    "objective_alignment",
    "constraint_gate",
    "final_score",
]
