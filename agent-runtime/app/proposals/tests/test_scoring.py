"""Unit tests for proposals/scoring.py — pure-function impact scoring.

No Lance setup, no LLM calls — runs anywhere with stdlib + pytest.
"""

from __future__ import annotations

import math
import pytest

from app.proposals import scoring as s
from app.proposals.scoring import SignalPayload


# ── _safe_mul ────────────────────────────────────────────────────────────


def test_safe_mul_basic():
    assert s._safe_mul(2, 3) == 6.0
    assert s._safe_mul(1.5, 2.0, 4.0) == 12.0


def test_safe_mul_none_zeros():
    # None in ANY factor → 0 (multiplicative composition drops candidate)
    assert s._safe_mul(2, None) == 0.0
    assert s._safe_mul(None, 3, 5) == 0.0
    assert s._safe_mul(2, 3, None, 5) == 0.0


def test_safe_mul_string_zeros():
    # Bad types → 0 (defensive against payload mis-types)
    assert s._safe_mul("not a number", 3) == 0.0


# ── information_gap ──────────────────────────────────────────────────────


def test_information_gap_basic():
    p = SignalPayload(gap_breadth=0.8, downstream_pressure=3)
    assert s.information_gap_impact(p) == pytest.approx(2.4)


def test_information_gap_zero_when_breadth_missing():
    p = SignalPayload(downstream_pressure=3)
    assert s.information_gap_impact(p) == 0.0


def test_information_gap_zero_when_pressure_missing():
    p = SignalPayload(gap_breadth=0.8)
    assert s.information_gap_impact(p) == 0.0


# ── contradiction_arc ────────────────────────────────────────────────────


def test_contradiction_user_user_highest():
    p = SignalPayload(source_kind_a="USER", source_kind_b="USER",
                      claim_centrality=1.0)
    assert s.contradiction_impact(p) == 10.0


def test_contradiction_agent_agent_mid():
    p = SignalPayload(source_kind_a="AGENT", source_kind_b="AGENT",
                      claim_centrality=2.0)
    assert s.contradiction_impact(p) == 6.0  # 3 × 2


def test_contradiction_agent_inferred_lowest():
    p = SignalPayload(source_kind_a="AGENT", source_kind_b="INFERRED",
                      claim_centrality=2.0)
    assert s.contradiction_impact(p) == 2.0  # 1 × 2


def test_contradiction_symmetric_lookup():
    # USER × INFERRED should be the same in either order
    p_ab = SignalPayload(source_kind_a="USER", source_kind_b="INFERRED",
                         claim_centrality=1.0)
    p_ba = SignalPayload(source_kind_a="INFERRED", source_kind_b="USER",
                         claim_centrality=1.0)
    assert s.contradiction_impact(p_ab) == s.contradiction_impact(p_ba) == 5.0


def test_contradiction_missing_kinds_zero():
    p = SignalPayload(claim_centrality=1.0)
    assert s.contradiction_impact(p) == 0.0


def test_contradiction_unknown_pair_zero():
    p = SignalPayload(source_kind_a="BOGUS", source_kind_b="ALSO_BOGUS",
                      claim_centrality=1.0)
    assert s.contradiction_impact(p) == 0.0


# ── reflective_heuristic + recurring_ritual ──────────────────────────────


def test_reflective_basic():
    p = SignalPayload(recurrence=5, success_rate=0.8,
                      token_savings_potential=1.5)
    assert s.reflective_heuristic_impact(p) == pytest.approx(6.0)


def test_recurring_ritual_same_formula():
    p = SignalPayload(recurrence=5, success_rate=0.8,
                      token_savings_potential=1.5)
    assert s.recurring_ritual_impact(p) == s.reflective_heuristic_impact(p)


# ── stance_arc ──────────────────────────────────────────────────────────


def test_stance_arc_basic():
    p = SignalPayload(polarity_distance=0.7, claim_stake=4.0)
    assert s.stance_arc_impact(p) == pytest.approx(2.8)


# ── low_stability ───────────────────────────────────────────────────────


def test_low_stability_basic():
    p = SignalPayload(failure_count=3, cluster_recall_frequency=2.5)
    assert s.low_stability_impact(p) == pytest.approx(7.5)


# ── stale_chain ─────────────────────────────────────────────────────────


def test_stale_chain_basic():
    p = SignalPayload(age_days=30, recall_count=4, topic_currency_index=0.9)
    assert s.stale_chain_impact(p) == pytest.approx(108.0)


# ── signal_impact dispatcher ────────────────────────────────────────────


def test_signal_impact_dispatch():
    p = SignalPayload(gap_breadth=0.5, downstream_pressure=2)
    assert s.signal_impact(s.SIGNAL_INFORMATION_GAP, p) == 1.0


def test_signal_impact_unknown_signal_zero():
    p = SignalPayload(gap_breadth=0.5, downstream_pressure=2)
    assert s.signal_impact("unknown_signal_type", p) == 0.0


# ── objective_alignment ──────────────────────────────────────────────────


def test_objective_alignment_basic():
    # information_gap default = {reduce_toil:0.5, improve_security:0.3, preserve_user_agency:0.2}
    # constitution        =    {reduce_toil:0.6, improve_security:0.3, preserve_user_agency:0.1}
    # dot product = 0.5*0.6 + 0.3*0.3 + 0.2*0.1 = 0.30 + 0.09 + 0.02 = 0.41
    out = s.objective_alignment(
        s.SIGNAL_INFORMATION_GAP,
        {"reduce_toil": 0.6, "improve_security": 0.3, "preserve_user_agency": 0.1},
    )
    assert out == pytest.approx(0.41)


def test_objective_alignment_empty_constitution_zero():
    assert s.objective_alignment(s.SIGNAL_INFORMATION_GAP, {}) == 0.0


def test_objective_alignment_unknown_signal_zero():
    assert s.objective_alignment("bogus", {"reduce_toil": 1.0}) == 0.0


def test_objective_alignment_missing_objective_treated_as_zero():
    # Constitution has only `reduce_toil` — others contribute 0.
    out = s.objective_alignment(
        s.SIGNAL_INFORMATION_GAP, {"reduce_toil": 1.0},
    )
    assert out == pytest.approx(0.5)  # 0.5 * 1.0 + 0 + 0


# ── constraint_gate ──────────────────────────────────────────────────────


def test_constraint_gate_open_when_no_violations():
    assert s.constraint_gate([]) == 1


def test_constraint_gate_closed_when_any_violation():
    assert s.constraint_gate(["cost_ceiling_exceeded"]) == 0
    assert s.constraint_gate(["a", "b"]) == 0


# ── final_score composition (multiplicative) ─────────────────────────────


def test_final_score_basic():
    payload = SignalPayload(gap_breadth=0.8, downstream_pressure=3)
    constitution = {
        "reduce_toil":          0.6,
        "improve_security":     0.3,
        "preserve_user_agency": 0.1,
    }
    final, breakdown = s.final_score(
        signal_type=s.SIGNAL_INFORMATION_GAP,
        payload=payload,
        constitution_objectives=constitution,
        constraint_violations=[],
    )
    # signal_impact = 0.8 * 3 = 2.4
    # objective_alignment = 0.5*0.6 + 0.3*0.3 + 0.2*0.1 = 0.41
    # constraint_gate = 1
    # final = 2.4 * 0.41 * 1 = 0.984
    assert breakdown["signal_impact"] == pytest.approx(2.4)
    assert breakdown["objective_alignment"] == pytest.approx(0.41)
    assert breakdown["constraint_gate"] == 1.0
    assert final == pytest.approx(0.984)


def test_final_score_constraint_zeros_out():
    payload = SignalPayload(gap_breadth=1.0, downstream_pressure=10)
    constitution = {"reduce_toil": 1.0}
    final, breakdown = s.final_score(
        signal_type=s.SIGNAL_INFORMATION_GAP,
        payload=payload,
        constitution_objectives=constitution,
        constraint_violations=["cost_ceiling_exceeded"],
    )
    # Even with a perfect signal_impact + alignment, the closed gate zeros it.
    assert breakdown["constraint_gate"] == 0.0
    assert final == 0.0


def test_final_score_weak_factor_zeros_out():
    # Per spec §3.7.2: "any single weak factor zeroes out the candidate"
    # Missing payload field → signal_impact = 0 → final = 0
    payload = SignalPayload()  # all None
    final, breakdown = s.final_score(
        signal_type=s.SIGNAL_INFORMATION_GAP,
        payload=payload,
        constitution_objectives={"reduce_toil": 1.0},
        constraint_violations=[],
    )
    assert breakdown["signal_impact"] == 0.0
    assert final == 0.0


# ── enumerations sanity ─────────────────────────────────────────────────


def test_signal_taxonomy_disjoint():
    """confirmed and deferred sets should NOT overlap."""
    assert not (s.CONFIRMED_SIGNAL_TYPES & s.DEFERRED_SIGNAL_TYPES)


def test_signal_taxonomy_all_have_impact_fn():
    """Every signal_type must have an impact formula."""
    for st in s.ALL_SIGNAL_TYPES:
        assert st in s._SIGNAL_IMPACT_FNS, f"missing impact fn for {st}"


def test_signal_taxonomy_all_have_default_alignment():
    """Every signal_type must have a default alignment vector."""
    for st in s.ALL_SIGNAL_TYPES:
        assert st in s.DEFAULT_ALIGNMENT_VECTORS, (
            f"missing default alignment for {st}"
        )
