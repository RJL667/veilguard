"""Unit tests for proposals/rank.py — Director Haiku rank pass.

All tests inject a mock ``adapter_call`` — no real LLM traffic.
"""

import pytest

from app.proposals.rank import (
    rank_candidates,
    VALID_ASSIGNEES,
    DEFAULT_RANK_MODEL,
)


CONSTITUTION = {
    "reduce_toil":          0.4,
    "improve_security":     0.3,
    "preserve_user_agency": 0.3,
}


def _candidate(
    signal_type="information_gap",
    node_ids=(101, 102),
    impact=2.5,
    brief="Research POPIA Section X.  Cite sources.",
    assignee="researcher",
    align=0.36,
):
    return {
        "signal_type":      signal_type,
        "signal_node_ids":  list(node_ids),
        "impact_score":     impact,
        "default_brief":    brief,
        "default_assignee": assignee,
        "default_align":    align,
        "payload_summary":  f"summary for {signal_type}",
    }


# ── basic happy path ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rank_empty_returns_empty():
    """No candidates → no LLM call, empty output."""
    call_count = 0
    async def adapter(**kwargs):
        nonlocal call_count
        call_count += 1
        return {"ranked": []}
    out = await rank_candidates(
        candidates=[],
        constitution_objectives=CONSTITUTION,
        adapter_call=adapter,
    )
    assert out == []
    assert call_count == 0


@pytest.mark.asyncio
async def test_rank_one_candidate_refined():
    c = _candidate()
    async def adapter(**kwargs):
        # Validate inputs the rank module sent us
        assert "rank_proposals" == kwargs["tools"][0]["name"]
        assert "Veilguard" in kwargs["system_prompt"]
        return {"ranked": [{
            "signal_node_ids":             [101, 102],
            "refined_brief":               "Research POPIA s.14 retention rules.  Cite the Information Regulator's 2023 enforcement notices.",
            "refined_assignee":            "researcher",
            "refined_objective_alignment": 0.48,
            "rationale":                   "Retention gaps map directly to improve_security objective.",
            "drop":                        False,
        }]}
    out = await rank_candidates(
        candidates=[c],
        constitution_objectives=CONSTITUTION,
        adapter_call=adapter,
    )
    assert len(out) == 1
    r = out[0]
    assert r["refined_brief"].startswith("Research POPIA")
    assert r["refined_assignee"] == "researcher"
    assert r["refined_objective_alignment"] == pytest.approx(0.48)
    assert r["drop"] is False
    assert "improve_security" in r["rationale"]


# ── validation: invalid assignee falls back to default ─────────────────


@pytest.mark.asyncio
async def test_invalid_assignee_falls_back_to_default():
    c = _candidate(assignee="critic-prose")
    async def adapter(**kwargs):
        return {"ranked": [{
            "signal_node_ids":             c["signal_node_ids"],
            "refined_brief":               "fine",
            "refined_assignee":            "totally-invented-agent-bot",  # LLM hallucination
            "refined_objective_alignment": 0.5,
            "rationale":                   "...",
            "drop":                        False,
        }]}
    out = await rank_candidates(
        candidates=[c],
        constitution_objectives=CONSTITUTION,
        adapter_call=adapter,
    )
    assert out[0]["refined_assignee"] == "critic-prose"  # fell back to default


@pytest.mark.asyncio
async def test_alignment_clipped_to_unit_interval():
    c = _candidate()
    async def adapter(**kwargs):
        return {"ranked": [{
            "signal_node_ids":             c["signal_node_ids"],
            "refined_brief":               "x",
            "refined_assignee":            "researcher",
            "refined_objective_alignment": 1.7,  # out of range
            "rationale":                   "",
            "drop":                        False,
        }, {  # second test of clipping
            "signal_node_ids":             [999],
            "refined_brief":               "x",
            "refined_assignee":            "researcher",
            "refined_objective_alignment": -0.5,  # negative
            "rationale":                   "",
            "drop":                        False,
        }]}
    c2 = _candidate(node_ids=[999])
    out = await rank_candidates(
        candidates=[c, c2],
        constitution_objectives=CONSTITUTION,
        adapter_call=adapter,
    )
    assert out[0]["refined_objective_alignment"] == 1.0   # clipped high
    assert out[1]["refined_objective_alignment"] == 0.0   # clipped low


# ── missing LLM response → passthrough preserves the candidate ─────────


@pytest.mark.asyncio
async def test_missing_response_passes_through_default():
    """LLM didn't return an entry for one of our candidates →
    that candidate gets its defaults back (NOT dropped silently)."""
    c1 = _candidate(node_ids=[1])
    c2 = _candidate(node_ids=[2], brief="default brief for #2")
    async def adapter(**kwargs):
        return {"ranked": [{   # only one entry — #2 missing
            "signal_node_ids":             [1],
            "refined_brief":               "refined #1",
            "refined_assignee":            "researcher",
            "refined_objective_alignment": 0.3,
            "rationale":                   "...",
            "drop":                        False,
        }]}
    out = await rank_candidates(
        candidates=[c1, c2],
        constitution_objectives=CONSTITUTION,
        adapter_call=adapter,
    )
    assert len(out) == 2
    # First refined
    assert out[0]["refined_brief"] == "refined #1"
    # Second falls back to its default brief (NOT dropped)
    assert out[1]["refined_brief"] == "default brief for #2"
    assert out[1]["drop"] is False


@pytest.mark.asyncio
async def test_adapter_exception_uses_defaults_for_all():
    """If the LLM call raises, every candidate gets its defaults
    rather than disappearing."""
    c1 = _candidate(node_ids=[1])
    c2 = _candidate(node_ids=[2])
    async def adapter(**kwargs):
        raise RuntimeError("Anthropic 503")
    out = await rank_candidates(
        candidates=[c1, c2],
        constitution_objectives=CONSTITUTION,
        adapter_call=adapter,
    )
    assert len(out) == 2
    for r in out:
        assert "(LLM rank pass unavailable" in r["rationale"]
        assert r["drop"] is False


@pytest.mark.asyncio
async def test_non_list_ranked_uses_defaults():
    """Bad shape from LLM → fall back to defaults, don't crash."""
    c = _candidate()
    async def adapter(**kwargs):
        return {"ranked": "not a list"}
    out = await rank_candidates(
        candidates=[c],
        constitution_objectives=CONSTITUTION,
        adapter_call=adapter,
    )
    assert out[0]["refined_brief"] == c["default_brief"]


# ── drop signal honoured ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_drop_flag_propagated():
    c = _candidate()
    async def adapter(**kwargs):
        return {"ranked": [{
            "signal_node_ids":             c["signal_node_ids"],
            "refined_brief":               "skip me",
            "refined_assignee":            "researcher",
            "refined_objective_alignment": 0.0,
            "rationale":                   "Not aligned with current objectives",
            "drop":                        True,
        }]}
    out = await rank_candidates(
        candidates=[c],
        constitution_objectives=CONSTITUTION,
        adapter_call=adapter,
    )
    assert out[0]["drop"] is True


# ── system prompt includes constitution ───────────────────────────────


@pytest.mark.asyncio
async def test_system_prompt_contains_constitution_objectives():
    """Sanity: the system prompt the LLM sees actually lists each
    objective + weight.  Sorted by weight DESC."""
    captured = {}
    async def adapter(**kwargs):
        captured["sys"] = kwargs["system_prompt"]
        return {"ranked": []}
    await rank_candidates(
        candidates=[],   # empty triggers early return — use 1 candidate
        constitution_objectives=CONSTITUTION,
        adapter_call=adapter,
    )
    # Empty candidates → adapter never called.  Test with 1 instead.
    c = _candidate()
    await rank_candidates(
        candidates=[c],
        constitution_objectives=CONSTITUTION,
        adapter_call=adapter,
    )
    sys = captured["sys"]
    assert "reduce_toil" in sys
    assert "improve_security" in sys
    assert "preserve_user_agency" in sys
    # Weight values inlined
    assert "0.40" in sys or "0.4" in sys
    # Valid assignees enumerated
    assert "researcher" in sys
    assert "critic-prose" in sys


# ── model default sanity ───────────────────────────────────────────────


def test_default_model_is_haiku():
    assert "haiku" in DEFAULT_RANK_MODEL.lower()


def test_valid_assignees_includes_all_personas():
    for a in ("researcher", "builder", "critic-claim", "critic-prose"):
        assert a in VALID_ASSIGNEES
