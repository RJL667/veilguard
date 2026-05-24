"""Unit tests for constitution.loader + scorer."""

from pathlib import Path

import pytest

from app.constitution.loader import load_constitution, ConstitutionError
from app.constitution.scorer import (
    DEFAULT_ALIGNMENT_VECTORS,
    objective_alignment,
    default_alignment_for_signal,
    constraint_gate,
    final_score,
)


# ── Loader ─────────────────────────────────────────────────────────────


_SAMPLE_CONSTITUTION = """\
# Veilguard Constitution

## Objectives

```
- id: reduce_toil
  weight: 0.40
  description: Eliminate repetitive work.

- id: improve_security
  weight: 0.30
  description: Strengthen the user's security.

- id: preserve_user_agency
  weight: 0.30
  description: Surface decisions, don't make them silently.
```

## Constraints

```
- id: no_hidden_automation
  rule: Tasks above $0.50 must surface to user.

- id: cost_ceiling_per_task
  rule: No single task may exceed $5.
```

## Metrics

```
- id: time_saved
  unit: minutes_per_week
```

constitution_version: 1
"""


class TestLoader:
    def test_loads_sample(self, tmp_path):
        path = tmp_path / "CONSTITUTION.md"
        path.write_text(_SAMPLE_CONSTITUTION, encoding="utf-8")
        result = load_constitution(path)
        assert len(result["objectives"]) == 3
        assert len(result["constraints"]) == 2
        assert result["version"] == 1

    def test_missing_file_raises(self):
        with pytest.raises(ConstitutionError):
            load_constitution(Path("/nonexistent.md"))

    def test_no_objectives_raises(self, tmp_path):
        path = tmp_path / "C.md"
        path.write_text("## Constraints\n```\n- id: x\n  rule: y\n```\n")
        with pytest.raises(ConstitutionError):
            load_constitution(path)

    def test_objective_missing_weight_raises(self, tmp_path):
        path = tmp_path / "C.md"
        path.write_text(
            "## Objectives\n```\n- id: reduce_toil\n```\n"
        )
        with pytest.raises(ConstitutionError):
            load_constitution(path)

    def test_crlf_in_source_handled(self, tmp_path):
        path = tmp_path / "C.md"
        path.write_bytes(_SAMPLE_CONSTITUTION.replace("\n", "\r\n").encode("utf-8"))
        result = load_constitution(path)
        assert len(result["objectives"]) == 3


# ── Scorer ─────────────────────────────────────────────────────────────


class TestObjectiveAlignment:
    def test_dot_product(self):
        delta = {"reduce_toil": 0.5, "improve_security": 0.5}
        objs = [
            {"id": "reduce_toil", "weight": 0.4},
            {"id": "improve_security", "weight": 0.3},
        ]
        score = objective_alignment(delta, objs)
        # 0.5*0.4 + 0.5*0.3 = 0.35
        assert score == pytest.approx(0.35)

    def test_empty_delta_zero(self):
        assert objective_alignment({}, [{"id": "x", "weight": 1.0}]) == 0.0

    def test_missing_objective_ignored(self):
        delta = {"unknown_objective": 1.0}
        objs = [{"id": "reduce_toil", "weight": 0.4}]
        assert objective_alignment(delta, objs) == 0.0


class TestDefaultAlignmentForSignal:
    def test_known_signal(self):
        objs = [{"id": "reduce_toil", "weight": 0.4}]
        score = default_alignment_for_signal("information_gap", objs)
        # information_gap default has reduce_toil=0.5
        assert score == pytest.approx(0.5 * 0.4)

    def test_unknown_signal_zero(self):
        objs = [{"id": "reduce_toil", "weight": 0.4}]
        assert default_alignment_for_signal("nonexistent_signal", objs) == 0.0


class TestConstraintGate:
    def test_under_cost_threshold_passes(self):
        proposal = {"estimated_cost_usd": 0.30}
        constraints = [{"id": "no_hidden_automation"}]
        gate, viols = constraint_gate(proposal, constraints)
        assert gate == 1
        assert viols == []

    def test_over_cost_threshold_blocks(self):
        proposal = {"estimated_cost_usd": 1.00, "user_approved": False}
        constraints = [{"id": "no_hidden_automation"}]
        gate, viols = constraint_gate(proposal, constraints)
        assert gate == 0
        assert "no_hidden_automation" in viols

    def test_user_approval_bypasses_no_hidden_automation(self):
        proposal = {"estimated_cost_usd": 1.00, "user_approved": True}
        constraints = [{"id": "no_hidden_automation"}]
        gate, viols = constraint_gate(proposal, constraints)
        assert gate == 1

    def test_cost_ceiling_per_task(self):
        proposal = {"estimated_cost_usd": 10.00, "user_approved": True}
        constraints = [{"id": "cost_ceiling_per_task"}]
        gate, viols = constraint_gate(proposal, constraints)
        assert gate == 0
        assert "cost_ceiling_per_task" in viols

    def test_preserve_provenance_blocks_inferred_to_blackboard(self):
        proposal = {
            "target": "org_blackboard",
            "source_kind": "INFERRED",
            "critic_promoted": False,
        }
        constraints = [{"id": "preserve_provenance"}]
        gate, viols = constraint_gate(proposal, constraints)
        assert gate == 0

    def test_critic_promoted_inferred_allowed(self):
        proposal = {
            "target": "org_blackboard",
            "source_kind": "INFERRED",
            "critic_promoted": True,
        }
        constraints = [{"id": "preserve_provenance"}]
        gate, viols = constraint_gate(proposal, constraints)
        assert gate == 1


class TestFinalScore:
    def test_composition(self):
        proposal = {"estimated_cost_usd": 0.10}
        constitution = {
            "objectives": [{"id": "reduce_toil", "weight": 0.4}],
            "constraints": [{"id": "no_hidden_automation"}],
        }
        score, viols = final_score(
            signal_impact=10.0,
            signal_type="information_gap",
            proposal=proposal,
            constitution=constitution,
        )
        # 10 * (0.5*0.4) * 1 = 2.0
        assert score == pytest.approx(2.0)
        assert viols == []

    def test_constraint_violation_zeroes_score(self):
        proposal = {"estimated_cost_usd": 10.0, "user_approved": True}
        constitution = {
            "objectives": [{"id": "reduce_toil", "weight": 0.4}],
            "constraints": [{"id": "cost_ceiling_per_task"}],
        }
        score, viols = final_score(
            signal_impact=10.0,
            signal_type="information_gap",
            proposal=proposal,
            constitution=constitution,
        )
        assert score == 0.0
        assert "cost_ceiling_per_task" in viols

    def test_no_constitution_passthrough(self):
        score, viols = final_score(
            signal_impact=5.0,
            signal_type="information_gap",
            proposal={},
            constitution=None,
        )
        assert score == 5.0
        assert viols == []
