"""Phase 6.9 — Constitution evaluator_id required tests.

AC-39 — loader refuses entries missing evaluator_id (when strict)
AC-40 — every existing objective has a registered evaluator on load
AC-41 — evaluator output is deterministic: same input → same verdict
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.constitution.evaluators import (
    all_registered,
    has_evaluator,
    lookup,
    register_evaluator,
    validate_constitution_evaluators,
)
from app.constitution.loader import ConstitutionError, load_constitution


# ── AC-39 — loader refuses missing evaluator_id ─────────────────────────


def _write_constitution(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "CONSTITUTION.md"
    p.write_text(body, encoding="utf-8")
    return p


def test_ac39_loader_refuses_objective_missing_evaluator_id(tmp_path: Path):
    """Strict-mode load with an objective lacking evaluator_id → ConstitutionError."""
    p = _write_constitution(tmp_path, _CONSTITUTION_NO_EVALUATOR_ID)
    with pytest.raises(ConstitutionError, match=r"evaluator_id"):
        load_constitution(p, require_evaluator_id=True)


def test_ac39_loader_refuses_evaluator_id_not_in_registry(tmp_path: Path):
    """Strict-mode load with an evaluator_id that isn't registered →
    ConstitutionError listing the registered ones."""
    p = _write_constitution(tmp_path, _CONSTITUTION_BAD_EVALUATOR_ID)
    with pytest.raises(ConstitutionError, match=r"NOT in the\s+registry"):
        load_constitution(p, require_evaluator_id=True)


def test_ac39_loader_accepts_valid_evaluator_ids(tmp_path: Path):
    """Strict-mode load with valid evaluator_ids → passes."""
    p = _write_constitution(tmp_path, _CONSTITUTION_VALID)
    cfg = load_constitution(p, require_evaluator_id=True)
    assert cfg["objectives"]
    assert any(o.get("evaluator_id") for o in cfg["objectives"])


def test_ac39_legacy_mode_does_not_enforce(tmp_path: Path):
    """Default (legacy) mode loads constitutions without evaluator_id."""
    p = _write_constitution(tmp_path, _CONSTITUTION_NO_EVALUATOR_ID)
    # Should NOT raise.
    cfg = load_constitution(p, require_evaluator_id=False)
    assert cfg["objectives"]


# ── AC-40 — every existing objective has a registered evaluator ────────


def test_ac40_validator_returns_no_errors_when_all_have_evaluator():
    objectives = [
        {"id": "o1", "weight": 1.0, "evaluator_id": "approval_rate_above_threshold"},
    ]
    constraints = [
        {"id": "c1", "rule": "x", "evaluator_id": "cost_ceiling_not_exceeded"},
    ]
    errs = validate_constitution_evaluators(objectives, constraints)
    assert errs == []


def test_ac40_validator_lists_missing_evaluator_id():
    objectives = [
        {"id": "o1", "weight": 1.0},  # no evaluator_id
    ]
    errs = validate_constitution_evaluators(objectives, [])
    assert len(errs) == 1
    assert "evaluator_id" in errs[0]


def test_ac40_validator_lists_unregistered_evaluator_id():
    objectives = [
        {"id": "o1", "weight": 1.0, "evaluator_id": "does_not_exist"},
    ]
    errs = validate_constitution_evaluators(objectives, [])
    assert len(errs) == 1
    assert "does_not_exist" in errs[0]


def test_ac40_legacy_exempt_downgrades_errors():
    objectives = [{"id": "o1", "weight": 1.0}]
    errs = validate_constitution_evaluators(
        objectives, [], legacy_exempt=True,
    )
    assert errs == []  # downgraded to warnings


def test_ac40_initial_registry_has_at_least_five_evaluators():
    """Phase 6.9 ships 5+ evaluators wired to existing objectives."""
    registered = all_registered()
    assert len(registered) >= 5, (
        f"Phase 6.9 expects ≥5 initial evaluators; got {sorted(registered)}"
    )


# ── AC-41 — evaluator output is deterministic ───────────────────────────


def test_ac41_approval_rate_deterministic():
    ev = lookup("approval_rate_above_threshold")
    assert ev is not None
    ctx = {
        "proposal_outcomes": [
            {"status": "approved"},
            {"status": "approved"},
            {"status": "shelved"},
        ],
        "threshold": 0.5,
    }
    r1 = ev(ctx)
    r2 = ev(ctx)
    r3 = ev(ctx)
    assert r1.verdict == r2.verdict == r3.verdict
    assert r1.details == r2.details == r3.details
    assert r1.reason == r2.reason == r3.reason
    # Sanity on the math: 2/3 = 0.667 ≥ 0.5 → pass.
    assert r1.verdict is True


def test_ac41_cost_ceiling_deterministic():
    ev = lookup("cost_ceiling_not_exceeded")
    assert ev is not None
    ctx = {"task_costs": [1.0, 2.5, 3.0], "ceiling_usd": 10.0}
    rs = [ev(ctx) for _ in range(5)]
    assert all(r.verdict == rs[0].verdict for r in rs)
    assert all(r.details == rs[0].details for r in rs)
    assert rs[0].verdict is True
    assert rs[0].details["total_usd"] == 6.5


def test_ac41_cost_ceiling_over_limit():
    ev = lookup("cost_ceiling_not_exceeded")
    ctx = {"task_costs": [5.0, 5.0, 5.0], "ceiling_usd": 10.0}
    r = ev(ctx)
    assert r.verdict is False
    assert r.details["total_usd"] == 15.0


def test_ac41_source_kind_trust_deterministic():
    ev = lookup("source_kind_trust_minimum")
    ctx = {"source_kinds": ["USER", "TOOL_RESULT", "WEB_FETCHED"]}
    r1 = ev(ctx)
    r2 = ev(ctx)
    assert r1.verdict == r2.verdict
    assert r1.details == r2.details
    # WEB_FETCHED not in default allowed set → reject.
    assert r1.verdict is False
    assert "WEB_FETCHED" in r1.details["rejected"]


def test_ac41_fairness_factor_deterministic():
    ev = lookup("fairness_factor_below_max")
    ctx = {
        "per_assignee_approvals": {"researcher": 5, "builder": 2, "critic": 1},
        "max_share": 0.50,
    }
    r1 = ev(ctx)
    r2 = ev(ctx)
    assert r1.verdict == r2.verdict
    # researcher = 5/8 = 0.625 > 0.50 → fail.
    assert r1.verdict is False


def test_ac41_user_user_emergency_lane():
    ev = lookup("user_user_emergency_lane_qualifies")
    ctx = {
        "claim_pair": {
            "claim_a": {"source_kind": "USER", "polarity": True, "triple_hash": "h1"},
            "claim_b": {"source_kind": "USER", "polarity": False, "triple_hash": "h1"},
        },
    }
    r1 = ev(ctx)
    r2 = ev(ctx)
    assert r1.verdict == r2.verdict is True


def test_ac41_no_evaluator_uses_time_or_random():
    """Sanity: invoke each evaluator twice with identical ctx and assert
    identical output.  Catches an evaluator that secretly imports time
    or random and varies per-call."""
    test_ctx_per_evaluator = {
        "approval_rate_above_threshold": {"proposal_outcomes": [], "threshold": 0.2},
        "cost_ceiling_not_exceeded":     {"task_costs": [], "ceiling_usd": 1.0},
        "source_kind_trust_minimum":     {"source_kinds": []},
        "fairness_factor_below_max":     {"per_assignee_approvals": {}, "max_share": 0.5},
        "user_user_emergency_lane_qualifies": {"claim_pair": {}},
    }
    for evaluator_id, ctx in test_ctx_per_evaluator.items():
        ev = lookup(evaluator_id)
        assert ev is not None, f"missing evaluator: {evaluator_id}"
        a = ev(ctx)
        b = ev(ctx)
        assert (a.verdict, a.details, a.reason) == (b.verdict, b.details, b.reason), (
            f"evaluator {evaluator_id!r} non-deterministic"
        )


# ── Fixture constitution texts ─────────────────────────────────────────


_CONSTITUTION_NO_EVALUATOR_ID = """\
# Veilguard Constitution

constitution_version: 1

## Objectives

```
- id: o1
  weight: 0.5
  description: First objective
- id: o2
  weight: 0.5
  description: Second objective
```

## Constraints

```
- id: c1
  rule: do_no_harm
```
"""

_CONSTITUTION_BAD_EVALUATOR_ID = """\
# Veilguard Constitution

constitution_version: 1

## Objectives

```
- id: o1
  weight: 1.0
  description: First objective
  evaluator_id: this_evaluator_does_not_exist
```

## Constraints

```
- id: c1
  rule: do_no_harm
  evaluator_id: cost_ceiling_not_exceeded
```
"""

_CONSTITUTION_VALID = """\
# Veilguard Constitution

constitution_version: 1

## Objectives

```
- id: o1
  weight: 0.5
  description: First objective
  evaluator_id: approval_rate_above_threshold
- id: o2
  weight: 0.5
  description: Second objective
  evaluator_id: cost_ceiling_not_exceeded
```

## Constraints

```
- id: c1
  rule: do_no_harm
  evaluator_id: source_kind_trust_minimum
```
"""
