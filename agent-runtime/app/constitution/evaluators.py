"""Phase 6.9 — Constitution evaluator registry.

Every constitution.json entry MUST point to a deterministic evaluator
in this registry via its `evaluator_id` field.  Entries without an
evaluator are aspiration, not policy — the loader refuses them.

An evaluator is a pure function `(context) -> EvaluatorResult` where
`context` is whatever the scorer collects (proposal payload, signal
metadata, recent outcomes for a tenant, etc.).  Result is True/False +
optional details.

Deterministic guarantee (AC-41): same context → same verdict.  No
time-of-day dependence (use fixed timestamps from context, not
`time.time()`); no external randomness; no network calls.

Phase 6.9 ships 5 evaluators wired to the existing constitution
objectives.  Phase 7 will expand the registry to cover every objective;
any objective lacking an evaluator is refused at load.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger("agent-runtime.constitution.evaluators")


@dataclass
class EvaluatorResult:
    """3-state result with details — same shape philosophy as
    `app/acceptance/executors.py:CheckResult`.
    """
    verdict: bool                # True = constraint satisfied / objective met
    details: dict[str, Any]
    reason: str = ""


Evaluator = Callable[[dict[str, Any]], EvaluatorResult]

# Registry: evaluator_id → callable.  Mutating from outside is allowed
# only via `register_evaluator()` so the registry stays auditable.
_REGISTRY: dict[str, Evaluator] = {}


def register_evaluator(evaluator_id: str) -> Callable[[Evaluator], Evaluator]:
    def deco(fn: Evaluator) -> Evaluator:
        if evaluator_id in _REGISTRY:
            raise ValueError(
                f"evaluator_id {evaluator_id!r} already registered"
            )
        _REGISTRY[evaluator_id] = fn
        return fn
    return deco


def lookup(evaluator_id: str) -> Evaluator | None:
    return _REGISTRY.get(evaluator_id)


def has_evaluator(evaluator_id: str) -> bool:
    return evaluator_id in _REGISTRY


def all_registered() -> frozenset[str]:
    return frozenset(_REGISTRY.keys())


# ── Initial evaluators (Phase 6.9 ships ~5; Phase 7 expands) ───────────


@register_evaluator("approval_rate_above_threshold")
def _approval_rate(ctx: dict[str, Any]) -> EvaluatorResult:
    """Last-N-days proposal approval rate ≥ threshold (default 0.2).

    Used by objectives that gate "are we generating useful proposals."
    Reads `ctx['proposal_outcomes']` = list of {status, approved_at, ...}
    and `ctx['threshold']` (defaults 0.2).
    """
    outcomes = ctx.get("proposal_outcomes") or []
    threshold = float(ctx.get("threshold", 0.2))
    if not outcomes:
        return EvaluatorResult(
            verdict=True,  # no data → don't block
            details={"outcomes": 0, "threshold": threshold},
            reason="no outcomes yet — pass by default",
        )
    approved = sum(
        1 for o in outcomes
        if (o.get("status") or "").lower() == "approved"
    )
    rate = approved / len(outcomes)
    return EvaluatorResult(
        verdict=rate >= threshold,
        details={"approved": approved, "total": len(outcomes), "rate": rate, "threshold": threshold},
        reason=f"approval_rate={rate:.3f} {'≥' if rate >= threshold else '<'} threshold={threshold}",
    )


@register_evaluator("cost_ceiling_not_exceeded")
def _cost_ceiling(ctx: dict[str, Any]) -> EvaluatorResult:
    """Sum of `cost_usd` over recent tasks ≤ ceiling.

    Reads `ctx['task_costs']` = list of floats and
    `ctx['ceiling_usd']` (required).
    """
    costs = ctx.get("task_costs") or []
    ceiling = ctx.get("ceiling_usd")
    if ceiling is None:
        return EvaluatorResult(
            verdict=False,
            details={"ceiling_usd": None},
            reason="ctx.ceiling_usd required",
        )
    total = sum(float(c) for c in costs)
    return EvaluatorResult(
        verdict=total <= float(ceiling),
        details={"total_usd": total, "ceiling_usd": ceiling, "n_tasks": len(costs)},
        reason=f"total=${total:.2f} {'≤' if total <= ceiling else '>'} ceiling=${ceiling:.2f}",
    )


@register_evaluator("source_kind_trust_minimum")
def _source_kind_trust(ctx: dict[str, Any]) -> EvaluatorResult:
    """Reject proposals derived from low-trust source_kinds.

    Reads `ctx['source_kinds']` (a list of source_kind strings) and
    `ctx['allowed']` (a set/list of permitted kinds).
    """
    seen = set(ctx.get("source_kinds") or [])
    allowed = set(ctx.get("allowed") or [
        "USER", "TOOL_RESULT", "AGENT_RESEARCHER",
        "AGENT_BUILDER", "AGENT_CRITIC", "INFERRED",
    ])
    rejected = seen - allowed
    return EvaluatorResult(
        verdict=not rejected,
        details={"seen": sorted(seen), "allowed": sorted(allowed), "rejected": sorted(rejected)},
        reason=(
            "all source_kinds allowed" if not rejected
            else f"rejected source_kinds: {sorted(rejected)}"
        ),
    )


@register_evaluator("fairness_factor_below_max")
def _fairness_factor(ctx: dict[str, Any]) -> EvaluatorResult:
    """Per-assignee approval share ≤ max_share (default 0.70).

    Reads `ctx['per_assignee_approvals']` = dict {assignee: approved_count}.
    """
    per_assignee = ctx.get("per_assignee_approvals") or {}
    max_share = float(ctx.get("max_share", 0.70))
    total = sum(per_assignee.values())
    if total == 0:
        return EvaluatorResult(
            verdict=True,
            details={"total": 0, "max_share": max_share},
            reason="no approvals yet",
        )
    over = {a: c / total for a, c in per_assignee.items() if c / total > max_share}
    return EvaluatorResult(
        verdict=not over,
        details={"shares": {a: c / total for a, c in per_assignee.items()}, "max_share": max_share},
        reason=(
            "fairness OK" if not over
            else f"assignees over max_share: {over}"
        ),
    )


@register_evaluator("user_user_emergency_lane_qualifies")
def _user_user_emergency_lane(ctx: dict[str, Any]) -> EvaluatorResult:
    """Emergency-lane bypass requires BOTH claims to have source_kind=USER
    and incompatible polarity on the same canonical_triple_hash.

    Reads `ctx['claim_pair']` = {claim_a: {source_kind, polarity, triple_hash}, claim_b: ...}.
    """
    pair = ctx.get("claim_pair") or {}
    a = pair.get("claim_a") or {}
    b = pair.get("claim_b") or {}
    if not (a and b):
        return EvaluatorResult(
            verdict=False,
            details={"have_pair": False},
            reason="claim_pair missing claim_a or claim_b",
        )
    both_user = (a.get("source_kind") == "USER" and b.get("source_kind") == "USER")
    same_hash = (
        a.get("triple_hash")
        and a.get("triple_hash") == b.get("triple_hash")
    )
    incompatible = a.get("polarity") != b.get("polarity") and a.get("polarity") is not None
    qualifies = both_user and same_hash and incompatible
    return EvaluatorResult(
        verdict=qualifies,
        details={
            "both_user": both_user,
            "same_hash": same_hash,
            "incompatible_polarity": incompatible,
        },
        reason=(
            "emergency-lane qualified" if qualifies
            else "emergency-lane criteria not all met"
        ),
    )


# ── Helpers for the loader ────────────────────────────────────────────


def validate_constitution_evaluators(
    objectives: list[dict[str, Any]],
    constraints: list[dict[str, Any]],
    *,
    legacy_exempt: bool = False,
) -> list[str]:
    """Return a list of error strings if any entry violates the
    evaluator_id requirement.  Empty list = OK.

    `legacy_exempt=True` (Phase 6.9 grace period for existing
    CONSTITUTION.md files) downgrades errors to warnings and returns [].
    """
    errors: list[str] = []
    for entry, kind in [(o, "objective") for o in objectives] + [
        (c, "constraint") for c in constraints
    ]:
        evaluator_id = entry.get("evaluator_id")
        if not evaluator_id:
            errors.append(
                f"{kind} {entry.get('id')!r} has no `evaluator_id` — "
                "entries without an evaluator are aspiration, not policy. "
                "Add an evaluator_id referencing a registered evaluator."
            )
            continue
        if not has_evaluator(evaluator_id):
            errors.append(
                f"{kind} {entry.get('id')!r} references "
                f"evaluator_id={evaluator_id!r} which is NOT in the "
                f"registry. Registered: {sorted(all_registered())}."
            )
    if legacy_exempt and errors:
        for e in errors:
            logger.warning(f"[constitution] legacy_exempt: {e}")
        return []
    return errors


__all__ = [
    "Evaluator",
    "EvaluatorResult",
    "register_evaluator",
    "lookup",
    "has_evaluator",
    "all_registered",
    "validate_constitution_evaluators",
]
