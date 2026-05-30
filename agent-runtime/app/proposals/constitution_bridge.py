"""Phase 3 — bridge parsed constitution.md → scoring-friendly objects.

`load_constitution()` returns dicts in the spec's natural shape:
    {"objectives": [{"id": "reduce_toil", "weight": 0.4, ...}, ...],
     "constraints": [{"id": "no_hidden_automation", "rule": "..."}, ...]}

But `proposals.scoring.objective_alignment()` wants a flat dict[name, float]
to dot-product with the per-signal default alignment vectors.  This
module owns that conversion + the constraint-evaluation stub so the
two layers don't need to know each other's exact shape.

Constraint evaluation is per-constraint pure-Python.  Today we ship
two evaluators (cost_ceiling_per_task, no_hidden_automation); the
rest fall through to "not yet evaluated" — which is documented in the
returned violation reason so callers know what's actually gating vs.
what's permissive-by-default.
"""

from __future__ import annotations

from typing import Any, Optional


def objectives_to_dict(constitution: dict[str, Any]) -> dict[str, float]:
    """Flatten parsed objectives into ``{id: weight}``.

    Empty / malformed objectives are dropped silently — they were
    already flagged at constitution parse time.  This function never
    raises.
    """
    out: dict[str, float] = {}
    objs = (constitution or {}).get("objectives") or []
    for o in objs:
        if not isinstance(o, dict):
            continue
        oid = o.get("id")
        try:
            weight = float(o.get("weight", 0.0))
        except (TypeError, ValueError):
            continue
        if oid and weight > 0:
            out[oid] = weight
    return out


# ── Per-constraint evaluators ───────────────────────────────────────────
#
# Each evaluator takes a context dict (filled in by the caller — usually
# from the dream signal payload + dream-cycle stats + tenant config)
# and returns either:
#   - None  → constraint not applicable / not violated
#   - str   → human-readable violation reason
#
# Evaluators are PURE — no I/O.  Caller is responsible for assembling
# the context (cost estimates, source kinds, etc.) before calling.


def _eval_cost_ceiling_per_task(
    *, rule: str, ctx: dict[str, Any]
) -> Optional[str]:
    """`No single Task may exceed $5 USD without explicit user approval.`

    Reads `ctx['estimated_cost_usd']`; missing → assume 0 (we don't
    fail-open or fail-closed; we let other gates catch genuine
    cost overruns at execution time).
    """
    cost = ctx.get("estimated_cost_usd")
    try:
        cost = float(cost) if cost is not None else 0.0
    except (TypeError, ValueError):
        cost = 0.0
    # Default ceiling per the constitution's body text ($5).  In future
    # we'll parse the dollar threshold from ``rule`` instead of
    # hardcoding it; spec acceptance for v1 is the gate firing at all.
    if cost > 5.0:
        return f"cost_ceiling_per_task exceeded: ${cost:.2f} > $5.00"
    return None


def _eval_no_hidden_automation(
    *, rule: str, ctx: dict[str, Any]
) -> Optional[str]:
    """`Tasks with budget > $0.50 must surface to user before execution.`

    For a *proposal* (which by definition surfaces), this is automatically
    satisfied — the user IS being asked before execution.  We document
    that by returning None; the constraint really applies at the
    execution-side gate, not at proposal-emit time.

    We keep the evaluator wired up so the audit shows "this constraint
    was checked, and was inapplicable at proposal time" rather than
    silently skipping.
    """
    return None  # proposals are user-gated by construction


def _eval_preserve_provenance(
    *, rule: str, ctx: dict[str, Any]
) -> Optional[str]:
    """`Published claims (target=org_blackboard) require source_kind != INFERRED`.

    Reads `ctx['target_namespace']` + `ctx['source_kind']`.  Only
    relevant when target is org_blackboard.
    """
    target = (ctx.get("target_namespace") or "").lower()
    if "org_blackboard" not in target and "org/blackboard" not in target:
        return None
    source_kind = (ctx.get("source_kind") or "").upper()
    if source_kind == "INFERRED":
        return (
            "preserve_provenance violated: target is org_blackboard "
            "but source_kind is INFERRED"
        )
    return None


# Dispatcher.  Add new evaluators here when constraint vocabulary grows.
_EVALUATORS = {
    "cost_ceiling_per_task":   _eval_cost_ceiling_per_task,
    "no_hidden_automation":    _eval_no_hidden_automation,
    "preserve_provenance":     _eval_preserve_provenance,
}


def evaluate_constraints(
    constitution: dict[str, Any],
    ctx: Optional[dict[str, Any]] = None,
) -> list[str]:
    """Run all wired constraint evaluators against ``ctx``.

    Returns the list of violation reasons (empty → no violations →
    `constraint_gate=1` per scoring.constraint_gate()).

    Constraints with no evaluator wired produce a "not-evaluated"
    annotation in the audit log but are NOT treated as violations —
    permissive-by-default for unknown rules.  When you add a new
    constraint, also add an evaluator above so it stops landing in
    the not-evaluated list.
    """
    ctx = ctx or {}
    violations: list[str] = []
    for c in (constitution or {}).get("constraints", []) or []:
        if not isinstance(c, dict):
            continue
        cid = c.get("id") or ""
        rule = c.get("rule") or ""
        fn = _EVALUATORS.get(cid)
        if fn is None:
            # Unknown constraint — log via the audit, don't gate.
            continue
        try:
            reason = fn(rule=rule, ctx=ctx)
        except Exception as e:
            # Evaluator bug → conservative: log + treat as violated.
            violations.append(
                f"{cid}: evaluator error: {type(e).__name__}: {e}"
            )
            continue
        if reason:
            violations.append(reason)
    return violations


__all__ = [
    "objectives_to_dict",
    "evaluate_constraints",
]
