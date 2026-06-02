"""Phase 3 — Director's Haiku rank pass over proposal candidates (§3.7.5).

Per spec §3.7.5: top-3 candidates by impact_score get deterministic
template briefs + default assignees (no LLM call).  Ranks 4-10 go
through **ONE** Haiku rank pass — a single structured-output call
that returns refined briefs + assignee picks + per-candidate
objective_alignment refinements + rationales.

Budget per spec: ~$0.001/cycle/tenant.  Haiku-only.

This module:
  * Builds the prompt + tool schema for structured output
  * Calls the AnthropicAdapter once
  * Parses + validates the response
  * Returns refined candidates that the caller persists

Contract — input:
    candidates: list of dicts with at least:
        signal_type:      str
        signal_node_ids:  list[int]
        impact_score:     float
        default_brief:    str   (template-rendered)
        default_assignee: str
        default_align:    float
        payload_summary:  str   (1-line summary the LLM sees)

Contract — output (one dict per surviving candidate):
    {
      "signal_node_ids":            same as input (correlation key)
      "refined_brief":              str
      "refined_assignee":           str  (must be a valid agent_id)
      "refined_objective_alignment": float in [0, 1]
      "rationale":                  str  (1-line "why this matters")
      "drop":                       bool (True → below cutoff, skip)
    }

Caller composes refined values back into a proposal row.  The
``drop=True`` cases are deferred per §3.7.4 (decay applies); we don't
discard the candidate, we mark it as "below cutoff this cycle."

Test-friendly: ``rank_candidates`` is async + accepts an injectable
``adapter_factory``.  Default factory wires the real AnthropicAdapter;
tests pass a mock that returns canned blocks.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


# Default rank-pass model (per agents/director.md `rank_pass=`).
DEFAULT_RANK_MODEL = "claude-haiku-4-5"

# Valid PROPOSAL-OWNER assignees the rank-pass LLM can choose.  Post-validated
# against this set so a hallucination doesn't land "researcher-bot" in the queue.
#
# [PROPOSAL_ASSIGNEE_NO_CRITIC_2026_06_02] Critics are intentionally ABSENT: a
# proposal's owner PRODUCES the deliverable, and critics are read-everywhere /
# write-nowhere (§4.4) — owning a producing task dead-ends it (no write_file /
# attach_output / submit_for_review). The critic split applies at REVIEW time
# (submit_for_review), not ownership. Add producer agent_ids here as needed.
VALID_ASSIGNEES = frozenset({
    "researcher",
    "builder",
    "phishing-analyst",
    "threat-analyst",
    "report-writer",
})


# ── Prompt assembly ────────────────────────────────────────────────────


def _build_system_prompt(constitution_objectives: dict[str, float]) -> str:
    """Static-ish system prompt explaining the task + the constitution.

    Kept deliberately short — Haiku doesn't need much context, and
    every extra token is a cost we'll pay 1× per dream cycle per
    tenant.  Objectives are inlined so the LLM can do
    `objective_alignment` math without a tool call.
    """
    obj_lines = "\n".join(
        f"  - {k}: weight={v:.2f}"
        for k, v in sorted(constitution_objectives.items(),
                           key=lambda kv: -kv[1])
    ) or "  (no objectives loaded)"
    return (
        "You are Veilguard's Director pre-evaluator.  Your job: take "
        "a list of candidate Task proposals derived from dream-signal "
        "nodes and decide which are worth surfacing to the user.\n\n"
        "Constitution objectives (higher weight = more important):\n"
        f"{obj_lines}\n\n"
        "For each candidate you keep, produce:\n"
        "  - refined_brief: clearer + more specific than the default\n"
        "  - refined_assignee: agent_id of who should do the work\n"
        "  - refined_objective_alignment: float in [0, 1] — dot product "
        "of the candidate's actual impact on the constitution objectives\n"
        "  - rationale: ONE sentence on why this matters\n"
        "  - drop: True if below the surfacing cutoff\n\n"
        f"Valid assignee values: {sorted(VALID_ASSIGNEES)}\n\n"
        "Use the rank_proposals tool exactly once.  Do NOT chain tool "
        "calls.  Stay under 500 tokens of output."
    )


def _build_user_prompt(candidates: list[dict[str, Any]]) -> str:
    """Render the candidate list as a numbered, parseable block."""
    lines = ["Candidates ranked 4-{n} by deterministic impact_score:".format(n=3 + len(candidates))]
    for i, c in enumerate(candidates, start=4):
        lines.append(
            f"\n#{i}  signal_type={c['signal_type']}\n"
            f"  impact_score={c['impact_score']:.3f}\n"
            f"  default_align={c.get('default_align', 0):.3f}\n"
            f"  default_assignee={c.get('default_assignee', '?')}\n"
            f"  default_brief={c.get('default_brief', '')[:240]}\n"
            f"  payload_summary={c.get('payload_summary', '(no summary)')}\n"
            f"  signal_node_ids={c.get('signal_node_ids', [])}"
        )
    return "\n".join(lines)


# ── Tool schema for structured output ──────────────────────────────────


def _build_rank_tool_schema() -> dict[str, Any]:
    """Anthropic tool schema forcing structured output.

    The LLM emits ONE call to this tool; the call's input is the
    parseable result.  Using a tool (not just JSON in text) makes
    parsing failure-proof — Anthropic validates against the schema
    before returning.
    """
    return {
        "name": "rank_proposals",
        "description": (
            "Submit the refined ranking for these candidates.  Call "
            "exactly once with the full list (including drops)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ranked": {
                    "type": "array",
                    "description": (
                        "One entry per input candidate, in any order. "
                        "Correlate by signal_node_ids."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "signal_node_ids": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "Same node IDs as the input row",
                            },
                            "refined_brief": {
                                "type": "string",
                                "description": (
                                    "Clearer + more specific than the "
                                    "default brief; ≤300 chars"
                                ),
                            },
                            "refined_assignee": {
                                "type": "string",
                                "description": (
                                    f"One of: {sorted(VALID_ASSIGNEES)}"
                                ),
                            },
                            "refined_objective_alignment": {
                                "type": "number",
                                "description": "[0, 1]",
                            },
                            "rationale": {
                                "type": "string",
                                "description": "One sentence on why this matters",
                            },
                            "drop": {
                                "type": "boolean",
                                "description": (
                                    "True if below the surfacing cutoff "
                                    "(deferred, not discarded)"
                                ),
                            },
                        },
                        "required": [
                            "signal_node_ids",
                            "refined_brief",
                            "refined_assignee",
                            "refined_objective_alignment",
                            "rationale",
                            "drop",
                        ],
                    },
                }
            },
            "required": ["ranked"],
        },
    }


# ── Default adapter factory ─────────────────────────────────────────────


async def _default_adapter_call(
    *,
    model: str,
    system_prompt: str,
    user_message: str,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run a single AnthropicAdapter call and return the parsed tool input.

    Lives behind a factory boundary so tests can mock it.  This is the
    only place that touches the live API in this module.
    """
    from llm.adapter import AnthropicAdapter   # local import — heavy
    adapter = AnthropicAdapter(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        agent_id="director-rank-pass",
    )
    result = await adapter.generate(user_message, label="proposals.rank")
    # Walk content_blocks for the tool_use block
    for block in result.content_blocks or []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            inp = block.get("input") or {}
            if isinstance(inp, dict):
                return inp
    # No tool_use → fall back to parsing the text as JSON
    text = (result.text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(
            "[proposals.rank] adapter returned no tool_use and "
            "non-JSON text; got: %r", text[:200],
        )
        return {"ranked": []}


# ── Public API ──────────────────────────────────────────────────────────


# Type alias for the injectable adapter call signature.
AdapterFn = Callable[..., Awaitable[dict[str, Any]]]


async def rank_candidates(
    *,
    candidates: list[dict[str, Any]],
    constitution_objectives: dict[str, float],
    model: str = DEFAULT_RANK_MODEL,
    adapter_call: Optional[AdapterFn] = None,
) -> list[dict[str, Any]]:
    """Run the Director Haiku rank pass over ``candidates``.

    Args:
        candidates: ranks 4-N from the deterministic top-N path; each
            dict must include signal_type, impact_score, default_brief,
            default_assignee, signal_node_ids (+ optional payload_summary).
        constitution_objectives: flat dict[id, weight] from the
            constitution.  Inlined into the system prompt.
        model: Anthropic model alias.  Defaults to
            ``claude-haiku-4-5`` per agents/director.md.
        adapter_call: optional async callable for tests.  Defaults to
            ``_default_adapter_call`` which makes a real Anthropic call.

    Returns:
        Refined list, one entry per input candidate.  Caller correlates
        by ``signal_node_ids``.  Entries with ``drop=True`` should be
        deferred (decay applies), not discarded.

    Robustness:
        * Empty input → empty output (no LLM call).
        * Adapter error → returns the input unchanged (preserves the
          default briefs so callers don't lose proposals to a hiccup).
        * Invalid assignee from the LLM → replaced with the default
          assignee + a warning logged.
        * Out-of-range alignment → clipped to [0, 1].
    """
    if not candidates:
        return []

    call = adapter_call or _default_adapter_call
    system_prompt = _build_system_prompt(constitution_objectives)
    user_message = _build_user_prompt(candidates)
    tools = [_build_rank_tool_schema()]

    try:
        result = await call(
            model=model,
            system_prompt=system_prompt,
            user_message=user_message,
            tools=tools,
        )
    except Exception as e:
        logger.exception(
            "[proposals.rank] adapter call failed: %s — falling back "
            "to default briefs", e,
        )
        return [_default_passthrough(c) for c in candidates]

    ranked = result.get("ranked", []) if isinstance(result, dict) else []
    if not isinstance(ranked, list):
        logger.warning(
            "[proposals.rank] adapter returned non-list 'ranked'; "
            "falling back to default briefs"
        )
        return [_default_passthrough(c) for c in candidates]

    # Correlate by signal_node_ids → fold into refined output
    by_key: dict[tuple, dict[str, Any]] = {}
    for entry in ranked:
        if not isinstance(entry, dict):
            continue
        node_ids = entry.get("signal_node_ids") or []
        key = tuple(sorted(int(n) for n in node_ids if isinstance(n, (int, float))))
        by_key[key] = entry

    out: list[dict[str, Any]] = []
    for c in candidates:
        node_ids = c.get("signal_node_ids") or []
        key = tuple(sorted(int(n) for n in node_ids))
        e = by_key.get(key)
        if e is None:
            # LLM dropped this candidate from its response — treat as
            # "no opinion": use defaults.  Better than losing it.
            out.append(_default_passthrough(c))
            continue
        # Validate + clip
        ass = (e.get("refined_assignee") or "").strip()
        if ass not in VALID_ASSIGNEES:
            logger.warning(
                "[proposals.rank] LLM picked invalid assignee %r "
                "for signal_node_ids=%s; using default %r",
                ass, node_ids, c.get("default_assignee"),
            )
            ass = c.get("default_assignee", "researcher")
        # [PROPOSAL_ASSIGNEE_NO_CRITIC_2026_06_02] Never let a critic become the
        # owner — even via a stale candidate default (the fallback above pulls
        # default_assignee without re-validating). Critics review at submit
        # time; they have no write tools and would dead-end the task.
        if ass in ("critic-claim", "critic-prose"):
            ass = "researcher"
        try:
            align = float(e.get("refined_objective_alignment") or c.get("default_align", 0.0))
        except (TypeError, ValueError):
            align = c.get("default_align", 0.0)
        align = max(0.0, min(1.0, align))
        out.append({
            "signal_node_ids":             list(node_ids),
            "refined_brief":               (e.get("refined_brief") or c.get("default_brief", "")).strip(),
            "refined_assignee":            ass,
            "refined_objective_alignment": align,
            "rationale":                   (e.get("rationale") or "").strip(),
            "drop":                        bool(e.get("drop", False)),
        })
    return out


def _default_passthrough(c: dict[str, Any]) -> dict[str, Any]:
    """Build a refined-entry from the candidate's defaults when the
    LLM call fails or skips it.  Preserves the candidate (doesn't
    drop it) so the cycle still surfaces something."""
    return {
        "signal_node_ids":             list(c.get("signal_node_ids") or []),
        "refined_brief":               c.get("default_brief", ""),
        "refined_assignee":            c.get("default_assignee", "researcher"),
        "refined_objective_alignment": float(c.get("default_align", 0.0) or 0.0),
        "rationale":                   "(LLM rank pass unavailable — using template)",
        "drop":                        False,
    }


__all__ = [
    "DEFAULT_RANK_MODEL",
    "VALID_ASSIGNEES",
    "rank_candidates",
]
