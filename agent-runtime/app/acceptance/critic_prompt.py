"""Phase 6.1 — Fresh-context critic prompt builder.

Critics MUST receive only `(spec, acceptance_criteria, artifact)`.
NOT the producer's trajectory / chain-of-thought / per-turn comments.

This module is the single source of truth for what gets inlined into a
critic's dispatch user_message.  Source-grep tests (AC-22) verify the
forbidden fields never appear in this module or its call sites.

Call site: `agent-runtime/app/workers/inbox_poller.py` critic-dispatch
branch.
"""

from __future__ import annotations

from typing import Any

# Tokens forbidden from appearing in critic dispatch prompts.  AC-22
# does a structural source grep for these strings across the critic
# prompt path.  Any field whose name implies producer-trajectory
# inheritance is here — debug flags / fallback branches don't get a
# pass.
FORBIDDEN_PRODUCER_FIELDS: frozenset[str] = frozenset({
    "producer_messages",
    "producer_trajectory",
    "parent_thread",
    "chain_of_thought",
    "producer_reasoning",
    "ic_internal_notes",
})


def build_critic_user_message(
    *,
    task_id: str,
    brief: str,
    deliverable_spec: str,
    acceptance_criteria: list[dict[str, Any]] | None,
    outputs: list[str] | None,
    inputs: list[str] | None,
) -> str:
    """Produce the critic's dispatch user message.

    Inputs are restricted to:
      - task identity (task_id, brief, deliverable_spec) — the contract
      - acceptance_criteria — the gate the critic walks
      - outputs — the artifact paths to inspect
      - inputs — upstream artifact paths (for cross-references)

    Inputs DELIBERATELY excluded:
      - producer's chain-of-thought
      - intermediate comments from the producing IC
      - any debug-mode trajectory dump

    The critic reads artifact CONTENT via tool calls (read_file,
    output_path_*), not via the prompt.  The prompt tells WHAT to check,
    not WHAT the producer thought.
    """
    acs_block = _format_acceptance_criteria(acceptance_criteria or [])
    outputs_block = _format_path_list("Artifacts to review", outputs)
    inputs_block = _format_path_list("Upstream inputs", inputs)
    # [CRITIC_NO_GET_TASK_LOOP_2026_05_29]  The persona's own system
    # prompt tells critic-prose to "Read the artifact at task.outputs[]"
    # — historically that meant "call get_task to fetch outputs first".
    # But this dispatch message ALREADY inlines task_id, brief, spec,
    # inputs, outputs, AND acceptance_criteria.  Without the explicit
    # guard the critic spins:  call get_task → result confuses model →
    # "I need to find the actual task ID" → repeat, blowing 8-10
    # turns of max_iterations on nothing.  Caught live 2026-05-29 with
    # critic-prose emitting 24 events of "I need to find the actual
    # task ID" on a single dispatch.  Mirror the same explicit guard
    # the iteration prompt already uses (inbox_poller line 985).
    # [CRITIC_SERVER_SIDE_AC_2026_05_29]  The mechanical acceptance
    # checks (output_path_exists, output_path_matches_regex, …) are now
    # run SERVER-SIDE inside `review_decision` — the critic does NOT have
    # and does NOT need an executor tool.  The previous prompt told it to
    # "run the appropriate check_kind executor and emit a criterion_check
    # comment", but the critic has no such tool, so it looped on
    # add_comment 9× and never reached review_decision → max_turns →
    # force-cancel on EVERY review.  New workflow: read the artifact,
    # make the SEMANTIC judgment, call review_decision.  The server runs
    # the mechanical gates and will auto-downgrade `approved` to
    # `changes_requested` if a required mechanical check fails — so the
    # critic can't rubber-stamp a missing file even if it tries.
    return (
        f"You are reviewing Task **{task_id}** (authoritative id — do NOT "
        f"call get_task; everything you need is below).\n\n"
        f"Brief: {brief}\n\n"
        f"Deliverable spec: {deliverable_spec}\n"
        f"{inputs_block}"
        f"{outputs_block}\n\n"
        f"{acs_block}\n\n"
        f"Do EXACTLY this, in order — it is a 2-step job, not a loop:\n"
        f"  1. Call `read_file` ONCE for each artifact path above to load "
        f"the content.\n"
        f"  2. Make your judgment and call `review_decision` "
        f"(approved | changes_requested | declined) with a one-sentence "
        f"note. THIS IS YOUR FINAL ACTION.\n\n"
        f"You judge SEMANTICS only — does the artifact satisfy the brief "
        f"and spec (scope, structure, quality)?  The mechanical "
        f"acceptance checks (file exists, size, regex) run AUTOMATICALLY "
        f"on the server when you call review_decision — you do NOT run "
        f"them yourself and you do NOT need to emit criterion_check "
        f"comments.\n\n"
        f"Hard rules:\n"
        f"  - Judge against EVERY required AC listed above before you decide. "
        f"Do not give benefit of the doubt: if any required AC is unmet, "
        f"unverifiable from the artifact, or the work only partially satisfies "
        f"the brief, the decision is changes_requested — never approved.\n"
        f"  - After reading the artifact, your VERY NEXT tool call MUST be "
        f"review_decision. Do not add_comment first — put your reasoning "
        f"in the review_decision `notes`.\n"
        f"  - If the artifact looks good, call review_decision(approved). "
        f"If the server's mechanical checks fail it will auto-downgrade — "
        f"you don't need to second-guess file existence.\n"
        f"  - Do NOT call get_task. Do NOT call read_file more than once "
        f"per path. Do NOT loop on add_comment — that is the #1 cause of "
        f"a review never completing."
    )


def _format_acceptance_criteria(acs: list[dict[str, Any]]) -> str:
    if not acs:
        return "**Acceptance criteria:** (none) — flag this as a contract violation."
    lines = ["**Acceptance criteria (walk each one):**"]
    for ac in acs:
        ac_id = ac.get("id", "?")
        statement = ac.get("statement", "")
        kind = ac.get("check_kind", "")
        req = "required" if ac.get("required", True) else "advisory"
        lines.append(f"  - {ac_id} [{req}] ({kind}): {statement}")
    return "\n".join(lines)


def _format_path_list(label: str, paths: list[str] | None) -> str:
    if not paths:
        return ""
    body = "\n".join(f"  - {p}" for p in paths)
    return f"\n\n**{label}:**\n{body}"
