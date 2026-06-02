"""Phase 3 — brief composition per signal_type.

Per spec §3.7.5: top-3 candidates get deterministic template-generated
briefs (no LLM call); ranks 4-10 get Director-LLM-drafted briefs from a
single rank-pass.  This module owns the templates and the substitution
machinery for the deterministic path.

Templates are STATIC config — change them here, not at call-sites.
Caller supplies a dict of substitution variables; missing keys land as
``<missing:key>`` in the output so the audit trail can flag the
template/payload mismatch instead of silently corrupting briefs.
"""

from __future__ import annotations

from typing import Mapping

from .scoring import (
    SIGNAL_INFORMATION_GAP,
    SIGNAL_CONTRADICTION_ARC,
    SIGNAL_REFLECTIVE_HEURISTIC,
    SIGNAL_RECURRING_RITUAL,
    SIGNAL_STANCE_ARC,
    SIGNAL_LOW_STABILITY,
    SIGNAL_STALE_SUPERSESSION,
)


# Per §3.7.5 — template strings keyed by signal_type.  Placeholders are
# `{name}`-style; missing substitutions become `<missing:name>` in the
# output (NOT a KeyError) so the rendered brief is still usable as a
# user-visible proposal — the user can spot the mistake and intervene.

BRIEF_TEMPLATES: dict[str, str] = {
    SIGNAL_INFORMATION_GAP: (
        "Research {gap_topic}. Cite sources. ≤500 words. "
        "Target: {target_namespace}."
    ),
    SIGNAL_CONTRADICTION_ARC: (
        "Investigate conflict between {claim_a_text} and {claim_b_text}. "
        "Determine which supersedes; cite evidence."
    ),
    SIGNAL_RECURRING_RITUAL: (
        "Codify procedure {pattern_id} as a skill. "
        "Trigger conditions + steps + 2 examples."
    ),
    SIGNAL_REFLECTIVE_HEURISTIC: (
        "Review pattern: {pattern_desc}. "
        "Propose org-memory lesson or skill."
    ),
    SIGNAL_STANCE_ARC: (
        "Multi-reviewer review: agents {agent_a} and {agent_b} "
        "disagree on {claim_text}."
    ),
    SIGNAL_LOW_STABILITY: (
        "Re-investigate cluster: {cluster_desc} "
        "({fail_count} failed claims)."
    ),
    SIGNAL_STALE_SUPERSESSION: (
        "Refresh assumptions: {chain_topic} last touched {days_ago}d ago."
    ),
}


# Default deliverable spec per signal_type.
#
# [PROACTIVE_AC_PATH_2026_06_02] Each spec MUST name a concrete,
# regex-extractable output path (team/drafts/<name>.md).  `/proposals/convert`
# + the MCP create_task tool synthesize a default `output_path_exists` AC by
# extracting a path from this spec; the receiving IC also reads this spec and
# writes there.  Earlier specs named no real path ("Land in team_knowledge",
# "agents/skills/<slug>.md") so the synthesized AC fell back to `deliverable.md`
# — a file the IC never wrote — and EVERY approved proactive task died at the
# Critic gate (UAT follow-up to F13).  Imperative "Write … to <path>" keeps the
# IC's output and the AC target aligned by construction.
DELIVERABLE_SPECS: dict[str, str] = {
    SIGNAL_INFORMATION_GAP: (
        "Write a markdown research note to team/drafts/research-note.md — "
        "≤500 words, ≥2 sources cited inline. Promote to team_knowledge on "
        "Critic approval."
    ),
    SIGNAL_CONTRADICTION_ARC: (
        "Write the investigation note to team/drafts/contradiction-note.md — "
        "which claim supersedes, with evidence citations."
    ),
    SIGNAL_RECURRING_RITUAL: (
        "Write the skill definition to team/drafts/skill-def.md — trigger "
        "pattern, step-by-step procedure, 2 worked examples."
    ),
    SIGNAL_REFLECTIVE_HEURISTIC: (
        "Write the lesson/skill proposal to team/drafts/heuristic-review.md — "
        "an org_memory lesson (rule + trigger + evidence) OR a skill; state "
        "which and why."
    ),
    SIGNAL_STANCE_ARC: (
        "Write the committee verdict to team/drafts/stance-verdict.md — which "
        "stance is canonical going forward, with explicit dissent log entries."
    ),
    SIGNAL_LOW_STABILITY: (
        "Write the cluster decision to team/drafts/cluster-review.md — "
        "stabilise with new evidence OR retire the claims with supersession notes."
    ),
    SIGNAL_STALE_SUPERSESSION: (
        "Write the refresh-or-retire decision to team/drafts/chain-refresh.md — "
        "per-claim rationale and any new evidence found."
    ),
}


class _SafeDict(dict):
    """Dict that renders missing keys as ``<missing:KEY>`` on format().

    Lets ``template.format_map(_SafeDict(payload))`` succeed even when
    the caller forgets a placeholder — the rendered brief is still
    usable; the artefact in the database flags exactly what was missing
    so the next caller can fix it without a stack trace.
    """
    def __missing__(self, key: str) -> str:
        return f"<missing:{key}>"


def render_brief(signal_type: str, payload: Mapping[str, object]) -> str:
    """Apply the static template for ``signal_type`` to ``payload``.

    Unknown signal_type → empty string so the caller can detect the
    mismatch (a proposal with empty brief should NEVER pass through
    pre-eval; this is a safety stop, not silent loss).
    """
    tmpl = BRIEF_TEMPLATES.get(signal_type)
    if not tmpl:
        return ""
    return tmpl.format_map(_SafeDict(payload))


def render_deliverable_spec(signal_type: str) -> str:
    """Default deliverable spec for ``signal_type``.

    Unknown signal_type → empty string (same safety stop as
    ``render_brief``).
    """
    return DELIVERABLE_SPECS.get(signal_type, "")


__all__ = [
    "BRIEF_TEMPLATES",
    "DELIVERABLE_SPECS",
    "render_brief",
    "render_deliverable_spec",
]
