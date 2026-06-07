"""TCMM record-turn shadow tool — schema + intercept.

Ported from agent-proxy/app/main.py's `_TCMM_RECORD_TURN_TOOL`,
`_inject_tcmm_record_tool`, and `_intercept_tcmm_record_tool_use`.

The shadow tool lives here (in the agent package) because it's a
turn-level concern — the model emits it at the END of every assistant
response, and the proxy/agent intercepts it BEFORE the user sees the
response.  It carries metadata about which memory blocks the answer
relied on, used by TCMM for heat-based promotion/decay.

Hooks fire in `Agent.run_turn`:
  - `inject_into_tools(tools)` — pre-LLM, prepends the shadow tool so
    the model knows to emit it
  - `intercept_response(content, stop_reason)` — post-LLM, removes the
    tool_use block before yielding content, captures its input as
    `flag_obj` for the subsequent ingest_assistant call
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("veilguard.agent.shadow_tool")


# Schema — kept in sync with agent-proxy/app/main.py _TCMM_RECORD_TURN_TOOL.
# When that file is eventually retired (PR #5b), this becomes the only
# source of truth.
TCMM_RECORD_TURN_TOOL: dict = {
    "name": "tcmm_record_turn",
    "description": (
        "Veilguard-internal: record metadata about this assistant turn "
        "for memory management.  Call this as the LAST action of every "
        "response, after any text and other tool calls.  The user never "
        "sees this tool.  Do not announce that you're calling it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "knowledge_class": {
                "type": "string",
                "enum": ["derived", "novel", "mixed"],
                "description": (
                    "derived = answer used only memory/general knowledge; "
                    "novel = contains new information worth remembering; "
                    "mixed = combination."
                ),
            },
            "used": {
                "type": "object",
                "description": (
                    "REQUIRED citation map.  Keys = memory block IDs "
                    "(as strings) the answer actually drew on.  Values "
                    "= relevance 0-1 (1 = directly quoted/restated, "
                    "0.5 = informed reasoning, <0.3 = barely used).  "
                    "Emit {} ONLY when zero memory blocks contributed "
                    "to this answer (pure general-knowledge response)."
                ),
                "additionalProperties": {"type": "number"},
            },
            "epoch_complete": {
                "type": "boolean",
                "description": (
                    "true if this turn closes an epoch — a coherent unit "
                    "of conversation that can be compacted.  Most turns "
                    "are false; mark true on natural completion points."
                ),
            },
            "emit_class": {
                "type": "string",
                # [TAXONOMY_FIX_2026-06-03] Must match the canonical episodic
                # ontology (core/episodic_ontology.py EPISODIC_CLASS_SET) — this
                # enum had drifted to a different taxonomy (factoid/instruction/
                # question/...), so every class except "decision" failed the
                # `_emit in EPISODIC_CLASS_SET` check in post_response and got
                # deferred to the AI Studio NLP worker (which can't run without
                # credits) → block_class stayed NULL forever. Kept byte-identical
                # to agent-proxy/app/main.py _TCMM_RECORD_TURN_TOOL.
                "enum": [
                    "FACT", "DECISION", "INSIGHT", "PROCEDURE", "STATE",
                    "INTENT", "DERIVED_FACT", "ARTIFACT", "AGENT_NOTE",
                    "CHATTER", "ACK", "QUERY", "TRANSIENT_DATA",
                    "EXECUTION_LOG",
                ],
                "description": (
                    "Required: classify this turn into the canonical "
                    "episodic ontology. Pick the SINGLE best fit.\n"
                    "- FACT: a verifiable statement about the world "
                    "(\"Paris is the capital of France\").\n"
                    "- DERIVED_FACT: a fact stitched together from "
                    "memory blocks (\"based on block 3+5, the user "
                    "prefers X\").\n"
                    "- DECISION: a choice or commitment made "
                    "(\"I'll use Python over Go for this\").\n"
                    "- INSIGHT: a new connection or realization "
                    "(\"so the bug is in the cache layer, not the "
                    "renderer\").\n"
                    "- PROCEDURE: action steps or how-to "
                    "(\"1. install X 2. configure Y\").\n"
                    "- STATE: current system or conversation state "
                    "(\"the build is failing at step 3\").\n"
                    "- INTENT: a stated goal or planned next action "
                    "(\"next I'll patch the renderer\").\n"
                    "- ARTIFACT: a piece of code/text/data you "
                    "produced for the user (a code block, a config).\n"
                    "- AGENT_NOTE: internal reasoning the user "
                    "doesn't need to retain (\"hmm, let me think\").\n"
                    "- CHATTER: small-talk, pleasantries (\"Hi\", "
                    "\"You're welcome\").\n"
                    "- ACK: pure acknowledgement (\"Got it.\", "
                    "\"Done.\").\n"
                    "- QUERY: a clarifying question back to the user "
                    "(\"Did you mean X or Y?\").\n"
                    "- TRANSIENT_DATA: ephemeral output like long "
                    "listings or table dumps not worth recalling.\n"
                    "- EXECUTION_LOG: traces of tool invocations or "
                    "command output (\"ran `ls`, got 12 files\")."
                ),
            },
            "redaction_audit": {
                "type": "object",
                "description": (
                    "OPTIONAL PII-boundary audit of THIS turn's visible text. "
                    "Only include it when you actually notice something — most "
                    "turns omit it. over_redacted: REF_* tokens that are clearly "
                    "PUBLIC entities (companies, products, security tools, threat "
                    "actors) wrongly tokenized as people — candidates to STOP "
                    "redacting. missed: literal PII strings you can SEE "
                    "un-redacted in the text (a person's name, email, account or "
                    "ID number) that SHOULD have been redacted."
                ),
                "properties": {
                    "over_redacted": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "token": {"type": "string"},
                                "reason": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                            "required": ["token"],
                        },
                    },
                    "missed": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "pii_type": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                            "required": ["text"],
                        },
                    },
                },
            },
        },
        "required": ["knowledge_class", "epoch_complete", "emit_class", "used"],
    },
}


def inject_into_tools(tools_list: Optional[list[dict]]) -> list[dict]:
    """Prepend the shadow tool to the user-provided tools list.

    Idempotent — re-injection on an already-augmented list is a no-op.
    Anthropic-shape only (no OpenAI variant here; ChatAgent today only
    runs against the Anthropic adapter).
    """
    out = list(tools_list) if tools_list else []
    for t in out:
        if isinstance(t, dict) and t.get("name") == "tcmm_record_turn":
            return out
    return [TCMM_RECORD_TURN_TOOL] + out


def intercept_response(
    content_blocks: Optional[list[dict]],
    stop_reason: Optional[str],
    sid: Optional[str] = None,
) -> tuple[list[dict], dict, str]:
    """Strip the shadow tool_use block from the response.

    Returns:
      (cleaned_blocks, flag_obj, new_stop_reason)

    flag_obj is the model's emitted input for `tcmm_record_turn`, ready
    to pass to `tcmm_client.ingest_assistant(..., flag_obj=...)`.

    new_stop_reason: if the ONLY tool_use in the response was the
    shadow tool, the turn is logically complete — downgrade
    "tool_use" → "end_turn" so downstream loops don't wait for a
    tool_result on a tool we consumed internally.
    """
    if not isinstance(content_blocks, list):
        return [], {}, stop_reason or "end_turn"

    cleaned: list[dict] = []
    flag_obj: dict = {}
    for b in content_blocks:
        if (
            isinstance(b, dict)
            and b.get("type") == "tool_use"
            and b.get("name") == "tcmm_record_turn"
        ):
            inp = b.get("input")
            if isinstance(inp, dict):
                flag_obj = inp
            # Drop from forwarded content.
            continue
        cleaned.append(b)

    new_stop = stop_reason or "end_turn"
    if stop_reason == "tool_use":
        has_real_tool_use = any(
            isinstance(b, dict) and b.get("type") == "tool_use"
            for b in cleaned
        )
        if not has_real_tool_use:
            new_stop = "end_turn"

    # [REDACTION_AUDIT_2026-06-07] fan out the model's PII-boundary audit
    # (best-effort; a DB blip must never break the turn).
    if flag_obj:
        process_redaction_audit(flag_obj, sid)

    return cleaned, flag_obj, new_stop


def _resolve_token(token: str, sid):
    """Resolve a REF_* token back to its original via the redactor's
    rehydrate_text (reads the per-session token map). None if it can't
    (no sid / no mapping) → the over_redacted candidate is skipped."""
    if not sid:
        return None
    try:
        from pii import get_redactor
        out = get_redactor().rehydrate_text(token, sid)
        if out and out != token:
            return out
    except Exception:
        pass
    return None


def process_redaction_audit(flag_obj: dict, sid=None) -> None:
    """[REDACTION_AUDIT_2026-06-07] Fan out the model's redaction_audit:
      missed         -> pii_deny_list  (FORCE-redact; safe direction, auto-applied)
      over_redacted  -> resolve REF_* token -> original -> redaction_suggestions
                        (whitelist CANDIDATE; gated later by the promoter/confirm)
    Best-effort: never raises (a DB hiccup must not break the turn)."""
    try:
        audit = (flag_obj or {}).get("redaction_audit")
        if not isinstance(audit, dict):
            return
        missed = audit.get("missed") or []
        over = audit.get("over_redacted") or []
        if not missed and not over:
            return
        import psycopg2
        from pii.session_store import _PG_DSN
        conn = psycopg2.connect(_PG_DSN, connect_timeout=3)
        conn.autocommit = True
        cur = conn.cursor()
        for m in missed:
            term = ((m.get("text") if isinstance(m, dict) else m) or "").strip()
            if len(term) < 2:
                continue
            ptype = m.get("pii_type") if isinstance(m, dict) else None
            cur.execute(
                "INSERT INTO pii_deny_list (term, entity_type, source, added_by, active) "
                "VALUES (%s,%s,'llm-audit','llm',true) "
                "ON CONFLICT (term) DO UPDATE SET active=true",
                (term, ptype),
            )
        for o in over:
            token = ((o.get("token") if isinstance(o, dict) else o) or "").strip()
            if "REF_" not in token:
                continue
            original = _resolve_token(token, sid)
            if not original or len(original) < 2:
                continue
            cur.execute(
                "INSERT INTO redaction_suggestions "
                "(term, kind, entity_type, support_count, status, last_seen) "
                "VALUES (%s,'whitelist','llm_over_redacted',1,'staged',now()) "
                "ON CONFLICT (term, kind) DO UPDATE SET "
                "support_count = redaction_suggestions.support_count + 1, last_seen=now()",
                (original,),
            )
        conn.close()
    except Exception:
        pass


__all__ = [
    "TCMM_RECORD_TURN_TOOL",
    "inject_into_tools",
    "intercept_response",
]
