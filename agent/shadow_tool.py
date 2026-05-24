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
                "enum": [
                    "factoid", "decision", "instruction", "question",
                    "summary", "code", "small_talk", "other",
                ],
                "description": "Top-level classification of this turn's content.",
            },
        },
        "required": ["knowledge_class", "used", "emit_class"],
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

    return cleaned, flag_obj, new_stop


__all__ = [
    "TCMM_RECORD_TURN_TOOL",
    "inject_into_tools",
    "intercept_response",
]
