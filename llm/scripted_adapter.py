"""ScriptedAdapter — drop-in replacement for AnthropicAdapter that
returns canned content keyed on (agent_id, turn_index).

Same interface as AnthropicAdapter (`.generate(user_msg, label=)` →
`AdapterResult`), so Agent.run_turn doesn't know it's being run on
fake data.  Lets the 4 existing scripted demos (Pattern A solo,
Pattern B delegation, Pattern C fanout, Critic-iterate) continue to
work through the SAME pipeline — TCMM render + PII redact + adapter +
rehydrate + TCMM ingest — without paying for or depending on a real LLM.

Usage:

    from llm.scripted_adapter import set_script, Turn, ToolCall

    set_script({
        "director": [
            Turn(text="Acknowledged.", tool_calls=[
                ToolCall(name="create_task", input={...}),
            ]),
            Turn(text="Task complete.", stop_reason="end_turn"),
        ],
    })

    # ... runtime instantiates DirectorAgent(persona, adapter_cls=
    #     ScriptedAdapter) and runs as normal.

The script is process-global (so demos can install it before invoking
agent.run_turn).  Tests use set_script and clear_script for isolation.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .adapter import AdapterResult

logger = logging.getLogger("veilguard.llm.scripted_adapter")


# ── Script DSL ──────────────────────────────────────────────────────────


@dataclass
class ToolCall:
    """A scripted tool_use the persona will emit on a given turn."""
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"tu_{uuid.uuid4().hex[:8]}")


@dataclass
class Turn:
    """One scripted invocation of a persona.

    stop_reason semantics:
      - "tool_use" (auto when tool_calls present): runtime dispatches
        tools, feeds tool_results back, calls this persona's NEXT
        scripted turn
      - "end_turn" (auto when no tool_calls): runtime stops looping
        this agent
    """
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.stop_reason is None:
            self.stop_reason = "tool_use" if self.tool_calls else "end_turn"


Script = dict[str, list[Turn]]


# ── Process-global script state ─────────────────────────────────────────


_CURRENT_SCRIPT: Optional[Script] = None
_TURN_COUNTERS: dict[str, int] = {}


def set_script(script: Script) -> None:
    """Install a script.  Resets per-persona turn counters."""
    global _CURRENT_SCRIPT, _TURN_COUNTERS
    _CURRENT_SCRIPT = script
    _TURN_COUNTERS = {}


def clear_script() -> None:
    set_script({})


def get_turn_counter(persona_id: str) -> int:
    return _TURN_COUNTERS.get(persona_id, 0)


# ── Adapter ─────────────────────────────────────────────────────────────


class ScriptedAdapter:
    """Same interface as AnthropicAdapter, returns canned content.

    Construction args mirror AnthropicAdapter exactly so swap is a
    one-liner in tests / demos.
    """

    def __init__(
        self,
        *,
        model: str,
        system_blocks: Optional[list[dict]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        agent_id: str = "",
    ):
        self._model = model
        self._agent_id = agent_id

    async def generate(
        self, user_message: str, *, label: str = "scripted"
    ) -> AdapterResult:
        # The label is "agent:<agent_id>" when called from Agent.run_turn.
        # Pull agent_id out so demos can key their scripts by persona.
        agent_id = self._agent_id or (
            label.split(":", 1)[1] if ":" in label else label
        )

        if _CURRENT_SCRIPT is None:
            return _empty_result(self._model)

        turns = _CURRENT_SCRIPT.get(agent_id, [])
        idx = _TURN_COUNTERS.get(agent_id, 0)
        _TURN_COUNTERS[agent_id] = idx + 1

        if idx >= len(turns):
            logger.debug(
                f"[scripted] {agent_id} turn {idx} out of script; "
                f"yielding empty end_turn"
            )
            return _empty_result(self._model)

        turn = turns[idx]
        logger.info(
            f"[scripted] {agent_id} turn {idx}: "
            f"text={turn.text[:40]!r} tools={[t.name for t in turn.tool_calls]}"
        )

        content_blocks: list[dict] = []
        if turn.text:
            content_blocks.append({"type": "text", "text": turn.text})
        for tc in turn.tool_calls:
            content_blocks.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": dict(tc.input),
            })

        usage = {
            "input_tokens": 200,
            "output_tokens": 80,
            "cache_creation_input_tokens": 0 if idx > 0 else 1000,
            "cache_read_input_tokens": 1000 if idx > 0 else 0,
        }

        return AdapterResult(
            text=turn.text,
            content_blocks=content_blocks,
            usage=usage,
            stop_reason=turn.stop_reason or "end_turn",
            mode="Scripted",
            model=self._model,
        )


def _empty_result(model: str) -> AdapterResult:
    return AdapterResult(
        text="(end of script)",
        content_blocks=[{"type": "text", "text": "(end of script)"}],
        usage={
            "input_tokens": 0, "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
        stop_reason="end_turn",
        mode="Scripted",
        model=model,
    )


__all__ = [
    "ToolCall",
    "Turn",
    "Script",
    "ScriptedAdapter",
    "set_script",
    "clear_script",
    "get_turn_counter",
]
