"""ScriptedBackend — deterministic fake LLM for demos + tests.

Returns predetermined tool_use / text-response sequences keyed on
(persona_id, turn_index).  Lets us exercise the full orchestration
pipeline without LLM cost, and lets demos drive specific scenarios
(Pattern A solo, Pattern B delegation, etc.).

How it works:
  - Each demo script declares a Script (dict from persona_id → list of
    Turn objects).  A Turn says "on the Nth invocation of this persona,
    yield this assistant message with these tool_uses".
  - On run_turn(), the backend looks up (persona_id, turn_index) in its
    script, yields the corresponding messages, increments the counter.
  - Tools are dispatched externally by hooks + MCP client (same as
    real SDK path); scripted backend just emits the tool_use blocks and
    waits for tool_result blocks to come back from the runtime.

Limitations:
  - Doesn't simulate streaming partial messages (text deltas) — yields
    the full assistant message in one block.  Sufficient for demos.
  - Doesn't simulate token usage realistically; returns fixed numbers
    so the audit pipeline has something to record.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from .base import BackendConfig

logger = logging.getLogger("agent-runtime.backends.scripted")


@dataclass
class ToolCall:
    """A scripted tool_use the persona will emit on a given turn."""

    name: str
    input: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"tu_{uuid.uuid4().hex[:8]}")


@dataclass
class Turn:
    """One scripted invocation of a persona.

    stop_reason controls whether the runtime's external loop continues
    (tool_use) or breaks (end_turn) after dispatching this turn's tools:

      - stop_reason="tool_use" (default when tool_calls exist): runtime
        dispatches tools, feeds results back, calls this persona for the
        NEXT scripted turn.  Use this for multi-step dispatches.
      - stop_reason="end_turn": runtime dispatches THIS turn's tools but
        then exits.  Use this to pause the script between distinct
        "wake-ups" of the same persona (e.g., Researcher submits, gets
        Critic feedback, then wakes again for the iterate).
    """

    text: str = ""                          # final assistant text
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None          # None = auto-pick
    # Auto-pick: tool_calls present → "tool_use"; otherwise → "end_turn".

    def __post_init__(self) -> None:
        if self.stop_reason is None:
            self.stop_reason = "tool_use" if self.tool_calls else "end_turn"


# A Script maps persona_id → list of Turns.  Turn N for persona P
# fires when persona P is invoked for the Nth time.  If the persona is
# invoked more than len(script[P]) times, we yield an empty end_turn
# (defensive default — better than IndexError).
Script = dict[str, list[Turn]]


# Process-global script registry.  Demos call `set_script(...)` before
# kicking off agent-runtime; the backend reads from this on each call.

_CURRENT_SCRIPT: Optional[Script] = None
_TURN_COUNTERS: dict[str, int] = {}


def set_script(script: Script) -> None:
    """Install the script for the current process.  Resets counters."""
    global _CURRENT_SCRIPT, _TURN_COUNTERS
    _CURRENT_SCRIPT = script
    _TURN_COUNTERS = {}


def get_turn_counter(persona_id: str) -> int:
    return _TURN_COUNTERS.get(persona_id, 0)


def clear_script() -> None:
    set_script({})


# ── Fake message classes ─────────────────────────────────────────────────
# Match the shape audit.tap_sdk_stream() + runtime._msg_to_events() expect.


class _FakeUsage:
    def __init__(
        self,
        input_tokens: int = 100,
        output_tokens: int = 50,
        cache_creation: int = 0,
        cache_read: int = 0,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation
        self.cache_read_input_tokens = cache_read


class _FakeBlock:
    def __init__(self, type_: str, **kwargs):
        self.type = type_
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeInnerMessage:
    def __init__(self, content, model, usage):
        self.content = content
        self.model = model
        self.usage = usage


class _FakeMsg:
    def __init__(self, type_: str, **kwargs):
        self.type = type_
        for k, v in kwargs.items():
            setattr(self, k, v)


# ── Backend ──────────────────────────────────────────────────────────────


class ScriptedBackend:
    """Deterministic backend for demos and tests.

    Not an ABC subclass anymore — just a plain class.  runtime.py
    instantiates it directly when BACKEND=scripted; live runs go through
    agent.Agent.run_turn() (in veilguard.agent).
    """

    name = "scripted"

    async def run_turn(self, config: BackendConfig) -> AsyncIterator[Any]:
        if _CURRENT_SCRIPT is None:
            yield {
                "type": "error",
                "code": "no_script",
                "message": (
                    "ScriptedBackend has no active script. "
                    "Call backends.scripted_backend.set_script(...) before invoking."
                ),
            }
            return

        persona_turns = _CURRENT_SCRIPT.get(config.persona_id, [])
        idx = _TURN_COUNTERS.get(config.persona_id, 0)
        _TURN_COUNTERS[config.persona_id] = idx + 1

        if idx >= len(persona_turns):
            logger.debug(
                f"[scripted] {config.persona_id} turn {idx} out of script; "
                "yielding empty end_turn"
            )
            usage = _FakeUsage()
            yield _FakeMsg(
                "assistant",
                message=_FakeInnerMessage(
                    content=[_FakeBlock("text", text="(end of script)")],
                    model=config.persona_model,
                    usage=usage,
                ),
            )
            yield _FakeMsg("result", result="(end of script)", stop_reason="end_turn")
            return

        turn = persona_turns[idx]
        logger.info(
            f"[scripted] {config.persona_id} turn {idx}: "
            f"text={turn.text[:40]!r} tools={[t.name for t in turn.tool_calls]}"
        )

        # Build the assistant message content blocks: text first, then tool_uses.
        content_blocks = []
        if turn.text:
            content_blocks.append(_FakeBlock("text", text=turn.text))
        for tc in turn.tool_calls:
            content_blocks.append(
                _FakeBlock("tool_use", name=tc.name, id=tc.id, input=tc.input)
            )

        usage = _FakeUsage(
            input_tokens=200,
            output_tokens=80,
            cache_creation=0 if idx > 0 else 1000,  # cold first turn
            cache_read=1000 if idx > 0 else 0,
        )

        yield _FakeMsg(
            "assistant",
            message=_FakeInnerMessage(
                content=content_blocks,
                model=config.persona_model,
                usage=usage,
            ),
        )

        # For tool-use turns: tools are dispatched OUTSIDE this backend
        # (by the runtime's MCP client + hooks).  Results come back via
        # the next call to run_turn for THIS persona.  So we end with a
        # result message whose stop_reason flags more turns coming.
        yield _FakeMsg("result", result=turn.text, stop_reason=turn.stop_reason)


__all__ = [
    "ToolCall",
    "Turn",
    "Script",
    "ScriptedBackend",
    "set_script",
    "clear_script",
    "get_turn_counter",
]
