"""Typed event envelope yielded by Agent.run_turn.

Mirrors what agent-runtime/app/runtime.py was already emitting so
existing SSE consumers (LibreChat, demos, tests) keep working.

Events are plain dicts (JSON-serializable) — not dataclasses — because
they're streamed over the wire and dict shape is the lingua franca.
The helpers here are just constructors so call sites don't typo keys.
"""

from __future__ import annotations

from typing import Any, Optional


# Event TYPES (string constants, used as `event["type"]`):
RUN_START      = "run_start"
ASSISTANT_TEXT = "assistant_text"
ASSISTANT      = "assistant"        # full assistant message with content blocks
TOOL_CALL      = "tool_call"
TOOL_DISPATCH  = "tool_dispatch"
TOOL_RESULT    = "tool_result"
FINAL_RESULT   = "final_result"
USAGE          = "usage"
RUN_END        = "run_end"
ERROR          = "error"


def run_start(*, run_id: str, agent_id: str, model: str, backend: str,
              started_at: float) -> dict:
    return {
        "type": RUN_START,
        "run_id": run_id,
        "agent_id": agent_id,
        "model": model,
        "backend": backend,
        "started_at": started_at,
    }


def assistant_text(text: str) -> dict:
    return {"type": ASSISTANT_TEXT, "text": text}


def assistant(content: list[dict], usage: dict, stop_reason: str) -> dict:
    return {
        "type": ASSISTANT,
        "message": {
            "content": content,
            "usage": usage,
            "stop_reason": stop_reason,
        },
    }


def tool_call(name: str, id: str, input: dict) -> dict:
    return {"type": TOOL_CALL, "name": name, "id": id, "input": input}


def tool_dispatch(name: str, id: str) -> dict:
    return {"type": TOOL_DISPATCH, "name": name, "id": id}


def tool_result(id: str, is_error: bool, content: Any = None) -> dict:
    return {"type": TOOL_RESULT, "id": id, "is_error": is_error,
            "content": content}


def final_result(result: str, stop_reason: str = "end_turn") -> dict:
    return {"type": FINAL_RESULT, "result": result, "stop_reason": stop_reason}


def usage(*, tokens_input_total: int = 0, tokens_input_new: int = 0,
          tokens_output: int = 0, cache_create: int = 0, cache_read: int = 0,
          model: str = "", iterations: int = 1) -> dict:
    total = tokens_input_total or (tokens_input_new + cache_create + cache_read)
    cache_hit_rate = round(cache_read / total, 4) if total else 0.0
    return {
        "type": USAGE,
        "tokens_input_total": total,
        "tokens_input_new": tokens_input_new,
        "tokens_output": tokens_output,
        "cache_create": cache_create,
        "cache_read": cache_read,
        "cache_hit_rate": cache_hit_rate,
        "model": model,
        "iterations": iterations,
    }


def run_end(*, run_id: str, ended_at: float, stop_reason: str) -> dict:
    return {
        "type": RUN_END,
        "run_id": run_id,
        "ended_at": ended_at,
        "stop_reason": stop_reason,
    }


def error(code: str, message: str) -> dict:
    return {"type": ERROR, "code": code, "message": message}


__all__ = [
    "RUN_START", "ASSISTANT_TEXT", "ASSISTANT", "TOOL_CALL",
    "TOOL_DISPATCH", "TOOL_RESULT", "FINAL_RESULT", "USAGE",
    "RUN_END", "ERROR",
    "run_start", "assistant_text", "assistant", "tool_call",
    "tool_dispatch", "tool_result", "final_result", "usage",
    "run_end", "error",
]
