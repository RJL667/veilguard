"""Backend config dataclass — shared by the scripted backend.

There was an LLMBackend abstraction here (SdkBackend / SsoBackend /
ScriptedBackend with a registry).  It was overengineered.  The live
LLM path is `agent.Agent.run_turn` (in the top-level `agent/` package); the only
remaining "backend" is `ScriptedBackend` for deterministic tests.

What's left here: the BackendConfig dataclass that the scripted backend
takes as input.  Kept as a typed struct so the call site doesn't pass a
sprawling kwargs dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BackendConfig:
    """Per-call configuration for the scripted backend.

    Used to be shared with SDK / SSO backends; those are gone.  Kept
    around because ScriptedBackend.run_turn() still takes it.
    """

    persona_id: str
    persona_system_prompt: str
    persona_model: str
    persona_tools: list[str]
    persona_subagents: dict[str, Any]
    system_blocks: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    mcp_servers: dict[str, Any]
    hooks: dict[str, Any]
    include_partial_messages: bool = True


__all__ = ["BackendConfig"]
