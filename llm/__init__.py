"""veilguard.llm — LLM call layer.

Two modules:
  adapter      — AnthropicAdapter (wraps TCMM's AnthropicGenerationAdapter)
  tcmm_client  — render + ingest hooks for TCMM memory (no LLM call)

Used by veilguard.agents.Agent.run_turn:
  1. tcmm_client.render_structured(...)   memory → blocks
  2. pii.redactor.redact_blocks(...)      cache-stable redaction
  3. adapter.AnthropicAdapter(...).generate(...)  LLM
  4. pii.redactor.rehydrate_*(...)        REF → original
  5. tcmm_client.ingest_assistant(...)    memory write
"""

from .adapter import AnthropicAdapter, AdapterResult
from .scripted_adapter import (
    ScriptedAdapter,
    ToolCall,
    Turn,
    Script,
    set_script,
    clear_script,
    get_turn_counter,
)
from .tcmm_client import (
    RenderResult,
    render_structured,
    pin_system_prompt,
    pin_tool_definitions,
    ingest_user,
    ingest_assistant,
    close,
    TCMM_URL,
)

__all__ = [
    "AnthropicAdapter",
    "AdapterResult",
    "ScriptedAdapter",
    "ToolCall",
    "Turn",
    "Script",
    "set_script",
    "clear_script",
    "get_turn_counter",
    "RenderResult",
    "render_structured",
    "pin_system_prompt",
    "pin_tool_definitions",
    "ingest_user",
    "ingest_assistant",
    "close",
    "TCMM_URL",
]
