"""veilguard.agent — unified Agent harness.

One pipeline, swappable persona + tools + model:

  agent.Agent.run_turn(messages, ctx)  →  AsyncIterator[Event]
    STEP 0  ingest raw user message → TCMM
    STEP 1  TCMM render → system blocks (raw, with cache_control)
    STEP 2  PII redact → block-stable bytes
    STEP 3  AnthropicAdapter → api.anthropic.com
    STEP 4  rehydrate REF tokens → caller sees raw
    STEP 5  ingest raw assistant text → TCMM

Subclasses:
  ChatAgent       — LibreChat user-facing (config from request body)
  (more in PR #4) — DirectorAgent / ICAgent / CriticAgent / ConsultantAgent
"""

from .base import Agent, TurnContext, _last_user_text
from .chat import ChatAgent, is_side_channel
from .director import DirectorAgent
from .ic import ICAgent
from .critic import CriticClaimAgent, CriticProseAgent
from .consultant import ConsultantAgent
from .registry import agent_for
from .persona import PersonaSpec, PersonaRegistry, load_personas
from . import events

__all__ = [
    "Agent",
    "TurnContext",
    "ChatAgent",
    "DirectorAgent",
    "ICAgent",
    "CriticClaimAgent",
    "CriticProseAgent",
    "ConsultantAgent",
    "agent_for",
    "is_side_channel",
    "PersonaSpec",
    "PersonaRegistry",
    "load_personas",
    "events",
]
