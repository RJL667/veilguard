"""Map persona.role → Agent subclass.

The runtime calls `agent_for(persona)` to get the right concrete class
for a persona it just loaded from agents/*.md.  This is the only place
that knows about the role-to-class mapping; everything else just sees
the Agent base interface.
"""

from __future__ import annotations

from .base import Agent
from .chat import ChatAgent
from .director import DirectorAgent
from .ic import ICAgent
from .critic import CriticClaimAgent, CriticProseAgent
from .consultant import ConsultantAgent
from .persona import PersonaSpec


# persona.role → Agent subclass.  Role strings come from the agent
# markdown frontmatter (see agents/director.md etc.).
_ROLE_TO_CLASS: dict[str, type[Agent]] = {
    "director":   DirectorAgent,
    "ic":         ICAgent,
    "consultant": ConsultantAgent,
    "chat":       ChatAgent,   # synthesized personas only; not loaded
}


# Persona ID → Agent subclass override (for the two Critics whose
# role is "ic" in their markdown but who need the critic split per §4.4).
_PERSONA_ID_TO_CLASS: dict[str, type[Agent]] = {
    "critic-claim": CriticClaimAgent,
    "critic-prose": CriticProseAgent,
}


def agent_for(persona: PersonaSpec) -> type[Agent]:
    """Return the Agent subclass for this persona.

    Lookup order:
      1. Persona-id override (critic-claim → CriticClaimAgent, etc.)
      2. Role mapping (director → DirectorAgent, ic → ICAgent, ...)
      3. Default to ICAgent (safe fallback for unknown roles)
    """
    cls = _PERSONA_ID_TO_CLASS.get(persona.agent_id)
    if cls is not None:
        return cls
    cls = _ROLE_TO_CLASS.get(persona.role)
    if cls is not None:
        return cls
    return ICAgent  # safe default


__all__ = ["agent_for"]
