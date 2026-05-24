"""DirectorAgent — the reactive orchestrator + proactive pre-evaluator.

Director's job (from MULTI_AGENT_PLATFORM.md §4.1):
  - Reactive stream: takes user messages, routes to ICs via create_task,
    answers solo from recall when possible, synthesizes when ICs return
  - Proactive stream: pre-evaluates dream-cycle proposals via a Haiku
    rank-pass before they hit the user's decision-ledger queue

Persona contract (agents/director.md):
  Model: reactive=claude-sonnet-4-5, rank_pass=claude-haiku-4-5
  Tools: create_task / assign_task / consult / final_synthesis,
         rank_proposals / convert_proposal / shelve_proposal /
         surface_org_memory_candidate, read_constitution / recall
  Role:  director  (only one per tenant; recall scope includes
         conv + team/knowledge, NEVER private agent memory)

This class is intentionally thin today.  As Phase 3 (proactive stream)
lands, the rank-pass entry point will live here as a separate method
that uses persona.model_for("rank_pass") instead of the default
reactive model.
"""

from __future__ import annotations

import logging

from .base import Agent

logger = logging.getLogger("veilguard.agent.director")


class DirectorAgent(Agent):
    """Tier-0 orchestrator.  Inherits the 5-step pipeline unchanged."""

    # Phase 3 hook: when the proactive pre-eval method lands, it'll
    # construct an adapter with model=persona.model_for("rank_pass")
    # (Haiku) instead of the default reactive (Sonnet).  For now the
    # base Agent.model() returns the reactive variant, which is what
    # we want for Director's user-facing turns.


__all__ = ["DirectorAgent"]
