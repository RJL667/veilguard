"""ConsultantAgent — existing security personas as on-demand specialists.

Consultants are the personas that pre-existed the multi-agent platform
(per memory `architecture_agents_persona_convention`):

  agents/phishing-analyst.md
  agents/threat-analyst.md
  agents/report-writer.md

Per spec §3.6 decision 2026-05-21: "Consultants read context from
task.inputs, not recall()" — they're invoked synchronously by an IC or
Director via the `consult` tool, given explicit inputs, and return a
result.  They don't see team memory; cross-team recall would leak.

Multi-tenant correctness (spec decision 2026-05-21): consultant
private memory is namespaced `(consultant_id, tenant_id)`.  The
PIISessionStore + TCMM both honor `tenant_id` in their session key, so
this is structurally enforced — no extra code needed in this subclass.
"""

from __future__ import annotations

from .base import Agent


class ConsultantAgent(Agent):
    """On-demand specialist.  Inherits the base pipeline."""


__all__ = ["ConsultantAgent"]
