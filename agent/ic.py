"""ICAgent — individual-contributor base for Researcher / Builder.

ICs (from MULTI_AGENT_PLATFORM.md §4.2-4.3):
  - Receive Tasks from Director via the inbox poller
  - Execute the brief; persist findings via ledger tools
    (accept_task → add_comment / attach_output → submit_for_review)
  - Submit completed work to a Critic for promotion

The two ICs that exist today:
  - Researcher (agents/researcher.md):
      tools  web (web_search, fetch), memory (recall, observe),
             task (accept_task, add_comment, attach_output,
             submit_for_review), peer (dm)
  - Builder (agents/builder.md):
      tools  filesystem (file_read, file_write, file_edit),
             shell (run_command, ...), client_daemon (host_file_*),
             memory (recall, observe), task (...)

Both share the same pipeline.  The only difference is the persona file
(model + tools + system prompt).  No method override needed today.
Future role-specific bits (e.g. Builder's approval-gate hook) would
land as instance behavior or subclasses of ICAgent.
"""

from __future__ import annotations

from .base import Agent


class ICAgent(Agent):
    """Researcher / Builder / general IC.  Inherits the base pipeline."""


__all__ = ["ICAgent"]
