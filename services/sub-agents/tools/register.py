"""Tool registration helper. Each tool module exposes a register(mcp) entry point."""

import importlib
import logging

_log = logging.getLogger("sub-agents")


def register_all(mcp):
    """Register the EXPOSED tool modules with the MCP server.

    Modules are imported by name and skipped if absent, so a deployment that
    lacks a newer module (e.g. an older tree without client_admin/host_docs)
    starts cleanly instead of crash-looping on ImportError.

    [DELEGATE_CONSOLIDATION_2026-06-06] tasks / managed_tasks / teams /
    messaging are intentionally NOT exposed: they wrote to in-process RAM
    stores that never reached the Director ledger (two parallel task systems),
    so a user-facing task_create / team_create / start_task silently no-op'd.
    Durable, tracked, governed delegation now has ONE front door --
    veilguard-mcp `delegate_to_org` -> agent-runtime ledger -> Work Queue.
    Inline "do it now" help stays on spawn_agent / spawn_agentic (agents).
    (scratchpad is also intentionally unregistered -- Claude misuses it as a
    place to "save" user info; TCMM handles memory.)
    """
    exposed = [
        "clipboard", "notifications", "schedules", "agents", "daemons",
        "utility", "verify", "transcripts", "playbooks", "file_tools",
        "ask_user", "tool_search", "plans", "client_admin", "host_docs",
    ]
    for name in exposed:
        try:
            mod = importlib.import_module(f"tools.{name}")
        except ImportError as e:
            _log.warning("[register] skipping unavailable module tools.%s: %s", name, e)
            continue
        mod.register(mcp)
