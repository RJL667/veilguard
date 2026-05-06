"""Tool registration helper. Each tool module calls register() to add its tools to the MCP server."""


def register_all(mcp):
    """Register all tool modules with the MCP server instance."""
    from tools import clipboard, notifications, schedules, agents, tasks
    from tools import managed_tasks, daemons, teams, messaging
    from tools import utility, verify, transcripts, playbooks, file_tools
    from tools import ask_user, tool_search, plans

    # NOTE: scratchpad NOT registered as MCP tool — Claude misuses it to "save" user info.
    # Scratchpad is available inside spawn_agentic via the agentic tool handler.
    # TCMM handles memory, not scratchpad.

    for mod in [clipboard, notifications, schedules, agents, tasks,
                managed_tasks, daemons, teams, messaging,
                utility, verify, transcripts, playbooks, file_tools,
                ask_user, tool_search, plans]:
        mod.register(mcp)
