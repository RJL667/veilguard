"""Tool registration helper. Each tool module calls register() to add its tools to the MCP server."""


def register_all(mcp):
    """Register all tool modules with the MCP server instance."""
    from tools import clipboard, notifications, schedules, agents, tasks
    from tools import managed_tasks, daemons, teams, messaging
    from tools import utility, verify, transcripts, playbooks, file_tools
    from tools import ask_user, tool_search, plans, client_admin
    from tools import host_docs

    # NOTE: scratchpad NOT registered as MCP tool — Claude misuses it to "save" user info.
    # Scratchpad is available inside spawn_agentic via the agentic tool handler.
    # TCMM handles memory, not scratchpad.

    for mod in [clipboard, notifications, schedules, agents, tasks,
                managed_tasks, daemons, teams, messaging,
                utility, verify, transcripts, playbooks, file_tools,
                ask_user, tool_search, plans,
                # Phase C/D: daemon task introspection + permission_level
                # MCP tools (list_my_tasks / task_status / cancel_task /
                # get_permission_level / set_permission_level).
                client_admin,
                # [HOST_DOC_READ_2026_06_01] read_pdf on the host (parses
                # Windows-path PDFs the container documents server can't see).
                host_docs]:
        mod.register(mcp)
