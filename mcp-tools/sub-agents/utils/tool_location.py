"""Tool location mapping — which tools run on client vs cloud."""

# Tools that execute on the client's local machine via WebSocket daemon
CLIENT_TOOLS = {
    # File operations
    "read_file",
    "write_file",
    "edit_file",
    "search_files",
    "grep",
    # Command execution
    "run_command",
    # Web operations
    "web_search",
    "web_fetch",
    # Host-exec tools
    "run_cmd",
    "run_powershell",
    "run_docker",
    "run_git",
    "host_file_read",
    "host_file_write",
}

# Tools that stay on the cloud (orchestration, memory, scratchpad)
CLOUD_TOOLS = {
    "write_scratchpad",
    "read_scratchpad",
    "tcmm_recall",
    "todo_write",
    # All MCP-registered tools (agents, tasks, daemons, etc.) are cloud-side
}


def is_client_tool(name: str) -> bool:
    """Check if a tool should be routed to the client daemon."""
    return name in CLIENT_TOOLS
