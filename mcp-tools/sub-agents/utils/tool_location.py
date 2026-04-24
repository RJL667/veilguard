"""Tool location mapping — which tools run on client vs cloud.

Client-routed tools require the user's Veilguard client daemon to be
connected (see Stage-1 multi-tenancy in core/client_bridge.py).
Without a live daemon, client-routed tool calls fail with
``No client daemon connected for this user``.  Keep this list narrow
— only tools that genuinely need the client's local environment
(filesystem / processes / user-local network) should be here.
``web_search`` was historically in this set but it doesn't need any
client-side state — searches go to Google either way — so it's been
moved to cloud where we back it with Vertex AI grounding and
authenticate via the VM's service account.  Users get search even
without a daemon running.
"""

# Tools that execute on the client's local machine via WebSocket daemon
CLIENT_TOOLS = {
    # File operations — need the client's filesystem
    "read_file",
    "write_file",
    "edit_file",
    "search_files",
    "grep",
    # Command execution — need the client's shell/processes
    "run_command",
    # Host-exec tools — same, host-side shells
    "run_cmd",
    "run_powershell",
    "run_docker",
    "run_git",
    "host_file_read",
    "host_file_write",
    # web_fetch was historically here for "fetch from user's IP".
    # Moved to CLOUD_TOOLS: sub-agent research calls were failing
    # with "No client daemon connected" whenever the Agents panel
    # session didn't have a daemon tied, and the cloud httpx path
    # is good enough for 99% of cases (public URLs). If a user ever
    # needs cookie'd / intranet fetches we'll add a per-call override.
}

# Tools that stay on the cloud (orchestration, memory, scratchpad,
# and anything that authenticates via the VM's Vertex SA).
CLOUD_TOOLS = {
    "write_scratchpad",
    "read_scratchpad",
    "tcmm_recall",
    "todo_write",
    # Google search — routed through agentic.py to the ``web`` MCP's
    # Vertex-grounded google_search.  Deliberately cloud-side so it
    # works regardless of whether the user has a client daemon
    # connected.  See agentic.py's handle_tool for the dispatch.
    "web_search",
    # URL fetch via server-side httpx (agentic.py handle_tool).
    # Follows redirects, html2text'd to ~3000 chars. No daemon needed.
    "web_fetch",
    # All MCP-registered tools (agents, tasks, daemons, etc.) are cloud-side
}


def is_client_tool(name: str) -> bool:
    """Check if a tool should be routed to the client daemon."""
    return name in CLIENT_TOOLS
