"""ToolSearch — discover and describe available tools by keyword.

Mirrors Claude Code's ToolSearch which lets the LLM find tools
it doesn't see in its initial context. With 61+ tools, the LLM
can't memorize them all — this helps it find the right one.
"""

from utils.safety import TOOL_RISK

# Tool catalog with descriptions for search
TOOL_CATALOG = {
    # File operations
    "read_file": "Read a file from the filesystem with line numbers",
    "write_file": "Create or overwrite a file",
    "edit_file": "Make targeted string replacement edits to a file",
    "search_files": "Search for files by glob pattern",
    "grep": "Search file contents for a regex pattern",
    # Shell & web
    "run_command": "Run a shell command (CMD/PowerShell/bash)",
    "web_search": "Search the web",
    "web_fetch": "Fetch a URL and return text content",
    # Agent spawning
    "spawn_agent": "Spawn a single sub-agent with a role for a focused task",
    "spawn_agentic": "Spawn an agentic agent that uses tools iteratively (multi-turn)",
    "parallel_agents": "Run multiple sub-agents in parallel",
    "coordinate": "Coordinate complex tasks with ULTRAPLAN (plan → execute → reconcile)",
    "pipeline": "Run sequential agent pipeline (each step feeds the next)",
    "review_loop": "Agent with automatic quality review loop (work → critique → revise)",
    "run_playbook": "Run a pre-built incident playbook",
    # Delegate durable, tracked, reviewed work to the Director org — the ONE
    # way to schedule governed work; it appears in the Work Queue. For quick
    # inline help that answers in this same turn, use spawn_agent/spawn_agentic.
    "delegate_to_org": "Hand a tracked, reviewed task to the agent org (async) — shows live in the Work Queue",
    # Daemons
    "start_daemon": "Start a persistent background daemon on a schedule",
    "stop_daemon": "Stop a running daemon",
    "list_daemons": "List all daemons and their status",
    "daemon_log": "View recent observations from a daemon",
    "daemon_wait": "Wait for a daemon to record its next observation(s)",
    # Schedules
    "schedule_task": "Schedule a recurring task on an interval",
    "run_schedule": "Manually trigger a scheduled task",
    "list_schedules": "List all scheduled tasks",
    "pause_schedule": "Pause a scheduled task",
    "resume_schedule": "Resume a paused scheduled task",
    "delete_schedule": "Delete a scheduled task",
    "ask_user": "Ask the user a question and wait for response",
    # Data
    "clipboard_copy": "Copy content to a named clipboard slot (in-memory)",
    "clipboard_paste": "Retrieve from clipboard slot",
    "clipboard_list": "List all clipboard slots",
    "scratchpad_write": "Write data to shared scratchpad (persistent file)",
    "scratchpad_read": "Read from shared scratchpad",
    "scratchpad_list": "List scratchpad files",
    # Verification & security
    "verify_output": "Adversarial verification of agent output",
    "security_review": "Security review of code or configs",
    # Utility
    "list_roles": "List available agent roles and tool risk levels",
    "list_skills": "List available agent skills",
    "list_playbooks": "List available incident playbooks",
    "agent_stats": "Show call statistics, costs, per-backend breakdown",
    "system_health": "Check health of all Veilguard services",
    "afk_status": "Check AFK mode status",
    "notify": "Send a Windows toast notification",
    "get_notifications": "Get all notifications sent this session",
    # Transcripts
    "transcript_list": "List all saved agent transcripts",
    "transcript_read": "Read a saved agent transcript",
}


def register(mcp):
    @mcp.tool()
    async def tool_search(query: str) -> str:
        """Search for available tools by keyword. Use when you're not sure which tool to use.

        WHEN TO USE: When you need to find the right tool for a task but aren't sure of its name.
        Returns matching tools with descriptions.

        Args:
            query: Search keyword (e.g. "file", "search", "daemon", "team", "schedule")
        """
        query_lower = query.lower()
        matches = []
        for name, desc in TOOL_CATALOG.items():
            if query_lower in name.lower() or query_lower in desc.lower():
                risk = TOOL_RISK.get(name, "?")
                matches.append(f"- **{name}** [{risk}]: {desc}")

        if not matches:
            return f"No tools matching '{query}'. Try broader terms like: file, search, agent, task, daemon, team, schedule, web"

        return f"# Tools matching '{query}' ({len(matches)} found)\n\n" + "\n".join(matches)
