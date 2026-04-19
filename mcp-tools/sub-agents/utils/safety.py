"""Veilguard Sub-Agents: Safety validation."""

import re

# Tool risk classification
TOOL_RISK = {
    # LOW — read-only, informational
    "clipboard_paste": "LOW", "clipboard_list": "LOW", "list_roles": "LOW",
    "check_task": "LOW", "get_result": "LOW", "list_tasks": "LOW",
    "list_schedules": "LOW", "list_daemons": "LOW", "daemon_log": "LOW",
    "scratchpad_read": "LOW", "scratchpad_list": "LOW", "agent_stats": "LOW",
    "get_notifications": "LOW",
    # MEDIUM — state changes, LLM calls
    "spawn_agent": "MEDIUM", "clipboard_copy": "MEDIUM", "scratchpad_write": "MEDIUM",
    "coordinate": "MEDIUM", "start_task": "MEDIUM",
    "smart_task": "MEDIUM", "parallel_agents": "MEDIUM", "review_loop": "MEDIUM",
    "start_parallel_tasks": "MEDIUM", "wait_for_tasks": "MEDIUM",
    # HIGH — persistent changes, external effects
    "start_daemon": "HIGH", "schedule_task": "HIGH", "cancel_task": "HIGH",
    "stop_daemon": "HIGH", "delete_schedule": "HIGH", "notify": "HIGH",
    "pause_schedule": "HIGH", "resume_schedule": "HIGH", "run_schedule": "HIGH",
    # Forge tools (external)
    "forge_tool": "HIGH", "run_tool": "HIGH", "edit_tool": "HIGH",
    "delete_tool": "HIGH", "test_tool": "MEDIUM", "read_tool": "LOW",
    "list_forged_tools": "LOW",
}

# Dangerous command patterns for agentic tool loop
AGENT_DANGEROUS_PATTERNS = [
    (re.compile(r"rm\s+-rf\s+/", re.IGNORECASE), "Recursive delete from root"),
    (re.compile(r"format\s+[a-z]:", re.IGNORECASE), "Format disk drive"),
    (re.compile(r"del\s+/[sfq]", re.IGNORECASE), "Recursive delete"),
    (re.compile(r"rmdir\s+/s", re.IGNORECASE), "Remove directory tree"),
    (re.compile(r"reg\s+delete", re.IGNORECASE), "Delete registry keys"),
    (re.compile(r"net\s+user\s+.*\s+/delete", re.IGNORECASE), "Delete user account"),
    (re.compile(r"shutdown", re.IGNORECASE), "Shutdown system"),
    (re.compile(r"bcdedit", re.IGNORECASE), "Modify boot config"),
    (re.compile(r"diskpart", re.IGNORECASE), "Disk partition tool"),
    (re.compile(r"cipher\s+/w:", re.IGNORECASE), "Secure wipe"),
    (re.compile(r"powershell\s+-enc", re.IGNORECASE), "Encoded PowerShell command"),
    (re.compile(r"\|\s*powershell", re.IGNORECASE), "Piped to PowerShell"),
]

# Protected filesystem paths
PROTECTED_PATHS = {".env", ".git", "credentials", "secrets", "id_rsa", ".ssh"}


def is_path_safe(path: str) -> tuple[bool, str]:
    """Check if a file path is safe to write to."""
    for p in PROTECTED_PATHS:
        if p in path.lower():
            return False, f"BLOCKED: Cannot write to protected path containing '{p}'"
    return True, ""


def validate_command(cmd: str) -> tuple[bool, str]:
    """Check command against dangerous patterns."""
    for pat, desc in AGENT_DANGEROUS_PATTERNS:
        if pat.search(cmd):
            return False, f"BLOCKED: {desc}"
    return True, ""


def yolo_approve(tool_name: str) -> bool:
    """Auto-approve LOW risk tools in coordinator pipelines."""
    return TOOL_RISK.get(tool_name, "MEDIUM") == "LOW"
