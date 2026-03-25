"""Sub-Agent MCP Tool Server for Veilguard.

Spawns sub-agents by calling the Gemini API directly.
Runs ON THE WINDOWS HOST (not inside Docker).

Start: python mcp-tools/sub-agents/server.py --sse --port 8809
"""

import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "sub-agents",
    instructions=(
        "Sub-agent orchestration tools. Spawn specialized AI workers to handle "
        "subtasks in parallel, with review loops, and role-based delegation."
    ),
)

# Config
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
DEFAULT_MODEL = os.environ.get("SUB_AGENT_MODEL", "gemini-2.5-flash")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
TIMEOUT = int(os.environ.get("SUB_AGENT_TIMEOUT", "120"))

# Load API key from .env if not in environment
if not GOOGLE_API_KEY:
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("GOOGLE_API_KEY="):
                GOOGLE_API_KEY = line.split("=", 1)[1].strip()
                break

# ── Predefined Roles ─────────────────────────────────────────────────────────

ROLES = {
    "analyst": {
        "name": "Phishing Analyst",
        "system": (
            "You are a senior phishing analyst at Phishield. You analyze emails, URLs, "
            "and attachments for phishing indicators. You identify social engineering tactics, "
            "check sender reputation, analyze URL patterns, and assess threat level. "
            "Always provide a confidence score (0-100) and clear reasoning."
        ),
    },
    "threat": {
        "name": "Threat Intelligence Analyst",
        "system": (
            "You are a threat intelligence analyst at Phishield. You research threat actors, "
            "TTPs (Tactics, Techniques, Procedures), IOCs (Indicators of Compromise), and "
            "map findings to MITRE ATT&CK framework. Provide structured intelligence reports."
        ),
    },
    "writer": {
        "name": "Report Writer",
        "system": (
            "You are a professional security report writer at Phishield. You take raw analysis "
            "and findings and produce clear, executive-ready reports. Use proper formatting, "
            "include risk ratings, recommendations, and action items. Write for both technical "
            "and non-technical audiences."
        ),
    },
    "coder": {
        "name": "Code Assistant",
        "system": (
            "You are a Python/JavaScript developer. Write clean, well-structured code. "
            "Include error handling and comments. When asked to fix code, explain what was wrong."
        ),
    },
    "researcher": {
        "name": "Research Assistant",
        "system": (
            "You are a thorough research assistant. Gather information, synthesize findings, "
            "and present them in a structured format with sources and confidence levels."
        ),
    },
    "critic": {
        "name": "Quality Reviewer",
        "system": (
            "You are a critical reviewer. Evaluate the provided work for accuracy, completeness, "
            "clarity, and quality. Identify specific issues and suggest concrete improvements. "
            "Rate overall quality as: PASS, NEEDS_REVISION, or FAIL. "
            "If PASS, respond with just 'PASS: [brief comment]'. "
            "If not, list specific issues to fix."
        ),
    },
}


async def _call_gemini(
    system_prompt: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> str:
    """Call Gemini API directly and return the response text."""
    if not GOOGLE_API_KEY:
        return "Error: GOOGLE_API_KEY not configured"

    headers = {
        "Authorization": f"Bearer {GOOGLE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": 4096,
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(GEMINI_URL, json=payload, headers=headers)

            if response.status_code != 200:
                return f"Error: Gemini API returned {response.status_code}: {response.text[:500]}"

            data = response.json()
            return data["choices"][0]["message"]["content"]
    except httpx.TimeoutException:
        return f"Error: Sub-agent timed out after {TIMEOUT}s"
    except Exception as e:
        return f"Error calling Gemini: {e}"


# ── Clipboard ─────────────────────────────────────────────────────────────────

_clipboard: dict[str, str] = {}  # named clipboard slots


@mcp.tool()
async def clipboard_copy(name: str, content: str) -> str:
    """Copy content to a named clipboard slot. Use to pass data between tools.

    Args:
        name: Slot name (e.g. "analysis", "data", "draft")
        content: Content to store
    """
    _clipboard[name] = content
    preview = content[:100] + "..." if len(content) > 100 else content
    return f"Copied to clipboard '{name}' ({len(content)} chars): {preview}"


@mcp.tool()
async def clipboard_paste(name: str) -> str:
    """Retrieve content from a named clipboard slot.

    Args:
        name: Slot name to retrieve
    """
    if name not in _clipboard:
        available = ", ".join(_clipboard.keys()) if _clipboard else "(empty)"
        return f"Error: Clipboard '{name}' not found. Available: {available}"
    return _clipboard[name]


@mcp.tool()
async def clipboard_list() -> str:
    """List all clipboard slots and their content preview."""
    if not _clipboard:
        return "Clipboard is empty."
    lines = ["# Clipboard Slots\n"]
    for name, content in _clipboard.items():
        preview = content[:80].replace("\n", " ") + "..." if len(content) > 80 else content.replace("\n", " ")
        lines.append(f"- **{name}** ({len(content)} chars): {preview}")
    return "\n".join(lines)


# ── Notifications ─────────────────────────────────────────────────────────────

_notifications: list[dict] = []


@mcp.tool()
async def notify(message: str, title: str = "Veilguard") -> str:
    """Send a Windows toast notification to the user.

    Use this to alert the user when a background task completes or needs attention.

    Args:
        message: Notification message
        title: Notification title (default "Veilguard")
    """
    import subprocess
    _notifications.append({"title": title, "message": message, "time": time.time()})

    # Windows balloon notification via NotifyIcon
    # Escape single quotes for PowerShell
    safe_title = title.replace("'", "''")
    safe_msg = message.replace("'", "''")

    ps_script = f"""
[void] [System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms')
$n = New-Object System.Windows.Forms.NotifyIcon
$n.Icon = [System.Drawing.SystemIcons]::Information
$n.BalloonTipTitle = '{safe_title}'
$n.BalloonTipText = '{safe_msg}'
$n.Visible = $true
$n.ShowBalloonTip(5000)
Start-Sleep -Seconds 6
$n.Dispose()
"""

    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return f"Notification sent: {title} — {message}"
    except Exception as e:
        return f"Notification stored (failed: {e}): {title} — {message}"


@mcp.tool()
async def get_notifications() -> str:
    """Get all notifications sent this session."""
    if not _notifications:
        return "No notifications."
    lines = ["# Notifications\n"]
    for n in _notifications:
        t = time.strftime("%H:%M:%S", time.localtime(n["time"]))
        lines.append(f"- [{t}] **{n['title']}**: {n['message']}")
    return "\n".join(lines)


# ── Scheduled Tasks ───────────────────────────────────────────────────────────

_schedules: dict[str, dict] = {}
_scheduler_running = False


async def _scheduler_loop():
    """Background loop that checks and runs scheduled tasks."""
    global _scheduler_running
    _scheduler_running = True
    while _scheduler_running:
        now = time.time()
        for sched_id, sched in list(_schedules.items()):
            if not sched.get("enabled", True):
                continue
            next_run = sched.get("next_run", 0)
            if now >= next_run:
                # Time to run
                interval = sched["interval_seconds"]
                sched["next_run"] = now + interval
                sched["last_run"] = now
                sched["run_count"] = sched.get("run_count", 0) + 1

                # Run the task
                task_text = sched["task"]
                role = sched.get("role", "researcher")
                sys_prompt = ROLES.get(role, ROLES["researcher"])["system"]
                role_name = ROLES.get(role, ROLES["researcher"])["name"]
                model = sched.get("model", DEFAULT_MODEL)

                task_id = _short_id()
                bg_task = BackgroundTask(
                    id=task_id,
                    task=f"[scheduled:{sched_id}] {task_text[:150]}",
                    role=role,
                    role_name=f"{role_name} (scheduled)",
                    status=TaskStatus.PENDING,
                    model=model,
                    created_at=now,
                )
                _tasks[task_id] = bg_task

                async def _run(bt=bg_task, sp=sys_prompt, tt=task_text, m=model):
                    bt.status = TaskStatus.RUNNING
                    bt.started_at = time.time()
                    try:
                        result = await _call_gemini(sp, tt, model=m)
                        bt.result = result
                        bt.status = TaskStatus.DONE
                    except Exception as e:
                        bt.error = str(e)
                        bt.status = TaskStatus.FAILED
                    finally:
                        bt.finished_at = time.time()

                asyncio.create_task(_run())
                sched["last_task_id"] = task_id

        await asyncio.sleep(10)  # Check every 10 seconds


def _ensure_scheduler():
    """Start the scheduler loop if not running."""
    global _scheduler_running
    if not _scheduler_running:
        asyncio.get_event_loop().create_task(_scheduler_loop())


@mcp.tool()
async def schedule_task(
    name: str,
    task: str,
    interval_minutes: int = 60,
    role: str = "researcher",
    model: str = "",
) -> str:
    """Schedule a recurring task that runs automatically on an interval.

    The task runs as a background sub-agent on the given interval.
    Results are stored as background tasks — use list_tasks to see them.

    Args:
        name: Unique name for this schedule (e.g. "threat-check", "report-gen")
        task: The task prompt to run each time
        interval_minutes: Run every N minutes (default 60, min 1)
        role: Agent role to use (default: researcher)
        model: Model override. Empty = gemini-2.5-flash.
    """
    _ensure_scheduler()

    interval = max(int(interval_minutes), 1) * 60
    sched_id = name.lower().replace(" ", "-")

    _schedules[sched_id] = {
        "name": name,
        "task": task,
        "interval_seconds": interval,
        "role": role,
        "model": model or DEFAULT_MODEL,
        "enabled": True,
        "next_run": time.time() + interval,  # First run after one interval
        "created_at": time.time(),
        "run_count": 0,
    }

    return (
        f"Scheduled task `{sched_id}` created.\n"
        f"Task: {task[:100]}...\n"
        f"Runs every {interval_minutes} min with role '{role}'\n"
        f"First run in {interval_minutes} min. Use `run_schedule(\"{sched_id}\")` to run immediately."
    )


@mcp.tool()
async def run_schedule(name: str) -> str:
    """Manually trigger a scheduled task to run now.

    Args:
        name: Schedule name/ID
    """
    sched_id = name.lower().replace(" ", "-")
    if sched_id not in _schedules:
        available = ", ".join(_schedules.keys()) if _schedules else "(none)"
        return f"Error: Schedule '{sched_id}' not found. Available: {available}"

    sched = _schedules[sched_id]
    sched["next_run"] = 0  # Trigger on next scheduler tick
    return f"Schedule `{sched_id}` will run on next tick (~10s). Use `list_tasks` to see the result."


@mcp.tool()
async def list_schedules() -> str:
    """List all scheduled tasks and their status."""
    if not _schedules:
        return "No scheduled tasks."

    lines = ["# Scheduled Tasks\n"]
    for sid, s in _schedules.items():
        status = "ENABLED" if s.get("enabled", True) else "PAUSED"
        interval_min = s["interval_seconds"] // 60
        runs = s.get("run_count", 0)
        last_tid = s.get("last_task_id", "—")
        next_in = max(0, int(s.get("next_run", 0) - time.time()))

        lines.append(
            f"- **{sid}** [{status}] every {interval_min}min | "
            f"runs: {runs} | next in: {next_in}s | last task: `{last_tid}`"
        )
        lines.append(f"  Task: {s['task'][:80]}...")

    return "\n".join(lines)


@mcp.tool()
async def pause_schedule(name: str) -> str:
    """Pause a scheduled task.

    Args:
        name: Schedule name/ID
    """
    sched_id = name.lower().replace(" ", "-")
    if sched_id not in _schedules:
        return f"Error: Schedule '{sched_id}' not found"
    _schedules[sched_id]["enabled"] = False
    return f"Schedule `{sched_id}` paused."


@mcp.tool()
async def resume_schedule(name: str) -> str:
    """Resume a paused scheduled task.

    Args:
        name: Schedule name/ID
    """
    sched_id = name.lower().replace(" ", "-")
    if sched_id not in _schedules:
        return f"Error: Schedule '{sched_id}' not found"
    _schedules[sched_id]["enabled"] = True
    _schedules[sched_id]["next_run"] = time.time() + _schedules[sched_id]["interval_seconds"]
    return f"Schedule `{sched_id}` resumed. Next run in {_schedules[sched_id]['interval_seconds'] // 60} min."


@mcp.tool()
async def delete_schedule(name: str) -> str:
    """Delete a scheduled task permanently.

    Args:
        name: Schedule name/ID
    """
    sched_id = name.lower().replace(" ", "-")
    if sched_id not in _schedules:
        return f"Error: Schedule '{sched_id}' not found"
    del _schedules[sched_id]
    return f"Schedule `{sched_id}` deleted."


# ── Background Task Store ─────────────────────────────────────────────────────


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackgroundTask:
    id: str
    task: str
    role: str
    role_name: str
    status: TaskStatus
    model: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None
    asyncio_task: Optional[asyncio.Task] = None


# In-memory task store
_tasks: dict[str, BackgroundTask] = {}


def _short_id() -> str:
    """Generate a short readable task ID."""
    return uuid.uuid4().hex[:8]


def _elapsed(start: float, end: Optional[float] = None) -> str:
    """Human-readable elapsed time."""
    secs = (end or time.time()) - start
    if secs < 60:
        return f"{secs:.1f}s"
    mins = secs / 60
    return f"{mins:.1f}m"


# ── Tools (Synchronous) ─────────────────────────────────────────────────────


@mcp.tool()
async def spawn_agent(task: str, role: str = "", system_prompt: str = "", model: str = "") -> str:
    """Spawn a sub-agent to handle a specific task.

    The sub-agent calls Gemini directly with a role-based system prompt.
    Use this to delegate subtasks to specialized workers.

    Args:
        task: The task/question for the sub-agent to handle
        role: Predefined role name: analyst, threat, writer, coder, researcher, critic.
              Or leave empty and provide a custom system_prompt.
        system_prompt: Custom system prompt (overrides role if both provided)
        model: Model to use. Empty = gemini-2.5-flash
    """
    # Resolve system prompt
    if system_prompt:
        sys_prompt = system_prompt
        role_name = "Custom Agent"
    elif role in ROLES:
        sys_prompt = ROLES[role]["system"]
        role_name = ROLES[role]["name"]
    else:
        available = ", ".join(ROLES.keys())
        return f"Error: Unknown role '{role}'. Available: {available}. Or provide a custom system_prompt."

    use_model = model if model else DEFAULT_MODEL

    result = await _call_gemini(sys_prompt, task, model=use_model)

    return f"# Sub-Agent: {role_name} ({use_model})\n\n{result}"


@mcp.tool()
async def parallel_agents(tasks: str) -> str:
    """Spawn multiple sub-agents in parallel and collect results.

    Each task runs concurrently — much faster than sequential spawning.

    Args:
        tasks: JSON string — list of task objects:
            [
                {"task": "Analyze this email header...", "role": "analyst"},
                {"task": "Write an executive summary...", "role": "writer"},
                {"task": "Check this URL for threats...", "role": "threat"}
            ]
            Each object can have: task (required), role, system_prompt, model
    """
    try:
        task_list = json.loads(tasks)
    except json.JSONDecodeError:
        return "Error: 'tasks' must be a valid JSON array"

    if not isinstance(task_list, list) or not task_list:
        return "Error: 'tasks' must be a non-empty JSON array"

    async def run_one(idx: int, item: dict) -> str:
        role = item.get("role", "")
        sys_prompt = item.get("system_prompt", "")
        model = item.get("model", DEFAULT_MODEL)
        task_text = item.get("task", "")

        if not task_text:
            return f"## Agent {idx + 1}: Error — no task provided"

        if sys_prompt:
            prompt = sys_prompt
            role_name = "Custom Agent"
        elif role in ROLES:
            prompt = ROLES[role]["system"]
            role_name = ROLES[role]["name"]
        else:
            prompt = "You are a helpful assistant."
            role_name = role or "General"

        result = await _call_gemini(prompt, task_text, model=model)
        return f"## Agent {idx + 1}: {role_name}\n\n{result}"

    # Run all in parallel
    results = await asyncio.gather(
        *[run_one(i, t) for i, t in enumerate(task_list)],
        return_exceptions=True,
    )

    output = [f"# Parallel Agents — {len(task_list)} tasks\n"]
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            output.append(f"## Agent {i + 1}: Error\n{r}\n")
        else:
            output.append(str(r) + "\n")

    return "\n---\n".join(output)


@mcp.tool()
async def review_loop(
    task: str,
    role: str = "coder",
    critic_prompt: str = "",
    max_rounds: int = 3,
    model: str = "",
) -> str:
    """Spawn a sub-agent with automatic quality review loop.

    1. Agent performs the task
    2. Critic reviews the output
    3. If critic says NEEDS_REVISION or FAIL, agent retries with feedback
    4. Repeats up to max_rounds times

    Args:
        task: The task for the agent to perform
        role: Role for the worker agent (default: coder)
        critic_prompt: Custom critic instructions. Empty = use default critic role.
        max_rounds: Maximum revision rounds (default 3)
        model: Model to use. Empty = gemini-2.5-flash
    """
    use_model = model if model else DEFAULT_MODEL

    # Worker config
    if role in ROLES:
        worker_prompt = ROLES[role]["system"]
        worker_name = ROLES[role]["name"]
    else:
        worker_prompt = "You are a helpful assistant."
        worker_name = role

    # Critic config
    critic_sys = critic_prompt if critic_prompt else ROLES["critic"]["system"]

    output_log = [f"# Review Loop: {worker_name} ({max_rounds} max rounds)\n"]
    current_task = task

    for round_num in range(1, max_rounds + 1):
        output_log.append(f"## Round {round_num}: Worker")

        # Worker produces output
        worker_result = await _call_gemini(worker_prompt, current_task, model=use_model)
        output_log.append(worker_result)

        # Critic reviews
        output_log.append(f"\n## Round {round_num}: Critic Review")
        review_input = (
            f"Original task:\n{task}\n\n"
            f"Worker output:\n{worker_result}\n\n"
            f"Review this output. Is it complete, accurate, and high quality?"
        )
        critic_result = await _call_gemini(critic_sys, review_input, model=use_model, temperature=0.3)
        output_log.append(critic_result)

        # Check if passed
        if critic_result.strip().upper().startswith("PASS"):
            output_log.append(f"\n**APPROVED in round {round_num}**")
            break

        # Prepare retry with feedback
        if round_num < max_rounds:
            current_task = (
                f"Original task:\n{task}\n\n"
                f"Your previous attempt:\n{worker_result}\n\n"
                f"Reviewer feedback:\n{critic_result}\n\n"
                f"Please revise your work based on the feedback above."
            )
            output_log.append(f"\n*Sending back for revision...*\n")
        else:
            output_log.append(f"\n**Max rounds reached — returning best effort**")

    return "\n\n".join(output_log)


@mcp.tool()
async def list_roles() -> str:
    """List all available predefined agent roles.

    Returns the role names, descriptions, and system prompts.
    """
    output = ["# Available Agent Roles\n"]
    for key, role in ROLES.items():
        output.append(f"## `{key}` — {role['name']}")
        output.append(f"{role['system'][:200]}...")
        output.append("")
    return "\n".join(output)


# ── Background Task Tools ─────────────────────────────────────────────────────


@mcp.tool()
async def start_task(task: str, role: str = "researcher", system_prompt: str = "", model: str = "") -> str:
    """Start a background task that runs asynchronously. Returns immediately with a task ID.

    The task runs in the background while you continue working. Use check_task or
    get_result to monitor progress and retrieve output when ready.

    Args:
        task: The task/question for the sub-agent
        role: Predefined role: analyst, threat, writer, coder, researcher, critic
        system_prompt: Custom system prompt (overrides role)
        model: Model to use. Empty = gemini-2.5-flash
    """
    # Resolve role
    if system_prompt:
        sys_prompt = system_prompt
        role_name = "Custom Agent"
    elif role in ROLES:
        sys_prompt = ROLES[role]["system"]
        role_name = ROLES[role]["name"]
    else:
        available = ", ".join(ROLES.keys())
        return f"Error: Unknown role '{role}'. Available: {available}"

    use_model = model if model else DEFAULT_MODEL
    task_id = _short_id()

    bg_task = BackgroundTask(
        id=task_id,
        task=task[:200],  # truncate for display
        role=role or "custom",
        role_name=role_name,
        status=TaskStatus.PENDING,
        model=use_model,
        created_at=time.time(),
    )
    _tasks[task_id] = bg_task

    async def _run():
        bg_task.status = TaskStatus.RUNNING
        bg_task.started_at = time.time()
        try:
            result = await _call_gemini(sys_prompt, task, model=use_model)
            bg_task.result = result
            bg_task.status = TaskStatus.DONE
        except asyncio.CancelledError:
            bg_task.status = TaskStatus.CANCELLED
            bg_task.error = "Cancelled by user"
        except Exception as e:
            bg_task.error = str(e)
            bg_task.status = TaskStatus.FAILED
        finally:
            bg_task.finished_at = time.time()

    bg_task.asyncio_task = asyncio.create_task(_run())

    return (
        f"Background task started: `{task_id}`\n"
        f"Role: {role_name} ({use_model})\n"
        f"Task: {task[:100]}...\n\n"
        f"Use `check_task(\"{task_id}\")` to monitor progress.\n"
        f"Use `get_result(\"{task_id}\")` when done."
    )


@mcp.tool()
async def check_task(task_id: str) -> str:
    """Check the status of a background task.

    Args:
        task_id: The task ID returned by start_task
    """
    if task_id not in _tasks:
        return f"Error: Unknown task ID '{task_id}'"

    t = _tasks[task_id]
    elapsed = _elapsed(t.created_at)

    status_emoji = {
        TaskStatus.PENDING: "PENDING",
        TaskStatus.RUNNING: "RUNNING",
        TaskStatus.DONE: "DONE",
        TaskStatus.FAILED: "FAILED",
        TaskStatus.CANCELLED: "CANCELLED",
    }

    lines = [
        f"Task `{task_id}`: {status_emoji[t.status]}",
        f"Role: {t.role_name} ({t.model})",
        f"Elapsed: {elapsed}",
        f"Task: {t.task[:100]}",
    ]

    if t.status == TaskStatus.DONE:
        lines.append(f"\nResult ready — use `get_result(\"{task_id}\")` to retrieve.")
    elif t.status == TaskStatus.FAILED:
        lines.append(f"\nError: {t.error}")

    return "\n".join(lines)


@mcp.tool()
async def get_result(task_id: str) -> str:
    """Get the full result of a completed background task.

    Args:
        task_id: The task ID returned by start_task
    """
    if task_id not in _tasks:
        return f"Error: Unknown task ID '{task_id}'"

    t = _tasks[task_id]

    if t.status == TaskStatus.RUNNING or t.status == TaskStatus.PENDING:
        elapsed = _elapsed(t.created_at)
        return f"Task `{task_id}` is still {t.status.value} ({elapsed} elapsed). Check back later."

    if t.status == TaskStatus.FAILED:
        return f"Task `{task_id}` FAILED: {t.error}"

    if t.status == TaskStatus.CANCELLED:
        return f"Task `{task_id}` was cancelled."

    runtime = _elapsed(t.started_at, t.finished_at) if t.started_at and t.finished_at else "?"
    return (
        f"# Task `{task_id}` — {t.role_name} (completed in {runtime})\n\n"
        f"{t.result}"
    )


@mcp.tool()
async def list_tasks() -> str:
    """List all background tasks and their current status."""
    if not _tasks:
        return "No background tasks."

    lines = ["# Background Tasks\n"]
    lines.append(f"{'ID':<10} {'Status':<12} {'Role':<20} {'Elapsed':<10} Task")
    lines.append("-" * 80)

    for t in sorted(_tasks.values(), key=lambda x: x.created_at, reverse=True):
        elapsed = _elapsed(t.created_at, t.finished_at if t.finished_at else None)
        lines.append(
            f"{t.id:<10} {t.status.value:<12} {t.role_name:<20} {elapsed:<10} {t.task[:30]}"
        )

    return "\n".join(lines)


@mcp.tool()
async def cancel_task(task_id: str) -> str:
    """Cancel a running background task.

    Args:
        task_id: The task ID to cancel
    """
    if task_id not in _tasks:
        return f"Error: Unknown task ID '{task_id}'"

    t = _tasks[task_id]
    if t.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
        return f"Task `{task_id}` is already {t.status.value}, cannot cancel."

    if t.asyncio_task and not t.asyncio_task.done():
        t.asyncio_task.cancel()

    t.status = TaskStatus.CANCELLED
    t.finished_at = time.time()
    return f"Task `{task_id}` cancelled."


@mcp.tool()
async def start_parallel_tasks(tasks: str) -> str:
    """Start multiple background tasks in parallel. Returns immediately with all task IDs.

    Like parallel_agents but non-blocking. Each task runs in the background.

    Args:
        tasks: JSON string — list of task objects:
            [{"task": "...", "role": "analyst"}, {"task": "...", "role": "writer"}]
    """
    try:
        task_list = json.loads(tasks)
    except json.JSONDecodeError:
        return "Error: 'tasks' must be a valid JSON array"

    if not isinstance(task_list, list) or not task_list:
        return "Error: 'tasks' must be a non-empty JSON array"

    task_ids = []
    for item in task_list:
        role = item.get("role", "researcher")
        sys_prompt = item.get("system_prompt", "")
        model = item.get("model", DEFAULT_MODEL)
        task_text = item.get("task", "")

        if not task_text:
            continue

        # Resolve role
        if sys_prompt:
            prompt = sys_prompt
            role_name = "Custom Agent"
        elif role in ROLES:
            prompt = ROLES[role]["system"]
            role_name = ROLES[role]["name"]
        else:
            prompt = "You are a helpful assistant."
            role_name = role or "General"

        use_model = model
        task_id = _short_id()

        bg_task = BackgroundTask(
            id=task_id,
            task=task_text[:200],
            role=role,
            role_name=role_name,
            status=TaskStatus.PENDING,
            model=use_model,
            created_at=time.time(),
        )
        _tasks[task_id] = bg_task

        async def _run(bt=bg_task, p=prompt, tt=task_text, m=use_model):
            bt.status = TaskStatus.RUNNING
            bt.started_at = time.time()
            try:
                result = await _call_gemini(p, tt, model=m)
                bt.result = result
                bt.status = TaskStatus.DONE
            except asyncio.CancelledError:
                bt.status = TaskStatus.CANCELLED
            except Exception as e:
                bt.error = str(e)
                bt.status = TaskStatus.FAILED
            finally:
                bt.finished_at = time.time()

        bg_task.asyncio_task = asyncio.create_task(_run())
        task_ids.append((task_id, role_name, task_text[:60]))

    lines = [f"# Started {len(task_ids)} background tasks\n"]
    for tid, rname, ttask in task_ids:
        lines.append(f"- `{tid}` — {rname}: {ttask}...")

    lines.append(f"\nUse `list_tasks()` to monitor all, or `check_task(\"<id>\")` for individual status.")
    return "\n".join(lines)


@mcp.tool()
async def smart_task(task: str, role: str = "researcher", system_prompt: str = "", model: str = "", wait_seconds: int = 0) -> str:
    """Start a background task with smart polling. If wait_seconds > 0, waits that long then returns status + result if done.

    This is the recommended way to run background tasks. The response tells you:
    - If wait_seconds=0: returns task ID + estimated completion time so you can decide when to check
    - If wait_seconds>0: waits up to that many seconds, then returns result if done or status if still running

    Use this instead of start_task + check_task for a smoother workflow.

    Args:
        task: The task for the sub-agent
        role: Predefined role: analyst, threat, writer, coder, researcher, critic
        system_prompt: Custom system prompt (overrides role)
        model: Model to use. Empty = gemini-2.5-flash
        wait_seconds: Seconds to wait before returning. 0 = return immediately. Max 90.
    """
    # Resolve role
    if system_prompt:
        sys_prompt = system_prompt
        role_name = "Custom Agent"
    elif role in ROLES:
        sys_prompt = ROLES[role]["system"]
        role_name = ROLES[role]["name"]
    else:
        available = ", ".join(ROLES.keys())
        return f"Error: Unknown role '{role}'. Available: {available}"

    use_model = model if model else DEFAULT_MODEL
    task_id = _short_id()

    # Estimate completion time based on task length
    task_len = len(task)
    if task_len < 200:
        est_seconds = 8
    elif task_len < 500:
        est_seconds = 15
    else:
        est_seconds = 25

    bg_task = BackgroundTask(
        id=task_id,
        task=task[:200],
        role=role or "custom",
        role_name=role_name,
        status=TaskStatus.PENDING,
        model=use_model,
        created_at=time.time(),
    )
    _tasks[task_id] = bg_task

    async def _run():
        bg_task.status = TaskStatus.RUNNING
        bg_task.started_at = time.time()
        try:
            result = await _call_gemini(sys_prompt, task, model=use_model)
            bg_task.result = result
            bg_task.status = TaskStatus.DONE
        except asyncio.CancelledError:
            bg_task.status = TaskStatus.CANCELLED
            bg_task.error = "Cancelled"
        except Exception as e:
            bg_task.error = str(e)
            bg_task.status = TaskStatus.FAILED
        finally:
            bg_task.finished_at = time.time()

    bg_task.asyncio_task = asyncio.create_task(_run())

    # If wait_seconds > 0, poll until done or timeout
    wait = min(max(wait_seconds, 0), 90)
    if wait > 0:
        deadline = time.time() + wait
        while time.time() < deadline:
            if bg_task.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED):
                break
            await asyncio.sleep(1)

        if bg_task.status == TaskStatus.DONE:
            runtime = _elapsed(bg_task.started_at, bg_task.finished_at)
            return (
                f"# Task `{task_id}` — {role_name} (completed in {runtime})\n\n"
                f"{bg_task.result}"
            )
        elif bg_task.status == TaskStatus.FAILED:
            return f"Task `{task_id}` FAILED: {bg_task.error}"
        else:
            elapsed = _elapsed(bg_task.created_at)
            return (
                f"Task `{task_id}` still {bg_task.status.value} after {wait}s wait ({elapsed} total).\n"
                f"Try again: `smart_task` is not reusable — use `check_task(\"{task_id}\")` or "
                f"`get_result(\"{task_id}\")` to poll.\n"
                f"Estimated remaining: ~{max(est_seconds - wait, 5)}s"
            )

    # Immediate return
    return (
        f"Background task `{task_id}` started.\n"
        f"Role: {role_name} ({use_model})\n"
        f"Estimated completion: ~{est_seconds}s\n\n"
        f"**Recommended**: Wait ~{est_seconds}s then call `check_task(\"{task_id}\")`. "
        f"If done, call `get_result(\"{task_id}\")` to retrieve output.\n"
        f"Or call `smart_task` next time with `wait_seconds={est_seconds}` to auto-wait."
    )


@mcp.tool()
async def wait_for_tasks(task_ids: str, timeout: int = 60) -> str:
    """Wait for one or more background tasks to complete, then return all results.

    Blocks until all tasks finish or timeout is reached. Use after start_parallel_tasks.

    Args:
        task_ids: Comma-separated task IDs (e.g. "a1b2c3d4,e5f6g7h8") or JSON array
        timeout: Max seconds to wait (default 60, max 120)
    """
    # Parse task IDs
    if task_ids.startswith("["):
        try:
            ids = json.loads(task_ids)
        except json.JSONDecodeError:
            return "Error: Invalid JSON array for task_ids"
    else:
        ids = [t.strip() for t in task_ids.split(",") if t.strip()]

    if not ids:
        return "Error: No task IDs provided"

    # Validate
    for tid in ids:
        if tid not in _tasks:
            return f"Error: Unknown task ID '{tid}'"

    wait = min(max(timeout, 1), 120)
    deadline = time.time() + wait

    # Poll until all done or timeout
    while time.time() < deadline:
        all_done = all(
            _tasks[tid].status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED)
            for tid in ids
        )
        if all_done:
            break
        await asyncio.sleep(1)

    # Collect results
    output = [f"# Results for {len(ids)} tasks\n"]
    for tid in ids:
        t = _tasks[tid]
        if t.status == TaskStatus.DONE:
            runtime = _elapsed(t.started_at, t.finished_at) if t.started_at and t.finished_at else "?"
            output.append(f"## `{tid}` — {t.role_name} (done in {runtime})\n\n{t.result}\n")
        elif t.status == TaskStatus.FAILED:
            output.append(f"## `{tid}` — {t.role_name} FAILED\n{t.error}\n")
        elif t.status == TaskStatus.CANCELLED:
            output.append(f"## `{tid}` — {t.role_name} CANCELLED\n")
        else:
            elapsed = _elapsed(t.created_at)
            output.append(f"## `{tid}` — {t.role_name} still {t.status.value} ({elapsed})\n")

    return "\n---\n".join(output)


# ── Generic Background Tool ───────────────────────────────────────────────────

BACKGROUND_SYSTEM_PROMPT = """\
You are a background worker agent. You have access to all Veilguard tools via the main system.
Complete the task described below thoroughly and return the full result.
Do NOT ask clarifying questions — use your best judgment.
Do NOT explain what you're going to do — just do it and return the result.
Be concise but complete in your output.\
"""


@mcp.tool()
async def background(task: str, wait_seconds: int = 15, model: str = "") -> str:
    """Run ANY task in the background using a Gemini sub-agent.

    This is the universal background tool. The sub-agent can do anything:
    create documents, analyze data, write code, research topics, generate reports.
    It delegates to Gemini which reasons about the task and produces the result.

    Examples:
        background("Create a PDF report summarizing Q1 sales data at /workspace/q1.pdf")
        background("Analyze the CSV at /workspace/data.csv and create a bar chart")
        background("Write a Python script that scrapes news headlines and save to /workspace/news.py")
        background("Research the top 5 ransomware groups active in 2025")

    Args:
        task: Natural language description of what to do. Be specific.
        wait_seconds: Auto-wait this many seconds. 0 = return immediately with task ID. Default 15. Max 90.
        model: Model override. Empty = gemini-2.5-flash.
    """
    use_model = model if model else DEFAULT_MODEL
    task_id = _short_id()

    # Estimate based on task complexity
    task_len = len(task)
    if task_len < 100:
        est = 8
    elif task_len < 300:
        est = 15
    else:
        est = 25

    bg_task = BackgroundTask(
        id=task_id,
        task=task[:200],
        role="background",
        role_name="Background Worker",
        status=TaskStatus.PENDING,
        model=use_model,
        created_at=time.time(),
    )
    _tasks[task_id] = bg_task

    async def _run():
        bg_task.status = TaskStatus.RUNNING
        bg_task.started_at = time.time()
        try:
            result = await _call_gemini(BACKGROUND_SYSTEM_PROMPT, task, model=use_model)
            bg_task.result = result
            bg_task.status = TaskStatus.DONE
        except asyncio.CancelledError:
            bg_task.status = TaskStatus.CANCELLED
            bg_task.error = "Cancelled"
        except Exception as e:
            bg_task.error = str(e)
            bg_task.status = TaskStatus.FAILED
        finally:
            bg_task.finished_at = time.time()

    bg_task.asyncio_task = asyncio.create_task(_run())

    # Auto-wait
    wait = min(max(wait_seconds, 0), 90)
    if wait > 0:
        deadline = time.time() + wait
        while time.time() < deadline:
            if bg_task.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED):
                break
            await asyncio.sleep(1)

        if bg_task.status == TaskStatus.DONE:
            runtime = _elapsed(bg_task.started_at, bg_task.finished_at)
            return f"# Background task `{task_id}` (done in {runtime})\n\n{bg_task.result}"
        elif bg_task.status == TaskStatus.FAILED:
            return f"Task `{task_id}` FAILED: {bg_task.error}"
        else:
            return (
                f"Task `{task_id}` still running after {wait}s.\n"
                f"Estimated: ~{est}s total. Use `check_task(\"{task_id}\")` to poll, "
                f"`get_result(\"{task_id}\")` when done."
            )

    return (
        f"Background task `{task_id}` started.\n"
        f"Estimated: ~{est}s. Use `check_task(\"{task_id}\")` after ~{est}s, "
        f"then `get_result(\"{task_id}\")` when done."
    )


if __name__ == "__main__":
    if "--sse" in sys.argv:
        port = 8809
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])

        import uvicorn
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route

        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as (read, write):
                await mcp._mcp_server.run(
                    read, write, mcp._mcp_server.create_initialization_options()
                )

        app = Starlette(
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        print(f"Starting sub-agents MCP server on http://0.0.0.0:{port}/sse")
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        mcp.run(transport="stdio")
