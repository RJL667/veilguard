"""Veilguard Sub-Agents: Shared mutable state.

All global state lives here. Every module imports from this file
instead of defining its own globals. This makes the dependency graph explicit.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ── Agent Statistics ─────────────────────────────────────────────────────────

agent_stats = {
    "total_calls": 0,
    "total_errors": 0,
    "by_backend": {
        "gemini": {"calls": 0, "errors": 0, "total_ms": 0},
        "claude": {"calls": 0, "errors": 0, "total_ms": 0},
    },
    "by_role": {},
}

call_log: list[dict] = []  # Last 100 calls
estimated_cost_usd = 0.0

# ── AFK Tracking ─────────────────────────────────────────────────────────────

last_user_activity = time.time()

# ── Session Save Counter ─────────────────────────────────────────────────────

calls_since_save = 0

# ── Clipboard ────────────────────────────────────────────────────────────────

clipboard: dict[str, str] = {}

# ── Notifications ────────────────────────────────────────────────────────────

notifications: list[dict] = []

# ── Schedules ────────────────────────────────────────────────────────────────

schedules: dict[str, dict] = {}
scheduler_running = False

# ── Agent Mailbox ────────────────────────────────────────────────────────────

agent_mailbox: dict[str, list[dict]] = {}


# ── Background Tasks ─────────────────────────────────────────────────────────

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
    asyncio_task: Optional[object] = None  # asyncio.Task


tasks: dict[str, BackgroundTask] = {}


# ── Managed Tasks (with dependencies) ────────────────────────────────────────

@dataclass
class ManagedTask:
    id: str
    title: str
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed, blocked
    depends_on: list = field(default_factory=list)
    assigned_to: str = ""
    result: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


managed_tasks: dict[str, ManagedTask] = {}


# ── Daemons ──────────────────────────────────────────────────────────────────

@dataclass
class Daemon:
    name: str
    task: str
    role: str
    interval_seconds: int
    backend: str
    model: str
    enabled: bool = True
    max_actions_per_window: int = 5
    action_count: int = 0
    window_start: float = 0.0
    last_run: float = 0.0
    next_run: float = 0.0
    run_count: int = 0
    observations: list = field(default_factory=list)
    asyncio_task: Optional[object] = None  # asyncio.Task


daemons: dict[str, Daemon] = {}


# ── Agent Teams ──────────────────────────────────────────────────────────────

@dataclass
class AgentTeam:
    name: str
    teammates: dict  # name -> {"role": str, "status": str, "task_id": str|None}
    task_ids: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


teams: dict[str, AgentTeam] = {}
