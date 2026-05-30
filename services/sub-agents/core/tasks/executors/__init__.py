"""Pluggable executors for the unified Task graph.

Phase B-1 ships two: DaemonExecutor (routes to user's bridge) and
SandboxExecutor (routes to system:sandbox bridge).  Both implement the
same start/cancel/status protocol so the dispatcher doesn't care which
one is selected — only the executor name on the Task does.

Phases B-2 through B-4 add:
  SubAgentExecutor   — wraps spawn_agentic / call_llm sub-agent spawn
  SchedulerExecutor  — periodic, fires sub-agent on tick
  ApprovalExecutor   — wraps request_approval WS round-trip
  DirectorExecutor   — HTTP RPC to agent-runtime container
"""

from .base import TaskExecutor, ExecutorResult
from .daemon import DaemonExecutor, SandboxExecutor
from .sub_agent import SubAgentExecutor
from .scheduler import SchedulerExecutor
from .approval import ApprovalExecutor

__all__ = [
    "TaskExecutor", "ExecutorResult",
    "DaemonExecutor", "SandboxExecutor",
    "SubAgentExecutor", "SchedulerExecutor",
    "ApprovalExecutor",
]
