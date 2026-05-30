"""TaskExecutor ABC — the contract every per-kind executor satisfies.

Executors are passive: the dispatcher owns scheduling, persistence,
status flips, and result handoff.  An executor only knows how to
ACTUALLY DO the work — call a WS, spawn a thread, post HTTP, etc.

Why ABC instead of a Protocol: we want pickle-friendliness for the
inevitable "queue executors to a background pool" follow-up, and ABC
gives us isinstance() checks at registration time.

Contract:

  ``start(task)`` is awaited by the dispatcher.  Returns an
  ``ExecutorResult`` carrying the final status + result/error.  The
  executor is responsible for any IO (WS, HTTP, subprocess).  Raising
  is allowed — the dispatcher catches and converts to FAILED.

  ``cancel(task)`` is best-effort.  Returns True if the executor
  thinks it successfully aborted.  Implementations:
    - DaemonExecutor: sends cancel_task WS to the daemon
    - SubAgentExecutor: cancels the asyncio.Task
    - SchedulerExecutor: removes the interval entry

  ``progress(task)`` (optional) returns a small dict with executor-
  specific live state — pending_requests, last_ping, etc.  Used by
  the tray + LLM-side debug tools.  Default returns None.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from ..model import Task, TaskStatus

logger = logging.getLogger("veilguard.tasks.executors")


@dataclass
class ExecutorResult:
    """What an executor returns when its work finishes."""
    status: TaskStatus
    result: str = ""
    error: str = ""
    extras: dict = field(default_factory=dict)


class TaskExecutor(ABC):
    """Override start() and cancel(); progress is optional."""

    # Identifying name — dispatcher routes Tasks by ``task.executor``
    # matching one of the registered executor's .name.
    name: str = ""

    @abstractmethod
    async def start(self, task: Task) -> ExecutorResult:
        """Execute `task` synchronously (well, async-ly).  Return result."""
        ...

    @abstractmethod
    async def cancel(self, task: Task) -> bool:
        """Best-effort cancel of an in-flight task."""
        ...

    async def progress(self, task: Task) -> Optional[dict]:
        """Live state for tray + debug.  Default: no extra info."""
        return None


__all__ = ["TaskExecutor", "ExecutorResult"]
