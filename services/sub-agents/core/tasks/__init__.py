"""Unified task graph — single Task model + pluggable executors.

Replaces the five disconnected task representations that lived around
the codebase before 2026-05-25:

  - daemon.TaskRegistry (RAM, daemon-local TaskState)
  - state.ManagedTask  (RAM, sub-agents, dependency graph)
  - state.BackgroundTask (RAM, spawned sub-agents)
  - state.Daemon       (RAM, scheduled workers)
  - agent_tasks Lance row (durable, agent-runtime ledger)

Single canonical type lives in ``model.py``.  Lance store in
``store.py`` is the persistence layer.  Dispatcher in ``dispatcher.py``
owns the lifecycle.  Executors (``executors/*``) implement the
per-kind start / cancel / status semantics.

The five legacy representations are NOT removed — they get adapted to
write shadow rows into this store so the unified queries see them too.
Future phases (B-2 through B-4) replace each one with native dispatcher
usage.
"""

from .model import Task, TaskKind, TaskStatus
from .store import TaskStore
from .dispatcher import TaskDispatcher

__all__ = [
    "Task", "TaskKind", "TaskStatus",
    "TaskStore", "TaskDispatcher",
]
