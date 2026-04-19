"""Veilguard Sub-Agents: Collection cleanup to prevent unbounded growth."""

import logging

from config import MAX_TASKS, MAX_NOTIFICATIONS, MAX_MAILBOX_PER_AGENT, MAX_MANAGED_TASKS
from core import state
from core.state import TaskStatus

logger = logging.getLogger("veilguard.storage")


def cleanup_collections():
    """Prune old entries from all growing collections."""
    # Prune completed background tasks
    if len(state.tasks) > MAX_TASKS:
        done = sorted(
            [(tid, t) for tid, t in state.tasks.items()
             if t.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED)],
            key=lambda x: x[1].created_at,
        )
        for tid, _ in done[:len(state.tasks) - MAX_TASKS]:
            del state.tasks[tid]

    # Prune notifications
    if len(state.notifications) > MAX_NOTIFICATIONS:
        state.notifications[:] = state.notifications[-MAX_NOTIFICATIONS:]

    # Prune mailboxes
    for agent_id in list(state.agent_mailbox.keys()):
        msgs = state.agent_mailbox[agent_id]
        if len(msgs) > MAX_MAILBOX_PER_AGENT:
            state.agent_mailbox[agent_id] = msgs[-MAX_MAILBOX_PER_AGENT:]

    # Prune managed tasks
    if len(state.managed_tasks) > MAX_MANAGED_TASKS:
        completed = sorted(
            [(tid, t) for tid, t in state.managed_tasks.items() if t.status in ("completed", "failed")],
            key=lambda x: x[1].created_at,
        )
        for tid, _ in completed[:len(state.managed_tasks) - MAX_MANAGED_TASKS]:
            del state.managed_tasks[tid]
