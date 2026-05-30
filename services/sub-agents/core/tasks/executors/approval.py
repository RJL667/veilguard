"""ApprovalExecutor — wraps the request_approval WS round-trip.

Used in B-3 to make approval requests show up in the unified Task
graph.  Today (B-2) the approval flow runs entirely inside
``approval.gate(...)`` and writes its own pii_audit row — that's fine
for audit, but it means the LLM's ``list_my_tasks`` doesn't see
"there's a toast on Rudolph's screen right now waiting for click."

With this executor:

  - When the approval gate fires, it submits an APPROVAL Task to the
    dispatcher BEFORE round-tripping through the bridge.
  - Task starts in AWAITING_USER status (a special transient state we
    added in B-1's TaskStatus enum exactly for this purpose).
  - On user click / timeout, the gate marks the task terminal with
    result = "approved" / "denied" / "timeout".
  - parent_task_id links the approval to the tool_call task that
    triggered it — so the tree shows: tool_call → approval (children
    of the same parent agent).

Cancel:
  Best-effort.  Sends a ``dismiss_approval`` WS message to the daemon
  (B-3 daemon-side handler).  If daemon dismisses the toast, great.
  Otherwise the toast naturally times out and reports back via the
  normal future-resolution path.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import TaskExecutor, ExecutorResult
from ..model import Task, TaskStatus

logger = logging.getLogger("veilguard.tasks.executors.approval")


class ApprovalExecutor(TaskExecutor):
    """Tracks an approval request in the unified task graph.

    Note: this executor is mostly PASSIVE.  The actual WS round-trip
    is driven by ``core.approval.gate()`` (already shipped in
    Phase A).  We exist so the approval shows up in list_my_tasks +
    can be cancelled via dispatcher.cancel().
    """

    name = "approval"

    async def start(self, task: Task) -> ExecutorResult:
        """Wait — the gate writes our terminal status externally.

        The dispatcher's _run will mark RUNNING (via mark_status) on
        entry; we then sleep here until the row's status flips to a
        terminal value.  In practice the wait time matches the toast
        timeout (default 60s).  Cap at 5min so a hung approval row
        eventually fails the dispatcher coroutine.
        """
        import asyncio
        import time
        from ..store import TaskStore

        store = TaskStore.get()
        deadline = time.time() + 300

        # The approval gate's bridge.request_approval call writes the
        # terminal status into Lance directly via mark_status (see
        # core.approval.gate _audit path which we extended in B-3).
        # We poll it back to a terminal state.
        while time.time() < deadline:
            fresh = store.get_task(task.task_id)
            if fresh is None:
                return ExecutorResult(
                    status=TaskStatus.FAILED,
                    error="ApprovalExecutor: task vanished from store",
                )
            if fresh.is_terminal:
                return ExecutorResult(
                    status=TaskStatus(fresh.status),
                    result=fresh.result, error=fresh.error,
                )
            await asyncio.sleep(0.5)

        logger.warning(
            f"[approval] task {task.task_id} stuck in non-terminal state "
            f"for 5min — declaring failure"
        )
        return ExecutorResult(
            status=TaskStatus.FAILED,
            error="approval task exceeded 5min wall-clock budget",
        )

    async def cancel(self, task: Task) -> bool:
        """Send dismiss_approval to the daemon (best-effort)."""
        try:
            from core.client_bridge import get_bridge
            bridge = get_bridge(task.user_id)
            if bridge is None or not bridge.connected:
                return False
            resp = await bridge.send_command(
                "dismiss_approval",
                {"request_id": task.task_id},
                timeout=3.0,
            )
            if isinstance(resp, dict):
                return bool(resp.get("dismissed"))
            return False
        except Exception as e:
            logger.warning(f"[approval] cancel raised: {e}")
            return False


__all__ = ["ApprovalExecutor"]
