"""SchedulerExecutor — periodic Tasks that spawn one sub-agent per tick.

A scheduled Task is a LONG-RUNNING parent that never finishes by itself
(unless explicitly cancelled).  Each interval, it spawns a CHILD task
(kind=SUB_AGENT, executor=sub_agent) that runs ONE pass of the
configured prompt, then disappears.

This matches the existing ``start_daemon`` semantics in
``tools/daemons.py`` — the daemon runs every N minutes, accumulates
observations, max_actions_per_window rate-limit.  In the unified
world: the daemon is one SCHEDULED Task, every run is a SUB_AGENT
Task with parent_task_id = the daemon's Task.

State on disk:

  Parent SCHEDULED Task:
    status:           running (forever until cancelled)
    interval_seconds: e.g. 1800 (30 min)
    next_run_at:      time.time() when the next tick should fire
    payload:          {role, model, backend, task, tools, max_turns,
                       max_actions_per_window}

  Child SUB_AGENT Tasks:
    parent_task_id:   the daemon's task_id
    root_task_id:     also the daemon's task_id (collapsed)
    status:           running → completed/failed/cancelled
    payload:          same {role, model, ..., task} copied from parent

Cancel:
  Cancelling the parent SCHEDULED Task removes it from the schedule
  ring + cancels any in-flight child (via dispatcher's cascade).

Persistence quirk:
  We don't persist the in-RAM scheduler loop — if sub-agents
  restarts, the parent SCHEDULED row sits idle until the next
  bootstrap that calls ``rehydrate()``.  That's the same behaviour
  the existing ``state.daemons`` had (with the added benefit that
  the parent row at least survives the restart in Lance).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from .base import TaskExecutor, ExecutorResult
from ..model import Task, TaskKind, TaskStatus

logger = logging.getLogger("veilguard.tasks.executors.scheduler")


class SchedulerExecutor(TaskExecutor):
    """Periodic Task driver.  Spawns child SUB_AGENT tasks on each tick."""

    name = "scheduler"

    def __init__(self):
        # task_id → asyncio.Task driving the tick loop.  Used to
        # cancel cleanly on cancel().
        self._tick_loops: dict[str, asyncio.Task] = {}

    async def start(self, task: Task) -> ExecutorResult:
        """Enter the periodic loop.  Never returns COMPLETED voluntarily —
        only when externally cancelled OR when next_run_at runs past
        an explicit ``deadline_ts`` in extras."""
        interval = max(int(task.interval_seconds or 60), 30)   # min 30s
        if not task.next_run_at:
            task.next_run_at = time.time() + interval

        # Action budget — same shape as legacy state.Daemon.
        max_actions = int((task.payload or {}).get(
            "max_actions_per_window", 5
        ))
        window_start = time.time()
        action_count = 0

        logger.info(
            f"[scheduler] task={task.task_id} interval={interval}s "
            f"max_actions={max_actions}/h"
        )
        self._tick_loops[task.task_id] = asyncio.current_task()  # type: ignore[assignment]

        try:
            while True:
                now = time.time()
                # Sleep until next_run_at
                wait = max(0.0, (task.next_run_at or now) - now)
                if wait > 0:
                    await asyncio.sleep(wait)

                # Reset the rate-limit window every hour.
                if time.time() - window_start > 3600:
                    window_start = time.time()
                    action_count = 0
                if action_count >= max_actions:
                    # Sleep out the remainder of the window.
                    sleep_for = 3600 - (time.time() - window_start)
                    if sleep_for > 0:
                        await asyncio.sleep(sleep_for)
                    continue

                # Spawn ONE child SUB_AGENT task per tick.
                await self._spawn_tick_child(task)
                action_count += 1
                task.next_run_at = time.time() + interval
        except asyncio.CancelledError:
            logger.info(
                f"[scheduler] task={task.task_id} cancelled — stopping loop"
            )
            raise
        finally:
            self._tick_loops.pop(task.task_id, None)

    async def _spawn_tick_child(self, parent: Task) -> None:
        """Create + submit a child SUB_AGENT Task carrying the same payload."""
        # Lazy import — avoid bringing dispatcher into module init.
        from ..dispatcher import TaskDispatcher

        child = Task.new(
            kind=TaskKind.SUB_AGENT,
            executor="sub_agent",
            user_id=parent.user_id,
            tenant_id=parent.tenant_id,
            conv_id=parent.conv_id,
            parent_task_id=parent.task_id,
            root_task_id=parent.root_task_id or parent.task_id,
            title=f"{parent.title or 'scheduled'} (tick @ {time.strftime('%H:%M')})",
            payload=dict(parent.payload or {}),
        )
        await TaskDispatcher.get().submit(child)
        logger.info(
            f"[scheduler] task={parent.task_id} spawned child={child.task_id}"
        )

    async def cancel(self, task: Task) -> bool:
        # Cancel our own tick loop AND let dispatcher cascade kill the
        # children (each is in _inflight tracked by its own asyncio.Task).
        loop = self._tick_loops.pop(task.task_id, None)
        if loop is not None and not loop.done():
            loop.cancel()
            return True
        return False

    async def progress(self, task: Task) -> Optional[dict]:
        loop = self._tick_loops.get(task.task_id)
        return {
            "loop_alive":  loop is not None and not loop.done(),
            "next_run_at": task.next_run_at,
            "interval_s":  task.interval_seconds,
        }


__all__ = ["SchedulerExecutor"]
