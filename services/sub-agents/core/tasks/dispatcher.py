"""TaskDispatcher — single entry point for submitting / cancelling tasks.

The dispatcher owns:

  - The mapping of executor-name → TaskExecutor instance
  - The lifecycle drive (PENDING → RUNNING → terminal)
  - Persistence (every status flip goes through TaskStore)
  - Cancel cascade (kill a parent → kill all live descendants)

What it doesn't own:

  - The actual work (executor does that)
  - Authorisation (caller is expected to have already gated)
  - Audit logging (the existing pii_audit table already covers tool
    calls; we don't double-write here)

Singleton per process so a single executor registry exists across the
sub-agents server's lifetime.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Optional

from .executors.base import TaskExecutor, ExecutorResult
from .executors.daemon import DaemonExecutor, SandboxExecutor
from .executors.sub_agent import SubAgentExecutor
from .executors.scheduler import SchedulerExecutor
from .executors.approval import ApprovalExecutor
from .model import Task, TaskKind, TaskStatus
from .store import TaskStore

logger = logging.getLogger("veilguard.tasks.dispatcher")


class TaskDispatcher:
    _instance: Optional["TaskDispatcher"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._store = TaskStore.get()
        self._executors: dict[str, TaskExecutor] = {}
        # In-flight asyncio.Tasks keyed on task_id — used by cancel
        # to interrupt local coroutines.  Cleaned up in _run when the
        # task finishes.
        self._inflight: dict[str, asyncio.Task] = {}

        # Register baseline executors.  B-1: daemon + sandbox.
        # B-2: sub_agent + scheduler.  B-3: approval.  B-4: director.
        self.register(DaemonExecutor())
        self.register(SandboxExecutor())
        self.register(SubAgentExecutor())
        self.register(SchedulerExecutor())
        self.register(ApprovalExecutor())

    @classmethod
    def get(cls) -> "TaskDispatcher":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ── Registration ──────────────────────────────────────────────────

    def register(self, executor: TaskExecutor) -> None:
        if not executor.name:
            raise ValueError(f"executor {type(executor).__name__} has empty .name")
        if executor.name in self._executors:
            logger.warning(
                f"[dispatcher] re-registering executor {executor.name!r} "
                "(replacing previous instance)"
            )
        self._executors[executor.name] = executor

    def executors(self) -> dict[str, TaskExecutor]:
        """Snapshot — for diagnostics."""
        return dict(self._executors)

    # ── Submit ────────────────────────────────────────────────────────

    async def submit(self, task: Task) -> Task:
        """Persist `task` and kick off its executor.

        Returns the (possibly mutated) Task.  The caller can await this
        directly for short-lived tasks (tool calls) or fire-and-forget
        for long-running ones (sub-agent runs).  Either way the row
        is durably persisted by the time submit() returns.

        Lineage: if `parent_task_id` is set but `root_task_id` is
        empty, we look up the parent and inherit its root.  This keeps
        list_subtree() queries fast — every descendant carries the
        root_id directly, no recursive joins needed.
        """
        # Lineage: collapse root if not pre-set.
        if task.parent_task_id and not task.root_task_id:
            parent = self._store.get_task(task.parent_task_id)
            if parent is not None:
                task.root_task_id = parent.root_task_id or parent.task_id
            else:
                # Orphan parent ref — record as own root.  Better than
                # failing the submit; the LLM may rerun the parent.
                logger.warning(
                    f"[dispatcher] submit task={task.task_id} "
                    f"references unknown parent={task.parent_task_id}; "
                    "treating as new root"
                )
                task.root_task_id = task.task_id

        if not task.root_task_id:
            task.root_task_id = task.task_id

        # Persist with status=PENDING so the row exists before we
        # kick off the executor.  Crash here = orphan PENDING row
        # the next cleanup sweep can detect.
        task.status = TaskStatus.PENDING.value
        self._store.upsert(task)

        # Spawn the executor coroutine.  We DON'T await it here so
        # callers can return immediately to the LLM; the executor
        # writes its own terminal-state row when it finishes.
        coro = self._run(task)
        atask = asyncio.create_task(coro)
        self._inflight[task.task_id] = atask
        # Auto-cleanup the inflight entry when the asyncio.Task is done.
        atask.add_done_callback(
            lambda _t, tid=task.task_id: self._inflight.pop(tid, None)
        )
        return task

    # ── Lifecycle drive ───────────────────────────────────────────────

    async def _run(self, task: Task) -> None:
        executor = self._executors.get(task.executor)
        if executor is None:
            logger.error(
                f"[dispatcher] no executor named {task.executor!r} for "
                f"task={task.task_id} kind={task.kind}"
            )
            self._store.mark_status(
                task.task_id, TaskStatus.FAILED,
                error=f"no executor for {task.executor!r}",
            )
            return

        # Flip to RUNNING.
        task.status = TaskStatus.RUNNING.value
        task.started_at = time.time()
        self._store.upsert(task)

        try:
            result = await executor.start(task)
        except asyncio.CancelledError:
            # Cancellation came from cancel() — already wrote
            # CANCELLED there; just bow out.
            logger.info(f"[dispatcher] task {task.task_id} cancelled mid-flight")
            raise
        except Exception as e:
            logger.exception(
                f"[dispatcher] executor {task.executor} raised "
                f"for task={task.task_id}: {e}"
            )
            self._store.mark_status(
                task.task_id, TaskStatus.FAILED, error=str(e)[:8000],
            )
            return

        # Persist terminal state from the executor's verdict.
        self._store.mark_status(
            task.task_id, result.status,
            error=result.error, result=result.result,
        )

    # ── Cancel ────────────────────────────────────────────────────────

    async def cancel(
        self, task_id: str, *, cascade: bool = True,
    ) -> dict:
        """Cancel `task_id` and (by default) all live descendants.

        Returns a dict mapping task_id → True/False for each task we
        attempted.  Terminal tasks are skipped (no-op).
        """
        target = self._store.get_task(task_id)
        if target is None:
            return {"ok": False, "reason": f"task {task_id!r} not found"}

        descendants: list[Task] = []
        if cascade:
            # Subtree query is one indexed read — root_task_id BTREE.
            root_id = target.root_task_id or target.task_id
            subtree = self._store.list_subtree(root_id, limit=500)
            # Filter to actual descendants of `task_id` (or target itself).
            keep = {target.task_id}
            changed = True
            while changed:
                changed = False
                for t in subtree:
                    if t.parent_task_id in keep and t.task_id not in keep:
                        keep.add(t.task_id)
                        changed = True
            descendants = [t for t in subtree if t.task_id in keep
                           and t.task_id != target.task_id]

        results: dict[str, bool] = {}
        for t in [target, *descendants]:
            if t.is_terminal:
                results[t.task_id] = False
                continue
            ok = await self._cancel_one(t)
            results[t.task_id] = ok
        return {"ok": True, "results": results}

    async def _cancel_one(self, task: Task) -> bool:
        # Tell the store CANCELLING first so concurrent readers see
        # an intermediate state.
        self._store.mark_status(task.task_id, TaskStatus.CANCELLING)

        executor = self._executors.get(task.executor)
        ex_ok = True
        if executor is not None:
            try:
                ex_ok = await executor.cancel(task)
            except Exception as e:
                logger.warning(
                    f"[dispatcher] cancel raised for {task.task_id}: {e}"
                )
                ex_ok = False

        # Cancel the local asyncio.Task too — it may be waiting on a
        # WS future the executor.cancel() just released.
        local = self._inflight.pop(task.task_id, None)
        if local is not None and not local.done():
            local.cancel()

        # Final state.
        self._store.mark_status(
            task.task_id, TaskStatus.CANCELLED,
            error="cancelled by dispatcher",
        )
        return ex_ok

    # ── Query (thin proxies to TaskStore) ─────────────────────────────

    def list_user(self, user_id: str, **kw) -> list[Task]:
        return self._store.list_user(user_id, **kw)

    def list_subtree(self, root_task_id: str, **kw) -> list[Task]:
        return self._store.list_subtree(root_task_id, **kw)

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._store.get_task(task_id)

    # ── Side-channel write (for daemon shadow rows) ──────────────────

    def record_external(self, task: Task) -> None:
        """For tasks the dispatcher DIDN'T submit but wants tracked.

        Used by:
          - The daemon's TaskRegistry (when running a tool call that
            wasn't routed via dispatcher — e.g. legacy path) writes a
            shadow row here so list_my_tasks sees it.
          - The agent-runtime container (when Director executes a tool
            without going through sub-agents' dispatcher).

        We persist as-is — caller is responsible for setting status +
        timestamps correctly.  No executor lookup, no _run call.
        """
        self._store.upsert(task)


__all__ = ["TaskDispatcher"]
