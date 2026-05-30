"""SubAgentExecutor — runs a spawned sub-agent via the existing
`_run_background_worker` mechanism.

This executor exists for TWO usage patterns:

  1. Native dispatch (forward-looking):
     A caller submits a Task(kind=SUB_AGENT, executor='sub_agent',
     payload={system_prompt, task, model, role, backend, tools,
     max_turns}).  The dispatcher calls start(); we delegate to
     ``_run_background_worker`` and return the result.

  2. Shadow dispatch (current path, B-2):
     Legacy ``start_task`` MCP tool keeps writing into
     ``state.tasks`` AND mirrors a Task row via
     ``dispatcher.record_external(...)``.  No executor.start() runs
     for that — the existing asyncio.create_task() drives the work.
     This executor's start() is the canonical implementation legacy
     code will migrate INTO in B-2.5.

Cancel:
  asyncio.Task.cancel() on the in-flight worker.  The dispatcher
  tracks the running coroutine in its _inflight dict and races us
  via that path when cancel cascade fires.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .base import TaskExecutor, ExecutorResult
from ..model import Task, TaskStatus

logger = logging.getLogger("veilguard.tasks.executors.sub_agent")


class SubAgentExecutor(TaskExecutor):
    """Spawns one sub-agent run (single LLM call OR agentic loop)."""

    name = "sub_agent"

    async def start(self, task: Task) -> ExecutorResult:
        # Lazy imports so circular dep risk stays low and the
        # executor module is cheap to import on cold start.
        try:
            from tools.tasks import _run_background_worker
            from config import ROLES, DEFAULT_MODEL
        except Exception as e:
            return ExecutorResult(
                status=TaskStatus.FAILED,
                error=f"SubAgentExecutor: import failed: {e}",
            )

        p = task.payload or {}
        task_text = (p.get("task") or task.title or "").strip()
        if not task_text:
            return ExecutorResult(
                status=TaskStatus.FAILED,
                error="SubAgentExecutor: payload['task'] missing",
            )

        # Resolve persona / system prompt the same way start_task does.
        # System prompt explicitly passed wins; otherwise role lookup.
        sys_prompt = p.get("system_prompt", "") or ""
        role = (p.get("role") or "").strip()
        if not sys_prompt:
            if role and role in ROLES:
                sys_prompt = ROLES[role].get("system", "")
            elif role:
                # Unknown role — caller's bug, surface clearly.
                return ExecutorResult(
                    status=TaskStatus.FAILED,
                    error=f"SubAgentExecutor: unknown role {role!r}",
                )

        model = p.get("model") or DEFAULT_MODEL
        backend = p.get("backend") or ""
        tools = bool(p.get("tools", True))
        max_turns = int(p.get("max_turns") or 7)

        logger.info(
            f"[sub_agent] task={task.task_id} role={role!r} model={model} "
            f"tools={tools} max_turns={max_turns}"
        )
        try:
            text = await _run_background_worker(
                sys_prompt, task_text,
                model=model, role=role, backend=backend,
                tools=tools, max_turns=max_turns,
            )
        except asyncio.CancelledError:
            # Dispatcher's cancel cascade — re-raise so it's surfaced.
            raise
        except Exception as e:
            logger.exception(
                f"[sub_agent] task={task.task_id} worker raised: {e}"
            )
            return ExecutorResult(
                status=TaskStatus.FAILED, error=str(e)[:8000],
            )
        return ExecutorResult(
            status=TaskStatus.COMPLETED,
            result=(text or "")[:8000],
        )

    async def cancel(self, task: Task) -> bool:
        # The dispatcher cancels the in-flight asyncio.Task directly
        # (see TaskDispatcher._cancel_one).  Nothing extra needed here
        # — the worker code already handles CancelledError cleanly
        # (state.tasks status becomes CANCELLED, see tools/tasks.py).
        return True


__all__ = ["SubAgentExecutor"]
