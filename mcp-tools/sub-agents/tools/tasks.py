"""Background task tools — start_task, check_task, get_result, smart_task, etc.

The three spawn-style tools (``start_task`` / ``smart_task`` /
``start_parallel_tasks``) default to **tool-enabled** background
workers — they run ``agentic_loop`` under the hood so each worker
has access to ``web_search``, ``web_fetch``, ``tcmm_recall``,
``write_scratchpad``, etc. This was previously ``call_llm`` which
is single-shot with no tool access — the 4/4 workers on the
Scattered Spider run all returned "I don't have web search
capability" because of that. Pass ``tools=False`` (or
``max_turns=1``) to fall back to the old single-call behaviour
for tasks that don't need tools (simple text generation,
reformatting, etc.).
"""

import asyncio
import json
import logging
import time

from config import ROLES, DEFAULT_MODEL, DEFAULT_BACKEND
from core import state
from core.state import BackgroundTask, TaskStatus
from core.llm import call_llm, resolve_backend
from core.agentic import agentic_loop, AGENT_TOOLS, EFFORT_LEVELS
from core.request_ctx import spawn_scope as _spawn_scope, capture_context as _capture_ctx
from utils.helpers import short_id, elapsed

logger = logging.getLogger("veilguard.tasks")


async def _run_background_worker(
    sys_prompt: str,
    task: str,
    *,
    model: str,
    role: str,
    backend: str = "",
    tools: bool = True,
    max_turns: int = 7,
) -> str:
    """Unified background-worker body.

    When ``tools=True`` (default) runs an agentic loop with
    AGENT_TOOLS (web_search, web_fetch, tcmm_recall, scratchpad,
    todo_write). When False, falls back to single-call ``call_llm``
    for lightweight text tasks. ``max_turns`` caps the agentic
    loop — 7 is a reasonable default, raise for deep-research
    workloads, drop to 1 to mimic the old behaviour.
    """
    if tools and max_turns > 1:
        return await agentic_loop(
            sys_prompt, task,
            max_turns=max_turns,
            backend=backend,
            model=model,
            tools=AGENT_TOOLS,
        )
    return await call_llm(sys_prompt, task, model=model, backend=backend, role=role)


def register(mcp):
    @mcp.tool()
    async def start_task(task: str, role: str = "researcher", system_prompt: str = "", model: str = "",
                          tools: bool = True, max_turns: int = 7) -> str:
        """Start a background task that runs asynchronously. Returns immediately with a task ID.

        Defaults to tool-enabled agentic execution: the worker runs
        a multi-turn loop with web_search/web_fetch/tcmm_recall/
        scratchpad access. Set ``tools=False`` (or ``max_turns=1``)
        for a single-shot text call.
        """
        if system_prompt:
            sys_prompt, role_name = system_prompt, "Custom Agent"
        elif role in ROLES:
            sys_prompt, role_name = ROLES[role]["system"], ROLES[role]["name"]
        else:
            return f"Error: Unknown role '{role}'"
        use_model = model if model else DEFAULT_MODEL
        task_id = short_id()
        bg = BackgroundTask(id=task_id, task=task[:200], role=role or "custom", role_name=role_name,
                            status=TaskStatus.PENDING, model=use_model, created_at=time.time())
        state.tasks[task_id] = bg
        # Capture parent identity NOW, synchronously in the handler,
        # before asyncio.create_task detaches the worker. See
        # capture_context docstring — we can't rely on contextvar
        # propagation through task boundaries in this codebase.
        captured_ctx = _capture_ctx()

        async def _run():
            async with _spawn_scope(f"bgtask-{task_id}", captured=captured_ctx):
                bg.status = TaskStatus.RUNNING
                bg.started_at = time.time()
                try:
                    bg.result = await _run_background_worker(
                        sys_prompt, task,
                        model=use_model, role=role,
                        tools=tools, max_turns=max_turns,
                    )
                    bg.status = TaskStatus.DONE
                except asyncio.CancelledError:
                    bg.status = TaskStatus.CANCELLED
                except Exception as e:
                    bg.error = str(e)
                    bg.status = TaskStatus.FAILED
                finally:
                    bg.finished_at = time.time()

        bg.asyncio_task = asyncio.create_task(_run())
        hint = ""
        if tools and max_turns > 1:
            hint = f" Agentic worker — expect ~{max_turns}min; use `wait_for_tasks('[\"{task_id}\"]', timeout=600)` to block until done."
        return f"Background task `{task_id}` started (tools={tools}, max_turns={max_turns}).{hint}"

    @mcp.tool()
    async def check_task(task_id: str) -> str:
        """Check the status of a background task."""
        if task_id not in state.tasks: return f"Error: Unknown task ID '{task_id}'"
        t = state.tasks[task_id]
        lines = [f"Task `{task_id}`: {t.status.value}", f"Role: {t.role_name} ({t.model})", f"Elapsed: {elapsed(t.created_at)}"]
        if t.status == TaskStatus.DONE: lines.append(f"\nResult ready — use `get_result(\"{task_id}\")`.")
        elif t.status == TaskStatus.FAILED: lines.append(f"\nError: {t.error}")
        elif t.status == TaskStatus.RUNNING:
            lines.append(f"\nStill running. Agentic workers typically take 5-10 min. "
                         f"Use `wait_for_tasks('[\"{task_id}\"]', timeout=600)` to block.")
        return "\n".join(lines)

    @mcp.tool()
    async def get_result(task_id: str) -> str:
        """Get the full result of a completed background task."""
        if task_id not in state.tasks: return f"Error: Unknown task ID '{task_id}'"
        t = state.tasks[task_id]
        if t.status in (TaskStatus.RUNNING, TaskStatus.PENDING):
            return (
                f"Task `{task_id}` still {t.status.value} "
                f"(elapsed {elapsed(t.created_at)}). Agentic workers take 5-10 min — "
                f"use `wait_for_tasks('[\"{task_id}\"]', timeout=600)` to block until it's done."
            )
        if t.status == TaskStatus.FAILED: return f"Task `{task_id}` FAILED: {t.error}"
        if t.status == TaskStatus.CANCELLED: return f"Task `{task_id}` was cancelled."
        runtime = elapsed(t.started_at, t.finished_at) if t.started_at and t.finished_at else "?"
        return f"# Task `{task_id}` — {t.role_name} (done in {runtime})\n\n{t.result}"

    @mcp.tool()
    async def list_tasks() -> str:
        """List all background tasks and their current status."""
        if not state.tasks: return "No background tasks."
        lines = ["# Background Tasks\n"]
        for t in sorted(state.tasks.values(), key=lambda x: x.created_at, reverse=True):
            e = elapsed(t.created_at, t.finished_at if t.finished_at else None)
            lines.append(f"- `{t.id}` {t.status.value} | {t.role_name} | {e} | {t.task[:40]}")
        return "\n".join(lines)

    @mcp.tool()
    async def cancel_task(task_id: str) -> str:
        """Cancel a running background task."""
        if task_id not in state.tasks: return f"Error: Unknown task ID '{task_id}'"
        t = state.tasks[task_id]
        if t.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
            return f"Task already {t.status.value}."
        if t.asyncio_task and not t.asyncio_task.done(): t.asyncio_task.cancel()
        t.status = TaskStatus.CANCELLED
        t.finished_at = time.time()
        return f"Task `{task_id}` cancelled."

    @mcp.tool()
    async def start_parallel_tasks(tasks: str, tools: bool = True, max_turns: int = 7) -> str:
        """Start multiple background tasks in parallel. Returns task IDs immediately.

        Defaults to tool-enabled agentic workers (web_search/
        web_fetch/tcmm_recall/scratchpad). Each task may override
        ``tools`` / ``max_turns`` per-item in the JSON array for
        fine-grained control. Without tools the previous behaviour
        on this tool was that every fan-out worker confabulated
        because it had no way to look things up — see the Scattered
        Spider post-mortem.
        """
        try:
            task_list = json.loads(tasks)
        except json.JSONDecodeError:
            return "Error: 'tasks' must be a valid JSON array"
        # Capture parent identity synchronously before fanning out.
        captured_ctx = _capture_ctx()
        task_ids = []
        for item in task_list:
            r = item.get("role", "researcher")
            sp = item.get("system_prompt", "")
            m = item.get("model", DEFAULT_MODEL)
            tt = item.get("task", "")
            if not tt: continue
            prompt = sp or ROLES.get(r, {}).get("system", "You are a helpful assistant.")
            rn = ROLES.get(r, {}).get("name", r or "General")
            tid = short_id()
            # Per-task overrides win; otherwise inherit the call-level defaults.
            task_tools = bool(item.get("tools", tools))
            task_max_turns = int(item.get("max_turns", max_turns))
            bg = BackgroundTask(id=tid, task=tt[:200], role=r, role_name=rn,
                                status=TaskStatus.PENDING, model=m, created_at=time.time())
            state.tasks[tid] = bg

            async def _run(bt=bg, p=prompt, t=tt, mo=m, ro=r, local_tid=tid,
                           use_tools=task_tools, mt=task_max_turns,
                           snap=captured_ctx):
                # Each parallel worker gets its OWN child cid so
                # their archive blocks land in distinct namespaces.
                async with _spawn_scope(f"pbg-{local_tid}", captured=snap):
                    bt.status = TaskStatus.RUNNING; bt.started_at = time.time()
                    try:
                        bt.result = await _run_background_worker(
                            p, t,
                            model=mo, role=ro,
                            tools=use_tools, max_turns=mt,
                        )
                        bt.status = TaskStatus.DONE
                    except Exception as e:
                        bt.error = str(e); bt.status = TaskStatus.FAILED
                    finally:
                        bt.finished_at = time.time()

            bg.asyncio_task = asyncio.create_task(_run())
            task_ids.append((tid, rn, tt[:60]))
        lines = [f"# Started {len(task_ids)} background tasks (tools={tools}, max_turns={max_turns})\n"]
        for tid, rn, tt in task_ids:
            lines.append(f"- `{tid}` — {rn}: {tt}...")
        return "\n".join(lines)

    @mcp.tool()
    async def smart_task(task: str, role: str = "researcher", system_prompt: str = "", model: str = "",
                         wait_seconds: int = 0, tools: bool = True, max_turns: int = 7) -> str:
        """Start a background task with smart polling. Recommended for async work.

        Defaults to tool-enabled agentic execution — same rationale
        as ``start_task``. Set ``tools=False`` / ``max_turns=1`` for
        the old single-shot behaviour when the task doesn't need
        web or memory lookups.
        """
        if system_prompt:
            sys_prompt, role_name = system_prompt, "Custom Agent"
        elif role in ROLES:
            sys_prompt, role_name = ROLES[role]["system"], ROLES[role]["name"]
        else:
            return f"Error: Unknown role '{role}'"
        use_model = model if model else DEFAULT_MODEL
        task_id = short_id()
        est = 8 if len(task) < 200 else (15 if len(task) < 500 else 25)
        bg = BackgroundTask(id=task_id, task=task[:200], role=role or "custom", role_name=role_name,
                            status=TaskStatus.PENDING, model=use_model, created_at=time.time())
        state.tasks[task_id] = bg
        captured_ctx = _capture_ctx()

        async def _run():
            async with _spawn_scope(f"smart-{task_id}", captured=captured_ctx):
                bg.status = TaskStatus.RUNNING; bg.started_at = time.time()
                try:
                    bg.result = await _run_background_worker(
                        sys_prompt, task,
                        model=use_model, role=role,
                        tools=tools, max_turns=max_turns,
                    )
                    bg.status = TaskStatus.DONE
                except Exception as e:
                    bg.error = str(e); bg.status = TaskStatus.FAILED
                finally:
                    bg.finished_at = time.time()

        bg.asyncio_task = asyncio.create_task(_run())
        # Agentic workers make ~20 tool calls over 7 turns and take
        # 5-10 min to complete (vs ~15s for the old single-shot
        # ``call_llm`` path). Raised the cap from 90s → 900s so
        # ``wait_seconds`` can actually wait for an agentic run.
        # Callers that don't want to block still pass 0 (default).
        wait = min(max(wait_seconds, 0), 900)
        if wait > 0:
            deadline = time.time() + wait
            while time.time() < deadline:
                if bg.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED): break
                await asyncio.sleep(1)
            if bg.status == TaskStatus.DONE:
                runtime = elapsed(bg.started_at, bg.finished_at)
                return f"# Task `{task_id}` — {role_name} (done in {runtime})\n\n{bg.result}"
            elif bg.status == TaskStatus.FAILED:
                return f"Task `{task_id}` FAILED: {bg.error}"
            return f"Task `{task_id}` still running. Use `check_task(\"{task_id}\")` to poll."
        return f"Background task `{task_id}` started. Est ~{est}s."

    @mcp.tool()
    async def wait_for_tasks(task_ids: str, timeout: int = 600) -> str:
        """Wait for multiple tasks to complete, then return all results.

        Default timeout raised to 600s (10 min) with a cap of 1200s
        (20 min) to match agentic workers — each one makes ~20 tool
        calls over up to 7 turns and typically takes 5-10 min. The
        old cap of 120s timed out mid-loop, forcing the planner to
        fall back to doing its own searches in parallel while the
        workers silently ran to completion. Callers that want a
        fast probe can still pass a short timeout (e.g. ``timeout=5``)
        and poll ``get_result`` themselves.
        """
        if task_ids.startswith("["):
            ids = json.loads(task_ids)
        else:
            ids = [t.strip() for t in task_ids.split(",") if t.strip()]
        for tid in ids:
            if tid not in state.tasks: return f"Error: Unknown task ID '{tid}'"
        deadline = time.time() + min(max(timeout, 1), 1200)
        while time.time() < deadline:
            if all(state.tasks[tid].status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED) for tid in ids):
                break
            await asyncio.sleep(1)
        output = [f"# Results for {len(ids)} tasks\n"]
        for tid in ids:
            t = state.tasks[tid]
            if t.status == TaskStatus.DONE:
                output.append(f"## `{tid}` — {t.role_name}\n\n{t.result}\n")
            elif t.status == TaskStatus.FAILED:
                output.append(f"## `{tid}` — FAILED\n{t.error}\n")
            else:
                output.append(f"## `{tid}` — {t.status.value}\n")
        return "\n---\n".join(output)
