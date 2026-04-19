"""Background task tools — start_task, check_task, get_result, smart_task, etc."""

import asyncio
import json
import time

from config import ROLES, DEFAULT_MODEL, DEFAULT_BACKEND
from core import state
from core.state import BackgroundTask, TaskStatus
from core.llm import call_llm, resolve_backend
from utils.helpers import short_id, elapsed


def register(mcp):
    @mcp.tool()
    async def start_task(task: str, role: str = "researcher", system_prompt: str = "", model: str = "") -> str:
        """Start a background task that runs asynchronously. Returns immediately with a task ID."""
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

        async def _run():
            bg.status = TaskStatus.RUNNING
            bg.started_at = time.time()
            try:
                bg.result = await call_llm(sys_prompt, task, model=use_model, role=role)
                bg.status = TaskStatus.DONE
            except asyncio.CancelledError:
                bg.status = TaskStatus.CANCELLED
            except Exception as e:
                bg.error = str(e)
                bg.status = TaskStatus.FAILED
            finally:
                bg.finished_at = time.time()

        bg.asyncio_task = asyncio.create_task(_run())
        return f"Background task `{task_id}` started. Use `check_task(\"{task_id}\")` to monitor."

    @mcp.tool()
    async def check_task(task_id: str) -> str:
        """Check the status of a background task."""
        if task_id not in state.tasks: return f"Error: Unknown task ID '{task_id}'"
        t = state.tasks[task_id]
        lines = [f"Task `{task_id}`: {t.status.value}", f"Role: {t.role_name} ({t.model})", f"Elapsed: {elapsed(t.created_at)}"]
        if t.status == TaskStatus.DONE: lines.append(f"\nResult ready — use `get_result(\"{task_id}\")`.")
        elif t.status == TaskStatus.FAILED: lines.append(f"\nError: {t.error}")
        return "\n".join(lines)

    @mcp.tool()
    async def get_result(task_id: str) -> str:
        """Get the full result of a completed background task."""
        if task_id not in state.tasks: return f"Error: Unknown task ID '{task_id}'"
        t = state.tasks[task_id]
        if t.status in (TaskStatus.RUNNING, TaskStatus.PENDING):
            return f"Task `{task_id}` still {t.status.value}. Check back later."
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
    async def start_parallel_tasks(tasks: str) -> str:
        """Start multiple background tasks in parallel. Returns task IDs immediately."""
        try:
            task_list = json.loads(tasks)
        except json.JSONDecodeError:
            return "Error: 'tasks' must be a valid JSON array"
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
            bg = BackgroundTask(id=tid, task=tt[:200], role=r, role_name=rn,
                                status=TaskStatus.PENDING, model=m, created_at=time.time())
            state.tasks[tid] = bg

            async def _run(bt=bg, p=prompt, t=tt, mo=m, ro=r):
                bt.status = TaskStatus.RUNNING; bt.started_at = time.time()
                try:
                    bt.result = await call_llm(p, t, model=mo, role=ro)
                    bt.status = TaskStatus.DONE
                except Exception as e:
                    bt.error = str(e); bt.status = TaskStatus.FAILED
                finally:
                    bt.finished_at = time.time()

            bg.asyncio_task = asyncio.create_task(_run())
            task_ids.append((tid, rn, tt[:60]))
        lines = [f"# Started {len(task_ids)} background tasks\n"]
        for tid, rn, tt in task_ids:
            lines.append(f"- `{tid}` — {rn}: {tt}...")
        return "\n".join(lines)

    @mcp.tool()
    async def smart_task(task: str, role: str = "researcher", system_prompt: str = "", model: str = "", wait_seconds: int = 0) -> str:
        """Start a background task with smart polling. Recommended for async work."""
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

        async def _run():
            bg.status = TaskStatus.RUNNING; bg.started_at = time.time()
            try:
                bg.result = await call_llm(sys_prompt, task, model=use_model, role=role)
                bg.status = TaskStatus.DONE
            except Exception as e:
                bg.error = str(e); bg.status = TaskStatus.FAILED
            finally:
                bg.finished_at = time.time()

        bg.asyncio_task = asyncio.create_task(_run())
        wait = min(max(wait_seconds, 0), 90)
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
    async def wait_for_tasks(task_ids: str, timeout: int = 60) -> str:
        """Wait for multiple tasks to complete, then return all results."""
        if task_ids.startswith("["):
            ids = json.loads(task_ids)
        else:
            ids = [t.strip() for t in task_ids.split(",") if t.strip()]
        for tid in ids:
            if tid not in state.tasks: return f"Error: Unknown task ID '{tid}'"
        deadline = time.time() + min(max(timeout, 1), 120)
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
