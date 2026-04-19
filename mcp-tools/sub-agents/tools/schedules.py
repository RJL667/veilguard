"""Scheduled task tools — recurring execution."""

import asyncio
import logging
import time

from config import DEFAULT_MODEL, ROLES
from core import state
from core.state import BackgroundTask, TaskStatus
from core.llm import call_llm
from utils.helpers import short_id

logger = logging.getLogger("veilguard.schedules")


async def _scheduler_loop():
    """Background loop that checks and runs scheduled tasks."""
    state.scheduler_running = True
    while state.scheduler_running:
        now = time.time()
        for sched_id, sched in list(state.schedules.items()):
            if not sched.get("enabled", True):
                continue
            if now >= sched.get("next_run", 0):
                interval = sched["interval_seconds"]
                sched["next_run"] = now + interval
                sched["last_run"] = now
                sched["run_count"] = sched.get("run_count", 0) + 1

                task_text = sched["task"]
                role = sched.get("role", "researcher")
                sys_prompt = ROLES.get(role, ROLES["researcher"])["system"]
                role_name = ROLES.get(role, ROLES["researcher"])["name"]
                model = sched.get("model", DEFAULT_MODEL)
                task_id = short_id()

                bg_task = BackgroundTask(
                    id=task_id, task=f"[scheduled:{sched_id}] {task_text[:150]}",
                    role=role, role_name=f"{role_name} (scheduled)",
                    status=TaskStatus.PENDING, model=model, created_at=now,
                )
                state.tasks[task_id] = bg_task

                async def _run(bt=bg_task, sp=sys_prompt, tt=task_text, m=model, r=role):
                    bt.status = TaskStatus.RUNNING
                    bt.started_at = time.time()
                    try:
                        bt.result = await call_llm(sp, tt, model=m, role=r)
                        bt.status = TaskStatus.DONE
                    except Exception as e:
                        bt.error = str(e)
                        bt.status = TaskStatus.FAILED
                    finally:
                        bt.finished_at = time.time()

                asyncio.create_task(_run())
                sched["last_task_id"] = task_id
        await asyncio.sleep(10)


def _ensure_scheduler():
    if not state.scheduler_running:
        asyncio.get_event_loop().create_task(_scheduler_loop())


def register(mcp):
    @mcp.tool()
    async def schedule_task(name: str, task: str, interval_minutes: int = 60, role: str = "researcher", model: str = "") -> str:
        """Schedule a recurring task that runs automatically on an interval."""
        _ensure_scheduler()
        interval = max(int(interval_minutes), 1) * 60
        sched_id = name.lower().replace(" ", "-")
        state.schedules[sched_id] = {
            "name": name, "task": task, "interval_seconds": interval,
            "role": role, "model": model or DEFAULT_MODEL, "enabled": True,
            "next_run": time.time() + interval, "created_at": time.time(), "run_count": 0,
        }
        return f"Scheduled task `{sched_id}` created. Runs every {interval_minutes} min."

    @mcp.tool()
    async def run_schedule(name: str) -> str:
        """Manually trigger a scheduled task to run now."""
        sched_id = name.lower().replace(" ", "-")
        if sched_id not in state.schedules:
            return f"Error: Schedule '{sched_id}' not found"
        state.schedules[sched_id]["next_run"] = 0
        return f"Schedule `{sched_id}` will run on next tick (~10s)."

    @mcp.tool()
    async def list_schedules() -> str:
        """List all scheduled tasks and their status."""
        if not state.schedules:
            return "No scheduled tasks."
        lines = ["# Scheduled Tasks\n"]
        for sid, s in state.schedules.items():
            status = "ENABLED" if s.get("enabled", True) else "PAUSED"
            lines.append(f"- **{sid}** [{status}] every {s['interval_seconds']//60}min | runs: {s.get('run_count',0)}")
        return "\n".join(lines)

    @mcp.tool()
    async def pause_schedule(name: str) -> str:
        """Pause a scheduled task."""
        sched_id = name.lower().replace(" ", "-")
        if sched_id not in state.schedules:
            return f"Error: Schedule '{sched_id}' not found"
        state.schedules[sched_id]["enabled"] = False
        return f"Schedule `{sched_id}` paused."

    @mcp.tool()
    async def resume_schedule(name: str) -> str:
        """Resume a paused scheduled task."""
        sched_id = name.lower().replace(" ", "-")
        if sched_id not in state.schedules:
            return f"Error: Schedule '{sched_id}' not found"
        state.schedules[sched_id]["enabled"] = True
        state.schedules[sched_id]["next_run"] = time.time() + state.schedules[sched_id]["interval_seconds"]
        return f"Schedule `{sched_id}` resumed."

    @mcp.tool()
    async def delete_schedule(name: str) -> str:
        """Delete a scheduled task permanently."""
        sched_id = name.lower().replace(" ", "-")
        if sched_id not in state.schedules:
            return f"Error: Schedule '{sched_id}' not found"
        del state.schedules[sched_id]
        return f"Schedule `{sched_id}` deleted."
