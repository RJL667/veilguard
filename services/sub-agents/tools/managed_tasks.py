"""Managed task tools — task_create, task_update, task_list, task_graph (with dependencies)."""

import uuid
import time

from core import state
from core.state import ManagedTask
from storage.session import save_session


def check_task_dependencies():
    """Auto-unblock tasks whose dependencies are all completed."""
    for tid, task in state.managed_tasks.items():
        if task.status == "blocked" and task.depends_on:
            all_done = all(
                state.managed_tasks.get(dep, ManagedTask(id="", title="", description="")).status == "completed"
                for dep in task.depends_on
            )
            if all_done:
                task.status = "pending"


def register(mcp):
    @mcp.tool()
    async def task_create(title: str, description: str = "", depends_on: str = "", assigned_to: str = "") -> str:
        """Create a managed task with optional dependencies."""
        tid = f"task-{uuid.uuid4().hex[:6]}"
        deps = [d.strip() for d in depends_on.split(",") if d.strip()] if depends_on else []
        for dep in deps:
            if dep not in state.managed_tasks: return f"Error: Dependency '{dep}' not found"
        status = "blocked" if deps else "pending"
        state.managed_tasks[tid] = ManagedTask(id=tid, title=title, description=description,
                                                status=status, depends_on=deps, assigned_to=assigned_to)
        save_session()
        return f"Task `{tid}` created: {title}" + (f" (blocked by: {', '.join(deps)})" if deps else "")

    @mcp.tool()
    async def task_update(task_id: str, status: str = "", result: str = "") -> str:
        """Update a managed task's status or result."""
        if task_id not in state.managed_tasks: return f"Error: Task '{task_id}' not found"
        t = state.managed_tasks[task_id]
        if status:
            t.status = status
            if status == "completed": t.completed_at = time.time()
        if result: t.result = result
        check_task_dependencies()
        save_session()
        return f"Task `{task_id}` updated: status={t.status}"

    @mcp.tool()
    async def task_list(filter: str = "all") -> str:
        """List all managed tasks."""
        check_task_dependencies()
        if not state.managed_tasks: return "No managed tasks."
        lines = ["# Managed Tasks\n"]
        for tid, t in sorted(state.managed_tasks.items(), key=lambda x: x[1].created_at):
            if filter != "all" and t.status != filter: continue
            dep_str = f" [blocked by: {', '.join(t.depends_on)}]" if t.depends_on and t.status == "blocked" else ""
            lines.append(f"- **{tid}** [{t.status.upper()}]: {t.title}{dep_str}")
        return "\n".join(lines) if len(lines) > 1 else "No tasks match filter."

    @mcp.tool()
    async def task_graph() -> str:
        """Show the task dependency graph."""
        if not state.managed_tasks: return "No managed tasks."
        lines = ["# Task Dependency Graph\n"]
        for tid, t in state.managed_tasks.items():
            deps = " → ".join(t.depends_on) if t.depends_on else "(no deps)"
            lines.append(f"  {deps} → [{t.status}] {tid}: {t.title}")
        return "\n".join(lines)
