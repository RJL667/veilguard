"""Agent team tools — collaborative multi-agent work."""

import asyncio
import json
import time
import uuid

from config import ROLES, DEFAULT_BACKEND, DEFAULT_MODEL
from core import state
from core.state import AgentTeam, ManagedTask
from core.llm import call_llm, resolve_backend
from tools.managed_tasks import check_task_dependencies
from storage.session import save_session


def register(mcp):
    @mcp.tool()
    async def team_create(name: str, teammates: str) -> str:
        """Create an agent team for collaborative multi-agent work."""
        try:
            specs = json.loads(teammates)
        except json.JSONDecodeError:
            return "Error: teammates must be valid JSON array"
        team_id = name.lower().replace(" ", "-")
        mates = {}
        for spec in specs:
            mn = spec.get("name", "")
            if mn: mates[mn] = {"role": spec.get("role", "researcher"), "status": "idle", "task_id": None}
        state.teams[team_id] = AgentTeam(name=team_id, teammates=mates)
        mate_list = ", ".join(f"{n} ({m['role']})" for n, m in mates.items())
        return f"Team `{team_id}` created: {mate_list}"

    @mcp.tool()
    async def team_assign(team: str, task: str, to: str, depends_on: str = "", backend: str = "", model: str = "") -> str:
        """Assign a task to a team member and start execution."""
        team_id = team.lower().replace(" ", "-")
        if team_id not in state.teams: return f"Error: Team '{team_id}' not found"
        t = state.teams[team_id]
        if to not in t.teammates: return f"Error: Teammate '{to}' not found"
        mate = t.teammates[to]
        role = mate["role"]
        tid = f"task-{uuid.uuid4().hex[:6]}"
        deps = [d.strip() for d in depends_on.split(",") if d.strip()] if depends_on else []
        status = "blocked" if deps else "pending"
        state.managed_tasks[tid] = ManagedTask(id=tid, title=f"[{team_id}/{to}] {task[:80]}",
                                                description=task, status=status, depends_on=deps, assigned_to=to)
        t.task_ids.append(tid)
        mate["task_id"] = tid

        if status == "pending":
            mate["status"] = "working"
            state.managed_tasks[tid].status = "in_progress"
            cfg = resolve_backend(backend)
            use_model = model if model else cfg["default_model"]
            backend_label = (backend or DEFAULT_BACKEND).lower()

            async def _run():
                sp = ROLES.get(role, {}).get("system", "You are helpful.")
                result = await call_llm(sp, task, model=use_model, backend=backend_label, role=role)
                state.managed_tasks[tid].status = "completed"
                state.managed_tasks[tid].result = result
                state.managed_tasks[tid].completed_at = time.time()
                mate["status"] = "idle"
                mate["task_id"] = None
                check_task_dependencies()
                save_session()

            asyncio.create_task(_run())
        return f"Task `{tid}` assigned to {to} ({role})"

    @mcp.tool()
    async def team_status(team: str) -> str:
        """Show team status — all members, tasks, and progress."""
        team_id = team.lower().replace(" ", "-")
        if team_id not in state.teams: return f"Error: Team '{team_id}' not found"
        t = state.teams[team_id]
        lines = [f"# Team `{team_id}`\n## Members"]
        for name, mate in t.teammates.items():
            lines.append(f"- **{name}** ({mate['role']}) [{mate['status']}]")
        if t.task_ids:
            lines.append("\n## Tasks")
            for tid in t.task_ids:
                mt = state.managed_tasks.get(tid)
                if mt: lines.append(f"- `{tid}` [{mt.status}]: {mt.title}")
        return "\n".join(lines)

    @mcp.tool()
    async def team_delete(team: str) -> str:
        """Disband an agent team and clean up its tasks.

        Args:
            team: Team name
        """
        team_id = team.lower().replace(" ", "-")
        if team_id not in state.teams:
            return f"Error: Team '{team_id}' not found"
        t = state.teams[team_id]
        # Cancel any running teammate tasks
        for mate in t.teammates.values():
            if mate.get("task_id") and mate["task_id"] in state.managed_tasks:
                state.managed_tasks[mate["task_id"]].status = "cancelled"
        del state.teams[team_id]
        return f"Team `{team_id}` disbanded."
