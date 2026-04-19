"""REST API endpoints for the Veilguard UI panel."""

import time
from starlette.responses import JSONResponse

from config import SCRATCHPAD_DIR
from core import state
from core.state import TaskStatus
from utils.helpers import elapsed
from utils.afk import is_afk
from tools.managed_tasks import check_task_dependencies


async def api_stats(request):
    return JSONResponse({
        "total_calls": state.agent_stats["total_calls"],
        "total_errors": state.agent_stats["total_errors"],
        "estimated_cost_usd": round(state.estimated_cost_usd, 4),
        "by_backend": state.agent_stats["by_backend"],
        "by_role": state.agent_stats["by_role"],
        "active_daemons": len([d for d in state.daemons.values() if d.enabled]),
        "running_tasks": len([t for t in state.tasks.values() if t.status == TaskStatus.RUNNING]),
        "afk": is_afk(),
        "idle_seconds": int(time.time() - state.last_user_activity),
        "recent_calls": state.call_log[-10:],
    })


async def api_tasks(request):
    bg = []
    for t in sorted(state.tasks.values(), key=lambda x: x.created_at, reverse=True)[:50]:
        bg.append({
            "id": t.id, "task": t.task, "role": t.role_name,
            "status": t.status.value, "model": t.model,
            "elapsed": elapsed(t.created_at, t.finished_at if t.finished_at else None),
            "result_preview": (t.result[:200] + "...") if t.result and len(t.result) > 200 else t.result,
            "error": t.error,
        })
    managed = []
    check_task_dependencies()
    for tid, mt in sorted(state.managed_tasks.items(), key=lambda x: x[1].created_at, reverse=True):
        managed.append({
            "id": mt.id, "title": mt.title, "description": mt.description[:200],
            "status": mt.status, "depends_on": mt.depends_on, "assigned_to": mt.assigned_to,
            "result_preview": (mt.result[:200] + "...") if mt.result and len(mt.result) > 200 else mt.result,
        })
    return JSONResponse({"background_tasks": bg, "managed_tasks": managed})


async def api_scratchpad(request):
    files = []
    if SCRATCHPAD_DIR.exists():
        for f in sorted(SCRATCHPAD_DIR.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True):
            content = f.read_text(encoding="utf-8", errors="replace")
            files.append({
                "name": f.stem, "size": f.stat().st_size,
                "age_hours": round((time.time() - f.stat().st_mtime) / 3600, 1),
                "preview": content[:300], "full_content": content[:5000],
            })
    return JSONResponse({"files": files})


async def api_daemons(request):
    daemons = []
    for did, d in state.daemons.items():
        daemons.append({
            "name": did, "task": d.task[:100], "role": d.role, "enabled": d.enabled,
            "backend": d.backend, "interval_minutes": d.interval_seconds // 60,
            "run_count": d.run_count, "next_run_seconds": max(0, int(d.next_run - time.time())),
            "last_observation": d.observations[-1] if d.observations else None,
        })
    return JSONResponse({"daemons": daemons})


async def api_teams(request):
    teams = []
    for tid, t in state.teams.items():
        members = [{"name": n, "role": m["role"], "status": m["status"]} for n, m in t.teammates.items()]
        tasks = []
        for task_id in t.task_ids:
            mt = state.managed_tasks.get(task_id)
            if mt: tasks.append({"id": mt.id, "title": mt.title, "status": mt.status})
        teams.append({"name": tid, "members": members, "tasks": tasks})
    return JSONResponse({"teams": teams})
