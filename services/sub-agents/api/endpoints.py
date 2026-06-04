"""REST API endpoints for the Veilguard UI panel."""

import time
from starlette.responses import JSONResponse

from config import SCRATCHPAD_DIR
from core import state
from core.state import TaskStatus
from utils.helpers import elapsed
from utils.afk import is_afk
from tools.managed_tasks import check_task_dependencies

import os as _osenv
# Org-status reads the ledger tables (agent_tasks / task_comments). When the
# ledger is on Postgres, query it directly instead of the Lance fallback.
_PG = _osenv.environ.get(
    "LEDGER_BACKEND", _osenv.environ.get("TCMM_STORAGE", "postgres")
).lower() in ("postgres", "postgresql")
_PG_DSN = _osenv.environ.get("TCMM_DATABASE_URL", "postgresql://tcmm:tcmm@localhost:5432/tcmm")


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


async def api_agents(request):
    """Phase 1 — Sidebar Agents view (read-only).

    Per `MULTI_AGENT_PLATFORM.md` §Phase 1: "Shows registered agents,
    last-active timestamp, the org chart, recent approvals.  No Task
    UI yet."  Sources:
      - agents/*.md persona files (canonical org structure)
      - agent_tasks Lance table (last-active inferred from updated_ts)
      - task_comments Lance table (recent-approvals via review_decision)

    The endpoint is dependency-tolerant: if Lance isn't reachable we
    still return the persona list with empty timestamps, because the
    UI shouldn't blank-out on a transient DB hiccup.
    """
    import os
    import re
    import time as _time
    from pathlib import Path

    # ── 1. Discover personas ─────────────────────────────────────────
    agents_dir_candidates = [
        Path(os.environ.get("VEILGUARD_AGENTS_DIR", "")) if os.environ.get("VEILGUARD_AGENTS_DIR") else None,
        Path(__file__).resolve().parents[3] / "agents",
        Path("C:/Users/rudol/Documents/veilguard/agents"),
        Path("/home/rudol/veilguard/agents"),
    ]
    agents_dir = next((p for p in agents_dir_candidates if p and p.is_dir()), None)
    personas: list[dict] = []
    if agents_dir is not None:
        for md in sorted(agents_dir.glob("*.md")):
            stem = md.stem
            # Skip the constitution / prompts notes — they're not personas
            if stem.upper() in ("CONSTITUTION", "PROMPTS", "README"):
                continue
            try:
                txt = md.read_text(encoding="utf-8", errors="replace")
            except Exception:
                txt = ""
            # Bold-Markdown KV frontmatter per spec §0.0.2.
            def _kv(key: str, default: str = "") -> str:
                m = re.search(rf"\*\*{re.escape(key)}\*\*[:：]?\s*(.+)", txt)
                return (m.group(1).strip() if m else default).strip("`")
            personas.append({
                "agent_id":   _kv("agent_id", stem),
                "role":       _kv("role", "consultant"),
                "manager_id": _kv("manager_id", ""),
                "team_id":    _kv("team_id", ""),
                "model":      _kv("Model", _kv("model", "")),
                "tools":      _kv("Tools", _kv("tools", "")),
                "source_path": str(md),
            })

    # ── 2. Last-active + task counts from agent_tasks ───────────────
    now = _time.time()
    by_owner: dict[str, dict] = {}
    if _PG:
        try:
            import psycopg2
            _conn = psycopg2.connect(_PG_DSN)
            with _conn.cursor() as _cur:
                _cur.execute("SELECT owner_id, status, COALESCE(updated_ts,0) FROM agent_tasks LIMIT 2000")
                for _owner, _st, _ts in _cur.fetchall():
                    if not _owner:
                        continue
                    _rec = by_owner.setdefault(_owner, {
                        "tasks_total": 0, "tasks_active": 0,
                        "tasks_done": 0, "tasks_cancelled": 0, "last_active_ts": 0.0})
                    _rec["tasks_total"] += 1
                    if _st in ("open", "accepted", "in_progress", "review"):
                        _rec["tasks_active"] += 1
                    elif _st == "done":
                        _rec["tasks_done"] += 1
                    elif _st == "cancelled":
                        _rec["tasks_cancelled"] += 1
                    if (_ts or 0.0) > _rec["last_active_ts"]:
                        _rec["last_active_ts"] = _ts or 0.0
            _conn.close()
        except Exception:
            pass
    try:
        if _PG:
            raise ImportError  # Postgres handled above; skip the Lance fallback
        import lancedb
        for db_path in (
            os.environ.get("VEILGUARD_AUDIT_DB_PATH"),
            "/tcmm-data/veilguard/tcmm.db",
            "C:/Users/rudol/Documents/veilguard/tcmm-data/veilguard/tcmm.db",
            "/home/rudol/veilguard/tcmm-data/veilguard/tcmm.db",
        ):
            if not db_path:
                continue
            try:
                db = lancedb.connect(db_path)
                t = db.open_table("agent_tasks")
                arr = (
                    t.search()
                    .select(["owner_id","status","updated_ts","id"])
                    .limit(2000)
                    .to_arrow()
                )
                for i in range(arr.num_rows):
                    owner = arr.column("owner_id")[i].as_py() or ""
                    if not owner:
                        continue
                    rec = by_owner.setdefault(owner, {
                        "tasks_total": 0, "tasks_active": 0,
                        "tasks_done": 0, "tasks_cancelled": 0,
                        "last_active_ts": 0.0,
                    })
                    rec["tasks_total"] += 1
                    st = arr.column("status")[i].as_py() or ""
                    if st in ("open","accepted","in_progress","review"):
                        rec["tasks_active"] += 1
                    elif st == "done":
                        rec["tasks_done"] += 1
                    elif st == "cancelled":
                        rec["tasks_cancelled"] += 1
                    ts = arr.column("updated_ts")[i].as_py() or 0.0
                    if ts > rec["last_active_ts"]:
                        rec["last_active_ts"] = ts
                break  # first reachable DB wins
            except Exception:
                continue
    except ImportError:
        pass  # lancedb not installed; degrade gracefully

    # ── 3. Recent approvals (last 5 review_decision comments) ──────
    recent_approvals: list[dict] = []
    if _PG:
        try:
            import psycopg2
            _conn = psycopg2.connect(_PG_DSN)
            with _conn.cursor() as _cur:
                _cur.execute("SELECT ts, author_id, task_id, body FROM task_comments "
                             "WHERE kind='review_decision' ORDER BY ts DESC LIMIT 5")
                for _ts, _author, _tid, _body in _cur.fetchall():
                    recent_approvals.append({"ts": _ts, "by": _author, "task_id": _tid,
                                             "verdict": (_body or "")[:60]})
            _conn.close()
        except Exception:
            pass
    try:
        if _PG:
            raise ImportError  # Postgres handled above; skip the Lance fallback
        import lancedb
        for db_path in (
            os.environ.get("VEILGUARD_AUDIT_DB_PATH"),
            "/tcmm-data/veilguard/tcmm.db",
            "C:/Users/rudol/Documents/veilguard/tcmm-data/veilguard/tcmm.db",
            "/home/rudol/veilguard/tcmm-data/veilguard/tcmm.db",
        ):
            if not db_path:
                continue
            try:
                db = lancedb.connect(db_path)
                c = db.open_table("task_comments")
                arr = (
                    c.search()
                    .where("kind = 'review_decision'")
                    .select(["ts","author_id","task_id","body"])
                    .limit(200)
                    .to_arrow()
                )
                rows = []
                for i in range(arr.num_rows):
                    rows.append({
                        "ts":         arr.column("ts")[i].as_py(),
                        "by":         arr.column("author_id")[i].as_py(),
                        "task_id":    arr.column("task_id")[i].as_py(),
                        "verdict":    (arr.column("body")[i].as_py() or "")[:60],
                    })
                rows.sort(key=lambda r: r["ts"], reverse=True)
                recent_approvals = rows[:5]
                break
            except Exception:
                continue
    except ImportError:
        pass

    # ── 4. Bridge / capability info per connected daemon (Phase 0.0.4) ─
    bridges_info: list[dict] = []
    try:
        from core.client_bridge import all_bridges
        for uid, b in all_bridges().items():
            bridges_info.append({
                "user_id":  uid,
                "client_id": getattr(b, "client_id", "") or "",
                "platform": getattr(b, "platform", "") or "",
                "client_capabilities": sorted(getattr(b, "capabilities", set()) or []),
                "server_capabilities_advertised": sorted(
                    getattr(b, "server_capabilities_advertised", set()) or []
                ),
                "connected": bool(getattr(b, "connected", False)),
            })
    except Exception:
        pass

    # ── 5. Compose response ─────────────────────────────────────────
    for p in personas:
        rec = by_owner.get(p["agent_id"], {})
        p["tasks_total"]    = rec.get("tasks_total", 0)
        p["tasks_active"]   = rec.get("tasks_active", 0)
        p["tasks_done"]     = rec.get("tasks_done", 0)
        p["tasks_cancelled"] = rec.get("tasks_cancelled", 0)
        last = rec.get("last_active_ts", 0.0) or 0.0
        p["last_active_ts"]    = last
        p["last_active_age_s"] = int(now - last) if last > 0 else None

    return JSONResponse({
        "agents":           personas,
        "agents_dir":       str(agents_dir) if agents_dir else None,
        "recent_approvals": recent_approvals,
        "bridges":          bridges_info,
        "generated_ts":     now,
    })
