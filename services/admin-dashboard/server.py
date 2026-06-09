"""Veilguard Admin Dashboard — FastAPI service.

Auth: LibreChat refreshToken cookie + ``role == "ADMIN"`` (see auth.py).

Surface (all auth-gated except /healthz):

  GET /                        dashboard HTML
  GET /healthz                 unauthenticated probe
  GET /api/health              one-shot service status
  GET /api/recall-perf         TCMM workers + queues + adapter
  GET /api/host                CPU / mem / disk / load
  GET /api/lance               table sizes + fragmentation + FTS
  GET /api/nlp-coverage        per-row NLP work coverage (?user_id= optional)
  GET /api/redactions/overview ?window=24h|7d|30d|custom &start_ts &end_ts
                               &user_id=...
  GET /api/redactions/recent   ?window= &user_id= &limit=
  GET /api/messages/per-user   ?window= top-N msg-volume table
  GET /api/users               combined LibreChat + redaction rollup
  GET /api/users/{uid}/detail  per-user drill-down

# Window encoding
The ``window`` query param accepts presets ``24h``, ``7d``, ``30d``,
``all``, or any string ending in ``h`` / ``d``. ``custom`` (or any
unrecognised value) defers to ``start_ts`` / ``end_ts`` Unix-second
parameters. Without explicit start/end, ``window`` resolves to
"now - N seconds" up to "now".
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from auth import require_admin  # noqa: E402
from data import (  # noqa: E402
    host_metrics,
    lance_stats,
    librechat_users,
    pii_audit_stats,
    tcmm_stats,
)


logger = logging.getLogger("veilguard.admin-dashboard")
logging.basicConfig(level=logging.INFO)


app = FastAPI(title="Veilguard Admin", version="2.0.0")
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")
templates = Jinja2Templates(directory=str(_HERE / "templates"))


# ── Time-window parsing ──────────────────────────────────────────────────────


_WINDOW_PRESETS = {
    "1h": 3600,
    "24h": 24 * 3600,
    "7d": 7 * 86400,
    "30d": 30 * 86400,
    "90d": 90 * 86400,
    "all": 365 * 86400,
}


_REL_RE = re.compile(r"^(\d+)\s*([hd])$")


def _resolve_window(
    window: Optional[str],
    start_ts: Optional[float],
    end_ts: Optional[float],
) -> tuple[float, float]:
    """Map (window?, start_ts?, end_ts?) → concrete (start_ts, end_ts).

    Priority: explicit start_ts/end_ts > preset > default 24h.
    """
    now = time.time()
    if start_ts is not None or end_ts is not None:
        return float(start_ts or 0), float(end_ts or now)
    if window:
        secs = _WINDOW_PRESETS.get(window)
        if secs is None:
            m = _REL_RE.match(window.strip())
            if m:
                n, unit = int(m.group(1)), m.group(2)
                secs = n * (3600 if unit == "h" else 86400)
        if secs:
            return now - secs, now
    return now - 24 * 3600, now


# ── Page ─────────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, user=Depends(require_admin)):
    return templates.TemplateResponse(
        request, "dashboard.html", {"user": user},
    )


# ── Probes ───────────────────────────────────────────────────────────────────


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "veilguard-admin"}


# ── API endpoints (auth-gated) ───────────────────────────────────────────────


@app.get("/api/health")
async def api_health(user=Depends(require_admin)):
    tcmm_h, lance_o = await asyncio.gather(
        tcmm_stats.health(),
        asyncio.to_thread(lance_stats.overview),
    )
    fts = await asyncio.to_thread(lance_stats.fts_index_status)
    return {
        "tcmm": tcmm_h,
        "lance": {
            "ok": "error" not in lance_o,
            "tables": len(lance_o.get("tables", [])),
            "error": lance_o.get("error"),
        },
        "fts_index": fts,
        "user": {"email": user.get("email"), "name": user.get("name")},
    }


@app.get("/api/recall-perf")
async def api_recall_perf(user=Depends(require_admin)):
    return await tcmm_stats.workers()


@app.get("/api/host")
async def api_host(user=Depends(require_admin)):
    return await asyncio.to_thread(host_metrics.overview)


@app.get("/api/lance")
async def api_lance(user=Depends(require_admin)):
    overview = await asyncio.to_thread(lance_stats.overview)
    fts = await asyncio.to_thread(lance_stats.fts_index_status)
    return {"overview": overview, "fts_index": fts}


@app.get("/api/nlp-coverage")
async def api_nlp_coverage(
    user_id: Optional[str] = None,
    user=Depends(require_admin),
):
    return await asyncio.to_thread(lance_stats.nlp_coverage, user_id)


@app.get("/api/redactions/overview")
async def api_redactions_overview(
    window: Optional[str] = "24h",
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    user_id: Optional[str] = None,
    user=Depends(require_admin),
):
    s, e = _resolve_window(window, start_ts, end_ts)
    data, names = await asyncio.gather(
        asyncio.to_thread(
            pii_audit_stats.overview,
            None,
            start_ts=s, end_ts=e, user_id=user_id,
        ),
        librechat_users.users_by_id(),
    )
    if user_id:
        u = names.get(user_id)
        if u:
            data["user"] = {**u, "user_id": user_id}
    return data


@app.get("/api/requests/{aid}")
async def api_request_detail(aid: int, user=Depends(require_admin)):
    """One audit row's full record incl. REDACTED content."""
    detail, names = await asyncio.gather(
        asyncio.to_thread(pii_audit_stats.request_detail, aid),
        librechat_users.users_by_id(),
    )
    if detail.get("error"):
        raise HTTPException(status_code=404, detail=detail["error"])
    u = names.get(detail.get("user_id") or "")
    if u:
        detail["user_name"] = u.get("name")
        detail["user_email"] = u.get("email")
    return detail


@app.get("/api/cache/overview")
async def api_cache_overview(
    window: Optional[str] = "24h",
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    user_id: Optional[str] = None,
    user=Depends(require_admin),
):
    """Prompt-cache (KV-cache) hit/write/read stats over a window.

    Reads the cache_create / cache_read / tokens_input columns Anthropic
    populates on each FROM_LLM audit row, aggregates them into hit-rate,
    token totals, time-series for a stacked chart, and a Sonnet-equivalent
    dollar savings figure. Same window/user_id contract as the other
    /api/redactions/* endpoints, so the dashboard's existing filter chips
    just work.
    """
    s, e = _resolve_window(window, start_ts, end_ts)
    data, names = await asyncio.gather(
        asyncio.to_thread(
            pii_audit_stats.cache_overview,
            None,
            start_ts=s, end_ts=e, user_id=user_id,
        ),
        librechat_users.users_by_id(),
    )
    if user_id:
        u = names.get(user_id)
        if u:
            data["user"] = {**u, "user_id": user_id}
    return data


@app.get("/api/redactions/recent")
async def api_redactions_recent(
    limit: int = 30,
    window: Optional[str] = "24h",
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    user_id: Optional[str] = None,
    user=Depends(require_admin),
):
    s, e = _resolve_window(window, start_ts, end_ts)
    rows, names = await asyncio.gather(
        asyncio.to_thread(
            pii_audit_stats.recent_redactions,
            limit, start_ts=s, end_ts=e, user_id=user_id,
        ),
        librechat_users.users_by_id(),
    )
    # Enrich each row with display name + email
    for r in rows:
        u = names.get(r.get("user_id") or "")
        if u:
            r["user_name"] = u.get("name")
            r["user_email"] = u.get("email")
    return rows


@app.get("/api/blocks/recent")
async def api_blocks_recent(
    limit: int = 40,
    user=Depends(require_admin),
):
    """Recent TCMM archive blocks + block_class — classification spot-check."""
    return await asyncio.to_thread(pii_audit_stats.recent_blocks, limit)


@app.get("/api/redactions/suspected-misses")
async def api_redactions_suspected_misses(
    limit: int = 40,
    window: Optional[str] = "24h",
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    user_id: Optional[str] = None,
    user=Depends(require_admin),
):
    """Raw PII the redactor MISSED in recent TO_LLM content (gap scanner)."""
    s, e = _resolve_window(window, start_ts, end_ts)
    return await asyncio.to_thread(
        pii_audit_stats.suspected_pii_misses,
        limit, start_ts=s, end_ts=e, user_id=user_id,
    )


@app.get("/api/messages/per-user")
async def api_messages_per_user(
    window: Optional[str] = "24h",
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    direction: Optional[str] = None,
    top_n: int = 50,
    user=Depends(require_admin),
):
    """Per-user message volume table.

    Returns ALL LibreChat users (left-joined with the audit data) so
    inactive users still appear with zero counts, instead of being
    filtered out when their last activity falls outside the window.
    Users who exist only in the audit log (test scripts like
    ``multi-tester``, ``e2e-tester``) still appear too — they show
    ``name=None`` and the dashboard renders that as ``—``.
    """
    s, e = _resolve_window(window, start_ts, end_ts)
    audit_rows, all_users, agent_rows = await asyncio.gather(
        asyncio.to_thread(
            pii_audit_stats.messages_per_user,
            None, top_n, start_ts=s, end_ts=e, direction=direction,
        ),
        librechat_users.all_users(),
        asyncio.to_thread(
            pii_audit_stats.per_agent,
            None, 50, start_ts=s, end_ts=e,
        ),
    )

    by_uid = {r["user_id"]: r for r in audit_rows}
    uid_map = {u["_id"]: u for u in all_users}

    rows: list[dict] = []
    seen_uids: set[str] = set()

    # 1. Every LibreChat user, with audit overlay if present.
    for u in all_users:
        uid = u["_id"]
        seen_uids.add(uid)
        a = by_uid.get(uid, {})
        rows.append({
            "user_id":         uid,
            "type":            "user",
            "name":            u.get("name"),
            "email":           u.get("email"),
            "role":            u.get("role"),
            "calls":           a.get("calls", 0),
            "to_llm":          a.get("to_llm", 0),
            "from_llm":        a.get("from_llm", 0),
            "to_llm_bytes":    a.get("to_llm_bytes", 0),
            "from_llm_bytes":  a.get("from_llm_bytes", 0),
            "tokens_in":       a.get("tokens_in", 0),
            "tokens_out":      a.get("tokens_out", 0),
            "conversations":   a.get("conversations", 0),
            "redactions":      a.get("redactions", {}),
            "redactions_total": a.get("redactions_total", 0),
            "last_seen":       a.get("last_seen", 0),
        })

    # 2. Audit-only user_ids (test scripts not in users collection).
    for uid, a in by_uid.items():
        if uid in seen_uids:
            continue
        rows.append({
            "user_id":         uid,
            "type":            "user",
            "name":            None,
            "email":           None,
            "role":            None,
            **{k: a.get(k) for k in ("calls", "to_llm", "from_llm",
                                     "to_llm_bytes", "from_llm_bytes",
                                     "tokens_in", "tokens_out",
                                     "conversations", "redactions",
                                     "redactions_total", "last_seen")},
        })

    # 3. Append the agent-runtime ICs (director/researcher/builder/critic)
    #    as their own rows in the SAME list + columns, attributed to the
    #    user they acted for.  Their activity is ALSO inside that user's
    #    totals above — they're the breakdown, not separate billing.
    for a in agent_rows:
        owner = uid_map.get(a.get("owner_user_id") or "", {})
        rows.append({
            "user_id":          "agent:" + a["agent_id"],
            "type":             "agent",
            "name":             a["agent_id"],
            "email":            owner.get("email"),
            "role":             "agent",
            "calls":            a.get("calls", 0),
            "to_llm":           a.get("to_llm", 0),
            "from_llm":         a.get("from_llm", 0),
            "to_llm_bytes":     a.get("to_llm_bytes", 0),
            "from_llm_bytes":   0,
            "tokens_in":        a.get("tokens_in", 0),
            "tokens_out":       a.get("tokens_out", 0),
            "conversations":    a.get("conversations", 0),
            "redactions":       a.get("redactions", {}),
            "redactions_total": a.get("redactions_total", 0),
            "last_seen":        a.get("last_seen", 0),
            "owner_user_id":    a.get("owner_user_id") or "",
        })

    rows.sort(key=lambda r: (r.get("to_llm", 0), r.get("calls", 0)), reverse=True)
    return {"rows": rows, "start_ts": s, "end_ts": e}


@app.get("/api/agents/per-agent")
async def api_agents_per_agent(
    window: Optional[str] = "24h",
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    user_id: Optional[str] = None,
    top_n: int = 30,
    user=Depends(require_admin),
):
    """Per-agent (IC) call + token rollup — *which agent spent the quota*.

    Distinct from /api/messages/per-user, which collapses every agent
    dispatch under the libreuser id the agents inherit.  This groups by
    ``extra.agent_id`` (director / researcher / builder / critic) so the
    operator can see the agents as distinct actors.
    """
    s, e = _resolve_window(window, start_ts, end_ts)
    rows = await asyncio.to_thread(
        pii_audit_stats.per_agent,
        None, top_n, start_ts=s, end_ts=e, user_id=user_id,
    )
    return {"rows": rows, "start_ts": s, "end_ts": e}


@app.get("/api/users")
async def api_users(
    window: Optional[str] = "7d",
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    user=Depends(require_admin),
):
    s, e = _resolve_window(window, start_ts, end_ts)
    users, summary, by_msg = await asyncio.gather(
        librechat_users.all_users(),
        librechat_users.usage_summary(),
        librechat_users.per_user_messages(50),
    )
    redact = await asyncio.to_thread(
        pii_audit_stats.per_user, None, 50, start_ts=s, end_ts=e,
    )
    redact_by_id = {r["user_id"]: r for r in redact}
    msg_by_id = {r["user_id"]: r for r in by_msg}

    rows = []
    for u in users:
        uid = u["_id"]
        m = msg_by_id.get(uid, {})
        r = redact_by_id.get(uid, {})
        rows.append(
            {
                "user_id": uid,
                "name": u.get("name"),
                "email": u.get("email"),
                "role": u.get("role"),
                "messages": m.get("messages", 0),
                "tokens": m.get("tokens", 0),
                "last_message": m.get("last_message"),
                # Window-scoped metrics
                "redactions_in_window": r.get("redactions_total", 0),
                "calls_in_window": r.get("calls", 0),
                "to_llm_in_window": r.get("to_llm", 0),
                "from_llm_in_window": r.get("from_llm", 0),
                "tokens_in_window": (r.get("tokens_in", 0) + r.get("tokens_out", 0)),
            }
        )
    rows.sort(key=lambda r: r.get("messages", 0), reverse=True)
    return {"summary": summary, "users": rows, "start_ts": s, "end_ts": e}


@app.get("/api/users/{user_id}/detail")
async def api_user_detail(
    user_id: str,
    window: Optional[str] = "7d",
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    user=Depends(require_admin),
):
    """Drill-down: everything we know about one user in the window."""
    s, e = _resolve_window(window, start_ts, end_ts)
    overview, recent, names, nlp = await asyncio.gather(
        asyncio.to_thread(
            pii_audit_stats.overview,
            None, start_ts=s, end_ts=e, user_id=user_id,
        ),
        asyncio.to_thread(
            pii_audit_stats.recent_redactions,
            50, start_ts=s, end_ts=e, user_id=user_id,
        ),
        librechat_users.users_by_id(),
        asyncio.to_thread(lance_stats.nlp_coverage, user_id),
    )
    info = names.get(user_id)
    if not info:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "user_id": user_id,
        "user": info,
        "start_ts": s,
        "end_ts": e,
        "overview": overview,
        "recent_redactions": recent,
        "nlp_coverage": nlp,
    }


# ── Entry point ──────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("ADMIN_PORT", "8820"))
    host = os.environ.get("ADMIN_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port, log_level="info")
