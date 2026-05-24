"""FastAPI entry — agent-runtime service.

Exposes:
  POST /agent/query     main entry; runs SDK query() with our middleware
  GET  /health          liveness probe (docker-compose healthcheck)
  GET  /agents          list registered personas (for sidebar UI)
  GET  /constitution    return parsed constitution (read-only)

Wiring:
  pii-proxy receives an Anthropic-bound request from LibreChat.
  If the request is "agent-runtime-enabled" (header or env flag), proxy
  forwards to http://agent-runtime:5000/agent/query.  Otherwise, proxy
  goes direct to api.anthropic.com (current behaviour).

  This service does NOT redact PII itself — that's pii-proxy's job and
  happens before the request reaches us.  We assume incoming content
  is already redacted.  We DO send audit rows to the same Lance table
  pii-proxy uses, so cost roll-up + replay queries work uniformly.
"""

import logging
import sys

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__, config
from .personas.loader import load_personas, PersonaRegistry
from .runtime import run_agent_query
from .constitution.loader import load_constitution, ConstitutionError
from .workers import inbox_poller

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agent-runtime")


# ── App + state ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Veilguard agent-runtime",
    version=__version__,
    description=(
        "Embedded Claude Agent SDK runtime. Owns the LLM-tool-LLM loop "
        "for Anthropic-bound conversations. PII proxy continues to handle "
        "redaction + multi-provider routing + audit."
    ),
)

_persona_registry: PersonaRegistry | None = None
_constitution: dict | None = None


@app.on_event("startup")
async def _startup() -> None:
    """Validate config + load personas + parse constitution.

    Fail fast and loud if anything's missing — better than discovering
    via a 500 on the first real request.
    """
    errs = config.validate()
    if errs:
        for e in errs:
            logger.error(f"[startup] {e}")
        # Don't abort — health check should still respond so ops sees the
        # container is alive but in bad state.  /agent/query will 503.
        return

    global _persona_registry, _constitution

    try:
        _persona_registry = load_personas(config.AGENTS_DIR)
        logger.info(
            f"[startup] loaded {len(_persona_registry)} personas: "
            f"{', '.join(p.agent_id for p in _persona_registry.all())}"
        )
    except Exception as e:
        logger.exception(f"[startup] persona load failed: {e}")
        _persona_registry = PersonaRegistry({})

    try:
        if config.CONSTITUTION_PATH.exists():
            _constitution = load_constitution(config.CONSTITUTION_PATH)
            objectives = _constitution.get("objectives", [])
            constraints = _constitution.get("constraints", [])
            logger.info(
                f"[startup] constitution loaded — "
                f"{len(objectives)} objectives, {len(constraints)} constraints"
            )
        else:
            logger.warning(
                f"[startup] CONSTITUTION.md not found at {config.CONSTITUTION_PATH} — "
                "proposals will run without objective alignment"
            )
            _constitution = None
    except ConstitutionError as e:
        logger.error(f"[startup] constitution parse failed: {e}")
        _constitution = None

    # Start the IC inbox poller (background asyncio task).
    # ICs (Researcher, Builder, critic-claim, critic-prose) receive
    # assigned Tasks via this loop, not via chat turns.
    if _persona_registry and len(_persona_registry) > 0:
        try:
            await inbox_poller.start(_persona_registry)
            logger.info("[startup] inbox poller started")
        except Exception as e:
            logger.exception(f"[startup] inbox poller failed to start: {e}")


@app.on_event("shutdown")
async def _shutdown() -> None:
    """Stop the inbox poller cleanly so in-flight dispatches finish."""
    try:
        await inbox_poller.stop()
        logger.info("[shutdown] inbox poller stopped")
    except Exception as e:
        logger.warning(f"[shutdown] inbox poller stop failed: {e}")


# ── Routes ───────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> JSONResponse:
    """Liveness + readiness probe.

    Returns 200 even if startup partially failed (e.g. CONSTITUTION.md
    missing).  Returns 503 only if the process is in a state that
    cannot serve /agent/query at all.
    """
    cfg_errs = config.validate()
    ready = not cfg_errs and _persona_registry is not None
    body = {
        "service": "agent-runtime",
        "version": __version__,
        "ready": ready,
        "personas_loaded": (
            len(_persona_registry) if _persona_registry else 0
        ),
        "constitution_loaded": _constitution is not None,
        "config_errors": cfg_errs,
    }
    code = 200 if ready else 503
    return JSONResponse(body, status_code=code)


@app.get("/agents")
async def list_agents() -> JSONResponse:
    """Return the registered persona set.  Used by sidebar Agents view."""
    if not _persona_registry:
        return JSONResponse({"agents": []})
    return JSONResponse({
        "agents": [
            {
                "agent_id": p.agent_id,
                "role": p.role,
                "manager_id": p.manager_id,
                "team_id": p.team_id,
                "model": p.model,
                "tool_count": len(p.tools),
            }
            for p in _persona_registry.all()
        ]
    })


@app.get("/constitution")
async def get_constitution() -> JSONResponse:
    """Return the parsed constitution (read-only).  Used for replay/audit
    UIs that need to show what objectives a decision was scored against.
    """
    if not _constitution:
        raise HTTPException(status_code=404, detail="constitution not loaded")
    return JSONResponse(_constitution)


@app.post("/agent/query")
async def agent_query(req: Request) -> StreamingResponse:
    """Main entry.  Runs SDK query() with all middleware wired in.

    Expected JSON body:
      {
        "conversation_id": "conv-...",     # cid
        "user_id": "user-...",
        "tenant_id": "tenant-...",
        "agent_id": "director",            # which persona drives this turn
        "messages": [...],                  # Anthropic-shape messages
        "stream": true                       # SSE if true, JSON if false
      }

    Returns SSE events (text/event-stream):
      data: {"type": "text_delta", "text": "..."}
      data: {"type": "tool_call", "tool": "...", "id": "..."}
      data: {"type": "tool_result", "id": "...", "content": "..."}
      data: {"type": "status_ping", "text": "..."}    # Director status
      data: {"type": "subagent_start", "agent_id": "..."}
      data: {"type": "subagent_end", "agent_id": "...", "summary": "..."}
      data: {"type": "final_result", "result": "..."}
      data: {"type": "usage", "tokens_in": N, "cache_read": M, ...}
    """
    if not _persona_registry:
        raise HTTPException(
            status_code=503,
            detail="persona registry not loaded; check /health",
        )

    try:
        body = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")

    required = ("conversation_id", "user_id", "tenant_id", "agent_id", "messages")
    missing = [k for k in required if k not in body]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"missing required fields: {missing}",
        )

    agent_id = body["agent_id"]
    persona = _persona_registry.get(agent_id)
    if persona is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown agent_id: {agent_id}",
        )

    # Dispatch into runtime.py — which owns all the middleware composition.
    # Streaming is the default; non-streaming returns the same generator
    # collected into one JSON response.
    stream = body.get("stream", True)
    if stream:
        return StreamingResponse(
            run_agent_query(
                persona=persona,
                conversation_id=body["conversation_id"],
                user_id=body["user_id"],
                tenant_id=body["tenant_id"],
                messages=body["messages"],
                registry=_persona_registry,
                constitution=_constitution,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming fallback (for tests, debugging) — collect into list.
    events = []
    async for ev in run_agent_query(
        persona=persona,
        conversation_id=body["conversation_id"],
        user_id=body["user_id"],
        tenant_id=body["tenant_id"],
        messages=body["messages"],
        registry=_persona_registry,
        constitution=_constitution,
    ):
        events.append(ev)
    return JSONResponse({"events": events})
