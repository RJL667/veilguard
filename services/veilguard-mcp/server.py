"""
Veilguard unified MCP server.

Single MCP entry-point that LibreChat (and any other MCP-capable host)
talks to for memory + tool-creation primitives. Started as just the two
new TCMM tools (search_memory, traverse_memory); the long-term plan is
to fold forge tools in here too so the LibreChat tool picker has one
"veilguard" group instead of three (forge, tcmm, …).

Architecture:
    LibreChat (Agents) ──MCP/SSE──▶ veilguard-mcp (this, port 8812)
                                          │
                                          └─HTTP─▶ tcmm-service (port 8811)
                                                       │
                                                       └─ adapters.memory_search

Why HTTP and not direct Python imports?
    The tcmm-service owns the per-conversation session pool, the FAISS
    indices, and the warm NLP/embedder. Spawning another TCMM in this
    process would double the memory + cold-start cost. HTTP is a thin
    adapter — ~1ms overhead per call against a localhost service that's
    already paying its loadout.

Headers:
    LibreChat populates ``x-conversation-id`` and ``x-user-id`` on every
    SSE/MCP message via the ``headers`` block in librechat.yaml. We
    extract them from the per-request context and forward them to
    tcmm-service so it routes to the right per-conversation TCMM.

Run:
    python services/veilguard-mcp/server.py
    # listens on http://0.0.0.0:8812/sse  (matches forge / sub-agents
    # transport, so LibreChat MCP wiring is identical)
"""

import os
import json
import logging
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP, Context

# ── Config ───────────────────────────────────────────────────────────
TCMM_SERVICE_URL = os.environ.get("TCMM_SERVICE_URL", "http://localhost:8811").rstrip("/")
PORT = int(os.environ.get("VEILGUARD_MCP_PORT", "8812"))
HOST = os.environ.get("VEILGUARD_MCP_HOST", "0.0.0.0")
# Per-call HTTP timeout. Search is bounded (recall pipeline is ~1-2s
# steady-state), traverse is even smaller. 30s is generous head-room.
HTTP_TIMEOUT = float(os.environ.get("VEILGUARD_MCP_TIMEOUT", "30"))
# Agent-runtime (the Director org). delegate_to_org POSTs tasks into its durable
# ledger. From the host this is the container's published port (5000:5000).
AGENT_RUNTIME_URL = os.environ.get("AGENT_RUNTIME_URL", "http://localhost:5000").rstrip("/")
INTERNAL_SECRET = os.environ.get("VEILGUARD_INTERNAL_SECRET", "")

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [VEILGUARD-MCP] %(message)s",
)
logger = logging.getLogger("veilguard-mcp")


# ── FastMCP server ───────────────────────────────────────────────────
mcp = FastMCP(
    "veilguard",
    instructions=(
        "Veilguard memory + tool-creation tools. Use search_memory to "
        "find relevant past facts, decisions, and conversations across "
        "all of this user's history. Use traverse_memory to read the "
        "temporal arc around a specific block surfaced by search. "
        "Memory is user-scoped by default — recall sees other "
        "conversations of the same user, not other users' data."
    ),
)


def _ctx_ids(ctx: Optional[Context]) -> tuple[str, str]:
    """Pull conversation_id + user_id from the MCP request headers.

    LibreChat wires ``x-conversation-id`` / ``x-user-id`` via the
    ``headers`` block in librechat.yaml. They land on the SSE message
    metadata and FastMCP exposes them on the per-call Context.

    Returns ("", "") when called outside a request context (e.g. from
    a test harness) — the tcmm-service tolerates blanks and will pick
    its active session, which is the right fallback for ad-hoc calls.
    """
    if ctx is None:
        return "", ""
    try:
        # FastMCP >=1.2 exposes request headers via request_context.request
        req = getattr(ctx, "request_context", None)
        request = getattr(req, "request", None) if req else None
        headers = getattr(request, "headers", None) if request else None
        if headers is None:
            return "", ""
        # Header lookup is case-insensitive; httpx-style Headers obj
        return (
            headers.get("x-conversation-id", "") or "",
            headers.get("x-user-id", "") or "",
        )
    except Exception:
        return "", ""


def _http_post(path: str, payload: dict) -> dict:
    """POST to tcmm-service and return parsed JSON. Re-raises on HTTP error."""
    url = f"{TCMM_SERVICE_URL}{path}"
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


# ── Tools ────────────────────────────────────────────────────────────

@mcp.tool()
def search_memory(
    query: str,
    ctx: Context,
    max_results: int = 10,
    temporal_window: int = 2,
    text_chars: int = 500,
    preview_chars: int = 100,
    include_dream: bool = True,
    scope: str = "user",
) -> str:
    """Search this user's persistent memory for content relevant to the query.

    Returns ranked hits with full text, the ISO date the block was
    created, semantic overlay (entities, topics, claims), block role,
    and a temporal preview of the N neighbours either side of each hit.

    Use this when the user asks about something they've discussed
    before, or when you need to ground an answer in past facts the
    short-term memory context didn't already include.

    Args:
        query: Natural-language query — the same phrasing you'd
            search for in a search engine.
        max_results: Cap on returned hits (default 10).
        temporal_window: Neighbours per side included as previews on
            each hit (default 2). Set 0 to skip previews.
        text_chars: Truncate each hit's main text at this many chars
            (default 500). Set 0 for full text.
        preview_chars: Truncate temporal-neighbour previews at this
            many chars (default 100).
        include_dream: Whether to include consolidated/dream knowledge
            nodes alongside raw conversational blocks (default true).
        scope: "user" (default) — recall across all of this user's
            conversations. "namespace" — only this conversation.

    Returns:
        JSON string with shape::

            {
              "query": str, "scope": str, "count": int,
              "results": [
                {
                  "aid": int, "score": float, "text": str,
                  "date": "YYYY-MM-DD HH:MM",
                  "role": "user"|"assistant"|"tool"|"system",
                  "entities": [...], "topics": [...], "claims": [...],
                  "temporal": {"prev": [...], "next": [...]}  # if window>0
                }, ...
              ]
            }
    """
    conv_id, user_id = _ctx_ids(ctx)
    payload = {
        "query": query,
        "conversation_id": conv_id,
        "user_id": user_id,
        "max_results": max_results,
        "temporal_window": temporal_window,
        "text_chars": text_chars,
        "preview_chars": preview_chars,
        "include_dream": include_dream,
        "scope": scope,
    }
    try:
        result = _http_post("/search", payload)
    except httpx.HTTPError as e:
        return json.dumps({"error": f"tcmm-service: {e}", "results": []})
    return json.dumps(result, default=str)


@mcp.tool()
def delegate_to_org(
    brief: str,
    ctx: Context,
    deliverable_spec: str = "",
    owner_hint: str = "",
) -> str:
    """Hand a task to the Veilguard agent ORG to work on asynchronously.

    Use this when the user wants real, tracked, governed work done by the
    organization -- NOT a quick inline answer. The task lands in the org's
    durable ledger; an IC (researcher/builder) executes it, a critic reviews
    it, and it appears live in the user's Work Queue panel with cost tracking.

    Rule of thumb: if the user needs the answer *now, in this conversation*,
    use your own tools (or a quick helper). If the work should be *tracked,
    reviewed, or done by the org over time*, use delegate_to_org.

    Args:
        brief: A clear, self-contained description of what the org should do.
            The assigned agent will NOT see this conversation, so put all the
            context it needs into the brief itself.
        deliverable_spec: Optional -- what the finished artifact should be,
            e.g. "a 300-500 word summary written to team/research/x.md with
            sections Overview / Findings / Notes". A concrete spec gives the
            critic a real acceptance check.
        owner_hint: Optional persona to assign -- one of: researcher, builder,
            critic-claim, critic-prose. Anything else (or blank) -> researcher.

    Returns:
        JSON: {"task_id", "owner_id", "status": "open", "message"} on success,
        or {"error": str} on failure.
    """
    _conv, user_id = _ctx_ids(ctx)
    if not user_id:
        return json.dumps({
            "error": "no user context -- cannot scope the task to a tenant; "
                     "call from a LibreChat session",
        })
    # Veilguard scopes the Work Queue by the LibreChat user id (veilguardClient.js
    # proxyRuntime sets tenant_id = user_id), so the delegated task must use the
    # same id for both to surface in the panel.
    payload = {
        "tenant_id": user_id,
        "user_id": user_id,
        "brief": brief,
        "deliverable_spec": deliverable_spec,
        "owner_hint": owner_hint,
    }
    headers = {"X-Internal-Secret": INTERNAL_SECRET} if INTERNAL_SECRET else {}
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            resp = client.post(f"{AGENT_RUNTIME_URL}/delegate", json=payload, headers=headers)
            resp.raise_for_status()
            return json.dumps(resp.json(), default=str)
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.text[:300]
        except Exception:
            pass
        return json.dumps({"error": f"agent-runtime /delegate {e.response.status_code}: {detail}"})
    except httpx.HTTPError as e:
        return json.dumps({"error": f"agent-runtime unreachable: {e}"})


@mcp.tool()
def traverse_memory(
    aid: int,
    ctx: Context,
    before: int = 3,
    after: int = 3,
    text_chars: int = 500,
    include_anchor: bool = True,
    scope: str = "user",
) -> str:
    """Walk the conversational arc around a specific memory block.

    Pure-read temporal walk: no recall, no scoring. Use this as a
    follow-up to search_memory when a hit looks relevant and you want
    to read the surrounding conversation (what was said before / after
    that exact block).

    Dream / consolidated-knowledge nodes don't have temporal
    neighbours — pass an episodic block's aid, not a dream aid.

    Args:
        aid: The anchor block's stable archive ID. Get this from the
            ``aid`` field of a search_memory hit.
        before: Number of blocks to walk backward from the anchor
            (default 3).
        after: Number of blocks to walk forward from the anchor
            (default 3).
        text_chars: Truncate each block's text at this many chars
            (default 500). Set 0 for full text.
        include_anchor: Whether to include the anchor block itself
            in the returned list (default true).
        scope: "user" (default) — resolve cross-namespace.
            "namespace" — only this conversation's chain.

    Returns:
        JSON string with shape::

            {
              "anchor_aid": int,
              "before_count": int, "after_count": int,
              "blocks": [
                {hit dict, "position": -N},
                ...
                {hit dict, "position": 0, "is_anchor": true},
                ...
                {hit dict, "position": N},
              ]
            }

        On error (anchor missing or dream-node anchor)::

            {"anchor_aid": int, "blocks": [], "error": str}
    """
    conv_id, user_id = _ctx_ids(ctx)
    payload = {
        "aid": aid,
        "conversation_id": conv_id,
        "user_id": user_id,
        "before": before,
        "after": after,
        "text_chars": text_chars,
        "include_anchor": include_anchor,
        "scope": scope,
    }
    try:
        result = _http_post("/traverse", payload)
    except httpx.HTTPError as e:
        return json.dumps({"error": f"tcmm-service: {e}", "blocks": []})
    return json.dumps(result, default=str)


# ── Run ──────────────────────────────────────────────────────────────
#
# We mirror forge's exact SSE wiring rather than calling ``mcp.sse_app()``
# directly. The high-level ``sse_app()`` helper in some FastMCP versions
# returns a Starlette app whose Host-header validation rejects requests
# coming from a Docker bridge with ``Host: host.docker.internal:8812``,
# producing HTTP 421 ("Misdirected Request"). Manually building the
# Starlette app with ``SseServerTransport`` and a bare ``Route`` matches
# what forge runs at :8810 — and forge has been working from the
# LibreChat container for months. Same code = same behaviour.
if __name__ == "__main__":
    import uvicorn
    from starlette.applications import Starlette
    from starlette.responses import Response, JSONResponse
    from starlette.routing import Route, Mount
    from mcp.server.sse import SseServerTransport

    sse = SseServerTransport("/messages/")

    # [SSE_NONE_RETURN_FIX 2026-06-03] Starlette 0.52+ requires route handlers
    # to return a Response; connect_sse() streams itself so returning None
    # crashes with "TypeError: 'NoneType' object is not callable" on teardown.
    # Return a Response shim that owns the SSE lifecycle. See sub-agents.
    class _SseResponse:
        """Minimal Response shim that owns the SSE stream lifecycle."""
        async def __call__(self, scope, receive, send):
            async with sse.connect_sse(scope, receive, send) as (read, write):
                await mcp._mcp_server.run(
                    read, write, mcp._mcp_server.create_initialization_options()
                )

    async def handle_sse(request):
        return _SseResponse()

    # 2026-05-14: LibreChat's MCP client probes
    #   HEAD /sse
    #   GET  /.well-known/oauth-protected-resource
    #   GET  /.well-known/oauth-protected-resource/sse
    # to detect OAuth-protected servers before opening the real SSE stream.
    # By default Starlette routes accept both GET and HEAD on the same path,
    # so HEAD /sse falls through to handle_sse → SseServerTransport, which
    # raises "Request validation failed" because there's no Accept:text/event-stream
    # on a HEAD probe. The uvicorn ASGI middleware then prints a 30-line
    # traceback. Add an explicit HEAD handler and OAuth-discovery stubs so
    # LibreChat's probes get a clean 204 / 404 instead of a 421 + exception.
    async def handle_sse_head(request):
        return Response(status_code=204)

    async def handle_oauth_discovery_miss(request):
        # 404 with an empty JSON body — explicit so it doesn't trigger the
        # default Starlette HTML 404 page.
        return JSONResponse({"error": "no OAuth on this server"}, status_code=404)

    app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Route("/sse", endpoint=handle_sse_head, methods=["HEAD"]),
            Route(
                "/.well-known/oauth-protected-resource",
                endpoint=handle_oauth_discovery_miss,
                methods=["GET", "HEAD"],
            ),
            Route(
                "/.well-known/oauth-protected-resource/sse",
                endpoint=handle_oauth_discovery_miss,
                methods=["GET", "HEAD"],
            ),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

    logger.info(
        f"Starting veilguard-mcp on {HOST}:{PORT}, "
        f"backing TCMM at {TCMM_SERVICE_URL}"
    )
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
