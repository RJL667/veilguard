"""Per-request identity context (user_id, conversation_id).

LibreChat forwards ``x-user-id`` and ``x-conversation-id`` headers to
every MCP request. ``RequestContextMiddleware`` in server.py populates
these contextvars for the duration of each request so downstream code
(client-daemon bridge lookup, TCMM calls, agent-state scoping) reads
them instead of the old module-level ``state.active_conversation_id``
global — that was race-prone under concurrent requests and had no
notion of user identity at all, which let one LibreChat user's tool
calls route to another user's client daemon.
"""

import contextvars


# Empty-string defaults are deliberate: code downstream should treat
# "" as "no user context" and fall back to a safe behaviour (usually
# refusing to act), not crash.
current_user_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "veilguard_user_id", default=""
)
current_conversation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "veilguard_conversation_id", default=""
)

# Lineage: when a spawn_* tool forks a child agent we mint a fresh
# conversation_id for it (``sub-<parent16>-<uuid8>``) so its internal
# reasoning lands in its own TCMM namespace instead of blending into
# the parent's archive — otherwise you can't tell which blocks were
# the parent thinking vs the sub-agent thinking. The parent's cid is
# simultaneously captured as ``lineage_parent_conv`` so downstream
# (pii-proxy → TCMM) can resolve the actual ``lineage.parents`` aid
# and inherit ``lineage.root`` for cross-namespace dream pooling.
# Both stay empty on top-level LibreChat turns — only spawn_* tools
# set them, and only for the lifetime of their subtree.
current_child_conversation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "veilguard_child_conversation_id", default=""
)
current_lineage_parent_conv: contextvars.ContextVar[str] = contextvars.ContextVar(
    "veilguard_lineage_parent_conv", default=""
)


def _header_from_mcp_request(name: str) -> str:
    """Read an HTTP header from the current MCP tool-dispatch request.

    Why this exists: FastMCP dispatches tool handlers in a DIFFERENT
    asyncio task from the ASGI POST handler that carried the
    ``x-user-id`` / ``x-conversation-id`` headers — so contextvars we
    set in ``RequestContextMiddleware`` don't propagate to where tools
    read them. The MCP SDK, however, DOES plumb the original Starlette
    Request through to the dispatch task via the ``SessionMessage``'s
    ``message_metadata.request_context`` field, and re-exposes it on
    its own ``request_ctx`` ContextVar which IS set from INSIDE the
    dispatch task in ``Server._handle_request`` (right before the tool
    handler runs). So reading the SDK's ``request_ctx`` inside a tool
    handler gives us the exact request that triggered this call — no
    global-state races, no last-writer-wins under concurrent users,
    no staleness. This is the authoritative per-request source.

    Returns ``""`` when no MCP request is active (e.g. background
    ``asyncio.create_task`` workers that detached from the handler's
    context — those use ``spawn_scope(captured=...)`` to pass identity
    explicitly instead).
    """
    try:
        from mcp.server.lowlevel.server import request_ctx as _mcp_request_ctx
        ctx = _mcp_request_ctx.get()  # raises LookupError if unset
    except LookupError:
        return ""
    except Exception:
        return ""
    req = getattr(ctx, "request", None)
    if req is None:
        return ""
    try:
        headers = getattr(req, "headers", {}) or {}
        val = headers.get(name, "") or ""
    except Exception:
        return ""
    # Same placeholder-sanitization as RequestContextMiddleware —
    # LibreChat occasionally forwards ``{{LIBRECHAT_*}}`` verbatim
    # on the initial SSE handshake / empty-context substitution
    # pass, and we don't want those polluting tenant lookups.
    if val.startswith("{{") and val.endswith("}}"):
        return ""
    return val


def _state_fallback(attr: str) -> str:
    """Last-resort read from ``core.state.<attr>``.

    Only hit when neither our own contextvar (set by ``spawn_scope``
    or middleware) NOR the MCP SDK's per-request ``request_ctx`` has
    a value — e.g. code paths that run outside any request context
    at all, or very old call sites that haven't been migrated to
    ``spawn_scope(captured=...)`` yet. Under multi-user concurrency
    these globals are last-writer-wins and can be stale, so the
    MCP-request header path above should always be preferred.
    """
    try:
        from core import state as _state  # local import to dodge cycles
        return getattr(_state, attr, "") or ""
    except Exception:
        return ""


def get_user_id() -> str:
    """Return the current request's LibreChat user_id or ``""``.

    Resolution order:
      1. Our own ``current_user_id`` contextvar — set by
         ``spawn_scope(captured=...)`` for background / sub-agent
         workers that explicitly inherit their parent's identity.
      2. ``x-user-id`` header of the MCP SDK's current per-request
         Starlette Request — authoritative for any tool handler
         dispatched via the SSE transport, safe under concurrent
         multi-user load because each handler invocation has its
         own SDK-scoped ``request_ctx``.
      3. ``state.active_user_id`` module global — legacy last-resort
         fallback, stale under concurrency (see ``_state_fallback``).
    """
    return (
        current_user_id.get()
        or _header_from_mcp_request("x-user-id")
        or _state_fallback("active_user_id")
    )


def get_conversation_id() -> str:
    """Return the current request's LibreChat conversation_id or ``""``.

    Same three-tier resolution as ``get_user_id``: our contextvar,
    then MCP SDK request header, then state global.
    """
    return (
        current_conversation_id.get()
        or _header_from_mcp_request("x-conversation-id")
        or _state_fallback("active_conversation_id")
    )


def get_child_conversation_id() -> str:
    """Return the current spawned sub-agent's child conversation_id.

    Empty string when not inside a spawn_* subtree — callers should
    fall back to ``get_conversation_id()`` (the LibreChat parent cid)
    in that case.
    """
    return current_child_conversation_id.get() or ""


def get_lineage_parent_conv() -> str:
    """Return the parent conversation_id that spawned the current sub-agent.

    Empty string outside a spawn_* subtree. Forwarded via metadata so
    TCMM can compute ``lineage.parents[0]`` from the latest aid in
    that namespace when it ingests the child's first block.
    """
    return current_lineage_parent_conv.get() or ""


def get_effective_conversation_id() -> str:
    """Return the cid an outbound LLM call should be tagged with.

    Prefers the spawned child cid (so sub-agent reasoning is isolated
    in its own namespace); falls back to the parent cid (LibreChat
    turn) when not inside a spawn.
    """
    return get_child_conversation_id() or get_conversation_id()


# ── Shared spawn scope helper ─────────────────────────────────────────
#
# Every spawn_* tool (in tools/agents.py AND tools/tasks.py) wraps its
# LLM-calling body with ``async with spawn_scope(kind): ...`` — this
# mints a child conversation_id and declares the parent cid as lineage
# hint. The child cid flows into TCMM as its own namespace; the parent
# cid travels in metadata so TCMM can stamp the real lineage edge.
#
# Lives here (not in tools/agents.py) so the background-task tools in
# tools/tasks.py don't need to reach across module boundaries. They
# use asyncio.create_task for fire-and-forget execution; contextvars
# set by spawn_scope BEFORE create_task's context snapshot are
# preserved for the detached _run coroutine — so background workers
# land under a stable child cid + lineage hint without extra plumbing.
import uuid as _uuid
from contextlib import asynccontextmanager as _asynccontextmanager


def capture_context() -> dict:
    """Snapshot the identity at the call site.

    Reads from (in priority order):

    1. Our per-request contextvars (set by RequestContextMiddleware).
    2. ``core.state.active_conversation_id`` / ``active_user_id`` —
       module-level globals that the middleware also updates.

    The fallback exists because FastMCP dispatches tool handlers
    in a DIFFERENT asyncio task than the ASGI POST handler that
    carried the HTTP headers. The MCP SDK's SSE server queues
    incoming messages into an internal stream and a separate loop
    pops them for dispatch — contextvars from the HTTP handler's
    context don't propagate across that boundary. The module
    globals ARE visible across tasks (they're just plain module
    attributes) so they catch the case where contextvars come
    back empty inside a tool handler.

    Handlers should call ``capture_context()`` synchronously at
    entry and pass the dict into ``spawn_scope(captured=...)`` —
    especially when the work is dispatched via
    ``asyncio.create_task`` (background workers in tasks.py).
    """
    # get_user_id / get_conversation_id already walk the full
    # resolution chain (our contextvar → MCP SDK request headers →
    # state global). Calling them here means background workers
    # get the SAME authoritative identity source that live tool
    # handlers see — no more relying on stale state globals as
    # the primary snapshot channel.
    return {
        "conversation_id": get_conversation_id(),
        "user_id": get_user_id(),
    }


@_asynccontextmanager
async def spawn_scope(kind: str, captured: dict | None = None):
    """Mint a child conversation_id + stamp lineage for a spawn_* tool.

    ``kind`` is a short label (agentic / coord / agent / parN /
    pipe-<id> / reviewloop / bgtask / parallel-bg) that makes it easy
    to eyeball the namespace and know what generated it without
    cracking open the archive.

    ``captured`` — optional dict from ``capture_context()`` with
    ``conversation_id`` and ``user_id`` from the handler's ASGI
    request. When provided, wins over contextvar lookups (which
    can be empty in detached ``asyncio.create_task`` coroutines
    because of the middleware reset race described in
    ``capture_context``'s docstring).
    """
    snap = captured or {}
    parent_cid = (snap.get("conversation_id") or get_conversation_id() or "")
    captured_uid = (snap.get("user_id") or get_user_id() or "")
    # Child cid MUST fit inside TCMM's 24-char namespace cap, otherwise
    # ``pool._normalize_id`` chops the discriminating uuid off the end
    # and every parallel worker collapses into one shared namespace —
    # we saw this on the 22 Apr Mexican-gov run where 4 workers all
    # landed in ``sub-9aa425aa-2d81-43-pbg``. Exact layout:
    #     sub-<parent[:7]>-<kind[:3]>-<uuid[:8]>
    #     4    +  7      + 1 +  3  + 1 +  8  = 24 chars
    # Parent prefix still recognisable (first 7 chars of parent cid),
    # kind truncated but readable (pbg / bgt / smr / agt / ...), and
    # the uuid survives so workers are distinct.
    parent_tag = (parent_cid[:7] if parent_cid else "noprnt7")
    kind_tag = (kind or "spn")[:3]
    child_cid = f"sub-{parent_tag}-{kind_tag}-{_uuid.uuid4().hex[:8]}"

    child_tok = current_child_conversation_id.set(child_cid)
    lineage_tok = current_lineage_parent_conv.set(parent_cid)
    # Also restore the user_id contextvar inside the scope so
    # ``call_llm`` picks up the right tenant on outgoing metadata
    # even in a background task where the middleware-set var
    # was already reset before we got here.
    user_tok = current_user_id.set(captured_uid)
    # And the parent conversation_id too so utils/tcmm.py's recall
    # and anything else keyed on get_conversation_id() inside the
    # worker still see the parent.
    conv_tok = current_conversation_id.set(parent_cid)
    try:
        yield child_cid
    finally:
        current_child_conversation_id.reset(child_tok)
        current_lineage_parent_conv.reset(lineage_tok)
        current_user_id.reset(user_tok)
        current_conversation_id.reset(conv_tok)
