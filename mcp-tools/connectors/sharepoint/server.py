"""FastMCP server exposing SharePoint as MCP tools.

LibreChat invokes these tools when the LLM calls
``sharepoint.search`` / ``sharepoint.read`` etc. The server resolves
the per-user OAuth token via the connector framework's
:class:`HttpCredentialResolver` against LibreChat's internal token
endpoint, then delegates to :class:`SharePointConnector`.

`read_file` wraps its response in the `_veilguard` envelope so the
TCMM ingest path (in `veilguard_adapter.py`) can extract acl /
tool_ref / etag / title into the archive entry's `extras_json`. The
PII proxy strips the envelope on the LLM-egress leg, so the LLM
sees only the inner `content`.

Run with::

    cd mcp-tools/connectors/sharepoint && python server.py
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import sys

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_CONNECTORS_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_CONNECTORS_DIR))
sys.path.insert(0, str(_CONNECTORS_DIR.parent / "_shared"))


from _base import (  # noqa: E402
    HttpCredentialResolver,
    ReauthenticationRequiredError,
    StaticCredentialResolver,
    UserContext,
    default_registry,
    wrap_with_provenance,
)
from connector import SharePointConnector  # noqa: E402  (local import)
from graph import GraphError  # noqa: E402

# request_ctx lives in mcp-tools/sub-agents/core/. We add the path
# rather than copy the helper because the get_user_id contract may
# evolve and we want SharePoint to track it without a sync.
_REQUEST_CTX_DIR = _CONNECTORS_DIR.parent / "sub-agents" / "core"
if _REQUEST_CTX_DIR.is_dir() and str(_REQUEST_CTX_DIR) not in sys.path:
    sys.path.insert(0, str(_REQUEST_CTX_DIR))

try:
    from request_ctx import get_user_id, get_conversation_id  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — degrades to env-based identity
    def get_user_id() -> str:
        return os.environ.get("VEILGUARD_DEFAULT_USER_ID", "")

    def get_conversation_id() -> str:
        return os.environ.get("VEILGUARD_DEFAULT_CONVERSATION_ID", "")


from mcp.server.fastmcp import FastMCP  # noqa: E402


logger = logging.getLogger("sharepoint-mcp")


# ─── credential resolver ──────────────────────────────────────────────


def _build_credential_resolver():
    """Choose the credential backend based on env.

    Default: HttpCredentialResolver pointing at LibreChat. For local
    development without a LibreChat-side token endpoint, set
    ``SHAREPOINT_STATIC_ACCESS_TOKEN=<token>`` and
    ``SHAREPOINT_STATIC_USER_ID=<user_id>`` — the resolver returns
    that token for the configured user.
    """
    static_token = os.environ.get("SHAREPOINT_STATIC_ACCESS_TOKEN")
    if static_token:
        from _base import OAuthToken
        user_id = os.environ.get("SHAREPOINT_STATIC_USER_ID", "default")
        resolver = StaticCredentialResolver()
        resolver.set(user_id, "sharepoint", OAuthToken(static_token))
        logger.warning(
            "[sharepoint] using SHAREPOINT_STATIC_ACCESS_TOKEN — "
            "do NOT use in production"
        )
        return resolver

    base = os.environ.get(
        "VEILGUARD_LIBRECHAT_URL", "http://host.docker.internal:3080"
    )
    return HttpCredentialResolver(base)


# ─── connector + server ───────────────────────────────────────────────


_connector = SharePointConnector(credentials=_build_credential_resolver())

# Register on the process-wide default registry so the recall layer
# can fan out hint() to this connector when imported as a library.
# When run as an MCP server, registration is benign (no recall layer
# is calling our hints — the LLM will reach this server via tool
# calls instead).
default_registry.register(_connector)


mcp = FastMCP(
    "sharepoint",
    instructions=(
        "Search and read documents from the user's SharePoint. "
        "Call `search_sharepoint` with a query to find candidate "
        "documents; the result lists item_id + drive_id pairs. "
        "Then call `read_file` with those IDs to fetch document "
        "content. Always cite the document title from the search "
        "result when answering a user."
    ),
)


# ─── tools ────────────────────────────────────────────────────────────


@mcp.tool()
async def search_sharepoint(query: str, top: int = 10) -> str:
    """Search SharePoint for documents matching ``query``.

    Returns a JSON array of {item_id, drive_id, title, score}. The
    LLM uses these to decide which document(s) to call ``read_file``
    on. Returns an empty array if no matches.
    """
    user_id = get_user_id()
    user_ctx = UserContext(tenant_id="", user_id=user_id, principals=[])
    try:
        # We use hint() rather than search() because hint() returns
        # Snippet objects with title/score; search() returns refs only.
        snippets = await _connector.hint(query, user_ctx, deadline_ms=10_000, top_k=top)
    except ReauthenticationRequiredError as e:
        return json.dumps({
            "error": "reauthentication_required",
            "connector": "sharepoint",
            "reason": e.reason,
            "user_action": "Reconnect SharePoint in your Veilguard settings.",
        })
    except GraphError as e:
        return json.dumps({
            "error": "graph_error",
            "status": e.status,
            "message": str(e),
        })

    return json.dumps([
        {
            "item_id": s.ref.args.get("item_id"),
            "drive_id": s.ref.args.get("drive_id"),
            "title": s.title,
            "score": s.score,
        }
        for s in snippets
    ])


@mcp.tool()
async def read_file(item_id: str, drive_id: str) -> str:
    """Read a SharePoint document's text content.

    Returns the document text, with file metadata (title, etag, ACL)
    riding along in a `_veilguard` envelope that the TCMM ingest
    path consumes. The LLM sees only the inner text content (the PII
    proxy strips the envelope on the egress leg).
    """
    user_id = get_user_id()
    user_ctx = UserContext(tenant_id="", user_id=user_id, principals=[])

    from _base import Ref
    ref = Ref(
        connector="sharepoint",
        tool="read",
        args={"item_id": item_id, "drive_id": drive_id},
    )

    try:
        content = await _connector.read(ref, user_ctx)
    except ReauthenticationRequiredError as e:
        return json.dumps({
            "error": "reauthentication_required",
            "connector": "sharepoint",
            "reason": e.reason,
            "user_action": "Reconnect SharePoint in your Veilguard settings.",
        })
    except ValueError as e:
        return json.dumps({"error": "bad_arguments", "message": str(e)})
    except GraphError as e:
        return json.dumps({
            "error": "graph_error",
            "status": e.status,
            "message": str(e),
        })

    return wrap_with_provenance(
        content.text,
        connector="sharepoint",
        tool_ref={
            "connector": "sharepoint",
            "tool": "read",
            "args": {"item_id": item_id, "drive_id": drive_id},
        },
        acl=content.acl,
        etag=content.etag,
        title=content.title,
    )


@mcp.tool()
async def list_folder(drive_id: str, item_id: str = "") -> str:
    """List items in a SharePoint folder.

    ``item_id`` empty means the drive root. Returns a JSON array of
    {item_id, drive_id, name, is_folder}. The LLM uses this to
    navigate folders before deciding which file to read.
    """
    user_id = get_user_id()
    user_ctx = UserContext(tenant_id="", user_id=user_id, principals=[])

    try:
        path = drive_id if not item_id else f"{drive_id}/{item_id}"
        refs = await _connector.list(user_ctx, path=path)
    except ReauthenticationRequiredError as e:
        return json.dumps({
            "error": "reauthentication_required",
            "connector": "sharepoint",
            "reason": e.reason,
        })
    except GraphError as e:
        return json.dumps({
            "error": "graph_error",
            "status": e.status,
            "message": str(e),
        })

    return json.dumps([
        {
            "item_id": r.args.get("item_id"),
            "drive_id": r.args.get("drive_id"),
        }
        for r in refs
    ])


# ─── entrypoint ───────────────────────────────────────────────────────


def _run_sse(port: int) -> None:
    """Run as a Starlette/SSE app so LibreChat can connect via its
    standard OAuth-aware MCP transport.

    When LibreChat speaks SSE to this server, it forwards
    ``X-User-ID`` and ``X-Conversation-ID`` headers (from the
    librechat.yaml ``headers`` block). We populate contextvars from
    those headers via :class:`_RequestContextMiddleware` so
    ``get_user_id()`` works inside tool handlers — matches the
    convention used by sub-agents and forge.

    OAuth itself is handled by LibreChat at
    ``/api/mcp/sharepoint/oauth/initiate`` and ``/callback`` (config
    in librechat.yaml). The user clicks "Connect SharePoint",
    completes the MS login flow, and the access token lands in
    LibreChat's token store. This server fetches the token via the
    internal credential endpoint (HttpCredentialResolver above) when
    a tool is invoked or when the connector framework runs hint
    fan-out during recall.
    """
    import uvicorn
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route
    from starlette.types import ASGIApp, Receive, Scope, Send

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as (in_stream, out_stream):
            await mcp._mcp_server.run(
                in_stream, out_stream, mcp._mcp_server.create_initialization_options()
            )

    class _RequestContextMiddleware:
        """Extract ``X-User-ID`` / ``X-Conversation-ID`` headers and set
        the contextvars that ``request_ctx.get_user_id()`` reads.

        Mirrors the pattern in sub-agents/server.py — incoming SSE
        connections carry the LibreChat-templated headers, but
        FastMCP doesn't automatically surface them to tool handlers.
        We do that here so the existing get_user_id helper is the
        single source of truth for tools.
        """

        def __init__(self, app: ASGIApp):
            self.app = app

        async def __call__(self, scope: Scope, receive: Receive, send: Send):
            if scope["type"] != "http":
                await self.app(scope, receive, send)
                return
            try:
                from request_ctx import (  # type: ignore[import-not-found]
                    set_user_id, set_conversation_id,
                )
            except ImportError:
                await self.app(scope, receive, send)
                return
            user_id = ""
            conv_id = ""
            for key, val in scope.get("headers", []):
                if key == b"x-user-id":
                    user_id = val.decode("utf-8", errors="replace")
                elif key == b"x-conversation-id":
                    conv_id = val.decode("utf-8", errors="replace")
            # Defensive: LibreChat has been observed to forward header
            # placeholders verbatim (e.g. ``{{LIBRECHAT_USER_ID}}``)
            # when the conversation hasn't authenticated yet — treat
            # those as empty rather than as a literal string.
            if user_id.startswith("{{") and user_id.endswith("}}"):
                user_id = ""
            if conv_id.startswith("{{") and conv_id.endswith("}}"):
                conv_id = ""
            tok_u = set_user_id(user_id) if user_id else None
            tok_c = set_conversation_id(conv_id) if conv_id else None
            try:
                await self.app(scope, receive, send)
            finally:
                if tok_u is not None:
                    from request_ctx import reset_user_id  # type: ignore[import-not-found]
                    reset_user_id(tok_u)
                if tok_c is not None:
                    from request_ctx import reset_conversation_id  # type: ignore[import-not-found]
                    reset_conversation_id(tok_c)

    routes = [
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ]
    app = Starlette(routes=routes)
    app = _RequestContextMiddleware(app)

    logger.info(
        f"Starting SharePoint MCP server on sse://0.0.0.0:{port}/sse "
        f"(librechat.yaml `url` should match)"
    )
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [sharepoint-mcp] %(message)s",
    )

    transport = os.environ.get("SHAREPOINT_TRANSPORT", "stdio").lower()
    if transport == "sse":
        port = int(os.environ.get("SHAREPOINT_PORT", "8812"))
        _run_sse(port)
    else:
        # stdio for local dev / direct LibreChat stdio integration
        # (no OAuth machinery — only viable with SHAREPOINT_STATIC_ACCESS_TOKEN
        # in env).
        mcp.run()
