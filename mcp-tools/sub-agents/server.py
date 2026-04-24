"""Veilguard Sub-Agent MCP Tool Server.

Multi-backend agent orchestration: Gemini + Claude.
53 tools across 14 modules: agents, tasks, daemons, teams, pipelines, playbooks, etc.

Start: python server.py --sse --port 8809
       python server.py --sse --port 8809 --local   (all tools execute locally, no client daemon needed)
"""

import logging
import os
import sys

LOCAL_MODE = "--local" in sys.argv

from mcp.server.fastmcp import FastMCP

# Configure logging
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("veilguard")

# ── MCP Server Instance ──────────────────────────────────────────────────────

mcp = FastMCP(
    "sub-agents",
    instructions=(
        "Sub-agent orchestration tools. Spawn specialized AI workers to handle "
        "subtasks in parallel, with coordinator/worker patterns, review loops, "
        "daemon mode, and multi-backend support (Gemini + Claude)."
    ),
)

# ── Register All Tools ───────────────────────────────────────────────────────

from tools.register import register_all
register_all(mcp)

logger.info("All tools registered")

# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Restore persisted state
    from storage.session import load_session
    load_session()

    if "--sse" in sys.argv:
        port = 8809
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])

        import uvicorn
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route
        from starlette.responses import JSONResponse, FileResponse, PlainTextResponse
        from starlette.middleware.cors import CORSMiddleware

        sse = SseServerTransport("/messages/")

        # Track the active conversation ID from LibreChat's MCP headers.
        # Kept as a legacy fallback for modules that haven't been ported
        # to core.request_ctx yet — the contextvar set by the middleware
        # below is the source of truth for new code.
        from core import state as _state
        _state.active_conversation_id = ""

        from core.request_ctx import (
            current_user_id as _ctx_user_id,
            current_conversation_id as _ctx_conv_id,
        )

        # Middleware populates per-request contextvars for every HTTP
        # request (was scoped to /messages/ only — that missed the REST
        # API endpoints entirely, which is why /api/client/* had no
        # notion of "who's asking").
        from starlette.middleware import Middleware
        from starlette.types import ASGIApp, Receive, Scope, Send

        class RequestContextMiddleware:
            def __init__(self, app: ASGIApp):
                self.app = app
            async def __call__(self, scope: Scope, receive: Receive, send: Send):
                if scope["type"] != "http":
                    await self.app(scope, receive, send)
                    return
                user_id = ""
                conv_id = ""
                for key, val in scope.get("headers", []):
                    if key == b"x-user-id":
                        user_id = val.decode("utf-8", errors="replace")
                    elif key == b"x-conversation-id":
                        conv_id = val.decode("utf-8", errors="replace")
                # Defensive: LibreChat's MCP SSE transport has been
                # observed to forward header placeholders verbatim
                # (e.g. ``{{LIBRECHAT_BODY_CONVERSATIONID}}``) when
                # its processMCPEnv substitution pass runs with an
                # empty requestBody — which happens on the initial
                # SSE handshake and can leak into message POSTs on
                # long-lived sessions. If we see an unsubstituted
                # ``{{...}}`` token, treat it as absent rather than
                # letting it corrupt TCMM namespaces + user_id
                # tenancy. Log once so we notice it's happening.
                if conv_id.startswith("{{") and conv_id.endswith("}}"):
                    logger.warning(
                        f"[middleware] unsubstituted x-conversation-id "
                        f"placeholder: {conv_id!r} — treating as empty"
                    )
                    conv_id = ""
                if user_id.startswith("{{") and user_id.endswith("}}"):
                    logger.warning(
                        f"[middleware] unsubstituted x-user-id "
                        f"placeholder: {user_id!r} — treating as empty"
                    )
                    user_id = ""
                user_tok = _ctx_user_id.set(user_id)
                conv_tok = _ctx_conv_id.set(conv_id)
                # Module-level globals: FastMCP dispatches tool
                # handlers in a separate asyncio task whose context
                # is NOT a copy of this ASGI request handler's
                # context — so our contextvars don't propagate to
                # the handlers at all. (The MCP SDK's SSE server
                # queues incoming messages into an internal stream
                # and a separate loop pops them for dispatch.) The
                # globals are the only reliable channel within a
                # single user's session. Safe under the current
                # single-user testing regime; revisit for
                # multi-tenancy when we run concurrent users on the
                # same sub-agents instance.
                if conv_id:
                    _state.active_conversation_id = conv_id
                if user_id:
                    _state.active_user_id = user_id
                try:
                    await self.app(scope, receive, send)
                finally:
                    _ctx_user_id.reset(user_tok)
                    _ctx_conv_id.reset(conv_tok)

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as (read, write):
                await mcp._mcp_server.run(
                    read, write, mcp._mcp_server.create_initialization_options()
                )

        async def handle_trigger(request):
            """Webhook trigger endpoint — fires a daemon or one-shot agent."""
            from core import state
            name = request.path_params.get("name", "")
            daemon_id = name.lower().replace(" ", "-")
            body = {}
            try:
                body = await request.json()
            except Exception:
                pass

            if daemon_id in state.daemons:
                state.daemons[daemon_id].next_run = 0
                return JSONResponse({"status": "triggered", "daemon": daemon_id})
            else:
                from tools.agents import register as _  # ensure loaded
                from core.llm import call_llm
                from config import ROLES
                role = body.get("role", "researcher")
                task = body.get("task", body.get("context", f"Webhook trigger: {name}"))
                sp = ROLES.get(role, {}).get("system", "You are helpful.")
                result = await call_llm(sp, task, role=role)
                return JSONResponse({"status": "executed", "result": result[:500]})

        # Import API endpoints
        from api.endpoints import api_stats, api_tasks, api_scratchpad, api_daemons, api_teams

        # ── Client Daemon WebSocket ─────────────────────────────────────
        from starlette.websockets import WebSocket, WebSocketDisconnect
        from core.client_bridge import (
            ClientBridge,
            set_bridge,
            get_bridge,
            all_bridges,
            register_user_token,
            validate_token,
        )
        import json as _json

        # Legacy shared CLIENT_TOKEN kept for backward-compat during
        # rollout. Per-user tokens (core/client_bridge.register_user_token)
        # supersede it. Can be removed once every daemon reinstalls.
        CLIENT_TOKEN = os.environ.get("CLIENT_TOKEN", "")

        def _request_user_id(request) -> str:
            """Extract x-user-id from a request (falls back to contextvar)."""
            uid = request.headers.get("x-user-id", "")
            if not uid:
                from core.request_ctx import get_user_id
                uid = get_user_id()
            return uid

        def _public_ws_url(request) -> str:
            """Build the daemon-facing WebSocket URL.

            Priority:
              1. ``VEILGUARD_PUBLIC_WS_URL`` env var — set this on prod
                 so the QR always shows the public domain regardless of
                 what proxy/host rewrote the Host header.
              2. ``X-Forwarded-Host`` header (added by cooperative
                 proxies).
              3. Fallback to the literal ``Host`` header.

            Previously used only (3), which on the LibreChat→sub-agents
            proxy hop echoed ``host.docker.internal:8809`` — unreachable
            from the user's laptop.  The daemon then tries to connect to
            a non-existent hostname and silently fails.
            """
            override = os.environ.get("VEILGUARD_PUBLIC_WS_URL", "").strip()
            if override:
                return override
            xfh = request.headers.get("x-forwarded-host", "").strip()
            host = xfh or request.headers.get("host", "localhost:8809")
            scheme = "wss" if request.url.scheme == "https" or xfh else "ws"
            return f"{scheme}://{host}/ws/client"

        def _public_download_url(request) -> str:
            override = os.environ.get("VEILGUARD_PUBLIC_DOWNLOAD_URL", "").strip()
            if override:
                return override
            xfh = request.headers.get("x-forwarded-host", "").strip()
            host = xfh or request.headers.get("host", "localhost:8809")
            scheme = "https" if request.url.scheme == "https" or xfh else "http"
            return f"{scheme}://{host}/download/VeilguardSetup.exe"

        async def api_client_install(request):
            """Return the daemon installer URL for the calling user."""
            return JSONResponse({
                "download_url": _public_download_url(request),
                "ws_url": _public_ws_url(request),
            })

        async def api_client_register(request):
            """Mint (or return) a per-user install token for the calling user.

            Called by the LibreChat Cowork panel once the user is
            authenticated. LibreChat forwards the user's id as
            ``x-user-id`` so we know whose token to mint. The QR / paste
            payload rendered in the panel embeds the returned fields.
            """
            user_id = _request_user_id(request)
            if not user_id:
                return JSONResponse(
                    {"error": "Missing x-user-id header"}, status_code=400,
                )
            regenerate = request.query_params.get("regenerate", "").lower() in (
                "1", "true", "yes",
            )
            token = register_user_token(user_id, regenerate=regenerate)
            return JSONResponse({
                "user_id": user_id,
                "token": token,
                "ws_url": _public_ws_url(request),
            })

        async def ws_client(websocket: WebSocket):
            """WebSocket endpoint for client daemon connection.

            Auth params must include ``user_id`` and ``token``. We look
            up the per-user token and compare in constant time — daemons
            can only connect as the LibreChat user they were installed
            for.
            """
            await websocket.accept()
            bridge = ClientBridge()
            registered_user_id = ""

            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=10)
                auth = _json.loads(raw)
                params = auth.get("params", {}) or {}
                user_id = params.get("user_id", "")
                token = params.get("token", "")
                client_id = params.get("client_id", "unknown")

                if not user_id:
                    await websocket.send_text(_json.dumps({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32001,
                            "message": (
                                "Missing user_id — re-install the Veilguard "
                                "client from LibreChat to get a per-user token."
                            ),
                        },
                    }))
                    await websocket.close()
                    return

                if not validate_token(user_id, token):
                    logger.warning(
                        f"[WS] Auth rejected for user={user_id} client={client_id}"
                    )
                    await websocket.send_text(_json.dumps({
                        "jsonrpc": "2.0",
                        "error": {"code": -32001, "message": "Invalid token"},
                    }))
                    await websocket.close()
                    return

                # Replace any existing daemon for this user — we allow
                # only one connected daemon per LibreChat user.
                existing = get_bridge(user_id)
                if existing is not None and existing is not bridge:
                    try:
                        existing.on_disconnect()
                    except Exception:
                        pass
                    set_bridge(None, user_id=user_id)

                bridge.ws = websocket
                bridge.connected = True
                bridge.client_id = client_id
                bridge.user_id = user_id
                # Host identity from the daemon's auth payload —
                # used by execute_remote to stamp a one-shot host
                # hint on the first run_command so Claude emits the
                # right shell syntax. Missing fields are harmless
                # (older daemons pre-dating this change just skip
                # the hint and Claude falls back to trial-and-error).
                bridge.platform = params.get("platform", "") or ""
                bridge.os_name = params.get("os_name", "") or ""
                bridge.os_release = params.get("os_release", "") or ""
                bridge.shell = params.get("shell", "") or ""
                set_bridge(bridge, user_id=user_id)
                registered_user_id = user_id

                await websocket.send_text(_json.dumps({
                    "jsonrpc": "2.0",
                    "result": {
                        "status": "authenticated",
                        "client_id": bridge.client_id,
                        "user_id": user_id,
                    },
                }))
                logger.info(
                    f"[WS] Client daemon connected: user={user_id} client={client_id} "
                    f"platform={bridge.platform or '?'} shell={bridge.shell or '?'}"
                )

            except Exception as e:
                logger.error(f"[WS] Auth failed: {e}")
                try:
                    await websocket.close()
                except Exception:
                    pass
                return

            # Message loop
            try:
                while True:
                    raw = await websocket.receive_text()
                    data = _json.loads(raw)

                    if data.get("method") == "ping":
                        bridge.last_ping = __import__("time").time()
                        await websocket.send_text(_json.dumps(
                            {"jsonrpc": "2.0", "result": "pong"}
                        ))
                    else:
                        bridge.on_message(data)

            except WebSocketDisconnect:
                bridge.on_disconnect()
                if registered_user_id and get_bridge(registered_user_id) is bridge:
                    set_bridge(None, user_id=registered_user_id)
            except Exception as e:
                logger.error(f"[WS] Error: {e}")
                bridge.on_disconnect()
                if registered_user_id and get_bridge(registered_user_id) is bridge:
                    set_bridge(None, user_id=registered_user_id)

        async def api_client_status(request):
            """Client daemon connection status for the calling user."""
            user_id = _request_user_id(request)
            if not user_id:
                return JSONResponse(
                    {"connected": False, "error": "Missing x-user-id header"},
                    status_code=400,
                )
            bridge = get_bridge(user_id)
            if bridge:
                return JSONResponse(bridge.status())
            return JSONResponse({
                "connected": False,
                "client_id": "",
                "user_id": user_id,
                "pending_requests": 0,
            })

        async def api_client_folders(request):
            """Get or set working folders on the calling user's daemon."""
            user_id = _request_user_id(request)
            if not user_id:
                return JSONResponse(
                    {"error": "Missing x-user-id header"}, status_code=400,
                )
            bridge = get_bridge(user_id)
            if not bridge or not bridge.connected:
                return JSONResponse({"error": "No client connected"}, status_code=503)

            if request.method == "POST":
                body = await request.json()
                folders = body.get("folders", [])
                result = await bridge.send_command("set_working_folders", {"folders": folders})
                return JSONResponse({"status": "ok", "result": result})
            else:
                result = await bridge.send_command("get_working_folders", {})
                if result and isinstance(result, dict):
                    return JSONResponse(result)
                return JSONResponse({"folders": []})

        async def api_client_browse(request):
            """Browse directories on the calling user's daemon."""
            user_id = _request_user_id(request)
            if not user_id:
                return JSONResponse(
                    {"error": "Missing x-user-id header"}, status_code=400,
                )
            bridge = get_bridge(user_id)
            if not bridge or not bridge.connected:
                return JSONResponse({"error": "No client connected"}, status_code=503)

            path = request.query_params.get("path", "")
            result = await bridge.send_command("list_directory", {"path": path})
            if result and isinstance(result, dict):
                return JSONResponse(result)
            return JSONResponse({"directories": []})

        # ── Client installer download ───────────────────────────────────
        # Serves VeilguardSetup.exe (and any other binaries we drop into
        # $VEILGUARD_DOWNLOADS_DIR) so users can grab the installer from
        # https://veilguard.phishield.com/api/sub-agents/download/<filename>.
        # Caddy strips /api/sub-agents/ so the backend sees /download/*.
        import pathlib as _pathlib
        _DEFAULT_DOWNLOADS = _pathlib.Path(
            os.environ.get(
                "VEILGUARD_DOWNLOADS_DIR",
                "/home/rudol/veilguard/downloads",
            )
        ).resolve()

        async def api_download(request):
            filename = request.path_params.get("filename", "")
            # Safety: forbid path traversal / absolute paths / empty names.
            if not filename or "/" in filename or "\\" in filename or filename.startswith("."):
                return PlainTextResponse("Not found", status_code=404)
            target = (_DEFAULT_DOWNLOADS / filename).resolve()
            # Ensure resolved path is still inside the downloads dir.
            try:
                target.relative_to(_DEFAULT_DOWNLOADS)
            except ValueError:
                return PlainTextResponse("Not found", status_code=404)
            if not target.is_file():
                return PlainTextResponse("Not found", status_code=404)
            # Content-Disposition: attachment with the original filename so
            # browsers save the .exe rather than attempting to render it.
            return FileResponse(
                str(target),
                filename=filename,
                media_type="application/octet-stream",
            )

        # ── Client version manifest ─────────────────────────────────────
        # The daemon polls this every 30min. Shape:
        #   {"version": "0.2.0",
        #    "url": "/download/VeilguardSetup.exe",
        #    "min_required": "0.1.0",
        #    "changelog": "..."}
        # The manifest file lives next to the installer in the downloads
        # dir so a publish_release.py script can atomically swap both in
        # one scp. If version.json is missing we return 503 (no update
        # available) — clients treat that as "stay put" and retry later.
        async def api_client_latest(request):
            manifest_file = _DEFAULT_DOWNLOADS / "version.json"
            if not manifest_file.is_file():
                return JSONResponse(
                    {"error": "No release manifest published"},
                    status_code=503,
                )
            try:
                import json as _json
                manifest = _json.loads(manifest_file.read_text(encoding="utf-8"))
            except Exception as e:
                return JSONResponse({"error": f"Manifest parse error: {e}"}, status_code=500)
            # Normalize relative URLs so clients don't need to know the
            # host. Keep absolute URLs untouched (useful if someone
            # points at a CDN).
            url = manifest.get("url", "")
            if url and not url.startswith(("http://", "https://", "/")):
                manifest["url"] = f"/download/{url}"
            return JSONResponse(manifest)

        import asyncio
        from starlette.routing import WebSocketRoute

        routes = [
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
            Route("/trigger/{name}", endpoint=handle_trigger, methods=["POST"]),
            Route("/api/stats", endpoint=api_stats),
            Route("/api/tasks", endpoint=api_tasks),
            Route("/api/scratchpad", endpoint=api_scratchpad),
            Route("/api/daemons", endpoint=api_daemons),
            Route("/api/teams", endpoint=api_teams),
            Route("/api/client/status", endpoint=api_client_status),
            Route("/api/client/install", endpoint=api_client_install),
            Route("/api/client/register", endpoint=api_client_register),
            Route("/api/client/folders", endpoint=api_client_folders, methods=["GET", "POST"]),
            Route("/api/client/browse", endpoint=api_client_browse),
            Route("/api/client/latest", endpoint=api_client_latest),
            Route("/download/{filename}", endpoint=api_download),
            WebSocketRoute("/ws/client", endpoint=ws_client),
        ]

        app = Starlette(routes=routes)

        # Populate per-request (user_id, conversation_id) contextvars.
        app = RequestContextMiddleware(app)

        app = CORSMiddleware(
            app, allow_origins=["http://localhost:3080", "http://127.0.0.1:3080"],
            allow_methods=["GET", "POST"], allow_headers=["*"],
        )

        mode = "LOCAL" if LOCAL_MODE else "REMOTE (client daemon)"
        logger.info(f"Starting sub-agents MCP server on http://0.0.0.0:{port}/sse [{mode}]")
        logger.info(f"  REST API: http://0.0.0.0:{port}/api/stats|tasks|scratchpad|daemons|teams")
        logger.info(f"  Client WS: ws://0.0.0.0:{port}/ws/client")
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        mcp.run(transport="stdio")
