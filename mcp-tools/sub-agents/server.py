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
        from starlette.responses import JSONResponse
        from starlette.middleware.cors import CORSMiddleware

        sse = SseServerTransport("/messages/")

        # Track the active conversation ID from LibreChat's MCP headers
        from core import state as _state
        _state.active_conversation_id = ""

        # Middleware to extract x-conversation-id from MCP /messages/ requests
        from starlette.middleware import Middleware
        from starlette.types import ASGIApp, Receive, Scope, Send

        class ConversationIdMiddleware:
            def __init__(self, app: ASGIApp):
                self.app = app
            async def __call__(self, scope: Scope, receive: Receive, send: Send):
                if scope["type"] == "http" and scope.get("path", "").startswith("/messages/"):
                    for key, val in scope.get("headers", []):
                        if key == b"x-conversation-id":
                            _state.active_conversation_id = val.decode("utf-8", errors="replace")
                            break
                await self.app(scope, receive, send)

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
        from core.client_bridge import ClientBridge, set_bridge, get_bridge
        import json as _json

        CLIENT_TOKEN = os.environ.get("CLIENT_TOKEN", "")
        import secrets as _secrets
        if not CLIENT_TOKEN:
            CLIENT_TOKEN = _secrets.token_hex(16)
            logger.info(f"Generated CLIENT_TOKEN: {CLIENT_TOKEN}")

        async def api_client_install(request):
            """Generate one-liner install command for the client daemon."""
            host = request.headers.get("host", "localhost:8809")
            scheme = "wss" if request.url.scheme == "https" else "ws"
            ws_url = f"{scheme}://{host}/ws/client"
            pip_cmd = f"pip install veilguard-client && veilguard --setup {ws_url} --token {CLIENT_TOKEN}"
            return JSONResponse({
                "install_command": pip_cmd,
                "ws_url": ws_url,
                "token": CLIENT_TOKEN,
            })

        async def ws_client(websocket: WebSocket):
            """WebSocket endpoint for client daemon connection."""
            await websocket.accept()
            bridge = ClientBridge()

            # Wait for auth message
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=10)
                auth = _json.loads(raw)
                token = auth.get("params", {}).get("token", "")

                if CLIENT_TOKEN and token != CLIENT_TOKEN:
                    await websocket.send_text(_json.dumps({
                        "jsonrpc": "2.0", "error": {"code": -32001, "message": "Invalid token"}
                    }))
                    await websocket.close()
                    return

                bridge.ws = websocket
                bridge.connected = True
                bridge.client_id = auth.get("params", {}).get("client_id", "unknown")
                set_bridge(bridge)

                await websocket.send_text(_json.dumps({
                    "jsonrpc": "2.0", "result": {"status": "authenticated", "client_id": bridge.client_id}
                }))
                logger.info(f"[WS] Client daemon connected: {bridge.client_id}")

            except Exception as e:
                logger.error(f"[WS] Auth failed: {e}")
                await websocket.close()
                return

            # Message loop
            try:
                while True:
                    raw = await websocket.receive_text()
                    data = _json.loads(raw)

                    if data.get("method") == "ping":
                        bridge.last_ping = __import__("time").time()
                        await websocket.send_text(_json.dumps({"jsonrpc": "2.0", "result": "pong"}))
                    else:
                        bridge.on_message(data)

            except WebSocketDisconnect:
                bridge.on_disconnect()
                if get_bridge() is bridge:
                    set_bridge(None)
            except Exception as e:
                logger.error(f"[WS] Error: {e}")
                bridge.on_disconnect()
                if get_bridge() is bridge:
                    set_bridge(None)

        async def api_client_status(request):
            """Client daemon connection status."""
            bridge = get_bridge()
            if bridge:
                return JSONResponse(bridge.status())
            return JSONResponse({"connected": False, "client_id": "", "pending_requests": 0})

        async def api_client_folders(request):
            """Get or set working folders on the client daemon."""
            bridge = get_bridge()
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
            """Browse directories on the client machine for the folder picker."""
            bridge = get_bridge()
            if not bridge or not bridge.connected:
                return JSONResponse({"error": "No client connected"}, status_code=503)

            path = request.query_params.get("path", "")
            result = await bridge.send_command("list_directory", {"path": path})
            if result and isinstance(result, dict):
                return JSONResponse(result)
            return JSONResponse({"directories": []})

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
            Route("/api/client/folders", endpoint=api_client_folders, methods=["GET", "POST"]),
            Route("/api/client/browse", endpoint=api_client_browse),
            WebSocketRoute("/ws/client", endpoint=ws_client),
        ]

        app = Starlette(routes=routes)

        # Wrap with conversation ID extraction middleware
        app = ConversationIdMiddleware(app)

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
