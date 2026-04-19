"""Cloud-side WebSocket bridge to client daemon.

Routes tool execution requests to the connected client daemon over
JSON-RPC 2.0 / WebSocket. Handles request/response correlation,
timeouts, and disconnection.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Optional

logger = logging.getLogger("veilguard.bridge")

# Global bridge instance
_bridge: Optional["ClientBridge"] = None


def get_bridge() -> Optional["ClientBridge"]:
    return _bridge


def set_bridge(bridge: "ClientBridge"):
    global _bridge
    _bridge = bridge


class ClientBridge:
    """Manages WebSocket connection to a single client daemon."""

    def __init__(self):
        self.ws = None
        self.connected = False
        self.client_id: str = ""
        self.last_ping: float = 0
        self.pending: dict[str, asyncio.Future] = {}

    async def send_command(self, method: str, params: dict, timeout: float = 10.0):
        """Send a generic JSON-RPC command to the daemon and await result."""
        if not self.connected or self.ws is None:
            return None

        request_id = f"cmd-{uuid.uuid4().hex[:8]}"
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self.pending[request_id] = future

        msg = json.dumps({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        })

        try:
            await self.ws.send_text(msg)
        except Exception:
            self.pending.pop(request_id, None)
            return None

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            # Result comes as string from on_message — try parsing as JSON
            if isinstance(result, str):
                try:
                    return json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    return result
            return result
        except asyncio.TimeoutError:
            self.pending.pop(request_id, None)
            return None

    async def execute_remote(self, tool: str, args: dict, timeout: float = 60.0) -> str:
        """Send a tool call to the client daemon and await the result."""
        if not self.connected or self.ws is None:
            return "Error: No client daemon connected. Start the Veilguard client."

        request_id = f"req-{uuid.uuid4().hex[:8]}"
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self.pending[request_id] = future

        msg = json.dumps({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "execute_tool",
            "params": {"tool": tool, "args": args},
        })

        try:
            await self.ws.send_text(msg)
            logger.debug(f"[BRIDGE] Sent {tool}({list(args.keys())}) id={request_id}")
        except Exception as e:
            self.pending.pop(request_id, None)
            return f"Error: Failed to send to client: {e}"

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            self.pending.pop(request_id, None)
            return f"Error: Client tool execution timed out after {timeout}s"

    def on_message(self, data: dict):
        """Handle an incoming message from the client daemon."""
        msg_id = data.get("id")

        # Heartbeat pong
        if data.get("method") == "pong" or data.get("result") == "pong":
            self.last_ping = time.time()
            return

        # Tool result
        if msg_id and msg_id in self.pending:
            future = self.pending.pop(msg_id)
            if future.done():
                return

            if "error" in data:
                err = data["error"]
                msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                future.set_result(f"Error: {msg}")
            else:
                future.set_result(data.get("result", ""))

    def on_disconnect(self):
        """Handle client daemon disconnection."""
        self.connected = False
        self.ws = None
        self.client_id = ""

        # Resolve all pending futures with error
        for rid, fut in list(self.pending.items()):
            if not fut.done():
                fut.set_result("Error: Client disconnected during tool execution")
        self.pending.clear()
        logger.warning("[BRIDGE] Client daemon disconnected")

    def status(self) -> dict:
        """Return bridge status for API."""
        return {
            "connected": self.connected,
            "client_id": self.client_id,
            "last_ping": self.last_ping,
            "pending_requests": len(self.pending),
        }
