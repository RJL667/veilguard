"""Cloud-side WebSocket bridge to client daemons.

One registered ``ClientBridge`` per LibreChat ``user_id`` — sub-agents
route tool-execution requests to the bridge that matches the currently
authenticated user.  Prior revision used a single module-level bridge,
which meant any LibreChat user's tool call routed to whichever daemon
had connected most recently — cross-user filesystem exposure.

Tokens are per-user too: ``/api/client/register`` mints one when the
user is authenticated to LibreChat; the daemon sends ``user_id`` +
``token`` during WebSocket auth and we refuse the connection if the
pair doesn't match.
"""

import asyncio
import hmac
import json
import logging
import os
import secrets
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger("veilguard.bridge")


# ── Per-user bridge registry ─────────────────────────────────────────

_registry: dict[str, "ClientBridge"] = {}


def get_bridge(user_id: str = "") -> Optional["ClientBridge"]:
    """Return the bridge registered for ``user_id``.

    Passing an empty ``user_id`` returns the only bridge when exactly
    one is connected (transition behaviour for callers that don't yet
    carry user context). When multiple daemons are registered and no
    user_id is supplied, returns ``None`` — better to fail visibly than
    route to the wrong user's machine.
    """
    if user_id:
        return _registry.get(user_id)
    if len(_registry) == 1:
        return next(iter(_registry.values()))
    return None


def set_bridge(bridge: Optional["ClientBridge"], user_id: str = "") -> None:
    """Register ``bridge`` for ``user_id`` or unregister when ``bridge`` is None."""
    if bridge is None:
        if user_id:
            _registry.pop(user_id, None)
        return
    if not user_id:
        user_id = getattr(bridge, "user_id", "") or "anonymous"
    _registry[user_id] = bridge


def all_bridges() -> dict[str, "ClientBridge"]:
    """Return a snapshot of ``{user_id: ClientBridge}`` — for admin / debug."""
    return dict(_registry)


# ── Per-user installation tokens ─────────────────────────────────────
# Tokens map LibreChat user_id → single-use secret the daemon forwards
# during WebSocket auth.  Installer fetches the token via
# /api/client/register while the user is authenticated to LibreChat
# and embeds it in the QR / paste string.

_TOKENS_PATH = Path(
    os.environ.get(
        "VEILGUARD_TOKENS_FILE",
        "/home/rudol/veilguard/.client-tokens.json",
    )
)
_tokens: dict[str, str] = {}
_tokens_loaded = False


def _ensure_tokens_loaded() -> None:
    global _tokens_loaded, _tokens
    if _tokens_loaded:
        return
    _tokens_loaded = True
    if _TOKENS_PATH.exists():
        try:
            _tokens = json.loads(_TOKENS_PATH.read_text())
        except Exception as e:
            logger.warning(f"Tokens file unreadable, starting fresh: {e}")
            _tokens = {}


def _save_tokens() -> None:
    try:
        _TOKENS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _TOKENS_PATH.write_text(json.dumps(_tokens, indent=2))
        try:
            os.chmod(_TOKENS_PATH, 0o600)
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"Failed to persist client tokens: {e}")


def get_user_token(user_id: str) -> Optional[str]:
    """Return the stored token for ``user_id`` or ``None``."""
    if not user_id:
        return None
    _ensure_tokens_loaded()
    return _tokens.get(user_id)


def register_user_token(user_id: str, regenerate: bool = False) -> str:
    """Return ``user_id``'s token, creating one on first call.

    ``regenerate=True`` mints a fresh token even if one already exists —
    use for token rotation or when revocation is needed.
    """
    if not user_id:
        raise ValueError("user_id required")
    _ensure_tokens_loaded()
    if regenerate or user_id not in _tokens:
        _tokens[user_id] = secrets.token_hex(16)
        _save_tokens()
    return _tokens[user_id]


def validate_token(user_id: str, token: str) -> bool:
    """Constant-time check that ``token`` matches ``user_id``'s record.

    Falls back to the legacy shared ``CLIENT_TOKEN`` env var so already-
    deployed daemons keep working during the rollout.  Once every daemon
    has been reinstalled with per-user tokens, remove the fallback.
    """
    if not user_id or not token:
        return False
    expected = get_user_token(user_id)
    if expected is not None:
        return hmac.compare_digest(expected, token)
    legacy = os.environ.get("CLIENT_TOKEN", "")
    if legacy:
        return hmac.compare_digest(legacy, token)
    return False


def revoke_user_token(user_id: str) -> None:
    _ensure_tokens_loaded()
    if _tokens.pop(user_id, None) is not None:
        _save_tokens()


# ── Bridge ───────────────────────────────────────────────────────────

class ClientBridge:
    """Manages one WebSocket connection to a client daemon."""

    def __init__(self):
        self.ws = None
        self.connected = False
        self.client_id: str = ""
        self.user_id: str = ""
        self.last_ping: float = 0
        self.pending: dict[str, asyncio.Future] = {}
        # Host identity announced by the daemon in its auth payload.
        # ``platform`` is sys.platform ("win32"/"linux"/"darwin"),
        # ``os_name`` is platform.system() ("Windows"/"Linux"/"Darwin"),
        # ``shell`` is the default shell the daemon will invoke. Used
        # to prepend a one-shot host hint to the first ``run_command``
        # response so Claude knows whether to emit CMD/PowerShell or
        # bash syntax — otherwise Claude defaults to Unix and wastes
        # the first call on Windows hosts trying ``pwd``/``ls -la``.
        self.platform: str = ""
        self.os_name: str = ""
        self.os_release: str = ""
        self.shell: str = ""
        self._host_hint_sent: bool = False

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
        except asyncio.TimeoutError:
            self.pending.pop(request_id, None)
            return f"Error: Client tool execution timed out after {timeout}s"

        # First ``run_command`` per bridge: prepend a one-line host
        # hint so Claude knows whether to emit Windows CMD or Unix
        # bash syntax for subsequent commands. Without this, Claude
        # defaults to Unix and the first call on Windows hosts wastes
        # a round-trip on ``pwd``/``ls -la``-style failures. We only
        # stamp it once per connected daemon — after that, the hint
        # is already in context and repeating it is noise. If the
        # tool call errored, don't consume the hint — leave it for
        # the next attempt so Claude still sees it on retry.
        if tool == "run_command" and self.platform and not self._host_hint_sent:
            is_error = isinstance(result, str) and result.startswith("Error:")
            if not is_error:
                shell_hint = (
                    f"CMD (use `cd` for current dir, `dir` for listing, "
                    f"`echo %OS%` for OS check)"
                    if self.platform.startswith("win")
                    else f"bash/zsh (`pwd`, `ls -la`, `uname -a`)"
                )
                hint = (
                    f"[Host: {self.os_name or self.platform} "
                    f"{self.os_release} — shell: {self.shell or 'default'}. "
                    f"Use {shell_hint}.]\n"
                )
                result = hint + (result if isinstance(result, str) else str(result))
                self._host_hint_sent = True

        return result

    def on_message(self, data: dict):
        """Handle an incoming message from the client daemon."""
        msg_id = data.get("id")

        if data.get("method") == "pong" or data.get("result") == "pong":
            self.last_ping = time.time()
            return

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
        # client_id / user_id are kept so we can log who left.

        for rid, fut in list(self.pending.items()):
            if not fut.done():
                fut.set_result("Error: Client disconnected during tool execution")
        self.pending.clear()
        logger.warning(
            f"[BRIDGE] Client daemon disconnected (user={self.user_id} "
            f"client_id={self.client_id})"
        )

    def status(self) -> dict:
        """Return bridge status for API."""
        return {
            "connected": self.connected,
            "client_id": self.client_id,
            "user_id": self.user_id,
            "last_ping": self.last_ping,
            "pending_requests": len(self.pending),
        }
