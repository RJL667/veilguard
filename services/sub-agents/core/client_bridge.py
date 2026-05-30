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


SANDBOX_USER_ID = "system:sandbox"


def _sandbox_token_from_env() -> Optional[str]:
    """The sandbox container ships with a pre-shared secret in env.

    Both sides (sub-agents process AND veilguard-sandbox container) read
    ``VEILGUARD_SANDBOX_TOKEN`` from the same `.env` file via Docker
    Compose.  If the env var is unset the sandbox identity is disabled
    — sub-agents won't accept WS auth as ``system:sandbox``, server
    agents fall back to "no daemon" errors when the user is offline.

    This is intentional: the sandbox is a TRUSTED SYSTEM COMPONENT, not
    a per-user identity, so it gets a separate auth channel from the
    per-user QR-token store.  Anyone with VEILGUARD_SANDBOX_TOKEN can
    impersonate the sandbox — keep it out of source control.
    """
    tok = os.environ.get("VEILGUARD_SANDBOX_TOKEN", "").strip()
    return tok or None


def validate_token(user_id: str, token: str) -> bool:
    """Constant-time check that ``token`` matches ``user_id``'s record.

    The legacy ``CLIENT_TOKEN`` env-var fallback was removed on
    2026-04-24 as part of the spear-phish incident response: it authed
    any ``user_id`` absent from ``.client-tokens.json`` against a single
    shared secret, which meant one leaked value gave impersonation
    across every not-yet-registered identity. Per-user tokens are the
    only path now; unknown user_ids fail closed.

    Exception: ``user_id == "system:sandbox"`` authenticates against the
    process-wide ``VEILGUARD_SANDBOX_TOKEN`` env var (Phase B).  This is
    a TRUSTED INTERNAL identity used by the sandbox container, not by
    end users.  When the env var is unset the sandbox path is closed.
    """
    if not user_id or not token:
        return False
    if user_id == SANDBOX_USER_ID:
        expected = _sandbox_token_from_env()
        if expected is None:
            return False
        return hmac.compare_digest(expected, token)
    expected = get_user_token(user_id)
    if expected is None:
        return False
    return hmac.compare_digest(expected, token)


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
        # [PHASE_0_0_4_CAPABILITY_HANDSHAKE_2026_05_27] Per-spec §0.4 +
        # §0.0.4 in MULTI_AGENT_PLATFORM.md.  The daemon declares its
        # supported features in its auth payload (capabilities=[…]);
        # the server stores them so call sites can degrade gracefully
        # for older daemons that pre-date a feature.  Empty set means
        # "vintage daemon, assume nothing".  Common feature names:
        #   - "request_approval"   — daemon can render toast + return user decision
        #   - "toast"              — daemon can show informational toasts
        #   - "execute_remote"     — daemon will exec shell tools (vs. local-only)
        #   - "file_io"            — daemon owns read_file / write_file / edit_file
        #   - "bypass_rules"       — daemon honours bypass-rule API
        self.capabilities: frozenset[str] = frozenset()
        # Spec §0.0.4: bidirectional handshake — track which capabilities
        # the SERVER advertised back to this daemon so admin tools can
        # show the negotiated set.
        self.server_capabilities_advertised: frozenset[str] = frozenset()

    def has_capability(self, name: str) -> bool:
        """Return True if the connected daemon advertised ``name``.

        Call sites should use this BEFORE invoking a feature path so we
        can fall back / log instead of timing out a feature the daemon
        doesn't support.  Always False when ``self.connected`` is False.
        """
        if not self.connected:
            return False
        return name in self.capabilities

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

    async def request_approval(
        self,
        *,
        tool: str,
        args: dict,
        conv_id: str = "",
        agent_id: str = "user",
        request_id: str = "",
        policy_decision: str = "",
        policy_reason: str = "",
        level: str = "",
        timeout_s: int = 60,
    ) -> dict:
        """Ask the daemon for user approval on a tool call.

        Round-trips a JSON-RPC ``request_approval`` to the daemon and
        awaits the user's click.  Mirrors ``execute_remote``'s future
        plumbing but uses a dedicated method name so the daemon can
        route to its toast/UI layer instead of the tool executor.

        Returns:
            {"approved": bool, "reason": str, "tool": str,
             "request_id": str} on success.
            {"approved": false, "reason": "<error>"} on transport failure.

        Caller is responsible for the outer timeout wrap — we still set
        one here as a defensive backstop in case the daemon's UI hangs.
        """
        if not self.connected or self.ws is None:
            return {"approved": False, "reason": "client daemon not connected"}

        # [PHASE_0_0_4_CAPABILITY_HANDSHAKE_2026_05_27]  Vintage daemons
        # that pre-date the handshake won't have advertised
        # 'request_approval' — sending a request_approval method to
        # them will deadlock (they don't know to reply).  Per spec
        # §0.0.4: fall back to DENY for background-routed tools when
        # the daemon can't gate.  Foreground (browser-driven) tools
        # still go through their own ALLOW path before reaching here.
        if not self.has_capability("request_approval"):
            logger.warning(
                "[bridge] daemon %r missing 'request_approval' "
                "capability (pre-§0.0.4 daemon) — denying %s",
                self.client_id or "?", tool,
            )
            return {
                "approved": False,
                "reason": (
                    "daemon does not advertise request_approval "
                    "capability (upgrade the Veilguard client to "
                    "land §0.0.4)"
                ),
                "tool": tool,
                "request_id": request_id or "",
            }

        # Use the caller's request_id when supplied so the audit row and
        # the daemon log line match.
        req_id = request_id or f"appr-{uuid.uuid4().hex[:8]}"
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self.pending[req_id] = future

        msg = json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "request_approval",
            "params": {
                "tool":            tool,
                "args":            args,
                "conv_id":         conv_id,
                "agent_id":        agent_id,
                "policy_decision": policy_decision,
                "policy_reason":   policy_reason,
                "level":           level,
                "timeout_s":       timeout_s,
            },
        })

        try:
            await self.ws.send_text(msg)
            logger.info(
                f"[BRIDGE] request_approval id={req_id} tool={tool!r} "
                f"conv={conv_id[:8]} level={level} timeout={timeout_s}s"
            )
        except Exception as e:
            self.pending.pop(req_id, None)
            return {"approved": False, "reason": f"send failed: {e}"}

        # +5s slack so the daemon-side timeout (timeout_s) fires first
        # and gives us a structured response instead of TimeoutError.
        try:
            result = await asyncio.wait_for(future, timeout=timeout_s + 5)
        except asyncio.TimeoutError:
            self.pending.pop(req_id, None)
            return {
                "approved": False,
                "reason": f"daemon did not respond in {timeout_s + 5}s",
            }

        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            # Daemon returned a bare string — treat as denial.
            return {"approved": False, "reason": result}
        return {"approved": False, "reason": "unparseable daemon response"}

    async def execute_remote(self, tool: str, args: dict, timeout: float = 60.0) -> str:
        """Send a tool call to the client daemon and await the result."""
        if not self.connected or self.ws is None:
            return "Error: No client daemon connected. Start the Veilguard client."

        # UTF-8 forcing for Windows shells.
        #
        # The client daemon does ``subprocess.run(..., text=True)`` with
        # bare UTF-8 decoding.  On Windows, child processes (especially
        # python.exe without PYTHONIOENCODING set) emit bytes in the
        # active codepage — typically cp1252.  When a script prints a
        # smart-quote / em-dash / pygame banner with extended chars,
        # the daemon's strict UTF-8 decode raises ``'utf-8' codec can't
        # decode byte 0x97 …``, which bubbles up as the entire tool
        # call's error.
        #
        # We can't change the daemon's decoding without rebuilding the
        # PyInstaller binary (intentionally not on the upgrade path),
        # but we CAN force the child to emit UTF-8 by prepending env-var
        # / chcp prelude to the command.  ``cmd.exe /c`` treats ``&``
        # as a separator that runs the next command unconditionally:
        #
        #   chcp 65001 > nul 2>&1    — UTF-8 console code page (affects
        #                              most native tools: git, dir, etc.)
        #   set PYTHONUTF8=1         — Python 3.7+ UTF-8 mode (overrides
        #                              locale, makes sys.std{in,out,err}
        #                              UTF-8 regardless of codepage)
        #   set PYTHONIOENCODING=...  — belt-and-suspenders for older
        #                              Python or third-party readers
        #
        # No behavioural change for Unix daemons (their child processes
        # already emit UTF-8 by default), but we still set the env vars
        # there for parity in case the same script is run on both hosts.
        if tool == "run_command":
            cmd = args.get("command", "") if isinstance(args, dict) else ""
            if cmd:
                if self.platform and self.platform.startswith("win"):
                    prelude = (
                        "chcp 65001 > nul 2>&1 & "
                        "set PYTHONUTF8=1 & "
                        "set PYTHONIOENCODING=utf-8:replace & "
                    )
                    args = {**args, "command": prelude + cmd}
                elif self.platform:
                    # Unix: export so it survives sub-shell spawns from
                    # the wrapping ``bash -c``.
                    prelude = (
                        "export PYTHONUTF8=1; "
                        "export PYTHONIOENCODING=utf-8:replace; "
                    )
                    args = {**args, "command": prelude + cmd}

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
        method = data.get("method", "") or ""

        if method == "pong" or data.get("result") == "pong":
            self.last_ping = time.time()
            return

        # Daemon-initiated notifications.  Until 2026-05-25 the bridge
        # was strictly cloud-asks/daemon-answers; the tray menu now
        # lets the user change permission_level directly, so we accept
        # a small set of inbound methods.  Anything not whitelisted
        # falls through to the response-future path below.
        if method == "set_permission_level":
            self._handle_inbound_set_permission_level(data.get("params") or {})
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

    def _handle_inbound_set_permission_level(self, params: dict) -> None:
        """User changed permission_level from the tray.  Persist it."""
        try:
            level = (params.get("level") or "").lower()
            scope = (params.get("scope") or "user").lower()
            source = params.get("source", "tray")
            conv_id = params.get("conv_id", "") or ""
            # Import locally so the bridge module stays lightweight
            # and tests don't pay the lance startup cost.
            from .client_settings import ClientSettingsStore, PERMISSION_LEVELS
            if level not in PERMISSION_LEVELS:
                logger.warning(
                    f"[BRIDGE] inbound set_permission_level: bad level "
                    f"{level!r} from user={self.user_id[:8]}"
                )
                return
            store = ClientSettingsStore.get()
            if scope == "conv" and conv_id:
                store.set_conv_override(self.user_id, conv_id, level)
                logger.info(
                    f"[BRIDGE] tray set conv override "
                    f"user={self.user_id[:8]} conv={conv_id[:8]} "
                    f"level={level} src={source}"
                )
            else:
                store.set_user_level(self.user_id, level)
                logger.info(
                    f"[BRIDGE] tray set user default "
                    f"user={self.user_id[:8]} level={level} src={source}"
                )
        except Exception as e:
            logger.warning(f"[BRIDGE] inbound set_permission_level failed: {e}")

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
        """Return bridge status for API.

        2026-05-18: surface ``platform``/``os_name``/``os_release``/
        ``shell`` so callers (the pii-proxy workspace injector, the
        admin dashboard) can render real OS identity instead of
        guessing from path-prefix heuristics. Daemons older than
        0.2.5 don't send these on auth — their values will be empty
        strings and consumers should treat that as "unknown".
        """
        return {
            "connected":        self.connected,
            "client_id":        self.client_id,
            "user_id":          self.user_id,
            "last_ping":        self.last_ping,
            "pending_requests": len(self.pending),
            "platform":         self.platform   or "",
            "os_name":          self.os_name    or "",
            "os_release":       self.os_release or "",
            "shell":            self.shell      or "",
        }
