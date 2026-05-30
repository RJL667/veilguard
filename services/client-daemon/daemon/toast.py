"""Windows toast notifications for tool-approval requests.

Architecture:

  Cloud sends `request_approval` over WS → veilguard_client.py routes
  to `ask_user_via_toast()` → toast appears with Approve / Deny /
  Always-for-this-conv buttons → click triggers a callback URL that
  the toast XML embeds (`activationType="protocol"`) → daemon's HTTP
  callback receiver captures the click → futures fire → daemon
  sends the decision back over WS.

Why HTTP callback instead of in-process toast click handlers:

  Windows toasts run in a separate XAML host process; the callback for
  a button click only reliably arrives via the Toast Activation API,
  which requires the app to be registered with COM as a TOAST_ACTIVATOR.
  PyInstaller-bundled daemons don't have that registration unless we
  ship an AppUserModelID + ToastActivator in the Inno Setup script —
  fragile, and the registration drifts every time the binary moves.

  The HTTP callback shim is a much smaller surface: bind a localhost
  HTTP server on a random port at toast-emit time, embed the URL in
  the toast's button `arguments` field, and unblock when the user's
  click triggers an HTTP GET on it.  Localhost-only, port closes
  immediately after the click — no persistent attack surface.

Fallback when winotify is unavailable:

  Headless / non-Windows / missing-dep builds will set HAS_TOAST_UI =
  False at import time.  Callers fall back to the Phase A auto-approve
  behavior.  We never raise from this module — UI failures are not
  fatal to the daemon.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import secrets
import threading
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

logger = logging.getLogger("veilguard.daemon.toast")

# ── Optional imports — module loads cleanly on non-Windows ─────────────


HAS_TOAST_UI = False
_winotify = None

if platform.system() == "Windows":
    try:
        import winotify
        HAS_TOAST_UI = True
        _winotify = winotify
        # Surface the loaded version so we can correlate behaviour
        # changes with winotify upgrades (silent show() failures in
        # certain Windows builds have been version-sensitive).
        _wv = getattr(winotify, "__version__", "?")
        logger.info(
            f"[toast] winotify loaded — version={_wv}; HAS_TOAST_UI=True"
        )
    except Exception as _e:   # pragma: no cover
        logger.warning(
            f"[toast] winotify unavailable ({_e}); toast UI disabled. "
            "Add `winotify` to requirements-windows.txt and rebuild "
            "the installer to enable Phase C UI."
        )


# ── Per-request click capture ─────────────────────────────────────────


@dataclass
class ApprovalDecision:
    approved: bool
    reason: str               # "user_clicked" | "always_for_conv" | ...
    persist_for_conv: bool = False   # write conv-override on user request


# Map of token → asyncio Future.  When the user clicks a toast button,
# the localhost HTTP handler resolves the future by token.  Tokens are
# 16-hex unguessable so a hostile process can't preempt approvals.
_pending: dict[str, asyncio.Future] = {}
_callback_server: Optional[HTTPServer] = None
_callback_port: int = 0
_server_lock = threading.Lock()


class _CallbackHandler(BaseHTTPRequestHandler):
    """Handles GET /approve|deny|always?token=... — fires the future."""

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler convention)
        try:
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            token = (qs.get("token") or [""])[0]
            action = parsed.path.strip("/").lower()  # "approve"/"deny"/"always"

            future = _pending.pop(token, None)
            if future is None or future.done():
                # Stale toast — common case: user clicks a notification
                # in Action Center that's left over from an already-
                # completed (or timed-out) approval.  Returning a bare
                # 404 looks like the click did nothing.  Surface what
                # actually happened so the user knows + closes the tab.
                self.send_response(404)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write((
                    "<!doctype html><html><head>"
                    "<meta charset='utf-8'>"
                    "<title>Veilguard</title>"
                    "<style>"
                    "html,body{margin:0;height:100%;background:#0d1117;"
                    "color:#e6edf3;font:13px/1.4 'Segoe UI',sans-serif;"
                    "display:flex;align-items:center;justify-content:center;}"
                    ".card{padding:24px 32px;border:1px solid #30363d;"
                    "border-radius:6px;background:#161b22;text-align:center;"
                    "max-width:420px;}"
                    ".card h3{margin:0 0 8px 0;color:#8b949e;}"
                    ".card p{margin:4px 0;color:#8b949e;}"
                    "</style></head><body>"
                    "<div class='card'>"
                    "<h3>This approval has already been handled</h3>"
                    "<p>It was approved or timed out earlier.</p>"
                    "<p>You can close this window.</p>"
                    "</div>"
                    "<script>setTimeout(function(){try{window.close();}catch(e){}}, 1500);</script>"
                    "</body></html>"
                ).encode("utf-8"))
                return

            if action == "approve":
                decision = ApprovalDecision(
                    approved=True, reason="user_clicked_approve",
                )
            elif action == "always":
                decision = ApprovalDecision(
                    approved=True, reason="user_clicked_always_for_conv",
                    persist_for_conv=True,
                )
            else:
                decision = ApprovalDecision(
                    approved=False, reason="user_clicked_deny",
                )

            # Hop back to the future's loop — it's on the daemon's main
            # asyncio loop, this handler runs on an HTTP server thread.
            loop = future.get_loop()
            loop.call_soon_threadsafe(
                lambda: future.set_result(decision) if not future.done() else None
            )

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            # Self-closing HTML so the browser tab flashes briefly then
            # disappears.  window.close() only works on tabs JS opened
            # itself — modern browsers reject it for user-opened tabs
            # — but for a fresh tab created by clicking a toast action
            # most browsers honour it.  Fallback: dark splash so even
            # if it sticks around it looks intentional.
            self.wfile.write((
                "<!doctype html><html><head>"
                "<meta charset='utf-8'>"
                "<title>Veilguard</title>"
                "<style>"
                "html,body{margin:0;height:100%;background:#0d1117;"
                "color:#e6edf3;font:13px/1.4 'Segoe UI',sans-serif;"
                "display:flex;align-items:center;justify-content:center;}"
                ".card{padding:24px 32px;border:1px solid #30363d;"
                "border-radius:6px;background:#161b22;text-align:center;}"
                f".card h3{{margin:0 0 6px 0;color:"
                f"{'#3fb950' if action == 'approve' else '#d29922' if action == 'always' else '#f85149'};"
                "}}"
                "</style></head><body>"
                "<div class='card'>"
                f"<h3>Veilguard: {action}d</h3>"
                "<p>You can close this window.</p>"
                "</div>"
                # Long enough that the user sees what happened, short
                # enough that the tab doesn't linger.  window.close()
                # works on tabs the browser opened in response to an
                # OS protocol handler (toast → http://localhost).
                "<script>"
                "setTimeout(function(){try{window.close();}catch(e){}}, 1500);"
                "</script>"
                "</body></html>"
            ).encode("utf-8"))
        except Exception as e:
            logger.warning(f"[toast] callback handler error: {e}")
            try:
                self.send_response(500)
                self.end_headers()
            except Exception:
                pass

    def log_message(self, *args, **kwargs):
        return    # silence BaseHTTPRequestHandler's stderr spam


def _ensure_callback_server() -> int:
    """Start the localhost HTTP callback shim once, return its port."""
    global _callback_server, _callback_port
    if _callback_server is not None:
        return _callback_port
    with _server_lock:
        if _callback_server is not None:
            return _callback_port
        server = HTTPServer(("127.0.0.1", 0), _CallbackHandler)
        port = server.server_address[1]
        threading.Thread(
            target=server.serve_forever, name="veilguard-toast-callback",
            daemon=True,
        ).start()
        _callback_server = server
        _callback_port = port
        logger.info(f"[toast] callback shim listening on 127.0.0.1:{port}")
        return port


# ── Public API ────────────────────────────────────────────────────────


def get_callback_base() -> str:
    """Return the localhost callback URL prefix (e.g. http://127.0.0.1:PORT).

    Lazy-starts the callback server on first call.  Same shim that
    Windows toasts launch — exposed publicly so the tray viewer can
    fire Approve/Deny without depending on the toast banner appearing
    (useful when Windows silences the banner / Focus Assist is on).
    """
    port = _ensure_callback_server()
    return f"http://127.0.0.1:{port}"


def resolve_pending(token: str, *, action: str) -> bool:
    """Direct-resolve a pending approval by token (no HTTP round-trip).

    The viewer's inline Approve/Deny buttons call this to short-circuit
    the localhost-HTTP path.  Returns True if the token was found and
    a decision was registered, False if it was already resolved or
    never registered.  Thread-safe: posts the decision via
    call_soon_threadsafe on the future's loop.
    """
    future = _pending.pop(token, None)
    if future is None or future.done():
        return False
    if action == "approve":
        decision = ApprovalDecision(
            approved=True, reason="user_clicked_approve_in_viewer",
        )
    elif action == "always":
        decision = ApprovalDecision(
            approved=True, reason="user_clicked_always_in_viewer",
            persist_for_conv=True,
        )
    else:
        decision = ApprovalDecision(
            approved=False, reason="user_clicked_deny_in_viewer",
        )
    loop = future.get_loop()
    loop.call_soon_threadsafe(
        lambda: future.set_result(decision) if not future.done() else None
    )
    return True


async def ask_user_via_toast(
    *,
    tool: str,
    args: dict,
    conv_id: str = "",
    agent_id: str = "user",
    policy_decision: str = "",
    level: str = "",
    timeout_s: int = 60,
    request_id: Optional[str] = None,
) -> ApprovalDecision:
    """Pop a Windows toast asking for approval; return the decision.

    Returns ApprovalDecision(approved=False, reason="timeout") if the
    user doesn't click within `timeout_s`.  Raises only if the toast
    couldn't even be displayed — callers should treat that as a
    fail-closed deny (with audit).

    `request_id` is the daemon's correlation id from the WS request.
    When supplied, it's used as the callback token so the viewer's
    inline Approve/Deny buttons (keyed by request_id in TaskRegistry)
    can resolve the same future the toast banner would resolve.
    """
    if not HAS_TOAST_UI:
        # Should never be called when HAS_TOAST_UI=False; defensive.
        return ApprovalDecision(approved=False, reason="toast_ui_unavailable")

    # Unify token with the daemon's request_id so viewer + toast share
    # the same key.  Fallback: opaque random token (original behaviour)
    # for any call site that doesn't pass one.
    token = request_id or secrets.token_hex(16)
    loop = asyncio.get_event_loop()
    future: asyncio.Future = loop.create_future()
    _pending[token] = future

    port = _ensure_callback_server()
    # `base` is the localhost HTTP fallback — kept for callers that
    # land here from a browser address bar.  Toast banner buttons now
    # use the ``veilguard://approve`` protocol scheme below instead so
    # clicking doesn't open a browser tab.
    base = f"http://127.0.0.1:{port}"
    # Single-token URI scheme dispatched to the daemon via
    # single_instance handoff.  Windows resolves veilguard:// to
    # VeilguardClient.exe (registered by the installer); the launched
    # secondary process hands off to the running daemon over TCP 9091
    # and exits silently.  No browser, no tab.
    proto_base = "veilguard://approve"
    logger.info(
        f"[toast] proto_base={proto_base} (fallback HTTP base={base})"
    )
    # Force-flush every handler so a crash AFTER this point still leaves
    # the trail intact on disk.  Default RotatingFileHandler buffers up
    # to 4 KB before flushing — a hard-kill drops that buffer.
    for _h in logging.getLogger().handlers:
        try:
            _h.flush()
        except Exception:
            pass

    body = (
        f"Tool: {tool}\n"
        f"Policy: {policy_decision or '?'}  |  Level: {level or '?'}\n"
        f"Args: {_truncate(args, 120)}\n"
        f"Conv: {conv_id[:12] or '-'}"
    )

    # winotify's Notification supports up to ~5 buttons via add_actions.
    logger.info(
        f"[toast] preparing notification tool={tool!r} token={token[:12]} "
        f"callback_base={base!r}"
    )
    # [TEST_HOOK_AUTO_DENY_2026_05_26] When VEILGUARD_DAEMON_LOG_FULL_TOKEN=1,
    # also emit the full token so an offline test harness can drive
    # /approve|/deny without UI clicks.  Production should leave this
    # env var UNSET — exposing the full token in logs widens the surface
    # for any reader of the log file.  This hook exists only for the
    # autonomous Phase 8 (strict-mode failure-injection) test.
    if os.environ.get("VEILGUARD_DAEMON_LOG_FULL_TOKEN", "").strip() == "1":
        logger.info(f"[toast] TEST_HOOK full_token={token} callback_base={base}")
    try:
        toast_obj = _winotify.Notification(
            app_id="Veilguard",
            title="Veilguard: tool approval required",
            msg=body,
            duration="long",   # 25 sec on screen vs 7 (short)
        )
        # Toast button launch URLs use the veilguard:// custom protocol
        # so Windows hands the click directly to a launched
        # VeilguardClient.exe (registered as the URL handler by the
        # installer).  That secondary instance hands off via TCP 9091
        # to the running daemon — which resolves the approval inline
        # via toast.resolve_pending.  No browser tab opens.
        #
        # The HTTP callback shim at ``base`` is still alive as a
        # fallback for manual / debugging use; the toast banner itself
        # no longer routes through it.  An earlier hypothesis that
        # Windows suppresses banners with non-http actions was wrong —
        # it was Focus Assist coincidence.
        toast_obj.add_actions(
            label="Approve",
            launch=f"{proto_base}?token={token}&action=approve",
        )
        toast_obj.add_actions(
            label="Deny",
            launch=f"{proto_base}?token={token}&action=deny",
        )
        toast_obj.add_actions(
            label="Always for this conv",
            launch=f"{proto_base}?token={token}&action=always",
        )
        # Audio cue — winotify defaults to silent; force a sound so the
        # user notices even when Focus Assist is suppressing banners.
        try:
            from winotify import audio as _wa
            toast_obj.set_audio(_wa.Default, loop=False)
        except Exception as _e:
            logger.debug(f"[toast] set_audio unavailable: {_e}")
        logger.info("[toast] calling toast_obj.show()")
        _show_result = toast_obj.show()
        # winotify returns None on success.  If show() returned anything
        # other than None, log it — could be a status code from
        # PowerShell New-BurntToastNotification on newer versions.
        logger.info(
            f"[toast] show() returned {_show_result!r}; if Windows shows "
            "no banner, check Settings → Notifications → Veilguard "
            "(or Focus Assist / Do Not Disturb)"
        )
    except BaseException as e:   # incl. SystemExit / KeyboardInterrupt
        _pending.pop(token, None)
        logger.error(
            f"[toast] show() failed: {type(e).__name__}: {e}",
            exc_info=True,
        )
        for _h in logging.getLogger().handlers:
            try: _h.flush()
            except Exception: pass
        return ApprovalDecision(
            approved=False, reason=f"toast_show_failed: {e}",
        )

    try:
        return await asyncio.wait_for(future, timeout=timeout_s)
    except asyncio.TimeoutError:
        _pending.pop(token, None)
        logger.info(
            f"[toast] timed out after {timeout_s}s waiting for user click "
            f"on tool={tool!r}"
        )
        return ApprovalDecision(approved=False, reason="timeout")


# ── Helpers ───────────────────────────────────────────────────────────


def _truncate(args: dict, n: int) -> str:
    try:
        import json
        s = json.dumps(args, ensure_ascii=False)
    except Exception:
        s = repr(args)
    return s[:n] + ("..." if len(s) > n else "")


__all__ = [
    "HAS_TOAST_UI",
    "ApprovalDecision",
    "ask_user_via_toast",
    "get_callback_base",
    "resolve_pending",
]
