"""Single-instance lock + argv handoff via multiprocessing.connection.

Why this exists:

  After we register ``veilguard://configure?...`` as a URL scheme,
  Windows launches a fresh ``VeilguardClient.exe`` every time the user
  clicks a pairing link.  If a daemon is ALREADY running:

    1. WS auth would race — two daemons same user_id, second one's
       auth succeeds and bumps the first one off (per-user bridge).
    2. The tray icon would multiply.
    3. The setup_server localhost:9090 would fail to bind.

  So the second instance has to detect "I'm a duplicate", hand its
  argv off to the running one, and exit.  Standard Windows app pattern.

Mechanism — multiprocessing.connection.Listener:

  - First instance binds ``Listener(("127.0.0.1", HANDOFF_PORT),
    authkey=<per-install secret>)`` in a daemon thread.  When messages
    arrive (pickled tuples), it dispatches them to the deep-link handler
    we wire in from veilguard_client.py.

  - Second instance tries the same port with ``Client(...)``.  If the
    connection succeeds, we're a duplicate — send argv, exit 0.  If it
    fails (port unbound / authkey wrong), we proceed as a fresh first
    instance.

  - The authkey lives in ``%LOCALAPPDATA%\\Veilguard\\install_secret``
    so other processes on the machine can't fake a handoff and trick
    the daemon into re-pairing with a hostile server.

  - Port 9091 is the default (9090 is taken by setup_server.py during
    first-run setup).  Override via VEILGUARD_HANDOFF_PORT env if you
    need to avoid a clash.

Cross-platform: ``multiprocessing.connection.Listener`` works on Linux
too (the sandbox container's headless build).  We just skip it when
headless because the container doesn't need re-pair UX.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import sys
import threading
import time
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("veilguard.daemon.single_instance")


HANDOFF_PORT = int(os.environ.get("VEILGUARD_HANDOFF_PORT", "9091"))


def _config_dir() -> Path:
    """OS-appropriate per-user dir for storing the handoff authkey.

    Windows: %LOCALAPPDATA%\\Veilguard\\
    Linux/macOS: ~/.veilguard/
    Stays in lock-step with where setup_server.py writes config.yaml so
    the daemon and the deep-link handler share state.
    """
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "Veilguard"
    return Path.home() / ".veilguard"


def _authkey() -> bytes:
    """Load or mint the per-install handoff authkey.

    The file is created with mode 0o600 on first run and lives in
    ``$LOCALAPPDATA/Veilguard/install_secret`` (Windows) or
    ``~/.veilguard/install_secret`` (Unix).  Both the first and second
    instance read the same file; processes outside the user's profile
    can't read it (NTFS ACL inherited from %LOCALAPPDATA%) so they
    can't impersonate.
    """
    path = _config_dir() / "install_secret"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            data = path.read_bytes()
            if len(data) >= 16:
                return data
        except OSError:
            pass
    # Mint a new one — 32 bytes of secrets.token_bytes is overkill but
    # fast and removes any guesswork.
    secret = secrets.token_bytes(32)
    try:
        path.write_bytes(secret)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass   # Windows ignores chmod; ACL is enough.
    except OSError as e:
        logger.warning(f"[single-instance] could not write authkey: {e}")
    return secret


# ── Sender (second-instance side) ───────────────────────────────────────


def send_handoff(argv: list[str], *, timeout_s: float = 2.0) -> bool:
    """Try to hand `argv` off to a running daemon.  Returns True on success.

    Called as the very first action of the second instance.  If we
    return True the caller MUST exit immediately so we don't double-up
    on the WS / tray.
    """
    try:
        conn = Client(
            ("127.0.0.1", HANDOFF_PORT),
            authkey=_authkey(),
        )
    except (ConnectionRefusedError, OSError):
        return False
    try:
        conn.send({"kind": "argv", "argv": list(argv), "ts": time.time()})
        # Wait briefly for an ack so the first instance has time to
        # actually process the message before we exit.
        if conn.poll(timeout=timeout_s):
            ack = conn.recv()
            logger.info(f"[single-instance] handoff ack: {ack}")
        return True
    except Exception as e:
        logger.warning(f"[single-instance] handoff send failed: {e}")
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ── Receiver (first-instance side) ──────────────────────────────────────


def start_listener(on_handoff: Callable[[list[str]], None]) -> Optional[threading.Thread]:
    """Spawn a daemon thread that accepts handoff messages.

    ``on_handoff(argv)`` is called from the listener thread with the
    second-instance's argv when a pairing link arrives.  Callback is
    responsible for re-routing into the daemon's deep-link handler.
    Listener exits when the daemon process exits (daemon=True).

    Returns None if the port can't be bound (another process owns it —
    we should have detected that via send_handoff already).
    """
    try:
        listener = Listener(
            ("127.0.0.1", HANDOFF_PORT),
            authkey=_authkey(),
        )
    except OSError as e:
        logger.warning(
            f"[single-instance] cannot bind handoff port {HANDOFF_PORT}: {e} "
            "(re-pair from another instance will not be received)"
        )
        return None

    def _serve():
        logger.info(
            f"[single-instance] handoff listener bound 127.0.0.1:{HANDOFF_PORT}"
        )
        while True:
            try:
                conn = listener.accept()
            except OSError as e:
                logger.debug(f"[single-instance] listener.accept ended: {e}")
                return
            try:
                msg = conn.recv()
                argv = list((msg or {}).get("argv") or [])
                logger.info(
                    f"[single-instance] received handoff "
                    f"argv={argv[:2]}{'…' if len(argv) > 2 else ''}"
                )
                try:
                    on_handoff(argv)
                    conn.send({"ok": True})
                except Exception as e:
                    logger.warning(f"[single-instance] on_handoff raised: {e}")
                    try:
                        conn.send({"ok": False, "error": str(e)})
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"[single-instance] connection broke: {e}")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

    t = threading.Thread(
        target=_serve, name="veilguard-handoff-listener", daemon=True,
    )
    t.start()
    return t


__all__ = ["HANDOFF_PORT", "send_handoff", "start_listener"]
