"""Veilguard URL-scheme handler — ``veilguard://configure?token=...``.

The LibreChat fork's Cowork onboarding banner renders a "Pair now"
button whose href is::

  veilguard://configure?
      token=<per-user-secret-from-/api/veilguard-client/register>
      &server=<wss-url>
      &user_id=<librechat-mongo-id>

Windows looks up ``HKCR\\veilguard`` (registered by installer.iss in
0.3.0+) and launches ``VeilguardClient.exe "veilguard://configure?…"``.
This module owns argv parsing + config.yaml writing + restart handoff.

Trust model:

  - The ``token`` is the per-user shared secret the cloud bridge uses
    to authenticate the WS connection.  Stealing this token gives
    impersonation, same as stealing the QR code, so handle it like a
    password.  Logged values are always truncated to first 8 chars.
  - The ``server`` URL gates which cloud the daemon talks to.  We only
    accept ``ws://`` / ``wss://`` schemes — anything else is rejected
    so a hostile link can't redirect the daemon to file:// or a
    payload server.
  - When a second instance arrives via single_instance handoff, we
    require the running daemon to ALREADY be authenticated as the same
    user_id.  Mismatched user_id = abort (we'd otherwise let a
    spear-phisher hijack an active session by serving a link that
    re-points the bridge).

Headless mode skips ALL of this — sandbox containers don't get URL
scheme registrations and their config comes from env vars.
"""

from __future__ import annotations

import logging
import os
import urllib.parse
from pathlib import Path
from typing import Optional

logger = logging.getLogger("veilguard.daemon.deeplink")


SCHEME = "veilguard"
HOST_CONFIGURE = "configure"
HOST_APPROVE   = "approve"   # veilguard://approve?token=...&action=approve|deny|always


def parse_approve(url: str) -> Optional[dict]:
    """Parse a ``veilguard://approve?token=X&action=approve|deny|always`` URL.

    Returns ``{"token": ..., "action": ...}`` on success, or None when
    the URL doesn't match this host (caller should fall through to
    other parsers).  Used to deliver toast-button clicks to the running
    daemon without opening a browser tab (Windows routes
    ``veilguard://...`` to the registered protocol handler, which is
    VeilguardClient.exe itself; single-instance handoff forwards it to
    the running daemon's loop).
    """
    if not is_deep_link(url):
        return None
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != SCHEME:
        return None
    host = (parsed.netloc or parsed.path).strip("/").lower()
    if host != HOST_APPROVE:
        return None
    q = urllib.parse.parse_qs(parsed.query)
    token = (q.get("token") or [""])[0]
    action = (q.get("action") or ["approve"])[0].lower()
    if action not in ("approve", "deny", "always"):
        action = "approve"
    if not token:
        return None
    return {"token": token, "action": action}


def is_deep_link(arg: str) -> bool:
    """True if `arg` looks like one of our URL-scheme invocations."""
    if not isinstance(arg, str):
        return False
    return arg.startswith(f"{SCHEME}://")


def parse_configure(url: str) -> Optional[dict]:
    """Parse a ``veilguard://configure?…`` URL into a config dict.

    Returns None when the URL is malformed or the host segment isn't
    ``configure``.  Raises ValueError when required params are missing
    or the server URL is unsafe so the caller can show a clear error.
    """
    if not is_deep_link(url):
        return None
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != SCHEME:
        return None
    # urlparse treats "configure" as the netloc when no // is at the
    # start; with our `veilguard://configure?…` syntax it lands in
    # ``netloc``.  Be defensive — accept either form.
    host = (parsed.netloc or parsed.path).strip("/").lower()
    if host != HOST_CONFIGURE:
        logger.warning(f"[deeplink] unknown host {host!r} in {url!r}")
        return None

    q = urllib.parse.parse_qs(parsed.query)
    def _one(name: str) -> str:
        vs = q.get(name) or []
        return vs[0] if vs else ""

    token   = _one("token")
    server  = _one("server")
    user_id = _one("user_id")
    client_id = _one("client_id") or ""
    # 0.4+: optional `env=` lets the cloud pre-name the environment so
    # the user sees "prod" / "local" / etc. in the tray menu instead of
    # an auto-derived hostname.  Optional — if absent we derive from
    # the server URL via daemon.env.derive_env_name.
    env_name = _one("env")

    missing = [n for n, v in (("token", token), ("server", server),
                              ("user_id", user_id)) if not v]
    if missing:
        raise ValueError(
            f"deep link missing required params: {', '.join(missing)}"
        )

    # Reject non-websocket schemes — protects against a hostile link
    # that tries to point the daemon at a file:// or http:// honeypot.
    if not (server.startswith("ws://") or server.startswith("wss://")):
        raise ValueError(
            f"server URL must be ws:// or wss:// (got {server!r})"
        )

    return {
        "server":    server,
        "token":     token,
        "user_id":   user_id,
        "client_id": client_id,
        "env_name":  env_name,
    }


def apply_configure(payload: dict, *, config_path: Optional[str] = None) -> str:
    """Persist a parsed configure payload to ``~/.veilguard/config.yaml``.

    0.4+: ADDS or UPDATES the named environment instead of overwriting
    the whole config.  A daemon already paired with `prod` can take a
    `veilguard://configure?env=local&...` link and end up with BOTH
    pairings active.  The spear-phish guard now operates per-env:
    refusing to overwrite an existing env's user_id with a different
    one (operator can disable / rename the env to recover).

    Atomic write via tmp+rename; preserves all non-env keys
    (project_root, timeout, auto_update, etc.).
    """
    try:
        import yaml
    except ImportError:
        yaml = None

    # Local import — avoid pulling env.py into module init order.
    from .env import (
        EnvConfig, parse_environments, write_environments,
        derive_env_name, merge_env,
    )

    if config_path is None:
        config_path = os.path.join(
            os.path.expanduser("~"), ".veilguard", "config.yaml",
        )
    cfg_dir = Path(config_path).parent
    cfg_dir.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if os.path.exists(config_path):
        try:
            if yaml is not None:
                with open(config_path, "r", encoding="utf-8") as f:
                    existing = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"[deeplink] could not read existing config: {e}")
            existing = {}

    envs = parse_environments(existing)

    # Resolve env name: explicit `env=` query param > derived from
    # server hostname.
    env_name = derive_env_name(
        payload["server"], hint=payload.get("env_name", "") or "",
    )

    # Per-env spear-phish guard: if an env with this name already has
    # a DIFFERENT user_id, refuse to silently clobber it.
    prev = envs.get(env_name)
    if prev is not None and prev.user_id and prev.user_id != payload["user_id"]:
        raise ValueError(
            f"deep link wants user_id={_short(payload['user_id'])} for env "
            f"{env_name!r} but config has user_id={_short(prev.user_id)}; "
            "refusing to clobber. Open the tray → 'Add server…' to pair "
            "as a different env name."
        )

    new_env = EnvConfig(
        name=env_name,
        server=payload["server"],
        token=payload["token"],
        user_id=payload["user_id"],
        enabled=True,
    )
    envs = merge_env(envs, new_env)

    merged = dict(existing)
    if payload.get("client_id"):
        merged["client_id"] = payload["client_id"]
    merged = write_environments(merged, envs)

    tmp = str(Path(config_path).with_suffix(".yaml.tmp"))
    if yaml is not None:
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.dump(merged, f, default_flow_style=False)
    else:
        import json as _json
        with open(tmp, "w", encoding="utf-8") as f:
            for k, v in merged.items():
                f.write(f"{k}: {_json.dumps(v)}\n")

    try:
        os.replace(tmp, config_path)
    except OSError:
        try: os.unlink(tmp)
        except Exception: pass
        raise

    try:
        os.chmod(config_path, 0o600)
    except OSError:
        pass

    logger.info(
        f"[deeplink] env={env_name} user={_short(payload['user_id'])} "
        f"server={payload['server']} → {config_path} "
        f"(total envs now: {len(envs)})"
    )
    return config_path


def _short(s: str) -> str:
    if not s:
        return "-"
    return s[:8] + ("…" if len(s) > 8 else "")


__all__ = [
    "SCHEME",
    "is_deep_link",
    "parse_configure",
    "apply_configure",
    "parse_approve",
]
