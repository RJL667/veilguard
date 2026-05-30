"""Multi-environment configuration for the client daemon.

Pre-0.4: config.yaml held a single set of credentials.  The daemon
opened one WS connection and could only be paired with one server at
a time — switching meant wiping creds + repairing.

0.4+: config holds a *list* of environments.  Each env has its own
{server, token, user_id, enabled}, and the daemon opens one WS per
enabled env.  A user with a prod LibreChat AND a local dev LibreChat
can have BOTH connected at the same time; tool calls from either side
route to the daemon over the appropriate bridge.

Backward compat:
  Old config shape (top-level server/token/user_id) is auto-migrated
  to a single env named "default" on first read.  No user action
  required.  Writes always use the new shape.

Shape:

    client_id:    "rudol-MSI"
    project_root: "C:\\Users\\rudol"
    environments:
      prod:
        server:   "wss://veilguard.phishield.com/ws/client"
        token:    "<prod-token>"
        user_id:  "<prod-libre-user-id>"
        enabled:  true
      local:
        server:   "ws://localhost:8809/ws/client"
        token:    "<local-token>"
        user_id:  "<local-libre-user-id>"
        enabled:  true

Env names are operator-chosen friendly identifiers.  Deep-link
``veilguard://configure?env=local&...`` lets the cloud pass the
intended name; if missing we derive from the server hostname.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger("veilguard.daemon.env")


# ── Single-env record ────────────────────────────────────────────────────


@dataclass
class EnvConfig:
    """One pairing's worth of state."""
    name:     str                                    # operator-chosen friendly name
    server:   str = ""                               # ws:// or wss://
    token:    str = ""
    user_id:  str = ""
    enabled:  bool = True

    def is_complete(self) -> bool:
        """True when we have everything needed to authenticate."""
        return bool(self.server and self.token and self.user_id)

    def to_dict(self) -> dict:
        return {
            "server":  self.server,
            "token":   self.token,
            "user_id": self.user_id,
            "enabled": bool(self.enabled),
        }


# ── Migration: old → new config shape ────────────────────────────────────


def parse_environments(config: dict) -> dict[str, EnvConfig]:
    """Return ``{name: EnvConfig}`` parsed from a config dict.

    Accepts both the new ``environments:`` map shape AND the legacy
    top-level ``server/token/user_id`` triplet (which is converted to
    a single env called "default").  When BOTH are present, the
    explicit ``environments:`` block wins; the top-level keys are
    treated as bootstrap leftovers and ignored.
    """
    out: dict[str, EnvConfig] = {}

    envs_block = config.get("environments")
    if isinstance(envs_block, dict) and envs_block:
        for name, body in envs_block.items():
            if not isinstance(body, dict):
                logger.warning(f"[env] skipping malformed env entry: {name!r}")
                continue
            out[str(name)] = EnvConfig(
                name=str(name),
                server=(body.get("server") or "").strip(),
                token=(body.get("token") or "").strip(),
                user_id=(body.get("user_id") or "").strip(),
                enabled=bool(body.get("enabled", True)),
            )

    # Legacy fallback: synthesise a "default" env from top-level keys
    # when no environments block was provided.  This keeps existing
    # ~/.veilguard/config.yaml files working on first 0.4 launch — they
    # get rewritten in the new shape on next write.
    if not out:
        server  = (config.get("server")  or "").strip()
        token   = (config.get("token")   or "").strip()
        user_id = (config.get("user_id") or "").strip()
        if server or token or user_id:
            out["default"] = EnvConfig(
                name="default", server=server, token=token,
                user_id=user_id, enabled=True,
            )

    return out


def write_environments(config: dict, envs: dict[str, EnvConfig]) -> dict:
    """Mutate `config` so it carries `envs` in the new shape.

    Strips the legacy top-level server/token/user_id keys so they
    don't drift out of sync with the environments map.  Returns the
    mutated config for chaining.
    """
    config["environments"] = {
        name: env.to_dict() for name, env in envs.items()
    }
    for legacy_key in ("server", "token", "user_id"):
        config.pop(legacy_key, None)
    return config


# ── Helpers ──────────────────────────────────────────────────────────────


_SAFE_NAME_RE = re.compile(r"[^a-z0-9._-]+", re.IGNORECASE)


def derive_env_name(server: str, *, hint: str = "") -> str:
    """Pick a friendly name for an env given its server URL.

    If `hint` is supplied (deep-link's ``env=`` param), it wins after
    sanitisation.  Otherwise we use the server hostname's first label
    — ``ws://localhost:8809/...`` → ``local``; ``wss://veilguard.phishield.com/...``
    → ``veilguard``.  Result is always lowercase, ascii-safe, ≤32 chars.
    """
    if hint:
        clean = _SAFE_NAME_RE.sub("-", hint.strip()).strip("-").lower()[:32]
        if clean:
            return clean

    try:
        parsed = urlparse(server)
        host = parsed.hostname or ""
    except Exception:
        host = ""
    # Use the first hostname label, special-casing localhost for clarity.
    if host == "localhost" or host.startswith("127."):
        return "local"
    first = (host or "default").split(".")[0].lower()
    first = _SAFE_NAME_RE.sub("-", first).strip("-")[:32] or "default"
    return first


def merge_env(
    existing: dict[str, EnvConfig], new_env: EnvConfig,
) -> dict[str, EnvConfig]:
    """Insert or update `new_env` in `existing` keyed on env.name.

    Replaces fields that the new env supplies; preserves ``enabled``
    flag of the old entry when the caller didn't override (so a user
    who disabled an env doesn't get it re-enabled by an inbound
    deep-link).
    """
    out = dict(existing)
    if new_env.name in out:
        prev = out[new_env.name]
        # Preserve disable choice unless explicitly re-enabled by the
        # incoming env (we conservatively only flip TO enabled here —
        # never silently disable).
        if not new_env.enabled and prev.enabled:
            new_env.enabled = prev.enabled
    out[new_env.name] = new_env
    return out


__all__ = [
    "EnvConfig",
    "parse_environments",
    "write_environments",
    "derive_env_name",
    "merge_env",
]
