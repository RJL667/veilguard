"""Centralised configuration via environment variables.

All env-var reads live here so the rest of the app can import typed
constants and tests can monkey-patch one place.

Default values match docker-compose.yml so a bare `docker compose up`
works without an .env file (for dev).  Production overrides via .env.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("agent-runtime.config")


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key, "").lower()
    if not raw:
        return default
    return raw in ("true", "1", "yes", "on")


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# ── Service identity + binding ───────────────────────────────────────────
SERVICE_NAME = "agent-runtime"
PORT = _env_int("AGENT_RUNTIME_PORT", 5000)
LOG_LEVEL = _env("LOG_LEVEL", "INFO")


# ── Anthropic API ────────────────────────────────────────────────────────
# The SDK reads ANTHROPIC_API_KEY from env directly; we surface it for
# explicit failure on missing key at startup rather than letting the SDK
# emit cryptic auth errors on first call.
ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY", "")
ANTHROPIC_API_URL = _env("ANTHROPIC_API_URL", "https://api.anthropic.com")


# ── TCMM integration ─────────────────────────────────────────────────────
# TCMM runs on the host (port 8811); reach it via host.docker.internal
# inside the container (same pattern as pii-proxy + admin-dashboard).
TCMM_ENABLED = _env_bool("TCMM_ENABLED", True)
TCMM_URL = _env("TCMM_URL", "http://host.docker.internal:8811")
TCMM_RENDER_TIMEOUT_S = _env_int("TCMM_RENDER_TIMEOUT_S", 30)


# ── Sub-agents service (MCP tool surface) ────────────────────────────────
SUB_AGENTS_URL = _env("SUB_AGENTS_URL", "http://host.docker.internal:8809")
VEILGUARD_INTERNAL_SECRET = _env("VEILGUARD_INTERNAL_SECRET", "")


# ── Persona + Constitution paths ─────────────────────────────────────────
# These live in the agents/ directory next to the multi-agent spec.
# Mounted into the container as /app/agents/.
AGENTS_DIR = Path(_env("AGENTS_DIR", "/app/agents"))
CONSTITUTION_PATH = AGENTS_DIR / "CONSTITUTION.md"


# ── Decision-ledger Lance directory ──────────────────────────────────────
# Same on-disk store as TCMM + pii_audit.  agent-runtime writes the
# following new tables here:
#   agent_tasks, task_comments, task_proposals, proposal_outcomes,
#   org_memory, client_tool_approvals, client_tool_bypass
#
# Discipline: this process MUST run as the same OS user that owns the
# Lance dir (typically rudol on the VM).  Mixing root + non-root writes
# silently breaks recall (see memory: architecture_lance_index_perms).
LEDGER_DB_PATH = Path(_env("LEDGER_DB_PATH", "/tcmm-data/veilguard/tcmm.db"))


# ── Audit DB (shared with pii-proxy) ─────────────────────────────────────
# We write to the SAME pii_audit table the proxy writes to.  This keeps
# cost roll-up queries single-table.  Phase 2 adds task_id column (via
# the existing forward-compat `extra` JSON field first; promoted later).
AUDIT_DB_PATH = Path(_env(
    "VEILGUARD_AUDIT_DB_PATH",
    "/tcmm-data/veilguard/tcmm.db",
))


# ── Prompt cache configuration ───────────────────────────────────────────
# 1h TTL pays off after ~0.83 extra hits per spec §3.7.0 breakeven math.
# Default ON for sub-agent calls; reactive Director may override per-call.
CACHE_TTL_DEFAULT = _env("CACHE_TTL_DEFAULT", "1h")


# ── Approval gate ────────────────────────────────────────────────────────
APPROVAL_TIMEOUT_S = _env_int("APPROVAL_TIMEOUT_S", 120)
APPROVAL_FAIL_CLOSED = _env_bool("APPROVAL_FAIL_CLOSED", True)


# ── Validation at import time ────────────────────────────────────────────
def validate() -> list[str]:
    """Return a list of fatal config errors (empty = OK).

    Called by main.py at startup so missing config fails fast with a
    readable message rather than crashing later on a real request.
    """
    errors: list[str] = []
    # ANTHROPIC_API_KEY was required by the legacy SDK path.  BACKEND=live
    # uses TCMM's AnthropicGenerationAdapter (OAuth bearer via the
    # long-lived setup-token written to ~/.claude/.credentials.json),
    # so the API key is no longer needed in the default path.  We only
    # warn now — not a fatal error — so /health stays green when the
    # bearer is in place.
    backend_name = os.environ.get("BACKEND", "live").lower()
    if backend_name == "live":
        # Live path uses OAuth bearer; API key not needed.
        if not ANTHROPIC_API_KEY:
            logger.info(
                "[config] ANTHROPIC_API_KEY unset — fine for BACKEND=live "
                "(OAuth bearer path)."
            )
    elif not ANTHROPIC_API_KEY:
        errors.append(
            f"ANTHROPIC_API_KEY is unset and BACKEND={backend_name!r} "
            "requires it"
        )
    if not AGENTS_DIR.exists():
        errors.append(
            f"AGENTS_DIR does not exist: {AGENTS_DIR} "
            "(should be the dir containing persona *.md files + CONSTITUTION.md)"
        )
    return errors
