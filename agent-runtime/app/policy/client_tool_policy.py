"""Client-tool policy — the capability matrix from spec §3.8.

Decides per (agent_role, tool, origin, arg) whether the call should be:
  - ALLOW    → execute immediately
  - APPROVE  → pause for user approval (Windows toast via client-daemon)
  - DENY     → reject with audit log

Inputs:
  - agent_id (string)         — the calling agent
  - agent_role (enum)         — director | ic | consultant
  - tool_name (string)        — e.g. "run_command", "read_file"
  - tool_args (dict)          — already-parsed call arguments
  - is_background (bool)      — true if cid starts with "sub-"

Output:
  - Decision enum + optional reason string

Spec §3.8 capability matrix:

  Tool                          | Director foreground | Builder delegated | Other ICs
  ------------------------------|---------------------|-------------------|----------
  read_file, search_files, grep | ALLOW               | ALLOW (path glob) | ALLOW (read-only)
  host_file_read                | ALLOW               | APPROVE           | APPROVE
  write_file, edit_file         | ALLOW               | APPROVE           | DENY
  host_file_write               | ALLOW               | DENY              | DENY
  run_command, run_powershell   | ALLOW               | APPROVE (per call)| DENY
  run_docker, run_git           | ALLOW               | APPROVE           | DENY

Arg sanitization (spec §3.8 / §11.5.4): reject any tool arg containing
control characters (0x00-0x1f except tab/newline) BEFORE glob matching.
Memory documents two CRLF bypass incidents already; this is the third
opportunity to prevent that class of attack.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger("agent-runtime.policy")


class Decision(str, Enum):
    ALLOW = "allow"
    APPROVE = "approve"
    DENY = "deny"


@dataclass(frozen=True)
class PolicyResult:
    decision: Decision
    reason: str = ""


# Tools that touch the user's local machine via the client-daemon WS.
# Background-origin calls to these must go through the approval gate.
CLIENT_TOOLS = frozenset({
    "read_file",
    "write_file",
    "edit_file",
    "search_files",
    "grep",
    "host_file_read",
    "host_file_write",
    "run_command",
    "run_cmd",
    "run_powershell",
    "run_docker",
    "run_git",
})

# Tools that are always read-only and pose minimal risk; ALLOW for any
# background agent so long as args pass sanitization.
SAFE_READ_TOOLS = frozenset({
    "read_file",
    "search_files",
    "grep",
})

# Tools that ARE writes but go through a path allow-list when from
# Builder background; DENY for other ICs.
GATED_WRITE_TOOLS = frozenset({
    "write_file",
    "edit_file",
})

# Tools that execute arbitrary commands; ALWAYS APPROVE for background.
EXEC_TOOLS = frozenset({
    "run_command",
    "run_cmd",
    "run_powershell",
    "run_docker",
    "run_git",
})

# Host-FS access (anywhere outside workspace).  Even Builder needs
# approval; other ICs get DENY (writes) or APPROVE (reads).
HOST_TOOLS = frozenset({
    "host_file_read",
    "host_file_write",
})


# Control-char regex: 0x00-0x1f EXCEPT 0x09 (tab) and 0x0a (newline).
# Per-spec §3.8: any of these in an arg → DENY before glob match.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f]")


def _has_control_chars(value: Any) -> bool:
    """True if any string in the arg value contains forbidden control chars."""
    if isinstance(value, str):
        return bool(_CONTROL_CHAR_RE.search(value))
    if isinstance(value, dict):
        return any(_has_control_chars(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_control_chars(v) for v in value)
    return False


def classify(
    *,
    agent_id: str,
    agent_role: str,
    tool_name: str,
    tool_args: dict[str, Any],
    is_background: bool,
) -> PolicyResult:
    """Decide what to do with a tool call.

    Pseudo-logic:
      1. Sanitize args first.  CRLF / control chars → DENY immediately
         regardless of role.
      2. If tool is not a client-tool, ALLOW (it's a server-side tool;
         policy doesn't apply).
      3. If not background (foreground Director / direct user call),
         ALLOW.
      4. Map (agent_role, tool category) to Decision per the matrix.
    """

    # ── Arg sanitization ────────────────────────────────────────────────
    # Reject control chars BEFORE policy lookup so a CRLF-bypass attempt
    # can never match a permissive rule.  Memory documents this exact
    # class of bug ≥ twice.
    if _has_control_chars(tool_args):
        return PolicyResult(
            decision=Decision.DENY,
            reason="arg contains control characters (CRLF/NUL/etc.); "
                   "rejected by policy before glob match",
        )

    # ── Server-side tool? ───────────────────────────────────────────────
    if tool_name not in CLIENT_TOOLS:
        return PolicyResult(decision=Decision.ALLOW, reason="server-side tool")

    # ── Foreground (Director-driven) ────────────────────────────────────
    if not is_background:
        return PolicyResult(decision=Decision.ALLOW, reason="foreground origin")

    # ── Background — apply the role matrix ──────────────────────────────
    role = agent_role.lower()

    if tool_name in SAFE_READ_TOOLS:
        return PolicyResult(decision=Decision.ALLOW, reason="safe read tool")

    if tool_name in HOST_TOOLS:
        if tool_name == "host_file_write":
            return PolicyResult(
                decision=Decision.DENY,
                reason="host_file_write denied for all background agents",
            )
        # host_file_read
        return PolicyResult(
            decision=Decision.APPROVE,
            reason="host file read requires approval (background origin)",
        )

    if tool_name in GATED_WRITE_TOOLS:
        # Only Builder can be APPROVED; other ICs are DENIED.
        if agent_id == "builder" or role == "ic" and agent_id == "builder":
            return PolicyResult(
                decision=Decision.APPROVE,
                reason="workspace write by Builder requires per-call approval",
            )
        return PolicyResult(
            decision=Decision.DENY,
            reason=f"{agent_id!r} cannot write (only Builder may, with approval)",
        )

    if tool_name in EXEC_TOOLS:
        # Only Builder can execute commands; even then, APPROVE each call.
        if agent_id == "builder":
            return PolicyResult(
                decision=Decision.APPROVE,
                reason="shell execution requires per-call approval (Builder background)",
            )
        return PolicyResult(
            decision=Decision.DENY,
            reason=f"{agent_id!r} cannot execute shell commands (background)",
        )

    # Unknown client tool — fail closed.
    logger.warning(
        f"[policy] unknown client tool {tool_name!r}; defaulting to DENY"
    )
    return PolicyResult(
        decision=Decision.DENY,
        reason=f"unknown client tool {tool_name!r}; fail-closed default",
    )


__all__ = [
    "Decision",
    "PolicyResult",
    "CLIENT_TOOLS",
    "classify",
]
