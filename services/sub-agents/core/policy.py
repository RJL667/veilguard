"""Canonical client-tool approval policy for the sub-agents service.

[POLICY_REHOME_2026_05_31]  The original `app.policy.client_tool_policy`
(agent-runtime) was removed 2026-05-25 when approval was centralised into
sub-agents, but the matrix was never re-homed — so `core.approval` fell back
to a blanket-APPROVE (later fail-closed) degraded mode.  This module restores
a real allow / approve / deny matrix, IN the service the gate lives in.

`core.approval.gate()` calls `classify(...)` and combines the verdict with the
user's permission_level (auto / confirm / strict):
    ALLOW   → runs (toast only at strict)
    APPROVE → toast at confirm/strict; silent at auto
    DENY    → blocked unconditionally

DECISION MATRIX (conservative; reviewer-friendly — tweak the sets below):

  read-only   read_file, search_files, grep, host_file_read   → ALLOW
              …but reading a PROTECTED path (.env/.ssh/id_rsa) → APPROVE
  write       write_file, edit_file, host_file_write           → APPROVE
              …writing a PROTECTED path                        → DENY
  exec        run_command, run_cmd, run_powershell,            → APPROVE
              run_docker, run_git                              …
              …command matches a dangerous pattern             → DENY
  other client tool (in CLIENT_TOOLS, no rule)                 → APPROVE
  non-client tool (cloud / orchestration / memory)            → ALLOW

`agent_role` / `is_background` are accepted for forward-compat (server/IC
callers) but do NOT currently relax the matrix — role-based trust is handled
upstream by the `system:sandbox` short-circuit in gate().  Keep it that way
unless you deliberately want ICs to bypass approval.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from utils.tool_location import CLIENT_TOOLS  # re-exported for approval.py
from utils.safety import validate_command, is_path_safe, PROTECTED_PATHS


class Decision(Enum):
    ALLOW = "allow"
    APPROVE = "approve"
    DENY = "deny"


@dataclass
class PolicyResult:
    decision: Decision
    reason: str


# Tool buckets (kept explicit so a reviewer can see exactly what's gated).
_READONLY_TOOLS = {"read_file", "search_files", "grep", "host_file_read"}
_WRITE_TOOLS = {"write_file", "edit_file", "host_file_write"}
_EXEC_TOOLS = {"run_command", "run_cmd", "run_powershell", "run_docker", "run_git"}


def _first_str_arg(args: dict, *keys: str) -> str:
    for k in keys:
        v = (args or {}).get(k)
        if isinstance(v, str) and v:
            return v
    return ""


def _is_protected_path(path: str) -> bool:
    low = (path or "").lower()
    return any(p in low for p in PROTECTED_PATHS)


def classify(
    *,
    tool_name: str,
    tool_args: Optional[dict] = None,
    agent_id: str = "user",
    agent_role: str = "user",
    is_background: bool = False,
    **_ignored: Any,
) -> PolicyResult:
    """Classify a single tool call into ALLOW / APPROVE / DENY."""
    args = tool_args or {}
    t = tool_name

    # ── Shell / command execution ────────────────────────────────────────
    if t in _EXEC_TOOLS:
        cmd = _first_str_arg(args, "command", "cmd", "script", "powershell")
        if cmd:
            ok, why = validate_command(cmd)
            if not ok:
                return PolicyResult(Decision.DENY, f"dangerous command blocked: {why}")
        return PolicyResult(Decision.APPROVE, f"shell execution ({t}) requires approval")

    # ── File writes ──────────────────────────────────────────────────────
    if t in _WRITE_TOOLS:
        path = _first_str_arg(args, "path", "file", "filename", "filepath", "target")
        if path:
            ok, why = is_path_safe(path)
            if not ok:
                return PolicyResult(Decision.DENY, f"protected-path write blocked: {why}")
        return PolicyResult(Decision.APPROVE, f"file write ({t}) requires approval")

    # ── Read-only ────────────────────────────────────────────────────────
    if t in _READONLY_TOOLS:
        path = _first_str_arg(args, "path", "file", "filename", "filepath", "pattern", "query")
        if _is_protected_path(path):
            return PolicyResult(
                Decision.APPROVE,
                "read of protected/secret path requires approval",
            )
        return PolicyResult(Decision.ALLOW, f"read-only ({t})")

    # ── Other client tools we don't have a rule for → conservative ───────
    if t in CLIENT_TOOLS:
        return PolicyResult(
            Decision.APPROVE, f"unclassified client tool ({t}) — approval required"
        )

    # ── Non-client tools (cloud / orchestration / memory) → low risk ─────
    return PolicyResult(Decision.ALLOW, f"non-client tool ({t})")


__all__ = ["Decision", "PolicyResult", "classify", "CLIENT_TOOLS"]
