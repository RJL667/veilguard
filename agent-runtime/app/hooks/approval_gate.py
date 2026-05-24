"""PreToolUse hook implementing the spec §3.8 approval gate.

The Claude Agent SDK fires `PreToolUse` BEFORE a tool actually executes.
We use it as the choke point that:
  1. Classifies the call via policy/client_tool_policy.py
  2. ALLOW → return immediately, SDK proceeds
  3. DENY → return permissionDecision="deny" + reason; SDK skips the call
  4. APPROVE → call client-daemon over WebSocket asking for user approval
     (Windows toast), await with timeout, then return allow/deny based
     on user's click

SDK hook contract (from claude-agent-sdk Python docs):

    async def my_hook(input_data: dict, tool_use_id: str, context: Any)
            -> dict:
        # input_data:
        #   hook_event_name: "PreToolUse"
        #   tool_name: str
        #   tool_input: dict
        # Return dict containing:
        #   {"hookSpecificOutput": {
        #       "hookEventName": "PreToolUse",
        #       "permissionDecision": "allow" | "deny" | "ask",
        #       "permissionDecisionReason": str,
        #   }}

We wire this hook with `HookMatcher(matcher="^mcp__client_daemon__", ...)`
so it ONLY fires for client-daemon-prefixed tools.  Server-side tools
(sub-agents service tools, recall, observe, etc.) skip the hook entirely.

The hook reads TenantContext via contextvars (the SDK doesn't propagate
per-request data through hooks), so the caller MUST establish context
via tenant.set_tenant_context() before invoking query().

Audit logging:
  Every decision is logged to the client_tool_approvals Lance table
  (see ledger/approvals.py).  This is the audit trail for "what was
  asked, what did the user say, when, why."
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..middleware import tenant
from ..policy.client_tool_policy import classify, Decision
from ..config import APPROVAL_TIMEOUT_S, APPROVAL_FAIL_CLOSED

logger = logging.getLogger("agent-runtime.hooks.approval")


# Client-daemon tool prefix the SDK's HookMatcher should target.
# MCP tool names in the SDK are exposed as `mcp__<server>__<tool>`.
CLIENT_DAEMON_PREFIX = "mcp__client_daemon__"


def _strip_prefix(tool_name: str) -> str:
    """Strip the MCP prefix so we feed the bare tool name to the policy."""
    if tool_name.startswith(CLIENT_DAEMON_PREFIX):
        return tool_name[len(CLIENT_DAEMON_PREFIX):]
    return tool_name


async def _request_user_approval(
    *,
    tool: str,
    args: dict[str, Any],
    agent_id: str,
    user_id: str,
    timeout_s: int,
) -> tuple[bool, str]:
    """Send a request_approval message to the client-daemon over WS and
    await the user's decision.

    Returns (approved, reason).

    Implementation note: the client-daemon side of this round-trip is
    in services/client-daemon/veilguard_client.py — adds a WS handler
    for `request_approval` that pops a Windows toast (winotify or
    winrt.windows.ui.notifications) and waits for click.

    For Phase 0.3 we ship a STUB that auto-denies if the daemon doesn't
    support the request_approval method (capability handshake per
    §0.4).  Real WS integration lands in Phase 0.3 implementation.
    """
    # Phase 0 stub.  Real WS round-trip in client-daemon update.
    # Falling back to fail-closed default per spec §0.4 + §3.8.
    if APPROVAL_FAIL_CLOSED:
        logger.warning(
            f"[approval] STUB request_approval for tool={tool!r} "
            f"agent={agent_id!r}; daemon WS integration not yet shipped, "
            "fail-closed default = DENY"
        )
        return False, "approval gate STUB; daemon integration pending"
    return True, "fail-open default (UNSAFE — only for dev)"


async def approval_gate_hook(
    input_data: dict[str, Any],
    tool_use_id: str,
    context: Any,
) -> dict[str, Any]:
    """PreToolUse hook.  Returns a permissionDecision dict for the SDK.

    Per SDK conventions, returning {} = allow with no comment.  Returning
    {"hookSpecificOutput": {...}} with permissionDecision="deny" blocks
    the tool call.  permissionDecision="ask" pauses the loop until the
    SDK's permission_prompt is satisfied (we use this for the toast).
    """
    if input_data.get("hook_event_name") != "PreToolUse":
        return {}

    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {}) or {}

    # Read tenant context from contextvars.  If unset, fail closed
    # (better than allowing a tool call without provenance).
    ctx = tenant.current()
    if ctx is None:
        logger.error(
            f"[approval] tool={tool_name!r}: NO TenantContext; "
            "fail-closed default = DENY"
        )
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    "agent-runtime: missing TenantContext "
                    "(programmer error; fail-closed)"
                ),
            }
        }

    bare_tool = _strip_prefix(tool_name)
    decision = classify(
        agent_id=ctx.agent_id,
        agent_role=_role_from_agent_id(ctx.agent_id),
        tool_name=bare_tool,
        tool_args=tool_input,
        is_background=ctx.is_background or tenant.is_subagent_cid(ctx.conversation_id),
    )

    # Always audit the decision regardless of outcome.  Pluggable: in
    # Phase 0.3 this writes to client_tool_approvals Lance table.
    _audit_decision(
        ctx=ctx,
        tool=tool_name,
        args=tool_input,
        decision=decision.decision.value,
        reason=decision.reason,
    )

    if decision.decision == Decision.ALLOW:
        return {}

    if decision.decision == Decision.DENY:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"policy DENY: {decision.reason}",
            }
        }

    # APPROVE → ask the user, awaiting up to APPROVAL_TIMEOUT_S.
    try:
        approved, reason = await asyncio.wait_for(
            _request_user_approval(
                tool=bare_tool,
                args=tool_input,
                agent_id=ctx.agent_id,
                user_id=ctx.user_id,
                timeout_s=APPROVAL_TIMEOUT_S,
            ),
            timeout=APPROVAL_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning(
            f"[approval] timeout after {APPROVAL_TIMEOUT_S}s for "
            f"tool={tool_name!r} agent={ctx.agent_id!r}; fail-closed = DENY"
        )
        _audit_decision(
            ctx=ctx,
            tool=tool_name,
            args=tool_input,
            decision="deny",
            reason=f"approval_timeout after {APPROVAL_TIMEOUT_S}s",
        )
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "user approval timed out",
            }
        }

    if approved:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "permissionDecisionReason": f"user approved: {reason}",
            }
        }
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": f"user denied: {reason}",
        }
    }


def _role_from_agent_id(agent_id: str) -> str:
    """Quick lookup so the hook doesn't need the full PersonaRegistry.

    The PersonaRegistry would be the clean answer but the hook signature
    is fixed by the SDK; we infer role from the agent_id naming convention
    (director → director, *-claim/*-prose/researcher/builder → ic, else
    consultant).  Phase 1 wires the registry via contextvars too.
    """
    if agent_id == "director":
        return "director"
    if agent_id in ("researcher", "builder", "critic-claim", "critic-prose"):
        return "ic"
    return "consultant"


def _audit_decision(
    *,
    ctx: tenant.TenantContext,
    tool: str,
    args: dict[str, Any],
    decision: str,
    reason: str,
) -> None:
    """Log the decision.  Phase 0.3 wires this to client_tool_approvals
    Lance table; Phase 0 just logs.
    """
    logger.info(
        f"[approval] decision={decision} tool={tool!r} "
        f"agent={ctx.agent_id!r} user={ctx.user_id!r} "
        f"reason={reason!r}"
    )


__all__ = [
    "CLIENT_DAEMON_PREFIX",
    "approval_gate_hook",
]
