"""Tool-call approval gate for the LibreChat / chat-driven path.

Sits between `agentic.execute_tool` and `bridge.execute_remote`. For every
client-tool invocation:

  1. Classify the tool via the static policy matrix
     (`agent-runtime/app/policy/client_tool_policy.py` — same matrix
     the Director / IC sub-agents use, deliberately shared so the
     classification is consistent across paths).
  2. Resolve the user's effective permission_level
     (conv override → user default → DEFAULT_LEVEL).
  3. Combine: DENY always blocks; ALLOW always runs;
     APPROVE behavior depends on level:
        auto    → silently run
        confirm → round-trip through the bridge for user approval
        strict  → round-trip even for ALLOW (Phase A: only APPROVE)
  4. If approval needed → bridge.request_approval(tool, args, conv_id, ...)
     pops a toast on the user's machine; await within timeout.
  5. Return (proceed: bool, reason: str).

The gate is async because the WS round-trip is async.  Callers should
wrap the return value in an `if not proceed: return f"Denied: {reason}"`
short-circuit before the actual tool execution.

Audit:
  Every decision (allow / deny / approve_granted / approve_denied / timeout)
  writes a row to the `pii_audit` table via `app.audit_db` (same writer
  the proxy uses) with direction='APPROVAL' so the dashboard can show
  the approval history alongside LLM turns.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

# Reach agent-runtime's policy module via sys.path probe — same shape as
# the chat_agent_handler's `/agent` mount discovery.  Inside the
# sub-agents container `/agent-runtime` is mounted as a sibling of
# `/app`; on host it's `Documents/veilguard/agent-runtime/`.
_HERE = Path(__file__).resolve()
for _candidate in [*_HERE.parents, Path("/")]:
    _ar = _candidate / "agent-runtime"
    if (_ar / "app" / "policy" / "client_tool_policy.py").is_file():
        if str(_ar) not in sys.path:
            sys.path.insert(0, str(_ar))
        break

try:
    from app.policy.client_tool_policy import (
        classify as _classify_policy,
        Decision as _PolicyDecision,
        CLIENT_TOOLS as _POLICY_CLIENT_TOOLS,
    )
    _POLICY_OK = True
except Exception as _e:  # pragma: no cover
    _POLICY_OK = False
    _classify_policy = None
    _PolicyDecision = None
    _POLICY_CLIENT_TOOLS: set = set()

from .client_settings import (
    ClientSettingsStore,
    PERMISSION_LEVELS,
    DEFAULT_LEVEL,
    DEFAULT_TIMEOUT_S,
)

logger = logging.getLogger("veilguard.approval")


# ── Gate result ──────────────────────────────────────────────────────────


class GateResult:
    __slots__ = ("proceed", "reason", "decision", "level", "request_id")

    def __init__(self, *, proceed: bool, reason: str,
                 decision: str, level: str, request_id: str = ""):
        self.proceed = proceed
        self.reason = reason
        self.decision = decision    # "allow" | "deny" | "approved" | "denied" | "timeout"
        self.level = level
        self.request_id = request_id

    def __repr__(self) -> str:
        return (f"GateResult(proceed={self.proceed}, decision={self.decision!r}, "
                f"level={self.level!r}, reason={self.reason!r})")


# ── Entrypoint ───────────────────────────────────────────────────────────


async def gate(
    *,
    tool: str,
    args: dict[str, Any],
    user_id: str,
    conv_id: str = "",
    bridge=None,                 # ClientBridge or None (no daemon → cloud-only)
    agent_id: str = "user",
    is_background: bool = False,
) -> GateResult:
    """Run the approval gate for one tool call.

    Returns `GateResult` carrying:
      - .proceed   : whether the caller should actually run the tool
      - .reason    : human-readable explanation (logged + surfaced to LLM
                     on deny)
      - .decision  : machine token for the audit row
      - .level     : effective permission_level at decision time
      - .request_id: WS request id if a daemon round-trip happened

    Phase A semantics:
      level=auto    → APPROVE collapses to ALLOW (no toast)
      level=confirm → APPROVE → WS request_approval → wait
      level=strict  → APPROVE → WS request_approval; ALLOW falls through

    The bridge is optional — if None, server-only execution path.  In
    that case APPROVE collapses to DENY (no way to ask the user) UNLESS
    the user-id is the sandbox identity (`system:sandbox`), which we
    treat as pre-approved for everything classified ALLOW or APPROVE.
    """
    request_id = uuid.uuid4().hex[:12]

    # ── Sandbox short-circuit ────────────────────────────────────────────
    # The server-side sandbox daemon (Phase B) runs unattended, so it
    # gets its own decision path — no toast.  Server agents calling the
    # sandbox already passed their own policy check (the Director hook).
    if user_id == "system:sandbox":
        return GateResult(
            proceed=True, reason="sandbox identity (pre-approved)",
            decision="allow", level="auto", request_id=request_id,
        )

    # ── Static policy classification ─────────────────────────────────────
    if _POLICY_OK and _classify_policy is not None:
        try:
            result = _classify_policy(
                agent_id=agent_id,
                agent_role=("user" if not is_background else "ic"),
                tool_name=tool,
                tool_args=args,
                is_background=is_background,
            )
            policy_decision = result.decision.value
            policy_reason = result.reason
        except Exception as e:
            logger.warning(f"[approval] policy classify failed: {e}")
            policy_decision = "approve"
            policy_reason = f"policy fault, defaulting to APPROVE: {e}"
    else:
        # If we can't load the policy module at all, fail closed for
        # CLIENT_TOOLS we don't recognise as server-only.
        logger.warning(
            "[approval] policy module not importable — degrading to "
            "blanket APPROVE for daemon-routed tools"
        )
        policy_decision = "approve"
        policy_reason = "policy module unavailable (fallback APPROVE)"

    # Hard DENY is final.
    if policy_decision == "deny":
        return GateResult(
            proceed=False, reason=f"policy: {policy_reason}",
            decision="deny", level="-", request_id=request_id,
        )

    # ── Resolve effective level ──────────────────────────────────────────
    try:
        level, timeout_s = ClientSettingsStore.get().get_effective(
            user_id, conv_id,
        )
    except Exception as e:
        logger.warning(f"[approval] settings lookup failed: {e}")
        level, timeout_s = DEFAULT_LEVEL, DEFAULT_TIMEOUT_S

    needs_approval = _needs_user_approval(policy_decision, level)
    if not needs_approval:
        return GateResult(
            proceed=True,
            reason=f"policy {policy_decision} @ level={level}",
            decision="allow", level=level, request_id=request_id,
        )

    # ── Round-trip through the bridge ────────────────────────────────────
    if bridge is None or not getattr(bridge, "connected", False):
        # No way to ask.  Phase A: fail closed.
        return GateResult(
            proceed=False,
            reason=(
                f"approval needed (policy={policy_decision} level={level}) "
                "but no client daemon connected to ask"
            ),
            decision="deny", level=level, request_id=request_id,
        )

    # ── Phase B-3: shadow Task row for the approval ──────────────────────
    # The unified store gets a row in AWAITING_USER status so
    # list_my_tasks shows "approval pending" alongside the running
    # tool_call.  We mark it terminal at the end.  Failure to write
    # is non-fatal — the legacy audit row still happens.
    _unified_task_id = ""
    try:
        from core.tasks import Task as _UTask, TaskKind as _UKind, TaskDispatcher as _UDisp, TaskStatus as _UStat
        approval_task = _UTask.new(
            kind=_UKind.APPROVAL,
            executor="approval",
            user_id=user_id,
            conv_id=conv_id,
            title=f"approve: {tool}",
            description=f"policy={policy_decision} level={level}",
            payload={"tool": tool, "args_keys": list(args.keys()),
                     "policy_decision": policy_decision, "level": level,
                     "timeout_s": timeout_s},
            task_id=f"appr-{request_id}",
        )
        approval_task.status = _UStat.AWAITING_USER.value
        approval_task.started_at = time.time()
        _UDisp.get().record_external(approval_task)
        _unified_task_id = approval_task.task_id
    except Exception as _e:
        logger.debug(f"[approval] shadow row write failed: {_e}")

    started = time.time()
    try:
        resp = await asyncio.wait_for(
            bridge.request_approval(
                tool=tool, args=args, conv_id=conv_id, agent_id=agent_id,
                request_id=request_id, policy_decision=policy_decision,
                policy_reason=policy_reason, level=level,
                timeout_s=timeout_s,
            ),
            timeout=timeout_s + 5,   # +5s slack for WS round-trip itself
        )
    except asyncio.TimeoutError:
        elapsed = int(time.time() - started)
        logger.warning(
            f"[approval] timeout id={request_id} tool={tool!r} "
            f"after {elapsed}s (level={level})"
        )
        _audit(
            user_id, conv_id, request_id, tool, args, level,
            decision="timeout",
            reason=f"user did not respond in {timeout_s}s",
        )
        _mark_unified_terminal(_unified_task_id, "timeout",
                               f"toast timeout {timeout_s}s")
        return GateResult(
            proceed=False, reason=f"approval timed out ({timeout_s}s)",
            decision="timeout", level=level, request_id=request_id,
        )

    approved = bool(resp.get("approved")) if isinstance(resp, dict) else False
    user_reason = (resp.get("reason") or "") if isinstance(resp, dict) else ""
    persist_for_conv = (
        bool(resp.get("persist_for_conv")) if isinstance(resp, dict) else False
    )
    decision = "approved" if approved else "denied"

    # User clicked "Always for this conv" on the toast → lock in an
    # auto-approve override for this conversation so subsequent tool
    # calls run silently.  We deliberately use level="auto" (NOT
    # "confirm") because the user's intent was "stop asking me for
    # this conversation" — confirm would keep popping toasts on
    # APPROVE-classified tools.
    if approved and persist_for_conv and conv_id:
        try:
            ClientSettingsStore.get().set_conv_override(
                user_id, conv_id, "auto",
            )
            logger.info(
                f"[approval] user opted-in 'always for conv' — wrote "
                f"override user={user_id[:8]} conv={conv_id[:8]} level=auto"
            )
            # Reflect in the decision so the audit row carries the
            # provenance ("user_clicked_always → conv override written").
            decision = "approved_persisted"
            user_reason = (user_reason or "user_clicked_always_for_conv") + \
                          " (conv override → auto persisted)"
        except Exception as e:
            logger.warning(
                f"[approval] persist_for_conv requested but write "
                f"failed: {e}; approval still granted for this call"
            )

    _audit(
        user_id, conv_id, request_id, tool, args, level,
        decision=decision, reason=user_reason or policy_reason,
    )
    _mark_unified_terminal(
        _unified_task_id, decision,
        user_reason or policy_reason,
    )

    return GateResult(
        proceed=approved,
        reason=(user_reason or ("user approved" if approved else "user denied")),
        decision=decision, level=level, request_id=request_id,
    )


# ── Helpers ──────────────────────────────────────────────────────────────


def _needs_user_approval(policy_decision: str, level: str) -> bool:
    """Map (policy_decision, level) → whether a toast must be raised."""
    if policy_decision == "allow":
        return level == "strict"
    if policy_decision == "approve":
        return level in ("confirm", "strict")
    # deny is already handled before this is called
    return False


def _audit(
    user_id: str, conv_id: str, request_id: str, tool: str,
    args: dict, level: str, *, decision: str, reason: str,
) -> None:
    """Best-effort write to the same pii_audit table the LLM turns use.

    Direction column gets a fresh value APPROVAL so the dashboard can
    distinguish these from TO_LLM / FROM_LLM rows.  Format mirrors what
    chat_agent_handler writes:
      content = JSON({tool, args, level, decision, request_id, ...})
    """
    try:
        # Reach the pii-proxy's audit_db sibling.  In the sub-agents
        # container it's reachable as `app.audit_db` only if the
        # /app/app/ tree is mounted — otherwise we fall back to a log
        # line (which is fine, the audit table is for the dashboard).
        import json
        try:
            from app import audit_db as _audit_db
            _audit_db.record(
                direction="APPROVAL",
                conversation_id=conv_id or "",
                user_id=user_id or "",
                model="",
                stream=False,
                content=json.dumps({
                    "request_id": request_id,
                    "tool": tool,
                    "args_keys": list((args or {}).keys()),
                    "level": level,
                    "decision": decision,
                    "reason": reason,
                }, ensure_ascii=False),
                extra={"path": "approval_gate"},
            )
        except Exception:
            logger.info(
                f"[approval] audit user={user_id[:8] if user_id else '-'} "
                f"conv={conv_id[:8] if conv_id else '-'} req={request_id} "
                f"tool={tool!r} level={level} decision={decision} "
                f"reason={reason!r}"
            )
    except Exception as e:   # never let audit failure propagate
        logger.debug(f"[approval] audit write failed: {e}")


def _mark_unified_terminal(task_id: str, decision: str, reason: str) -> None:
    """Mark the shadow APPROVAL Task row terminal.  Best-effort."""
    if not task_id:
        return
    try:
        from core.tasks import TaskDispatcher, TaskStatus
        # Map gate decision strings to TaskStatus.
        smap = {
            "approved":           TaskStatus.COMPLETED,
            "approved_persisted": TaskStatus.COMPLETED,
            "denied":             TaskStatus.CANCELLED,
            "timeout":            TaskStatus.FAILED,
            "deny":               TaskStatus.CANCELLED,
        }
        st = smap.get(decision, TaskStatus.FAILED)
        TaskDispatcher.get()._store.mark_status(
            task_id, st,
            result=reason if st == TaskStatus.COMPLETED else "",
            error=reason if st != TaskStatus.COMPLETED else "",
        )
    except Exception as e:
        logger.debug(f"[approval] unified terminal mark failed: {e}")


__all__ = ["GateResult", "gate"]
