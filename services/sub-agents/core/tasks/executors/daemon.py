"""DaemonExecutor + SandboxExecutor.

Both route tool calls through ``ClientBridge.execute_remote`` — same
mechanism, different target identity:

  DaemonExecutor  — bridge looked up by ``task.user_id`` (the calling
                    LibreChat user's MongoDB _id).  Tool runs on that
                    user's Windows machine.

  SandboxExecutor — bridge looked up by ``"system:sandbox"`` (Phase B
                    headless container).  Tool runs in the server-side
                    sandbox.  Used by the agentic.handle_tool fallback
                    when the user's daemon is offline AND the tool is
                    in SANDBOX_FALLBACK_TOOLS.

Payload shape (both):
  {"tool": str, "args": dict}

Why two executors and not one with a parameter:
  - Permission gating differs (sandbox auto-approves; user daemon
    goes through the approval gate).  Cleaner to have two named
    executors than to branch inside one.
  - LLM-visible task introspection can show "executor: daemon" vs
    "executor: sandbox" so the user knows which machine ran the call.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import TaskExecutor, ExecutorResult
from ..model import Task, TaskStatus

logger = logging.getLogger("veilguard.tasks.executors.daemon")


# ── Shared core ─────────────────────────────────────────────────────────


async def _execute_via_bridge(
    task: Task, *, target_user_id: str, label: str,
) -> ExecutorResult:
    """The actual WS dispatch.  Returns ExecutorResult."""
    from core.client_bridge import get_bridge

    payload = task.payload or {}
    tool = payload.get("tool") or ""
    args = payload.get("args") or {}
    if not tool:
        return ExecutorResult(
            status=TaskStatus.FAILED,
            error=(
                "DaemonExecutor: task.payload missing 'tool' key; "
                "cannot dispatch"
            ),
        )

    bridge = get_bridge(target_user_id)
    if bridge is None or not bridge.connected:
        return ExecutorResult(
            status=TaskStatus.FAILED,
            error=(
                f"{label}: no connected bridge for "
                f"user_id={target_user_id[:12]!r}"
            ),
        )

    try:
        result = await bridge.execute_remote(tool, args)
    except Exception as e:
        logger.exception(f"[{label}] execute_remote raised: {e}")
        return ExecutorResult(status=TaskStatus.FAILED, error=str(e))

    # bridge.execute_remote returns a string; "Error:" prefix means
    # tool-side failure (not transport failure).  Surface as FAILED so
    # tree-walking tools / cancel cascade can react.
    if isinstance(result, str) and result.startswith("Error:"):
        return ExecutorResult(
            status=TaskStatus.FAILED,
            error=result[:8000],
            result=result[:1000],
        )
    out = result if isinstance(result, str) else str(result)
    return ExecutorResult(
        status=TaskStatus.COMPLETED,
        result=out[:8000],
    )


async def _cancel_via_bridge(task: Task, *, target_user_id: str) -> bool:
    """Send ``cancel_task`` to the target daemon."""
    from core.client_bridge import get_bridge
    bridge = get_bridge(target_user_id)
    if bridge is None or not bridge.connected:
        return False
    try:
        resp = await bridge.send_command(
            "cancel_task", {"task_id": task.task_id}, timeout=5.0,
        )
    except Exception as e:
        logger.warning(f"[cancel] bridge.send_command failed: {e}")
        return False
    if isinstance(resp, dict):
        return bool(resp.get("cancelled"))
    if isinstance(resp, str):
        try:
            import json
            parsed = json.loads(resp)
            return bool((parsed or {}).get("cancelled"))
        except Exception:
            pass
    return False


# ── Concrete executors ──────────────────────────────────────────────────


class DaemonExecutor(TaskExecutor):
    """Routes to the calling user's own Windows daemon."""

    name = "daemon"

    async def start(self, task: Task) -> ExecutorResult:
        return await _execute_via_bridge(
            task, target_user_id=task.user_id, label="DaemonExecutor",
        )

    async def cancel(self, task: Task) -> bool:
        return await _cancel_via_bridge(task, target_user_id=task.user_id)

    async def progress(self, task: Task) -> Optional[dict]:
        from core.client_bridge import get_bridge
        bridge = get_bridge(task.user_id)
        if bridge is None:
            return {"bridge": "absent"}
        return {
            "bridge": "connected" if bridge.connected else "disconnected",
            "client_id": bridge.client_id,
            "platform": bridge.platform,
            "pending": len(bridge.pending or {}),
        }


class SandboxExecutor(TaskExecutor):
    """Routes to the server-side veilguard-sandbox container."""

    name = "sandbox"

    SANDBOX_USER_ID = "system:sandbox"

    async def start(self, task: Task) -> ExecutorResult:
        return await _execute_via_bridge(
            task, target_user_id=self.SANDBOX_USER_ID,
            label="SandboxExecutor",
        )

    async def cancel(self, task: Task) -> bool:
        return await _cancel_via_bridge(
            task, target_user_id=self.SANDBOX_USER_ID,
        )

    async def progress(self, task: Task) -> Optional[dict]:
        from core.client_bridge import get_bridge
        bridge = get_bridge(self.SANDBOX_USER_ID)
        if bridge is None:
            return {"bridge": "absent"}
        return {
            "bridge": "connected" if bridge.connected else "disconnected",
            "client_id": bridge.client_id,
            "platform": bridge.platform,
            "pending": len(bridge.pending or {}),
        }


__all__ = ["DaemonExecutor", "SandboxExecutor"]
