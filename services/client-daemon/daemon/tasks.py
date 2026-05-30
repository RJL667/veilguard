"""Daemon-side task tracking.

Replaces the bare ``active_tasks = set()`` that veilguard_client.py used
purely for asyncio GC reference holding.  The new ``TaskRegistry`` keeps
structured state per task so:

  - The tray UI can list "running / completed / pending approval"
  - LLMs can query via the new MCP tools (`list_my_tasks`, `task_status`)
  - The user can cancel an in-flight tool call from the tray
  - Crash recovery has something to inspect

Schema kept small on purpose — tasks are ephemeral, not a ledger.  When
the daemon restarts, in-flight tasks become orphans; completed tasks
beyond ``COMPLETED_KEEP_LAST`` get pruned to keep memory bounded.

Thread safety: a single asyncio loop drives the daemon, so dict
mutations from coroutine code are safe without locks.  Outside-loop
readers (the tray Tk thread polling the registry) get an atomic
``snapshot()`` that copies the dicts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("veilguard.daemon.tasks")


# ── Constants ────────────────────────────────────────────────────────────


COMPLETED_KEEP_LAST = 50    # tray "completed" list cap (LRU-by-finished_at)
PENDING_KEEP_LAST   = 20    # approval queue cap (rare to exceed even 5)


class TaskKind(str, Enum):
    TOOL_CALL = "tool_call"           # execute_tool from cloud
    APPROVAL  = "approval"            # request_approval pending user click


class TaskStatus(str, Enum):
    RUNNING        = "running"
    COMPLETED      = "completed"
    FAILED         = "failed"
    CANCELLED      = "cancelled"
    AWAITING_USER  = "awaiting_user"  # toast on screen, no click yet


# ── Record ───────────────────────────────────────────────────────────────


@dataclass
class TaskState:
    task_id:        str
    kind:           str           # TaskKind value
    status:         str           # TaskStatus value
    tool:           str = ""      # tool name (for TOOL_CALL + APPROVAL)
    args_keys:      list[str] = field(default_factory=list)
    args_preview:   str = ""      # first ~80 chars of arg payload, for UI
    conv_id:        str = ""      # for grouping by conversation
    agent_id:       str = "user"
    created_at:     float = field(default_factory=time.time)
    started_at:     Optional[float] = None
    finished_at:    Optional[float] = None
    result_preview: str = ""      # truncated result string for UI
    error:          str = ""
    # The actual asyncio.Task / Future the registry holds onto so it
    # can cancel it later.  Excluded from to_dict() / snapshots — UI
    # consumers should never need the live handle.
    _future:        Optional[Any] = None

    def to_dict(self) -> dict:
        """JSON-safe shape for tray / MCP / WS responses.

        IMPORTANT: we DO NOT use ``dataclasses.asdict()`` here because
        asdict() recursively deepcopy()'s every field — including
        ``_future`` (which holds the live ``asyncio.Task`` reference
        for tool calls in the running bucket).  asyncio.Task isn't
        picklable, so the deepcopy raises
        ``cannot pickle '_asyncio.Task' object`` and the caller (the
        tray viewer's polling refresh) silently dies.  Manually
        building the dict avoids the deepcopy entirely.
        """
        return {
            "task_id":        self.task_id,
            "kind":           self.kind,
            "status":         self.status,
            "tool":           self.tool,
            "args_keys":      list(self.args_keys),
            "args_preview":   self.args_preview,
            "conv_id":        self.conv_id,
            "agent_id":       self.agent_id,
            "created_at":     self.created_at,
            "started_at":     self.started_at,
            "finished_at":    self.finished_at,
            "result_preview": self.result_preview,
            "error":          self.error,
        }


# ── Registry ─────────────────────────────────────────────────────────────


class TaskRegistry:
    """In-memory store of daemon-side tasks.

    Single instance, accessed via ``get()``.  Thread-safety: only mutated
    from the asyncio loop; readers from other threads (the Tk tray) call
    ``snapshot()`` which dict-copies under a short lock.
    """

    _instance: Optional["TaskRegistry"] = None

    def __init__(self):
        self._running:   dict[str, TaskState] = {}
        self._pending:   dict[str, TaskState] = {}     # awaiting user click
        self._completed: dict[str, TaskState] = {}     # capped LRU
        self._lock = threading.Lock()
        self._listeners: list[Any] = []   # callables fired on state change

    @classmethod
    def get(cls) -> "TaskRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Event hooks ──────────────────────────────────────────────────────
    # Tray code subscribes to repaint its icon / menu when tasks change.

    def add_listener(self, cb) -> None:
        self._listeners.append(cb)

    def _notify(self) -> None:
        for cb in list(self._listeners):
            try:
                cb()
            except Exception as e:
                logger.warning(f"[tasks] listener {cb!r} raised: {e}")

    # ── Tool-call lifecycle ──────────────────────────────────────────────

    def start_tool(
        self, task_id: str, tool: str, args: dict, *,
        conv_id: str = "", agent_id: str = "user",
        future: Optional[Any] = None,
    ) -> TaskState:
        rec = TaskState(
            task_id=task_id, kind=TaskKind.TOOL_CALL.value,
            status=TaskStatus.RUNNING.value, tool=tool,
            args_keys=list((args or {}).keys()),
            args_preview=_preview_args(args),
            conv_id=conv_id, agent_id=agent_id,
            started_at=time.time(), _future=future,
        )
        with self._lock:
            self._running[task_id] = rec
        self._notify()
        return rec

    def finish_tool(
        self, task_id: str, *, result: str = "", error: str = "",
        cancelled: bool = False,
    ) -> Optional[TaskState]:
        with self._lock:
            rec = self._running.pop(task_id, None)
            if rec is None:
                return None
            rec.finished_at = time.time()
            rec.result_preview = (result or "")[:200]
            rec.error = error
            if cancelled:
                rec.status = TaskStatus.CANCELLED.value
            elif error:
                rec.status = TaskStatus.FAILED.value
            else:
                rec.status = TaskStatus.COMPLETED.value
            self._completed[task_id] = rec
            self._evict_completed()
        self._notify()
        return rec

    # ── Approval lifecycle (Phase C) ─────────────────────────────────────

    def start_approval(
        self, request_id: str, tool: str, args: dict, *,
        conv_id: str = "", agent_id: str = "user",
        policy_decision: str = "", level: str = "",
        timeout_s: int = 60,
    ) -> TaskState:
        rec = TaskState(
            task_id=request_id, kind=TaskKind.APPROVAL.value,
            status=TaskStatus.AWAITING_USER.value, tool=tool,
            args_keys=list((args or {}).keys()),
            args_preview=_preview_args(args),
            conv_id=conv_id, agent_id=agent_id,
            started_at=time.time(),
            # Cram extra context into result_preview for the tray menu —
            # cheap and avoids growing the dataclass.
            result_preview=(
                f"policy={policy_decision} level={level} timeout={timeout_s}s"
            ),
        )
        with self._lock:
            self._pending[request_id] = rec
            self._evict_pending()
        self._notify()
        return rec

    def finish_approval(
        self, request_id: str, *, approved: bool, reason: str = "",
    ) -> Optional[TaskState]:
        with self._lock:
            rec = self._pending.pop(request_id, None)
            if rec is None:
                return None
            rec.finished_at = time.time()
            rec.status = (
                TaskStatus.COMPLETED.value if approved
                else TaskStatus.CANCELLED.value
            )
            rec.result_preview = (
                f"approved: {reason}" if approved else f"denied: {reason}"
            )
            self._completed[request_id] = rec
            self._evict_completed()
        self._notify()
        return rec

    # ── Cancel (called from tray "Cancel" menu OR cloud cancel_task) ────

    def cancel_running(self, task_id: str) -> bool:
        """Cancel an in-flight tool call.  Returns True if it was running."""
        with self._lock:
            rec = self._running.get(task_id)
            if rec is None:
                return False
            future = rec._future
        if future is not None and not future.done():
            future.cancel()
            return True
        return False

    # ── Snapshots ────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Atomic dict-copy for tray polling / MCP responses."""
        with self._lock:
            return {
                "running":   [r.to_dict() for r in self._running.values()],
                "pending":   [r.to_dict() for r in self._pending.values()],
                "completed": [
                    r.to_dict() for r in
                    sorted(self._completed.values(),
                           key=lambda x: x.finished_at or 0,
                           reverse=True)
                ],
                "snapshot_at": time.time(),
            }

    def get_task(self, task_id: str) -> Optional[dict]:
        with self._lock:
            for bucket in (self._running, self._pending, self._completed):
                rec = bucket.get(task_id)
                if rec is not None:
                    return rec.to_dict()
        return None

    def counts(self) -> dict:
        with self._lock:
            return {
                "running":   len(self._running),
                "pending":   len(self._pending),
                "completed": len(self._completed),
            }

    # ── Eviction ─────────────────────────────────────────────────────────

    def _evict_completed(self) -> None:
        if len(self._completed) <= COMPLETED_KEEP_LAST:
            return
        ordered = sorted(
            self._completed.items(),
            key=lambda kv: kv[1].finished_at or 0,
        )
        excess = len(self._completed) - COMPLETED_KEEP_LAST
        for task_id, _ in ordered[:excess]:
            self._completed.pop(task_id, None)

    def _evict_pending(self) -> None:
        if len(self._pending) <= PENDING_KEEP_LAST:
            return
        # Should never realistically happen — toasts are 1:1 with user
        # attention.  If we hit the cap, the oldest pending is abandoned
        # (no toast click possible anymore); log loudly.
        ordered = sorted(
            self._pending.items(),
            key=lambda kv: kv[1].created_at,
        )
        excess = len(self._pending) - PENDING_KEEP_LAST
        for task_id, rec in ordered[:excess]:
            logger.warning(
                f"[tasks] pending approval queue full; abandoning "
                f"id={task_id} tool={rec.tool!r}"
            )
            self._pending.pop(task_id, None)


# ── Helpers ──────────────────────────────────────────────────────────────


def _preview_args(args: Optional[dict]) -> str:
    if not args:
        return ""
    try:
        s = json.dumps(args, ensure_ascii=False)[:80]
    except Exception:
        s = repr(args)[:80]
    return s


__all__ = [
    "TaskKind", "TaskStatus", "TaskState", "TaskRegistry",
    "COMPLETED_KEEP_LAST", "PENDING_KEEP_LAST",
]
