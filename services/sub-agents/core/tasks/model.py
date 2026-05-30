"""The canonical Task model.

A single dataclass that subsumes every "work item" the system tracks:

  - A daemon tool call (read_file, run_command)
  - A sub-agent run (Researcher / Builder / Critic)
  - A scheduled / periodic daemon
  - A user approval request
  - A high-level managed task with a dependency graph

Discriminated by ``kind`` + routed by ``executor``.  Each executor
interprets ``payload`` according to its own contract — for tool_call,
payload = {tool, args}; for sub_agent, payload = {persona, task,
model, backend}; etc.  The Task type stays narrow and the executors
own the polymorphism.

Lineage:

  ``parent_task_id`` — immediate parent (the agent that spawned this)
  ``root_task_id``   — top-of-tree (denormalised for cheap subtree
                       queries; equals task_id when self is a root)
  ``depends_on``     — explicit dependencies (ManagedTask-style; runs
                       block until all listed task_ids reach a
                       terminal state)

Lifecycle:

  pending → running → completed | failed | cancelled
                   ↘ awaiting_user (approval) → completed / cancelled

  Terminal states never transition.  Cancel races the executor; if
  the executor finished first the cancel is a no-op.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class TaskKind(str, Enum):
    """What KIND of work this Task represents.

    Each kind maps to (roughly) one executor.  The mapping isn't
    strictly 1:1 — TOOL_CALL routes to either DaemonExecutor or
    SandboxExecutor depending on bridge availability — but kind is
    the LLM-facing discriminator and what list_my_tasks groups by.
    """
    TOOL_CALL    = "tool_call"        # daemon-routed execute_tool
    SUB_AGENT    = "sub_agent"        # spawn_agentic / Researcher / Builder
    SCHEDULED    = "scheduled"        # periodic daemon (start_daemon)
    APPROVAL     = "approval"         # request_approval round-trip
    MANAGED      = "managed"          # user-defined task with dep graph
    DIRECTOR_RUN = "director_run"     # agent-runtime Director invocation


class TaskStatus(str, Enum):
    PENDING        = "pending"
    RUNNING        = "running"
    AWAITING_USER  = "awaiting_user"    # toast on screen, no click yet
    COMPLETED      = "completed"
    FAILED         = "failed"
    CANCELLED      = "cancelled"
    CANCELLING     = "cancelling"       # transient — descendants are being killed

    @classmethod
    def terminal(cls) -> set[str]:
        return {cls.COMPLETED.value, cls.FAILED.value, cls.CANCELLED.value}


# ── The dataclass ─────────────────────────────────────────────────────────


@dataclass
class Task:
    """Canonical unit of tracked work.

    Construct with ``Task.new(...)`` so the auto-fields (task_id +
    root_task_id collapse + timestamps) come out right.  Direct
    instantiation is fine too but you have to remember to set them.
    """
    # ── Identity ─────────────────────────────────────────────────────────
    task_id:        str
    user_id:        str
    tenant_id:      str = ""
    conv_id:        str = ""

    # ── Lineage ──────────────────────────────────────────────────────────
    parent_task_id: str = ""
    root_task_id:   str = ""              # == task_id when this is a root
    depends_on:     list[str] = field(default_factory=list)

    # ── Classification ───────────────────────────────────────────────────
    kind:           str = TaskKind.TOOL_CALL.value
    executor:       str = ""              # which executor owns this task

    # ── Content ──────────────────────────────────────────────────────────
    title:          str = ""              # human-readable name (tray / LLM)
    description:    str = ""              # longer explanation
    payload:        dict = field(default_factory=dict)
                                          # executor-specific input
    # ── Lifecycle ────────────────────────────────────────────────────────
    status:         str = TaskStatus.PENDING.value
    created_at:     float = field(default_factory=time.time)
    started_at:     Optional[float] = None
    finished_at:    Optional[float] = None

    # ── Outcome ──────────────────────────────────────────────────────────
    result:         str = ""              # truncated; full result in extras
    error:          str = ""

    # ── Schedule (only for kind=SCHEDULED) ───────────────────────────────
    interval_seconds: int = 0
    next_run_at:    Optional[float] = None

    # ── Catch-all ────────────────────────────────────────────────────────
    # JSON-encoded grab bag for executor-specific fields that don't fit
    # the typed columns.  Examples: daemon args, sub-agent persona,
    # constitution score, etc.  Keep small; large blobs belong in
    # separate tables.
    extras_json:    str = ""

    # ── Constructors ─────────────────────────────────────────────────────

    @classmethod
    def new(
        cls,
        *,
        kind: TaskKind,
        executor: str,
        user_id: str,
        title: str = "",
        payload: Optional[dict] = None,
        parent_task_id: str = "",
        conv_id: str = "",
        tenant_id: str = "",
        depends_on: Optional[list[str]] = None,
        task_id: Optional[str] = None,
        **extra_fields,
    ) -> "Task":
        """Mint a new Task with sensible defaults.

        Auto-fills:
          task_id     = `kind[:4]-<uuid8>` (e.g. ``tool-a1b2c3d4``)
          root_task_id = task_id if no parent; else inherit from parent
                         (caller must pass parent's root_task_id via
                         extra_fields["root_task_id"] for non-root tasks)
        """
        tid = task_id or f"{kind.value[:4]}-{uuid.uuid4().hex[:8]}"
        return cls(
            task_id=tid,
            user_id=user_id,
            tenant_id=tenant_id or user_id,
            conv_id=conv_id,
            parent_task_id=parent_task_id,
            root_task_id=extra_fields.pop("root_task_id", "") or
                         (tid if not parent_task_id else ""),
            depends_on=list(depends_on or []),
            kind=kind.value if isinstance(kind, TaskKind) else str(kind),
            executor=executor,
            title=title,
            description=extra_fields.pop("description", ""),
            payload=dict(payload or {}),
            **extra_fields,
        )

    # ── Convenience ──────────────────────────────────────────────────────

    @property
    def is_terminal(self) -> bool:
        return self.status in TaskStatus.terminal()

    @property
    def is_root(self) -> bool:
        return not self.parent_task_id

    def to_dict(self) -> dict:
        """JSON-safe shape for WS / MCP responses + Lance writes."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        """Reverse of to_dict — used by TaskStore when reading rows."""
        # Tolerate unknown keys so a future schema add doesn't break
        # older readers.
        known = {f.name for f in cls.__dataclass_fields__.values()}
        clean = {k: v for k, v in d.items() if k in known}
        return cls(**clean)


__all__ = ["Task", "TaskKind", "TaskStatus"]
