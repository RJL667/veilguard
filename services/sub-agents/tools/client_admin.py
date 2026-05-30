"""MCP tools — surfacing daemon tasks + permission level to the LLM.

Phase D of the Veilguard client expansion:

  list_my_tasks(scope)          → snapshot of daemon-local tasks
  daemon_task_status(task_id)   → detail for one task
  cancel_daemon_task(task_id)   → best-effort cancel of an in-flight
                                   client-tool call on the daemon
                                   (renamed from cancel_task to avoid
                                   colliding with tools/tasks.py's
                                   pre-existing cancel_task which
                                   cancels SERVER-side background
                                   tasks; both lived under the same
                                   bare name and the MCP server
                                   rejected the second registration)

  get_permission_level()         → current effective (user-default,
                                   conv-override-or-none, effective)
  set_permission_level(level,    → upsert the user default or set the
                       scope)      conv override

All tools resolve the current user/conv via request_ctx — same flow as
every other MCP tool here.  Cross-tenant access is blocked structurally:
the daemon RPC only ever reaches THIS user's bridge.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from core import request_ctx
from core.client_bridge import get_bridge
from core.client_settings import (
    ClientSettingsStore,
    PERMISSION_LEVELS,
    DEFAULT_LEVEL,
)

logger = logging.getLogger("veilguard.tools.client_admin")


# ── Helpers ──────────────────────────────────────────────────────────────


async def _bridge_send(method: str, params: dict) -> Any:
    """Round-trip a JSON-RPC method to the user's daemon.  Returns None
    if there's no live daemon."""
    user_id = request_ctx.get_user_id()
    if not user_id:
        return None
    bridge = get_bridge(user_id)
    if not bridge or not bridge.connected:
        return None
    try:
        return await bridge.send_command(method, params, timeout=10.0)
    except Exception as e:
        logger.warning(f"[client_admin] bridge.{method} failed: {e}")
        return None


# ── Registration ─────────────────────────────────────────────────────────


def register(mcp):

    @mcp.tool()
    async def list_my_tasks(scope: str = "all", since_seconds: int = 0) -> str:
        """List EVERY task running on your behalf — daemon tool calls,
        sub-agents, scheduled workers, approval prompts.

        WHEN TO USE: the user asks "what are you doing?" / "what's
        running?", or before launching a long task you want to confirm
        nothing else is blocking, or to find a task_id to cancel.

        Args:
            scope: "all" (default) | "running" | "pending" | "completed"
                   - running   = in-flight (tool calls + sub-agents)
                   - pending   = approvals waiting for user click
                   - completed = recently finished

        Returns markdown grouped by kind (tool_call / sub_agent / etc.).
        Each task carries an executor tag (daemon / sandbox / etc.) so
        you can tell whether work ran on the user's machine, in the
        server sandbox, or via agent-runtime.

        Implementation (Phase B-1): queries the unified TaskStore Lance
        table.  As B-2/3/4 land, more task kinds appear here without
        any change to this tool.  Daemon-local in-memory state is
        merged in for in-flight tool calls that haven't yet flushed to
        Lance.
        """
        scope_clean = (scope or "all").lower().strip()
        if scope_clean not in ("all", "running", "pending", "completed"):
            return f"Error: scope must be one of all/running/pending/completed"

        user_id = request_ctx.get_user_id()
        if not user_id:
            return "Error: no user context (missing x-user-id header)."

        # ── Unified read: Lance TaskStore ────────────────────────────
        try:
            from core.tasks import TaskDispatcher, TaskStatus
            dispatcher = TaskDispatcher.get()

            scope_to_statuses = {
                "running":   [TaskStatus.RUNNING.value],
                "pending":   [TaskStatus.PENDING.value,
                              TaskStatus.AWAITING_USER.value],
                "completed": [TaskStatus.COMPLETED.value,
                              TaskStatus.FAILED.value,
                              TaskStatus.CANCELLED.value],
            }
            since_ts = (time.time() - since_seconds) if since_seconds > 0 else 0
            if scope_clean == "all":
                tasks = dispatcher.list_user(
                    user_id, since_ts=since_ts, limit=200,
                )
            else:
                tasks = dispatcher.list_user(
                    user_id,
                    statuses=scope_to_statuses[scope_clean],
                    since_ts=since_ts, limit=200,
                )
        except Exception as e:
            return f"Error: TaskStore unavailable: {e}"

        # ── Merge in daemon's in-memory in-flight state ──────────────
        # Tool calls landed via dispatcher already have Lance rows.
        # Tool calls that took the legacy direct-bridge path
        # (dispatcher fallback) don't — daemon's TaskRegistry knows
        # about them.  Fetch and shadow-merge so the LLM sees them too.
        daemon_snap = await _bridge_send("list_local_tasks", {}) or {}
        if isinstance(daemon_snap, str):
            try:
                import json as _json
                daemon_snap = _json.loads(daemon_snap)
            except Exception:
                daemon_snap = {}
        existing_ids = {t.task_id for t in tasks}
        for bucket_key in ("running", "pending", "completed"):
            for legacy in (daemon_snap.get(bucket_key) or []):
                lid = legacy.get("task_id")
                if not lid or lid in existing_ids:
                    continue
                # Adapt daemon dict-shape to a faux Task for rendering.
                _faux = _FauxTaskFromLegacy(legacy)
                if scope_clean == "all" or _faux.status in scope_to_statuses[scope_clean]:
                    tasks.append(_faux)

        # ── Render ───────────────────────────────────────────────────
        by_kind: dict[str, list] = {}
        for t in tasks:
            by_kind.setdefault(t.kind or "tool_call", []).append(t)

        lines: list[str] = []
        for kind in sorted(by_kind.keys()):
            bucket = by_kind[kind]
            lines.append(f"# {kind} ({len(bucket)})")
            # Sort: pending first, then by created_at desc
            bucket.sort(key=lambda x: (
                0 if x.status == "pending" else
                1 if x.status == "running" else 2,
                -(x.created_at or 0),
            ))
            for t in bucket[:30]:    # cap output per kind
                lines.append(_fmt_unified_line(t))
            if not bucket:
                lines.append("_(none)_")
            lines.append("")

        if not by_kind:
            return f"# No tasks ({scope_clean})\n_(nothing to show)_"
        return "\n".join(lines).strip()

    @mcp.tool()
    async def daemon_task_status(task_id: str) -> str:
        """Get the full state of one daemon task by ID.

        Useful after `list_my_tasks` to see error / result_preview /
        timestamps for a specific task.  Returns "Not found" if the
        task has been evicted from the completed cap (~50).
        """
        if not task_id:
            return "Error: task_id required"
        rec = await _bridge_send("task_state", {"task_id": task_id})
        if rec is None:
            return f"Task `{task_id}` not found (or daemon offline)."
        if isinstance(rec, str):
            try:
                import json
                rec = json.loads(rec)
            except Exception:
                return f"Error: malformed daemon response"
        return _fmt_task_detail(rec or {})

    @mcp.tool()
    async def cancel_daemon_task(task_id: str) -> str:
        """Cancel a running CLIENT-tool call on the daemon.

        Distinct from the cancel_task in tools/tasks.py which cancels
        SERVER-side background tasks.  This one cancels the in-flight
        run_command / read_file / etc. that the daemon is executing on
        the user's machine — useful when the LLM realises mid-call
        that it shouldn't have started that grep across C:\.

        Only `running` tasks can be cancelled — pending approvals
        (toasts) are cancelled by the user dismissing the toast, and
        completed tasks are inherently uncancellable.  Returns
        confirmation or "not running".
        """
        if not task_id:
            return "Error: task_id required"
        resp = await _bridge_send("cancel_task", {"task_id": task_id})
        if resp is None:
            return f"Error: daemon offline; cannot cancel."
        if isinstance(resp, str):
            try:
                import json
                resp = json.loads(resp)
            except Exception:
                return f"Error: malformed daemon response"
        ok = bool((resp or {}).get("cancelled"))
        if ok:
            return f"Cancelled task `{task_id}`."
        return (
            f"Could not cancel `{task_id}` — not running "
            "(may have already completed or never existed)."
        )

    @mcp.tool()
    async def get_permission_level() -> str:
        """Show the current user permission level + conv override (if any).

        Three levels exist:
          - auto:    ALLOW + APPROVE tools run silently; DENY blocks
          - confirm: APPROVE tools pop a Windows toast for user approval
          - strict:  every CLIENT_TOOL pops a toast regardless of class

        Returns markdown showing both the user default and any active
        per-conversation override.
        """
        user_id = request_ctx.get_user_id()
        conv_id = request_ctx.get_conversation_id() or ""
        if not user_id:
            return "Error: no user context in this MCP call."
        store = ClientSettingsStore.get()
        user_lvl, timeout_s = store.get_user_level(user_id)
        override = store.get_conv_override(user_id, conv_id) if conv_id else None
        effective = override or user_lvl
        return (
            f"# Permission level\n"
            f"- **Effective**: `{effective}`\n"
            f"- User default:   `{user_lvl}`  "
            f"(toast timeout: {timeout_s}s)\n"
            f"- Conv override:  `{override or '(none)'}`\n"
            f"- Levels available: {', '.join(PERMISSION_LEVELS)}\n"
        )

    @mcp.tool()
    async def set_permission_level(level: str, scope: str = "user") -> str:
        """Set the user default or per-conversation permission level.

        Args:
            level: one of `auto`, `confirm`, `strict`
            scope: `user` (default) — persist as the user's default
                   `conv` — apply only to the current conversation

        The change takes effect on the NEXT tool call — there's no
        rebroadcast to the daemon UI today (the tray reads the
        effective level on each toast).
        """
        level = (level or "").lower().strip()
        scope = (scope or "user").lower().strip()
        if level not in PERMISSION_LEVELS:
            return (
                f"Error: level must be one of "
                f"{', '.join(PERMISSION_LEVELS)} (got {level!r})"
            )
        if scope not in ("user", "conv"):
            return f"Error: scope must be 'user' or 'conv' (got {scope!r})"

        user_id = request_ctx.get_user_id()
        conv_id = request_ctx.get_conversation_id() or ""
        if not user_id:
            return "Error: no user context in this MCP call."

        store = ClientSettingsStore.get()
        try:
            if scope == "user":
                store.set_user_level(user_id, level)
                return (
                    f"Set permission level to **{level}** for your account. "
                    f"Conv overrides still apply."
                )
            if not conv_id:
                return "Error: scope='conv' requires an active conversation."
            store.set_conv_override(user_id, conv_id, level)
            return (
                f"Set permission override to **{level}** for THIS "
                f"conversation only.  Your account default unchanged."
            )
        except Exception as e:
            return f"Error: failed to persist setting: {e}"


# ── Formatters ───────────────────────────────────────────────────────────


def _fmt_task_line(r: dict) -> str:
    """One-line summary for the legacy dict-shape bullet list."""
    tid = r.get("task_id", "?")
    tool = r.get("tool", "")
    status = r.get("status", "?")
    args_preview = (r.get("args_preview") or "")[:60]
    return f"- `{tid}` · **{tool}** · _{status}_ · {args_preview}"


def _fmt_unified_line(t) -> str:
    """One-line summary for a unified Task object.  Defensive against
    both Task-from-store and the _FauxTaskFromLegacy adapter.
    """
    tid = getattr(t, "task_id", "?")
    title = getattr(t, "title", "") or (
        # Fall back to payload tool name if title's empty
        (getattr(t, "payload", {}) or {}).get("tool", "?")
    )
    status = getattr(t, "status", "?")
    executor = getattr(t, "executor", "?")
    parent = getattr(t, "parent_task_id", "")
    parent_tag = f" ↑{parent[:8]}" if parent else ""
    return f"- `{tid}` · **{title}** · _{status}_ · @{executor}{parent_tag}"


class _FauxTaskFromLegacy:
    """Duck-typed Task adapter for daemon TaskRegistry rows.

    Lets us merge daemon snapshot dicts into the same render path as
    unified Tasks without two formatters.  Keeps the field names
    aligned with model.Task.
    """
    __slots__ = ("task_id", "title", "status", "kind", "executor",
                 "parent_task_id", "payload", "created_at")

    def __init__(self, d: dict):
        self.task_id = d.get("task_id", "?")
        self.title = d.get("tool", "")
        self.status = d.get("status", "?")
        self.kind = "tool_call" if d.get("kind", "").endswith("call") else \
                    "approval" if d.get("kind", "").startswith("appr") else \
                    d.get("kind", "tool_call")
        self.executor = "daemon-legacy"     # signals it bypassed dispatcher
        self.parent_task_id = ""
        self.payload = {"tool": d.get("tool", ""),
                        "args_preview": d.get("args_preview", "")}
        self.created_at = float(d.get("created_at") or 0)


def _fmt_task_detail(r: dict) -> str:
    """Multi-line full detail view."""
    import time
    def _t(ts):
        if not ts: return "-"
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

    lines = [
        f"# Task `{r.get('task_id', '?')}`",
        f"- Kind:     `{r.get('kind', '?')}`",
        f"- Status:   `{r.get('status', '?')}`",
        f"- Tool:     `{r.get('tool', '')}`",
        f"- Agent:    `{r.get('agent_id', '?')}`",
        f"- Conv:     `{r.get('conv_id', '')}`",
        f"- Created:  {_t(r.get('created_at'))}",
        f"- Started:  {_t(r.get('started_at'))}",
        f"- Finished: {_t(r.get('finished_at'))}",
    ]
    args_preview = r.get("args_preview")
    if args_preview:
        lines.append(f"- Args:     `{args_preview}`")
    result = r.get("result_preview") or r.get("error") or ""
    if result:
        lines.append("")
        lines.append("**Result / error preview:**")
        lines.append("```")
        lines.append(result[:1000])
        lines.append("```")
    return "\n".join(lines)
