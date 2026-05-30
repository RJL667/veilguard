"""Phase 7.5 — Stall detector worker (Magentic-One Progress Ledger pattern).

Periodically walks `agent_tasks` for "stuck" rows that indicate an IC
has wedged mid-execution.  When a stall is detected, emits a
`task_stalled` comment on the task's SHA chain and (optionally)
broadcasts an SSE event so the sidebar can surface the stall.

Stall criteria (any one triggers):

  1. **Lease expired, status still in_progress** — IC claimed the
     task, never updated lease_until, never moved status to `review`
     or `done`.  Inbox-poller's startup sweep would catch a *crash*
     (orphaned lease at next boot) but not a *hang* (process still up,
     just frozen).

  2. **No heartbeat in N×lease_ttl_s** — the Phase 6.3 heartbeats
     stopped landing, but `lease_until` is still in the future (i.e.
     the IC explicitly set a long lease then went silent).

  3. **Progress ledger frozen** — Magentic-One spec §3.7: if the task
     has a progress_ledger with steps, and the most recent step's `ts`
     is older than the stall threshold despite the task still being
     `in_progress`, the agent isn't making forward progress.

The stall comment carries the trigger reason in its body so a
reviewing agent (or human via the sidebar) can decide whether to
re-dispatch (clear the lease so inbox-poller re-claims), escalate
(promote to Critic), or cancel.

Idempotency: each Task gets at most one open `task_stalled` comment.
Re-running this worker on the same stuck task is a no-op.

Phase 7.5 spec line:
> Stall detector worker (Magentic-One Progress Ledger pattern)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from ..ledger.store import LedgerStore, ns_filter

logger = logging.getLogger("agent-runtime.proposals.stall_detector")


# ── Tunables ─────────────────────────────────────────────────────────────


# How long after lease_until expires before we call it a stall.
# Below this the inbox-poller's stale-heartbeat sweep can still reclaim
# the row organically.  Above this it's been long enough that nothing
# is going to recover without intervention.
STALL_LEASE_GRACE_S = 5 * 60.0     # 5 min beyond expiry

# How stale the latest progress_ledger step can be before we flag.
# Set higher than typical inter-turn latency so we don't false-positive
# on a slow LLM call.  Magentic-One uses ~30 turns; we use wall-clock.
STALL_PROGRESS_FROZEN_S = 30 * 60.0     # 30 min without a step

# How long after the last heartbeat before "silent stall" fires.
# Inbox-poller writes a heartbeat per turn (see Phase 6.3).
STALL_HEARTBEAT_SILENT_S = 15 * 60.0    # 15 min without a beat


# Stall trigger types (carried in comment body for downstream routing).
TRIGGER_LEASE_EXPIRED  = "lease_expired"
TRIGGER_HEARTBEAT_SILENT = "heartbeat_silent"
TRIGGER_PROGRESS_FROZEN = "progress_frozen"


# ── Detection ────────────────────────────────────────────────────────────


def _open_in_progress_tasks(
    *, tenant_id: str, user_id: str, limit: int = 1000,
) -> list[dict[str, Any]]:
    """Pull all in_progress tasks for one (tenant, user).  Limited to
    `limit` to bound a runaway tenant; production scale far below it."""
    try:
        tbl = LedgerStore.get().table("agent_tasks")
        where = (
            f"{ns_filter(tenant_id, user_id)} "
            f"AND status = 'in_progress'"
        )
        arr = tbl.search().where(where).limit(limit).to_arrow()
    except Exception as e:
        logger.warning(
            f"[stall_detector] in_progress scan failed for "
            f"{tenant_id}/{user_id}: {e}"
        )
        return []
    return [
        {col: arr.column(col)[i].as_py() for col in arr.column_names}
        for i in range(arr.num_rows)
    ]


def _last_heartbeat_for_task(task_id: str) -> float | None:
    """Return the most-recent `last_beat_at` for this task, or None."""
    try:
        tbl = LedgerStore.get().table("agent_task_heartbeats")
        arr = (
            tbl.search()
            .where(f"task_id = '{task_id}'")
            .limit(500)
            .to_arrow()
            .to_pylist()
        )
        if not arr:
            return None
        return max(r.get("last_beat_at") or 0.0 for r in arr)
    except Exception:
        return None


def _existing_open_stall_comment(task_id: str, tenant_id: str, user_id: str) -> bool:
    """Idempotency check — does this task already have an open
    `task_stalled` comment?  Open == no `stall_cleared` follow-up."""
    try:
        from ..ledger import comments as _c
        rows = _c.list_comments(
            task_id=task_id, tenant_id=tenant_id, user_id=user_id, limit=500,
        )
    except Exception:
        return False
    # Walk newest-first; a `stall_cleared` after a `task_stalled` means
    # the stall was resolved and we can flag again.
    saw_stall = False
    for row in reversed(rows):
        body = (row.get("body") or "")
        if body.startswith("[stall_cleared]"):
            return False
        if body.startswith("[task_stalled]"):
            saw_stall = True
            return True
    return saw_stall


def detect_stalls(*, tenant_id: str, user_id: str, now: float | None = None) -> list[dict[str, Any]]:
    """Return a list of stall findings for this tenant.  Pure read —
    no writes here; `emit_stall_comments` does the side effects.

    Each finding is a dict:
        {
          "task_id":    str,
          "trigger":    one of TRIGGER_* constants,
          "reason":     human-readable why-string,
          "evidence":   {key: value, ...}    # for the comment body
        }
    """
    if now is None:
        now = time.time()
    tasks = _open_in_progress_tasks(tenant_id=tenant_id, user_id=user_id)
    findings: list[dict[str, Any]] = []
    for t in tasks:
        tid = t.get("id")
        if not tid:
            continue
        lease_until = float(t.get("lease_until") or 0.0)
        # Trigger 1: lease expired by more than grace.
        if 0 < lease_until < (now - STALL_LEASE_GRACE_S):
            findings.append({
                "task_id": tid,
                "trigger": TRIGGER_LEASE_EXPIRED,
                "reason":  (
                    f"lease expired {now - lease_until:.0f}s ago "
                    f"(grace={STALL_LEASE_GRACE_S:.0f}s); "
                    f"task still in_progress"
                ),
                "evidence": {
                    "lease_until":   lease_until,
                    "lease_owner":   t.get("lease_owner"),
                    "now":           now,
                },
            })
            continue   # don't double-flag a lease-expired task on heartbeat too

        # Trigger 2: heartbeat silent.
        last_beat = _last_heartbeat_for_task(tid)
        if last_beat is not None and (now - last_beat) > STALL_HEARTBEAT_SILENT_S:
            findings.append({
                "task_id": tid,
                "trigger": TRIGGER_HEARTBEAT_SILENT,
                "reason":  (
                    f"no heartbeat for {now - last_beat:.0f}s "
                    f"(threshold={STALL_HEARTBEAT_SILENT_S:.0f}s); "
                    f"task still in_progress with lease in future"
                ),
                "evidence": {
                    "last_beat_at": last_beat,
                    "lease_until":  lease_until,
                    "now":          now,
                },
            })
            continue

        # Trigger 3: progress frozen (Magentic-One pattern).
        extras_raw = t.get("extras_json") or ""
        try:
            extras = json.loads(extras_raw) if extras_raw else {}
        except (json.JSONDecodeError, TypeError):
            extras = {}
        progress = extras.get("progress_ledger") or []
        if isinstance(progress, list) and progress:
            try:
                last_step_ts = float(progress[-1].get("ts") or 0.0)
            except (AttributeError, TypeError):
                last_step_ts = 0.0
            if last_step_ts > 0 and (now - last_step_ts) > STALL_PROGRESS_FROZEN_S:
                findings.append({
                    "task_id": tid,
                    "trigger": TRIGGER_PROGRESS_FROZEN,
                    "reason":  (
                        f"progress_ledger frozen for {now - last_step_ts:.0f}s "
                        f"(threshold={STALL_PROGRESS_FROZEN_S:.0f}s); "
                        f"latest step idx={len(progress) - 1}"
                    ),
                    "evidence": {
                        "last_step_ts":   last_step_ts,
                        "progress_steps": len(progress),
                        "now":            now,
                    },
                })
    return findings


# ── Emission ─────────────────────────────────────────────────────────────


def emit_stall_comments(
    *,
    tenant_id: str,
    user_id: str,
    findings: list[dict[str, Any]],
    actor_agent_id: str = "stall-detector",
) -> int:
    """Write one `task_stalled` comment per finding that doesn't
    already have an open stall.  Returns count of comments written.

    The comment kind is `blocker_raised` so existing kind-based readers
    (sidebar Work Queue, Director's open-blocker probe) see it without
    schema changes.  The body prefix `[task_stalled]` is the marker
    downstream filters can use.
    """
    from ..ledger import comments as _c
    written = 0
    for f in findings:
        tid = f["task_id"]
        if _existing_open_stall_comment(tid, tenant_id, user_id):
            continue
        body = (
            f"[task_stalled] trigger={f['trigger']}\n"
            f"{f['reason']}\n\n"
            f"evidence={json.dumps(f['evidence'], default=str)}"
        )
        try:
            _c.add_comment(
                task_id=tid,
                tenant_id=tenant_id, user_id=user_id,
                author_id=actor_agent_id,
                kind="blocker_raised",
                body=body,
                extras_json=json.dumps({
                    "stall_detector": True,
                    "trigger": f["trigger"],
                }),
            )
            written += 1
            # Best-effort SSE event so the sidebar sees it live.
            try:
                from ..events import broadcast
                broadcast({
                    "type":      "task_stalled",
                    "tenant_id": tenant_id, "user_id": user_id,
                    "task_id":   tid,
                    "trigger":   f["trigger"],
                    "reason":    f["reason"],
                })
            except Exception:
                pass
        except Exception as e:
            logger.warning(
                f"[stall_detector] couldn't add stall comment for {tid}: {e}"
            )
    if written:
        logger.info(
            f"[stall_detector] emitted {written} task_stalled comments "
            f"for {tenant_id}/{user_id}"
        )
    return written


# ── Worker ───────────────────────────────────────────────────────────────


def run_one_cycle(*, tenant_id: str, user_id: str) -> dict[str, Any]:
    """One-pass scan + emit for a single (tenant, user).  Idempotent."""
    findings = detect_stalls(tenant_id=tenant_id, user_id=user_id)
    written  = emit_stall_comments(
        tenant_id=tenant_id, user_id=user_id, findings=findings,
    )
    summary = {
        "tenant_id": tenant_id, "user_id": user_id,
        "findings": len(findings),
        "emitted":  written,
        "by_trigger": _count_by(findings, "trigger"),
    }
    return summary


def _count_by(findings: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in findings:
        v = f.get(key) or "?"
        out[v] = out.get(v, 0) + 1
    return out


class StallDetectorWorker:
    """Periodic stall sweep across all active tenants.  Cheap — one
    Lance scan per tenant per cycle, no LLM calls.  Default cadence
    every 5 min (sub-`STALL_LEASE_GRACE_S` so a stall is flagged within
    one cycle of crossing the threshold)."""

    def __init__(self, interval_seconds: float = 300.0):
        self.interval_seconds = float(interval_seconds)
        self._stop_evt = asyncio.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    def discover_active_pairs(self) -> list[tuple[str, str]]:
        """Find (tenant_id, user_id) tuples with at least one
        in_progress task — those are the only places stalls can hide."""
        try:
            tbl = LedgerStore.get().table("agent_tasks")
            arr = (
                tbl.search()
                .where("status = 'in_progress'")
                .select(["tenant_id", "user_id"])
                .limit(5000)
                .to_arrow()
            )
        except Exception as e:
            logger.warning(f"[stall_detector] discover pairs failed: {e}")
            return []
        pairs = set()
        for i in range(arr.num_rows):
            t = arr.column("tenant_id")[i].as_py() or ""
            u = arr.column("user_id")[i].as_py() or ""
            if t and u:
                pairs.add((t, u))
        return sorted(pairs)

    async def run(self) -> None:
        logger.info(
            f"[stall_detector] worker starting; "
            f"interval={self.interval_seconds:.0f}s "
            f"lease_grace={STALL_LEASE_GRACE_S:.0f}s "
            f"hb_silent={STALL_HEARTBEAT_SILENT_S:.0f}s "
            f"progress_frozen={STALL_PROGRESS_FROZEN_S:.0f}s"
        )
        try:
            while not self._stop_evt.is_set():
                try:
                    pairs = self.discover_active_pairs()
                    for tenant_id, user_id in pairs:
                        try:
                            summary = run_one_cycle(
                                tenant_id=tenant_id, user_id=user_id,
                            )
                            if summary["findings"]:
                                logger.info(
                                    f"[stall_detector] cycle: {summary}"
                                )
                        except Exception as e:
                            logger.exception(
                                f"[stall_detector] cycle failed for "
                                f"{tenant_id}/{user_id}: {e}"
                            )
                except Exception as e:
                    logger.exception(f"[stall_detector] sweep error: {e}")
                try:
                    await asyncio.wait_for(
                        self._stop_evt.wait(),
                        timeout=self.interval_seconds,
                    )
                except asyncio.TimeoutError:
                    pass
        finally:
            logger.info("[stall_detector] worker stopped")


__all__ = [
    "STALL_LEASE_GRACE_S",
    "STALL_PROGRESS_FROZEN_S",
    "STALL_HEARTBEAT_SILENT_S",
    "TRIGGER_LEASE_EXPIRED",
    "TRIGGER_HEARTBEAT_SILENT",
    "TRIGGER_PROGRESS_FROZEN",
    "detect_stalls",
    "emit_stall_comments",
    "run_one_cycle",
    "StallDetectorWorker",
]
