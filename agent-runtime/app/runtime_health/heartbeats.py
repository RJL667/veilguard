"""Phase 6.3 — Lease TTL + heartbeats.

The turn cap (Phase 6.0.1) catches *live* dithering — agents that
keep emitting LLM turns without converging.  It does NOT catch *dead*
workers holding stale claims: a worker that crashes mid-task leaves
its lease in place but no turns are accruing, so the turn cap never
fires.

Heartbeats close that gap.  Workers write a heartbeat row to
`agent_task_heartbeats` at every turn boundary.  The inbox-poller
sweeps for rows where `now - last_beat_at > lease_ttl_s` and
auto-reclaims the task with a `lease_expired` audit comment.

This module is the pure helpers — `is_stale`, `select_orphans`.  The
actual Lance writes live in `app/memory/writers.py:record_heartbeat`
(future addition) + `app/workers/inbox_poller.py` sweep.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

# 5 minutes — matches the existing LEASE_DURATION_S in inbox_poller.  A
# worker missing 5 minutes of beats is considered dead; reclaim the task.
DEFAULT_LEASE_TTL_S = 300.0


@dataclass(frozen=True)
class HeartbeatRow:
    """One heartbeat record.  Lance schema is a separate table; this
    is the in-process projection used by the sweep logic."""
    task_id:         str
    worker_id:       str
    last_beat_at:    float
    lease_ttl_s:     float = DEFAULT_LEASE_TTL_S


def is_stale(row: HeartbeatRow, now: float | None = None) -> bool:
    """True iff the row's worker has not beaten in > lease_ttl_s."""
    t = time.time() if now is None else now
    return (t - row.last_beat_at) > row.lease_ttl_s


def select_orphans(
    rows: list[HeartbeatRow],
    now: float | None = None,
) -> list[HeartbeatRow]:
    """Return the subset of heartbeat rows that are stale.

    Caller (inbox_poller sweep) iterates the returned list and
    auto-reclaims each task, writing a `lease_expired` audit comment
    via `app/memory/writers.py:log_decision`.
    """
    return [r for r in rows if is_stale(r, now=now)]
