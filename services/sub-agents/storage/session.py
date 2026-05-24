"""Veilguard Sub-Agents: Session persistence."""

import json
import logging
import time

from config import SESSION_DIR, TRANSCRIPT_DIR
from core import state
from core.state import TaskStatus

logger = logging.getLogger("veilguard.storage")


def save_session():
    """Persist current state to disk.

    Called every 5 LLM calls (counter in core/llm.py) and once on
    graceful shutdown via the Starlette on_shutdown hook in server.py.
    Worst-case loss on a hard crash is up to 5 calls of state plus any
    daemon observations between saves — observations also stream to
    ``DAEMON_DATA_DIR/<daemon>.jsonl`` continuously, so the historical
    record survives even if state.json is mid-write at crash.
    """
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    try:
        # Build a JSON-safe view of every Daemon. Skip:
        #   - ``asyncio_task`` — coroutine handle, un-picklable, recreated
        #     fresh by start_persistent_loops() on next boot.
        #   - ``observations`` — already streamed to per-daemon JSONL
        #     files; restoring them from there avoids duplicating up to
        #     50 observation dicts × N daemons in state.json.
        # Everything else (rate-limit window, run_count, scheduling
        # cursors) IS persisted so a restart can't be used to bypass
        # the per-window action cap and so daemons resume on schedule
        # rather than firing immediately.
        daemons_serialized = {}
        for did, d in state.daemons.items():
            daemons_serialized[did] = {
                "name": d.name,
                "task": d.task,
                "role": d.role,
                "interval_seconds": d.interval_seconds,
                "backend": d.backend,
                "model": d.model,
                "enabled": d.enabled,
                "max_actions_per_window": d.max_actions_per_window,
                "action_count": d.action_count,
                "window_start": d.window_start,
                "last_run": d.last_run,
                "next_run": d.next_run,
                "run_count": d.run_count,
            }

        saved = {
            "stats": state.agent_stats,
            "cost": state.estimated_cost_usd,
            "schedules": state.schedules,
            "daemons": daemons_serialized,
            "managed_tasks": {
                tid: {
                    "id": t.id, "title": t.title, "description": t.description[:200],
                    "status": t.status, "depends_on": t.depends_on,
                    "assigned_to": t.assigned_to,
                }
                for tid, t in state.managed_tasks.items()
            },
        }
        path = SESSION_DIR / "state.json"
        path.write_text(json.dumps(saved, default=str, indent=2), encoding="utf-8")
        logger.debug("Session saved")
    except Exception as e:
        logger.error(f"Session save failed: {e}")


def load_session():
    """Restore state from disk on startup.

    NOTE: this populates the in-memory dicts (``state.schedules``,
    ``state.daemons``, etc.) but does NOT start asyncio tasks — those
    require a running event loop, which doesn't exist yet at module
    import time. Call ``start_persistent_loops()`` from a Starlette
    on_startup hook to actually resume daemons + the scheduler after
    the loop is up.

    Pre-0.2.5 this function silently restored ``state.schedules`` but
    no scheduler loop was ever started after a restart-from-saved-
    state, so saved schedules sat there idle. The two-phase approach
    (load_session() → start_persistent_loops()) closes that gap.
    """
    path = SESSION_DIR / "state.json"
    if not path.exists():
        return
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        state.agent_stats.update(saved.get("stats", {}))
        state.estimated_cost_usd = float(saved.get("cost", 0))
        state.schedules.update(saved.get("schedules", {}))

        # Reconstruct Daemon dataclasses. asyncio_task stays None until
        # start_persistent_loops() runs. observations starts empty here
        # — we rehydrate it from the JSONL file in start_persistent_loops
        # too (single place that touches JSONL on boot).
        from core.state import Daemon
        for did, d in saved.get("daemons", {}).items():
            try:
                state.daemons[did] = Daemon(**d)
            except TypeError as e:
                # Schema drift (e.g. old state.json predates a new field)
                # — log and skip rather than crashing the whole load.
                logger.warning(f"Skipping daemon {did} during restore: {e}")

        logger.info(
            f"Session restored: {state.agent_stats['total_calls']} calls, "
            f"${state.estimated_cost_usd:.4f}, "
            f"{len(state.daemons)} daemons, {len(state.schedules)} schedules"
        )
    except Exception as e:
        logger.error(f"Session load failed: {e}")


async def start_persistent_loops():
    """Start asyncio tasks for restored daemons and the scheduler.

    Must be called AFTER the asyncio event loop is running — typically
    from Starlette's ``on_startup`` lifespan hook. load_session() runs
    at module-import time when no event loop exists, so it can only
    populate the dicts; this function does the work that requires a
    live loop.

    Idempotent: safe to call multiple times. Each daemon's loop is
    only (re-)started if its ``asyncio_task`` is None or already done,
    so a second invocation is a no-op rather than a double-start.
    """
    import asyncio

    # Local imports avoid a top-level cycle. tools.daemons depends on
    # config + core.state, which storage.session also imports — keeping
    # the tools import inside this function means session.py can be
    # imported during early bootstrap without pulling the daemon code.
    from tools.daemons import _daemon_loop
    from tools.schedules import _ensure_scheduler

    # 1. Rehydrate each daemon's observation list from its JSONL log.
    # daemon_log shows the in-memory list, so without this rehydrate
    # users would see "no observations yet" for a daemon that's run
    # 1000 times. Cap at the last 50 to match the in-flight cap in
    # _daemon_loop (line "if len(daemon.observations) > 50: ...").
    from config import DAEMON_DATA_DIR
    rehydrated = 0
    for did, d in state.daemons.items():
        log_path = DAEMON_DATA_DIR / f"{d.name}.jsonl"
        if not log_path.is_file():
            continue
        try:
            lines = log_path.read_text(encoding="utf-8").splitlines()
            d.observations = []
            for line in lines[-50:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    d.observations.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            rehydrated += 1
        except OSError as e:
            logger.warning(f"Could not rehydrate observations for {did}: {e}")

    # 2. Restart enabled daemons. Skip already-running ones (idempotency).
    started = 0
    for did, d in state.daemons.items():
        if not d.enabled:
            continue
        existing = d.asyncio_task
        if existing is not None and not existing.done():
            continue
        d.asyncio_task = asyncio.create_task(_daemon_loop(d))
        started += 1

    # 3. Ensure the scheduler loop is running if we have any schedules.
    # _ensure_scheduler is the existing entry point — pre-0.2.5 it was
    # only called from schedule_task, which meant restored schedules
    # never fired. Calling it here from the post-load hook closes that
    # gap.
    if state.schedules:
        _ensure_scheduler()
        scheduler_status = f"started ({len(state.schedules)} schedules)"
    else:
        scheduler_status = "idle (no schedules)"

    logger.info(
        f"Persistent loops: {started} daemons restarted, "
        f"{rehydrated} observation logs rehydrated, scheduler {scheduler_status}"
    )


def log_transcript(agent_id: str, role: str, content: str):
    """Append an entry to the transcript log."""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"{agent_id}.jsonl"
    entry = {"role": role, "agent": agent_id, "time": time.time(), "content": content[:2000]}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
