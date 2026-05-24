"""Daemon tools — persistent background monitors (KAIROS-inspired)."""

import asyncio
import json
import logging
import re
import time

from config import ROLES, DEFAULT_BACKEND, DAEMON_DATA_DIR
from core import state
from core.state import Daemon
from core.llm import call_llm, resolve_backend
from utils.afk import is_afk
from utils.tool_timing import resolve_timeout, record_duration, format_hint

logger = logging.getLogger("veilguard.daemons")


async def _daemon_loop(daemon: Daemon):
    """Background loop for a single daemon."""
    while daemon.enabled:
        try:
            now = time.time()
            if now < daemon.next_run:
                await asyncio.sleep(min(10, daemon.next_run - now))
                continue
            if now - daemon.window_start > 3600:
                daemon.window_start = now
                daemon.action_count = 0
            effective_budget = daemon.max_actions_per_window * (2 if is_afk() else 1)
            if daemon.action_count >= effective_budget:
                await asyncio.sleep(60)
                continue

            daemon.last_run = now
            daemon.run_count += 1
            daemon.action_count += 1
            daemon.next_run = now + daemon.interval_seconds

            role_prompt = ROLES.get(daemon.role, ROLES.get("researcher", {})).get("system", "You are a helpful assistant.")
            result = await call_llm(
                role_prompt + "\n\nYou are running as a background daemon. Be concise. Report only findings or anomalies.",
                daemon.task, backend=daemon.backend, model=daemon.model, role=daemon.role,
            )
            observation = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "run": daemon.run_count, "result": result[:2000]}
            daemon.observations.append(observation)
            if len(daemon.observations) > 50: daemon.observations = daemon.observations[-50:]

            DAEMON_DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(DAEMON_DATA_DIR / f"{daemon.name}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(observation) + "\n")
            logger.info(f"Daemon {daemon.name} run {daemon.run_count} complete")

        except asyncio.CancelledError:
            break
        except Exception as e:
            daemon.observations.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "run": daemon.run_count, "error": str(e)})
            logger.error(f"Daemon {daemon.name} error: {e}")
            await asyncio.sleep(60)


def register(mcp):
    @mcp.tool()
    async def start_daemon(name: str, task: str, interval_minutes: int = 30, role: str = "researcher",
                           backend: str = "", model: str = "", max_actions_per_hour: int = 5) -> str:
        """Start a persistent background daemon that runs a task on a schedule."""
        daemon_id = name.lower().replace(" ", "-")
        if daemon_id in state.daemons and state.daemons[daemon_id].enabled:
            return f"Error: Daemon '{daemon_id}' already running."
        cfg = resolve_backend(backend)
        use_model = model if model else cfg["default_model"]
        daemon = Daemon(name=daemon_id, task=task, role=role, interval_seconds=max(int(interval_minutes), 1) * 60,
                        backend=(backend or DEFAULT_BACKEND).lower(), model=use_model,
                        max_actions_per_window=max(max_actions_per_hour, 1),
                        next_run=time.time() + 5, window_start=time.time())
        daemon.asyncio_task = asyncio.create_task(_daemon_loop(daemon))
        state.daemons[daemon_id] = daemon
        return f"Daemon `{daemon_id}` started. Every {interval_minutes}min, role={role}."

    @mcp.tool()
    async def stop_daemon(name: str) -> str:
        """Stop a running daemon."""
        daemon_id = name.lower().replace(" ", "-")
        if daemon_id not in state.daemons: return f"Error: Daemon '{daemon_id}' not found"
        d = state.daemons[daemon_id]
        d.enabled = False
        if d.asyncio_task and not d.asyncio_task.done(): d.asyncio_task.cancel()
        return f"Daemon `{daemon_id}` stopped after {d.run_count} runs."

    @mcp.tool()
    async def list_daemons() -> str:
        """List all daemons and their status."""
        if not state.daemons: return "No daemons running."
        lines = ["# Daemons\n"]
        for did, d in state.daemons.items():
            next_in = max(0, int(d.next_run - time.time()))
            lines.append(f"- **{did}** [{'RUNNING' if d.enabled else 'STOPPED'}] every {d.interval_seconds//60}min | runs: {d.run_count} | next: {next_in}s")
        return "\n".join(lines)

    @mcp.tool()
    async def daemon_log(name: str, last_n: int = 5) -> str:
        """View recent observations from a daemon."""
        daemon_id = name.lower().replace(" ", "-")
        if daemon_id not in state.daemons: return f"Error: Daemon '{daemon_id}' not found"
        obs = state.daemons[daemon_id].observations[-last_n:]
        if not obs: return f"Daemon `{daemon_id}` has no observations yet."
        lines = [f"# Daemon `{daemon_id}` — last {len(obs)} observations\n"]
        for o in obs:
            lines.append(f"### Run {o.get('run','?')} @ {o.get('time','?')}")
            if "error" in o: lines.append(f"**ERROR:** {o['error']}")
            else: lines.append(o.get("result", "(no output)"))
        return "\n".join(lines)

    @mcp.tool()
    async def daemon_wait(name: str, since_run: int = -1, pattern: str = "",
                          timeout_seconds: int = 0) -> str:
        """Block until the daemon produces a new observation, then return it.

        Use this instead of polling daemon_log when you want to wait for the
        next daemon update. Returns immediately if a matching observation
        already exists (run > since_run); otherwise blocks up to
        timeout_seconds for one to arrive.

        WHEN TO USE: long-running daemons where you want to react to the next
        finding (e.g. "wake me when the inbox-watcher daemon finds a flagged
        email"). Cheaper than polling daemon_log every minute — one blocking
        call instead of N round-trips.

        Args:
            name: Daemon name (same as for start_daemon / daemon_log).
            since_run: Wait for an observation with run > this value. Pass -1
                       (default) to wait for the NEXT observation after now.
            pattern: Optional case-insensitive regex. If set, only observations
                     whose result/error matches the pattern count as "new".
            timeout_seconds: Max wait. Pass 0 (default) for "auto-pick" — the
                             system uses the daemon's interval and your past
                             call history to choose. Pass an explicit value
                             only when you have a specific reason. Capped at
                             1800s (30min) regardless.
        """
        daemon_id = name.lower().replace(" ", "-")
        if daemon_id not in state.daemons:
            return f"Error: Daemon '{daemon_id}' not found"
        daemon = state.daemons[daemon_id]

        if since_run < 0:
            since_run = daemon.run_count

        # Smart-default the timeout. Tool-derived hint = 2x the daemon's
        # configured interval (a daemon that fires every 30min produces
        # observations roughly every 30min, so waiting one full interval
        # plus headroom is the right ballpark). History from past
        # daemon_wait calls on this same daemon overrides once we have ≥3
        # samples — real cadence beats configured cadence.
        resolved = resolve_timeout(
            tool_name="daemon_wait",
            key=daemon_id,
            explicit=timeout_seconds if timeout_seconds > 0 else None,
            derived_default=lambda: max(60, 2 * daemon.interval_seconds),
            hard_min=1,
            hard_max=1800,
            fallback=300,
        )

        try:
            pattern_re = re.compile(pattern, re.IGNORECASE) if pattern else None
        except re.error as e:
            return f"Error: invalid regex `{pattern}`: {e}"

        def _match(obs):
            if obs.get("run", 0) <= since_run:
                return False
            if pattern_re is None:
                return True
            blob = (obs.get("result", "") or "") + " " + (obs.get("error", "") or "")
            return bool(pattern_re.search(blob))

        wait_start = time.time()
        deadline = wait_start + resolved.timeout_seconds

        # Fast path: matching observation may already exist. Don't record
        # this as a real "wait" — we didn't actually block, so adding 0s
        # to the history would skew p95 toward zero and break future
        # defaults.
        for obs in daemon.observations:
            if _match(obs):
                return _format_observation(daemon_id, obs)

        # Slow path: poll the in-memory list with bounded sleeps.
        # Sleep cadence: short (1s) at first so we react quickly, then
        # back off — the daemon's interval is in minutes, no need to spin.
        poll = 1.0
        while time.time() < deadline:
            await asyncio.sleep(min(poll, max(0.1, deadline - time.time())))
            poll = min(poll * 1.5, 15.0)
            for obs in daemon.observations:
                if _match(obs):
                    elapsed = time.time() - wait_start
                    record_duration("daemon_wait", daemon_id, elapsed, success=True)
                    return (_format_observation(daemon_id, obs)
                            + format_hint(resolved, actual_elapsed=elapsed))
            if not daemon.enabled:
                return (f"Daemon `{daemon_id}` was stopped while waiting "
                        f"(last run {daemon.run_count}). No new observation.")

        # Timeout. Record the failed wait so format_hint can flag the
        # divergence from history (or lack of it) — over time the
        # auto-picked default will trend up if this daemon really does
        # need longer waits.
        record_duration("daemon_wait", daemon_id, resolved.timeout_seconds, success=False)
        return (f"Daemon `{daemon_id}`: no new observation"
                f"{' matching pattern' if pattern else ''} within "
                f"{resolved.timeout_seconds}s (last run was {daemon.run_count})."
                + format_hint(resolved, hit_timeout=True))


def _format_observation(daemon_id: str, obs: dict) -> str:
    body = obs.get("result") or f"ERROR: {obs.get('error', '(unknown)')}"
    return (f"# Daemon `{daemon_id}` — new observation\n\n"
            f"### Run {obs.get('run','?')} @ {obs.get('time','?')}\n"
            f"{body}")
