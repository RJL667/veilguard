"""Daemon tools — persistent background monitors (KAIROS-inspired)."""

import asyncio
import json
import logging
import time

from config import ROLES, DEFAULT_BACKEND, DAEMON_DATA_DIR
from core import state
from core.state import Daemon
from core.llm import call_llm, resolve_backend
from utils.afk import is_afk

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
