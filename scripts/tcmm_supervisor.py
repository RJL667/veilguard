"""TCMM supervisor — keep tcmm-service alive across crashes.

Fixes the F15 cascade: TCMM (host Python process on :8811) dies
unpredictably during this dev workflow — `taskkill /F /T` against
related processes sweeps it, Docker Desktop crashes have killed it,
some agent-runtime restarts have correlated with TCMM dying.

This supervisor:
  1. Launches tcmm-service (server.py) as a subprocess
  2. Polls its /health every N seconds
  3. If the process is dead OR /health doesn't respond for `dead_after_s`
     seconds, kills the survivor and relaunches.
  4. Logs every event so you can see when/why TCMM died

Run with::
    python scripts/tcmm_supervisor.py

To stop: Ctrl-C — supervisor terminates the child cleanly.

Env vars (optional):
    VEILGUARD_TCMM_SERVER_PY  default `~/.veilguard/mcp-tools/tcmm-service/server.py`
    TCMM_PORT                 default 8811 (also passed through to the child)
    VEILGUARD_TCMM_HEALTH_INTERVAL_S   default 10
    VEILGUARD_TCMM_DEAD_AFTER_S        default 30  (consecutive bad probes before relaunch)
    NLP_BACKEND               default ai_studio
    VEILGUARD_NLP_OOP_DISABLED  default 0
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now_str()}] [tcmm-supervisor] {msg}", flush=True)


def _resolve_server_py() -> Path:
    raw = os.environ.get("VEILGUARD_TCMM_SERVER_PY", "")
    if raw:
        p = Path(raw).expanduser()
        if p.is_file():
            return p
    # Default lookup paths. [DEDUP_2026-06-03] The local-first dev tree is the
    # SINGLE source of truth — the old default pointed at a .veilguard/mcp-tools
    # COPY that drifted from this one. Prefer the dev-tree master now; the VM
    # path stays last so this script still works if ever run on the VM.
    candidates = [
        Path("C:/Users/rudol/Documents/veilguard/services/tcmm-service/server.py"),
        Path.home() / "Documents" / "veilguard" / "services" / "tcmm-service" / "server.py",
        Path("/home/rudol/veilguard/services/tcmm-service/server.py"),
        Path("/home/rudol/veilguard/mcp-tools/tcmm-service/server.py"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise RuntimeError(
        "couldn't find tcmm-service/server.py — set "
        "VEILGUARD_TCMM_SERVER_PY=<path> and retry"
    )


def _is_healthy(port: int, timeout_s: float = 3.0) -> bool:
    """Hit /health.  True iff HTTP 200."""
    try:
        with urllib.request.urlopen(
            f"http://localhost:{port}/health", timeout=timeout_s
        ) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _launch_tcmm(server_py: Path, port: int) -> subprocess.Popen:
    """Spawn server.py with the env we want.  Returns the Popen handle."""
    env = os.environ.copy()
    env["TCMM_PORT"] = str(port)
    env.setdefault("NLP_BACKEND", "ai_studio")
    env.setdefault("VEILGUARD_NLP_OOP_DISABLED", "0")
    cwd = server_py.parent
    log_path = (
        Path(os.environ.get("VEILGUARD_TCMM_LOG_DIR", str(cwd / "logs")))
        / f"tcmm-{time.strftime('%Y%m%d-%H%M%S')}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")
    _log(f"spawning TCMM: {server_py} (port={port}, log={log_path})")
    proc = subprocess.Popen(
        [sys.executable, str(server_py)],
        cwd=str(cwd),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    return proc


def _terminate(proc: subprocess.Popen, timeout_s: float = 8.0) -> None:
    """Try graceful then forceful termination."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=timeout_s)
            return
        except subprocess.TimeoutExpired:
            pass
        proc.kill()
        proc.wait(timeout=4.0)
    except Exception as e:
        _log(f"terminate error (continuing): {e}")


def main() -> int:
    port = int(os.environ.get("TCMM_PORT", "8811"))
    health_interval_s = float(
        os.environ.get("VEILGUARD_TCMM_HEALTH_INTERVAL_S", "10")
    )
    dead_after_s = float(os.environ.get("VEILGUARD_TCMM_DEAD_AFTER_S", "30"))
    server_py = _resolve_server_py()

    _log(
        f"starting; port={port}, health_interval={health_interval_s}s, "
        f"dead_after={dead_after_s}s, server_py={server_py}"
    )

    # If TCMM is already up on this port, supervisor stays passive — it
    # just monitors and only spawns when it's gone.  Prevents two
    # instances fighting over the port.
    if _is_healthy(port):
        _log("TCMM already healthy on this port — entering passive monitor")
        proc: subprocess.Popen | None = None
    else:
        proc = _launch_tcmm(server_py, port)

    bad_streak_started: float | None = None
    restart_count = 0
    stop_requested = False

    def _on_signal(signum, _frame):
        nonlocal stop_requested
        _log(f"got signal {signum}; shutting down")
        stop_requested = True

    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_signal)

    try:
        while not stop_requested:
            healthy = _is_healthy(port)
            # If we own a proc, also check it's still alive at the OS level.
            proc_alive = (proc is None) or (proc.poll() is None)

            if healthy:
                if bad_streak_started is not None:
                    _log(
                        f"TCMM recovered after "
                        f"{time.time() - bad_streak_started:.1f}s of bad probes"
                    )
                    bad_streak_started = None
            else:
                if bad_streak_started is None:
                    bad_streak_started = time.time()
                    _log(
                        f"TCMM /health unhealthy "
                        f"(proc_alive={proc_alive}); waiting up to "
                        f"{dead_after_s:.0f}s before relaunch"
                    )
                elapsed = time.time() - bad_streak_started
                if elapsed >= dead_after_s or not proc_alive:
                    exit_code = (
                        proc.returncode if (proc and proc.poll() is not None) else None
                    )
                    _log(
                        f"TCMM dead "
                        f"(unhealthy_for={elapsed:.1f}s, proc_alive={proc_alive}, "
                        f"exit_code={exit_code}); relaunching"
                    )
                    if proc is not None:
                        _terminate(proc)
                    proc = _launch_tcmm(server_py, port)
                    restart_count += 1
                    bad_streak_started = None
                    _log(f"TCMM relaunched (restart_count={restart_count})")

            time.sleep(health_interval_s)
    finally:
        if proc is not None and proc.poll() is None:
            _log("terminating child TCMM on supervisor exit")
            _terminate(proc)
        _log(f"supervisor exiting; total_restarts={restart_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
