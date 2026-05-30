"""Host-level CPU / memory / disk / load metrics.

Reads from ``/host/proc`` when the dashboard runs in a container with
the host's ``/proc`` bind-mounted there (the deployed pattern); falls
back to the local ``/proc`` so it still works on bare-metal dev.

We deliberately do NOT depend on ``psutil`` here — the in-tree readers
are cheap, deterministic, and avoid pulling in another C extension.

# Threading
``cpu_percent()`` is calculated against a snapshot stored in process
memory; the first call returns 0% (no baseline) and subsequent calls
return the rolling delta. The dashboard refreshes every 10s, so the
second poll onward gives a real number.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

# Bind-mount target inside the container; falls back to the actual /proc
PROC = Path("/host/proc") if Path("/host/proc/stat").exists() else Path("/proc")
ROOT_FS = Path("/host") if Path("/host/proc").exists() else Path("/")


_last_cpu_snapshot: dict[str, Any] = {"ts": 0.0, "totals": None}


def _read_cpu_totals() -> tuple[int, int]:
    """Return ``(busy_jiffies, total_jiffies)`` from /proc/stat first line."""
    line = (PROC / "stat").read_text().splitlines()[0]
    fields = [int(x) for x in line.split()[1:]]
    # user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
    total = sum(fields)
    return total - idle, total


def cpu_percent() -> float:
    """Rolling CPU% across all cores (since previous poll)."""
    global _last_cpu_snapshot
    busy, total = _read_cpu_totals()
    prev = _last_cpu_snapshot.get("totals")
    _last_cpu_snapshot = {"ts": time.time(), "totals": (busy, total)}
    if not prev:
        return 0.0
    db, dt = busy - prev[0], total - prev[1]
    if dt <= 0:
        return 0.0
    return round(100.0 * db / dt, 1)


def memory() -> dict[str, Any]:
    """Memory totals + used/available, all in bytes + percent."""
    fields: dict[str, int] = {}
    for line in (PROC / "meminfo").read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        fields[k.strip()] = int(v.strip().split()[0]) * 1024  # kB → bytes
    total = fields.get("MemTotal", 0)
    avail = fields.get("MemAvailable", fields.get("MemFree", 0))
    used = max(total - avail, 0)
    pct = round(100.0 * used / total, 1) if total else 0.0
    return {"total": total, "used": used, "available": avail, "percent": pct}


def loadavg() -> list[float]:
    """1-, 5-, 15-minute load averages."""
    parts = (PROC / "loadavg").read_text().split()
    return [float(p) for p in parts[:3]]


def disk(path: str = "/tcmm-data") -> dict[str, Any]:
    """Disk usage for the LanceDB path (or fallback to /)."""
    target = Path(path) if Path(path).exists() else Path("/")
    try:
        total, used, free = shutil.disk_usage(target)
        pct = round(100.0 * used / total, 1) if total else 0.0
        return {
            "path": str(target),
            "total": total,
            "used": used,
            "free": free,
            "percent": pct,
        }
    except Exception as e:
        return {"path": str(target), "error": str(e)}


def uptime_seconds() -> float:
    try:
        return float((PROC / "uptime").read_text().split()[0])
    except Exception:
        return 0.0


def overview() -> dict[str, Any]:
    """One-shot snapshot for the dashboard."""
    return {
        "cpu_percent": cpu_percent(),
        "cpu_count": os.cpu_count(),
        "memory": memory(),
        "load_avg": loadavg(),
        "disk_lance": disk(os.environ.get("ADMIN_LANCE_DIR_DISK", "/tcmm-data")),
        "uptime_seconds": uptime_seconds(),
        "host_proc_mounted": PROC == Path("/host/proc"),
    }
