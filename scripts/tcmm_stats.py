"""
TCMM + LanceDB performance/state snapshot.

Captures one-line JSON per invocation to /var/log/tcmm-stats.jsonl
(also stdout) — append-only, easy to tail/grep/jq. Designed to be
called from a systemd timer every few minutes so you can see drift
in RSS, fragment counts, version counts, and /health latency over
time without manually SSH'ing in.

Captured per snapshot:
  - timestamp (epoch + iso)
  - host: free_mb, used_mb, available_mb, swap_used_mb
  - tcmm: pid, rss_mb, vsz_mb, etime_seconds, healthy, latency_ms
  - lance.<table>: data_mb, versions_mb, indices_mb, fragments, versions, indices

Tail in real time:
    tail -f /var/log/tcmm-stats.jsonl | jq

Trend RSS over the last hour:
    tail -200 /var/log/tcmm-stats.jsonl | jq '[.ts, .tcmm.rss_mb] | @tsv'

Trend archive.lance fragments:
    tail -200 /var/log/tcmm-stats.jsonl \
      | jq '[.ts_iso, .lance["archive.lance"].fragments] | @tsv'

Tunables via env vars:
    LANCE_DB_DIR        — path to tcmm.db (default below)
    TCMM_HEALTH_URL     — health endpoint (default http://localhost:8811/health)
    TCMM_STATS_LOG      — output JSONL path (default /var/log/tcmm-stats.jsonl)
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request as urlreq
from urllib.error import URLError

DB_DIR = Path(os.environ.get(
    "LANCE_DB_DIR",
    "/home/rudol/veilguard/tcmm-data/veilguard/tcmm.db",
))
HEALTH_URL = os.environ.get("TCMM_HEALTH_URL", "http://localhost:8811/health")
STATS_LOG = Path(os.environ.get("TCMM_STATS_LOG", "/var/log/tcmm-stats.jsonl"))

# Tables we care about. Order: most-frequently-changing first so partial
# captures (if the script is killed) still capture the interesting bits.
TABLES = [
    "archive.lance",
    "embeddings.lance",
    "pii_audit.lance",
    "sparse_archive.lance",
    "sparse_claims.lance",
    "sparse_topic.lance",
    "sparse_entity.lance",
    "_sequences.lance",
    "dream_archive.lance",
    "sparse_dream.lance",
    "sparse_dream_topic.lance",
    "sparse_dream_entity.lance",
    "sparse.lance",
]


def dir_size_mb(path: Path) -> float:
    if not path.is_dir():
        return 0.0
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return round(total / (1024 * 1024), 2)


def count_files(path: Path, glob: str) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for _ in path.glob(glob))


def count_subdirs(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.is_dir())


def host_memory() -> dict:
    """Parse /proc/meminfo for memory state."""
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                m = re.match(r"^(\w+):\s+(\d+)\s+kB", line)
                if m:
                    info[m.group(1)] = int(m.group(2))
    except OSError:
        return {}
    return {
        "total_mb": info.get("MemTotal", 0) // 1024,
        "free_mb": info.get("MemFree", 0) // 1024,
        "available_mb": info.get("MemAvailable", 0) // 1024,
        "buffers_mb": info.get("Buffers", 0) // 1024,
        "cached_mb": info.get("Cached", 0) // 1024,
        "swap_total_mb": info.get("SwapTotal", 0) // 1024,
        "swap_free_mb": info.get("SwapFree", 0) // 1024,
    }


def tcmm_process_stats() -> dict:
    """Resolve the TCMM systemd MainPID and read its /proc stats.

    Returns {"running": False} if the service isn't running or the PID
    can't be read. Doesn't import psutil — pure stdlib so the script
    runs anywhere Python 3.8+ is available.
    """
    try:
        out = subprocess.run(
            ["systemctl", "show", "veilguard-tcmm", "-p", "MainPID", "--value"],
            capture_output=True, text=True, timeout=3,
        )
        pid_str = out.stdout.strip()
        if not pid_str or pid_str == "0":
            return {"running": False}
        pid = int(pid_str)
    except (subprocess.SubprocessError, ValueError):
        return {"running": False}

    proc = Path(f"/proc/{pid}")
    if not proc.is_dir():
        return {"running": False, "pid": pid}

    stats: dict = {"running": True, "pid": pid}

    # /proc/<pid>/status: VmRSS, VmSize, etc. (in kB)
    try:
        with open(proc / "status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    stats["rss_mb"] = int(line.split()[1]) // 1024
                elif line.startswith("VmSize:"):
                    stats["vsz_mb"] = int(line.split()[1]) // 1024
                elif line.startswith("Threads:"):
                    stats["threads"] = int(line.split()[1])
    except OSError:
        pass

    # /proc/<pid>/stat: field 22 is starttime in clock ticks since boot.
    try:
        with open(proc / "stat") as f:
            fields = f.read().split()
        clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        start_ticks = int(fields[21])
        with open("/proc/uptime") as f:
            uptime_s = float(f.read().split()[0])
        stats["etime_seconds"] = round(uptime_s - start_ticks / clk_tck, 1)
    except (OSError, ValueError, IndexError):
        pass

    return stats


def tcmm_health() -> dict:
    """Hit the TCMM /health endpoint with a 4s timeout, measure latency."""
    t0 = time.time()
    try:
        with urlreq.urlopen(HEALTH_URL, timeout=4) as resp:
            elapsed = time.time() - t0
            return {
                "healthy": resp.status == 200,
                "status_code": resp.status,
                "latency_ms": round(elapsed * 1000, 1),
            }
    except URLError as e:
        return {
            "healthy": False,
            "error": str(e.reason if hasattr(e, "reason") else e),
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }


def lance_table_stats(table_path: Path) -> dict:
    """Filesystem-level stats for one Lance table. Avoids importing lance
    so this script runs even when TCMM is mid-restart and lance imports
    might temporarily race shared state."""
    if not table_path.is_dir():
        return {"present": False}

    return {
        "present": True,
        "total_mb": dir_size_mb(table_path),
        "data_mb": dir_size_mb(table_path / "data"),
        "versions_mb": dir_size_mb(table_path / "_versions"),
        "indices_mb": dir_size_mb(table_path / "_indices"),
        "transactions_mb": dir_size_mb(table_path / "_transactions"),
        "deletions_mb": dir_size_mb(table_path / "_deletions"),
        "fragments": count_files(table_path / "data", "*.lance"),
        "versions": count_files(table_path / "_versions", "*.manifest"),
        "indices": count_subdirs(table_path / "_indices"),
    }


def collect() -> dict:
    now = time.time()
    snapshot = {
        "ts": int(now),
        "ts_iso": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "host": host_memory(),
        "tcmm": {**tcmm_process_stats(), **tcmm_health()},
        "lance": {t: lance_table_stats(DB_DIR / t) for t in TABLES},
    }
    # Roll-up totals across tables for quick scanning without jq gymnastics.
    snapshot["lance_total"] = {
        "total_mb": round(sum(s.get("total_mb", 0) for s in snapshot["lance"].values()), 2),
        "fragments": sum(s.get("fragments", 0) for s in snapshot["lance"].values()),
        "versions": sum(s.get("versions", 0) for s in snapshot["lance"].values()),
    }
    return snapshot


def main() -> int:
    snap = collect()
    line = json.dumps(snap, separators=(",", ":"))

    # Append to log file (best-effort — don't crash if /var/log isn't
    # writable; the systemd unit runs as root so this normally works).
    try:
        STATS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(STATS_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as e:
        print(f"WARN: could not write to {STATS_LOG}: {e}", file=sys.stderr)

    # Always also print to stdout for ad-hoc runs / one-off invocations.
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
