"""Pretty-printed snapshot of TCMM + LanceDB + VM memory.

Reads the latest line from /var/log/tcmm-stats.jsonl (written by
tcmm_stats.py via the systemd timer) and renders a human-friendly
table. If the log is empty or stale, runs tcmm_stats.py first to
generate a fresh snapshot.

Usage:
    python3 scripts/tcmm_health_pretty.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LOG = Path(os.environ.get("TCMM_STATS_LOG", "/var/log/tcmm-stats.jsonl"))
STATS_SCRIPT = Path("/home/rudol/veilguard/scripts/tcmm_stats.py")


def latest_snapshot() -> dict:
    """Return the most recent stats snapshot, refreshing it if stale."""
    if LOG.is_file():
        try:
            last = LOG.read_text(encoding="utf-8").strip().splitlines()[-1]
            snap = json.loads(last)
            age = time.time() - snap.get("ts", 0)
            if age < 60:
                return snap
        except (OSError, json.JSONDecodeError, IndexError):
            pass
    # Stale or unreadable — refresh
    if STATS_SCRIPT.is_file():
        subprocess.run([sys.executable, str(STATS_SCRIPT)],
                        capture_output=True, timeout=30)
        last = LOG.read_text(encoding="utf-8").strip().splitlines()[-1]
        return json.loads(last)
    raise SystemExit("No stats available and no stats script to refresh")


def fmt_mb(mb: float) -> str:
    if mb >= 1024:
        return f"{mb/1024:.2f} GB"
    return f"{mb:.1f} MB"


def main() -> int:
    s = latest_snapshot()
    age_s = int(time.time() - s["ts"])

    print()
    print(f"=== Snapshot from {s['ts_iso']} ({age_s}s ago) ===")

    # ── VM memory ───────────────────────────────────────────────────
    h = s["host"]
    total = h["total_mb"]
    avail = h["available_mb"]
    used = total - avail
    pct = used / total * 100 if total else 0
    swap_used = h["swap_total_mb"] - h["swap_free_mb"]

    print()
    print("### VM MEMORY ###")
    print(f"  total           : {fmt_mb(total):>10s}")
    print(f"  used            : {fmt_mb(used):>10s}  ({pct:.0f}%)")
    print(f"  available       : {fmt_mb(avail):>10s}")
    print(f"  free            : {fmt_mb(h['free_mb']):>10s}")
    print(f"  buff/cache      : {fmt_mb(h['buffers_mb']+h['cached_mb']):>10s}")
    if h["swap_total_mb"]:
        print(f"  swap used       : {fmt_mb(swap_used):>10s} of {fmt_mb(h['swap_total_mb'])}")
    else:
        print(f"  swap            : (none configured)")

    # ── TCMM process ────────────────────────────────────────────────
    t = s["tcmm"]
    print()
    print("### TCMM PROCESS ###")
    if not t.get("running"):
        print(f"  NOT RUNNING")
    else:
        rss_mb = t.get("rss_mb", 0)
        vsz_mb = t.get("vsz_mb", 0)
        et = t.get("etime_seconds", 0)
        et_h = et / 3600
        print(f"  pid             : {t.get('pid', '?')}")
        print(f"  rss             : {fmt_mb(rss_mb):>10s}  ({rss_mb/total*100:.1f}% of VM)")
        print(f"  vsz             : {fmt_mb(vsz_mb):>10s}")
        print(f"  threads         : {t.get('threads', '?')}")
        print(f"  uptime          : {et:.0f}s ({et_h:.1f}h)")
        if t.get("healthy"):
            print(f"  /health         : 200 in {t.get('latency_ms','?')}ms")
        else:
            print(f"  /health         : UNHEALTHY ({t.get('error', t.get('status_code', '?'))})")

    # ── Lance per-table breakdown ───────────────────────────────────
    print()
    print("### LANCE TABLES ###")
    cols = (
        f"  {'table':<28s} {'total':>10s} {'data':>10s} "
        f"{'versions':>9s} {'indices':>9s} {'frags':>6s} {'vers':>5s}"
    )
    print(cols)
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*9} {'-'*9} {'-'*6} {'-'*5}")

    populated = []
    empty = []
    for name, st in s["lance"].items():
        if not st.get("present"):
            empty.append(f"{name} (missing)")
            continue
        if st["total_mb"] == 0:
            empty.append(name)
            continue
        populated.append((name, st))

    # Sort by total_mb descending so the big ones are at the top
    populated.sort(key=lambda x: -x[1]["total_mb"])
    for name, st in populated:
        print(
            f"  {name:<28s} "
            f"{fmt_mb(st['total_mb']):>10s} "
            f"{fmt_mb(st['data_mb']):>10s} "
            f"{fmt_mb(st['versions_mb']):>9s} "
            f"{fmt_mb(st['indices_mb']):>9s} "
            f"{st['fragments']:>6d} "
            f"{st['versions']:>5d}"
        )

    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*9} {'-'*9} {'-'*6} {'-'*5}")
    rt = s["lance_total"]
    print(
        f"  {'TOTAL':<28s} "
        f"{fmt_mb(rt['total_mb']):>10s} "
        f"{'':>10s} {'':>9s} {'':>9s} "
        f"{rt['fragments']:>6d} "
        f"{rt['versions']:>5d}"
    )

    if empty:
        print()
        print(f"  empty tables: {', '.join(empty)}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
