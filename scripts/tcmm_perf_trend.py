"""Render trend view of TCMM + LanceDB metrics from the stats JSONL log.

Read-only. Does not touch any state. Defaults to the last ~2 hours
(24 snapshots @ 5-min cadence). Pass --hours N to widen.

Usage:
    python3 scripts/tcmm_perf_trend.py
    python3 scripts/tcmm_perf_trend.py --hours 24
    python3 scripts/tcmm_perf_trend.py --since-restart  # only since latest TCMM PID change
"""
import argparse
import json
import os
import sys
from pathlib import Path

LOG = Path(os.environ.get("TCMM_STATS_LOG", "/var/log/tcmm-stats.jsonl"))


def load_snapshots(hours: float = 2.0, since_restart: bool = False) -> list:
    if not LOG.is_file():
        print(f"ERROR: {LOG} not found", file=sys.stderr)
        return []
    rows = []
    with open(LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return []

    # Window
    if since_restart:
        # Walk backwards, keep snapshots until pid changes
        latest_pid = rows[-1].get("tcmm", {}).get("pid")
        kept = []
        for r in reversed(rows):
            if r.get("tcmm", {}).get("pid") != latest_pid:
                break
            kept.append(r)
        rows = list(reversed(kept))
    else:
        cutoff = rows[-1]["ts"] - int(hours * 3600)
        rows = [r for r in rows if r["ts"] >= cutoff]

    return rows


def fmt_mb(mb: float) -> str:
    if mb >= 1024:
        return f"{mb/1024:.1f}G"
    return f"{mb:.0f}M"


def render(rows: list) -> None:
    if not rows:
        print("(no snapshots in window)")
        return

    span_hours = (rows[-1]["ts"] - rows[0]["ts"]) / 3600

    print(f"\n=== {len(rows)} snapshots over {span_hours:.1f}h "
          f"({rows[0]['ts_iso'][:19]} → {rows[-1]['ts_iso'][:19]}) ===\n")

    # Header — keep widths fixed
    h = (f"  {'time':<19s}  {'vm-used':>8s}  {'rss':>6s}  {'health':>7s}  "
         f"{'archive':>9s}  {'frags':>6s}  {'vers':>6s}")
    print(h)
    print(f"  {'-'*19}  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*9}  {'-'*6}  {'-'*6}")

    # Table — sample every Nth so we keep <=30 lines visible
    step = max(1, len(rows) // 30)
    for r in rows[::step]:
        host = r["host"]
        used_mb = host["total_mb"] - host["available_mb"]
        tcmm = r["tcmm"]
        rss = tcmm.get("rss_mb", 0)
        lat = tcmm.get("latency_ms", 0)
        arc = r["lance"].get("archive.lance", {})
        arc_total = arc.get("total_mb", 0)
        arc_frags = arc.get("fragments", 0)
        arc_vers = arc.get("versions", 0)
        print(
            f"  {r['ts_iso'][:19]:<19s}  "
            f"{fmt_mb(used_mb):>8s}  "
            f"{fmt_mb(rss):>6s}  "
            f"{lat:>5.1f}ms  "
            f"{fmt_mb(arc_total):>9s}  "
            f"{arc_frags:>6d}  "
            f"{arc_vers:>6d}"
        )

    # Min/max/delta summary
    print()
    def stats(extract):
        vals = [extract(r) for r in rows]
        return min(vals), max(vals), vals[-1] - vals[0]

    rss_min, rss_max, rss_delta = stats(lambda r: r["tcmm"].get("rss_mb", 0))
    lat_min, lat_max, _ = stats(lambda r: r["tcmm"].get("latency_ms", 0))
    arc_total_min, arc_total_max, arc_total_delta = stats(
        lambda r: r["lance"].get("archive.lance", {}).get("total_mb", 0)
    )
    frag_min, frag_max, frag_delta = stats(
        lambda r: r["lance"].get("archive.lance", {}).get("fragments", 0)
    )
    ver_min, ver_max, ver_delta = stats(
        lambda r: r["lance"].get("archive.lance", {}).get("versions", 0)
    )

    print("  min/max over window:")
    print(f"    rss              : {rss_min}M → {rss_max}M  (delta over window: "
          f"{'+' if rss_delta >= 0 else ''}{rss_delta}M)")
    print(f"    /health latency  : {lat_min:.1f}ms → {lat_max:.1f}ms")
    print(f"    archive total    : {arc_total_min:.0f}M → {arc_total_max:.0f}M  "
          f"(delta: {'+' if arc_total_delta >= 0 else ''}{arc_total_delta:.0f}M)")
    print(f"    archive frags    : {frag_min} → {frag_max}  "
          f"(delta: {'+' if frag_delta >= 0 else ''}{frag_delta})")
    print(f"    archive versions : {ver_min} → {ver_max}  "
          f"(delta: {'+' if ver_delta >= 0 else ''}{ver_delta})")

    if span_hours > 0.1:
        print()
        rate_frags = frag_delta / span_hours
        rate_vers = ver_delta / span_hours
        rate_mb = arc_total_delta / span_hours
        print("  rate (per hour):")
        print(f"    fragments        : {'+' if rate_frags >= 0 else ''}{rate_frags:.0f}/h "
              f"({rate_frags/60:+.1f}/min)")
        print(f"    versions         : {'+' if rate_vers >= 0 else ''}{rate_vers:.0f}/h "
              f"({rate_vers/60:+.1f}/min)")
        print(f"    archive bytes    : {'+' if rate_mb >= 0 else ''}{rate_mb:.0f} MB/h")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--since-restart", action="store_true",
                    help="Only show snapshots since the latest TCMM PID")
    args = ap.parse_args()

    rows = load_snapshots(hours=args.hours, since_restart=args.since_restart)
    render(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
