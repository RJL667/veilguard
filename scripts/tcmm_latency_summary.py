"""Parse latency from existing logs (pii-proxy + TCMM) and report
percentiles. Read-only — never writes.

Per turn the pii-proxy logs ~5 timestamps, in order:

    >>> POST [anthropic] /v1/messages (stream=..., conv=...)
        ... (PII redaction, request prep, header-rewrite) ...
    HTTP Request: POST http://172.17.0.1:8811/pre_request "200 OK"
    [TCMM] Memory=N chars, format=anthropic
        ... (proxy POSTs to Anthropic) ...
    HTTP Request: POST https://api.anthropic.com/v1/messages "200 OK"
    INFO:     172.18.0.3:... - "POST /anthropic/v1/messages HTTP/1.1" 200 OK
    HTTP Request: POST http://172.17.0.1:8811/post_response "200 OK"

We pair them up sequentially (single-stream service, requests rarely
overlap by more than one) and compute four buckets:

    TCMM pre_request   : >>> -> pre_request 200
    Anthropic call     : pre_request 200 -> anthropic.com 200
    TCMM post_response : anthropic 200 -> post_response 200
    pii-proxy total    : >>> -> uvicorn access-log 200 (end of turn)

Usage:
    python3 scripts/tcmm_latency_summary.py
    python3 scripts/tcmm_latency_summary.py --window 4h
"""
import argparse
import re
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime


_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
_OPEN = re.compile(r">>> POST \[anthropic\]")
_PRE_DONE = re.compile(r"HTTP Request: POST http://[^/]+/pre_request \"HTTP/1\.1 200 OK\"")
_ANTHROPIC_DONE = re.compile(r"HTTP Request: POST https://api\.anthropic\.com/v1/messages \"HTTP/1\.1 (\d+)")
_PROXY_DONE = re.compile(r'INFO:\s+\S+\s+- "POST /anthropic/v1/messages HTTP/1\.1" (\d+)')
_POST_DONE = re.compile(r"HTTP Request: POST http://[^/]+/post_response \"HTTP/1\.1 200 OK\"")


def _parse_ts(line: str) -> float | None:
    m = _TS.match(line)
    if not m:
        return None
    try:
        return datetime.strptime(
            m.group(1) + "." + m.group(2) + "000",
            "%Y-%m-%d %H:%M:%S.%f",
        ).timestamp()
    except ValueError:
        return None


def parse_proxy(window: str) -> list[dict]:
    """Walk the pii-proxy log and pair >>> with the next milestones.

    Tolerant of missing checkpoints — if a stream doesn't log the
    closing INFO line we just leave that field None and still report
    the partial data.
    """
    out = subprocess.run(
        ["docker", "logs", "veilguard-pii-proxy-1", "--since", window],
        capture_output=True, text=True, errors="replace",
    )
    raw = out.stdout + out.stderr

    turns: list[dict] = []
    current: dict | None = None

    for line in raw.splitlines():
        ts = _parse_ts(line)
        if ts is None:
            # Lines without a timestamp (e.g. the INFO uvicorn line) are
            # still meaningful — but we need to give them the timestamp
            # of the most recent timestamped line. Skip; only timestamped
            # lines drive state.
            if current is None:
                continue
            # uvicorn access log line — we can detect it but have no
            # timestamp; just mark proxy_done by the next ts we see.
            mp = _PROXY_DONE.search(line)
            if mp:
                current["proxy_done_pending"] = int(mp.group(1))
            continue

        # Promote a pending proxy_done to ts (uvicorn line had no ts of
        # its own, so we backdate it ~0.001s before the next event).
        if current and current.get("proxy_done_pending") is not None and current.get("proxy_done") is None:
            current["proxy_done"] = ts
            current["proxy_status"] = current.pop("proxy_done_pending")

        if _OPEN.search(line):
            if current and current.get("proxy_done") is None:
                # Old turn never closed cleanly; archive partial
                turns.append(current)
            current = {"open": ts}
        elif current is not None:
            if _PRE_DONE.search(line) and "pre_done" not in current:
                current["pre_done"] = ts
            elif (m := _ANTHROPIC_DONE.search(line)):
                current["anthropic_done"] = ts
                current["anthropic_status"] = int(m.group(1))
            elif (m := _PROXY_DONE.search(line)):
                current["proxy_done"] = ts
                current["proxy_status"] = int(m.group(1))
            elif _POST_DONE.search(line):
                current["post_done"] = ts
                # post_response is the last event in a turn
                turns.append(current)
                current = None
    if current is not None:
        turns.append(current)
    return turns


def buckets(turns: list[dict]) -> dict[str, list[float]]:
    """Convert raw turn dicts into per-bucket millisecond lists."""
    out: dict[str, list[float]] = defaultdict(list)
    for t in turns:
        if "open" in t and "pre_done" in t:
            out["TCMM pre_request"].append((t["pre_done"] - t["open"]) * 1000)
        if "pre_done" in t and "anthropic_done" in t:
            out["Anthropic call  "].append((t["anthropic_done"] - t["pre_done"]) * 1000)
        if "anthropic_done" in t and "post_done" in t:
            out["TCMM post_response"].append((t["post_done"] - t["anthropic_done"]) * 1000)
        if "open" in t and "proxy_done" in t:
            out["pii-proxy total"].append((t["proxy_done"] - t["open"]) * 1000)
    return out


def parse_tcmm_perf(window_lines: int = 20000) -> dict[str, list[float]]:
    """TCMM logs PERF lines with `+Xms` increments. Sample top buckets."""
    out = subprocess.run(
        ["tail", "-n", str(window_lines), "/var/log/veilguard-tcmm.log"],
        capture_output=True, text=True, errors="replace",
    )
    raw = out.stdout

    # Pattern: optional indent, [PERF/SECTION] some-label +Xms
    # Sometimes ends with total=Xms instead of +Xms
    pat = re.compile(r"\[PERF/([\w_]+)\]\s+([\w\s+\-_/().]+?)(?:\s+\+([0-9.]+)ms|\s+total=([0-9.]+)ms)")
    by_label: dict[str, list[float]] = defaultdict(list)
    for line in raw.splitlines():
        for m in pat.finditer(line):
            section = m.group(1)
            label = m.group(2).strip()
            ms = float(m.group(3) or m.group(4))
            key = f"[{section}] {label}"
            by_label[key].append(ms)
    return dict(by_label)


def percentiles(vals: list[float]) -> dict | None:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    pick = lambda q: s[max(0, min(n - 1, int(q * n)))]
    return {
        "n": n,
        "min": s[0],
        "p50": pick(0.50),
        "p95": pick(0.95),
        "p99": pick(0.99),
        "max": s[-1],
        "mean": statistics.mean(s),
    }


def fmt(ms: float) -> str:
    if ms >= 1000:
        return f"{ms/1000:.2f}s"
    if ms >= 10:
        return f"{ms:.0f}ms"
    return f"{ms:.1f}ms"


def render_table(title: str, data: dict[str, list[float]],
                  sort_key: str = "n", top: int | None = None) -> None:
    print(f"\n### {title} ###")
    if not data:
        print("  (no data)")
        return
    print(
        f"  {'bucket':<28s}  {'n':>4s}  {'min':>7s}  "
        f"{'p50':>7s}  {'p95':>7s}  {'p99':>7s}  {'max':>7s}  {'mean':>7s}"
    )
    print(
        f"  {'-'*28}  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*7}  "
        f"{'-'*7}  {'-'*7}  {'-'*7}"
    )
    items = []
    for label, vals in data.items():
        p = percentiles(vals)
        if p:
            items.append((label, p))
    if sort_key == "p95":
        items.sort(key=lambda kv: -kv[1]["p95"])
    elif sort_key == "n":
        items.sort(key=lambda kv: -kv[1]["n"])
    if top:
        items = items[:top]
    for label, p in items:
        print(
            f"  {label:<28s}  {p['n']:>4d}  "
            f"{fmt(p['min']):>7s}  {fmt(p['p50']):>7s}  "
            f"{fmt(p['p95']):>7s}  {fmt(p['p99']):>7s}  "
            f"{fmt(p['max']):>7s}  {fmt(p['mean']):>7s}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="1h",
                    help="Docker logs --since window. Default 1h.")
    ap.add_argument("--tcmm-lines", type=int, default=30000,
                    help="How many TCMM log lines to sample (default 30k = ~1h)")
    args = ap.parse_args()

    print(f"=== latency summary, window={args.window} ===")

    turns = parse_proxy(args.window)
    print(f"  parsed {len(turns)} pii-proxy turns "
          f"(complete = those with both >>> and proxy_done markers)")
    render_table("end-to-end per turn (pii-proxy view)", buckets(turns))

    tcmm = parse_tcmm_perf(args.tcmm_lines)
    render_table(
        "TCMM internal PERF buckets (top 12 by call count)",
        tcmm, sort_key="n", top=12,
    )

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
