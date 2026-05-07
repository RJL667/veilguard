"""perf_audit.py — append perf-test results to a Lance audit table.

Creates the ``perf_metrics`` table on first run if it doesn't exist;
otherwise opens and appends. Each call to ``record(...)`` adds one row.
Schema is forward-compatible: new columns can be added later via Lance's
schema-evolution without rewriting existing rows.

Usage from the perf test script:
    python perf_audit.py record --label cold-1 --conv perf-cold-X-1 \
        --e2e_ms 6720 --ttfb_ms 6720 --pre_ms 1660 --status 200 \
        --model claude-haiku-4-5 --pass cold

Usage to read recent rows for spot-check:
    python perf_audit.py tail --limit 20
"""
import argparse
import os
import sys
import time
import uuid
import lancedb
import pyarrow as pa

DB_DIR = os.environ.get(
    "TCMM_DATA_DIR",
    "/home/rudol/veilguard/tcmm-data/veilguard/tcmm.db",
)
TABLE = "perf_metrics"

# Schema. Keep it explicit (rather than letting from_pylist infer) so
# every row has the same column types regardless of which test wrote it.
SCHEMA = pa.schema([
    pa.field("ts",            pa.float64()),     # unix seconds
    pa.field("run_id",        pa.string()),      # group by test invocation
    pa.field("pass_label",    pa.string()),      # "cold" or "warm"
    pa.field("call_label",    pa.string()),      # "cold-1", "warm-3", etc.
    pa.field("conv_id",       pa.string()),
    pa.field("user_id",       pa.string()),
    pa.field("model",         pa.string()),
    pa.field("prompt_chars",  pa.int64()),
    # End-to-end (curl-perceived).
    pa.field("e2e_ms",        pa.int64()),
    pa.field("ttfb_ms",       pa.int64()),
    pa.field("http_status",   pa.int64()),
    pa.field("response_bytes", pa.int64()),
    # TCMM legs (from /var/log/veilguard-tcmm.log [PRE]/[POST] entries).
    pa.field("tcmm_pre_ms",   pa.int64()),
    pa.field("tcmm_post_ms",  pa.int64()),
    pa.field("recalled",      pa.int64()),
    # Anthropic accounting (from pii-proxy [CACHE] log line).
    pa.field("tokens_input",  pa.int64()),
    pa.field("tokens_output", pa.int64()),
    pa.field("cache_create",  pa.int64()),
    pa.field("cache_read",    pa.int64()),
    # Free-form notes / diagnostics column.
    pa.field("notes",         pa.string()),
])


def _open_or_create():
    db = lancedb.connect(DB_DIR)
    if TABLE not in db.table_names():
        # Empty table with the schema. Lance happily creates a 0-row
        # table from an Arrow Table that has no rows.
        empty = pa.Table.from_pylist([], schema=SCHEMA)
        db.create_table(TABLE, data=empty, mode="create")
    return db.open_table(TABLE)


def cmd_record(args):
    tbl = _open_or_create()
    row = {
        "ts": float(args.ts) if args.ts else time.time(),
        "run_id": args.run_id,
        "pass_label": args.pass_label,
        "call_label": args.call_label,
        "conv_id": args.conv,
        "user_id": args.user or "",
        "model": args.model or "",
        "prompt_chars": int(args.prompt_chars or 0),
        "e2e_ms": int(args.e2e_ms or 0),
        "ttfb_ms": int(args.ttfb_ms or 0),
        "http_status": int(args.status or 0),
        "response_bytes": int(args.bytes_recv or 0),
        "tcmm_pre_ms": int(args.pre_ms or 0),
        "tcmm_post_ms": int(args.post_ms or 0),
        "recalled": int(args.recalled or 0),
        "tokens_input": int(args.tin or 0),
        "tokens_output": int(args.tout or 0),
        "cache_create": int(args.cc or 0),
        "cache_read": int(args.cr or 0),
        "notes": args.notes or "",
    }
    rec = pa.Table.from_pylist([row], schema=SCHEMA)
    tbl.add(rec)
    print(f"  appended row: {args.call_label} e2e={args.e2e_ms}ms pre={args.pre_ms}ms")


def cmd_tail(args):
    tbl = _open_or_create()
    arr = tbl.to_arrow()
    print(f"perf_metrics rows: {len(arr)}")
    # Sort by ts desc, take top N
    df = arr.to_pandas().sort_values("ts", ascending=False).head(args.limit)
    cols = ["call_label", "pass_label", "model", "e2e_ms", "tcmm_pre_ms",
            "recalled", "cache_read", "cache_create", "tokens_input",
            "tokens_output", "http_status"]
    print(df[cols].to_string(index=False))


def cmd_summary(args):
    tbl = _open_or_create()
    df = tbl.to_arrow().to_pandas()
    if len(df) == 0:
        print("(no rows yet)")
        return
    if args.run_id:
        df = df[df["run_id"] == args.run_id]
    print(f"Aggregate over {len(df)} rows:")
    for pl in ("cold", "warm"):
        sub = df[df["pass_label"] == pl]
        if len(sub) == 0:
            continue
        print(f"\n  pass={pl}  n={len(sub)}")
        for c in ("e2e_ms", "tcmm_pre_ms", "tokens_input", "tokens_output",
                 "cache_create", "cache_read"):
            if c not in sub or sub[c].sum() == 0:
                continue
            print(f"    {c:<14} avg={sub[c].mean():>8.0f}  "
                  f"p50={sub[c].median():>8.0f}  "
                  f"min={sub[c].min():>8.0f}  max={sub[c].max():>8.0f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)

    pr = sp.add_parser("record")
    pr.add_argument("--ts", type=float)
    pr.add_argument("--run_id", required=True)
    pr.add_argument("--pass_label", choices=["cold", "warm"], required=True,
                    dest="pass_label")
    pr.add_argument("--call_label", required=True)
    pr.add_argument("--conv", required=True)
    pr.add_argument("--user", default="")
    pr.add_argument("--model", default="")
    pr.add_argument("--prompt_chars", type=int, default=0)
    pr.add_argument("--e2e_ms", type=int)
    pr.add_argument("--ttfb_ms", type=int)
    pr.add_argument("--status", type=int)
    pr.add_argument("--bytes_recv", type=int)
    pr.add_argument("--pre_ms", type=int)
    pr.add_argument("--post_ms", type=int)
    pr.add_argument("--recalled", type=int)
    pr.add_argument("--tin", type=int)
    pr.add_argument("--tout", type=int)
    pr.add_argument("--cc", type=int)
    pr.add_argument("--cr", type=int)
    pr.add_argument("--notes", default="")
    pr.set_defaults(func=cmd_record)

    pt = sp.add_parser("tail")
    pt.add_argument("--limit", type=int, default=20)
    pt.set_defaults(func=cmd_tail)

    ps = sp.add_parser("summary")
    ps.add_argument("--run_id", default="")
    ps.set_defaults(func=cmd_summary)

    args = p.parse_args()
    args.func(args)
