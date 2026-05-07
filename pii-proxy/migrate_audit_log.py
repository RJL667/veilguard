"""One-shot migration: parse /app/logs/pii_audit.log into LanceDB.

Run inside the running pii-proxy container so we use the exact same
Lance connection / schema as live writes:

    sudo docker exec veilguard-pii-proxy-1 python /app/migrate_audit_log.py

The parser walks the text log, extracts the timestamp + direction +
conv_id + metadata + content for each entry, and enqueues rows through
``app.audit_db.AuditDB.enqueue``.  Timestamps are preserved
(``created_at`` overrides the default "now").

Safe to re-run — if you re-seed against a partially-migrated table you
will get duplicate rows (the log file is the source of truth and has no
unique id to dedupe against).  So only run on empty-table or
destructive-reset scenarios.  Use ``--dry-run`` to see what WOULD be
enqueued without writing.
"""
import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure ``app.audit_db`` resolves when run via ``python /app/migrate_audit_log.py``.
sys.path.insert(0, "/app")
from app.audit_db import AuditDB  # noqa: E402


# Log entry layout written by audit_log() in main.py:
#
#   2026-04-21 12:39:42,751
#   ================================================================================
#   [TO_LLM] conv=684c55b4-e81 model=claude-opus-4-7
#   ────────────────────────────────────────────────────────────────────────────────
#   <content — may span many lines, contains any characters>
#   ================================================================================
#
# We use a non-greedy multiline DOTALL match to pick up each block.
_ENTRY_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\s*\n"
    r"=+\n"
    r"\[(?P<direction>TO_LLM|FROM_LLM)\] conv=(?P<conv>\S+)(?:\s+(?P<extras>.+?))?\n"
    r"─+\n"
    r"(?P<content>.*?)\n"
    r"=+",
    re.DOTALL,
)

# k=v pairs in the header line (model=..., stream=...).  Uses \S+ on
# the value so we don't choke on model IDs with hyphens.
_KV_RE = re.compile(r"(\w+)=(\S+)")


def parse_entries(log_text: str):
    for m in _ENTRY_RE.finditer(log_text):
        ts_raw = m.group("ts")
        try:
            # strptime understands ``,%f`` for milliseconds.
            ts = datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S,%f").timestamp()
        except ValueError:
            continue
        extras = m.group("extras") or ""
        meta = dict(_KV_RE.findall(extras))
        yield {
            "ts": ts,
            "direction": m.group("direction"),
            "conversation_id": m.group("conv"),
            "model": meta.get("model") if meta.get("model") and meta.get("model") != "?" else None,
            "stream": meta.get("stream") in ("anthropic", "openai", "true"),
            "content": m.group("content").strip(),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/app/logs/pii_audit.log")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse + count without writing to DB")
    ap.add_argument("--limit", type=int, default=None,
                    help="migrate at most N entries (handy for spot-checks)")
    args = ap.parse_args()

    log_path = Path(args.path)
    if not log_path.exists():
        print(f"log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    text = log_path.read_text(encoding="utf-8", errors="replace")
    print(f"reading {log_path} ({len(text):,} bytes)")

    entries = list(parse_entries(text))
    print(f"parsed {len(entries)} entries")
    if args.limit:
        entries = entries[: args.limit]
        print(f"limit applied → migrating first {len(entries)}")

    if args.dry_run:
        # Show a sample — first + last 2 entries.
        for e in entries[:2] + entries[-2:]:
            preview = e["content"][:120].replace("\n", " ⏎ ")
            print(
                f"  {datetime.fromtimestamp(e['ts']).isoformat()} "
                f"{e['direction']:<9} conv={e['conversation_id'][:12]:<12} "
                f"model={e['model'] or '-':<24} len={len(e['content']):>6} "
                f"  {preview}"
            )
        return

    db = AuditDB.get()
    if db is None or db.__class__.__name__ == "_NullAuditDB":
        print("AuditDB unavailable — check VEILGUARD_AUDIT_DB_PATH.", file=sys.stderr)
        sys.exit(1)

    for e in entries:
        db.enqueue({
            "user_id":         "",   # legacy log — user_id not captured pre-migration
            "conversation_id": e["conversation_id"],
            "direction":       e["direction"],
            "model":           e["model"],
            "stream":          e["stream"],
            "content":         e["content"],
            "created_at":      e["ts"],
            "extra":           '{"source":"legacy_log_migration"}',
        })

    # The writer thread batches every few seconds.  Sleep long enough
    # for the queue to drain.  Worst case the migration undercounts
    # the confirmation (rows still land after we exit).
    print(f"enqueued {len(entries)} rows; waiting 5s for queue drain…")
    time.sleep(5)
    print("done")


if __name__ == "__main__":
    main()
