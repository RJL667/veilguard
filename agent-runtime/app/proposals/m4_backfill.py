"""Phase 7 M4 — one-shot historical migration of `task_comments[kind=comment]`
to TCMM `agent/<author_id>/observations/`.

After Phase 7 M4 wired `record_discussion_comment` as the canonical
destination for non-state-machine task comments, the legacy
`add_comment(kind="comment")` rows that landed in the ledger SHA chain
BEFORE the wire-up should be observed into TCMM so recall can see them.

Idempotency: each migrated row gets `extras_json.migrated_to_tcmm =
<tcmm_obs_id>` so a re-run skips it.  Original SHA-chain row stays —
removing it would break the chain.  This is a pure ADDITIVE migration:
add the TCMM observation, mark the ledger row as mirrored.

Run via:
    docker exec veilguard-agent-runtime-1 python -m app.proposals.m4_backfill

Or call `run_migration()` from a worker.  Pure async; safe to invoke
multiple times.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

logger = logging.getLogger("agent-runtime.m4_backfill")


async def run_migration(
    *,
    db_path: str = "/tcmm-data/veilguard/tcmm.db",
    batch_limit: int = 1000,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Walk `task_comments` for `kind=comment`, observe each to TCMM,
    mark the ledger row.  Returns a summary.

    Args:
        db_path:    Lance DB path.  Default matches production.
        batch_limit: max comments to process this call.  Default 1000.
        dry_run:    When True, count + log but don't write to TCMM
                    and don't mark the ledger row.  For preview.

    Returns:
        {
          "candidates":    int,   # rows matching kind=comment
          "skipped_done":  int,   # already had migrated_to_tcmm marker
          "skipped_empty": int,   # body too short / empty
          "migrated":      int,   # successfully observed + marked
          "failed":        int,   # TCMM write returned False
          "dry_run":       bool,
        }
    """
    import lancedb
    from ..memory.phase_7_writers import record_discussion_comment

    try:
        db = lancedb.connect(db_path)
        tbl = db.open_table("task_comments")
    except Exception as e:
        return {"error": f"db open failed: {e}",
                "candidates": 0, "migrated": 0}

    # Pull all kind=comment rows (small table — typically <10K rows).
    arr = (
        tbl.search()
        .where("kind = 'comment'")
        .limit(batch_limit)
        .to_arrow()
        .to_pylist()
    )
    candidates = len(arr)
    skipped_done = 0
    skipped_empty = 0
    migrated = 0
    failed = 0

    for row in arr:
        # Idempotency check.
        extras_raw = row.get("extras_json") or ""
        try:
            extras = json.loads(extras_raw) if extras_raw else {}
        except (json.JSONDecodeError, TypeError):
            extras = {}
        if extras.get("migrated_to_tcmm"):
            skipped_done += 1
            continue

        body = (row.get("body") or "").strip()
        if not body:
            skipped_empty += 1
            continue

        if dry_run:
            migrated += 1
            continue

        ok = await record_discussion_comment(
            task_id=row.get("task_id") or "",
            tenant_id=row.get("tenant_id") or "",
            user_id=row.get("user_id") or "",
            author_id=row.get("author_id") or "unknown-historical",
            body=body,
            note=False,
        )
        if not ok:
            failed += 1
            continue

        # Mark ledger row so re-run skips it.  Use a synthetic marker
        # rather than the real tcmm_obs_id because TCMM doesn't expose
        # row ids back to callers; the marker just needs to be truthy.
        new_extras = {**extras, "migrated_to_tcmm": int(time.time())}
        try:
            from ..ledger.store import LedgerStore, ns_filter
            ut = LedgerStore.get().table("task_comments")
            where = (
                f"{ns_filter(row.get('tenant_id') or '', row.get('user_id') or '')} "
                f"AND id = '{row.get('id')}'"
            )
            ut.update(where=where, values={"extras_json": json.dumps(new_extras)})
            migrated += 1
        except Exception as e:
            logger.warning(
                f"[m4_backfill] couldn't mark ledger row {row.get('id')}: {e}"
            )
            # Counted as migrated since TCMM write succeeded; the next
            # run will see the row again (missing marker) and re-observe.
            # That's idempotent at the TCMM side via content-hash dedup,
            # but produces a duplicate observation.  Log loudly.
            migrated += 1

    summary = {
        "candidates":    candidates,
        "skipped_done":  skipped_done,
        "skipped_empty": skipped_empty,
        "migrated":      migrated,
        "failed":        failed,
        "dry_run":       dry_run,
    }
    logger.info(f"[m4_backfill] cycle: {summary}")
    return summary


def main() -> None:
    """CLI entry-point for ad-hoc backfill runs."""
    import sys
    dry = "--dry-run" in sys.argv
    summary = asyncio.run(run_migration(dry_run=dry))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
