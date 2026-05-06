"""
LanceDB maintenance for TCMM.

Compacts fragmented Lance tables and prunes old version manifests +
their unique on-disk data. Run with TCMM stopped — concurrent writes
during cleanup_old_versions can race a reader holding a stale version
ref. The systemd unit veilguard-tcmm-maintenance.service handles the
stop/start dance automatically; for manual runs:

    sudo systemctl stop veilguard-tcmm
    python3 /home/rudol/veilguard/scripts/lance_maintenance.py
    sudo systemctl start veilguard-tcmm

Each table's reduction is logged to stdout. Idempotent — re-running
when there's nothing to compact is a fast no-op.

Tunables via env vars:
    LANCE_DB_DIR       — path to tcmm.db (default below)
    LANCE_KEEP_HOURS    — version retention in hours (default 1)
    LANCE_KEEP_VERSIONS — minimum versions to retain (default 5)
"""

import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import lance

DB_DIR = Path(os.environ.get(
    "LANCE_DB_DIR",
    "/home/rudol/veilguard/tcmm-data/veilguard/tcmm.db",
))
# Retention policy. Learned the hard way that ``older_than`` measured in
# DAYS protects too much — TCMM commits manifests fast enough that a
# 7-day window kept all 9000+ manifests intact (cleanup was a no-op).
# 1 hour gives enough headroom for "oh shit, revert that last write"
# time-travel without accumulating per-write garbage. retain_versions
# is a floor so quiet periods still keep at least N recent versions.
KEEP_HOURS = int(os.environ.get("LANCE_KEEP_HOURS", "1"))
KEEP_VERSIONS = int(os.environ.get("LANCE_KEEP_VERSIONS", "5"))
KEEP_NEWER_THAN = timedelta(hours=KEEP_HOURS)

# Order matters — biggest first so a crash mid-script doesn't leave the
# largest table in an inconsistent state. archive.lance is the prod
# behemoth (2.5 GB at last check). The smaller sparse_* tables are
# included because they accumulate index versions too.
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


def size_mb(path: Path) -> float:
    """Total size of a directory tree in MB."""
    if not path.is_dir():
        return 0.0
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total / (1024 * 1024)


def maintain(table_path: Path) -> tuple[float, float]:
    """Run compaction + cleanup on one Lance table. Returns (before_mb, after_mb)."""
    if not table_path.is_dir():
        print(f"  SKIP {table_path.name} — not present", flush=True)
        return 0.0, 0.0

    before_mb = size_mb(table_path)
    try:
        ds = lance.dataset(str(table_path))
    except Exception as e:
        print(f"  SKIP {table_path.name} — open failed: {e}", flush=True)
        return before_mb, before_mb

    frag_before = len(ds.get_fragments())
    ver_before = len(ds.versions())
    print(
        f"\n[{table_path.name}] before: {before_mb:>7.1f} MB | "
        f"{frag_before:>5} fragments | {ver_before:>5} versions",
        flush=True,
    )

    # 1. Compact small fragments. This is MVCC-safe but we still run
    # with TCMM stopped — see module docstring.
    t0 = time.time()
    print("  → compact_files()...", flush=True)
    try:
        ds.optimize.compact_files()
    except Exception as e:
        print(f"     compact failed: {e}", flush=True)

    # 2. Drop old version manifests + their unique data. This is what
    # actually frees disk for tables that have been heavily updated.
    # We pass BOTH ``older_than`` and ``retain_versions`` — Lance keeps
    # the union of "newer than X" and "in the most recent N", so we
    # never wipe past the floor even if the cluster was idle.
    print(
        f"  → cleanup_old_versions(older_than={KEEP_HOURS}h, "
        f"retain_versions={KEEP_VERSIONS})...",
        flush=True,
    )
    try:
        result = ds.cleanup_old_versions(
            older_than=KEEP_NEWER_THAN,
            retain_versions=KEEP_VERSIONS,
            delete_unverified=True,
            error_if_tagged_old_versions=False,
        )
        # Lance >=4 returns a CleanupStats. Older versions return None.
        if result is not None:
            br = getattr(result, "bytes_removed", None)
            ov = getattr(result, "old_versions", None)
            if br is not None and ov is not None:
                print(
                    f"     removed {ov} old versions, "
                    f"{br / (1024 * 1024):.1f} MB of unique data",
                    flush=True,
                )
    except TypeError:
        # Older lance versions don't accept all kwargs. Fall back.
        try:
            ds.cleanup_old_versions(older_than=KEEP_NEWER_THAN)
        except Exception as e:
            print(f"     cleanup failed: {e}", flush=True)
    except Exception as e:
        print(f"     cleanup failed: {e}", flush=True)

    # Re-open to see post-cleanup state
    ds = lance.dataset(str(table_path))
    after_mb = size_mb(table_path)
    frag_after = len(ds.get_fragments())
    ver_after = len(ds.versions())
    saved_pct = (before_mb - after_mb) / before_mb * 100 if before_mb > 0 else 0
    elapsed = time.time() - t0

    print(
        f"  after:  {after_mb:>7.1f} MB | "
        f"{frag_after:>5} fragments | {ver_after:>5} versions | "
        f"freed {before_mb - after_mb:.1f} MB ({saved_pct:.0f}%) in {elapsed:.1f}s",
        flush=True,
    )
    return before_mb, after_mb


def main() -> int:
    print(f"=== LanceDB maintenance: {DB_DIR} ===", flush=True)
    print(
        f"=== keep versions newer than {KEEP_HOURS}h "
        f"(floor: retain last {KEEP_VERSIONS}) ===",
        flush=True,
    )

    if not DB_DIR.is_dir():
        print(f"ERROR: {DB_DIR} not found", file=sys.stderr, flush=True)
        return 1

    t_start = time.time()
    total_before = 0.0
    total_after = 0.0

    for table in TABLES:
        b, a = maintain(DB_DIR / table)
        total_before += b
        total_after += a

    elapsed = time.time() - t_start
    saved = total_before - total_after
    pct = saved / total_before * 100 if total_before > 0 else 0

    print(
        f"\n=== TOTAL: {total_before:.1f} MB → {total_after:.1f} MB "
        f"(freed {saved:.1f} MB / {pct:.0f}%) in {elapsed:.1f}s ===",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
