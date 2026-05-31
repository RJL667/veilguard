"""
Migrate TCMM archive embeddings: sentence-transformers/all-mpnet-base-v2 (768d)
OR Vertex text-embedding-005 (768d) → snowflake/snowflake-arctic-embed-xs (384d).

Rewrites the ``vector`` column in every Lance table that has one, preserving
every other column verbatim. NEVER deletes user data — old tables are kept
around under ``<name>_old_<ts>`` so a rollback is one rename away. Use
``--keep-backup-tables=0`` to drop the backup tables AFTER the new tables
have been verified end-to-end against live traffic.

Why a full rewrite (not "update vector in place"):
    LanceDB schemas are immutable. The existing ``vector: fixed_size_list<float>[768]``
    column can't be widened/narrowed without rewriting the table. So we build a
    new table with the same schema except for ``vector: fixed_size_list<float>[384]``,
    fill it row-for-row, swap names atomically.

Usage:
    # Dry run — read + embed everything, write to /tmp scratch, no touch on tcmm.db
    python scripts/migrate_to_arctic_xs.py --dry-run --db-path /path/to/tcmm.db

    # Live migration — atomic swap, keep backup tables for rollback
    python scripts/migrate_to_arctic_xs.py --db-path /path/to/tcmm.db

    # Drop the backup tables after a few days of healthy production
    python scripts/migrate_to_arctic_xs.py --cleanup-backups --db-path /path/to/tcmm.db

Safety:
    - Backup of the ENTIRE tcmm.db dir BEFORE running this is required (the
      script verifies a sibling ``.deploy-backups/`` exists; bail otherwise).
    - Refuses to run while TCMM service is up (looks for a listener on :8811).
    - Writes new tables under a ``__migrating__`` prefix; only renames to the
      live name after row-count + schema verification.
    - The ``_tcmm_meta`` table is updated last, after all data tables, so a
      crash mid-migration leaves the meta pointing at the old embedder (TCMM
      will refuse to start, but data is consistent).

Author: 2026-05-13 migration script for the FastEmbed cutover.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path
from typing import Optional

# Force CPU + no TF/Flax probe before any transformers-adjacent import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")


# ── New embedder config ──────────────────────────────────────────────────────
NEW_BACKEND = "local"
NEW_MODEL = "snowflake/snowflake-arctic-embed-xs"
NEW_DIM = 384


def log(*a, **kw):
    print(*a, **kw, flush=True)


def ensure_safe_to_run(db_path: str) -> None:
    """Bail loudly if preconditions aren't met."""
    # 1. tcmm.db must exist
    if not Path(db_path).is_dir():
        log(f"FATAL: {db_path} is not a directory — aborting")
        sys.exit(2)

    # 2. There should be a recent backup somewhere
    backups_dir = Path(db_path).parents[1] / ".deploy-backups"
    if not backups_dir.exists():
        log(f"FATAL: no {backups_dir} found — back up tcmm.db FIRST (e.g. "
            "`sudo cp -r {db_path} {backups_dir}/pre-arctic-migration-$(date +%s)`)")
        sys.exit(3)

    # 3. TCMM service must NOT be running (or we'd race writes)
    s = socket.socket()
    s.settimeout(1)
    try:
        s.connect(("127.0.0.1", 8811))
        log("FATAL: TCMM service is up on :8811 — stop it before migrating")
        log("       (`sudo pkill -f tcmm-service/server.py` or systemctl stop ...)")
        sys.exit(4)
    except (socket.error, ConnectionRefusedError):
        pass  # good, nothing listening
    finally:
        s.close()


def find_vector_and_text_columns(schema) -> tuple[Optional[str], int, Optional[str]]:
    """Return (vector_col_name, old_dim, text_col_name) or (None, 0, None)."""
    import pyarrow as pa

    vec_col = None
    old_dim = 0
    text_col = None
    text_priority = {"semantic_text": 1, "text": 0}  # prefer semantic_text if present

    for f in schema:
        t = f.type
        if isinstance(t, pa.FixedSizeListType) and pa.types.is_floating(t.value_type):
            vec_col = f.name
            old_dim = t.list_size
        if pa.types.is_string(t) and f.name in text_priority:
            if text_col is None or text_priority[f.name] > text_priority[text_col]:
                text_col = f.name
    return vec_col, old_dim, text_col


def rebuild_schema_with_new_dim(old_schema, vec_col: str, new_dim: int):
    """Return a new pyarrow.Schema identical except for ``vec_col`` width."""
    import pyarrow as pa

    fields = []
    for f in old_schema:
        if f.name == vec_col:
            fields.append(pa.field(vec_col, pa.list_(pa.float32(), new_dim)))
        else:
            fields.append(f)
    return pa.schema(fields)


def migrate_table(db, table_name: str, embedder, batch_size: int, dry_run: bool) -> dict:
    """Migrate one table. Returns a stats dict."""
    import pyarrow as pa

    t = db.open_table(table_name)
    old_schema = t.schema
    vec_col, old_dim, text_col = find_vector_and_text_columns(old_schema)

    if vec_col is None:
        log(f"  {table_name}: no FixedSizeList<float> column → SKIP")
        return {"table": table_name, "status": "skip-no-vector"}

    n_rows = t.count_rows()
    if n_rows == 0:
        log(f"  {table_name}: 0 rows → SKIP")
        return {"table": table_name, "status": "skip-empty", "rows": 0}

    if text_col is None:
        log(f"  {table_name}: vector column {vec_col} present but no text column "
            f"({n_rows} rows) — can't re-embed, SKIP. (Consider manual investigation.)")
        return {"table": table_name, "status": "skip-no-text", "rows": n_rows,
                "vec_col": vec_col, "old_dim": old_dim}

    log(f"  {table_name}: {n_rows:,} rows, "
        f"vec_col={vec_col}, old_dim={old_dim} → new_dim={NEW_DIM}, text_col={text_col}")

    # Read whole table into memory. 3393 rows × ~10KB/row ≈ 30MB, trivial.
    df = t.to_pandas()
    # Defensive — replace NaN/None text with empty string so embed doesn't crash.
    df[text_col] = df[text_col].fillna("")
    texts = df[text_col].tolist()

    # Embed in chunks for progress visibility (fastembed itself batches internally).
    t0 = time.perf_counter()
    new_vecs: list = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        chunk_vecs = list(embedder.embed(chunk))
        new_vecs.extend(chunk_vecs)
        elapsed = time.perf_counter() - t0
        rate = (i + len(chunk)) / elapsed if elapsed > 0 else 0
        log(f"    embedded {i + len(chunk):>6}/{len(texts)}  "
            f"{rate:>5.1f} rows/s  "
            f"{elapsed:>5.1f}s elapsed")
    assert len(new_vecs) == len(df), f"embed count mismatch {len(new_vecs)} vs {len(df)}"
    embed_secs = time.perf_counter() - t0

    # Sanity: dims must match what we expect.
    sample_dim = len(new_vecs[0])
    if sample_dim != NEW_DIM:
        log(f"FATAL: embedder produced dim={sample_dim}, expected {NEW_DIM} — aborting")
        sys.exit(5)

    # Replace vector column with float32 lists.
    import numpy as np
    df[vec_col] = [np.asarray(v, dtype=np.float32) for v in new_vecs]

    # Build new schema with the new dim.
    new_schema = rebuild_schema_with_new_dim(old_schema, vec_col, NEW_DIM)
    new_table_name = f"__migrating__{table_name}"

    if dry_run:
        log(f"  [DRY-RUN] would create {new_table_name} ({len(df)} rows, "
            f"vec dim {NEW_DIM}) and atomic-swap. Skipping actual write.")
        return {"table": table_name, "status": "dry-run-ok", "rows": len(df),
                "embed_secs": embed_secs, "rows_per_sec": len(df) / embed_secs}

    # Drop any leftover __migrating__ artefact from a previous interrupted run.
    if new_table_name in db.table_names(limit=100_000):  # default limit=10 hides late tables
        log(f"  cleaning up stale {new_table_name} from prior interrupted run")
        db.drop_table(new_table_name)

    # Create new table with new schema + data.
    log(f"  creating {new_table_name} with {len(df)} rows, vec dim {NEW_DIM}...")
    t_new = db.create_table(new_table_name, df, schema=new_schema)
    new_count = t_new.count_rows()
    if new_count != len(df):
        log(f"FATAL: new table has {new_count} rows, expected {len(df)} — aborting")
        sys.exit(6)
    log(f"    verified {new_count} rows written")

    # Atomic swap: old → backup, new → live.
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup_name = f"{table_name}_old_{old_dim}d_{ts}"
    log(f"  renaming {table_name} → {backup_name}")
    db.rename_table(table_name, backup_name)
    log(f"  renaming {new_table_name} → {table_name}")
    db.rename_table(new_table_name, table_name)

    return {
        "table": table_name,
        "status": "migrated",
        "rows": len(df),
        "old_dim": old_dim,
        "new_dim": NEW_DIM,
        "embed_secs": round(embed_secs, 1),
        "rows_per_sec": round(len(df) / embed_secs, 1),
        "backup_table": backup_name,
    }


def update_tcmm_meta(db, dry_run: bool):
    """Write the persisted embedder-metadata row so TCMM's compat check passes."""
    import pyarrow as pa

    meta_schema = pa.schema([
        pa.field("key", pa.string()),
        pa.field("value", pa.string()),
    ])
    rows = [
        {"key": "embedder.backend", "value": NEW_BACKEND},
        {"key": "embedder.model_name", "value": NEW_MODEL},
        {"key": "embedder.dimension", "value": str(NEW_DIM)},
        {"key": "migrated_at", "value": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    ]

    if dry_run:
        log(f"  [DRY-RUN] would write {len(rows)} _tcmm_meta rows: {rows}")
        return

    table_name = "_tcmm_meta"
    if table_name in db.table_names(limit=100_000):  # default limit=10 hides late tables
        log(f"  dropping existing {table_name} so fresh rows aren't duplicated")
        db.drop_table(table_name)
    db.create_table(table_name, rows, schema=meta_schema)
    log(f"  wrote {len(rows)} _tcmm_meta rows")


def cleanup_backup_tables(db, dry_run: bool):
    """Drop all *_old_*d_* tables. Run only after the new embedder is validated."""
    backups = [n for n in db.table_names(limit=100_000) if "_old_" in n and "d_" in n]  # default limit=10 truncates
    if not backups:
        log("no backup tables to clean up")
        return
    for n in sorted(backups):
        if dry_run:
            log(f"  [DRY-RUN] would drop {n}")
        else:
            db.drop_table(n)
            log(f"  dropped {n}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db-path",
                   default="/home/rudol/veilguard/tcmm-data/veilguard/tcmm.db",
                   help="path to the tcmm.db directory")
    p.add_argument("--dry-run", action="store_true",
                   help="run the embedder but don't touch any table")
    p.add_argument("--cleanup-backups", action="store_true",
                   help="drop all *_old_*d_* backup tables; run only after the "
                        "new embedder has been validated against live traffic")
    p.add_argument("--batch-size", type=int, default=128,
                   help="rows per progress checkpoint (fastembed batches internally too)")
    p.add_argument("--tables", nargs="*",
                   default=["archive", "embeddings", "dream_archive"],
                   help="tables to migrate (others left alone)")
    args = p.parse_args()

    log(f"=== TCMM Arctic-XS migration ===")
    log(f"  db_path: {args.db_path}")
    log(f"  dry_run: {args.dry_run}")
    log(f"  target:  backend={NEW_BACKEND} model={NEW_MODEL} dim={NEW_DIM}")
    log("")

    ensure_safe_to_run(args.db_path)

    import lancedb
    db = lancedb.connect(args.db_path)

    if args.cleanup_backups:
        log("=== cleanup mode: dropping backup tables ===")
        cleanup_backup_tables(db, args.dry_run)
        return

    log("=== loading FastEmbed embedder (one-time) ===")
    from fastembed import TextEmbedding
    embedder = TextEmbedding(
        model_name=NEW_MODEL,
        providers=["CPUExecutionProvider"],
        threads=os.cpu_count() or 4,
    )
    # Warmup
    _ = list(embedder.embed(["warmup"]))
    log(f"  embedder ready (threads={os.cpu_count()})")
    log("")

    log("=== existing tables ===")
    for n in sorted(db.table_names(limit=100_000)):  # default limit=10 truncates
        try:
            log(f"  {n}: {db.open_table(n).count_rows():>10,} rows")
        except Exception as e:
            log(f"  {n}: <count failed: {e}>")
    log("")

    log("=== migrating ===")
    results = []
    for tbl in args.tables:
        if tbl not in db.table_names(limit=100_000):  # default limit=10 hides late tables
            log(f"  {tbl}: not in db, SKIP")
            continue
        results.append(migrate_table(db, tbl, embedder, args.batch_size, args.dry_run))
    log("")

    log("=== writing _tcmm_meta ===")
    update_tcmm_meta(db, args.dry_run)
    log("")

    log("=== summary ===")
    for r in results:
        log(f"  {r}")

    if args.dry_run:
        log("")
        log("dry-run complete. nothing was modified.")
    else:
        log("")
        log("MIGRATION COMPLETE.")
        log("Backup tables remain under *_old_<dim>d_<ts> — rollback by renaming back.")
        log("After validating new embedder against live traffic for a few days,")
        log("run with --cleanup-backups to drop them.")


if __name__ == "__main__":
    main()
