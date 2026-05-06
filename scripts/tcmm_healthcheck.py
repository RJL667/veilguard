"""
TCMM / LanceDB content health check.

Opens each Lance table under the TCMM database and reports:
  - Row counts + schema column counts per table
  - For ``archive.lance`` specifically: % of rows that have populated
    embeddings, topics, entities, semantic_links, entity_links,
    topic_links, contextual_links, claims, lineage, temporal links,
    plus per-row averages.
  - Embedding-vector quality: dimension, % non-zero, mean norm.
  - Recent-write activity: how recent is the latest row.

Run as a one-shot on the VM. Doesn't write anything — pure read,
safe to run while TCMM is live.

    sudo -u rudol /usr/bin/python3 /home/rudol/veilguard/scripts/tcmm_healthcheck.py

Tunables via env:
    LANCE_DB_DIR       — path to tcmm.db (default below)
    HEALTHCHECK_SAMPLE — rows to sample per check (default 500)
"""

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lance

DB_DIR = Path(os.environ.get(
    "LANCE_DB_DIR",
    "/home/rudol/veilguard/tcmm-data/veilguard/tcmm.db",
))
# Default to "no sampling" — read everything. With 1.4K rows the whole
# archive fits in <100 MB of RAM (the vector column is 768 floats × 4
# bytes = 3 KB/row × 1394 ≈ 4 MB). Set HEALTHCHECK_SAMPLE=N to cap if
# this ever grows past comfortable RAM. 0 means "all rows".
SAMPLE = int(os.environ.get("HEALTHCHECK_SAMPLE", "0"))


# Tables we expect, in the order we want to print them. Matches the
# layout we found on prod: archive is the keystone, sparse_* are the
# per-feature inverted indices, embeddings.lance is the secondary
# embedding store.
TABLES = [
    "archive.lance",
    "embeddings.lance",
    "sparse_archive.lance",
    "sparse_topic.lance",
    "sparse_entity.lance",
    "sparse_claims.lance",
    "_sequences.lance",
    "pii_audit.lance",
    "dream_archive.lance",
    "sparse_dream.lance",
    "sparse_dream_topic.lance",
    "sparse_dream_entity.lance",
    "sparse.lance",
]


def hr(title: str = "") -> None:
    bar = "─" * 76
    if title:
        print(f"\n{bar}\n  {title}\n{bar}")
    else:
        print(bar)


def fmt_pct(num: int, denom: int) -> str:
    if denom == 0:
        return "n/a"
    return f"{num / denom * 100:5.1f}% ({num}/{denom})"


def open_dataset(path: Path):
    if not path.is_dir():
        return None
    try:
        return lance.dataset(str(path))
    except Exception as e:
        print(f"  ! could not open {path.name}: {e}")
        return None


def basic_stats(name: str, ds) -> dict:
    """Row count + schema summary + size on disk for one table."""
    if ds is None:
        return {"rows": 0, "fields": 0}
    n_rows = ds.count_rows()
    n_fields = len(ds.schema)
    n_versions = len(ds.versions())
    n_fragments = len(ds.get_fragments())
    print(
        f"  rows={n_rows:>8,d}  fields={n_fields:>3d}  "
        f"fragments={n_fragments:>4d}  versions={n_versions:>4d}"
    )
    return {
        "rows": n_rows,
        "fields": n_fields,
        "fragments": n_fragments,
        "versions": n_versions,
    }


def archive_content_health(ds) -> None:
    """Sample archive.lance rows + report per-field population.

    The archive table is where TCMM stores everything: blocks, their
    embeddings, the topic/entity/claim extractions, and the
    semantic/contextual/topic/entity link maps. If any of these
    fields are 0% populated, the corresponding pipeline stage is
    broken (or hasn't been enabled).
    """
    if ds is None:
        print("  ! archive.lance not present")
        return

    total = ds.count_rows()
    if total == 0:
        print("  ! archive table is empty — no blocks have been ingested")
        return

    # Read the full table by default (SAMPLE=0). The percentages we
    # compute are sample-size-sensitive — sampling 500 of 1,394 rows
    # gives a coverage estimate, not the exact figure. With small
    # tables we just read all of it; SAMPLE>0 caps for huge tables.
    if SAMPLE > 0 and total > SAMPLE:
        sample_n = SAMPLE
        print(f"\n  Sampling {sample_n:,} rows of {total:,} (SAMPLE cap)...")
        sample = ds.to_table(limit=sample_n).to_pylist()
    else:
        sample_n = total
        print(f"\n  Reading all {total:,} rows for exact content stats...")
        sample = ds.to_table().to_pylist()

    # Per-field populated counters
    counts = {
        "vector_present":     0,
        "vector_nonzero":     0,
        "topics":             0,
        "entities":           0,
        "claims":             0,
        "topic_dicts":        0,
        "entity_dicts":       0,
        "semantic_links":     0,
        "entity_links":       0,
        "topic_links":        0,
        "contextual_links":   0,
        "lineage_parents":    0,
        "temporal_prev":      0,
        "temporal_next":      0,
        "source_block_ids":   0,
        "behavioural_json":   0,
        "semantic_json":      0,
    }

    # Sums for per-row averages (only counted when populated)
    sums = {k: 0 for k in counts}

    # Vector norm stats
    vec_norms: list = []
    vec_dim = None

    # Recency
    latest_created_ts = 0

    for row in sample:
        v = row.get("vector")
        if v is not None and len(v) > 0:
            counts["vector_present"] += 1
            vec_dim = len(v)
            # Cheap non-zero check: norm > tiny threshold
            norm = math.sqrt(sum(float(x) * float(x) for x in v[:64]))
            if norm > 1e-3:
                counts["vector_nonzero"] += 1
                vec_norms.append(norm)

        for list_field in ("topics", "entities", "claims",
                           "source_block_ids",
                           "topic_dicts", "entity_dicts",
                           "semantic_links", "entity_links",
                           "topic_links", "contextual_links"):
            val = row.get(list_field) or []
            if val:
                counts[list_field] += 1
                sums[list_field] += len(val)

        lineage = row.get("lineage") or {}
        if lineage and (lineage.get("parents") or []):
            counts["lineage_parents"] += 1
            sums["lineage_parents"] += len(lineage["parents"])

        temporal = row.get("temporal") or {}
        if temporal:
            if temporal.get("prev_aid"):
                counts["temporal_prev"] += 1
            if temporal.get("next_aid"):
                counts["temporal_next"] += 1

        for s in ("behavioural_json", "semantic_json"):
            v = row.get(s) or ""
            if v and v != "{}":
                counts[s] += 1

        ts = row.get("created_ts") or 0
        if ts and ts > latest_created_ts:
            latest_created_ts = ts

    n = sample_n

    print()
    print("  Embedding pipeline:")
    print(f"    vector present           {fmt_pct(counts['vector_present'], n)}"
          f"  dim={vec_dim}")
    print(f"    vector non-zero          {fmt_pct(counts['vector_nonzero'], n)}"
          f"  (zero-vectors mean embedding worker hasn't filled them in)")
    if vec_norms:
        mean_norm = sum(vec_norms) / len(vec_norms)
        print(f"    vector mean L2 norm      {mean_norm:.3f} "
              f"(typical normalized embedding ≈ 1.0; 64-element prefix used)")

    print()
    print("  Extracted features:")
    print(f"    topics                   {fmt_pct(counts['topics'], n)}"
          f"  avg/row={sums['topics'] / max(counts['topics'], 1):.1f}")
    print(f"    entities                 {fmt_pct(counts['entities'], n)}"
          f"  avg/row={sums['entities'] / max(counts['entities'], 1):.1f}")
    print(f"    claims                   {fmt_pct(counts['claims'], n)}"
          f"  avg/row={sums['claims'] / max(counts['claims'], 1):.1f}")
    print(f"    topic_dicts (typed)      {fmt_pct(counts['topic_dicts'], n)}"
          f"  avg/row={sums['topic_dicts'] / max(counts['topic_dicts'], 1):.1f}")
    print(f"    entity_dicts (typed)     {fmt_pct(counts['entity_dicts'], n)}"
          f"  avg/row={sums['entity_dicts'] / max(counts['entity_dicts'], 1):.1f}")

    print()
    print("  Graph / link structure:")
    print(f"    semantic_links           {fmt_pct(counts['semantic_links'], n)}"
          f"  avg/row={sums['semantic_links'] / max(counts['semantic_links'], 1):.1f}")
    print(f"    entity_links             {fmt_pct(counts['entity_links'], n)}"
          f"  avg/row={sums['entity_links'] / max(counts['entity_links'], 1):.1f}")
    print(f"    topic_links              {fmt_pct(counts['topic_links'], n)}"
          f"  avg/row={sums['topic_links'] / max(counts['topic_links'], 1):.1f}")
    print(f"    contextual_links         {fmt_pct(counts['contextual_links'], n)}"
          f"  avg/row={sums['contextual_links'] / max(counts['contextual_links'], 1):.1f}")
    print(f"    lineage.parents          {fmt_pct(counts['lineage_parents'], n)}"
          f"  avg/row={sums['lineage_parents'] / max(counts['lineage_parents'], 1):.1f}")
    print(f"    temporal.prev_aid        {fmt_pct(counts['temporal_prev'], n)}")
    print(f"    temporal.next_aid        {fmt_pct(counts['temporal_next'], n)}")
    print(f"    source_block_ids         {fmt_pct(counts['source_block_ids'], n)}"
          f"  avg/row={sums['source_block_ids'] / max(counts['source_block_ids'], 1):.1f}")

    print()
    print("  Residual JSON fields (rarely populated by design):")
    print(f"    behavioural_json         {fmt_pct(counts['behavioural_json'], n)}")
    print(f"    semantic_json            {fmt_pct(counts['semantic_json'], n)}")

    if latest_created_ts:
        # Field is in step-units historically but recent versions store
        # epoch seconds — try to interpret.
        try:
            if latest_created_ts > 10**9:  # plausible epoch
                latest_iso = datetime.fromtimestamp(
                    latest_created_ts, tz=timezone.utc
                ).isoformat()
                age_h = (time.time() - latest_created_ts) / 3600
                print()
                print(f"  Latest sampled created_ts: {latest_iso}  "
                      f"({age_h:.1f}h ago)")
        except (ValueError, OSError):
            pass


def embeddings_table_health(ds) -> None:
    """Sister embedding table — separate from the archive's vector column.

    Used as overflow / per-aid embedding store. We just check it has rows
    and the vector column is the same dim as the archive's.
    """
    if ds is None:
        print("  ! embeddings.lance not present")
        return
    total = ds.count_rows()
    if total == 0:
        print("  ! embeddings table is empty")
        return
    sample = ds.to_table(limit=min(50, total)).to_pylist()
    n = len(sample)
    nonzero = 0
    dim = None
    for r in sample:
        v = r.get("vector") or r.get("embedding") or []
        if v:
            dim = len(v)
            if any(abs(float(x)) > 1e-3 for x in v[:64]):
                nonzero += 1
    print(f"  total embeddings rows = {total:,}")
    print(f"  sample non-zero       = {fmt_pct(nonzero, n)}  dim={dim}")


def sparse_table_health(name: str, ds) -> None:
    """Inverted-index tables (sparse_topic, sparse_entity, etc.).

    Each row is typically (term, list of aids that mention it). A high
    row count with a sensible avg fan-out means the index is working.
    """
    if ds is None:
        return
    total = ds.count_rows()
    if total == 0:
        print(f"  {name:<28} EMPTY")
        return
    # Just count rows + show fields. Sampling a few to inspect shape.
    sample = ds.to_table(limit=5).to_pylist()
    fields = list(sample[0].keys()) if sample else []
    print(f"  {name:<28} rows={total:>8,d}  fields={','.join(fields[:5])}"
          f"{'...' if len(fields) > 5 else ''}")


def main() -> int:
    print(f"=== TCMM / LanceDB health check ===")
    print(f"=== db: {DB_DIR}")
    print(f"=== sample size: {SAMPLE} rows per content check")

    if not DB_DIR.is_dir():
        print(f"ERROR: {DB_DIR} not found", file=sys.stderr)
        return 1

    # ── Per-table headline stats ─────────────────────────────────────
    hr("Per-table row + schema summary")
    summaries = {}
    for table in TABLES:
        path = DB_DIR / table
        if not path.is_dir():
            continue
        print(f"\n[{table}]")
        ds = open_dataset(path)
        summaries[table] = basic_stats(table, ds)

    # ── Archive deep-dive ────────────────────────────────────────────
    hr("archive.lance content health (the keystone table)")
    archive_ds = open_dataset(DB_DIR / "archive.lance")
    archive_content_health(archive_ds)

    # ── Embeddings sister table ──────────────────────────────────────
    hr("embeddings.lance health (overflow embedding store)")
    embeddings_ds = open_dataset(DB_DIR / "embeddings.lance")
    embeddings_table_health(embeddings_ds)

    # ── Sparse indices ───────────────────────────────────────────────
    hr("Sparse inverted indices (topic / entity / claim / archive)")
    for sparse in ("sparse_archive.lance", "sparse_topic.lance",
                   "sparse_entity.lance", "sparse_claims.lance",
                   "sparse_dream.lance", "sparse_dream_topic.lance",
                   "sparse_dream_entity.lance", "sparse.lance",
                   "dream_archive.lance"):
        path = DB_DIR / sparse
        if not path.is_dir():
            continue
        ds = open_dataset(path)
        sparse_table_health(sparse, ds)

    # ── Roll-up ──────────────────────────────────────────────────────
    hr("Roll-up")
    total_rows = sum(s.get("rows", 0) for s in summaries.values())
    total_fragments = sum(s.get("fragments", 0) for s in summaries.values())
    total_versions = sum(s.get("versions", 0) for s in summaries.values())
    print(f"\n  total rows across all tables  : {total_rows:>10,d}")
    print(f"  total fragments               : {total_fragments:>10,d}")
    print(f"  total versions                : {total_versions:>10,d}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
