"""Create Lance indices on the archive table for fast recall.

Two classes of index:

1. **Scalar indices** (BTREE-like) on the columns that every recall
   query filters by: ``namespace`` (conversation scope),
   ``user_id`` (tenant isolation). Without these, every query does a
   full table scan looking for matching rows. At 2K rows that's ~5ms,
   at 10K it's ~25-50ms, at 100K it's 250ms+.

2. **Vector index** (IVF-PQ) on the ``vector`` column. Dense
   similarity search currently does a flat L2 scan over all rows that
   pass the where-clause filter. With IVF-PQ at ~sqrt(N) partitions,
   search time drops from O(N) to O(sqrt(N)).

Idempotent: running this multiple times skips already-created indices.
"""
import lancedb
import sys

DB_PATH = "/home/rudol/veilguard/tcmm-data/veilguard/tcmm.db"


def main():
    db = lancedb.connect(DB_PATH)
    t = db.open_table("archive")

    rows = t.count_rows()
    print(f"archive rows: {rows}")

    existing = {i.name for i in t.list_indices()}
    print(f"existing indices: {existing or '(none)'}")

    # Scalar indices — cheap to build, big win on where-clause filtering.
    # Each is a BITMAP or BTREE-like structure. LanceDB picks the right
    # kind based on cardinality.
    for col in ["namespace", "user_id", "aid"]:
        idx_name = f"{col}_idx"
        if idx_name in existing or any(col in i.columns for i in t.list_indices()):
            print(f"  skip scalar {col} (already indexed)")
            continue
        print(f"  creating scalar index on {col}...")
        # replace=True so repeat runs don't complain; it's a no-op if
        # the index already exists under the same spec.
        t.create_scalar_index(col, replace=True)
        print(f"    done")

    # Vector index — needs enough rows for meaningful partitions.
    # Lance recommends >1000 rows before IVF-PQ pays off (smaller
    # datasets are faster with flat scan). We're at ~2K so on the
    # edge; doing it now so we're ready as the archive grows.
    has_vector_index = any("vector" in i.columns for i in t.list_indices())
    if has_vector_index:
        print("  skip vector index (already present)")
    elif rows < 256:
        print(f"  skip vector index (only {rows} rows — Lance needs ≥256 for IVF-PQ)")
    else:
        # num_partitions = round(sqrt(N)) is the textbook rule; for
        # N=2K that's ~45, for N=10K it's ~100. We bump to 64 min so
        # small archives still get decent partitioning.
        import math
        num_partitions = max(64, int(math.sqrt(rows)))
        # num_sub_vectors divides the 768-dim vector into smaller
        # chunks for Product Quantization. Must evenly divide the
        # dimension. 96 sub-vectors × 8 floats each = 768. Good
        # recall/speed tradeoff for text embeddings.
        num_sub_vectors = 96
        print(
            f"  creating IVF-PQ vector index "
            f"(num_partitions={num_partitions}, num_sub_vectors={num_sub_vectors})..."
        )
        t.create_index(
            vector_column_name="vector",
            metric="cosine",
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            replace=True,
        )
        print("    done")

    # Embeddings table (if present) — dense retrieval hits this too.
    for table_name in ("embeddings", "sparse_archive"):
        if table_name not in db.table_names():
            continue
        tt = db.open_table(table_name)
        print(f"\n{table_name} rows: {tt.count_rows()}")
        for col in ["namespace", "user_id"]:
            try:
                tt.create_scalar_index(col, replace=True)
                print(f"  scalar {col} index created")
            except Exception as e:
                print(f"  skip scalar {col}: {e}")

    print("\n=== final state ===")
    for name in db.table_names():
        try:
            tbl = db.open_table(name)
            idx = tbl.list_indices()
            if idx:
                print(f"{name}: {[(i.name, i.columns) for i in idx]}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
