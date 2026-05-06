"""Quick per-user breakdown of archive.lance contents.

Reports rows / namespace / latest activity grouped by user_id, plus
the same for sparse_archive (so we can verify the inverted index has
content for every user).
"""
import os
import time
from collections import Counter
from pathlib import Path

import lance

DB_DIR = Path(os.environ.get(
    "LANCE_DB_DIR",
    "/home/rudol/veilguard/tcmm-data/veilguard/tcmm.db",
))


def per_user(table_name: str, columns: list) -> None:
    path = DB_DIR / table_name
    if not path.is_dir():
        print(f"[{table_name}] not present")
        return

    ds = lance.dataset(str(path))
    total = ds.count_rows()
    print(f"\n=== {table_name} : {total:,} rows ===")

    if total == 0:
        return

    arr = ds.to_table(columns=columns).to_pylist()

    by_user = Counter()
    by_ns_per_user: dict = {}
    latest_per_user: dict = {}
    for r in arr:
        uid = (r.get("user_id") or "").strip() or "(empty)"
        ns = r.get("namespace") or "(empty)"
        ts = r.get("created_ts") or 0
        by_user[uid] += 1
        by_ns_per_user.setdefault(uid, Counter())[ns] += 1
        if ts > latest_per_user.get(uid, 0):
            latest_per_user[uid] = ts

    now = time.time()
    print(f"  {'user_id':<30s}  {'rows':>6s}  {'latest activity':<30s}  namespaces")
    print(f"  {'-' * 30}  {'-' * 6}  {'-' * 30}  {'-' * 30}")
    for uid, n in sorted(by_user.items(), key=lambda kv: -kv[1]):
        last = latest_per_user.get(uid, 0)
        if last and last > 10**9:
            age_h = (now - last) / 3600
            last_str = (time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(last))
                        + f"  ({age_h:.1f}h ago)")
        else:
            last_str = f"(raw={last})"
        ns_summary = ", ".join(
            f"{ns}={c}" for ns, c in by_ns_per_user[uid].most_common(5)
        )
        print(f"  {uid:<30s}  {n:>6d}  {last_str:<30s}  {ns_summary}")


def main() -> int:
    per_user("archive.lance", ["user_id", "namespace", "created_ts"])
    # sparse_archive has user_id + namespace too but no timestamp
    per_user("sparse_archive.lance", ["user_id", "namespace"])
    per_user("embeddings.lance", ["user_id", "namespace"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
