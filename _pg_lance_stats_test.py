"""TCMM-health dashboard panel on Postgres: lance_stats.overview /
nlp_coverage / fts_index_status read the CTI tables (base_node/archive) instead
of LanceDB."""
import sys, os
TC = r"C:\Users\rudol\.gemini\antigravity\tcmm\TCMM"
VG = r"C:\Users\rudol\Documents\veilguard"
sys.path.insert(0, os.path.join(VG, "services", "admin-dashboard"))
sys.path.insert(0, TC)
DSN = "postgresql://tcmm:tcmm@localhost:5433/tcmm"
os.environ["TCMM_STORAGE"] = "postgres"
os.environ["TCMM_DATABASE_URL"] = DSN
import psycopg2, numpy as np

fails = []
def chk(n, c, x=""):
    print(("  OK   " if c else "  FAIL ")+n+("" if c else f"   <<< {x}"))
    if not c: fails.append(n)

# clean-slate the CTI at dim=8 (provider recreates via migration → all indexes)
c = psycopg2.connect(DSN); c.autocommit = True
with c.cursor() as cur:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("""DROP VIEW IF EXISTS v_archive CASCADE; DROP VIEW IF EXISTS v_dream CASCADE;
    DROP TABLE IF EXISTS archive CASCADE; DROP TABLE IF EXISTS dream CASCADE;
    DROP TABLE IF EXISTS dream_edges CASCADE; DROP TABLE IF EXISTS block_vectors CASCADE;
    DROP TABLE IF EXISTS base_node CASCADE; DROP SEQUENCE IF EXISTS node_aid_seq CASCADE;""")
c.close()

from core.providers.postgres import PostgresStorageProvider
from data import lance_stats as ls

NS, UID = "lstat", "lstatuser"
prov = PostgresStorageProvider(dsn=DSN, namespace=NS, user_id=UID, dim=8)
for i in range(3):
    a = prov.next_id()
    prov[a] = {"text": f"row {i}", "block_class": "FACT", "recallable": True,
               "topics": ["t1", "t2"], "entities": ["e1"], "claims": ["c1"],
               "semantic_links": {"100": 0.5}}
    prov.store_embedding(a, (np.ones(8, dtype=np.float32) / np.sqrt(8)), "archive")
a = prov.next_id()
prov[a] = {"text": "bare", "recallable": False}   # no topics/entities/vector

ov = ls.overview()
chk("overview is postgres-backed", ov.get("backend") == "postgres", ov.get("backend"))
chk("overview lists base_node + archive with rows",
    {t["name"] for t in ov["tables"]} >= {"base_node", "archive"} and
    any(t["name"] == "base_node" and t.get("rows", 0) >= 4 for t in ov["tables"]),
    [(t["name"], t.get("rows")) for t in ov["tables"]])

cov = ls.nlp_coverage(UID)
chk("nlp_coverage.archive_rows == 4", cov.get("archive_rows") == 4, cov.get("archive_rows"))
chk("topics covered == 3", cov["stages"]["topics"]["covered"] == 3, cov["stages"].get("topics"))
chk("entities covered == 3", cov["stages"]["entities"]["covered"] == 3, cov["stages"].get("entities"))
chk("semantic_links covered == 3", cov["stages"]["semantic_links"]["covered"] == 3, cov["stages"].get("semantic_links"))
chk("block_class covered == 3", cov["stages"]["block_class"]["covered"] == 3, cov["stages"].get("block_class"))
chk("vector_coverage covered == 3", cov["vector_coverage"]["covered"] == 3, cov.get("vector_coverage"))
chk("recallable_total == 3", cov.get("recallable_total") == 3, cov.get("recallable_total"))
chk("class_breakdown has FACT(3 recallable)",
    cov.get("class_breakdown", {}).get("FACT", {}).get("recallable") == 3, cov.get("class_breakdown"))
chk("embeddings_by_type archive == 3", cov["embeddings_by_type"].get("archive") == 3, cov.get("embeddings_by_type"))

fts = ls.fts_index_status()
chk("fts_index_status built (ix_base_fts present)", fts.get("built") is True, fts)
chk("fts reports vector index too", fts.get("vector_index") is True, fts)

c = psycopg2.connect(DSN); c.autocommit = True
with c.cursor() as cur:
    cur.execute("DELETE FROM archive a USING base_node b WHERE a.aid=b.aid AND b.user_id=%s", [UID])
    cur.execute("DELETE FROM base_node WHERE user_id=%s", [UID])
c.close()
print("\n" + "="*60)
print("LANCE-STATS-ON-PG RESULT:", "ALL PASS" if not fails else f"FAILED {fails}")
print("="*60)
sys.exit(1 if fails else 0)
