"""Self-contained ledger Lance->Postgres copy test: build a tiny Lance ledger
with a row in two tables, migrate, verify the rows land in Postgres."""
import sys, os, tempfile
AR = r"C:\Users\rudol\Documents\veilguard\agent-runtime"
sys.path.insert(0, AR)
DSN = "postgresql://tcmm:tcmm@localhost:5433/tcmm"
os.environ["LEDGER_BACKEND"] = "postgres"
os.environ["TCMM_DATABASE_URL"] = DSN
import pyarrow as pa, psycopg2
from app.ledger.schemas import TABLE_SCHEMAS
from app.ledger.store import LedgerStore
from app.ledger.pg_store import PgLedgerStore
from app.ledger.migrate_ledger import migrate_ledger

TABLES = ["task_proposals", "agent_tasks"]

def default_for(t):
    if pa.types.is_integer(t): return 1
    if pa.types.is_floating(t): return 1.0
    if pa.types.is_boolean(t): return False
    if pa.types.is_string(t) or pa.types.is_large_string(t): return "x"
    if pa.types.is_list(t) or pa.types.is_large_list(t): return []
    if pa.types.is_timestamp(t): return 0
    return None

def row_for(name):
    schema = TABLE_SCHEMAS[name]
    if callable(schema):
        schema = schema()
    return {f.name: default_for(f.type) for f in schema}

# clean the PG ledger tables
conn = psycopg2.connect(DSN); conn.autocommit = True
with conn.cursor() as c:
    for t in TABLES:
        c.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
conn.close()

# build a tiny Lance ledger
ld = tempfile.mkdtemp(prefix="ledger_src_")
store = LedgerStore(db_path=ld)
for t in TABLES:
    store.table(t).add([row_for(t)])

counts = migrate_ledger(ld, DSN)
dst = PgLedgerStore(DSN)
fails = []
def chk(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ")+name+("" if cond else f"   <<< {extra}"))
    if not cond: fails.append(name)

for t in TABLES:
    chk(f"{t}: migrate reports 1 row", counts.get(t) == 1, counts.get(t))
    chk(f"{t}: 1 row present in postgres", dst.table(t).count_rows() == 1, dst.table(t).count_rows())

print("\n" + "#"*60)
print("LEDGER-MIGRATE RESULT:", "ALL PASS" if not fails else f"FAILED {fails}")
print("#"*60)
sys.exit(1 if fails else 0)
