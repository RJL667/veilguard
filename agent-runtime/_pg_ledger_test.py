"""W4 foundation test: the Lance-table-API shim (PgTable) over Postgres handles
the full ledger CRUD surface the consumers use — add / search().where() /
merge_insert upsert / count_rows / update / delete — on a real ledger schema."""
import sys, os, datetime
sys.path.insert(0, r"C:\Users\rudol\Documents\veilguard\agent-runtime")
os.environ["TCMM_DATABASE_URL"] = "postgresql://tcmm:tcmm@localhost:5433/tcmm"
import pyarrow as pa, psycopg2
from app.ledger.schemas import TABLE_SCHEMAS
from app.ledger.pg_store import PgLedgerStore

DSN = "postgresql://tcmm:tcmm@localhost:5433/tcmm"
c = psycopg2.connect(DSN); c.autocommit = True
with c.cursor() as cur:
    for name in TABLE_SCHEMAS:
        cur.execute(f"DROP TABLE IF EXISTS {name} CASCADE")
c.close()

def mkrow(schema, **over):
    row = {}
    for f in schema:
        if f.name in over:
            row[f.name] = over[f.name]; continue
        t = f.type
        if pa.types.is_string(t): row[f.name] = "x"
        elif pa.types.is_boolean(t): row[f.name] = False
        elif pa.types.is_floating(t): row[f.name] = 0.0
        elif pa.types.is_integer(t): row[f.name] = 0
        elif pa.types.is_timestamp(t): row[f.name] = datetime.datetime.now(datetime.timezone.utc)
        elif pa.types.is_list(t): row[f.name] = []
        else: row[f.name] = {}
    return row

fails = []
def chk(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ")+name+("" if cond else f"   <<< {extra}"));
    if not cond: fails.append(name)

store = PgLedgerStore.get()
chk("PgLedgerStore created all ledger tables", len(store._created) == len(TABLE_SCHEMAS), f"{len(store._created)}/{len(TABLE_SCHEMAS)}")

t = store.table("agent_tasks")
t.add([mkrow(t.schema, id="T1", status="open", user_id="u1")])
rows = t.search().where("user_id = 'u1'").to_arrow().to_pylist()
chk("add + search().where().to_arrow().to_pylist()", len(rows) == 1 and rows[0]["id"] == "T1", rows)

# merge_insert upsert: T1 -> done, T2 new
t.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute([
    mkrow(t.schema, id="T1", status="done", user_id="u1"),
    mkrow(t.schema, id="T2", status="open", user_id="u1"),
])
r1 = t.search().where("id = 'T1'").to_pylist()
chk("merge_insert updated existing (T1 -> done)", r1 and r1[0]["status"] == "done", r1)
chk("merge_insert inserted new (T2)", t.count_rows("id = 'T2'") == 1)
chk("count_rows(filter)", t.count_rows("user_id = 'u1'") == 2)

t.update(where="id = 'T1'", values={"status": "closed"})
chk("update(where, values)", t.search().where("id = 'T1'").to_pylist()[0]["status"] == "closed")

t.delete("id = 'T2'")
chk("delete(filter)", t.count_rows() == 1)

# a different table + jsonb column round-trip
hb = store.table("agent_task_heartbeats")
hb.add([mkrow(hb.schema, id="H1", task_id="T1", user_id="u1")])
chk("second table (heartbeats) add + count", hb.count_rows() == 1)

# LedgerStore.get() routing + the inbox_poller `store._db.open_table(name)` path
os.environ["LEDGER_BACKEND"] = "postgres"
from app.ledger.store import LedgerStore
gs = LedgerStore.get()
chk("LedgerStore.get() routes to PgLedgerStore", type(gs).__name__ == "PgLedgerStore", type(gs).__name__)
chk("store._db.open_table(name) proxy works (inbox_poller path)",
    gs._db.open_table("agent_tasks").count_rows() == 1)

print("\n" + "#"*64)
print("W4 LEDGER-SHIM RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
print("#"*64)
sys.exit(1 if fails else 0)
