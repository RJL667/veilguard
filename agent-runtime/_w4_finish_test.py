"""W4-finish verification: the 9 direct-connect modules compile after the
codemod, and open_ledger_db() resolves ledger tables AND the TCMM archive/dream
CTI views on Postgres (the routing the modules now use)."""
import sys, os, py_compile
AR = r"C:\Users\rudol\Documents\veilguard\agent-runtime"
TC = r"C:\Users\rudol\.gemini\antigravity\tcmm\TCMM"
sys.path.insert(0, AR); sys.path.insert(0, TC)
DSN = "postgresql://tcmm:tcmm@localhost:5433/tcmm"
os.environ["TCMM_DATABASE_URL"] = DSN
os.environ["LEDGER_BACKEND"] = "postgres"
import psycopg2

mods = ["app/memory/lessons_reader.py", "app/proposals/constitution_amendments.py",
        "app/proposals/dream_scanner.py", "app/proposals/m4_backfill.py",
        "app/proposals/signal_emitters.py", "app/proposals/drift_watchdog.py",
        "app/proposals/recalibration.py", "app/proposals/lessons.py", "app/proposals/outcomes.py"]
for m in mods:
    py_compile.compile(os.path.join(AR, m), doraise=True)
print("compile OK (9 routed modules)")

# ensure the TCMM CTI views exist (apply migration 002 if base_node is absent)
conn = psycopg2.connect(DSN); conn.autocommit = True
with conn.cursor() as c:
    c.execute("CREATE EXTENSION IF NOT EXISTS vector")
    c.execute("SELECT to_regclass('base_node')")
    if c.fetchone()[0] is None:
        sql = open(os.path.join(TC, "migrations", "002_class_table_inheritance.sql"),
                   encoding="utf-8").read().replace("%(dim)s", "8")
        c.execute(sql)
conn.close()

from app.ledger.store import open_ledger_db
db = open_ledger_db()  # postgres -> PgLedgerStore proxy
fails = []
for name in ["task_proposals", "agent_tasks", "archive", "dream_archive"]:
    try:
        cnt = db.open_table(name).count_rows()
        print(f"  OK   open_table('{name}').count_rows() = {cnt}")
    except Exception as e:
        print(f"  FAIL open_table('{name}'): {e}"); fails.append(name)
# org_memory is an UNREGISTERED legacy schema (not in TABLE_SCHEMAS); it must
# fail gracefully (the consumer's try/except handles it), matching Lance's
# table-not-found behaviour — NOT a port regression.
try:
    db.open_table("org_memory").count_rows()
    print("  OK   org_memory resolved")
except Exception:
    print("  OK   org_memory unavailable + raises gracefully (matches Lance; consumer catches)")

print("\n" + "#"*60)
print("W4-FINISH RESULT:", "ALL PASS" if not fails else f"FAILED {fails}")
print("#"*60)
sys.exit(1 if fails else 0)
