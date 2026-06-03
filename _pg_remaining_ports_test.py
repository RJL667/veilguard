"""Verify the remaining Lance→PG ports: sub-agents TaskStore + ClientSettingsStore,
and tcmm_stats NLP-progress — all on Postgres."""
import sys, os
VG = r"C:\Users\rudol\Documents\veilguard"
DSN = "postgresql://tcmm:tcmm@localhost:5433/tcmm"
os.environ["TCMM_STORAGE"] = "postgres"
os.environ["VEILGUARD_AUDIT_BACKEND"] = "postgres"
os.environ["LEDGER_BACKEND"] = "postgres"
os.environ["TCMM_DATABASE_URL"] = DSN
import psycopg2
c = psycopg2.connect(DSN); c.autocommit = True
with c.cursor() as cur:
    cur.execute("DROP TABLE IF EXISTS tasks")
    cur.execute("DROP TABLE IF EXISTS client_settings")
    cur.execute("DROP TABLE IF EXISTS client_settings_conv")
c.close()

fails = []
def chk(n, cond, x=""):
    print(("  OK   " if cond else "  FAIL ")+n+("" if cond else f"   <<< {x}")); fails.append(n) if not cond else None

# ── sub-agents TaskStore ──
sys.path.insert(0, os.path.join(VG, "services", "sub-agents"))
from core.tasks.store import TaskStore
from core.tasks.model import Task
ts = TaskStore()
ts.upsert(Task(task_id="t1", user_id="u1", root_task_id="t1", kind="tool_call",
               title="T", payload={"a": 1}, status="pending"))
ts.upsert(Task(task_id="t2", user_id="u1", root_task_id="t1", parent_task_id="t1",
               kind="tool_call", title="child", status="pending"))
g = ts.get_task("t1")
chk("TaskStore upsert+get round-trip", g and g.task_id == "t1" and g.payload.get("a") == 1, g)
chk("TaskStore mark_status", ts.mark_status("t1", "running") and ts.get_task("t1").status == "running")
chk("TaskStore list_user (2 tasks)", len(ts.list_user("u1")) == 2, len(ts.list_user("u1")))
chk("TaskStore list_children of t1", len(ts.list_children("t1")) == 1, len(ts.list_children("t1")))
chk("TaskStore list_subtree(t1) = 2", len(ts.list_subtree("t1")) == 2, len(ts.list_subtree("t1")))

# ── sub-agents ClientSettingsStore ──
from core.client_settings import ClientSettingsStore
cs = ClientSettingsStore()
cs.set_user_level("u1", "strict")
chk("ClientSettings user level persists", cs.get_user_level("u1") == ("strict", 60), cs.get_user_level("u1"))
cs.set_conv_override("u1", "conv1", "auto")
chk("conv override wins in get_effective", cs.get_effective("u1", "conv1")[0] == "auto", cs.get_effective("u1", "conv1"))
cs.clear_conv_override("u1", "conv1")
chk("after clear -> user default", cs.get_effective("u1", "conv1")[0] == "strict", cs.get_effective("u1", "conv1"))

# ── admin tcmm_stats NLP progress (Postgres) ──
sys.path.insert(0, os.path.join(VG, "services", "admin-dashboard"))
from data import tcmm_stats
prog = tcmm_stats._pg_nlp_progress()
chk("tcmm_stats pg progress shape", set(prog) >= {"archive_total", "nlp_processed", "embed_processed"}, prog)

with psycopg2.connect(DSN) as cc, cc.cursor() as cur:
    cur.execute("DROP TABLE IF EXISTS tasks")
    cur.execute("DROP TABLE IF EXISTS client_settings")
    cur.execute("DROP TABLE IF EXISTS client_settings_conv")
print("\n" + "="*58)
print("REMAINING-PORTS-ON-PG:", "ALL PASS" if not fails else f"FAILED {fails}")
print("="*58)
sys.exit(1 if fails else 0)
