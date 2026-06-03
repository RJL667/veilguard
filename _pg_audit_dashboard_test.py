"""Admin dashboard on Postgres: write pii_audit rows via the (PG-backend)
audit_db writer, then read them back through the dashboard's stats layer
(overview / per_user / per_agent / cache_overview / recent_redactions /
request_detail) and verify shapes — proving the whole panel works on Postgres."""
import sys, os, time
VG = r"C:\Users\rudol\Documents\veilguard"
sys.path.insert(0, VG)
sys.path.insert(0, os.path.join(VG, "llm"))
sys.path.insert(0, os.path.join(VG, "services", "admin-dashboard"))
DSN = "postgresql://tcmm:tcmm@localhost:5433/tcmm"
os.environ["VEILGUARD_AUDIT_BACKEND"] = "postgres"
os.environ["VEILGUARD_AUDIT_DATABASE_URL"] = DSN
os.environ["TCMM_DATABASE_URL"] = DSN
import psycopg2

fails = []
def chk(n, c, x=""):
    print(("  OK   " if c else "  FAIL ")+n+("" if c else f"   <<< {x}"))
    if not c: fails.append(n)

c = psycopg2.connect(DSN); c.autocommit = True
with c.cursor() as cur:
    cur.execute("DROP TABLE IF EXISTS pii_audit")
c.close()

import audit_db
from data import pii_audit_stats as pas

UID = "a1b2c3d4e5f6a7b8c9d0e1f2"   # hex-24 tenant (survives record()'s bad-user guard)
def rec(direction, conv, content, **kw):
    audit_db.record(direction=direction, conversation_id=conv, content=content,
                    user_id=UID, model="claude-haiku-4-5", **kw)

# two request/response pairs, with redaction tokens + cache usage + agent tags
rec("TO_LLM",  "c1", "hi REF_PERSON_1 and REF_EMAIL_2", extra={"agent_id": "researcher", "task_id": "t1"})
rec("FROM_LLM","c1", "reply", tokens_input=1000, tokens_output=50, cache_read=800, cache_create=0,
    extra={"agent_id": "researcher", "task_id": "t1"})
rec("TO_LLM",  "c2", "world REF_PHONE_1", extra={"agent_id": "builder"})
rec("FROM_LLM","c2", "ok", tokens_input=500, tokens_output=20, cache_read=0, cache_create=400,
    extra={"agent_id": "builder"})

# the writer flushes on a 2s timer; wait for the background drain
time.sleep(3)

c = psycopg2.connect(DSN); c.autocommit = True
with c.cursor() as cur:
    cur.execute("SELECT count(*) FROM pii_audit")
    n = cur.fetchone()[0]
c.close()
chk("4 rows written to Postgres pii_audit", n == 4, n)

ov = pas.overview(window_hours=1)
chk("overview.total_rows == 4", ov.get("total_rows") == 4, ov.get("total_rows"))
chk("overview.tokens_by_kind has PERSON/EMAIL/PHONE",
    set(ov.get("tokens_by_kind", {})) >= {"PERSON", "EMAIL", "PHONE"}, ov.get("tokens_by_kind"))
chk("overview.users_seen == 1", ov.get("users_seen") == 1, ov.get("users_seen"))

pu = pas.per_user(window_hours=1)
chk("per_user shows the tenant w/ 2 calls",
    any(u["user_id"] == UID and u["calls"] == 2 for u in pu), pu)

pa_ = pas.per_agent(window_hours=1)
chk("per_agent shows researcher + builder",
    {a["agent_id"] for a in pa_} >= {"researcher", "builder"}, [a["agent_id"] for a in pa_])

co = pas.cache_overview(window_hours=1)
chk("cache_overview.from_llm_rows == 2", co.get("from_llm_rows") == 2, co.get("from_llm_rows"))
chk("cache_overview.cache_read_tokens == 800", co.get("cache_read_tokens") == 800, co.get("cache_read_tokens"))
chk("cache_overview.cache_hits == 1", co.get("cache_hits") == 1, co.get("cache_hits"))

rr = pas.recent_redactions(window_hours=1, limit=10)
chk("recent_redactions returns 2 TO_LLM rows", len(rr) == 2, len(rr))
chk("recent_redactions paired the response tokens (1000 from c1)",
    any(r.get("tokens_input") == 1000 for r in rr), [r.get("tokens_input") for r in rr])
chk("recent_redactions carries agent_id attribution",
    any(r.get("agent_id") in ("researcher", "builder") for r in rr), [r.get("agent_id") for r in rr])

aid0 = rr[0]["aid"]
rd = pas.request_detail(aid0)
chk("request_detail returns redacted content w/ REF_ tokens",
    "REF_" in (rd.get("content") or ""), rd.get("content"))

c = psycopg2.connect(DSN); c.autocommit = True
with c.cursor() as cur:
    cur.execute("DROP TABLE IF EXISTS pii_audit")
c.close()
print("\n" + "="*60)
print("AUDIT-DASHBOARD-ON-PG RESULT:", "ALL PASS" if not fails else f"FAILED {fails}")
print("="*60)
sys.exit(1 if fails else 0)
