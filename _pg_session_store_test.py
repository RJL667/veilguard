"""PII session store on Postgres: append-only REF-token mapping, case-insensitive
PERSON dedup, cross-instance persistence, rehydrate + rehydrate_any."""
import sys, os
sys.path.insert(0, r"C:\Users\rudol\Documents\veilguard")
DSN = "postgresql://tcmm:tcmm@localhost:5433/tcmm"
os.environ["VEILGUARD_AUDIT_BACKEND"] = "postgres"
os.environ["VEILGUARD_AUDIT_DATABASE_URL"] = DSN
import psycopg2
c = psycopg2.connect(DSN); c.autocommit = True
with c.cursor() as cur: cur.execute("DROP TABLE IF EXISTS pii_session_mapping")
c.close()

from pii.session_store import PIISessionStore, SessionId
fails = []
def chk(n, cond, x=""):
    print(("  OK   " if cond else "  FAIL ")+n+("" if cond else f"   <<< {x}")); fails.append(n) if not cond else None

s = PIISessionStore()
sid = SessionId("tenantA", "conv1")
t1 = s.add_mapping(sid, "PERSON", "Alice")
t1b = s.add_mapping(sid, "PERSON", "alice")     # case-insensitive PERSON → same token
t2 = s.add_mapping(sid, "EMAIL", "a@b.com")
s.flush()
chk("PERSON -> REF_PERSON_1", t1 == "REF_PERSON_1", t1)
chk("case-insensitive PERSON same token", t1b == t1, t1b)
chk("EMAIL -> REF_EMAIL_1", t2 == "REF_EMAIL_1", t2)

# fresh instance = cross-process: reads persisted rows from PG (no shared memo)
s2 = PIISessionStore()
chk("cross-instance query finds token", s2._query_token(sid, "PERSON", "alice") == "REF_PERSON_1",
    s2._query_token(sid, "PERSON", "alice"))
reh = s2.rehydrate(sid, "Hello REF_PERSON_1 and REF_EMAIL_1")
chk("rehydrate swaps tokens -> originals", "Alice" in reh and "a@b.com" in reh, reh)
chk("rehydrate_any (session-less)", "Alice" in s2.rehydrate_any("see REF_PERSON_1"),
    s2.rehydrate_any("see REF_PERSON_1"))

# counter continues across instances (append-only)
t3 = s2.add_mapping(sid, "PERSON", "Bob"); s2.flush()
chk("counter continues -> REF_PERSON_2", t3 == "REF_PERSON_2", t3)

# sub-cid resolves to parent (shared mapping)
sub = SessionId("tenantA", "sub-conv1-researcher")
chk("sub-cid resolves to parent's token", s2.add_mapping(sub, "PERSON", "Alice") == "REF_PERSON_1",
    s2.add_mapping(sub, "PERSON", "Alice"))

with psycopg2.connect(DSN) as cc, cc.cursor() as cur:
    cur.execute("DROP TABLE IF EXISTS pii_session_mapping")
print("\n" + "="*56)
print("PII-SESSION-STORE-ON-PG:", "ALL PASS" if not fails else f"FAILED {fails}")
print("="*56)
sys.exit(1 if fails else 0)
