#!/usr/bin/env python3
"""PII allow-list SHADOW-MODE signal collector (v1, read-only).

Reads the already-stored typed entities (`entity_dicts`) from TCMM memory,
keeps the NON-person entities the boundary redactor WOULD flag as PII, and
logs them to `redaction_suggestions`. Changes NOTHING live — no allow_list
edit, no enrichment/schema/redactor change. Run inside the pii-proxy
container (has `pii` + the PG):
    docker exec -i veilguard-pii-proxy-1 python3 - < _redaction_shadow_scan.py
"""
import sys, json
for p in ["/", "/app", "/app/.."]:
    sys.path.insert(0, p)
import psycopg2
from pii import get_redactor
from pii.session_store import _PG_DSN

# Entity types that are (or may be) private PII -> NEVER whitelist candidates.
PRIVATE_TYPES = {"person", "people", "name", "email", "phone", "contact",
                 "ssn", "id_number", "account", "address"}
# Types that are noise for whitelisting (ephemeral refs), skip silently.
SKIP_TYPES = {"identifier", "topic", "date", "time", "expense", "transport"}

r = get_redactor()
sid = "redaction-shadow-scan"

conn = psycopg2.connect(_PG_DSN)
conn.autocommit = True
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS redaction_suggestions (
  term            text,
  kind            text,            -- 'whitelist' | 'denylist'
  entity_type     text,
  support_count   int  DEFAULT 0,  -- # distinct blocks corroborating
  redactor_flags  bool DEFAULT true,
  first_seen      timestamptz DEFAULT now(),
  last_seen       timestamptz DEFAULT now(),
  status          text DEFAULT 'staged',  -- staged|proposed|applied|rejected
  PRIMARY KEY (term, kind)
);
""")

cur.execute("SELECT entity_dicts FROM v_archive "
            "WHERE entity_dicts IS NOT NULL AND entity_dicts::text NOT IN ('[]','null')")
rows = cur.fetchall()

from collections import defaultdict
agg = defaultdict(lambda: {"count": 0, "type": ""})
scanned = 0
for (ed,) in rows:
    if not ed:
        continue
    if isinstance(ed, str):
        try: ed = json.loads(ed)
        except Exception: continue
    seen = set()
    for e in (ed or []):
        if not isinstance(e, dict):
            continue
        name = (e.get("name") or "").strip()
        etype = (e.get("type") or "").strip().lower()
        if len(name) < 3 or not name[0].isupper():
            continue                       # only proper-noun-ish (what NER over-flags)
        if etype in PRIVATE_TYPES or etype in SKIP_TYPES:
            continue                       # never whitelist private; skip noise
        scanned += 1
        # Does the boundary redactor actually flag this non-person entity as PII?
        try:
            red = r.redact_text(name, sid)
        except Exception:
            continue
        if "REF_" not in red:
            continue                       # redactor leaves it alone -> nothing to fix
        if name in seen:
            continue
        seen.add(name)
        agg[name]["count"] += 1
        agg[name]["type"] = etype

for term, v in agg.items():
    cur.execute(
        """INSERT INTO redaction_suggestions (term, kind, entity_type, support_count, last_seen)
           VALUES (%s,'whitelist',%s,%s, now())
           ON CONFLICT (term, kind) DO UPDATE
             SET support_count=EXCLUDED.support_count, entity_type=EXCLUDED.entity_type,
                 last_seen=now()""",
        (term, v["type"], v["count"]))

print(f"scanned {len(rows)} blocks, {scanned} candidate-entity occurrences")
print(f"=> {len(agg)} distinct WHITELIST candidates (NLP non-person + redactor flags as PII)\n")
cur.execute("SELECT term, entity_type, support_count FROM redaction_suggestions "
            "WHERE kind='whitelist' ORDER BY support_count DESC, term LIMIT 35")
print(f"{'cnt':>4}  {'type':<14} term")
for term, etype, cnt in cur.fetchall():
    print(f"{cnt:>4}  {(etype or '?'):<14} {term}")
