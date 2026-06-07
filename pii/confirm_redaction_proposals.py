#!/usr/bin/env python3
"""Confirm-gate for the self-maintaining whitelist loop.

The promoter holds borderline whitelist candidates (organizations, locations,
low-support, ambiguous surnames) as status='proposed' rather than auto-applying
them — because wrongly whitelisting a real name is a permanent PII leak. A judge
(human, or a high model) then rules each proposal public vs private/ambiguous:

  CONFIRM -> applied to pii_allow_list (SPAN-ALIGNED, like the promoter).
  REJECT  -> status='rejected' (ambiguous surname / junk), never whitelisted.

Here the verdicts are the high-model judge's call for the current batch. In
production this is an LLM-judge call per proposal (or a human review queue);
the apply/reject mechanism is identical. Run in the pii-proxy container.
"""
import sys

for _p in ["/", "/app", "/app/.."]:
    sys.path.insert(0, _p)
import psycopg2
from pii.session_store import _PG_DSN

# ---- High-model judge verdicts (lowercased term -> public/safe?) ------------
CONFIRM = {
    "dragonforce", "cobalt strike", "byovd", "sslvpn",
    "akira edr", "akira ransomware", "marsh mclennan",
}  # public threat-actors / tooling / techniques / a public insurer — not persons
REJECT = {
    "gallagher",                    # insurance broker BUT also a common surname -> leak trap
    "akira edr-killer corruption",  # junk phrase, not an entity
    "testschedulerlifecycle",       # test/code class identifier, not a public entity
}

_R = None


def _spans(term):
    """Redactor's detected span(s) within term — the allow_list is span-exact."""
    global _R
    try:
        if _R is None:
            from pii import get_redactor
            _R = get_redactor()
        from pii.redactor import PII_ENTITIES
        res = _R.analyzer.analyze(text=term, language="en", entities=PII_ENTITIES)
        s = {term[x.start:x.end] for x in res if x.score >= 0.5}
        return s or {term}
    except Exception:
        return {term}


def main():
    conn = psycopg2.connect(_PG_DSN)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT term, entity_type FROM redaction_suggestions "
                "WHERE kind='whitelist' AND status='proposed'")
    rows = cur.fetchall()

    applied, rejected, held = [], [], []
    for term, etype in rows:
        tl = (term or "").lower().strip()
        if tl in CONFIRM:
            wl = _spans(term)
            for span in wl:
                cur.execute(
                    "INSERT INTO pii_allow_list (term, entity_type, source, added_by, added_at, active) "
                    "VALUES (%s,%s,'confirm-promote','judge',now(),true) "
                    "ON CONFLICT (term) DO UPDATE SET active=true, source='confirm-promote'",
                    (span, etype),
                )
            cur.execute("UPDATE redaction_suggestions SET status='applied', last_seen=now() "
                        "WHERE term=%s AND kind='whitelist'", (term,))
            applied.append((term, sorted(wl)))
        elif tl in REJECT:
            cur.execute("UPDATE redaction_suggestions SET status='rejected', last_seen=now() "
                        "WHERE term=%s AND kind='whitelist'", (term,))
            rejected.append(term)
        else:
            held.append(term)

    print(f"=== confirm run: {len(applied)} CONFIRMED, {len(rejected)} REJECTED, {len(held)} still held ===")
    for t, w in applied:
        print(f"  + {t!r:<30} -> whitelisted spans {w}")
    for t in rejected:
        print(f"  x {t!r:<30} (ambiguous surname / junk — never whitelisted)")
    for t in held:
        print(f"  ? {t!r:<30} (no verdict — left proposed)")


if __name__ == "__main__":
    main()
