#!/usr/bin/env python3
"""Promote redaction_suggestions -> pii_allow_list with ASYMMETRIC safety gates.

The whitelist direction is the dangerous one: wrongly whitelisting a real
name = a name is never redacted again = permanent silent PII leak. So
promotion is deliberately conservative and biased toward "hold, don't apply":

  SKIP/REJECT  junk artifacts (REF_*/token/placeholder — the NLP extracting
               redaction tokens from text ABOUT redaction) and person-ish types.
  AUTO-APPLY   only CLEARLY non-person public types (technology/tool/product/
               malware/...) with support_count >= K  -> insert into pii_allow_list.
  PROPOSE      everything else — organizations (can be surname-companies),
               locations, low-support, and any term in the hard-deny surname
               seed -> status='proposed', held for an LLM/human confirm step.

Idempotent: re-running only advances staged/proposed rows. Run in the
pii-proxy container (has `pii` + the PG):
    docker exec -i veilguard-pii-proxy-1 python3 - < promote_redaction_suggestions.py
"""
import os
import sys

for _p in ["/", "/app", "/app/.."]:
    sys.path.insert(0, _p)
import psycopg2
from pii.session_store import _PG_DSN

K = int(os.environ.get("PROMOTE_SUPPORT_K", "2"))  # corroboration threshold for auto-apply

# Redaction-token artifacts — NEVER real entities, never whitelist.
JUNK = ("ref_", "token", "placeholder")
# Types that are (or may be) a private person -> never auto-whitelist.
PERSON_TYPES = {"person", "people", "name", "email", "phone", "contact",
                "id", "ssn", "account", "address", "location"}
# Clearly NON-person public types -> eligible for auto-apply.
AUTO_TYPES = {"technology", "tool", "product", "technique", "malware",
              "software", "threat_actor", "ransomware", "protocol", "standard"}
# Common surnames that are ALSO words/companies -> the canonical leak trap.
# Even when typed "organization", these are held for confirm, never auto-applied.
HARD_DENY = {"gallagher", "walker", "carter", "mason", "cooper", "porter",
             "hunter", "fletcher", "tanner", "parker", "archer", "turner",
             "baker", "cook", "ward", "cole", "reed", "bell", "wood", "gray",
             "young", "hill", "green", "stone", "fox", "wells", "banks",
             "marshall", "marsh", "cross", "king", "price", "hunt", "knight"}

_REDACTOR = None


def _detected_spans(term):
    """The redactor's ACTUAL detected PERSON-ish span(s) within ``term`` — what
    the allow_list must match (Presidio's allow_list is span-exact). Presidio
    detects e.g. 'Akira EDR' inside the NLP entity 'Akira EDR-killer', so
    whitelisting the raw phrase never matches the redaction. Falls back to the
    raw term when nothing is detected."""
    global _REDACTOR
    try:
        if _REDACTOR is None:
            from pii import get_redactor
            _REDACTOR = get_redactor()
        from pii.redactor import PII_ENTITIES
        res = _REDACTOR.analyzer.analyze(text=term, language="en", entities=PII_ENTITIES)
        spans = {term[x.start:x.end] for x in res if x.score >= 0.5}
        return spans or {term}
    except Exception:
        return {term}


def main():
    conn = psycopg2.connect(_PG_DSN)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        "SELECT term, entity_type, COALESCE(support_count,0) "
        "FROM redaction_suggestions "
        "WHERE kind='whitelist' AND status IN ('staged','proposed')"
    )
    rows = cur.fetchall()

    applied, proposed, skipped = [], [], []
    for term, etype, support in rows:
        tl = (term or "").lower().strip()
        et = (etype or "").lower().strip()
        if any(j in tl for j in JUNK):
            skipped.append((term, "junk-artifact"))
            new_status = "rejected"
        elif et in PERSON_TYPES:
            skipped.append((term, f"person-type:{et}"))
            new_status = "rejected"
        elif len(tl.split()) == 1 and tl in HARD_DENY:
            proposed.append((term, "ambiguous-surname (hard-deny seed)"))
            new_status = "proposed"
        elif et in AUTO_TYPES and support >= K:
            # [SPAN_ALIGN] whitelist the redactor's ACTUAL detected span(s), not
            # the raw NLP phrase — Presidio's allow_list match is span-exact.
            wl = _detected_spans(term)
            for span in wl:
                cur.execute(
                    "INSERT INTO pii_allow_list (term, entity_type, source, added_by, added_at, active) "
                    "VALUES (%s,%s,'auto-promote','promoter',now(),true) "
                    "ON CONFLICT (term) DO UPDATE SET active=true, source='auto-promote'",
                    (span, etype),
                )
            applied.append((term, f"{et}/support={support} -> spans {sorted(wl)}"))
            new_status = "applied"
        else:
            reason = f"{et or '?'}/support={support}" + (
                " <K" if support < K else " non-auto-type (needs confirm)")
            proposed.append((term, reason))
            new_status = "proposed"
        cur.execute(
            "UPDATE redaction_suggestions SET status=%s, last_seen=now() "
            "WHERE term=%s AND kind='whitelist'",
            (new_status, term),
        )

    print(f"=== promotion run: {len(applied)} APPLIED, {len(proposed)} PROPOSED, {len(skipped)} SKIPPED (K={K}) ===")
    print("APPLIED -> pii_allow_list (auto, clearly non-person + corroborated):")
    for t, r in applied:
        print(f"  + {t!r:<32} {r}")
    print("PROPOSED (held for LLM/human confirm — never auto-applied):")
    for t, r in proposed:
        print(f"  ? {t!r:<32} {r}")
    print("SKIPPED/REJECTED:")
    for t, r in skipped:
        print(f"  x {t!r:<32} {r}")


if __name__ == "__main__":
    main()
