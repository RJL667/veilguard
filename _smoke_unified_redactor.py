"""Smoke test for the unified veilguard.pii redactor (PII_UNIFY_2026_05_29).

Run: VEILGUARD_PII_DB_PATH=<tmp> python _smoke_unified_redactor.py
Exercises proxy-style string sids, redact_json on an Anthropic body, the
FROZEN block cache, CLEAN-skip, metadata stripping, rehydration, and
fail-closed.
"""
import os, sys, tempfile, json

os.environ.setdefault("VEILGUARD_PII_DB_PATH", tempfile.mkdtemp(prefix="pii_smoke_"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pii import get_redactor, RedactionUnavailable, get_store, SessionId  # noqa: E402

MAGIC = "You are a Claude agent, built on Anthropic's Claude Agent SDK."
r = get_redactor()
ok = True
def check(name, cond):
    global ok
    print(("PASS" if cond else "FAIL"), name)
    ok = ok and cond

# 1. redact_text with a plain STRING sid (proxy per-user key).
sid = "pii-petrus"
out = r.redact_text("Email Alice Smith at alice@acme.com about the deal.", sid)
check("1 redact_text str sid detects email+person", "REF_" in out and "alice@acme.com" not in out)

# 2. redact_json on an Anthropic-shape body.
body = {
    "model": "claude-opus-4-8",
    "system": [
        {"type": "text", "text": MAGIC},  # CLEAN — must pass untouched
        {"type": "text",
         "text": "--- MEMORY ---\nContact: Bob Jones, bob@acme.com, +27 21 555 0100",
         "cache_control": {"type": "ephemeral", "ttl": "1h"},
         "_vg_aid": 7, "_vg_stability": "byte_stable"},
    ],
    "messages": [
        {"role": "user", "content": "Call Carol Wang on 0821234567 please."},
        {"role": "assistant", "content": "Sure, I'll reach REF_PERSON_99 — newline=test"},
    ],
}
red = r.redact_json(body, sid)
sysblocks = red["system"]
check("2a magic-prefix block unchanged (CLEAN skip)", sysblocks[0]["text"] == MAGIC)
check("2b memory block redacted", "bob@acme.com" not in sysblocks[1]["text"] and "REF_" in sysblocks[1]["text"])
check("2c cache_control preserved on memory block", sysblocks[1].get("cache_control") == {"type": "ephemeral", "ttl": "1h"})
check("2d _vg_*/_skip_pii stripped before wire", not any(k.startswith("_vg") or k == "_skip_pii" for k in sysblocks[1]))
check("2e user message redacted", "REF_" in red["messages"][0]["content"])
check("2f assistant message NOT re-redacted (verbatim)", red["messages"][1]["content"] == body["messages"][1]["content"])

# 3. AID cache: redact same body again → byte-identical + an aid-cache hit.
hits_before = r._aid_hits
# also flip the cache marker to prove the aid key ignores marker churn
body2 = json.loads(json.dumps(body)); body2["system"][1]["cache_control"]["ttl"] = "5m"
red2 = r.redact_json(body2, sid)
check("3a second pass redaction byte-identical (deterministic)",
      red2["system"][1]["text"] == red["system"][1]["text"])
check("3b aid cache hit despite cache-marker flip", r._aid_hits > hits_before)

# 4. Rehydrate round-trips the memory block.
rehydrated = r.rehydrate_text(sysblocks[1]["text"], sid)
check("4a rehydrate restores email", "bob@acme.com" in rehydrated)
check("4b rehydrate restores person", "Bob Jones" in rehydrated)
check("4c rehydrate fast-path no-op when no REF_", r.rehydrate_text("plain text", sid) == "plain text")

# 5. Fail-closed: analyzer failure must RAISE, not return raw.
class Boom:
    def analyze(self, *a, **k): raise RuntimeError("presidio down")
_orig = r.analyzer
r.analyzer = Boom()
raised = False
try:
    r.redact_text("Fresh PII: Dave Lee dave@x.com unseen-line-zzz", "pii-other")
except RedactionUnavailable:
    raised = True
finally:
    r.analyzer = _orig
check("5 fail-closed raises RedactionUnavailable on analyzer failure", raised)

# 6. /rehydrate-style session-less lookup.
glob = get_store().rehydrate_any(sysblocks[1]["text"])
check("6 rehydrate_any (session-less) restores values", "bob@acme.com" in glob)

print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
