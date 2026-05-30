"""Local 'run it' — blazingly-fast redaction demo (PII_UNIFY_2026_05_29).

Simulates a realistic multi-turn conversation through the UNIFIED redactor
(the proxy's exact call: redactor.redact_json(body, sid)) and prints
per-turn latency + block-cache behavior.  Then runs a BASELINE that clears
the caches every turn (≈ the old uncached proxy redactor) to quantify the
win.  Finally proves correctness: no raw PII survives, and rehydration
restores it.

Run:  python _perf_local_demo.py
"""
import os, sys, json, time, tempfile

os.environ.setdefault("VEILGUARD_PII_DB_PATH", tempfile.mkdtemp(prefix="pii_perf_"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pii import get_redactor  # noqa: E402

MAGIC = "You are a Claude agent, built on Anthropic's Claude Agent SDK."

# A static, PII-free "preamble" + "tool schema" bulk (CLEAN / FROZEN).
PREAMBLE = ("VEILGUARD OPERATING CONTEXT.\n" + ("Policy line about safe tool use and tenant isolation. " * 400))
TOOLS_BLOCK = "TOOL SCHEMAS\n" + json.dumps(
    [{"name": f"tool_{i}", "description": "does a thing " * 30,
      "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}}}
     for i in range(18)]
)

# Frozen memory fragments, each with real PII, byte-stable across turns.
FROZEN_FRAGS = [
    "[Memory index=0 | role=USER | src=live]\nAlice Johnson emailed alice.johnson@acme.co.za about the Q3 plan; call her on 082 123 4567.",
    "[Memory index=1 | role=ASSISTANT | src=live]\nNoted. Looped in Brian Naidoo (brian.naidoo@acme.co.za) and finance.",
    "[Memory index=2 | role=USER | src=live]\nPriya Pillay said account 6271045839 and ID 8001015009087 are correct.",
    "[Memory index=3 | role=ASSISTANT | src=live]\nConfirmed with Thabo Mokoena at +27 31 555 0199.",
]

def build_body(turn: int) -> dict:
    """A growing conversation: stable prefix + one new live fragment + a new user msg each turn."""
    live = [
        f"[Memory index={4+t} | role=USER | src=live]\nTurn {t}: Sipho Dlamini (sipho{t}@acme.co.za) flagged invoice {1000+t}."
        for t in range(turn + 1)
    ]
    system = (
        [{"type": "text", "text": MAGIC}]                                  # CLEAN
        + [{"type": "text", "text": PREAMBLE}]                             # CLEAN (fingerprint)
        + [{"type": "text", "text": TOOLS_BLOCK,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}}]          # FROZEN (no PII, cached)
        + [{"type": "text", "text": f} for f in FROZEN_FRAGS]              # FROZEN (PII, cached)
        + [{"type": "text", "text": x} for x in live]                      # per-fragment: each turn adds ONE new block
    )
    return {
        "model": "claude-opus-4-8",
        "system": system,
        "messages": [
            {"role": "assistant", "content": "Prior assistant turn — must NOT be re-scanned."},
            {"role": "user",
             "content": f"Turn {turn}: please email Nomsa Khumalo nomsa.k{turn}@acme.co.za the summary."},
        ],
        "tools": [{"name": "noop", "description": "x", "input_schema": {"type": "object"}}],
    }

def total_chars(body):
    return sum(len(b["text"]) for b in body["system"]) + sum(
        len(m["content"]) for m in body["messages"])

TURNS = 6
sid = "pii-perfdemo"

print(f"=== payload ~{total_chars(build_body(TURNS-1))//1000} KB system+msgs, {TURNS} turns ===\n")

# ---- UNIFIED (caches ON) ----
r = get_redactor()
print("UNIFIED redactor (block cache + clean-skip + line cache):")
unified_ms = []
for t in range(TURNS):
    body = build_body(t)
    t0 = time.perf_counter()
    red = r.redact_json(body, sid)
    ms = (time.perf_counter() - t0) * 1000
    unified_ms.append(ms)
    # correctness: no raw PII in the redacted system/messages
    blob = json.dumps(red)
    leaked = [s for s in ["alice.johnson@acme.co.za", "082 123 4567", "8001015009087",
                          "brian.naidoo@acme.co.za", "Nomsa Khumalo"] if s in blob]
    print(f"  turn {t}: {ms:7.1f} ms   hits={r._block_hits} miss={r._block_misses} "
          f"clean_skipped={r._clean_skipped}   leaked={leaked or 'none'}")

# ---- BASELINE (simulate old uncached redactor: clear caches each turn) ----
print("\nBASELINE (caches cleared each turn ~= old uncached proxy redactor):")
base_ms = []
for t in range(TURNS):
    r._block_cache.clear()
    r._clean_skip = False  # force-scan everything, like the old path that never skipped
    body = build_body(t)
    t0 = time.perf_counter()
    r.redact_json(body, sid)
    ms = (time.perf_counter() - t0) * 1000
    base_ms.append(ms)
    print(f"  turn {t}: {ms:7.1f} ms")
r._clean_skip = True

# ---- correctness: rehydrate restores a frozen fragment ----
body = build_body(TURNS - 1)
red = r.redact_json(body, sid)
frag = red["system"][3]["text"]                 # first frozen PII fragment
rehydrated = r.rehydrate_text(frag, sid)
restored = all(s in rehydrated for s in ["Alice Johnson", "alice.johnson@acme.co.za"])
asst_intact = red["messages"][0]["content"] == "Prior assistant turn — must NOT be re-scanned."

print("\n=== SUMMARY ===")
print(f"UNIFIED  cold(turn0)={unified_ms[0]:.1f}ms   warm avg(turns1+)={sum(unified_ms[1:])/(TURNS-1):.1f}ms")
print(f"BASELINE cold(turn0)={base_ms[0]:.1f}ms       avg(all)={sum(base_ms)/TURNS:.1f}ms")
warm = sum(unified_ms[1:])/(TURNS-1)
base = sum(base_ms[1:])/(TURNS-1)
print(f"WARM SPEEDUP: {base/warm:.1f}x  ({base:.1f}ms → {warm:.1f}ms per warm turn)")
print(f"correctness: rehydrate_restores_frozen={restored}  assistant_untouched={asst_intact}")
print("RESULT:", "PASS" if (restored and asst_intact) else "FAIL")
