"""Benchmark the AID-keyed redaction (PII_AID_CACHE_2026_05_30).

Proves the corrected design:
  - redaction cache keyed on the live-memory-block aid (_vg_aid), redact ONCE
  - immutable blocks (preamble, tool schema, markers) NEVER redacted
  - the latest prompt redacted every turn
  - IMMUNE to cache-marker churn: cache_control flips turn-to-turn, aid hits
    are unaffected (the old content-hash key would have churned).
"""
import os, sys, time, tempfile, copy
os.environ.setdefault("VEILGUARD_PII_DB_PATH", tempfile.mkdtemp(prefix="pii_aid_"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pii import get_redactor  # noqa: E402

MAGIC = "You are a Claude agent, built on Anthropic's Claude Agent SDK."
PREAMBLE = "VEILGUARD OPERATING CONTEXT. " + ("policy " * 300)
TOOLS = "TOOL SCHEMAS " + ("schema desc " * 400)

# Stable memory CONTENT per aid (the renderer frames around this; content
# itself is byte-stable for a given aid across turns + tier migration).
MEM = {
    0: "Alice Johnson emailed alice.johnson@acme.co.za; call her on 082 123 4567.",
    1: "Brian Naidoo (brian.naidoo@acme.co.za) joined finance.",
    2: "Priya Pillay said ID 8001015009087 is correct.",
    3: "Thabo Mokoena is at +27 31 555 0199.",
}

def build(turn):
    """System blocks for `turn`.  Immutable blocks tagged _vg_clean; each
    live memory block tagged _vg_aid.  cache_control is FLIPPED every turn
    (ttl + which block carries the breakpoint) to simulate marker churn."""
    ttl = "1h" if turn % 2 == 0 else "5m"          # marker churn
    breakpoint_aid = turn % 4                       # breakpoint moves each turn
    sysblocks = [
        {"type": "text", "text": MAGIC},                                  # CLEAN (fingerprint)
        {"type": "text", "text": PREAMBLE, "_vg_clean": True},            # CLEAN (immutable)
        {"type": "text", "text": TOOLS, "_vg_clean": True,                # CLEAN (tool schema)
         "cache_control": {"type": "ephemeral", "ttl": ttl}},
    ]
    # live memory blocks: aids 0..(3+turn). content stable per aid.
    for aid in range(4 + turn):
        content = MEM.get(aid) or f"Memory atom {aid}: contact person{aid}@acme.co.za."
        blk = {"type": "text", "text": content, "_vg_aid": aid, "_vg_stability": "byte_stable"}
        if aid == breakpoint_aid:                   # breakpoint hops between memory blocks
            blk["cache_control"] = {"type": "ephemeral", "ttl": ttl}
        sysblocks.append(blk)
    msgs = [{"role": "user", "content": f"Turn {turn}: email Nomsa Khumalo nomsa{turn}@acme.co.za."}]
    return sysblocks, msgs

r = get_redactor()
print("turn | sys_blocks | aid_hits aid_miss clean_skip | leaked | ms")
prev_hits = prev_miss = prev_clean = 0
for turn in range(6):
    sysb, msgs = build(turn)
    t0 = time.perf_counter()
    red_sys = r.redact_memory_blocks(copy.deepcopy(sysb), "pii-aidbench")
    red_msg = r.redact_messages(copy.deepcopy(msgs), "pii-aidbench")
    ms = (time.perf_counter() - t0) * 1000
    dh, dm, dc = r._aid_hits - prev_hits, r._aid_misses - prev_miss, r._clean_skipped - prev_clean
    prev_hits, prev_miss, prev_clean = r._aid_hits, r._aid_misses, r._clean_skipped
    # leak check: raw PII must not survive anywhere in system or messages
    import json as _j
    blob = _j.dumps(red_sys) + _j.dumps(red_msg)
    leaks = [s for s in ["alice.johnson@acme.co.za","082 123 4567","8001015009087",
                         "brian.naidoo@acme.co.za","Nomsa Khumalo"] if s in blob]
    # confirm immutable blocks untouched + meta stripped
    tools_clean = "REF_" not in red_sys[2]["text"] and "_vg_clean" not in red_sys[2]
    print(f"  {turn}  |    {len(sysb):2d}      |   +{dh}      +{dm}      +{dc}     | {leaks or 'none'} | {ms:6.1f}  "
          f"tools_untouched={tools_clean}")

print(f"\nTOTALS: aid_hits={r._aid_hits} aid_misses={r._aid_misses} "
      f"clean_skipped={r._clean_skipped} uncached={r._uncached_redactions}")
print("EXPECT: misses = 1/turn after turn0 (only the NEW aid); hits = carried aids; "
      "clean_skipped = 3/turn (magic+preamble+tools); cache-marker flips cause ZERO extra misses.")
