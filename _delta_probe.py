"""Does redaction process only the DELTA of TCMM memory blocks?

Instruments analyzer.analyze() to count how many lines actually hit spaCy
per turn, under the two ways TCMM memory grows:

  A. PER-FRAGMENT blocks (TCMM's real working-tier model: each turn appends
     a NEW `[Memory index=N]` block; prior blocks are byte-identical).
  B. ONE GROWING blob (a single tier block that grows in place each turn).
"""
import os, sys, tempfile
os.environ.setdefault("VEILGUARD_PII_DB_PATH", tempfile.mkdtemp(prefix="pii_delta_"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pii import get_redactor  # noqa: E402

r = get_redactor()

# Instrument analyze(): record #lines per call.
_orig = r.analyzer.analyze
calls = []
def _wrap(*a, **k):
    txt = k.get("text", a[0] if a else "")
    calls.append(txt.count("\n") + 1 if txt else 0)
    return _orig(*a, **k)
r.analyzer.analyze = _wrap

def frag(n):
    return (f"[Memory index={n} | role=USER | src=live]\n"
            f"Person_{n} Smith emailed person{n}@acme.co.za about item {n}.")

print("A. PER-FRAGMENT blocks (TCMM working-tier model):")
for turn in range(5):
    calls.clear()
    blocks = [{"type": "text", "text": frag(n)} for n in range(turn + 1)]
    r.redact_render_blocks(blocks, "pii-fragmodel")
    print(f"  turn {turn}: blocks={turn+1}  block_miss_this_turn={1 if turn==0 else len(calls)>=0 and sum(1 for _ in calls)} "
          f"analyze_calls={len(calls)}  lines_analyzed={sum(calls)}")

print("\nB. ONE GROWING blob (single tier block grows in place):")
r2 = get_redactor()  # same singleton
for turn in range(5):
    calls.clear()
    blob = "\n".join(frag(n) for n in range(turn + 1))   # whole live tier as ONE block
    r2.redact_render_blocks([{"type": "text", "text": blob}], "pii-blobmodel")
    print(f"  turn {turn}: block_lines={blob.count(chr(10))+1}  "
          f"analyze_calls={len(calls)}  lines_analyzed={sum(calls)}  (only NEW lines should hit spaCy)")
