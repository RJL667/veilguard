"""Boundary skip when render pre-redacts (PII_BOUNDARY_FAILSAFE_2026-06-10).

flag OFF → redact_memory_blocks redacts system blocks.
flag ON  → ONLY blocks the render hook tagged `_vg_pii: "redacted"` pass
           through (evidence, not trust). Untagged blocks are STILL
           redacted at the boundary — env skew (flag on, hook off) can
           cost duplicate work but can never ship raw PII. The latest
           PROMPT is redacted regardless.
"""
import os, sys, tempfile
os.environ.setdefault("VEILGUARD_PII_DB_PATH", tempfile.mkdtemp(prefix="pii_skip_"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pii import get_redactor  # noqa: E402

ok = True
def check(n, c):
    global ok; print(("PASS" if c else "FAIL"), n); ok = ok and c

r = get_redactor()

raw_block = lambda: [{"type": "text", "text": "Contact Bob Jones bob@acme.com",
                      "_vg_aid": 9, "_vg_stability": "byte_stable",
                      "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
pre_redacted_block = lambda: [{"type": "text", "text": "Contact REF_PERSON_1 REF_EMAIL_1",
                               "_vg_aid": 9, "_vg_pii": "redacted",
                               "cache_control": {"type": "ephemeral", "ttl": "1h"}}]

# flag OFF (default): system memory gets redacted at the boundary.
r._redact_in_render = False
out = r.redact_memory_blocks(raw_block(), "pii-skip")
check("OFF: boundary redacts system memory", "bob@acme.com" not in out[0]["text"] and "REF_" in out[0]["text"])

# flag ON + _vg_pii tag: pre-redacted memory passes through untouched.
r._redact_in_render = True
pin = pre_redacted_block()
out = r.redact_memory_blocks(pin, "pii-skip")
check("ON+tag: pre-redacted text passed through verbatim", out[0]["text"] == "Contact REF_PERSON_1 REF_EMAIL_1")
check("ON+tag: _vg_* metadata stripped for wire", "_vg_aid" not in out[0] and "_vg_pii" not in out[0])
check("ON+tag: cache_control preserved", out[0].get("cache_control") == {"type": "ephemeral", "ttl": "1h"})

# flag ON, NO tag: the failsafe — raw memory is STILL redacted at the
# boundary even though the flag claims render-side redaction. This is the
# exact config skew (REDACT_IN_RENDER=1, RENDER_PII_HOOK unset) that
# leaked the raw live-memory tail to Anthropic before 2026-06-10.
out = r.redact_memory_blocks(
    [{"type": "text", "text": "Reach Dave Smith dave@corp.co.za",
      "_vg_aid": 11, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
    "pii-skip",
)
check("ON+untagged: boundary STILL redacts (failsafe)",
      "dave@corp.co.za" not in out[0]["text"] and "REF_" in out[0]["text"])

# flag ON: the latest PROMPT (messages) is STILL redacted (not gated).
msgs = [{"role": "user", "content": "Email Carol Wang carol@acme.com now"}]
rmsg = r.redact_messages(msgs, "pii-skip")
check("ON: latest prompt STILL redacted", "carol@acme.com" not in rmsg[0]["content"] and "REF_" in rmsg[0]["content"])

r._redact_in_render = False
print("\nRESULT:", "ALL PASS" if ok else "FAILURES")
sys.exit(0 if ok else 1)
