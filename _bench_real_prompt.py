"""Benchmark redaction on a REAL captured Anthropic request body.

Source: anthropic_dump_payloads_full.json (a real proxied turn).
Shows per-block classification (CLEAN-skip vs scanned), confirms the
`tools` field is never touched, and times cold vs warm redaction.
"""
import os, sys, json, time, tempfile
os.environ.setdefault("VEILGUARD_PII_DB_PATH", tempfile.mkdtemp(prefix="pii_bench_"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pii import get_redactor  # noqa: E402

_DUMP = os.environ.get("DUMP_PATH", r"C:\Users\rudol\.veilguard\anthropic_dump_payloads_full.json")
req = json.load(open(_DUMP, encoding="utf-8"))[0]["request"]
sid = "pii-realbench"
r = get_redactor()

sysb = req.get("system") or []
sys_chars = sum(len(b.get("text", "")) for b in sysb if isinstance(b, dict))
tools = req.get("tools") or []
tools_chars = len(json.dumps(tools))
print(f"REAL prompt: {len(sysb)} system blocks ({sys_chars} chars), "
      f"{len(tools)} tools ({tools_chars} chars), {len(req.get('messages', []))} messages\n")

# Per-block classification (what redact_render_blocks decides).
print("per system-block decision:")
for i, b in enumerate(sysb):
    t = b.get("text", "") if isinstance(b, dict) else ""
    cls = r._classify(b) if isinstance(b, dict) and b.get("type") == "text" else "non-text"
    clean = isinstance(b, dict) and (b.get("_skip_pii") is True or r._classify(b) == "CLEAN")
    print(f"  [{i}] {len(t):6d}c  {'CLEAN-SKIP' if clean else 'SCANNED   '}  {t[:55]!r}")

# Cold + warm timing on the FULL body (proxy's exact call).
import copy
t0 = time.perf_counter(); red1 = r.redact_json(copy.deepcopy(req), sid); cold = (time.perf_counter()-t0)*1000
t0 = time.perf_counter(); red2 = r.redact_json(copy.deepcopy(req), sid); warm = (time.perf_counter()-t0)*1000

# Was the tools field touched?
tools_untouched = json.dumps(red1.get("tools")) == json.dumps(req.get("tools"))
# Byte-stable across turns?
stable = json.dumps(red1.get("system")) == json.dumps(red2.get("system"))

print(f"\ncold redact (turn 1): {cold:.1f} ms")
print(f"warm redact (turn 2): {warm:.1f} ms   block_cache hits={r._block_hits} miss={r._block_misses} clean_skipped={r._clean_skipped}")
print(f"tools field UNTOUCHED (never redacted): {tools_untouched}")
print(f"redacted system byte-stable across turns: {stable}")
# Show whether any REF tokens were minted (memory had PII?)
ref_in_sys = sum(b.get('text','').count('REF_') for b in red1['system'] if isinstance(b,dict))
print(f"REF tokens minted in system memory: {ref_in_sys}")
