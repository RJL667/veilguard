"""Preamble core/delta split — [PREAMBLE_CORE_DELTA_2026-06-01].

Two invariants:
  1. render_preamble(with_orchestration=True) is byte-IDENTICAL to the
     historical template — the Director/IC live path keeps its exact
     cache slot (no invalidation).
  2. render_preamble(with_orchestration=False) is the lean universal core:
     the spawn/sub-agent/long-running-task paragraphs are gone, but every
     universal section (identity, style, answer contract, POPIA, etc.)
     and the generic "don't call tools when chatting" style rule remain.

Self-contained; run: python agent/tests/test_preamble.py
"""

import importlib.util
import os
import sys

# Load agent/preamble.py DIRECTLY (it has no heavy deps) rather than via
# `import agent.preamble`, which would trigger agent/__init__.py → pii →
# presidio → torch (heavy / unavailable in a bare test shell).
_PREAMBLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "preamble.py"
)
_spec = importlib.util.spec_from_file_location("vg_preamble_under_test", _PREAMBLE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
render_preamble = _mod.render_preamble
_VEILGUARD_PREAMBLE_TEMPLATE = _mod._VEILGUARD_PREAMBLE_TEMPLATE

passed = 0
failed = 0


def test(name, cond, msg=""):
    global passed, failed
    if cond:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} — {msg}")
        failed += 1


full = render_preamble(with_orchestration=True)
lean = render_preamble(with_orchestration=False)

# 1. Full variant unchanged (cache stability for the live path).
test("full == historical template (byte-identical)",
     full == _VEILGUARD_PREAMBLE_TEMPLATE)
test("default (no kwarg) == full",
     render_preamble() == _VEILGUARD_PREAMBLE_TEMPLATE)

# 2. Lean strips the orchestration-only paragraphs.
for phrase in (
    "Lineage: sub-agent conversations you spawn",
    "When you spawn sub-agents (via spawn_agent",
    "Long-running tasks (5-10 minutes) submitted via start_task",
    "wait_for_tasks",
):
    test(f"lean drops orchestration phrase: {phrase[:42]!r}",
         phrase not in lean, "phrase still present in lean preamble")

# 3. Lean keeps every universal section + the generic style rule.
for phrase in (
    "IDENTITY & TRUST MODEL",
    "STYLE RULES",
    "tcmm_record_turn",
    "POPIA & DATA PROTECTION",
    "RECALL FAILURE MODES",
    "CITATIONS & EVIDENCE HYGIENE",
    "FINAL OPERATIONAL CHECKLIST",
    # §2 style rule that *mentions* spawn_agent as an example must STAY —
    # it's a generic "don't call tools when just chatting" rule.
    "Do NOT call tools",
    "End of preamble. Memory context follows below.",
):
    test(f"lean keeps core phrase: {phrase[:42]!r}",
         phrase in lean, "core phrase missing from lean preamble")

# 4. Lean is strictly shorter.
test("lean is shorter than full",
     len(lean) < len(full),
     f"lean={len(lean)} full={len(full)}")

print(f"\nfull={len(full)} chars  lean={len(lean)} chars  "
      f"saved={len(full) - len(lean)} chars")
print(f"{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
