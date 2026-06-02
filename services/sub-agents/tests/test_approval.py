"""Canonical approval-gate + policy tests.

Covers two things added 2026-05-31:
  1. core/policy.py — the re-homed client-tool decision matrix
     (ALLOW / APPROVE / DENY), restoring the policy brain that was
     removed from agent-runtime 2026-05-25 and never replaced.
  2. core/approval.py fail-closed backstop — if the policy module ever
     fails to load, the gate must force an approval toast (never silently
     run a client tool), even at permission_level=auto.

Self-contained script-style harness (matches test_all.py).  Run:
    python tests/test_approval.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.approval as ap          # noqa: E402
from core.policy import classify, Decision  # noqa: E402

passed = 0
failed = 0


def test(name, condition, msg=""):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} — {msg}")
        failed += 1


# ── Group A: the policy matrix (classify, direct) ────────────────────────
print("=== A: policy matrix (classify) ===")
test("read_file -> ALLOW",
     classify(tool_name="read_file", tool_args={"path": "notes.txt"}).decision is Decision.ALLOW)
test("read_file .env -> APPROVE (no silent secret read)",
     classify(tool_name="read_file", tool_args={"path": "proj/.env"}).decision is Decision.APPROVE)
test("write_file -> APPROVE",
     classify(tool_name="write_file", tool_args={"path": "out.txt"}).decision is Decision.APPROVE)
test("write_file .ssh/id_rsa -> DENY",
     classify(tool_name="write_file", tool_args={"path": "home/.ssh/id_rsa"}).decision is Decision.DENY)
test("run_command 'ls' -> APPROVE",
     classify(tool_name="run_command", tool_args={"command": "ls -la"}).decision is Decision.APPROVE)
test("run_command 'rm -rf /' -> DENY",
     classify(tool_name="run_command", tool_args={"command": "rm -rf /"}).decision is Decision.DENY)
test("grep -> ALLOW",
     classify(tool_name="grep", tool_args={"pattern": "TODO"}).decision is Decision.ALLOW)
test("cloud tool tcmm_recall -> ALLOW",
     classify(tool_name="tcmm_recall", tool_args={}).decision is Decision.ALLOW)


# ── Group B: gate() integration with the real matrix loaded ──────────────
print("=== B: gate() with policy matrix loaded ===")
test("_POLICY_OK is True (local policy module loaded)", ap._POLICY_OK is True,
     f"_POLICY_OK={ap._POLICY_OK} — core.policy failed to import")


class _StubBridge:
    connected = True
    user_id = "stub"

    def __init__(self):
        self.asked = False

    async def request_approval(self, **kw):
        self.asked = True
        return {"approved": True, "reason": "stub user approved"}


class _StubSettings:
    def __init__(self, level):
        self._lvl = level

    def get_effective(self, user_id, conv_id):
        return (self._lvl, 2)


def _patch_settings(level):
    ap.ClientSettingsStore.get = staticmethod(lambda: _StubSettings(level))


def _gate(**kw):
    return asyncio.run(ap.gate(**kw))


# Dangerous command -> DENY is final, regardless of level (no toast).
_patch_settings("auto")
b = _StubBridge()
r = _gate(tool="run_command", args={"command": "shutdown /s"},
          user_id="u:1", conv_id="b1", bridge=b)
test("dangerous run_command -> blocked (deny, no toast)",
     r.proceed is False and r.decision == "deny" and b.asked is False,
     f"proceed={r.proceed} decision={r.decision} asked={b.asked}")

# read_file + auto -> ALLOW runs silently.
b = _StubBridge()
r = _gate(tool="read_file", args={"path": "notes.txt"},
          user_id="u:1", conv_id="b2", bridge=b)
test("read_file + auto runs silently (allow)",
     r.proceed is True and b.asked is False,
     f"proceed={r.proceed} asked={b.asked}")

# write_file + confirm -> APPROVE pops a toast.
_patch_settings("confirm")
b = _StubBridge()
r = _gate(tool="write_file", args={"path": "out.txt", "content": "x"},
          user_id="u:1", conv_id="b3", bridge=b)
test("write_file + confirm asks (approve->toast)",
     b.asked is True and r.proceed is True,
     f"asked={b.asked} proceed={r.proceed}")

# sandbox identity short-circuits to allow (multi-agent flow intact).
b = _StubBridge()
r = _gate(tool="run_command", args={"command": "rm -rf /"},
          user_id="system:sandbox", conv_id="b4", bridge=b)
test("sandbox identity short-circuits to allow",
     r.proceed is True and b.asked is False,
     f"proceed={r.proceed} asked={b.asked}")


# ── Group C: fail-closed backstop (simulate policy module missing) ───────
print("=== C: fail-closed when policy unavailable ===")
_saved_ok, _saved_fn = ap._POLICY_OK, ap._classify_policy
ap._POLICY_OK = False
ap._classify_policy = None
try:
    _patch_settings("auto")
    b = _StubBridge()
    r = _gate(tool="run_command", args={"command": "ls"},
              user_id="u:1", conv_id="c1", bridge=b)
    test("policy-down + auto forces approval (no silent run)",
         b.asked is True,
         f"asked={b.asked} proceed={r.proceed}")

    r = _gate(tool="run_command", args={"command": "ls"},
              user_id="u:1", conv_id="c2", bridge=None)
    test("policy-down + auto + no bridge denies (fail closed)",
         r.proceed is False,
         f"proceed={r.proceed} decision={r.decision}")
finally:
    ap._POLICY_OK, ap._classify_policy = _saved_ok, _saved_fn

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
