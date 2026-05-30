"""Stale test — `app.policy.client_tool_policy` was removed 2026-05-25.

Per `app/runtime.py:57-61`:
  > The local approval_gate_hook + client_tool_policy modules were the
  > SDK-PreToolUse path.  All tool dispatch now flows through
  > tool_dispatcher → sub-agents (Path 2 HTTP), and sub-agents runs the
  > canonical approval gate (services/sub-agents/core/approval.py).

The canonical approval-gate tests now live in
`services/sub-agents/tests/test_approval.py` and exercise the matrix
this file used to.  If a §3.8 capability-matrix regression needs a
test in agent-runtime, write it against `tool_dispatcher` (which
proxies to sub-agents) — NOT against the removed local policy module.
"""

import pytest


def test_legacy_policy_module_intentionally_removed():
    """Documentation test: the import path this file used to depend on
    is gone and should NOT come back.  If you're re-adding policy logic
    to agent-runtime, talk to ops first — the architectural decision
    was to centralise approval in sub-agents."""
    with pytest.raises(ModuleNotFoundError):
        import app.policy.client_tool_policy  # noqa: F401
