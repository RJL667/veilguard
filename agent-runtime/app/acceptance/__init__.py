"""Phase 6.0 — Acceptance criteria executor framework.

Public surface:
  - CHECK_KINDS               — the 7 mechanical kinds + manual_user
  - MECHANICAL_CHECK_KINDS    — subset Director must include ≥1 of
  - EXECUTOR_REGISTRY         — kind → executor mapping
  - run_check(...)            — dispatch one AC, return CheckResult
  - CheckResult               — {status: 'pass'|'fail'|'error', evidence, reason}

See `agent-runtime/tests/PHASE_6_ACCEPTANCE.md` AC-4 through AC-8,
AC-26, AC-27 for the gating criteria.
"""

from .executors import (
    CHECK_KINDS,
    MECHANICAL_CHECK_KINDS,
    EXECUTOR_REGISTRY,
    CheckResult,
    run_check,
)

__all__ = [
    "CHECK_KINDS",
    "MECHANICAL_CHECK_KINDS",
    "EXECUTOR_REGISTRY",
    "CheckResult",
    "run_check",
]
