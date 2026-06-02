"""F10 — `/apr/resume` + `/apr/status` endpoints + sticky-breaker recovery.

Phase 6.7's APR circuit breaker is **sticky**: once tripped,
`apr_should_pause_dispatch()` short-circuits on `is_tripped()` and never
auto-recovers (even after the rolling window empties). `apr_resume()` is the
ONLY unblock. Before 2026-06-02 it had **no wired HTTP route** — `apr.py`
docstrings referenced `/apr/resume` + `/apr/status` but `main.py` never
registered them, so a tripped breaker wedged inbox-poller dispatch until a
process restart (UAT finding F10, surfaced live: a burst of high-token Director
queries tripped it and the only recovery was `docker restart`). These tests pin
the fix + document the sticky behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.runtime_health.apr import (
    GLOBAL_APR,
    apr_resume,
    apr_snapshot,
    apr_should_pause_dispatch,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(autouse=True)
def _reset_apr():
    """GLOBAL_APR is a module singleton shared across the suite — clear it
    before and after so these tests neither inherit nor leak breaker state."""
    GLOBAL_APR.clear()
    yield
    GLOBAL_APR.clear()


def test_apr_endpoints_registered_in_main():
    """Source-grep: both routes + handlers must stay wired (the F10 gap was
    that they were referenced in docstrings but never registered)."""
    src = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert '@app.get("/apr/status")' in src
    assert "async def apr_status" in src
    assert '@app.post("/apr/resume")' in src
    assert "async def apr_resume_endpoint" in src


def test_breaker_is_sticky_until_resume():
    """Root cause of F10: a tripped breaker stays tripped even with an empty
    window — `breaker_should_fire()` would be False on <5 samples, but
    `is_tripped()` wins, so dispatch stays paused until an explicit resume."""
    GLOBAL_APR.trip()
    assert GLOBAL_APR.is_tripped() is True
    assert apr_should_pause_dispatch() is True  # sticky, not data-driven here
    snap = apr_snapshot()
    assert snap["state"] == "tripped"
    assert snap["tripped_at"] is not None


def test_apr_resume_clears_breaker_and_dispatch_resumes():
    """The fix's contract: apr_resume() clears the tripped state + window so
    `apr_should_pause_dispatch()` returns False and the poller resumes."""
    GLOBAL_APR.trip()
    assert apr_should_pause_dispatch() is True
    apr_resume()
    assert GLOBAL_APR.is_tripped() is False
    assert apr_should_pause_dispatch() is False
    snap = apr_snapshot()
    assert snap["state"] == "normal"
    assert snap["tripped_at"] is None
    assert snap["n_samples"] == 0
