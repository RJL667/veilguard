"""F14 + Phase 6.5 — `truncate_with_marker` + the runtime tool-result cap.

F14 (surfaced live 2026-06-02): `runtime.py` capped EVERY tool result to 1200
chars when rebuilding the next-turn context (the LOOP_CONTEXT_FIX action log).
So a critic re-reading a >1.2 KB artifact across turns saw it as
"truncated/corrupted" and rejected valid work — the systemic reason multi-agent
tasks died at the Critic gate. The cap is now deliverable-sized (≥12 KB,
env-tunable) and uses the explicit Phase-6.5 marker instead of a silent slice.
"""

from __future__ import annotations

from app.runtime_health.truncation import (
    truncate_with_marker,
    has_truncation_marker,
)


def test_body_under_cap_is_unchanged():
    body = "x" * 500
    out = truncate_with_marker(body, max_bytes=12000)
    assert out == body
    assert not has_truncation_marker(out)


def test_body_over_cap_truncated_and_marked():
    body = "y" * 20000
    out = truncate_with_marker(body, max_bytes=12000)
    assert len(out.encode("utf-8")) <= 12000 + 120  # + marker overhead
    assert has_truncation_marker(out)
    assert "of 20000 bytes" in out  # original size reported


def test_realistic_artifact_not_truncated_at_runtime_default():
    """F14 regression: a typical ~3.8 KB markdown note (the size that read as
    'truncated at line 18' under the old 1200 cap) must pass through whole at
    the runtime default cap."""
    from app.runtime import _TOOL_RESULT_LOG_CAP
    assert _TOOL_RESULT_LOG_CAP >= 12000, "default cap regressed below 12KB"
    note = "# Research Note\n\n" + ("A finding with an inline citation. " * 130)
    assert len(note) > 1200  # would have been truncated under the old cap
    out = truncate_with_marker(note, max_bytes=_TOOL_RESULT_LOG_CAP)
    assert out == note
    assert not has_truncation_marker(out)
