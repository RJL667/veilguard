"""AC-5 — Happy-path golden tests for every executor.

Each executor returns `pass` on a hand-crafted fixture that should pass.
Goal: catch "implementation never returns pass even when artifact is
correct" — the inverse of the "always returns pass" bug AC-6 catches.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.acceptance import (
    CHECK_KINDS,
    EXECUTOR_REGISTRY,
    MECHANICAL_CHECK_KINDS,
    run_check,
)


# ── AC-4 — Registry completeness ────────────────────────────────────────


def test_ac4_executor_registry_has_six_mechanical_plus_manual_and_llm_verify():
    """AC-4 (post-Phase 7.5): 6 mechanical kinds + manual_user + llm_verify.
    llm_verify is in CHECK_KINDS (it's executable) but NOT in
    MECHANICAL_CHECK_KINDS — the Director's "≥1 required mechanical AC"
    rule keeps the iron rule intact (llm_verify can lift quality but
    cannot be the sole gate)."""
    expected_kinds = {
        "claim_count", "claim_predicate",
        "output_path_exists", "output_path_matches_regex",
        "output_path_jsonschema", "test_passes",
        "manual_user", "llm_verify",
    }
    expected_mechanical = {
        "claim_count", "claim_predicate",
        "output_path_exists", "output_path_matches_regex",
        "output_path_jsonschema", "test_passes",
    }
    assert set(EXECUTOR_REGISTRY.keys()) == expected_kinds
    assert "llm_verify" in EXECUTOR_REGISTRY  # Phase 7.5
    assert CHECK_KINDS == frozenset(expected_kinds)
    assert MECHANICAL_CHECK_KINDS == frozenset(expected_mechanical)
    # Iron rule survives: llm_verify and manual_user cannot count
    # toward the mechanical-required AC requirement.
    assert "llm_verify" not in MECHANICAL_CHECK_KINDS
    assert "manual_user" not in MECHANICAL_CHECK_KINDS


# ── AC-5 — Happy path per executor ──────────────────────────────────────


def test_output_path_exists_happy(tmp_path: Path):
    """A non-empty file at the given path passes the check."""
    p = tmp_path / "deliverable.md"
    p.write_text("# A deliverable\n\nWith real content.\n")
    r = run_check(
        "output_path_exists",
        {"path": str(p), "min_bytes": 1},
        {"workspace_root": tmp_path},
    )
    assert r.status == "pass", f"expected pass, got {r.status}: {r.reason}"
    assert r.evidence["size_bytes"] > 0
    assert "path_sha256" in r.evidence  # AC-26 — evidence hash present


def test_output_path_matches_regex_happy(tmp_path: Path):
    p = tmp_path / "redactor.py"
    p.write_text("def redact_phone(s: str) -> str:\n    return s\n")
    r = run_check(
        "output_path_matches_regex",
        {"path": str(p), "pattern": r"^def\s+redact_phone\(s:\s*str\)\s*->\s*str:"},
        {"workspace_root": tmp_path},
    )
    assert r.status == "pass", f"expected pass, got {r.status}: {r.reason}"
    assert r.evidence["match_span"][0] == 0
    assert "body_sha256" in r.evidence


def test_output_path_jsonschema_happy(tmp_path: Path):
    pytest.importorskip("jsonschema")
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"name": "agent", "version": 1}))
    schema = {
        "type": "object",
        "required": ["name", "version"],
        "properties": {
            "name":    {"type": "string"},
            "version": {"type": "integer"},
        },
    }
    r = run_check(
        "output_path_jsonschema",
        {"path": str(p), "schema": schema},
        {"workspace_root": tmp_path},
    )
    assert r.status == "pass", f"expected pass, got {r.status}: {r.reason}"
    assert "body_sha256" in r.evidence


def test_test_passes_happy(tmp_path: Path):
    """`exit 0` command should pass."""
    r = run_check(
        "test_passes",
        {"cmd": "exit 0", "cwd": str(tmp_path)},
        {"workspace_root": tmp_path},
    )
    assert r.status == "pass", f"expected pass, got {r.status}: {r.reason}"
    assert r.evidence["exit_code"] == 0
    assert "stdout_hash" in r.evidence  # AC-26 — evidence hash


def test_claim_count_happy():
    """Three matching claims, op=='>=', n=2 → pass."""
    claims = [
        {"category": "finding", "body": "A"},
        {"category": "finding", "body": "B"},
        {"category": "finding", "body": "C"},
        {"category": "note",    "body": "D"},
    ]
    r = run_check(
        "claim_count",
        {"predicate": {"category": "finding"}, "op": ">=", "n": 2},
        {"claims": claims},
    )
    assert r.status == "pass", f"expected pass, got {r.status}: {r.reason}"
    assert r.evidence["matched"] == 3


def test_claim_predicate_happy():
    """At least one claim matches; must_exist=True → pass."""
    claims = [
        {"category": "decision", "body": "approved"},
        {"category": "finding",  "body": "X"},
    ]
    r = run_check(
        "claim_predicate",
        {"predicate": {"category": "decision"}, "must_exist": True},
        {"claims": claims},
    )
    assert r.status == "pass", f"expected pass, got {r.status}: {r.reason}"
    assert r.evidence["matched"] >= 1


def test_manual_user_never_passes():
    """manual_user always returns `error` (never auto-pass)."""
    r = run_check(
        "manual_user",
        {"question": "Approve this deploy?"},
        {},
    )
    # Per executor docs: manual_user returns 'error' (awaiting_user=True).
    # AC-25 / spec: never auto-passes.
    assert r.status == "error"
    assert r.evidence.get("awaiting_user") is True
