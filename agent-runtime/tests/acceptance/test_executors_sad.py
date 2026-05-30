"""AC-6 — Sad-path golden tests for every executor.

Each executor returns `fail` (or `error`) with a non-empty `reason` on a
fixture that should NOT pass.  Prevents "always returns pass" — the
worst-class bug for a verification system.

Three-state semantics per AC-27:
  fail  = check ran cleanly, artifact wrong
  error = check itself couldn't execute (cmd not found, regex compile, ...)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.acceptance import run_check


def test_output_path_exists_missing_file(tmp_path: Path):
    r = run_check(
        "output_path_exists",
        {"path": str(tmp_path / "does_not_exist.md")},
        {"workspace_root": tmp_path},
    )
    assert r.status == "fail"
    assert r.evidence["exists"] is False
    assert r.reason, "fail must carry a non-empty reason"


def test_output_path_matches_regex_no_match(tmp_path: Path):
    p = tmp_path / "report.md"
    p.write_text("# Report\n\nNo conclusion section here.\n")
    r = run_check(
        "output_path_matches_regex",
        {"path": str(p), "pattern": r"^##\s+Conclusion"},
        {"workspace_root": tmp_path},
    )
    assert r.status == "fail"
    assert "not found" in r.reason.lower()


def test_output_path_jsonschema_validation_fail(tmp_path: Path):
    pytest.importorskip("jsonschema")
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"name": "agent"}))  # missing required `version`
    schema = {
        "type": "object",
        "required": ["name", "version"],
    }
    r = run_check(
        "output_path_jsonschema",
        {"path": str(p), "schema": schema},
        {"workspace_root": tmp_path},
    )
    assert r.status == "fail"
    assert "validation" in r.reason.lower()


def test_output_path_jsonschema_parse_error_is_error_not_fail(tmp_path: Path):
    """AC-27: parse failure is `error`, not `fail`."""
    pytest.importorskip("jsonschema")
    p = tmp_path / "broken.json"
    p.write_text("{not json,")
    r = run_check(
        "output_path_jsonschema",
        {"path": str(p), "schema": {"type": "object"}},
        {"workspace_root": tmp_path},
    )
    assert r.status == "error", (
        "JSON parse error must be `error` (check unable to execute), not `fail` "
        "(per AC-27 three-state semantics)"
    )


def test_test_passes_wrong_exit_code(tmp_path: Path):
    r = run_check(
        "test_passes",
        {"cmd": "exit 7", "cwd": str(tmp_path)},
        {"workspace_root": tmp_path},
    )
    assert r.status == "fail"
    assert r.evidence["exit_code"] == 7
    assert "exit 7" in r.reason or "expected 0" in r.reason


def test_test_passes_command_not_found_is_error(tmp_path: Path):
    """AC-27: exit 127 = command not found = `error`, not `fail`."""
    r = run_check(
        "test_passes",
        {"cmd": "this_command_definitely_does_not_exist_xyz_42", "cwd": str(tmp_path)},
        {"workspace_root": tmp_path},
    )
    # Either FileNotFoundError → 'error', or shell returns 127 → 'error'.
    assert r.status == "error", (
        f"command-not-found must be `error`, got {r.status}: {r.reason}"
    )


def test_test_passes_timeout_is_error(tmp_path: Path):
    """Timeout = `error`, not `fail` (check couldn't complete)."""
    import sys
    if sys.platform == "win32":
        cmd = "ping -n 60 127.0.0.1 > nul"
    else:
        cmd = "sleep 60"
    r = run_check(
        "test_passes",
        {"cmd": cmd, "cwd": str(tmp_path), "timeout_s": 1},
        {"workspace_root": tmp_path},
    )
    assert r.status == "error"
    assert "timed out" in r.reason.lower()


def test_claim_count_too_few():
    """Want >=3, only 1 matches → fail."""
    claims = [
        {"category": "finding", "body": "A"},
        {"category": "note",    "body": "B"},
    ]
    r = run_check(
        "claim_count",
        {"predicate": {"category": "finding"}, "op": ">=", "n": 3},
        {"claims": claims},
    )
    assert r.status == "fail"
    assert r.evidence["matched"] == 1


def test_claim_predicate_no_match():
    """No matches, must_exist=True → fail."""
    claims = [{"category": "finding", "body": "X"}]
    r = run_check(
        "claim_predicate",
        {"predicate": {"category": "decision"}, "must_exist": True},
        {"claims": claims},
    )
    assert r.status == "fail"
    assert r.evidence["matched"] == 0


def test_unknown_check_kind_is_error():
    """AC-27: unknown kind = `error`, not raise."""
    r = run_check("not_a_real_kind", {}, {})
    assert r.status == "error"
    assert "unknown check_kind" in r.reason
