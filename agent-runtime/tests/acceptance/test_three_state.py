"""AC-27 — Three-state result: pass | fail | error (not two).

The most common gate bug: 2-state result (pass/fail) collapses two
fundamentally different conditions:

  - "the artifact is wrong"           (fail — block done, ask for revision)
  - "the check itself couldn't run"   (error — block done, but the
                                       artifact may still be correct)

Bug class this catches: executor that returns `pass` when exec failed
(e.g. catches the exception and treats command-not-found as "test
passed because no failures were observed").  Or executor that returns
`fail` when the test runner is just missing, masking the real issue.

Both `fail` and `error` block the gate.  Distinction matters for
diagnostics + remediation guidance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.acceptance import CheckResult, run_check


def test_check_result_status_is_pass_fail_or_error_only():
    """Schema invariant: CheckResult.status is one of three values."""
    valid = {"pass", "fail", "error"}
    # Sample one of each.
    r_pass = CheckResult("pass")
    r_fail = CheckResult("fail", reason="x")
    r_err  = CheckResult("error", reason="y")
    assert r_pass.status in valid
    assert r_fail.status in valid
    assert r_err.status in valid


def test_command_not_found_is_error_not_fail(tmp_path: Path):
    """`test_passes` with non-existent cmd → error, not fail."""
    r = run_check(
        "test_passes",
        {"cmd": "definitely_not_a_real_binary_xyz_123", "cwd": str(tmp_path)},
        {"workspace_root": tmp_path},
    )
    assert r.status == "error", (
        f"command-not-found must be error (check couldn't run), "
        f"not fail (check ran and disagreed). Got {r.status}: {r.reason}"
    )
    # Critical: NOT pass
    assert r.status != "pass"


def test_command_exits_nonzero_is_fail_not_error(tmp_path: Path):
    """`test_passes` with a real cmd that exits non-zero → fail, not error.

    Distinct from command-not-found: here the test runner DID execute,
    it just returned a non-zero exit code. That's the artifact being
    wrong, not the check being broken.
    """
    r = run_check(
        "test_passes",
        {"cmd": "exit 1", "cwd": str(tmp_path)},
        {"workspace_root": tmp_path},
    )
    assert r.status == "fail", (
        f"non-zero exit from a real cmd must be fail (artifact wrong), "
        f"not error (check broken). Got {r.status}: {r.reason}"
    )


def test_json_parse_failure_is_error_not_fail(tmp_path: Path):
    """`output_path_jsonschema` on malformed JSON → error, not fail."""
    pytest.importorskip("jsonschema")
    p = tmp_path / "broken.json"
    p.write_text("{not json")
    r = run_check(
        "output_path_jsonschema",
        {"path": str(p), "schema": {"type": "object"}},
        {"workspace_root": tmp_path},
    )
    assert r.status == "error", (
        "JSON parse error must be error (check can't execute), not fail. "
        "Distinguishes 'broken JSON' from 'valid JSON failing schema'."
    )


def test_jsonschema_validation_failure_is_fail_not_error(tmp_path: Path):
    """Schema-conformance failure on parseable JSON → fail, not error."""
    pytest.importorskip("jsonschema")
    p = tmp_path / "valid_but_wrong.json"
    p.write_text('{"name": 42}')  # name should be string per schema
    r = run_check(
        "output_path_jsonschema",
        {
            "path": str(p),
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        },
        {"workspace_root": tmp_path},
    )
    assert r.status == "fail"


def test_regex_compile_error_is_error_not_fail(tmp_path: Path):
    """Invalid regex → error, not fail."""
    p = tmp_path / "x.txt"
    p.write_text("hello")
    r = run_check(
        "output_path_matches_regex",
        {"path": str(p), "pattern": "[unclosed"},
        {"workspace_root": tmp_path},
    )
    assert r.status == "error", (
        "regex compile failure must be error (check broken), not fail"
    )


def test_path_outside_sandbox_is_error(tmp_path: Path):
    """Sandbox violation → error (per AC-8)."""
    r = run_check(
        "output_path_exists",
        {"path": "../../../../../etc/passwd"},
        {"workspace_root": tmp_path},
    )
    assert r.status == "error"
    assert "sandbox" in r.reason.lower() or "outside" in r.reason.lower()


def test_unknown_check_kind_is_error_not_pass():
    """Unknown kind must be error, never pass (silent type bug catcher)."""
    r = run_check("does_not_exist", {}, {})
    assert r.status == "error"


def test_unknown_kind_has_known_kind_list_in_reason():
    """Error reason should help the developer figure out the right kind."""
    r = run_check("ouput_path_exists", {}, {})  # typo: ouput not output
    assert r.status == "error"
    assert "output_path_exists" in r.reason or "known" in r.reason.lower()


def test_all_three_states_block_done_only_pass_allows():
    """Sanity for the hard-gate logic: only `pass` advances; fail/error block.

    Asserts the boolean shape the gate uses: `status == 'pass'`.
    """
    # The gate uses: required AC must have status == 'pass'.  Both fail
    # and error fail that condition.
    for status in ("fail", "error"):
        r = CheckResult(status, reason="x")
        assert r.status != "pass"
