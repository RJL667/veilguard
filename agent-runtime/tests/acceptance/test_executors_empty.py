"""AC-7 — Empty-input safety: no false-positive on empty.

The most common gate-implementation bug: regex `.*` returns pass on an
empty file because `.*` matches empty string.  Or `claim_count(n>=0)`
trivially passes on empty list.  Or `output_path_exists(min_bytes=0)`
passes on a 0-byte file.

These tests deliberately construct empty/trivial inputs and assert the
executors return `fail`, not pass.
"""

from __future__ import annotations

from pathlib import Path

from app.acceptance import run_check


def test_no_false_pass_on_empty_file_with_regex_wildcard(tmp_path: Path):
    """Empty file + regex `.*` must NOT pass — regex would trivially match."""
    p = tmp_path / "empty.txt"
    p.write_text("")
    r = run_check(
        "output_path_matches_regex",
        {"path": str(p), "pattern": ".*", "flags": "s"},
        {"workspace_root": tmp_path},
    )
    assert r.status == "fail", (
        f"empty file + `.*` regex must fail (anti-false-positive), "
        f"got {r.status}: {r.reason}"
    )
    assert "empty" in r.reason.lower() or "whitespace" in r.reason.lower()


def test_no_false_pass_on_whitespace_only_file(tmp_path: Path):
    """File with only whitespace fails even when regex matches whitespace."""
    p = tmp_path / "whitespace.txt"
    p.write_text("   \n\t  \n")
    r = run_check(
        "output_path_matches_regex",
        {"path": str(p), "pattern": r"\s+"},
        {"workspace_root": tmp_path},
    )
    assert r.status == "fail", "whitespace-only file must fail"


def test_no_false_pass_on_zero_byte_file_with_default_min_bytes(tmp_path: Path):
    """`output_path_exists` with default min_bytes=1 fails on 0-byte file."""
    p = tmp_path / "stub.md"
    p.write_text("")
    r = run_check(
        "output_path_exists",
        {"path": str(p)},  # default min_bytes=1
        {"workspace_root": tmp_path},
    )
    assert r.status == "fail"
    assert "stub" in r.reason.lower() or "partial" in r.reason.lower() or "0 bytes" in r.reason or "0 byte" in r.reason


def test_no_false_pass_on_claim_count_with_empty_list():
    """`claim_count(>=1)` on empty list must fail, not pass."""
    r = run_check(
        "claim_count",
        {"predicate": {"category": "finding"}, "op": ">=", "n": 1},
        {"claims": []},
    )
    assert r.status == "fail"
    assert r.evidence["matched"] == 0


def test_no_false_pass_on_claim_predicate_with_empty_list():
    """`claim_predicate(must_exist=True)` on empty list must fail."""
    r = run_check(
        "claim_predicate",
        {"predicate": {"category": "decision"}, "must_exist": True},
        {"claims": []},
    )
    assert r.status == "fail"
    assert r.evidence["matched"] == 0


def test_claim_count_op_eq_zero_on_empty_does_pass():
    """Sanity: explicit `== 0` on empty list DOES pass (correct behaviour)."""
    r = run_check(
        "claim_count",
        {"predicate": {"category": "finding"}, "op": "==", "n": 0},
        {"claims": []},
    )
    assert r.status == "pass"  # this is the one case where empty list passes


def test_test_passes_empty_cmd_is_error(tmp_path: Path):
    """Empty `cmd` is `error` (can't run), not silently `pass`."""
    r = run_check(
        "test_passes",
        {"cmd": "", "cwd": str(tmp_path)},
        {"workspace_root": tmp_path},
    )
    assert r.status == "error"
