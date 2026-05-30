"""AC-26 — Evidence hash present on every executor result.

The critic re-runs the same check on review and compares evidence to
detect partial-write / retry-with-different-content corruption.
Without evidence hashes, this attack succeeds:

  Builder writes report.md with content X → AC passes →
  Builder crashes mid-write → retry writes report.md with content Y →
  AC passes again →  but content Y is NOT what the first AC saw.

With evidence hashes, the critic stores `path_sha256(X)` on the first
pass; on review re-check it computes `path_sha256(Y)`, sees mismatch,
returns changes_requested.
"""

from __future__ import annotations

from pathlib import Path

from app.acceptance import run_check


def test_output_path_exists_carries_path_sha256(tmp_path: Path):
    p = tmp_path / "deliverable.md"
    p.write_text("# Real content\n")
    r = run_check(
        "output_path_exists",
        {"path": str(p)},
        {"workspace_root": tmp_path},
    )
    assert r.status == "pass"
    # 64-hex-char SHA256
    assert "path_sha256" in r.evidence
    assert len(r.evidence["path_sha256"]) == 64
    assert all(c in "0123456789abcdef" for c in r.evidence["path_sha256"])


def test_path_sha256_changes_when_content_changes(tmp_path: Path):
    """The hash discriminates content X from content Y."""
    p = tmp_path / "report.md"
    p.write_text("Content X")
    r1 = run_check(
        "output_path_exists",
        {"path": str(p)},
        {"workspace_root": tmp_path},
    )
    p.write_text("Content Y")
    r2 = run_check(
        "output_path_exists",
        {"path": str(p)},
        {"workspace_root": tmp_path},
    )
    assert r1.status == "pass" and r2.status == "pass"
    assert r1.evidence["path_sha256"] != r2.evidence["path_sha256"], (
        "evidence hash must differ when file content differs — "
        "this is the AC-26 anti-corruption invariant"
    )


def test_output_path_matches_regex_carries_body_sha256(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_text("hello world")
    r = run_check(
        "output_path_matches_regex",
        {"path": str(p), "pattern": "hello"},
        {"workspace_root": tmp_path},
    )
    assert r.status == "pass"
    assert "body_sha256" in r.evidence
    assert len(r.evidence["body_sha256"]) == 64


def test_test_passes_carries_stdout_hash(tmp_path: Path):
    r = run_check(
        "test_passes",
        {"cmd": "echo hello", "cwd": str(tmp_path)},
        {"workspace_root": tmp_path},
    )
    assert r.status == "pass"
    assert "stdout_hash" in r.evidence
    assert len(r.evidence["stdout_hash"]) == 64
    # exit_code is part of the evidence
    assert r.evidence["exit_code"] == 0


def test_claim_count_carries_predicate_match_count():
    """claim_count uses match-count as its evidence."""
    r = run_check(
        "claim_count",
        {"predicate": {"category": "x"}, "op": ">=", "n": 1},
        {"claims": [{"category": "x"}, {"category": "x"}]},
    )
    assert r.status == "pass"
    assert r.evidence.get("predicate_match_count") == 2
