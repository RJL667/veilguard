"""Phase 7.5 — llm_verify executor + pairing rule.

Covers:
  * Happy path: pass / fail / error verdicts surface correctly.
  * Iron rule: llm_verify alone cannot satisfy create_task; needs a
    mechanical pair.
  * Cost guards: oversized rubric, oversized artifact rejected.
  * Sandbox: target_path escaping rejected.
  * Evidence shape: rubric_sha + artifact_sha + verdict + confidence.
  * Judge contract: missing judge → error; bad verdict → error.
"""

from __future__ import annotations

import hashlib
import json
import pytest


from app.acceptance.executors import (
    CHECK_KINDS,
    MECHANICAL_CHECK_KINDS,
    run_check,
    CheckResult,
)


# ── Constants ───────────────────────────────────────────────────────────


def test_llm_verify_in_check_kinds():
    assert "llm_verify" in CHECK_KINDS


def test_llm_verify_NOT_in_mechanical_set():
    """The iron rule: llm_verify is not mechanical.  Director's
    `create_task` validator rejects any task whose only required ACs
    are llm_verify (or manual_user)."""
    assert "llm_verify" not in MECHANICAL_CHECK_KINDS


# ── Happy path ──────────────────────────────────────────────────────────


def _judge_pass(prompt: str, model: str) -> dict:
    return {"verdict": "pass", "confidence": 0.92, "reason": "Looks good."}


def _judge_fail(prompt: str, model: str) -> dict:
    return {"verdict": "fail", "confidence": 0.78,
            "reason": "Missing required citation."}


def test_llm_verify_pass_with_inline_target():
    r = run_check(
        "llm_verify",
        {"rubric": "The memo cites a source.",
         "target_text": "Per the FAIS Act §3, the agent must..."},
        {"llm_judge": _judge_pass},
    )
    assert r.status == "pass"
    assert r.evidence["verdict"] == "pass"
    assert r.evidence["confidence"] == 0.92
    assert r.evidence["artifact_source"] == "inline"
    assert "rubric_sha" in r.evidence
    assert "artifact_sha" in r.evidence


def test_llm_verify_fail_blocks_gate():
    r = run_check(
        "llm_verify",
        {"rubric": "Cites a source.", "target_text": "vibes only"},
        {"llm_judge": _judge_fail},
    )
    assert r.status == "fail"
    assert "citation" in r.reason.lower()


# ── Errors ──────────────────────────────────────────────────────────────


def test_llm_verify_requires_rubric():
    r = run_check("llm_verify", {"target_text": "x"}, {"llm_judge": _judge_pass})
    assert r.status == "error"
    assert "rubric" in r.reason.lower()


def test_llm_verify_rubric_too_long_rejects():
    r = run_check(
        "llm_verify",
        {"rubric": "x" * 4001, "target_text": "y"},
        {"llm_judge": _judge_pass},
    )
    assert r.status == "error"
    assert "rubric too long" in r.reason


def test_llm_verify_requires_target():
    r = run_check(
        "llm_verify",
        {"rubric": "ok"},
        {"llm_judge": _judge_pass},
    )
    assert r.status == "error"
    assert "target_path" in r.reason or "target_text" in r.reason


def test_llm_verify_artifact_too_large_rejects():
    """Cost guard: 50KB cap unless allow_large=True."""
    r = run_check(
        "llm_verify",
        {"rubric": "ok", "target_text": "x" * 50_001},
        {"llm_judge": _judge_pass},
    )
    assert r.status == "error"
    assert "cap" in r.reason.lower() or "50" in r.reason


def test_llm_verify_artifact_large_with_override_runs():
    r = run_check(
        "llm_verify",
        {"rubric": "ok", "target_text": "x" * 50_001, "allow_large": True},
        {"llm_judge": _judge_pass},
    )
    assert r.status == "pass"


def test_llm_verify_missing_judge_in_context_errors():
    """If the caller forgot to inject `ctx['llm_judge']`, fail loudly
    rather than silently no-op."""
    r = run_check(
        "llm_verify",
        {"rubric": "ok", "target_text": "y"},
        context={},  # no llm_judge
    )
    assert r.status == "error"
    assert "llm_judge" in r.reason


def test_llm_verify_bad_judge_verdict_errors():
    """Judge returned something other than pass/fail."""
    def _bad_judge(p, m):
        return {"verdict": "maybe", "confidence": 0.5}

    r = run_check(
        "llm_verify",
        {"rubric": "ok", "target_text": "y"},
        {"llm_judge": _bad_judge},
    )
    assert r.status == "error"
    assert "verdict" in r.reason.lower()


def test_llm_verify_judge_raises_errors():
    """Judge call exploded — we report error, not crash the gate."""
    def _explosive_judge(p, m):
        raise RuntimeError("API died")

    r = run_check(
        "llm_verify",
        {"rubric": "ok", "target_text": "y"},
        {"llm_judge": _explosive_judge},
    )
    assert r.status == "error"
    assert "API died" in r.reason or "judge" in r.reason.lower()


def test_llm_verify_judge_returns_non_dict_errors():
    def _string_judge(p, m):
        return "pass"

    r = run_check(
        "llm_verify",
        {"rubric": "ok", "target_text": "y"},
        {"llm_judge": _string_judge},
    )
    assert r.status == "error"


# ── Sandbox ─────────────────────────────────────────────────────────────


def test_llm_verify_path_escape_rejected(tmp_path):
    """target_path that resolves outside workspace_root must error."""
    r = run_check(
        "llm_verify",
        {"rubric": "ok", "target_path": "../../etc/passwd"},
        {"llm_judge": _judge_pass, "workspace_root": str(tmp_path)},
    )
    assert r.status == "error"
    assert "sandbox" in r.reason.lower() or "outside" in r.reason.lower()


def test_llm_verify_target_path_missing_is_fail_not_error(tmp_path):
    """If the file doesn't exist, that's a `fail` (artifact missing),
    not an `error` (couldn't run check)."""
    r = run_check(
        "llm_verify",
        {"rubric": "ok", "target_path": "does_not_exist.md"},
        {"llm_judge": _judge_pass, "workspace_root": str(tmp_path)},
    )
    assert r.status == "fail"


def test_llm_verify_target_path_reads_file(tmp_path):
    f = tmp_path / "report.md"
    f.write_text("# Report\n\nGrounded in the cited FAIS source.\n", encoding="utf-8")
    r = run_check(
        "llm_verify",
        {"rubric": "Cites a source.", "target_path": "report.md"},
        {"llm_judge": _judge_pass, "workspace_root": str(tmp_path)},
    )
    assert r.status == "pass"
    assert r.evidence["artifact_source"] == "file:report.md"


# ── Evidence integrity ──────────────────────────────────────────────────


def test_llm_verify_evidence_carries_rubric_and_artifact_hashes():
    """Re-running the check with the same rubric + artifact should
    produce the same hashes — that's how the Critic detects builder
    partial-write / retry-different-content corruption (AC-26)."""
    r1 = run_check(
        "llm_verify",
        {"rubric": "exact rubric", "target_text": "exact artifact"},
        {"llm_judge": _judge_pass},
    )
    r2 = run_check(
        "llm_verify",
        {"rubric": "exact rubric", "target_text": "exact artifact"},
        {"llm_judge": _judge_pass},
    )
    assert r1.evidence["rubric_sha"]   == r2.evidence["rubric_sha"]
    assert r1.evidence["artifact_sha"] == r2.evidence["artifact_sha"]


def test_llm_verify_evidence_changes_when_artifact_changes():
    """Same rubric, different artifact → different artifact_sha."""
    r1 = run_check(
        "llm_verify",
        {"rubric": "r", "target_text": "alpha"},
        {"llm_judge": _judge_pass},
    )
    r2 = run_check(
        "llm_verify",
        {"rubric": "r", "target_text": "beta"},
        {"llm_judge": _judge_pass},
    )
    assert r1.evidence["artifact_sha"] != r2.evidence["artifact_sha"]
    assert r1.evidence["rubric_sha"]   == r2.evidence["rubric_sha"]


# ── Iron-rule integration via create_task ───────────────────────────────


def test_iron_rule_llm_verify_alone_rejected_by_create_task():
    """Direct integration test of the pairing rule: a task whose only
    required AC is llm_verify must be rejected by `create_task`."""
    from app.ledger import tasks as _t

    acs = [
        {
            "id":         "AC-1",
            "statement":  "Memo is grounded in cited source.",
            "check_kind": "llm_verify",
            "check_args": {"rubric": "x"},
            "required":   True,
        },
    ]
    with pytest.raises(ValueError, match="mechanical check_kind"):
        _t.create_task(
            tenant_id="t1", user_id="u1",
            owner_id="researcher",
            assigner_id="director",
            brief="test",
            deliverable_spec="test",
            acceptance_criteria=acs,
        )


def test_iron_rule_llm_verify_paired_with_mechanical_accepted():
    """When llm_verify is paired with a mechanical required AC, the
    task is allowed (still rejected here because LedgerStore isn't
    initialised in unit-test env — but the rejection should be from
    storage, not from the iron-rule validator)."""
    from app.ledger import tasks as _t

    acs = [
        {
            "id":         "AC-1",
            "statement":  "File exists.",
            "check_kind": "output_path_exists",
            "check_args": {"path": "out/report.md"},
            "required":   True,
        },
        {
            "id":         "AC-2",
            "statement":  "Memo is grounded.",
            "check_kind": "llm_verify",
            "check_args": {"rubric": "Grounded in citation"},
            "required":   True,
        },
    ]
    # We don't have a Lance store in this test env, so create_task will
    # raise — but the message should NOT mention the iron rule.
    try:
        _t.create_task(
            tenant_id="t1", user_id="u1",
            owner_id="researcher",
            assigner_id="director",
            brief="test",
            deliverable_spec="test",
            acceptance_criteria=acs,
        )
    except Exception as e:
        assert "mechanical check_kind" not in str(e), (
            "iron-rule validator wrongly rejected llm_verify + mechanical pair"
        )
