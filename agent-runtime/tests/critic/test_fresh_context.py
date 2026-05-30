"""Phase 6.1 — Fresh-context critic tests.

AC-21 — captured critic prompt contains spec/AC/artifact, MUST NOT
        contain producer trajectory tokens.
AC-22 — structural source grep: forbidden field names do NOT appear in
        the critic prompt builder OR the dispatch path under any flag.
AC-23 — negative fixture: deliberately wire a leak and assert the
        leak-detector (AC-22) catches it.  Tests the test.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.acceptance.critic_prompt import (
    FORBIDDEN_PRODUCER_FIELDS,
    build_critic_user_message,
)


# ── AC-21 — prompt content ──────────────────────────────────────────────


def test_ac21_critic_prompt_contains_spec_ac_artifact():
    prompt = build_critic_user_message(
        task_id="task-abc",
        brief="Implement redact_phone",
        deliverable_spec="≤500 words + tests, claim-only deliverable.",
        acceptance_criteria=[
            {
                "id": "AC-1",
                "statement": "report.md exists",
                "check_kind": "output_path_exists",
                "check_args": '{"path": "report.md"}',
                "required": True,
            },
            {
                "id": "AC-2",
                "statement": "tests pass",
                "check_kind": "test_passes",
                "check_args": '{"cmd": "pytest -q"}',
                "required": True,
            },
        ],
        outputs=["workspace/task-abc/report.md"],
        inputs=["workspace/upstream/spec.md"],
    )
    # Task identity present
    assert "task-abc" in prompt
    assert "Implement redact_phone" in prompt
    assert "≤500 words" in prompt
    # ACs inlined
    assert "AC-1" in prompt
    assert "AC-2" in prompt
    assert "output_path_exists" in prompt
    assert "test_passes" in prompt
    # Artifact paths present
    assert "workspace/task-abc/report.md" in prompt
    # Critic given the gate-discipline rule
    assert "review_decision" in prompt
    assert "changes_requested" in prompt
    assert "do not give benefit of the doubt" in prompt.lower() or "every required ac" in prompt.lower()


def test_ac21_critic_prompt_contains_no_producer_trajectory():
    """The prompt must not include any forbidden producer-trajectory tokens."""
    prompt = build_critic_user_message(
        task_id="task-abc",
        brief="Real brief",
        deliverable_spec="Real spec",
        acceptance_criteria=[{
            "id": "AC-1", "statement": "x", "check_kind": "output_path_exists",
            "check_args": "{}", "required": True,
        }],
        outputs=["a.md"],
        inputs=None,
    )
    lower = prompt.lower()
    for forbidden in FORBIDDEN_PRODUCER_FIELDS:
        assert forbidden not in lower, (
            f"forbidden token {forbidden!r} appeared in critic prompt — "
            "producer-trajectory leak"
        )
    # Common leak signatures from prior implementations
    leak_phrases = [
        "ic's submit_for_review",
        "ic narrative",
        "producer comments",
        "agent's reasoning",
        "read task comments (get_task) for full context",
    ]
    for phrase in leak_phrases:
        assert phrase not in lower, (
            f"leaky phrase {phrase!r} found in critic prompt"
        )


# ── AC-22 — structural source grep ──────────────────────────────────────


CRITIC_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "app" / "acceptance" / "critic_prompt.py"
)
INBOX_POLLER_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "app" / "workers" / "inbox_poller.py"
)


def test_ac22_critic_prompt_module_has_no_forbidden_references():
    """The critic prompt builder module must not reference forbidden fields
    in code paths that build the prompt.

    Note: FORBIDDEN_PRODUCER_FIELDS itself contains these strings as data
    (the enum of names to reject). The check excludes that constant
    definition: we grep AFTER the FORBIDDEN_PRODUCER_FIELDS definition.
    """
    src = CRITIC_PROMPT_PATH.read_text(encoding="utf-8")
    # Drop everything inside the FORBIDDEN_PRODUCER_FIELDS frozenset literal
    src_after_const = re.sub(
        r"FORBIDDEN_PRODUCER_FIELDS:\s*frozenset\[str\]\s*=\s*frozenset\(\{[^}]*\}\)",
        "FORBIDDEN_PRODUCER_FIELDS_REDACTED",
        src,
        flags=re.DOTALL,
    )
    for forbidden in FORBIDDEN_PRODUCER_FIELDS:
        # forbid the name appearing in code (not the constant data)
        assert forbidden not in src_after_const, (
            f"{forbidden!r} appears in critic_prompt.py outside the "
            "FORBIDDEN_PRODUCER_FIELDS allow-list — possible leak"
        )


def test_ac22_inbox_poller_critic_branch_has_no_producer_message_inlining():
    """The critic dispatch branch in inbox_poller.py must not inline
    producer narrative.  Specifically: it must NOT tell the critic to
    "read task comments (get_task) for full context including the IC's
    submit_for_review" — that's the old pre-Phase-6.1 leak."""
    src = INBOX_POLLER_PATH.read_text(encoding="utf-8")
    leaky_phrases = [
        "Read task comments (get_task) for full context",
        "including the IC's submit_for_review",
    ]
    for phrase in leaky_phrases:
        assert phrase not in src, (
            f"inbox_poller.py still has the pre-Phase-6.1 leak phrase "
            f"{phrase!r} — Phase 6.1 build_critic_user_message not wired"
        )


# ── AC-23 — negative fixture / tests the test ──────────────────────────


def test_ac23_negative_fixture_catches_chain_of_thought_leak():
    """Deliberately construct a prompt with a leak. Assert the
    AC-21 invariant test would catch it.

    This is the meta-test: if AC-21 only checks for the *absence* of
    producer fields but the actual leak is a different phrasing, AC-21
    can pass while the system is broken. AC-23 fixes that by deliberately
    injecting a leak and confirming the detection mechanism fires.
    """
    leaky_prompt = (
        "You have been assigned review of Task task-x.\n\n"
        "Brief: X\n\n"
        "Producer's chain_of_thought:\n"
        "  - Step 1: I tried A\n"
        "  - Step 2: I tried B and that worked\n"
        # ↑ this should trip the FORBIDDEN_PRODUCER_FIELDS detector
    )
    # Re-run the same invariant the prompt builder is supposed to satisfy.
    lower = leaky_prompt.lower()
    leaks_found = [f for f in FORBIDDEN_PRODUCER_FIELDS if f in lower]
    assert leaks_found, (
        "negative fixture should have triggered the leak detector — "
        "either the leaky_prompt content drifted or FORBIDDEN_PRODUCER_FIELDS "
        "is missing the relevant entries"
    )
    assert "chain_of_thought" in leaks_found


def test_ac23_negative_fixture_via_actual_invariant():
    """Cross-check: assert that if we ran the same assertion AC-21 uses
    against the leaky prompt, AssertionError fires."""
    leaky_prompt = "Here is the producer_messages dump from the IC: ..."
    with pytest.raises(AssertionError):
        lower = leaky_prompt.lower()
        for forbidden in FORBIDDEN_PRODUCER_FIELDS:
            assert forbidden not in lower, (
                f"forbidden token {forbidden!r} appeared in critic prompt"
            )
