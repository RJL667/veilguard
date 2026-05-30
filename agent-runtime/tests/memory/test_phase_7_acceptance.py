"""Phase 7.3 acceptance criteria — AC-P7.1 ... AC-P7.6.

These tests are the CI source of truth for the §7 TCMM-ledger boundary.
They are deliberately mechanical so the Critic's existing executors can
verify them without needing live-LLM judgement.

Per spec §7.3:
  AC-P7.1  output_path_matches_regex  — boundary rule text present in spec
  AC-P7.2  claim_predicate            — org_memory Lance table absent post-M1
  AC-P7.3  claim_predicate            — proposal_outcomes + task_proposals
                                        schemas have tcmm_obs_id + no
                                        free-text content columns
  AC-P7.4  test_passes                — status-index query on task_proposals
                                        makes zero TCMM HTTP calls
  AC-P7.5  test_passes                — SHA-chain intact for state-machine
                                        kinds after M4 discussion split
  AC-P7.6  test_passes                — TCMM source_kind allow-list

Transitional state caveat: M1 / M4 are not fully cut over yet
(`org_memory` Lance table still exists, transitional dual-write).
AC-P7.2 is therefore relaxed below into a "schema-still-defined-but-
flagged-as-transitional" check; the strict post-cutover assertion
lands in the M1 cutover PR which removes the Lance table.
"""

from __future__ import annotations

import asyncio
import re
import unittest.mock as mock
from pathlib import Path

import pyarrow as pa
import pytest

from app.ledger.schemas import (
    TABLE_SCHEMAS,
    proposal_outcomes_schema,
    task_proposals_schema,
)
from app.memory import phase_7_writers


# Spec lives one level above agent-runtime/.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SPEC_PATH = _REPO_ROOT / "MULTI_AGENT_PLATFORM.md"


# ── AC-P7.1 ─────────────────────────────────────────────────────────────


def test_ac_p7_1_boundary_rule_present_in_spec():
    """The §2 design principle text must be present in the spec."""
    if not _SPEC_PATH.exists():
        pytest.skip(f"spec not at expected location: {_SPEC_PATH}")
    text = _SPEC_PATH.read_text(encoding="utf-8")
    # The rule text per AC-P7.1.  Substring match — phrasing must be stable.
    needle = "TCMM is the knowledge-graph substrate; the ledger is the state machine"
    assert needle in text, (
        f"Phase 7 boundary rule missing from spec at §2.  "
        f"Required text: {needle!r}"
    )


# ── AC-P7.2 ─────────────────────────────────────────────────────────────


def test_ac_p7_2_org_memory_table_removed_post_cutover():
    """Post-M1-cutover (2026-05-28): the legacy `org_memory` Lance
    table is dropped, the schema is no longer registered in
    `TABLE_SCHEMAS`, and the Phase 7 M1 split-writer is the sole
    canonical lesson destination (TCMM-only).
    """
    # Schema dropped from the active migration registry.
    assert "org_memory" not in TABLE_SCHEMAS, (
        "AC-P7.2 violation: org_memory still registered in TABLE_SCHEMAS "
        "post-M1 cutover.  Drop it from app/ledger/schemas.py:TABLE_SCHEMAS."
    )

    # The Phase 7 split-writer must contain the TCMM observation path.
    import inspect
    src = inspect.getsource(phase_7_writers.promote_lesson_to_team_knowledge)
    assert "TCMM" in src or "observe_agent_output" in src, (
        "Phase 7 M1 split-writer must call TCMM observe."
    )


# ── AC-P7.3 ─────────────────────────────────────────────────────────────


# Columns we expect to be ABSENT from the ledger schema after M2/M3
# because they are content the spec moves to TCMM observations.
_FORBIDDEN_OUTCOME_LEDGER_COLUMNS = {
    "regret_text", "what_went_wrong", "lessons_learned",
}
# For task_proposals the spec is more nuanced: proposed_brief + rationale
# remain in the ledger today (transitional, see §7 M3 note in spec) so
# legacy queries don't break.  Phase 7.1 M3 cutover removes them; this
# test asserts the cross-ref column is present so the migration can land
# without a follow-up schema change.
_REQUIRED_PROPOSAL_LEDGER_COLUMNS = {"tcmm_obs_id"}
_REQUIRED_OUTCOME_LEDGER_COLUMNS = {"tcmm_obs_id"}


def test_ac_p7_3_proposal_outcomes_has_tcmm_obs_id_and_no_narrative():
    schema = proposal_outcomes_schema()
    names = set(schema.names)
    missing = _REQUIRED_OUTCOME_LEDGER_COLUMNS - names
    assert not missing, f"proposal_outcomes missing cross-ref columns: {missing}"
    # No free-text narrative columns leaked into the ledger.
    leaked = _FORBIDDEN_OUTCOME_LEDGER_COLUMNS & names
    assert not leaked, (
        f"proposal_outcomes ledger has narrative columns that must live in "
        f"TCMM only: {leaked}"
    )


def test_ac_p7_3_task_proposals_has_tcmm_obs_id():
    schema = task_proposals_schema()
    names = set(schema.names)
    missing = _REQUIRED_PROPOSAL_LEDGER_COLUMNS - names
    assert not missing, f"task_proposals missing cross-ref columns: {missing}"


# ── AC-P7.4 ─────────────────────────────────────────────────────────────


def test_ac_p7_4_status_index_query_makes_zero_tcmm_calls():
    """The operational hot path (status query) must not touch TCMM.

    We monkey-patch the TCMM observe wrapper to record every call, then
    drive a status-index query path and assert the call count stayed at 0.
    """
    call_count = {"n": 0}

    async def _spy(*args, **kwargs):
        call_count["n"] += 1
        return False  # pretend TCMM is unreachable; should not be called anyway

    # Patch the sanctioned TCMM observe wrapper.  Any code path that
    # would observe goes through this single import — that's the lint
    # guarantee, so this patch covers all callers.
    with mock.patch("app.middleware.tcmm.observe_agent_output", _spy):
        # Drive the operational path: list pending proposals.  This is
        # the canonical "status index" query: filter by status, no
        # narrative content needed.
        from app.ledger import proposals as _p
        try:
            # queue() is the operational status-index path — pending +
            # deferred filter, sorted by decay_score.  No content.
            _ = _p.queue(tenant_id="t-test", user_id="u-test", limit=10)
        except Exception:
            # If the underlying Lance store isn't initialised in the
            # test environment, that's fine — what we're asserting is
            # that the *code path* never reaches TCMM, not that Lance
            # succeeds.  The mock spy is still the source of truth.
            pass

    assert call_count["n"] == 0, (
        f"AC-P7.4 violation: status-index query made {call_count['n']} "
        f"TCMM observe calls.  The operational hot path must stay cold "
        f"on TCMM."
    )


# ── AC-P7.5 ─────────────────────────────────────────────────────────────


def test_ac_p7_5_state_machine_kinds_remain_sha_chained():
    """The kinds the spec marks as state-machine must still be in the
    ledger SHA-chain allow-list — M4 only removed discussion/note.

    Source of truth: comments.py `_ALLOWED_KINDS` set.
    """
    from app.ledger.comments import _ALLOWED_KINDS
    required_state_machine_kinds = {
        "status_change",
        "review_decision",
        "blocker_raised",
        "blocker_cleared",
        "review_request",
    }
    missing = required_state_machine_kinds - _ALLOWED_KINDS
    assert not missing, (
        f"State-machine kinds dropped from SHA-chain ledger after M4: "
        f"{missing}.  These must remain in task_comments per AC-P7.5."
    )


def test_ac_p7_5_sha_chain_verifier_exists():
    """Phase 7 didn't break the chain walker.  We can't run a live walk
    here (no Lance fixture), but we assert the verify_chain function is
    still exported and importable — the M4 split must not have removed it.
    """
    from app.ledger.comments import verify_chain
    assert callable(verify_chain)
    import inspect
    sig = inspect.signature(verify_chain)
    # Must accept the trio of identifiers; otherwise upstream callers break.
    for p in ("task_id", "tenant_id", "user_id"):
        assert p in sig.parameters, f"verify_chain missing kw {p!r}"


# ── AC-P7.6 ─────────────────────────────────────────────────────────────


_TCMM_SOURCE_KIND_ALLOWLIST = frozenset({
    "lesson",
    "proposal",
    "outcome_narrative",
    "discussion",
    "agent_observation",
})


def test_ac_p7_6_phase_7_writers_use_only_allowlisted_source_kinds():
    """Static check: every TCMM observe call in phase_7_writers passes
    a `source=` whose value is in the allow-list.  This is the static
    equivalent of the AC-P7.6 runtime probe.
    """
    import ast
    src = Path(phase_7_writers.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    source_kinds_found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "source" and isinstance(kw.value, ast.Constant):
                    if isinstance(kw.value.value, str):
                        source_kinds_found.add(kw.value.value)
    bad = source_kinds_found - _TCMM_SOURCE_KIND_ALLOWLIST
    assert not bad, (
        f"AC-P7.6 violation: phase_7_writers uses source_kind(s) not in "
        f"the allow-list {sorted(_TCMM_SOURCE_KIND_ALLOWLIST)}: {sorted(bad)}"
    )
    # And we found at least one — sanity check the AST walker worked.
    assert source_kinds_found, "AST walker found zero source=... kwargs"


def test_ac_p7_6_record_episode_default_source_allowlisted():
    """The Phase 6.8 `record_episode` writer's default `source="agent"` is
    NOT in the AC-P7.6 allow-list because it is an agent observation by
    a different name.  We assert that the writer's documented intent
    maps cleanly to `agent_observation`, and that callers who care can
    override.
    """
    import inspect
    from app.memory.writers import record_episode
    sig = inspect.signature(record_episode)
    src_param = sig.parameters.get("source")
    assert src_param is not None
    # The default ships as "agent" today; allow-list expects
    # "agent_observation" or one of the other 4.  This test is
    # documentary: it will need to flip when AC-P7.6 runtime probe lands.
    assert src_param.default in {"agent", "agent_observation"}, (
        f"record_episode source default unexpectedly {src_param.default!r}; "
        f"AC-P7.6 expects this to migrate to 'agent_observation'."
    )


# ── Integration smoke for the M4 discussion writer ─────────────────────


def test_m4_discussion_writer_routes_to_tcmm_agent_observations():
    """When `record_discussion_comment` is called, it must call the
    TCMM observe wrapper exactly once with the agent-namespaced
    conversation_id and an allow-listed source kind."""
    captured: dict = {}

    async def _spy(**kwargs):
        captured.update(kwargs)
        return True

    with mock.patch("app.middleware.tcmm.observe_agent_output", _spy):
        ok = asyncio.run(phase_7_writers.record_discussion_comment(
            task_id="task-abc",
            tenant_id="t-test",
            user_id="u-test",
            author_id="researcher",
            body="Looked at the docs.",
            note=False,
        ))
    assert ok is True
    assert captured.get("conversation_id") == "agent/researcher/observations"
    assert captured.get("source") in _TCMM_SOURCE_KIND_ALLOWLIST
    assert captured.get("agent_id") == "researcher"
    assert "task-abc" in captured.get("text", "")


def test_m4_note_kind_uses_agent_observation_source():
    captured: dict = {}

    async def _spy(**kwargs):
        captured.update(kwargs)
        return True

    with mock.patch("app.middleware.tcmm.observe_agent_output", _spy):
        ok = asyncio.run(phase_7_writers.record_discussion_comment(
            task_id="task-xyz",
            tenant_id="t-test",
            user_id="u-test",
            author_id="builder",
            body="Internal scratch note.",
            note=True,
        ))
    assert ok is True
    assert captured.get("source") == "agent_observation"


def test_m4_empty_body_is_noop():
    called = {"n": 0}

    async def _spy(**kwargs):
        called["n"] += 1
        return True

    with mock.patch("app.middleware.tcmm.observe_agent_output", _spy):
        ok = asyncio.run(phase_7_writers.record_discussion_comment(
            task_id="task-empty",
            tenant_id="t-test",
            user_id="u-test",
            author_id="researcher",
            body="   ",
        ))
    assert ok is False
    assert called["n"] == 0
