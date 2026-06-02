"""F13 — proactive `/proposals/convert` must synthesize a default AC.

The convert endpoint used to call `create_task` with NO `acceptance_criteria`,
so the Phase 6.0.2 contract rejected EVERY proposal approval with a 400 — the
whole proactive "approve → Task" payoff was dead (surfaced live in the Tier-1.3
UAT, 2026-06-02). The fix routes both the convert path and the Director's MCP
`create_task` tool through one shared `synthesize_default_acceptance_criteria`
helper so they can't drift again.
"""

from __future__ import annotations

import time

import pytest

from app.ledger import tasks as T

TEN, USR = "t-f13", "u-f13"


@pytest.fixture
def store(tmp_path):
    from app.ledger.store import LedgerStore
    LedgerStore._instance = None
    inst = LedgerStore.get(db_path=str(tmp_path / "ledger.db"))
    inst.table("agent_tasks")
    yield inst
    LedgerStore._instance = None


# ── helper unit tests ────────────────────────────────────────────────────


def test_synthesize_worker_extracts_path_from_spec():
    acs = T.synthesize_default_acceptance_criteria(
        "researcher", "Write the findings to team/drafts/x.md please"
    )
    assert len(acs) == 1
    ac = acs[0]
    assert ac["check_kind"] == "output_path_exists"
    assert ac["required"] is True
    assert ac["check_args"]["path"] == "team/drafts/x.md"
    assert ac["check_args"]["min_bytes"] == 1


def test_synthesize_worker_defaults_when_no_path():
    acs = T.synthesize_default_acceptance_criteria("builder", "do some analysis")
    assert acs[0]["check_args"]["path"] == "deliverable.md"


def test_synthesize_coordinator_returns_empty():
    # director / team-lead coordinate fan-outs — zero ACs, close via autoclose.
    assert T.synthesize_default_acceptance_criteria("director", "x.md") == []
    assert T.synthesize_default_acceptance_criteria("team-lead", "x.md") == []


# ── the F13 contract: synthesized AC satisfies create_task WITHOUT exempt ──


def test_synthesized_ac_satisfies_phase_6_0_2(store):
    """The whole point of F13: a proposal's assignee + spec, run through the
    helper, must let create_task succeed with NO legacy-exempt — exactly what
    the /proposals/convert endpoint now does."""
    assignee, spec = "researcher", "memo at team/drafts/m.md"
    acs = T.synthesize_default_acceptance_criteria(assignee, spec)
    tid = T.create_task(
        tenant_id=TEN, user_id=USR, owner_id=assignee,
        brief="proactive task", deliverable_spec=spec,
        acceptance_criteria=acs,  # NOT _phase_6_legacy_exempt
    )
    assert tid  # did not raise → the convert path can create the task


def test_every_proactive_spec_yields_concrete_ac_path():
    """[PROACTIVE_AC_PATH] Every signal-type deliverable spec must name a real,
    extractable path so the synthesized AC doesn't fall back to 'deliverable.md'
    (a file the IC never writes → the approved proactive task dies at the Critic).
    Regression for the F13 follow-up surfaced live 2026-06-02."""
    from app.proposals.briefs import DELIVERABLE_SPECS, render_deliverable_spec
    for sig in DELIVERABLE_SPECS:
        spec = render_deliverable_spec(sig)
        ac = T.synthesize_default_acceptance_criteria("researcher", spec)[0]
        path = ac["check_args"]["path"]
        assert path != "deliverable.md", (
            f"{sig}: deliverable_spec names no concrete path → AC falls back to "
            f"deliverable.md, which the IC won't write"
        )
        assert path.startswith("team/drafts/"), (
            f"{sig}: synthesized AC path {path!r} should be under team/drafts/"
        )
