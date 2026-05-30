"""Phase 8.0 — Snapshot anchor primitive tests.

Covers:
  * Identity: same inputs → same anchor_hash.
  * Sensitivity: changing any one substrate flips the hash.
  * Version bump: SNAPSHOT_ANCHOR_VERSION participates in the hash.
  * decision_point participates in the hash (same substrate, different
    decision point = different anchor).
  * Validators: missing required fields raise ValueError.
  * to_dict round-trip preserves fields.
"""

from __future__ import annotations

import pytest

from app.replay import (
    SnapshotAnchor,
    compute_anchor,
    SNAPSHOT_ANCHOR_VERSION,
)


def _kwargs(**overrides) -> dict:
    base = dict(
        decision_point="director.route",
        agent_id="director",
        tenant_id="t1",
        user_id="u1",
        task_graph={"task-1": {"status": "open", "owner": "researcher"}},
        approvals=[{"id": "appr-1", "decision": "allow"}],
        tool_outputs={"task-1": {"web_fetch": "...content..."}},
        constitution_version=4,
        tcmm_snapshot_ref="archive[aid<=120]",
        alignment_weights_ver=2,
        dream_graph_signals_ref="dream_cycle_id=cyc-abc",
        ts=1000.0,
    )
    base.update(overrides)
    return base


# ── Identity ────────────────────────────────────────────────────────────


def test_anchor_hash_identical_inputs_match():
    a1 = compute_anchor(**_kwargs())
    a2 = compute_anchor(**_kwargs())
    assert a1.anchor_hash == a2.anchor_hash


def test_anchor_hash_independent_of_ts():
    """Timestamp must NOT participate in anchor_hash — only the
    substrate state does.  Otherwise two anchors computed milliseconds
    apart would never match."""
    a1 = compute_anchor(**_kwargs(ts=1000.0))
    a2 = compute_anchor(**_kwargs(ts=2000.0))
    assert a1.anchor_hash == a2.anchor_hash


def test_anchor_hash_independent_of_extras():
    """Extras are diagnostic-only — they must NOT affect anchor_hash."""
    a1 = compute_anchor(**_kwargs(extras={"note": "first"}))
    a2 = compute_anchor(**_kwargs(extras={"note": "second"}))
    assert a1.anchor_hash == a2.anchor_hash


# ── Sensitivity (each input must flip the hash) ─────────────────────────


def test_anchor_hash_changes_with_task_graph():
    a1 = compute_anchor(**_kwargs())
    a2 = compute_anchor(**_kwargs(
        task_graph={"task-1": {"status": "in_progress"}},  # changed
    ))
    assert a1.anchor_hash != a2.anchor_hash


def test_anchor_hash_changes_with_approvals():
    a1 = compute_anchor(**_kwargs())
    a2 = compute_anchor(**_kwargs(
        approvals=[{"id": "appr-2", "decision": "deny"}],
    ))
    assert a1.anchor_hash != a2.anchor_hash


def test_anchor_hash_changes_with_tool_outputs():
    a1 = compute_anchor(**_kwargs())
    a2 = compute_anchor(**_kwargs(
        tool_outputs={"task-1": {"web_fetch": "DIFFERENT"}},
    ))
    assert a1.anchor_hash != a2.anchor_hash


def test_anchor_hash_changes_with_constitution_version():
    a1 = compute_anchor(**_kwargs(constitution_version=4))
    a2 = compute_anchor(**_kwargs(constitution_version=5))
    assert a1.anchor_hash != a2.anchor_hash


def test_anchor_hash_changes_with_tcmm_snapshot_ref():
    a1 = compute_anchor(**_kwargs(tcmm_snapshot_ref="archive[aid<=120]"))
    a2 = compute_anchor(**_kwargs(tcmm_snapshot_ref="archive[aid<=125]"))
    assert a1.anchor_hash != a2.anchor_hash


def test_anchor_hash_changes_with_alignment_weights_ver():
    a1 = compute_anchor(**_kwargs(alignment_weights_ver=2))
    a2 = compute_anchor(**_kwargs(alignment_weights_ver=3))
    assert a1.anchor_hash != a2.anchor_hash


def test_anchor_hash_changes_with_dream_graph_signals_ref():
    a1 = compute_anchor(**_kwargs(dream_graph_signals_ref="cyc-A"))
    a2 = compute_anchor(**_kwargs(dream_graph_signals_ref="cyc-B"))
    assert a1.anchor_hash != a2.anchor_hash


def test_anchor_hash_changes_with_decision_point():
    """Same substrate, different decision point = different anchor.
    Critical: a single substrate state can be the input to many
    decisions (route, synthesize, propose); they must not collide."""
    a1 = compute_anchor(**_kwargs(decision_point="director.route"))
    a2 = compute_anchor(**_kwargs(decision_point="director.synthesize"))
    assert a1.anchor_hash != a2.anchor_hash


# ── Schema integrity ────────────────────────────────────────────────────


def test_anchor_version_stamped():
    a = compute_anchor(**_kwargs())
    assert a.version == SNAPSHOT_ANCHOR_VERSION


def test_anchor_required_fields_present():
    a = compute_anchor(**_kwargs())
    for fld in (
        "anchor_id", "version", "decision_point", "agent_id",
        "tenant_id", "user_id", "ts",
        "task_graph_hash", "approvals_hash", "tool_outputs_hash",
        "constitution_version", "tcmm_snapshot_ref",
        "alignment_weights_ver", "dream_graph_signals_ref",
        "anchor_hash",
    ):
        assert hasattr(a, fld), f"missing field {fld}"


def test_anchor_to_dict_roundtrip():
    a = compute_anchor(**_kwargs())
    d = a.to_dict()
    assert d["anchor_id"] == a.anchor_id
    assert d["anchor_hash"] == a.anchor_hash
    assert d["task_graph_hash"] == a.task_graph_hash


# ── Validators ──────────────────────────────────────────────────────────


def test_compute_anchor_rejects_empty_decision_point():
    with pytest.raises(ValueError, match="decision_point"):
        compute_anchor(**_kwargs(decision_point=""))


def test_compute_anchor_rejects_missing_tenant():
    with pytest.raises(ValueError):
        compute_anchor(**_kwargs(tenant_id=""))


def test_compute_anchor_rejects_missing_agent():
    with pytest.raises(ValueError):
        compute_anchor(**_kwargs(agent_id=""))
