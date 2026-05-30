"""Tests for the create_proposal dedup behaviour added 2026-05-28.

Verifies:
  1. First emission of a (tenant, user, signal_type, node_ids) tuple
     inserts a new row.
  2. Re-emission of the same tuple while the row is still pending or
     deferred bumps `recurrence_count` and refreshes `last_surfaced_ts`
     instead of writing a duplicate.
  3. node_ids order doesn't affect dedup — [1,2,3] matches [3,2,1].
  4. Different node_ids on the same (tenant, user, signal_type) creates
     a NEW row.
  5. Different tenant — even with same node_ids — creates a NEW row
     (no cross-tenant collision).
  6. A row whose status moved past pending/deferred (e.g. shelved) does
     NOT block a fresh emission — the signal has come back after the
     user closed it out, so we want a new surface.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def fresh_ledger(monkeypatch):
    """Spin up an isolated LedgerStore against a tmp Lance dir.

    NB: `config.LEDGER_DB_PATH` is loaded at module-import time, so
    setting the env var alone won't redirect the store.  We patch the
    config module's attribute directly + reset the singleton.
    """
    tmp = Path(tempfile.mkdtemp(prefix="proposal_dedup_test_"))
    db_path = tmp / "tcmm.db"
    monkeypatch.setenv("LEDGER_DB_PATH", str(db_path))
    monkeypatch.setenv("VEILGUARD_AUDIT_DB_PATH", str(db_path))

    # Override the module-level constant the store reads at __init__.
    from app import config as _config
    monkeypatch.setattr(_config, "LEDGER_DB_PATH", db_path)

    from app.ledger import store as _store
    _store.LedgerStore._instance = None
    yield tmp
    _store.LedgerStore._instance = None


def _create(p, **overrides):
    """Convenience — minimal create_proposal call with defaults."""
    from app.ledger import proposals
    args = dict(
        tenant_id="t1", user_id="u1",
        signal_type="low_stability_cluster",
        signal_node_ids=[101, 102, 103],
        impact_score=10.0,
        proposed_brief="Refresh cluster",
        proposed_assignee="researcher",
    )
    args.update(overrides)
    return proposals.create_proposal(**args)


def test_first_emission_inserts(fresh_ledger):
    pid = _create(None)
    assert pid.startswith("prop-"), pid


def test_same_node_ids_bumps_recurrence(fresh_ledger):
    pid1 = _create(None)
    pid2 = _create(None)
    assert pid2 == pid1, "duplicate emission should return existing pid"

    from app.ledger.store import LedgerStore
    tbl = LedgerStore.get().table("task_proposals")
    arr = tbl.search().where(f"id = '{pid1}'").limit(1).to_arrow()
    assert arr.num_rows == 1
    rc = arr.column("recurrence_count")[0].as_py()
    assert rc == 2, f"recurrence_count should be 2 after one dedup, got {rc}"


def test_node_id_order_does_not_block_dedup(fresh_ledger):
    pid1 = _create(None, signal_node_ids=[1, 2, 3])
    pid2 = _create(None, signal_node_ids=[3, 2, 1])
    assert pid2 == pid1, "node_id ORDER must not matter for dedup"


def test_different_node_ids_inserts_new(fresh_ledger):
    pid1 = _create(None, signal_node_ids=[1, 2, 3])
    pid2 = _create(None, signal_node_ids=[4, 5, 6])
    assert pid2 != pid1, "different node_ids must NOT dedup"


def test_different_tenant_no_cross_collision(fresh_ledger):
    pid1 = _create(None, tenant_id="tenant-A", user_id="tenant-A")
    pid2 = _create(None, tenant_id="tenant-B", user_id="tenant-B")
    assert pid2 != pid1, (
        "same node_ids under DIFFERENT tenants must not collide — that "
        "would leak proposals across tenant boundaries"
    )


def test_decay_score_floor_is_max_of_old_and_new(fresh_ledger):
    """A re-emission with HIGHER impact_score should raise decay_score
    back up (the signal is fresher / stronger).  A LOWER score should
    not pull it down (we want re-surfacing, not de-prioritisation).
    """
    pid1 = _create(None, impact_score=10.0)
    _create(None, impact_score=15.0)
    _create(None, impact_score=5.0)
    from app.ledger.store import LedgerStore
    tbl = LedgerStore.get().table("task_proposals")
    arr = tbl.search().where(f"id = '{pid1}'").limit(1).to_arrow()
    decay = arr.column("decay_score")[0].as_py()
    assert decay == 15.0, f"decay_score should stay at max-seen 15.0, got {decay}"


def test_shelved_does_not_block_fresh_emission(fresh_ledger):
    """Once the user shelves a proposal, a future emission on the same
    cluster MUST surface a new row (otherwise the shelving permanently
    silences a recurring signal — bad).
    """
    pid1 = _create(None)
    from app.ledger.store import LedgerStore
    tbl = LedgerStore.get().table("task_proposals")
    import time as _t
    tbl.update(
        where=f"id = '{pid1}'",
        values={"status": "shelved", "updated_ts": _t.time()},
    )
    pid2 = _create(None)
    assert pid2 != pid1, (
        "shelved row must NOT block fresh emissions; got same pid"
    )
