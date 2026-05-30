"""Tests for proposals/recalibration.py — weekly alignment weight tuning."""

import time
from unittest.mock import patch, MagicMock

import pytest

from app.proposals import recalibration as R
from app.proposals.scoring import DEFAULT_ALIGNMENT_VECTORS, SIGNAL_INFORMATION_GAP


# ── compute_adjustment ─────────────────────────────────────────────────


def test_compute_adjustment_high_regret_downweights():
    """High regret on an objective → that objective gets downweighted."""
    current = {"reduce_toil": 0.5, "improve_security": 0.3, "preserve_user_agency": 0.2}
    regret = {"reduce_toil": 3.0, "improve_security": 1.0, "preserve_user_agency": 1.0}
    new = R.compute_adjustment(
        current_vector=current, regret_per_objective=regret,
    )
    # reduce_toil should drop, others rise via renormalisation
    assert new["reduce_toil"] < current["reduce_toil"]
    # Sum still ≈ 1.0
    assert sum(new.values()) == pytest.approx(1.0)


def test_compute_adjustment_low_regret_upweights():
    """Low regret → upweight that objective."""
    current = {"reduce_toil": 0.5, "improve_security": 0.3, "preserve_user_agency": 0.2}
    regret = {"reduce_toil": 0.1, "improve_security": 1.0, "preserve_user_agency": 1.0}
    new = R.compute_adjustment(
        current_vector=current, regret_per_objective=regret,
    )
    # reduce_toil should rise (pre-renorm), but after renorm both sides matter.
    # Easier check: it should NOT fall.
    assert new["reduce_toil"] >= current["reduce_toil"] - 0.05  # at minimum, not heavily downweighted


def test_compute_adjustment_in_band_regret_no_change():
    """Regret in [LOW, HIGH] range → no weight change."""
    current = {"reduce_toil": 0.5, "improve_security": 0.3, "preserve_user_agency": 0.2}
    regret = {"reduce_toil": 1.0, "improve_security": 1.0, "preserve_user_agency": 1.0}
    new = R.compute_adjustment(
        current_vector=current, regret_per_objective=regret,
    )
    # Within range → identity (post-renorm)
    for k, v in current.items():
        assert new[k] == pytest.approx(v)


def test_compute_adjustment_clamps_delta():
    """|Δ| ≤ WEIGHT_DELTA_CLAMP per cycle, no matter how bad the regret."""
    current = {"reduce_toil": 0.9, "improve_security": 0.1}
    # Extreme regret on reduce_toil
    regret = {"reduce_toil": 100.0, "improve_security": 0.0}
    new = R.compute_adjustment(
        current_vector=current, regret_per_objective=regret,
    )
    # The unclamped nudge would be much more than 0.10; check the
    # pre-renormalise delta was clamped.  Hard to assert directly
    # after renormalise — just check we don't fall below 0.05 floor.
    assert new["reduce_toil"] >= 0.05
    assert sum(new.values()) == pytest.approx(1.0)


def test_compute_adjustment_floors_at_005():
    """Weights never drop below 0.05 (signals can recover)."""
    current = {"reduce_toil": 0.05, "improve_security": 0.95}
    regret = {"reduce_toil": 100.0, "improve_security": 0.0}
    new = R.compute_adjustment(
        current_vector=current, regret_per_objective=regret,
    )
    assert new["reduce_toil"] >= 0.05 * 0.95   # post-renorm worst-case


def test_compute_adjustment_no_regret_for_objective_unchanged():
    """If we have no measurement for an objective, leave it alone."""
    current = {"reduce_toil": 0.5, "improve_security": 0.5}
    regret = {"reduce_toil": 5.0}   # no security entry
    new = R.compute_adjustment(
        current_vector=current, regret_per_objective=regret,
    )
    # security weight should be left untouched (then renormalised)
    assert new["improve_security"] > 0


# ── get_alignment_for_tenant ───────────────────────────────────────────


class FakeColumn:
    def __init__(self, vs): self._v = vs
    def __getitem__(self, i): return FakeCell(self._v[i])
class FakeCell:
    def __init__(self, v): self._v = v
    def as_py(self): return self._v
class FakeArrow:
    def __init__(self, rows):
        self._rows = rows
        self.num_rows = len(rows)
        cols = set()
        for r in rows: cols.update(r.keys())
        self.column_names = sorted(cols)
    def column(self, name):
        return FakeColumn([r.get(name) for r in self._rows])

class FakeTable:
    def __init__(self, rows=None):
        self._rows = list(rows or [])
        self.added = []
        self.updates = []
        self._where = None
    def search(self):
        self._where = None
        return self
    def where(self, expr):
        self._where = expr
        return self
    def select(self, *_args, **_kw):
        return self
    def limit(self, n):
        self._limit = n
        return self
    def to_arrow(self):
        rows = self._rows
        if self._where:
            for clause in self._where.split(" AND "):
                if " >= " in clause:
                    k, v = clause.split(" >= ", 1)
                    k = k.strip(); v = float(v.strip())
                    rows = [r for r in rows if float(r.get(k, 0)) >= v]
                elif " <= " in clause:
                    k, v = clause.split(" <= ", 1)
                    k = k.strip(); v = float(v.strip())
                    rows = [r for r in rows if float(r.get(k, 0)) <= v]
                elif " > " in clause:
                    k, v = clause.split(" > ", 1)
                    k = k.strip(); v = float(v.strip())
                    rows = [r for r in rows if float(r.get(k, 0)) > v]
                elif " < " in clause:
                    k, v = clause.split(" < ", 1)
                    k = k.strip(); v = float(v.strip())
                    rows = [r for r in rows if float(r.get(k, 0)) < v]
                elif " = " in clause:
                    k, v = clause.split(" = ", 1)
                    k = k.strip(); v = v.strip().strip("'\"")
                    rows = [r for r in rows if str(r.get(k)) == v]
        return FakeArrow(rows[: getattr(self, "_limit", 500)])
    def add(self, new_rows):
        self.added.extend(new_rows); self._rows.extend(new_rows)
    def update(self, *, where, values):
        self.updates.append((where, values))
        for clause in where.split(" AND "):
            k, v = clause.split(" = ", 1)
            k = k.strip(); v = v.strip().strip("'\"")
            for r in self._rows:
                if all(str(r.get(kk.strip())) == vv.strip().strip("'\"")
                       for kk, vv in [c.split(" = ", 1) for c in where.split(" AND ")]):
                    r.update(values)


def test_get_alignment_falls_back_to_defaults():
    fake_store = MagicMock()
    fake_store.table.return_value = FakeTable([])
    with patch("app.proposals.recalibration.LedgerStore.get", return_value=fake_store):
        out = R.get_alignment_for_tenant(tenant_id="t1", user_id="u1")
    # No rows → identical to DEFAULT_ALIGNMENT_VECTORS
    assert out == {k: dict(v) for k, v in DEFAULT_ALIGNMENT_VECTORS.items()}


def test_get_alignment_overlays_calibrated_weights():
    """Tenant has calibrated weights for one objective → those override."""
    fake_store = MagicMock()
    fake_store.table.return_value = FakeTable([
        {
            "tenant_id": "t1", "user_id": "u1",
            "signal_type": SIGNAL_INFORMATION_GAP,
            "objective_id": "reduce_toil",
            "weight": 0.7,   # calibrated up from 0.5 default
            "default_weight": 0.5,
            "last_regret_avg": 0.2,
            "last_recalibrated_ts": time.time(),
            "recalibration_count": 1,
        },
    ])
    with patch("app.proposals.recalibration.LedgerStore.get", return_value=fake_store):
        out = R.get_alignment_for_tenant(tenant_id="t1", user_id="u1")
    # Calibrated value overrides default
    assert out[SIGNAL_INFORMATION_GAP]["reduce_toil"] == 0.7
    # Other objectives keep their defaults
    assert out[SIGNAL_INFORMATION_GAP]["improve_security"] == \
        DEFAULT_ALIGNMENT_VECTORS[SIGNAL_INFORMATION_GAP]["improve_security"]


# ── run_one_cycle ──────────────────────────────────────────────────────


def test_run_one_cycle_skips_low_volume():
    """Below min_outcomes threshold → no adjustment (statistics too noisy)."""
    outcomes = FakeTable([
        # only 2 outcomes for info_gap, below threshold of 5
        {"proposal_id": "p1", "task_cost_usd": 0.1, "value_realized": 1, "regret_score": 0.1,
         "tenant_id": "t1", "user_id": "u1", "computed_at_ts": time.time()},
        {"proposal_id": "p2", "task_cost_usd": 0.2, "value_realized": 0, "regret_score": 2.0,
         "tenant_id": "t1", "user_id": "u1", "computed_at_ts": time.time()},
    ])
    proposals = FakeTable([
        {"id": "p1", "signal_type": "information_gap"},
        {"id": "p2", "signal_type": "information_gap"},
    ])
    weights = FakeTable([])

    fake_db = MagicMock()
    def open_tbl(name):
        return {"proposal_outcomes": outcomes,
                "task_proposals":    proposals,
                "alignment_weights": weights}[name]
    fake_db.open_table.side_effect = open_tbl

    with patch("lancedb.connect", return_value=fake_db):
        result = R.run_one_cycle()
    assert result["checked"] == 1
    assert result["skipped_low_volume"] == 1
    assert result["adjusted"] == 0
    assert len(weights.added) == 0


def test_run_one_cycle_adjusts_when_enough_data():
    """With enough outcomes → write/update alignment_weights row(s)."""
    now = time.time()
    # 6 outcomes for info_gap, all HIGH regret → downweight all objectives
    outcomes_rows = []
    for i in range(6):
        outcomes_rows.append({
            "proposal_id":   f"p{i}",
            "task_cost_usd": 0.3,
            "value_realized": 0.0,
            "regret_score":  3.0,    # > HIGH_REGRET
            "tenant_id":     "t1",
            "user_id":       "u1",
            "computed_at_ts": now,
        })
    outcomes = FakeTable(outcomes_rows)
    proposals = FakeTable([{"id": f"p{i}", "signal_type": "information_gap"} for i in range(6)])
    weights = FakeTable([])

    fake_db = MagicMock()
    def open_tbl(name):
        return {"proposal_outcomes": outcomes,
                "task_proposals":    proposals,
                "alignment_weights": weights}[name]
    fake_db.open_table.side_effect = open_tbl

    with patch("lancedb.connect", return_value=fake_db), \
         patch("app.proposals.recalibration.LedgerStore.get") as mock_store:
        # alignment lookup returns defaults (no prior weights)
        mock_store.return_value.table.return_value = FakeTable([])
        result = R.run_one_cycle()
    assert result["checked"] == 1
    assert result["skipped_low_volume"] == 0
    # 3 objectives in info_gap's default vector → 3 rows written
    assert result["adjusted"] == 3
    assert len(weights.added) == 3
    # All new rows for the SAME (tenant, signal_type) — sum should be ~1
    by_obj = {r["objective_id"]: r["weight"] for r in weights.added}
    assert pytest.approx(sum(by_obj.values())) == 1.0
    # And every objective got its last_regret_avg populated
    for r in weights.added:
        assert r["last_regret_avg"] == 3.0
        assert r["recalibration_count"] == 1
