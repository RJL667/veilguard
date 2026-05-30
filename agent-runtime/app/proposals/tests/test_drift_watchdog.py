"""Tests for proposals/drift_watchdog.py — auto-pause on volume spike."""

import time
from unittest.mock import patch, MagicMock

import pytest

from app.proposals import drift_watchdog as dw


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
        self._where = None
    def search(self):
        self._where = None
        return self
    def where(self, expr):
        self._where = expr
        return self
    def select(self, *_args, **_kw): return self
    def limit(self, n):
        self._limit = n
        return self
    def to_arrow(self):
        rows = self._rows
        if self._where:
            # parse "tenant_id = 'X' AND user_id = 'Y' AND created_ts >= N AND created_ts < M"
            for clause in self._where.split(" AND "):
                if " = " in clause:
                    k, v = clause.split(" = ", 1)
                    k = k.strip(); v = v.strip().strip("'\"")
                    rows = [r for r in rows if str(r.get(k)) == v]
                elif " >= " in clause:
                    k, v = clause.split(" >= ", 1)
                    k = k.strip(); v = float(v)
                    rows = [r for r in rows if float(r.get(k, 0)) >= v]
                elif " < " in clause:
                    k, v = clause.split(" < ", 1)
                    k = k.strip(); v = float(v)
                    rows = [r for r in rows if float(r.get(k, 0)) < v]
        return FakeArrow(rows[: getattr(self, "_limit", 5000)])


def _proposal(tenant="t1", user="u1", created_ts=None):
    return {
        "id": f"prop-{int(created_ts or 0)}",
        "tenant_id": tenant, "user_id": user,
        "created_ts": created_ts or time.time(),
    }


# ── evaluate_tenant ─────────────────────────────────────────────────────


def test_evaluate_tenant_no_drift_when_volume_steady():
    now = time.time()
    day = 86400.0
    # Steady 2 proposals/day for 30 days.  Use day-CENTRE times to
    # avoid bucket-boundary contamination.
    rows = []
    for d in range(30):
        # bucket d covers [now - (30-d)*day, now - (29-d)*day)
        bucket_centre = now - (30 - d) * day + day / 2
        for _ in range(2):
            rows.append(_proposal(created_ts=bucket_centre))
    tbl = FakeTable(rows)
    ev = dw.evaluate_tenant(tbl=tbl, tenant_id="t1", user_id="u1", now_ts=now)
    # median should be 2, today=2, threshold=6 → no drift
    assert ev["median"] == 2.0
    assert ev["today_count"] == 2
    assert ev["drifted"] is False


def test_evaluate_tenant_drift_when_today_spikes():
    now = time.time()
    day = 86400.0
    rows = []
    # 2/day for 29 prior days using day-centre times
    for d in range(29):
        bucket_centre = now - (30 - d) * day + day / 2
        for _ in range(2):
            rows.append(_proposal(created_ts=bucket_centre))
    # 10 today (last bucket — centre)
    today_centre = now - day / 2
    for _ in range(10):
        rows.append(_proposal(created_ts=today_centre))
    tbl = FakeTable(rows)
    ev = dw.evaluate_tenant(tbl=tbl, tenant_id="t1", user_id="u1", now_ts=now)
    assert ev["median"] == 2.0
    assert ev["today_count"] == 10
    assert ev["threshold"] == 6.0  # 3 × 2
    assert ev["drifted"] is True


def test_evaluate_tenant_no_drift_when_median_too_low():
    """Low-volume tenants don't drift on a single noisy day."""
    now = time.time()
    day = 86400.0
    rows = []
    # 0 prior days, 5 today
    for _ in range(5):
        rows.append(_proposal(created_ts=now - 100))
    tbl = FakeTable(rows)
    ev = dw.evaluate_tenant(tbl=tbl, tenant_id="t1", user_id="u1", now_ts=now)
    assert ev["median"] == 0.0
    assert ev["today_count"] == 5
    # median < DRIFT_MIN_MEDIAN (1.0) → drift gate held closed
    assert ev["drifted"] is False


# ── run_one_cycle: end-to-end with mocked proactive_config.pause ──────


def test_run_one_cycle_auto_pauses_drifted_tenant():
    now = time.time()
    day = 86400.0
    # tenant t1 — drifted (10 today vs 2 median).  Day-centre times.
    rows = []
    for d in range(29):
        bucket_centre = now - (30 - d) * day + day / 2
        for _ in range(2):
            rows.append(_proposal(tenant="t1", user="u1", created_ts=bucket_centre))
    today_centre = now - day / 2
    for _ in range(10):
        rows.append(_proposal(tenant="t1", user="u1", created_ts=today_centre))
    # tenant t2 — not drifted (3/day steady, day-centre)
    for d in range(30):
        bucket_centre = now - (30 - d) * day + day / 2
        for _ in range(3):
            rows.append(_proposal(tenant="t2", user="u2", created_ts=bucket_centre))
    fake_tbl = FakeTable(rows)
    fake_db = MagicMock()
    fake_db.open_table.return_value = fake_tbl

    pause_calls = []
    def fake_pause(*, tenant_id, user_id, reason):
        pause_calls.append({"tenant_id": tenant_id, "user_id": user_id, "reason": reason})
        return MagicMock(paused=True)

    fake_unpaused = MagicMock(paused=False)
    with patch("lancedb.connect", return_value=fake_db), \
         patch("app.proposals.drift_watchdog.proactive_config.pause", side_effect=fake_pause), \
         patch("app.proposals.drift_watchdog.proactive_config.get_or_default",
               return_value=fake_unpaused):
        summary = dw.run_one_cycle()

    assert summary["checked"] == 2
    assert summary["paused"] == 1
    assert len(pause_calls) == 1
    assert pause_calls[0]["tenant_id"] == "t1"
    assert "signal_quality_drift" in pause_calls[0]["reason"]
    assert "today=10" in pause_calls[0]["reason"]


def test_run_one_cycle_skips_already_paused():
    """Idempotent: drifted but already-paused tenants don't re-pause."""
    now = time.time()
    day = 86400.0
    rows = []
    for d in range(29):
        bucket_centre = now - (30 - d) * day + day / 2
        for _ in range(2):
            rows.append(_proposal(created_ts=bucket_centre))
    today_centre = now - day / 2
    for _ in range(10):
        rows.append(_proposal(created_ts=today_centre))
    fake_tbl = FakeTable(rows)
    fake_db = MagicMock()
    fake_db.open_table.return_value = fake_tbl

    pause_calls = []
    def fake_pause(**kw):
        pause_calls.append(kw)
        return MagicMock()

    fake_already_paused = MagicMock(paused=True)
    with patch("lancedb.connect", return_value=fake_db), \
         patch("app.proposals.drift_watchdog.proactive_config.pause", side_effect=fake_pause), \
         patch("app.proposals.drift_watchdog.proactive_config.get_or_default",
               return_value=fake_already_paused):
        summary = dw.run_one_cycle()

    assert summary["checked"] == 1
    assert summary["paused"] == 0
    assert len(pause_calls) == 0    # already paused → no re-call


def test_discover_active_pairs_dedupes():
    rows = [
        _proposal(tenant="t1", user="u1"),
        _proposal(tenant="t1", user="u1"),
        _proposal(tenant="t2", user="u2"),
    ]
    pairs = dw.discover_active_pairs(FakeTable(rows))
    assert pairs == [("t1", "u1"), ("t2", "u2")]
