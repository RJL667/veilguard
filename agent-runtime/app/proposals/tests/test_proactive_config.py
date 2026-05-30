"""Tests for ledger.proactive_config — per-tenant proactive-stream gates."""

from unittest.mock import patch, MagicMock
import pytest

from app.ledger import proactive_config as pc


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
    def limit(self, n):
        self._limit = n
        return self
    def to_arrow(self):
        rows = self._rows
        if self._where and " AND " in self._where:
            # simple tenant_id = 'X' AND user_id = 'Y' parser
            parts = self._where.split(" AND ")
            for p in parts:
                k, v = p.split(" = ", 1)
                k = k.strip(); v = v.strip().strip("'\"")
                rows = [r for r in rows if str(r.get(k)) == v]
        return FakeArrow(rows[: getattr(self, "_limit", 100)])
    def add(self, new_rows):
        self.added.extend(new_rows)
        self._rows.extend(new_rows)
    def update(self, *, where, values):
        # apply values to matching rows
        self.updates.append((where, values))
        # mimic where=" AND " parsing same as above
        parts = where.split(" AND ")
        for r in self._rows:
            match = True
            for p in parts:
                k, v = p.split(" = ", 1)
                k = k.strip(); v = v.strip().strip("'\"")
                if str(r.get(k)) != v:
                    match = False
                    break
            if match:
                r.update(values)


# ── get_or_default ──────────────────────────────────────────────────────


def test_get_or_default_synthesises_when_no_row():
    fake = FakeTable([])
    fake_store = MagicMock()
    fake_store.table.return_value = fake
    with patch("app.ledger.proactive_config.LedgerStore.get", return_value=fake_store):
        cfg = pc.get_or_default("t1", "u1")
    assert cfg.is_default is True
    assert cfg.proactive_stream_enabled is True
    assert cfg.proactive_cycles_per_day == 12
    assert cfg.proactive_approval_cap_per_day == 20
    assert cfg.cost_ceiling_per_tenant_per_day_usd == 5.0
    assert cfg.paused is False
    assert cfg.tenant_id == "t1"
    assert cfg.user_id == "u1"


def test_get_or_default_reads_existing_row():
    fake = FakeTable([{
        "tenant_id": "t1", "user_id": "u1",
        "proactive_stream_enabled": False,
        "proactive_cycles_per_day": 6,
        "proactive_approval_cap_per_day": 5,
        "cost_ceiling_per_tenant_per_day_usd": 1.5,
        "paused": True,
        "paused_reason": "signal_quality_drift",
        "paused_at_ts": 1779000000.0,
    }])
    fake_store = MagicMock()
    fake_store.table.return_value = fake
    with patch("app.ledger.proactive_config.LedgerStore.get", return_value=fake_store):
        cfg = pc.get_or_default("t1", "u1")
    assert cfg.is_default is False
    assert cfg.proactive_stream_enabled is False
    assert cfg.proactive_cycles_per_day == 6
    assert cfg.cost_ceiling_per_tenant_per_day_usd == 1.5
    assert cfg.paused is True
    assert cfg.paused_reason == "signal_quality_drift"


def test_stream_active_combines_enabled_and_paused():
    fake = FakeTable([])
    fake_store = MagicMock()
    fake_store.table.return_value = fake
    with patch("app.ledger.proactive_config.LedgerStore.get", return_value=fake_store):
        cfg = pc.get_or_default("t1", "u1")
    assert cfg.stream_active is True   # default config → active

    cfg.proactive_stream_enabled = False
    assert cfg.stream_active is False

    cfg.proactive_stream_enabled = True
    cfg.paused = True
    assert cfg.stream_active is False  # paused → inactive even if enabled


# ── upsert ──────────────────────────────────────────────────────────────


def test_upsert_inserts_when_no_row():
    fake = FakeTable([])
    fake_store = MagicMock()
    fake_store.table.return_value = fake
    with patch("app.ledger.proactive_config.LedgerStore.get", return_value=fake_store):
        cfg = pc.upsert(
            tenant_id="t1", user_id="u1",
            proactive_cycles_per_day=6,
            cost_ceiling_per_tenant_per_day_usd=2.0,
        )
    assert len(fake.added) == 1
    row = fake.added[0]
    assert row["tenant_id"] == "t1"
    assert row["proactive_cycles_per_day"] == 6
    assert row["cost_ceiling_per_tenant_per_day_usd"] == 2.0
    # Fields not specified should be defaults
    assert row["proactive_stream_enabled"] is True
    assert row["proactive_approval_cap_per_day"] == 20
    # Returned cfg reflects the insert
    assert cfg.proactive_cycles_per_day == 6


def test_upsert_updates_when_row_exists():
    fake = FakeTable([{
        "tenant_id": "t1", "user_id": "u1",
        "proactive_stream_enabled": True,
        "proactive_cycles_per_day": 12,
        "proactive_approval_cap_per_day": 20,
        "cost_ceiling_per_tenant_per_day_usd": 5.0,
        "paused": False,
        "paused_reason": None,
        "paused_at_ts": None,
    }])
    fake_store = MagicMock()
    fake_store.table.return_value = fake
    with patch("app.ledger.proactive_config.LedgerStore.get", return_value=fake_store):
        cfg = pc.upsert(
            tenant_id="t1", user_id="u1",
            proactive_cycles_per_day=6,
        )
    assert len(fake.added) == 0   # update, not insert
    assert len(fake.updates) == 1
    # the second-pass read sees the updated cycles
    assert cfg.proactive_cycles_per_day == 6


def test_pause_sets_paused_at_ts():
    fake = FakeTable([])
    fake_store = MagicMock()
    fake_store.table.return_value = fake
    with patch("app.ledger.proactive_config.LedgerStore.get", return_value=fake_store):
        cfg = pc.pause(tenant_id="t1", user_id="u1", reason="cost_runaway")
    assert cfg.paused is True
    assert cfg.paused_reason == "cost_runaway"
    # paused_at_ts is set during insert
    inserted = fake.added[0]
    assert inserted["paused_at_ts"] is not None


def test_resume_clears_paused_state():
    fake = FakeTable([{
        "tenant_id": "t1", "user_id": "u1",
        "proactive_stream_enabled": True,
        "proactive_cycles_per_day": 12,
        "proactive_approval_cap_per_day": 20,
        "cost_ceiling_per_tenant_per_day_usd": 5.0,
        "paused": True,
        "paused_reason": "test",
        "paused_at_ts": 1779000000.0,
    }])
    fake_store = MagicMock()
    fake_store.table.return_value = fake
    with patch("app.ledger.proactive_config.LedgerStore.get", return_value=fake_store):
        cfg = pc.resume(tenant_id="t1", user_id="u1")
    assert cfg.paused is False
    assert cfg.paused_reason is None
