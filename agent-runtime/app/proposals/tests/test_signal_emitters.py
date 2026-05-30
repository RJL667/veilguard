"""Tests for proposals/signal_emitters.py — low_stability + stale_chain."""

import time
from unittest.mock import patch, MagicMock
import pytest
import pyarrow as pa

from app.proposals import signal_emitters as SE


# Fake table that supports schema + select + add — emitters use real
# pyarrow column lookups so we need a slightly fuller fake.


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
    def __init__(self, rows=None, schema_fields=None):
        self._rows = list(rows or [])
        self.added = []
        self._where = None
        # Build a pyarrow-like schema from the keys of the first row
        self._fields = schema_fields or []
        if not self._fields and self._rows:
            for k in self._rows[0].keys():
                self._fields.append(self._make_field(k, type(self._rows[0][k])))

    def _make_field(self, name, py_type):
        # Crude mapping — fine for tests since SE checks str(f.type)
        if py_type is int:    t = pa.int64()
        elif py_type is float: t = pa.float64()
        elif py_type is bool: t = pa.bool_()
        elif py_type is list: t = pa.list_(pa.int64())
        elif py_type is dict: t = pa.struct([pa.field("k", pa.string())])
        else:                  t = pa.string()
        return pa.field(name, t, nullable=True)

    @property
    def schema(self):
        return pa.schema(self._fields) if self._fields else pa.schema([])

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
                clause = clause.strip()
                if " = " in clause:
                    k, v = clause.split(" = ", 1)
                    k = k.strip(); v = v.strip().strip("'\"")
                    rows = [r for r in rows if str(r.get(k)) == v]
        return FakeArrow(rows[: getattr(self, "_limit", 5000)])

    def add(self, new_rows):
        self.added.extend(new_rows)
        self._rows.extend(new_rows)

    def count_rows(self, filter=None):
        # Test fixture — emitter's empty-archive guard calls
        # `tbl.count_rows()` to short-circuit before `.select(...)`
        # would fail on an empty-schema lance dataset.
        if filter:
            # rough where filter — just substring match on 'k = v' pairs
            rows = self._rows
            for clause in filter.split(" AND "):
                clause = clause.strip()
                if " = " in clause:
                    k, v = clause.split(" = ", 1)
                    k = k.strip(); v = v.strip().strip("'\"")
                    rows = [r for r in rows if str(r.get(k)) == v]
            return len(rows)
        return len(self._rows)


# ── emit_low_stability_clusters ────────────────────────────────────────


def test_low_stability_emits_when_density_below_floor():
    """5 archive rows on same topic, all low density → emit."""
    archive_rows = [
        {
            "aid": i, "topics": ["POPIA-retention"], "claims": [f"claim {i}"],
            "density_score": 0.15, "timestamp": time.time(),
            "user_id": "u1", "namespace": "x", "block_class": "",
            "source_block_ids": [i*10],
        }
        for i in range(5)
    ]
    archive_tbl = FakeTable(archive_rows, schema_fields=[
        pa.field("aid", pa.int64()),
        pa.field("topics", pa.list_(pa.string())),
        pa.field("claims", pa.list_(pa.string())),
        pa.field("density_score", pa.float64()),
        pa.field("timestamp", pa.float64()),
        pa.field("user_id", pa.string()),
        pa.field("namespace", pa.string()),
        pa.field("block_class", pa.string()),
        pa.field("source_block_ids", pa.list_(pa.int64())),
        pa.field("text", pa.string()),
        pa.field("heat", pa.float64()),
    ])
    dream_tbl = FakeTable([], schema_fields=[
        pa.field(name, pa.string(), nullable=True)
        for name in ("aid","namespace","user_id","id","text","origin",
                     "block_class","topics","claims","source_block_ids",
                     "density_score","heat","timestamp","created_ts",
                     "session_id","session_date","recallable","vector",
                     "knowledge_class","temporal","archive_stats","lineage",
                     "extras_json","extracted_by","behavioural_json",
                     "suppresses_json","semantic_json","entropy_static_json",
                     "semantic_text","semantic_text_tail","fingerprint")
    ])

    out = SE.emit_low_stability_clusters(
        archive_tbl=archive_tbl, dream_tbl=dream_tbl,
    )
    assert out["emitted"] == 1
    assert len(dream_tbl.added) == 1
    written = dream_tbl.added[0]
    # block_class field is restricted by what made it through the
    # schema filter — but text always survives
    assert "POPIA-retention" in str(written.get("text") or "")


def test_low_stability_skips_high_density_topics():
    """Topics with density >= floor → skip."""
    archive_rows = [
        {
            "aid": i, "topics": ["clean-topic"], "claims": [f"c{i}"],
            "density_score": 0.80,
            "timestamp": time.time(), "user_id": "u1",
            "source_block_ids": [i],
            "namespace": "", "block_class": "",
        }
        for i in range(5)
    ]
    archive_tbl = FakeTable(archive_rows, schema_fields=[
        pa.field("aid", pa.int64()),
        pa.field("topics", pa.list_(pa.string())),
        pa.field("claims", pa.list_(pa.string())),
        pa.field("density_score", pa.float64()),
        pa.field("timestamp", pa.float64()),
        pa.field("user_id", pa.string()),
        pa.field("namespace", pa.string()),
        pa.field("block_class", pa.string()),
        pa.field("source_block_ids", pa.list_(pa.int64())),
    ])
    dream_tbl = FakeTable([], schema_fields=[pa.field("x", pa.string(), nullable=True)])

    out = SE.emit_low_stability_clusters(
        archive_tbl=archive_tbl, dream_tbl=dream_tbl,
    )
    assert out["emitted"] == 0


# ── emit_stale_supersession_chains ─────────────────────────────────────


def test_stale_chain_emits_when_topic_old_enough():
    """All rows on topic timestamped > 60d ago → emit."""
    old_ts = time.time() - 100 * 86400.0
    archive_rows = [
        {
            "aid": i, "topics": ["old-topic"], "claims": [f"c{i}"],
            "density_score": 0.5,
            "timestamp": old_ts, "user_id": "u1",
            "source_block_ids": [i],
            "namespace": "", "block_class": "",
        }
        for i in range(4)   # ≥ STALE_CHAIN_MIN_CLAIMS=3
    ]
    archive_tbl = FakeTable(archive_rows, schema_fields=[
        pa.field("aid", pa.int64()),
        pa.field("topics", pa.list_(pa.string())),
        pa.field("claims", pa.list_(pa.string())),
        pa.field("density_score", pa.float64()),
        pa.field("timestamp", pa.float64()),
        pa.field("user_id", pa.string()),
        pa.field("namespace", pa.string()),
        pa.field("block_class", pa.string()),
        pa.field("source_block_ids", pa.list_(pa.int64())),
    ])
    dream_tbl = FakeTable([], schema_fields=[pa.field("x", pa.string(), nullable=True)])

    out = SE.emit_stale_supersession_chains(
        archive_tbl=archive_tbl, dream_tbl=dream_tbl,
    )
    assert out["emitted"] == 1


def test_stale_chain_skips_recent_topics():
    """Recent timestamps → skip."""
    recent_ts = time.time() - 1 * 86400.0  # 1d ago
    archive_rows = [
        {
            "aid": i, "topics": ["fresh-topic"], "claims": [f"c{i}"],
            "timestamp": recent_ts, "user_id": "u1",
            "source_block_ids": [i], "density_score": 0.5,
            "namespace": "", "block_class": "",
        }
        for i in range(5)
    ]
    archive_tbl = FakeTable(archive_rows, schema_fields=[
        pa.field("aid", pa.int64()),
        pa.field("topics", pa.list_(pa.string())),
        pa.field("claims", pa.list_(pa.string())),
        pa.field("timestamp", pa.float64()),
        pa.field("density_score", pa.float64()),
        pa.field("user_id", pa.string()),
        pa.field("namespace", pa.string()),
        pa.field("block_class", pa.string()),
        pa.field("source_block_ids", pa.list_(pa.int64())),
    ])
    dream_tbl = FakeTable([], schema_fields=[pa.field("x", pa.string(), nullable=True)])

    out = SE.emit_stale_supersession_chains(
        archive_tbl=archive_tbl, dream_tbl=dream_tbl,
    )
    assert out["emitted"] == 0


# ── _scan_topics ───────────────────────────────────────────────────────


# ── Empty-archive guard (2026-05-28 regression test) ──────────────────


def test_scan_topics_returns_empty_on_zero_row_archive():
    """[EMPTY_ARCHIVE_GUARD_2026_05_28] regression — an empty archive
    table (0 rows + uninitialised schema, common on local dev and
    freshly-provisioned tenants) must NOT raise SchemaError when the
    emitter tries to `.select(['aid', ...])`.  Pre-patch behaviour
    was a WARNING log every 24h cycle indefinitely.
    """
    tbl = FakeTable(rows=[], schema_fields=[])
    grouped = SE._scan_topics(tbl=tbl)
    assert grouped == {}, (
        "empty archive should short-circuit before .select() raises"
    )


def test_scan_topics_returns_empty_on_partial_schema():
    """Same guard — if the table has a non-empty schema but is missing
    the `aid` or `topics` columns the emitter reads, we MUST short-
    circuit rather than letting Lance raise downstream.
    """
    # Schema lacks 'aid' AND 'topics' — the two columns the emitter
    # reads via `.select([...])` AND iterates over in the projection.
    tbl = FakeTable(rows=[], schema_fields=[
        pa.field("some_other_field", pa.string()),
    ])
    grouped = SE._scan_topics(tbl=tbl)
    assert grouped == {}


def test_emit_low_stability_on_empty_archive_is_zero():
    """End-to-end view of the guard at the emit_low_stability level."""
    archive = FakeTable(rows=[], schema_fields=[])
    dream = FakeTable(rows=[], schema_fields=[
        pa.field("aid",         pa.int64()),
        pa.field("user_id",     pa.string()),
        pa.field("topics",      pa.list_(pa.string())),
        pa.field("block_class", pa.string()),
    ])
    res = SE.emit_low_stability_clusters(archive_tbl=archive, dream_tbl=dream)
    assert res["emitted"] == 0
    assert res["topics_scanned"] == 0
    assert dream.added == []


def test_emit_stale_chain_on_empty_archive_is_zero():
    archive = FakeTable(rows=[], schema_fields=[])
    dream = FakeTable(rows=[], schema_fields=[
        pa.field("aid",         pa.int64()),
        pa.field("user_id",     pa.string()),
        pa.field("topics",      pa.list_(pa.string())),
        pa.field("block_class", pa.string()),
    ])
    res = SE.emit_stale_supersession_chains(archive_tbl=archive, dream_tbl=dream)
    assert res["emitted"] == 0
    assert res["topics_scanned"] == 0
    assert dream.added == []


def test_scan_topics_groups_by_primary_topic():
    rows = [
        {"aid": 1, "topics": ["A"], "claims": []},
        {"aid": 2, "topics": ["A", "B"], "claims": []},  # primary = A
        {"aid": 3, "topics": ["B"], "claims": []},
        {"aid": 4, "topics": [], "claims": []},          # skipped (no topics)
    ]
    tbl = FakeTable(rows, schema_fields=[
        pa.field("aid", pa.int64()),
        pa.field("topics", pa.list_(pa.string())),
        pa.field("claims", pa.list_(pa.string())),
        pa.field("density_score", pa.float64()),
        pa.field("timestamp", pa.float64()),
        pa.field("user_id", pa.string()),
        pa.field("namespace", pa.string()),
        pa.field("block_class", pa.string()),
        pa.field("source_block_ids", pa.list_(pa.int64())),
    ])
    grouped = SE._scan_topics(tbl=tbl)
    assert len(grouped["A"]) == 2
    assert len(grouped["B"]) == 1
    # "no topics" row is dropped
    assert sum(len(v) for v in grouped.values()) == 3
