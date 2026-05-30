"""Tests for proposals/constitution_amendments.py — amendment proposal writer."""

from unittest.mock import patch, MagicMock
import pytest

from app.proposals import constitution_amendments as CA


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
        self._where = None
    def search(self):
        self._where = None
        return self
    def where(self, expr):
        self._where = expr
        return self
    def select(self, *_a, **_kw):
        return self
    def limit(self, n):
        self._limit = n
        return self
    def to_arrow(self):
        rows = self._rows
        if self._where:
            for clause in self._where.split(" AND "):
                clause = clause.strip()
                if " >= " in clause:
                    k, v = clause.split(" >= ", 1)
                    rows = [r for r in rows if float(r.get(k.strip(), 0)) >= float(v.strip())]
                elif " = " in clause:
                    k, v = clause.split(" = ", 1)
                    k = k.strip(); v = v.strip().strip("'\"")
                    rows = [r for r in rows if str(r.get(k)) == v]
        return FakeArrow(rows[: getattr(self, "_limit", 500)])
    def add(self, new_rows):
        self.added.extend(new_rows); self._rows.extend(new_rows)


# ── draft_amendment_brief ──────────────────────────────────────────────


def test_draft_brief_includes_trigger_rule_and_confidence():
    lesson = {
        "trigger": "Critic skips review",
        "rule": "Require Critic on team_knowledge writes",
        "confidence": 0.82,
        "reinforcement_count": 7,
    }
    out = CA.draft_amendment_brief(lesson)
    assert "Critic skips review" in out["brief"]
    assert "Require Critic" in out["brief"]
    assert "0.82" in out["brief"]
    assert "7" in out["brief"]
    assert "unified-diff" in out["spec"]
    assert "title" in out


# ── propose_one ────────────────────────────────────────────────────────


async def _rpwc_stub(**kwargs):
    """Stub for record_proposal_with_content — returns (pid, None) and
    records the kwargs for assertion."""
    _rpwc_stub.calls.append(kwargs)  # type: ignore[attr-defined]
    return (f"prop-amendment-{len(_rpwc_stub.calls)}", None)  # type: ignore[attr-defined]


def _mk_archive_mock(rows):
    """Build a Lance-table-shaped mock whose
    .search().limit().to_arrow().to_pylist() returns `rows`."""
    arr_mock = MagicMock()
    arr_mock.to_pylist.return_value = rows
    q = MagicMock()
    q.where.return_value = q
    q.limit.return_value = q
    q.select.return_value = q
    q.to_arrow.return_value = arr_mock
    tbl = MagicMock()
    tbl.search.return_value = q
    return tbl


def _reset_rpwc_stub():
    _rpwc_stub.calls = []  # type: ignore[attr-defined]


async def test_propose_one_writes_proposal_for_eligible_lesson():
    proposals_tbl = FakeTable([])
    fake_store = MagicMock()
    fake_store.table.return_value = proposals_tbl
    _reset_rpwc_stub()
    with patch("app.proposals.constitution_amendments.LedgerStore.get", return_value=fake_store), \
         patch("app.memory.phase_7_writers.record_proposal_with_content", side_effect=_rpwc_stub):
        pid = await CA.propose_one(lesson={
            "id":                  "lsn-x",
            "tenant_id":           "t1",
            "user_id":             "u1",
            "trigger":             "X",
            "rule":                "Y",
            "confidence":          0.8,
            "reinforcement_count": 6,
        })
    assert pid == "prop-amendment-1"
    assert len(_rpwc_stub.calls) == 1  # type: ignore[attr-defined]
    kwargs = _rpwc_stub.calls[0]  # type: ignore[attr-defined]
    assert kwargs["signal_type"] == CA.SIGNAL_CONSTITUTION_AMENDMENT
    assert kwargs["proposed_assignee"] == "critic-prose"
    assert "lsn-x" in kwargs["rationale"]


async def test_propose_one_skips_if_already_proposed():
    """Idempotency: same lesson_id in rationale → skip."""
    existing = {
        "id":          "prop-exists",
        "rationale":   "constitution_amendment from lesson lsn-x ...",
        "status":      "pending",
        "tenant_id":   "t1",
        "user_id":     "u1",
        "signal_type": CA.SIGNAL_CONSTITUTION_AMENDMENT,
    }
    proposals_tbl = FakeTable([existing])
    fake_store = MagicMock()
    fake_store.table.return_value = proposals_tbl
    _reset_rpwc_stub()
    with patch("app.proposals.constitution_amendments.LedgerStore.get", return_value=fake_store), \
         patch("app.memory.phase_7_writers.record_proposal_with_content", side_effect=_rpwc_stub):
        pid = await CA.propose_one(lesson={
            "id":                  "lsn-x",
            "tenant_id":           "t1",
            "user_id":             "u1",
            "trigger":             "X",
            "rule":                "Y",
            "confidence":          0.8,
            "reinforcement_count": 6,
        })
    assert pid is None
    assert len(_rpwc_stub.calls) == 0  # type: ignore[attr-defined]


async def test_propose_one_skips_when_missing_required_fields():
    proposals_tbl = FakeTable([])
    fake_store = MagicMock()
    fake_store.table.return_value = proposals_tbl
    _reset_rpwc_stub()
    with patch("app.proposals.constitution_amendments.LedgerStore.get", return_value=fake_store), \
         patch("app.memory.phase_7_writers.record_proposal_with_content", side_effect=_rpwc_stub):
        # missing user_id
        assert await CA.propose_one(lesson={"id": "lsn-x", "tenant_id": "t1"}) is None
        # missing id
        assert await CA.propose_one(lesson={"tenant_id": "t1", "user_id": "u1"}) is None
    assert len(_rpwc_stub.calls) == 0  # type: ignore[attr-defined]


# ── run_one_cycle ──────────────────────────────────────────────────────


async def test_run_one_cycle_proposes_eligible_skips_others():
    """Post-M1 cutover: lessons come from TCMM archive via lessons_reader,
    not the dropped org_memory Lance table.  We stub the archive scan
    to return one lesson row that meets the amendment thresholds."""
    archive_rows = [
        # Eligible: [lesson] prefix + active + confidence + reinforcement met.
        {
            "aid": 100, "user_id": "u1",
            "text": "[lesson] trigger=T rule=R confidence=0.80",
            "extracted_by": "agent:critic-prose",
            "namespace": "agent_critic-prose_obser",
            "created_ts": 1000.0,
        },
    ] * 6  # Repeat to bump reinforcement count via distinct authors
    # Make extracted_by distinct so reinforcement_count >= 5.
    for i, r in enumerate(archive_rows):
        archive_rows[i] = {**r, "extracted_by": f"agent:critic-{i}"}

    proposals_tbl = FakeTable([])
    fake_store = MagicMock()
    fake_store.table.return_value = proposals_tbl
    fake_store._db.open_table.return_value = _mk_archive_mock(archive_rows)

    fake_db = MagicMock()
    fake_db.open_table.return_value = _mk_archive_mock(archive_rows)

    create_calls = []
    async def fake_rpwc(**kwargs):
        create_calls.append(kwargs)
        return (f"prop-{len(create_calls)}", None)

    with patch("lancedb.connect", return_value=fake_db), \
         patch("app.ledger.store.LedgerStore.get", return_value=fake_store), \
         patch("app.proposals.constitution_amendments.LedgerStore.get", return_value=fake_store), \
         patch("app.memory.phase_7_writers.record_proposal_with_content", side_effect=fake_rpwc):
        out = await CA.run_one_cycle()

    # Post-M1: reader collapses observations-per-trigger into one
    # lesson; we expect exactly one amendment proposal.
    assert out["scanned"] >= 1
    assert len(create_calls) == 1
    # The rationale carries the lesson id from the TCMM reader.
    assert "lsn-tcmm-100" in create_calls[0]["rationale"]
