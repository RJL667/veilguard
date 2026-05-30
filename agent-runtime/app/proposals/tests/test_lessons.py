"""Tests for proposals/lessons.py — reflective_heuristic → org_memory promotion."""

from unittest.mock import patch, MagicMock
import pytest

from app.proposals import lessons as L


# ── Fake Lance table ───────────────────────────────────────────────────


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
    def limit(self, n):
        self._limit = n
        return self
    def to_arrow(self):
        rows = self._rows
        if self._where:
            for clause in self._where.split(" AND "):
                k, v = clause.split(" = ", 1)
                k = k.strip(); v = v.strip().strip("'\"")
                rows = [r for r in rows if str(r.get(k)) == v]
        return FakeArrow(rows[: getattr(self, "_limit", 500)])
    def add(self, new_rows):
        self.added.extend(new_rows)
        self._rows.extend(new_rows)


# ── extract_lesson_from_dream_row ──────────────────────────────────────


def test_extract_returns_none_for_wrong_block_class():
    row = {"block_class": "INFORMATION_GAP", "text": "x", "claims": []}
    assert L.extract_lesson_from_dream_row(row) is None


def test_extract_returns_none_for_empty_content():
    row = {"block_class": "REFLECTIVE_HEURISTIC", "text": "", "claims": []}
    assert L.extract_lesson_from_dream_row(row) is None


def test_extract_uses_claims_when_present():
    row = {
        "block_class": "REFLECTIVE_HEURISTIC",
        "text": "ignored when claims present",
        "claims": ["When critic skips review", "Add a Critic to the Task"],
        "aid": 42,
    }
    out = L.extract_lesson_from_dream_row(row)
    assert out is not None
    assert out["trigger"] == "When critic skips review"
    assert out["rule"] == "Add a Critic to the Task"
    assert out["promoted_from"] == "dream-aid-42"


def test_extract_falls_back_to_text_split():
    row = {
        "block_class": "RECURRING_RITUAL",
        "text": "On every new KYC. Run the standard 5-step checklist.",
        "claims": [],
        "aid": 99,
    }
    out = L.extract_lesson_from_dream_row(row)
    assert out is not None
    assert out["trigger"] == "On every new KYC"
    assert "5-step checklist" in out["rule"]


# ── evaluate_promotion ─────────────────────────────────────────────────


def test_evaluate_all_thresholds_met():
    d = L.evaluate_promotion(
        confidence=0.6, reinforcement_count=3, distinct_agents=2,
    )
    assert d["promote"] is True
    assert d["reasons"] == []


def test_evaluate_low_confidence_blocks():
    d = L.evaluate_promotion(
        confidence=0.3, reinforcement_count=10, distinct_agents=5,
    )
    assert d["promote"] is False
    assert any("confidence" in r for r in d["reasons"])


def test_evaluate_low_reinforcement_blocks():
    d = L.evaluate_promotion(
        confidence=0.9, reinforcement_count=1, distinct_agents=5,
    )
    assert d["promote"] is False
    assert any("reinforcement_count" in r for r in d["reasons"])


def test_evaluate_single_agent_blocks_per_spec_3_8_5():
    """Spec §3.8.5 — cross-agent reinforcement required."""
    d = L.evaluate_promotion(
        confidence=0.9, reinforcement_count=10, distinct_agents=1,
    )
    assert d["promote"] is False
    assert any("distinct_agents" in r for r in d["reasons"])


def test_evaluate_amendment_eligible_when_high_reinforcement():
    """≥5 reinforcements + ≥0.75 confidence → constitution-amendment-eligible."""
    d = L.evaluate_promotion(
        confidence=0.8, reinforcement_count=6, distinct_agents=2,
    )
    assert d["promote"] is True
    assert d["amendment_eligible"] is True


def test_evaluate_promote_but_not_amendment_eligible():
    """Promotes but not amendment-eligible (under 5 reinforcements)."""
    d = L.evaluate_promotion(
        confidence=0.8, reinforcement_count=3, distinct_agents=2,
    )
    assert d["promote"] is True
    assert d["amendment_eligible"] is False


# ── promote_one ────────────────────────────────────────────────────────


async def test_promote_one_happy_path_writes_row():
    """Post-M1 cutover: promote_one routes through TCMM split-writer.
    We mock the split-writer to capture the trigger/rule/confidence it
    was called with."""
    tbl = FakeTable([])
    row = {
        "block_class": "REFLECTIVE_HEURISTIC",
        "text": "trigger here. rule here.",
        "claims": ["When X happens", "Do Y in response"],
        "aid": 7,
    }
    captured: dict = {}

    async def _writer_stub(**kw):
        captured.update(kw)
        return True

    with patch("app.proposals.lessons._existing_promoted_from", return_value=False), \
         patch("app.memory.phase_7_writers.promote_lesson_to_team_knowledge",
               side_effect=_writer_stub):
        result = await L.promote_one(
            lessons_tbl=tbl,
            tenant_id="t1", user_id="u1",
            dream_row=row, confidence=0.7,
            reinforcement_count=3, distinct_agents=2,
            reinforced_by_agent_ids=["agent:researcher", "agent:critic-claim"],
        )
    assert result["promoted"] is True
    assert result["lesson_id"] is not None
    assert captured.get("trigger") == "When X happens"
    assert captured.get("rule")    == "Do Y in response"
    assert captured.get("confidence") == pytest.approx(0.7)
    assert captured.get("promoted_from") == "dream-aid-7"


async def test_promote_one_blocks_low_confidence():
    tbl = FakeTable([])
    row = {
        "block_class": "REFLECTIVE_HEURISTIC",
        "claims": ["x", "y"],
        "aid": 7,
    }
    result = await L.promote_one(
        lessons_tbl=tbl,
        tenant_id="t1", user_id="u1",
        dream_row=row, confidence=0.2,
        reinforcement_count=5, distinct_agents=3,
    )
    assert result["promoted"] is False
    assert len(tbl.added) == 0
    assert any("confidence" in r for r in result["reasons"])


async def test_promote_one_idempotent():
    """Already-promoted (promoted_from match in TCMM) → skip."""
    tbl = FakeTable([])
    row = {
        "block_class": "REFLECTIVE_HEURISTIC",
        "claims": ["When X", "Do Y"],
        "aid": 7,
    }
    # Stub the TCMM-archive idempotency check to report "already there".
    with patch("app.proposals.lessons._existing_promoted_from", return_value=True), \
         patch("app.memory.phase_7_writers.promote_lesson_to_team_knowledge") as writer:
        result = await L.promote_one(
            lessons_tbl=tbl,
            tenant_id="t1", user_id="u1",
            dream_row=row, confidence=0.8,
            reinforcement_count=5, distinct_agents=2,
        )
    assert result["promoted"] is False
    assert any("already promoted" in r for r in result["reasons"])
    writer.assert_not_called()


async def test_promote_one_skips_non_heuristic_rows():
    tbl = FakeTable([])
    row = {
        "block_class": "INFORMATION_GAP",
        "claims": ["x"], "aid": 1,
    }
    result = await L.promote_one(
        lessons_tbl=tbl,
        tenant_id="t1", user_id="u1",
        dream_row=row, confidence=0.9,
        reinforcement_count=10, distinct_agents=3,
    )
    assert result["promoted"] is False
    assert "not a reflective_heuristic" in result["reasons"][0]


# ── run_one_cycle ──────────────────────────────────────────────────────


async def test_run_one_cycle_promotes_eligible_and_skips_others():
    dream_rows = [
        # promotable: reflective_heuristic, high density, 2 source agents
        {"block_class": "REFLECTIVE_HEURISTIC",
         "text": "On every onboarding. Run KYC checklist.",
         "claims": ["On every onboarding", "Run KYC checklist"],
         "user_id": "u1", "aid": 1,
         "density_score": 0.8,
         "source_block_ids": [101, 102, 103],
         "extracted_by": "agent:researcher,agent:critic-claim"},
        # NOT promotable: low density (confidence < 0.5)
        {"block_class": "REFLECTIVE_HEURISTIC",
         "text": "Maybe. Probably not.",
         "claims": ["Maybe", "Probably not"],
         "user_id": "u1", "aid": 2,
         "density_score": 0.2,
         "source_block_ids": [201], "extracted_by": "agent:researcher"},
        # NOT promotable: wrong block_class
        {"block_class": "INFORMATION_GAP",
         "text": "POPIA s14 gap", "claims": ["gap"],
         "user_id": "u1", "aid": 3,
         "density_score": 0.9,
         "source_block_ids": [301], "extracted_by": "agent:researcher"},
    ]
    dream_tbl = FakeTable(dream_rows)

    fake_db = MagicMock()
    # Post-M1 cutover: only dream_archive opens via lancedb.connect; lessons
    # land via the Phase 7 split-writer (mocked).
    fake_db.open_table.side_effect = lambda n: {"dream_archive": dream_tbl}.get(n) \
        or (_ for _ in ()).throw(KeyError(n))

    writer_calls: list[dict] = []
    async def _writer_stub(**kw):
        writer_calls.append(kw)
        return True

    with patch("lancedb.connect", return_value=fake_db), \
         patch("app.proposals.lessons._existing_promoted_from", return_value=False), \
         patch("app.memory.phase_7_writers.promote_lesson_to_team_knowledge",
               side_effect=_writer_stub):
        result = await L.run_one_cycle()
    assert result["scanned"] == 3
    assert result["promoted"] == 1
    assert result["skipped"] == 1   # the low-confidence one
    assert len(writer_calls) == 1
    assert writer_calls[0]["trigger"] == "On every onboarding"


async def test_run_one_cycle_amendment_eligible_counted():
    """A row with ≥5 source_blocks + density ≥0.75 + 2 distinct agents
    should be amendment-eligible."""
    dream_rows = [
        {"block_class": "REFLECTIVE_HEURISTIC",
         "text": "Strong pattern. Codify rule.",
         "claims": ["Strong pattern", "Codify rule"],
         "user_id": "u1", "aid": 1,
         "density_score": 0.85,
         "source_block_ids": [101, 102, 103, 104, 105, 106],
         "extracted_by": "agent:researcher,agent:critic-claim"},
    ]
    dream_tbl = FakeTable(dream_rows)
    fake_db = MagicMock()
    fake_db.open_table.side_effect = lambda n: {"dream_archive": dream_tbl}.get(n) \
        or (_ for _ in ()).throw(KeyError(n))

    async def _writer_stub(**kw):
        return True

    with patch("lancedb.connect", return_value=fake_db), \
         patch("app.proposals.lessons._existing_promoted_from", return_value=False), \
         patch("app.memory.phase_7_writers.promote_lesson_to_team_knowledge",
               side_effect=_writer_stub):
        result = await L.run_one_cycle()
    assert result["promoted"] == 1
    assert result["amendment_candidates"] == 1
