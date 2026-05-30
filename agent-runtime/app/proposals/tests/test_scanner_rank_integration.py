"""Integration test: DreamScanner top-N / rank-pass slicing.

Validates that ranks 1..deterministic_top_n use template defaults
and ranks N+1..max go through the rank pass.  Doesn't touch Lance
or Anthropic — patches both layers.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch, MagicMock

import pytest

from app.proposals.dream_scanner import DreamScanner


CONSTITUTION = {
    "objectives": [
        {"id": "reduce_toil",          "weight": 0.4},
        {"id": "improve_security",     "weight": 0.3},
        {"id": "preserve_user_agency", "weight": 0.3},
    ],
    "constraints": [],
}


def _build_dream_row(*, aid: int, block_class: str, source_block_ids: list[int],
                    text: str = "test", topics=None, claims=None,
                    heat: float = 0.5, density: float = 0.5):
    """Minimal dream_archive row dict that exercises the scanner."""
    return {
        "aid":              aid,
        "block_class":      block_class,
        "user_id":          "tenant-x",
        "text":             text,
        "topics":           topics or [],
        "claims":           claims or [],
        "source_block_ids": source_block_ids,
        "origin":           "INFERRED",
        "heat":             heat,
        "density_score":    density,
        "timestamp":        1779800000.0,
    }


class _FakeLanceColumn:
    def __init__(self, values):
        self._v = values
    def __getitem__(self, i):
        return _FakeLanceCell(self._v[i])

class _FakeLanceCell:
    def __init__(self, v):
        self._v = v
    def as_py(self):
        return self._v

class _FakeLanceArrow:
    """Minimal pyarrow-like object the scanner reads from."""
    def __init__(self, rows: list[dict]):
        self._rows = rows
        self.num_rows = len(rows)
        keys: set = set()
        for r in rows:
            keys.update(r.keys())
        self.column_names = sorted(keys)
    def column(self, name):
        return _FakeLanceColumn([r.get(name) for r in self._rows])


@pytest.mark.asyncio
async def test_rank_pass_only_applies_to_ranks_above_top_n():
    """5 candidates, top_n=2 → ranks 3,4,5 should land in rank_pass input."""
    # Build 5 candidates so top-2 use templates, last 3 go through Haiku
    rows = [
        _build_dream_row(aid=100+i, block_class="INFORMATION_GAP",
                         source_block_ids=[i],
                         topics=[f"topic-{i}"],
                         heat=1.0 - i*0.1, density=0.3)
        for i in range(5)
    ]

    captured_rank_inputs = []

    async def fake_rank(*, candidates, constitution_objectives, model):
        # Capture what got sent to the rank pass
        captured_rank_inputs.append(candidates)
        # Return refined versions of each
        return [
            {
                "signal_node_ids":             c["signal_node_ids"],
                "refined_brief":               f"LLM-refined: {c['default_brief'][:30]}",
                "refined_assignee":            "critic-prose",
                "refined_objective_alignment": 0.7,
                "rationale":                   "ranked by Haiku",
                "drop":                        False,
            } for c in candidates
        ]

    created_proposals = []

    def fake_create_proposal(**kwargs):
        created_proposals.append(kwargs)
        return f"prop-{kwargs['signal_type']}-{len(created_proposals)}"

    fake_table = MagicMock()
    fake_table.search.return_value.limit.return_value.to_arrow.return_value = _FakeLanceArrow(rows)

    fake_db = MagicMock()
    fake_db.open_table.return_value = fake_table

    with patch("app.proposals.dream_scanner._props.create_proposal", side_effect=fake_create_proposal), \
         patch("lancedb.connect", return_value=fake_db), \
         patch("app.proposals.dream_scanner.rank_candidates", side_effect=fake_rank) \
            if False else patch.dict("os.environ", {}):
        # Use a direct rank_candidates import-path patch on the module
        scanner = DreamScanner(
            deterministic_top_n=2,
            max_per_cycle=10,
            per_signal_cap=10,   # disable per-signal cap so all 5 emit
            rank_pass_enabled=True,
            constitution=CONSTITUTION,
        )
        # Patch the rank function the scanner imports inline
        from app.proposals import dream_scanner as _ds
        # Monkey-patch the rank_candidates symbol resolution by
        # injecting at the module level used in _scan_once.
        with patch.object(_ds, "rank_candidates", side_effect=fake_rank, create=True):
            # Also need to patch the deferred import inside _scan_once
            import app.proposals.rank as _rank_mod
            with patch.object(_rank_mod, "rank_candidates", side_effect=fake_rank):
                result = await scanner._scan_once()

    # 5 candidates total, top-2 deterministic, ranks 3-5 (3 candidates) → rank pass
    assert result["candidates"] == 5
    assert result["rank_passed"] == 3
    assert len(captured_rank_inputs) == 1
    assert len(captured_rank_inputs[0]) == 3, "ranks 3-5 should hit rank pass"
    # 5 proposals created (top-2 + rank-passed 3)
    assert len(created_proposals) == 5
    # Top-2 use template assignee (researcher for info_gap)
    top_two = created_proposals[:2]
    for p in top_two:
        assert p["proposed_assignee"] == "researcher", \
            "top-N should use default assignee"
        assert not p["proposed_brief"].startswith("LLM-refined:"), \
            "top-N should use template brief"
    # Bottom-3 should use rank-pass refined assignee (critic-prose) and brief
    bottom_three = created_proposals[2:]
    for p in bottom_three:
        assert p["proposed_assignee"] == "critic-prose", \
            "rank-passed candidates should use refined assignee"
        assert p["proposed_brief"].startswith("LLM-refined:"), \
            "rank-passed candidates should use refined brief"


@pytest.mark.asyncio
async def test_rank_pass_disabled_uses_all_defaults():
    """Default config: rank pass off → ALL candidates use template defaults."""
    rows = [
        _build_dream_row(aid=100+i, block_class="INFORMATION_GAP",
                         source_block_ids=[i],
                         topics=[f"topic-{i}"], heat=0.5, density=0.3)
        for i in range(5)
    ]

    rank_calls = []
    async def fake_rank(**kwargs):
        rank_calls.append(kwargs)
        return []

    created_proposals = []
    def fake_create_proposal(**kwargs):
        created_proposals.append(kwargs)
        return "prop-x"

    fake_table = MagicMock()
    fake_table.search.return_value.limit.return_value.to_arrow.return_value = _FakeLanceArrow(rows)
    fake_db = MagicMock()
    fake_db.open_table.return_value = fake_table

    with patch("app.proposals.dream_scanner._props.create_proposal", side_effect=fake_create_proposal), \
         patch("lancedb.connect", return_value=fake_db), \
         patch.dict("os.environ", {}, clear=False):
        import app.proposals.rank as _rank_mod
        with patch.object(_rank_mod, "rank_candidates", side_effect=fake_rank):
            scanner = DreamScanner(
                deterministic_top_n=2,
                max_per_cycle=10,
                per_signal_cap=10,
                rank_pass_enabled=False,  # explicit off
                constitution=CONSTITUTION,
            )
            result = await scanner._scan_once()

    assert result["rank_passed"] == 0
    assert len(rank_calls) == 0, "rank pass should NOT have been invoked"
    # All 5 still emitted using defaults
    assert len(created_proposals) == 5
    for p in created_proposals:
        assert p["proposed_assignee"] == "researcher"


@pytest.mark.asyncio
async def test_rank_pass_drop_skips_emission():
    """drop=True from LLM → proposal NOT emitted (deferred per §3.7.4)."""
    rows = [
        _build_dream_row(aid=200, block_class="INFORMATION_GAP",
                         source_block_ids=[1], topics=["x"], heat=0.5),
        _build_dream_row(aid=201, block_class="INFORMATION_GAP",
                         source_block_ids=[2], topics=["y"], heat=0.4),
    ]

    async def fake_rank(*, candidates, constitution_objectives, model):
        # Drop both (they're below the top-1 cutoff anyway)
        return [{
            "signal_node_ids":             c["signal_node_ids"],
            "refined_brief":               "x",
            "refined_assignee":            "researcher",
            "refined_objective_alignment": 0.1,
            "rationale":                   "below cutoff",
            "drop":                        True,
        } for c in candidates]

    created = []
    def fake_create_proposal(**kwargs):
        created.append(kwargs)
        return "prop-x"

    fake_table = MagicMock()
    fake_table.search.return_value.limit.return_value.to_arrow.return_value = _FakeLanceArrow(rows)
    fake_db = MagicMock()
    fake_db.open_table.return_value = fake_table

    with patch("app.proposals.dream_scanner._props.create_proposal", side_effect=fake_create_proposal), \
         patch("lancedb.connect", return_value=fake_db):
        import app.proposals.rank as _rank_mod
        with patch.object(_rank_mod, "rank_candidates", side_effect=fake_rank):
            scanner = DreamScanner(
                deterministic_top_n=1,    # top-1 deterministic
                max_per_cycle=10,
                rank_pass_enabled=True,
                constitution=CONSTITUTION,
            )
            result = await scanner._scan_once()

    # Top-1 emits (uses defaults), rank-1 dropped
    assert result["emitted"] == 1
    assert result["rank_passed"] == 1   # the rank pass DID run on 1 candidate
    assert len(created) == 1, "drop=True should skip create_proposal"
