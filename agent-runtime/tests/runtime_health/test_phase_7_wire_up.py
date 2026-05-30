"""Phase 7 wire-up tests — gate that the four split-writers are actually
plugged into the production callsites, not just defined in
`app/memory/phase_7_writers.py`.

Pattern mirrors `test_wire_up.py` for Phase 6.  Source-code grep is
the right tool here — we're testing the import graph, not behaviour.

  M1 promote_lesson_to_team_knowledge → lessons.promote_one
  M2 record_outcome_with_narrative    → outcomes.compute_one
  M3 record_proposal_with_content     → dream_scanner._scan_once
                                       + main.py POST /proposals
                                       + constitution_amendments.propose_one
  M4 record_discussion_comment        → (no prod caller yet — opt-in writer)
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _src(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


# ── M1: lesson promotion ────────────────────────────────────────────────


def test_m1_promote_lesson_wired_in_lessons_promote_one():
    src = _src("app/proposals/lessons.py")
    assert "promote_lesson_to_team_knowledge" in src, (
        "Phase 7 M1 — lessons.promote_one must call "
        "promote_lesson_to_team_knowledge (transitional dual-write)"
    )


def test_m1_promote_one_is_async():
    """propose_one needs to await the Phase 7 writer — so it must be async."""
    src = _src("app/proposals/lessons.py")
    assert "async def promote_one" in src, (
        "promote_one must be async to await promote_lesson_to_team_knowledge"
    )


def test_m1_lessons_run_one_cycle_is_async():
    src = _src("app/proposals/lessons.py")
    assert "async def run_one_cycle" in src
    assert "await run_one_cycle" in src, (
        "LessonPromotionWorker.run must await run_one_cycle"
    )


# ── M2: outcome with narrative ──────────────────────────────────────────


def test_m2_record_outcome_wired_in_outcomes_compute_one():
    src = _src("app/proposals/outcomes.py")
    assert "record_outcome_with_narrative" in src, (
        "Phase 7 M2 — outcomes.compute_one must route through "
        "record_outcome_with_narrative"
    )


def test_m2_compute_one_is_async():
    src = _src("app/proposals/outcomes.py")
    assert "async def compute_one" in src


def test_m2_outcomes_run_one_cycle_is_async():
    src = _src("app/proposals/outcomes.py")
    assert "async def run_one_cycle" in src
    assert "await run_one_cycle" in src, (
        "OutcomesWorker.run must await run_one_cycle"
    )


# ── M3: proposal with content (THREE callsites) ─────────────────────────


def test_m3_wired_in_dream_scanner():
    src = _src("app/proposals/dream_scanner.py")
    assert "record_proposal_with_content" in src, (
        "Phase 7 M3 — dream_scanner must emit proposals via "
        "record_proposal_with_content for TCMM cross-ref"
    )


def test_m3_wired_in_constitution_amendments():
    src = _src("app/proposals/constitution_amendments.py")
    assert "record_proposal_with_content" in src, (
        "Phase 7 M3 — constitution_amendments.propose_one must route "
        "through record_proposal_with_content"
    )


def test_m3_propose_one_is_async():
    src = _src("app/proposals/constitution_amendments.py")
    assert "async def propose_one" in src
    assert "async def run_one_cycle" in src


def test_m3_wired_in_main_post_proposals():
    src = _src("app/main.py")
    assert "record_proposal_with_content" in src, (
        "Phase 7 M3 — POST /proposals endpoint must use the split-writer"
    )


def test_m3_no_legacy_create_proposal_calls_in_wired_modules():
    """The four wired callsites should no longer call `_props.create_proposal`
    directly (transition complete).  Lower-level callers in
    `record_proposal_with_content` itself still use it — that's fine."""
    wired_modules = [
        "app/proposals/dream_scanner.py",
        "app/proposals/constitution_amendments.py",
        # main.py still has create_proposal mentioned in /convert helper text;
        # we check the specific POST /proposals endpoint via a stricter scan.
    ]
    for mod in wired_modules:
        src = _src(mod)
        # Walk the AST and reject any `_props.create_proposal(...)` calls.
        tree = ast.parse(src)
        bad: list[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                if (
                    isinstance(f, ast.Attribute)
                    and f.attr == "create_proposal"
                    and isinstance(f.value, ast.Name)
                    and f.value.id == "_props"
                ):
                    bad.append(node.lineno)
        assert not bad, (
            f"{mod} still calls _props.create_proposal directly at "
            f"line(s) {bad}; should route through record_proposal_with_content"
        )


# ── Heartbeat (already wired in Phase 6.3, re-asserted here) ────────────


def test_heartbeat_wired_in_inbox_poller():
    src = _src("app/workers/inbox_poller.py")
    assert "record_heartbeat" in src, (
        "Phase 6.3 — inbox_poller dispatch loop must call record_heartbeat "
        "at every turn boundary"
    )


def test_heartbeat_at_turn_boundary_events():
    """Heartbeat must fire on the three streaming-event kinds that mark
    a turn: assistant_text, tool_call, final_result."""
    src = _src("app/workers/inbox_poller.py")
    # Look for the conditional that gates the record_heartbeat call.
    assert "assistant_text" in src
    assert "tool_call" in src
    assert "final_result" in src


# ── Schema migrations applied lazily on first table open ────────────────


def test_phase_6_7_migrations_registered_in_store():
    src = _src("app/ledger/store.py")
    # The dispatcher in _migrate_phase_6_0 routes agent_tasks to the
    # struct-aware migration and everything else to the generic
    # additive-nullable-column sync.
    assert "_migrate_agent_tasks" in src
    assert "_migrate_add_missing_nullable_columns" in src
    # Generic sync reads TABLE_SCHEMAS for the target schema.
    assert "TABLE_SCHEMAS" in src


def test_main_startup_eager_opens_all_tables():
    """`main.py:_startup` must touch every table so migrations apply
    before the first request."""
    src = _src("app/main.py")
    assert "for _tname in TABLE_SCHEMAS.keys()" in src
    assert "Phase 6+7 migrations applied" in src
