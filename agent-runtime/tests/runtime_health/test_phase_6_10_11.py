"""Phase 6.10 (Repository abstraction) + 6.11 (Director interface).

AC-42 — every Lance table access in app/ routes through a Repository
AC-43 — migration-trigger metrics emitted
AC-44 — Director exposes route/synthesize/propose as separate awaitables
AC-45 — per-method telemetry distinct from generic agent_query
"""

from __future__ import annotations

import asyncio
import inspect

from app.storage.repository import (
    REPOSITORY_REGISTRY,
    emit_migration_metrics,
    get_repository,
    should_migrate,
    update_p95_mutation_latency,
    update_row_count,
)


# ── Phase 6.10 ──────────────────────────────────────────────────────────


def test_ac42_all_known_tables_registered():
    """Every ledger table named in PHASE_6_ACCEPTANCE.md has a Repository.

    `org_memory` dropped 2026-05-28 (Phase 7 M1 cutover) — lessons now
    live in TCMM `archive`.  Removed from this expected set.
    """
    expected = {
        "agent_tasks", "task_comments", "task_proposals",
        "proposal_outcomes", "alignment_weights",
        "client_tool_approvals", "client_tool_bypass",
        "tenant_proactive_config", "a2a_external_keys",
        "agent_task_heartbeats",  # Phase 6.3
        # vector-side
        "archive", "dream_archive", "embeddings", "sparse_archive",
    }
    registered = set(REPOSITORY_REGISTRY.keys())
    missing = expected - registered
    assert not missing, f"Repositories missing for: {sorted(missing)}"


def test_ac42_table_kinds_match_spec():
    """The classification of each table follows §9 decision-log."""
    must_be_mutable = {
        "agent_tasks", "task_proposals", "alignment_weights",
        "tenant_proactive_config", "a2a_external_keys",
        # `org_memory` removed from mutable-set 2026-05-28 (M1 cutover).
    }
    must_be_append = {
        "task_comments", "client_tool_approvals", "client_tool_bypass",
        "proposal_outcomes",
    }
    must_be_vector = {"archive", "dream_archive", "embeddings", "sparse_archive"}
    for name in must_be_mutable:
        meta = get_repository(name)
        assert meta is not None and meta.table_kind == "mutable_transactional", (
            f"{name} should be mutable_transactional, got "
            f"{meta.table_kind if meta else None}"
        )
    for name in must_be_append:
        meta = get_repository(name)
        assert meta is not None and meta.table_kind == "append_analytical"
    for name in must_be_vector:
        meta = get_repository(name)
        assert meta is not None and meta.table_kind == "vector"


def test_ac43_migration_metrics_emit_keys_per_table():
    """emit_migration_metrics returns a per-table dict with the expected keys."""
    snap = emit_migration_metrics()
    assert "agent_tasks" in snap
    keys = set(snap["agent_tasks"].keys())
    expected_keys = {
        "row_count", "p95_mutation_latency_ms",
        "last_sampled_ts", "n_cross_row_txn_needed",
    }
    assert expected_keys.issubset(keys)


def test_ac43_should_migrate_fires_on_row_count():
    """>500K rows on a mutable_transactional table triggers migration."""
    update_row_count("agent_tasks", 1_000_000)
    should, reason = should_migrate("agent_tasks")
    assert should is True
    assert "row_count" in reason
    # Reset to avoid bleed into other tests.
    update_row_count("agent_tasks", 0)


def test_ac43_should_migrate_fires_on_latency():
    """>100ms p95 mutation latency triggers migration."""
    update_p95_mutation_latency("agent_tasks", 250.0)
    should, reason = should_migrate("agent_tasks")
    assert should is True
    assert "latency" in reason
    update_p95_mutation_latency("agent_tasks", 0.0)


def test_ac43_vector_tables_never_migrate_to_postgres():
    """Vector tables stay on Lance regardless of size."""
    update_row_count("archive", 10_000_000)
    should, reason = should_migrate("archive")
    assert should is False
    assert "vector" in reason or "Lance" in reason
    update_row_count("archive", 0)


# ── Phase 6.11 ──────────────────────────────────────────────────────────


def test_ac44_director_exposes_route_synthesize_propose_methods():
    """Static introspection: DirectorAgent has the three methods."""
    from agent.director import DirectorAgent
    for method in ("route", "synthesize", "propose"):
        m = getattr(DirectorAgent, method, None)
        assert m is not None, f"DirectorAgent.{method} not defined"
        assert inspect.iscoroutinefunction(m), (
            f"DirectorAgent.{method} must be async (`route() / synthesize() / "
            f"propose()` are awaitables per Phase 6.11 interface)"
        )


def test_ac45_director_methods_emit_per_method_latency():
    """Each method records its latency into DIRECTOR_METHOD_LATENCY_MS."""
    from agent.director import (
        DIRECTOR_METHOD_LATENCY_MS,
        DirectorAgent,
        director_p95_ms,
    )

    # Reset state for the test.
    for buf in DIRECTOR_METHOD_LATENCY_MS.values():
        buf.clear()

    # Construct a Director with the minimum persona surface.  The
    # methods themselves are async, so drive them through an event loop.
    class _MinimalPersona:
        agent_id = "director-test"
        system_prompt = ""
        model_for = staticmethod(lambda kind: "claude-sonnet-4-5")
    agent = DirectorAgent.__new__(DirectorAgent)
    agent.persona = _MinimalPersona()  # type: ignore[attr-defined]

    async def drive():
        await agent.route({"brief": "test"})
        await agent.synthesize("task-1", [{"output": "x"}])
        await agent.propose([{"id": "p-1"}, {"id": "p-2"}])
    asyncio.run(drive())

    # Each method got recorded.
    for method in ("route", "synthesize", "propose"):
        buf = DIRECTOR_METHOD_LATENCY_MS[method]
        assert len(buf) >= 1, f"{method} latency not recorded"
        # All values are non-negative.
        assert all(v >= 0.0 for v in buf)
        # p95 helper works.
        p95 = director_p95_ms(method)
        assert p95 is not None
        assert p95 >= 0.0


def test_ac45_method_latencies_are_distinct_buckets():
    """The three methods record into separate buckets — Phase 8 split
    trigger ('Director p95 > 2× slowest specialist') reads each
    independently."""
    from agent.director import DIRECTOR_METHOD_LATENCY_MS

    expected = {"route", "synthesize", "propose"}
    assert set(DIRECTOR_METHOD_LATENCY_MS.keys()) == expected
