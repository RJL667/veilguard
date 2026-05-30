"""Phase 6.10 — Repository abstraction with table tagging + migration triggers.

Each ledger table is wrapped in a Repository that exposes a small,
backend-agnostic surface (`get`, `upsert`, `delete`, `query`, `count`).
Today every Repository's `backend` is `lance`.  When trigger conditions
fire (Prometheus metrics in `emit_migration_metrics`), `mutable_transactional`
tables swap to `postgres` while the rest stay on Lance.

Migration triggers (instrumented in `emit_migration_metrics`):
  - tasks row count > 500K
  - agent_tasks p95 mutation latency > 100ms
  - any operation requires cross-row transactional guarantee

This module is intentionally light — Phase 6.10 ships the protocol,
the registry, and the metric emission.  Actual swap-to-Postgres logic
arrives in Phase 7+ when a trigger fires.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

logger = logging.getLogger("agent-runtime.storage.repository")


TableKind = Literal["mutable_transactional", "append_analytical", "vector"]
Backend = Literal["lance", "postgres", "lance+postgres"]


@dataclass(frozen=True)
class RepositoryMeta:
    """Static metadata for a Repository.  Drives migration decisions."""
    table_name: str
    table_kind: TableKind
    backend:    Backend = "lance"


@dataclass
class _MigrationMetric:
    """Per-table runtime metric used to decide migration triggers."""
    row_count:               int = 0
    p95_mutation_latency_ms: float = 0.0
    last_sampled_ts:         float = 0.0
    n_cross_row_txn_needed:  int = 0


class Repository(Protocol):
    """Marker protocol — concrete Repositories implement the methods
    their callers need.  Kept light: Phase 6.10 doesn't force a full
    CRUD interface because the existing ledger modules already provide
    one; this protocol is the migration-readiness gate."""

    meta: RepositoryMeta


# Registry: table_name → meta.  Concrete Repository subclasses (or
# functional callers) register here so `emit_migration_metrics()` can
# walk every known table.
REPOSITORY_REGISTRY: dict[str, RepositoryMeta] = {}
_METRICS: dict[str, _MigrationMetric] = {}


def register_repository(meta: RepositoryMeta) -> None:
    if meta.table_name in REPOSITORY_REGISTRY:
        existing = REPOSITORY_REGISTRY[meta.table_name]
        if existing != meta:
            raise ValueError(
                f"repository {meta.table_name!r} already registered with "
                f"different meta: {existing} vs {meta}"
            )
        return
    REPOSITORY_REGISTRY[meta.table_name] = meta
    _METRICS[meta.table_name] = _MigrationMetric()
    logger.debug(f"[storage] registered repository: {meta}")


def get_repository(table_name: str) -> RepositoryMeta | None:
    return REPOSITORY_REGISTRY.get(table_name)


# ── Initial registrations ──────────────────────────────────────────────


# These mirror the table classifications in §9 decision-log "LanceDB
# acknowledged-eventually-wrong" + Phase 6.10 + Phase 7 migrations.
_INITIAL: list[RepositoryMeta] = [
    # mutable_transactional — Postgres migration candidates
    RepositoryMeta("agent_tasks",            "mutable_transactional"),
    RepositoryMeta("task_proposals",         "mutable_transactional"),
    RepositoryMeta("alignment_weights",      "mutable_transactional"),
    RepositoryMeta("tenant_proactive_config","mutable_transactional"),
    RepositoryMeta("a2a_external_keys",      "mutable_transactional"),
    # append_analytical — stays on Lance
    RepositoryMeta("task_comments",          "append_analytical"),
    RepositoryMeta("client_tool_approvals",  "append_analytical"),
    RepositoryMeta("client_tool_bypass",     "append_analytical"),
    RepositoryMeta("proposal_outcomes",      "append_analytical"),
    # `org_memory` removed 2026-05-28 (Phase 7 M1 cutover).  Lessons
    # now live in TCMM `archive`; no Repository-tagged ledger table.
    # Phase 6.3 — high-frequency heartbeat writes; append_analytical
    # (we read by "most recent per task_id" not by exact key).
    RepositoryMeta("agent_task_heartbeats",  "append_analytical"),
    # vector — stays on Lance, but listed for completeness
    RepositoryMeta("archive",                "vector"),
    RepositoryMeta("dream_archive",          "vector"),
    RepositoryMeta("embeddings",             "vector"),
    RepositoryMeta("sparse_archive",         "vector"),
]
for _m in _INITIAL:
    register_repository(_m)


# ── Migration metric emission ──────────────────────────────────────────


def update_row_count(table_name: str, row_count: int) -> None:
    """Called from a sampler (e.g. once per minute) to refresh row counts."""
    m = _METRICS.get(table_name)
    if m is None:
        return
    m.row_count = row_count
    m.last_sampled_ts = time.time()


def update_p95_mutation_latency(table_name: str, p95_ms: float) -> None:
    """Called from the mutation hot-path (sampled, not per-mutation)."""
    m = _METRICS.get(table_name)
    if m is None:
        return
    m.p95_mutation_latency_ms = p95_ms
    m.last_sampled_ts = time.time()


def emit_migration_metrics() -> dict[str, dict[str, float]]:
    """Return a snapshot of per-table metrics for Prometheus-style scrape.

    Caller (e.g. /metrics endpoint) iterates the dict and emits
    `repo.<table>.row_count` + `repo.<table>.p95_mutation_latency_ms`.

    Phase 6.10 ships emission; the actual triggers (compare row_count
    vs 500K, latency vs 100ms) fire in Phase 7's migration decision
    module — not here.
    """
    out: dict[str, dict[str, float]] = {}
    for name, m in _METRICS.items():
        out[name] = {
            "row_count":               float(m.row_count),
            "p95_mutation_latency_ms": m.p95_mutation_latency_ms,
            "last_sampled_ts":         m.last_sampled_ts,
            "n_cross_row_txn_needed":  float(m.n_cross_row_txn_needed),
        }
    return out


def should_migrate(table_name: str) -> tuple[bool, str]:
    """Check trigger conditions; return (should_migrate, reason).

    Phase 6.10 — purely advisory.  Phase 7 hooks this into the
    migration decision module to actually flip the backend.
    """
    meta = REPOSITORY_REGISTRY.get(table_name)
    m = _METRICS.get(table_name)
    if meta is None or m is None:
        return (False, "unknown table")
    if meta.table_kind != "mutable_transactional":
        return (False, f"table_kind={meta.table_kind} — Lance is the right home")
    if m.row_count > 500_000:
        return (True, f"row_count={m.row_count} > 500K threshold")
    if m.p95_mutation_latency_ms > 100.0:
        return (True, f"p95_mutation_latency_ms={m.p95_mutation_latency_ms:.1f} > 100ms threshold")
    if m.n_cross_row_txn_needed > 0:
        return (True, f"cross_row_txn_needed={m.n_cross_row_txn_needed} (any > 0 triggers)")
    return (False, "under all triggers")
