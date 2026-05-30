"""Phase 6.10 — Repository abstraction for eventual Postgres migration.

Wraps each Lance table with a typed Repository.  Today all backends
return Lance.  When migration triggers fire (>500K rows, p95 mutation
>100ms, cross-row transaction needed), `mutable_transactional` tables
swap to PostgreSQL while `append_analytical` + `vector` stay on Lance.

The Repository protocol exists so the swap is a wiring change, not a
rewrite.  See `MULTI_AGENT_PLATFORM.md` §9 decision-log "Phase 6.10
Repository abstraction" for the trigger conditions.
"""

from .repository import (
    Repository,
    REPOSITORY_REGISTRY,
    TableKind,
    Backend,
    register_repository,
    get_repository,
    emit_migration_metrics,
)

__all__ = [
    "Repository",
    "REPOSITORY_REGISTRY",
    "TableKind",
    "Backend",
    "register_repository",
    "get_repository",
    "emit_migration_metrics",
]
