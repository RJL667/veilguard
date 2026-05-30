"""Phase 8 — Runtime replay determinism.

Spec hard gate (decision log 2026-05-27): before any feature that
mutates system behavior over time ships beyond Phase 6.9, runtime
replay must produce deterministic reproduction of at least one full
decision trace (proposal → critic decision → outcome → recalibration
delta).

This package is the implementation surface.  Phase 8.0 (2026-05-28)
landed the **snapshot anchor primitive** — the content-hash of every
input substrate at a Director decision point.  Anchors are the
load-bearing data structure: they let the future re-execution harness
detect when the *inputs* match a prior run, so any output divergence
is genuine LLM non-determinism rather than substrate drift.

Phase 8.0 scope:
  - SnapshotAnchor dataclass + hash computation
  - compute_anchor(...) — pure function over input substrate refs
  - record_anchor(...) — append anchor row to ledger
  - Tests for hash stability + sensitivity

Deferred to focused follow-up session:
  - Replay harness (frozen-snapshot reconstruction)
  - Reasoning-envelope comparison (token-set / claim-set / structural)
  - Per-anchor test scaffolding (1 week per spec)
"""

from .snapshot_anchor import (
    SnapshotAnchor,
    compute_anchor,
    record_anchor,
    SNAPSHOT_ANCHOR_VERSION,
)

__all__ = [
    "SnapshotAnchor",
    "compute_anchor",
    "record_anchor",
    "SNAPSHOT_ANCHOR_VERSION",
]
