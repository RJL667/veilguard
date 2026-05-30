"""Phase 8.0 — Snapshot anchor primitive.

A `SnapshotAnchor` is the content-hash of every input substrate that
fed a Director (or other agent's) decision at time T.  Anchors let the
replay harness say "if all six input hashes match a prior run, any
output difference is LLM non-determinism — not substrate drift", which
is the foundation of runtime replay determinism.

Anchor inputs (per spec 2026-05-27):
  - task_graph_hash         — `sha256(canonical Task graph at T)`
  - approvals_hash          — `sha256(client_tool_approvals at T)`
  - tool_outputs_hash       — `sha256(canonical tool-output record at T)`
  - constitution_version    — int, integer version of CONSTITUTION.md
  - tcmm_snapshot_ref       — opaque pointer (e.g. archive aid range)
  - alignment_weights_ver   — int, alignment_weights schema version
  - dream_graph_signals_ref — opaque pointer (e.g. last dream cycle id)

Plus metadata:
  - anchor_id, decision_point, agent_id, tenant_id, user_id, ts

The composite `anchor_hash` is the hash that locks together all the
inputs so a single equality check tells you "same substrate state".

Storage: anchors live in `agent_tasks.extras_json` (Phase 8.0 minimal
footprint — no new Lance table yet).  When the replay harness lands,
it may promote anchors to a dedicated table.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

logger = logging.getLogger("agent-runtime.replay.snapshot_anchor")


# Version stamp for the anchor schema.  Bumped whenever the input set
# or hash algorithm changes — replay harness uses this to decide
# whether two anchors are comparable at all.
SNAPSHOT_ANCHOR_VERSION = 1


# ── Data class ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SnapshotAnchor:
    """One anchor row.  Immutable by design — anchors are append-only."""

    # Identity / context
    anchor_id:       str
    version:         int
    decision_point:  str        # e.g. 'director.route', 'critic.review_decision'
    agent_id:        str
    tenant_id:       str
    user_id:         str
    ts:              float

    # Input substrate hashes
    task_graph_hash:         str
    approvals_hash:          str
    tool_outputs_hash:       str
    constitution_version:    int
    tcmm_snapshot_ref:       str
    alignment_weights_ver:   int
    dream_graph_signals_ref: str

    # Composite hash — locks all inputs together for the equality check.
    anchor_hash: str

    # Optional context the operator may want at replay time.  Doesn't
    # participate in the anchor hash — purely diagnostic.
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Hashing ──────────────────────────────────────────────────────────────


def _h(data: Any) -> str:
    """Canonical SHA-256 of any JSON-able payload (sorted keys, no
    whitespace) — deterministic across Python versions."""
    if isinstance(data, str):
        return hashlib.sha256(data.encode("utf-8")).hexdigest()
    if isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":"),
                   default=str).encode("utf-8")
    ).hexdigest()


def _composite_hash(
    *,
    task_graph_hash: str,
    approvals_hash: str,
    tool_outputs_hash: str,
    constitution_version: int,
    tcmm_snapshot_ref: str,
    alignment_weights_ver: int,
    dream_graph_signals_ref: str,
    decision_point: str,
    version: int,
) -> str:
    """Build the anchor's composite hash from its inputs.

    Order is fixed and documented — changing it bumps
    SNAPSHOT_ANCHOR_VERSION so old anchors aren't compared against new.
    """
    payload = {
        "_version":                 version,
        "decision_point":           decision_point,
        "task_graph_hash":          task_graph_hash,
        "approvals_hash":           approvals_hash,
        "tool_outputs_hash":        tool_outputs_hash,
        "constitution_version":     int(constitution_version),
        "tcmm_snapshot_ref":        tcmm_snapshot_ref,
        "alignment_weights_ver":    int(alignment_weights_ver),
        "dream_graph_signals_ref":  dream_graph_signals_ref,
    }
    return _h(payload)


# ── Public API ───────────────────────────────────────────────────────────


def compute_anchor(
    *,
    decision_point: str,
    agent_id: str,
    tenant_id: str,
    user_id: str,
    task_graph: Any,
    approvals: Any,
    tool_outputs: Any,
    constitution_version: int = 0,
    tcmm_snapshot_ref: str = "",
    alignment_weights_ver: int = 0,
    dream_graph_signals_ref: str = "",
    ts: Optional[float] = None,
    extras: Optional[dict[str, Any]] = None,
) -> SnapshotAnchor:
    """Pure function — compute the anchor for one decision point.

    Inputs are payload-objects (dicts, lists, primitive types); this
    function hashes each then composes the lock.  Callers extract the
    substrate state however they like (e.g. for `task_graph` they may
    pass the canonical Task row dict; for `approvals` the relevant
    `client_tool_approvals` slice).

    `tcmm_snapshot_ref` and `dream_graph_signals_ref` are passed as
    opaque strings because the full state would be too large to
    inline — caller's responsibility to make them deterministic
    pointers (e.g. "archive[aid<=12345]", "dream_cycle_id=cyc-abc").

    No I/O.  Test-friendly.  Use `record_anchor` to persist.
    """
    if not decision_point:
        raise ValueError("compute_anchor requires non-empty decision_point")
    if not (agent_id and tenant_id and user_id):
        raise ValueError(
            "compute_anchor requires agent_id, tenant_id, user_id"
        )

    task_graph_hash   = _h(task_graph)
    approvals_hash    = _h(approvals)
    tool_outputs_hash = _h(tool_outputs)

    composite = _composite_hash(
        task_graph_hash=task_graph_hash,
        approvals_hash=approvals_hash,
        tool_outputs_hash=tool_outputs_hash,
        constitution_version=constitution_version,
        tcmm_snapshot_ref=tcmm_snapshot_ref,
        alignment_weights_ver=alignment_weights_ver,
        dream_graph_signals_ref=dream_graph_signals_ref,
        decision_point=decision_point,
        version=SNAPSHOT_ANCHOR_VERSION,
    )

    return SnapshotAnchor(
        anchor_id=f"anch-{uuid.uuid4().hex[:12]}",
        version=SNAPSHOT_ANCHOR_VERSION,
        decision_point=decision_point,
        agent_id=agent_id,
        tenant_id=tenant_id,
        user_id=user_id,
        ts=ts if ts is not None else time.time(),
        task_graph_hash=task_graph_hash,
        approvals_hash=approvals_hash,
        tool_outputs_hash=tool_outputs_hash,
        constitution_version=int(constitution_version),
        tcmm_snapshot_ref=str(tcmm_snapshot_ref),
        alignment_weights_ver=int(alignment_weights_ver),
        dream_graph_signals_ref=str(dream_graph_signals_ref),
        anchor_hash=composite,
        extras=dict(extras or {}),
    )


def record_anchor(
    *,
    anchor: SnapshotAnchor,
    task_id: Optional[str] = None,
) -> bool:
    """Persist `anchor` into `agent_tasks.extras_json[snapshot_anchors]`.

    Phase 8.0 stores anchors inline on the relevant Task row so the
    minimal footprint doesn't add a new Lance table or migration.  The
    extras_json grows as `{snapshot_anchors: [...anchor dicts...]}`.
    When the replay harness lands, anchors may migrate to a dedicated
    table — at that point this function becomes a thin write to the
    new table and the per-Task accessor becomes a read-side join.

    Returns True on success, False on write failure (logs the error
    rather than raising — anchors are best-effort observability, never
    block the decision they describe).
    """
    if task_id is None:
        # No-op anchor (e.g. for the Director's pre-task routing
        # decision that has no Task yet) — caller can still hold the
        # anchor in memory for the replay harness to pick up later.
        return True
    try:
        from ..ledger.store import LedgerStore, ns_filter
        from ..ledger import tasks as _tasks
        task = _tasks.get_task(task_id, anchor.tenant_id, anchor.user_id)
        if task is None:
            logger.debug(
                f"[snapshot_anchor] task {task_id} not found; "
                f"anchor {anchor.anchor_id} not persisted"
            )
            return False
        extras_raw = task.get("extras_json") or ""
        try:
            extras = json.loads(extras_raw) if extras_raw else {}
        except (json.JSONDecodeError, TypeError):
            extras = {}
        anchors = list(extras.get("snapshot_anchors") or [])
        # Idempotency — don't duplicate if the same anchor_id is
        # somehow recorded twice.
        if any(a.get("anchor_id") == anchor.anchor_id for a in anchors):
            return True
        anchors.append(anchor.to_dict())
        extras["snapshot_anchors"] = anchors
        tbl = LedgerStore.get().table("agent_tasks")
        tbl.update(
            where=(
                f"{ns_filter(anchor.tenant_id, anchor.user_id)} "
                f"AND id = '{task_id}'"
            ),
            values={
                "extras_json": json.dumps(extras),
                "updated_ts":  time.time(),
            },
        )
        return True
    except Exception as e:
        logger.warning(
            f"[snapshot_anchor] record failed for {anchor.anchor_id}: {e}"
        )
        return False
