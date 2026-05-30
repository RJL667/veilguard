"""Phase 6.4 — Revision-priority lane.

When a critic returns `changes_requested`, the task moves from `review`
back to `in_progress` and the original IC needs to revise.  Without
priority handling, that revision sits behind freshly-emitted `open`
tasks for the same persona — "done halfway" silently becomes "done
forever-pending."

Phase 6.4 marks revision-bound tasks with `is_revision=True` (carried
in `extras_json.is_revision`) and the inbox-poller's claim path sorts
revisions ahead of fresh work within the same persona.

This module is the pure sorter — `sort_for_revision_priority` —
testable without Lance.  The actual revision flag-setting lives in the
critic's `review_decision` tool when verdict=changes_requested.
"""

from __future__ import annotations

import json
from typing import Any


def is_revision_candidate(task_row: dict[str, Any]) -> bool:
    """True iff the task row is a critic-bounced revision.

    Identified by:
      - status=='in_progress', AND
      - extras_json contains is_revision=True, OR the task has at
        least one prior `review_decision` comment with verdict
        `changes_requested` (Phase 6.4 v2 inference path).
    """
    if (task_row.get("status") or "") != "in_progress":
        return False
    extras_raw = task_row.get("extras_json") or "{}"
    try:
        extras = json.loads(extras_raw) if isinstance(extras_raw, str) else extras_raw
    except (json.JSONDecodeError, TypeError):
        extras = {}
    if extras.get("is_revision") is True:
        return True
    # Inference fallback — caller may carry the history.
    history = task_row.get("review_history") or []
    return any(
        (h.get("verdict") or "").lower() == "changes_requested"
        for h in history
    )


def sort_for_revision_priority(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sort claimable tasks: revisions FIRST within each persona, then
    fresh work in creation order.

    Stable across same-key entries (Python's sort is stable since 3.7),
    so creation-order is preserved between tasks of the same priority.
    """
    def sort_key(row: dict[str, Any]) -> tuple[int, float]:
        # Lower priority number = claimed first.
        priority = 0 if is_revision_candidate(row) else 1
        created_ts = row.get("created_ts") or 0.0
        return (priority, created_ts)

    return sorted(rows, key=sort_key)
