"""Phase 4 — constitution amendment writer (§3.10).

Spec: "Constitution-amendment proposals — high-confidence,
multiply-reinforced lessons (`reinforcement_count >= 5 AND
confidence >= 0.75`) become eligible for promotion to constitution
amendment, surfaced in the same review queue with distinct visual
treatment."

This module:
  1. Scans `org_memory` for lessons that meet amendment thresholds
     AND haven't already been proposed as an amendment.
  2. For each, drafts a constitution-edit proposal in a structured
     shape the user can review + apply manually.
  3. Writes one row per amendment to `task_proposals.lance` with
     signal_type='constitution_amendment' so the proposals queue
     surfaces it alongside other proposals.

Idempotent: lessons that already have a `task_proposals` row with
`signal_type=constitution_amendment` AND `rationale` containing
the lesson_id are skipped.

The actual constitution file edit is NEVER automatic — it's a
user-gated decision because the constitution is the steering wheel.
This module surfaces the AMENDMENT PROPOSAL; the user clicks
Approve to convert into a Task that the user (or critic-prose)
executes manually.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Optional

from ..ledger.store import LedgerStore
from ..ledger import proposals as _props
from .lessons import (
    CONST_AMENDMENT_CONFIDENCE_MIN,
    CONST_AMENDMENT_REINFORCEMENT_MIN,
)

logger = logging.getLogger(__name__)


SIGNAL_CONSTITUTION_AMENDMENT = "constitution_amendment"


def _existing_amendment_proposal(
    *,
    tenant_id: str,
    user_id: str,
    lesson_id: str,
) -> Optional[str]:
    """Returns existing proposal_id for this lesson_id or None."""
    try:
        tbl = LedgerStore.get().table("task_proposals")
        arr = (
            tbl.search()
            .where(
                f"tenant_id = '{tenant_id}' AND user_id = '{user_id}' "
                f"AND signal_type = '{SIGNAL_CONSTITUTION_AMENDMENT}'"
            )
            .select(["id", "rationale", "status"])
            .limit(500)
            .to_arrow()
        )
        for i in range(arr.num_rows):
            rationale = (arr.column("rationale")[i].as_py() or "")
            if lesson_id in rationale:
                return arr.column("id")[i].as_py()
    except Exception:
        pass
    return None


def draft_amendment_brief(lesson: dict[str, Any]) -> dict[str, str]:
    """Compose a constitution-amendment brief from a lesson row.

    The brief is the operational diff the user reviews.  We don't
    invent constitution objective/constraint keys — we propose a
    structured suggestion the user can adapt.
    """
    trigger = (lesson.get("trigger") or "").strip()
    rule = (lesson.get("rule") or "").strip()
    conf = float(lesson.get("confidence") or 0.0)
    reinf = int(lesson.get("reinforcement_count") or 0)
    return {
        "title": (
            f"Constitution amendment from lesson "
            f"({reinf}× reinforced, conf={conf:.2f})"
        ),
        "brief": (
            f"Propose adding to CONSTITUTION.md as either an "
            f"objective tweak or a new constraint:\n\n"
            f"  - Trigger:  {trigger}\n"
            f"  - Rule:     {rule}\n\n"
            f"Evidence: lesson reinforced {reinf} times across distinct "
            f"agents; confidence {conf:.2f} (≥ "
            f"{CONST_AMENDMENT_CONFIDENCE_MIN:.2f} threshold).\n\n"
            f"User decides: (a) amend objectives (re-weight existing or "
            f"add new); (b) add a constraint rule; or (c) leave as "
            f"org_memory lesson only.  This proposal is the surface "
            f"point — no constitution file is edited automatically."
        ),
        "spec": (
            "Output: a unified-diff against agents/CONSTITUTION.md "
            "(or a written rationale for why no edit is needed).  "
            "User-applied edit only; this Task is NOT for the agent "
            "to edit the constitution autonomously."
        ),
    }


async def propose_one(*, lesson: dict[str, Any]) -> Optional[str]:
    """Emit a single constitution-amendment proposal for ``lesson``.

    Returns the new proposal_id, or None if skipped (idempotency,
    missing fields, etc.).

    Phase 7 M3 — routes through `record_proposal_with_content` so the
    drafted brief lives in TCMM and the ledger row carries the cross-ref.
    """
    tenant_id = lesson.get("tenant_id") or ""
    user_id = lesson.get("user_id") or ""
    lesson_id = lesson.get("id") or ""
    if not (tenant_id and user_id and lesson_id):
        return None
    if _existing_amendment_proposal(
        tenant_id=tenant_id, user_id=user_id, lesson_id=lesson_id,
    ):
        return None
    drafted = draft_amendment_brief(lesson)
    from ..memory.phase_7_writers import record_proposal_with_content
    pid, _tcmm_obs_id = await record_proposal_with_content(
        tenant_id=tenant_id,
        user_id=user_id,
        signal_type=SIGNAL_CONSTITUTION_AMENDMENT,
        signal_node_ids=[],
        impact_score=10.0,   # always high — these are rare + load-bearing
        proposed_brief=drafted["brief"],
        # [F15_2026_06_02] A producer writes the amendment draft/diff; critics
        # have no write tools (§4.4) so a critic-owned amendment task dead-ends.
        # The Critic reviews the draft at submit_for_review time.
        proposed_assignee="researcher",
        proposed_deliverable_spec=drafted["spec"],
        rationale=(
            f"constitution_amendment from lesson {lesson_id} "
            f"(conf={lesson.get('confidence', 0):.2f}, "
            f"reinf={lesson.get('reinforcement_count', 0)})"
        ),
        objective_alignment=None,
        constraint_violations=[],
    )
    return pid


async def run_one_cycle(
    *,
    db_path: str = "/tcmm-data/veilguard/tcmm.db",
) -> dict[str, Any]:
    """Scan TCMM-archive lessons for amendment-eligible candidates; emit proposals.

    Post-M1 cutover: lessons live in TCMM `archive`, not the dropped
    org_memory Lance table.  We enumerate active lessons across every
    tenant that has one (tenant discovery by user_id column) and
    filter to those above the amendment thresholds.
    """
    try:
        import lancedb
        from ..ledger.store import open_ledger_db; db = open_ledger_db(db_path)
        a = db.open_table("archive")
    except Exception as e:
        return {"scanned": 0, "proposed": 0, "error": f"open: {e}"}

    # Discover tenant set by scanning lesson rows.
    try:
        arr = a.search().limit(1_000_000).to_arrow().to_pylist()
    except Exception as e:
        return {"scanned": 0, "proposed": 0, "error": f"scan: {e}"}
    tenants = sorted({
        (r.get("user_id") or "")
        for r in arr
        if (r.get("text") or "").lstrip().startswith("[lesson]")
        and (r.get("user_id") or "")
    })

    from ..memory.lessons_reader import find_amendment_eligible_lessons
    scanned = 0
    proposed = 0
    skipped_dup = 0
    for uid in tenants:
        # v1: tenant_id == user_id
        candidates = find_amendment_eligible_lessons(
            tenant_id=uid, user_id=uid,
            confidence_min=CONST_AMENDMENT_CONFIDENCE_MIN,
            reinforcement_min=CONST_AMENDMENT_REINFORCEMENT_MIN,
            limit=500,
        )
        scanned += len(candidates)
        for lesson in candidates:
            pid = await propose_one(lesson=lesson)
            if pid:
                proposed += 1
            else:
                skipped_dup += 1
    out = {
        "scanned":      scanned,
        "proposed":     proposed,
        "skipped_dup":  skipped_dup,
    }
    if proposed:
        logger.info(f"[constitution_amendments] cycle: {out}")
    return out


class ConstitutionAmendmentWorker:
    """Weekly scan for amendment-eligible lessons.  Cheap; idempotent."""

    def __init__(
        self,
        *,
        interval_seconds: float = 7 * 24 * 3600.0,
        db_path: str = "/tcmm-data/veilguard/tcmm.db",
    ):
        self.interval_seconds = float(interval_seconds)
        self.db_path = db_path
        self._stop_evt = asyncio.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    async def run(self) -> None:
        logger.info(
            f"[constitution_amendments] worker starting; "
            f"interval={self.interval_seconds:.0f}s "
            f"conf_min={CONST_AMENDMENT_CONFIDENCE_MIN} "
            f"reinf_min={CONST_AMENDMENT_REINFORCEMENT_MIN}"
        )
        try:
            while not self._stop_evt.is_set():
                try:
                    await run_one_cycle(db_path=self.db_path)
                except Exception as e:
                    logger.exception(f"[constitution_amendments] cycle error: {e}")
                try:
                    await asyncio.wait_for(
                        self._stop_evt.wait(),
                        timeout=self.interval_seconds,
                    )
                except asyncio.TimeoutError:
                    pass
        finally:
            logger.info("[constitution_amendments] worker stopped")


__all__ = [
    "SIGNAL_CONSTITUTION_AMENDMENT",
    "draft_amendment_brief",
    "propose_one",
    "run_one_cycle",
    "ConstitutionAmendmentWorker",
]
