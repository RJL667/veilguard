"""Phase 7.1 M1 cutover — TCMM-backed lesson reader.

After M1 cutover, lessons no longer live in the `org_memory` Lance
table; they are TCMM observations written by
`promote_lesson_to_team_knowledge` to `team/knowledge/<team_id>` with
the deterministic prefix `[lesson]`.

This module provides the read-path API the legacy `org_memory`
callers used to use, but resolves lessons by scanning the TCMM
`archive` Lance table for rows whose text body starts with the
marker prefix.

Why scan archive directly (not /search):
  * Deterministic enumeration — /search returns ranked hits;
    review-queue + amendment-eligibility need a deterministic list.
  * Works when TCMM HTTP is offline — same Lance file as the writer.
  * No namespace/scope ceremony — we filter by extracted_by + prefix.

Lifecycle metadata (status / decay / expiry / reinforcement_count)
moved with the cutover:

  * `status`         — defaults to 'active' for every lesson in the
                       TCMM archive.  Retirement happens by writing a
                       follow-up `[lesson_retired]` observation; the
                       reader applies last-write-wins per (trigger).
  * `confidence`     — parsed from the body.
  * `reinforcement_count` — count of distinct extracted_by tags
                       across all `[lesson]` observations that share
                       (tenant_id, user_id, trigger).
  * `review_after`   — Phase 7.5 follow-up; for now every lesson is
                       review-eligible from `created_ts + 30d`.
  * `expires_at`     — same: created_ts + (1 / decay_per_week) weeks.

The shape returned by `list_lessons` matches the old org_memory row
shape so callers don't have to change their downstream code.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

logger = logging.getLogger("agent-runtime.memory.lessons_reader")


# ── Parsing ──────────────────────────────────────────────────────────────


_LESSON_PREFIX = "[lesson]"
_RETIRED_PREFIX = "[lesson_retired]"

# Match the body the Phase 7 M1 writer emits.  TCMM may strip newlines
# during ingest (concatenated key=value pairs), so we use
# lookahead-bounded captures that stop at the next known marker rather
# than ^/$ line anchors.  Markers must be in this order in the body:
#   [lesson] trigger=<X> rule=<Y> confidence=0.873
#                       ──────    ──────────────
# The trailing `(?=\s+rule=|\s*$)` etc. is the bound.
_TRIGGER_RE     = re.compile(r"\btrigger\s*=\s*(.+?)(?=\s+rule\s*=|\s+confidence\s*=|$)")
_RULE_RE        = re.compile(r"\brule\s*=\s*(.+?)(?=\s+confidence\s*=|\s+trigger\s*=|$)")
_CONFIDENCE_RE  = re.compile(r"\bconfidence\s*=\s*([\d.]+)")


def parse_lesson_body(text: str) -> dict[str, Any] | None:
    """Parse a `[lesson] ...` body into a lesson dict; None if shape
    doesn't match.

    Expected body (per `promote_lesson_to_team_knowledge`):
        [lesson] trigger=<X>
        rule=<Y>
        confidence=0.873
    """
    if not text or not text.lstrip().startswith(_LESSON_PREFIX):
        return None
    # Strip the prefix line, then parse remaining lines.
    body = text.lstrip()
    # The first line carries `[lesson] trigger=X`; subsequent lines
    # are key=value pairs.
    after_marker = body[len(_LESSON_PREFIX):].lstrip()
    trigger_match = _TRIGGER_RE.search(after_marker)
    rule_match    = _RULE_RE.search(after_marker)
    conf_match    = _CONFIDENCE_RE.search(after_marker)
    if not trigger_match or not rule_match:
        return None
    try:
        confidence = float(conf_match.group(1)) if conf_match else 0.0
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "trigger":    trigger_match.group(1).strip(),
        "rule":       rule_match.group(1).strip(),
        "confidence": confidence,
    }


# ── Reader ───────────────────────────────────────────────────────────────


def list_lessons(
    *,
    tenant_id: str,
    user_id: str,
    status: str = "active",
    limit: int = 500,
    now_ts: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Return lessons for (tenant, user) by scanning TCMM `archive`.

    Returns dicts shaped to match the legacy org_memory row so
    downstream code doesn't need to change:

        {
          "id":                   <aid as str>,
          "tenant_id":            tenant_id,
          "user_id":              user_id,
          "trigger":              parsed,
          "rule":                 parsed,
          "confidence":           parsed,
          "status":               "active" or "retired",
          "reinforcement_count":  computed from distinct extracted_by,
          "created_ts":           archive row's created_ts,
          "review_after":         created_ts + 30d (Phase 7.5 may evolve),
          "expires_at":           created_ts + 365d (placeholder),
          "promoted_from":        same as "id" in TCMM-only world,
          "kind":                 "lesson",
        }

    Args:
        status:  "active" (default) returns lessons not retired.
                 "retired" returns retirements only.
                 "all" returns everything.
        limit:   max rows to scan.  Bounded to prevent runaway.
    """
    from ..ledger.store import LedgerStore
    try:
        tbl = LedgerStore.get()._db.open_table("archive")
    except Exception as e:
        logger.warning(f"[lessons_reader] archive open failed: {e}")
        return []

    # Scan the whole archive.  Lance's row count is bounded by the
    # writer-output rate × retention; production caps the file under
    # ~1M rows (memory: architecture_lance_maintenance.md).  Pulling
    # the whole table is cheap and avoids missing lessons that fell
    # outside an arbitrary head-window.  We bound the RETURNED list
    # by `limit`, not the scan.
    try:
        arr = tbl.search().limit(1_000_000).to_arrow().to_pylist()
    except Exception as e:
        logger.warning(f"[lessons_reader] archive scan failed: {e}")
        return []

    # [F4c_LESSON_NAMESPACE_2026_05_28] Phase 7 M1 spec said lessons
    # would live in `team/knowledge/<team_id>`, but the middleware
    # hardcodes `agent/<agent_id>/observations/<user_id>` and ignores
    # the caller's conv_id — so lessons actually land in the
    # critic-prose agent namespace.  We filter on the reliable
    # invariant instead: text prefix `[lesson]` AND row's user_id
    # column matches.  Tenant isolation is preserved because TCMM's
    # per-row user_id column is the source of truth for tenant scope
    # (memory `architecture_tcmm_api.md`).

    lessons: list[dict[str, Any]] = []
    retired_triggers: set[str] = set()
    by_trigger_eb: dict[str, set[str]] = {}

    for row in arr:
        text = row.get("text") or ""
        row_user_id = (row.get("user_id") or "")
        # Tenant isolation: TCMM stores user_id per row.  In v1, team_id
        # == tenant_id == user_id (single-user-per-tenant assumption).
        if row_user_id != user_id:
            continue

        # [CHANNEL_2026_05_29 / P6] Channel cutover: lessons now live in the
        # `team_knowledge` channel. Skip rows whose channel is set to
        # something else; rows with NULL channel are legacy (pre-backfill)
        # and fall through to the `[lesson]`-prefix discriminator below
        # (the transitional OR-clause, spec §6/§8). Replaces the
        # [F4c_LESSON_NAMESPACE_2026_05_28] prefix-only scan rationale.
        _row_ch = (row.get("channel") or "").strip()
        if _row_ch and _row_ch != "team_knowledge":
            continue

        # Retirement marker — record the trigger.
        if text.lstrip().startswith(_RETIRED_PREFIX):
            t = _TRIGGER_RE.search(text)
            if t:
                retired_triggers.add(t.group(1).strip())
            continue

        parsed = parse_lesson_body(text)
        if parsed is None:
            continue

        eb = (row.get("extracted_by") or "").strip()
        # Reinforcement = unique authors per (trigger).
        by_trigger_eb.setdefault(parsed["trigger"], set()).add(eb)

        created_ts = float(row.get("created_ts") or 0.0)
        aid_str    = str(row.get("aid") or "")
        lessons.append({
            "id":                   f"lsn-tcmm-{aid_str}",
            "tenant_id":            tenant_id,
            "user_id":              user_id,
            "trigger":              parsed["trigger"],
            "rule":                 parsed["rule"],
            "confidence":           parsed["confidence"],
            "status":               "active",   # filtered below
            "kind":                 "lesson",
            "promoted_from":        eb or aid_str,
            "created_ts":           created_ts,
            "updated_ts":           created_ts,
            "expires_at":           created_ts + 365 * 86400,
            "review_after":         created_ts + 30 * 86400,
            "last_reinforced_ts":   created_ts,
            "reinforcement_count":  1,        # filled in below
            "reinforced_by_agent_ids": [eb] if eb else [],
            "_tcmm_aid":            row.get("aid"),
        })

    # Collapse to one lesson per (trigger).  Multiple TCMM observations
    # of the same trigger represent reinforcement, not separate lessons
    # — surface a SINGLE row whose reinforcement_count reflects the
    # distinct-author cardinality and whose canonical entry is the
    # oldest observation (so `created_ts`/`promoted_from` is stable).
    by_trigger: dict[str, dict[str, Any]] = {}
    for L in lessons:
        trig = L["trigger"]
        existing = by_trigger.get(trig)
        if existing is None or L["created_ts"] < existing["created_ts"]:
            by_trigger[trig] = L

    out: list[dict[str, Any]] = []
    for L in by_trigger.values():
        if L["trigger"] in retired_triggers:
            L["status"] = "retired"
        L["reinforcement_count"] = len(
            by_trigger_eb.get(L["trigger"], set())
        ) or 1
        L["reinforced_by_agent_ids"] = sorted(
            by_trigger_eb.get(L["trigger"], set())
        )
        if status == "active" and L["status"] != "active":
            continue
        if status == "retired" and L["status"] != "retired":
            continue
        out.append(L)
        if len(out) >= limit:
            break

    return out


# ── Specialised reads ────────────────────────────────────────────────────


def find_lessons_due_for_review(
    *,
    tenant_id: str,
    user_id: str,
    now_ts: Optional[float] = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Phase 3 / §3.10 — active lessons whose `review_after` has passed."""
    now = now_ts or time.time()
    return [
        L for L in list_lessons(
            tenant_id=tenant_id, user_id=user_id, status="active", limit=limit,
        )
        if (L.get("review_after") or 0) <= now
    ]


def find_amendment_eligible_lessons(
    *,
    tenant_id: str,
    user_id: str,
    confidence_min: float,
    reinforcement_min: int,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Constitution amendments — active lessons meeting threshold."""
    return [
        L for L in list_lessons(
            tenant_id=tenant_id, user_id=user_id, status="active", limit=limit,
        )
        if (L.get("confidence") or 0) >= confidence_min
        and (L.get("reinforcement_count") or 0) >= reinforcement_min
    ]


# ── Backfill from legacy org_memory ──────────────────────────────────────


async def backfill_from_org_memory(
    *,
    db_path: str = "/tcmm-data/veilguard/tcmm.db",
    dry_run: bool = False,
) -> dict[str, Any]:
    """One-shot M1 cutover migration: read every row in the legacy
    `org_memory` Lance table and observe it into TCMM via the Phase 7
    M1 split-writer.

    Idempotency: relies on TCMM's content-hash dedup at the archive
    level.  Re-running is safe — re-observing the same text returns 0
    new rows.

    Run once before dropping the org_memory table.
    """
    import lancedb
    try:
        db = lancedb.connect(db_path)
        tbl = db.open_table("org_memory")
    except Exception as e:
        return {"error": f"org_memory open failed: {e}",
                "candidates": 0, "migrated": 0}

    try:
        rows = tbl.search().limit(50000).to_arrow().to_pylist()
    except Exception as e:
        return {"error": f"org_memory scan failed: {e}",
                "candidates": 0, "migrated": 0}

    candidates = len(rows)
    migrated = 0
    failed   = 0
    skipped  = 0

    if dry_run:
        return {"candidates": candidates, "dry_run": True,
                "would_migrate": candidates}

    from .phase_7_writers import promote_lesson_to_team_knowledge
    for r in rows:
        trigger = (r.get("trigger") or "").strip()
        rule    = (r.get("rule") or "").strip()
        if not (trigger and rule):
            skipped += 1
            continue
        ok = await promote_lesson_to_team_knowledge(
            tenant_id=r.get("tenant_id") or "",
            user_id=r.get("user_id") or "",
            team_id=r.get("tenant_id") or "",  # v1
            trigger=trigger,
            rule=rule,
            confidence=float(r.get("confidence") or 0.0),
            promoted_from=r.get("promoted_from") or r.get("id"),
        )
        if ok:
            migrated += 1
        else:
            failed += 1

    return {
        "candidates": candidates,
        "migrated":   migrated,
        "failed":     failed,
        "skipped":    skipped,
        "dry_run":    False,
    }


__all__ = [
    "list_lessons",
    "find_lessons_due_for_review",
    "find_amendment_eligible_lessons",
    "parse_lesson_body",
    "backfill_from_org_memory",
]
