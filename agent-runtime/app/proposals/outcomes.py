"""Phase 3 — proposal_outcomes regret writer (§3.7.7).

For every proposal that became a Task (status=approved with
resulting_task_id set) and whose Task reached a terminal state ≥30
days ago, compute:

    regret_score = task_cost / max(value_realized, ε)

…and write one row to ``proposal_outcomes.lance``.  Weekly
recalibration (§3.7.2 end) reads from this table.

### v1 cost/value heuristics

`task_cost` — from `pii_audit` rows with matching `task_id`
(Phase 2's typed-column work plumbed task_id end-to-end).  Sum of
`cost_usd` or compute from token_in/out × rate cards when the
cost column is missing.

`value_realized` — v1 PROXY: number of dream_archive rows whose
`source_block_ids` overlap with the task's output AIDs.  This
approximates "did downstream consolidation use this task's
artifacts."  The spec calls out producer-self-recalls exclusion
(§11.6.2 gaming mitigation) — v1 doesn't have separate
producer/consumer tracking, so we treat any reference as value;
v2 will need agent-id-scoped recall accounting.

### Cadence

Daily.  Lighter than the LifecycleWorker's hourly scan because the
30d-after-completion gate means there's nothing to do on most
cycles.  Configurable via ``VEILGUARD_OUTCOMES_INTERVAL_S``.

### Idempotency

Outcomes are written exactly once per proposal_id.  On each cycle
we scan for eligible proposals whose proposal_id is NOT already in
`proposal_outcomes`.  No upserts; if the heuristic improves, the
row reflects the version of the heuristic at write time.  Weekly
recalibration uses what's there.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Optional

from .scoring import ALL_SIGNAL_TYPES
from ..ledger.store import LedgerStore

logger = logging.getLogger(__name__)


# Default interval — once per day.  Tunable per env.
DEFAULT_INTERVAL_S = 24 * 3600.0

# Outcome eligibility window — task must have been terminal for ≥30d
# before its regret is computed.  Per spec §3.7.7.
OUTCOME_AGE_THRESHOLD_S = 30 * 24 * 3600.0

# Floor for value_realized so the division doesn't explode.
VALUE_REALIZED_EPSILON = 0.1


# [PRICING_2026_05_28]  Per-model rate cards in USD per million tokens.
# Source: memory `reference_anthropic_pricing.md` (Opus 4.5/4.6/4.7).
# Output rate for Opus not yet confirmed by user; using Anthropic's
# public Opus 4 price point ($75/M output) as a placeholder — flag this
# in the spec when the real number lands.  Sonnet + Haiku are public.
# Keys must match the `model` column values in pii_audit.
_RATE_CARD_USD_PER_M: dict[str, dict[str, float]] = {
    # claude-opus-4.5/4.6/4.7 — $5 in, $0.50 cache_read, $10 cache_write 1h
    "claude-opus-4-5":           {"in": 5.0,  "out": 75.0, "cache_rd": 0.50, "cache_wr": 10.0},
    "claude-opus-4-6":           {"in": 5.0,  "out": 75.0, "cache_rd": 0.50, "cache_wr": 10.0},
    "claude-opus-4-7":           {"in": 5.0,  "out": 75.0, "cache_rd": 0.50, "cache_wr": 10.0},
    # claude-sonnet-4-5/4-6 — public: $3 in, $15 out, $0.30 cache_rd, $6 cache_wr
    "claude-sonnet-4-5":         {"in": 3.0,  "out": 15.0, "cache_rd": 0.30, "cache_wr": 6.0},
    "claude-sonnet-4-6":         {"in": 3.0,  "out": 15.0, "cache_rd": 0.30, "cache_wr": 6.0},
    # claude-haiku-4-5 — public: $0.80 in, $4 out, $0.08 cache_rd, $1.60 cache_wr
    "claude-haiku-4-5":          {"in": 0.80, "out": 4.0,  "cache_rd": 0.08, "cache_wr": 1.60},
    "claude-haiku-4-5-20251001": {"in": 0.80, "out": 4.0,  "cache_rd": 0.08, "cache_wr": 1.60},
}
# Fallback for unknown models — conservative Sonnet rates.
_FALLBACK_RATE = _RATE_CARD_USD_PER_M["claude-sonnet-4-6"]


def _cost_from_tokens(
    model: str,
    tokens_in: int,
    tokens_out: int,
    cache_create: int,
    cache_read: int,
) -> float:
    """Apply the rate card to one pii_audit row's token counts.

    Anthropic's `input_tokens` excludes cache reads + creates (per
    memory `architecture_token_accounting.md`), so all four columns
    are billable-distinct.  Sum them at per-million resolution.
    """
    card = _RATE_CARD_USD_PER_M.get(model) or _FALLBACK_RATE
    return (
        tokens_in    * card["in"]       / 1_000_000.0 +
        tokens_out   * card["out"]      / 1_000_000.0 +
        cache_read   * card["cache_rd"] / 1_000_000.0 +
        cache_create * card["cache_wr"] / 1_000_000.0
    )


def _compute_task_cost(
    *,
    audit_tbl: Any,
    task_id: str,
) -> tuple[float, dict[str, int]]:
    """Sum cost across pii_audit rows tagged with this task_id.

    Returns (total_usd, token_totals_dict) where token_totals has
    keys tokens_input, tokens_output, cache_create, cache_read.

    [PRICING_2026_05_28] pii_audit doesn't store a precomputed cost_usd
    column — we derive it from tokens × per-model rate card every
    scan.  Live verified: previous SELECT cost_usd failed with
    `No field named cost_usd`, regret_score=0 everywhere.
    """
    try:
        arr = (
            audit_tbl.search()
            .where(f"task_id = '{task_id}'")
            .select([
                "model", "tokens_input", "tokens_output",
                "cache_create", "cache_read",
            ])
            .limit(5000)
            .to_arrow()
        )
    except Exception as e:
        logger.warning(f"[outcomes] cost scan failed for {task_id}: {e}")
        return 0.0, {}
    total = 0.0
    tokens_in = tokens_out = cache_cr = cache_rd = 0
    for i in range(arr.num_rows):
        try:
            model = arr.column("model")[i].as_py() or ""
            ti = int(arr.column("tokens_input")[i].as_py() or 0)
            to = int(arr.column("tokens_output")[i].as_py() or 0)
            cc = int(arr.column("cache_create")[i].as_py() or 0)
            cr = int(arr.column("cache_read")[i].as_py() or 0)
            total += _cost_from_tokens(model, ti, to, cc, cr)
            tokens_in  += ti
            tokens_out += to
            cache_cr   += cc
            cache_rd   += cr
        except Exception:
            continue
    return total, {
        "tokens_input":  tokens_in,
        "tokens_output": tokens_out,
        "cache_create":  cache_cr,
        "cache_read":    cache_rd,
    }


def _compute_value_realized(
    *,
    dream_tbl: Any,
    task_outputs: list[str],
    task_id: str,
) -> float:
    """v1 proxy: count of dream_archive rows whose source_block_ids
    overlap with the task's output AIDs.

    This is intentionally loose — v2 should track per-agent recall
    counts excluding producer-self-recalls (§11.6.2 gaming mitigation).
    For v1 we just want the column populated so weekly recalibration
    has SOMETHING to aggregate.

    `task_outputs` are typically file paths today (e.g.
    "team/drafts/foo.md"), not block AIDs.  We can't trivially map
    paths to AIDs without an extra lookup, so this function falls
    back to "count of dream rows that mention any output path
    string in their `claims` or `text`".  Crude but documentable.
    """
    if not task_outputs or not dream_tbl:
        return 0.0
    # Filter terms — strip leading 'team/drafts/' style prefixes so we
    # match on the document slug too.
    needles = set()
    for p in task_outputs:
        if not isinstance(p, str):
            continue
        needles.add(p)
        # last path segment (slug)
        slug = p.rsplit("/", 1)[-1]
        if slug:
            needles.add(slug)
    if not needles:
        return 0.0
    try:
        # Scan a generous window of recent dream rows; in practice the
        # recall reuse signal lives in the last few thousand rows.
        arr = (
            dream_tbl.search()
            .select(["text", "claims", "source_block_ids", "namespace"])
            .limit(5000)
            .to_arrow()
        )
    except Exception as e:
        logger.warning(f"[outcomes] dream scan failed: {e}")
        return 0.0
    hits = 0
    for i in range(arr.num_rows):
        text = (arr.column("text")[i].as_py() or "")
        claims_list = arr.column("claims")[i].as_py() or []
        ns = (arr.column("namespace")[i].as_py() or "")
        # Don't count the task's own namespace — that's a self-recall.
        if task_id and task_id in ns:
            continue
        if any(n in text for n in needles):
            hits += 1
            continue
        for c in claims_list:
            if isinstance(c, str) and any(n in c for n in needles):
                hits += 1
                break
    return float(hits)


def _outcome_id() -> str:
    return f"outc-{uuid.uuid4().hex[:12]}"


def _write_outcome(
    *,
    outcomes_tbl: Any,
    proposal_id: str,
    resulting_task_id: str,
    task_status: str,
    task_cost_usd: float,
    value_realized: float,
    objective_deltas: Optional[dict[str, float]] = None,
    tenant_id: str = "",
    user_id: str = "",
    token_totals: Optional[dict[str, int]] = None,
) -> str:
    """Insert one proposal_outcomes row.  Returns the new outcome id."""
    import json as _json
    now = time.time()
    regret = float(task_cost_usd) / max(float(value_realized), VALUE_REALIZED_EPSILON)
    oid = _outcome_id()
    extras = {
        "token_totals":   token_totals or {},
        "heuristic_ver":  "v1",
        "epsilon_used":   VALUE_REALIZED_EPSILON,
    }
    row = {
        "id":                    oid,
        "kind":                  "outcome",
        "status":                "computed",
        "parent_id":             None,
        "lineage_chain":         [],
        "tenant_id":             tenant_id,
        "user_id":               user_id,
        "created_by_agent_id":   "proposals-outcomes",
        "created_ts":            now,
        "updated_ts":            now,
        "cost_attributed_usd":   float(task_cost_usd),
        "proposal_id":           proposal_id,
        "resulting_task_id":     resulting_task_id,
        "task_status":           task_status,
        "task_cost_usd":         float(task_cost_usd),
        "value_realized":        float(value_realized),
        "regret_score":          regret,
        "objective_deltas_json": _json.dumps(objective_deltas or {}),
        "computed_at_ts":        now,
        "extras_json":           _json.dumps(extras),
    }
    outcomes_tbl.add([row])
    return oid


def _outcome_already_exists(
    *,
    outcomes_tbl: Any,
    proposal_id: str,
) -> bool:
    """Idempotency check — already computed for this proposal?"""
    try:
        arr = (
            outcomes_tbl.search()
            .where(f"proposal_id = '{proposal_id}'")
            .select(["id"])
            .limit(1)
            .to_arrow()
        )
        return arr.num_rows > 0
    except Exception:
        return False


def find_eligible_proposals(
    *,
    proposals_tbl: Any,
    tasks_tbl: Any,
    outcomes_tbl: Any,
    age_threshold_s: float = OUTCOME_AGE_THRESHOLD_S,
) -> list[dict[str, Any]]:
    """Return proposals that should have an outcome row written.

    Eligibility:
      * status='approved' AND resulting_task_id is set
      * NO outcome row yet for this proposal_id (idempotency)
      * resulting_task is in terminal state (done | cancelled)
      * resulting_task.updated_ts is ≥ age_threshold_s old
    """
    now = time.time()
    cutoff = now - age_threshold_s
    try:
        arr = (
            proposals_tbl.search()
            .where("status = 'approved'")
            .select([
                "id", "resulting_task_id", "signal_type",
                "tenant_id", "user_id", "proposed_assignee",
                "impact_score",
            ])
            .limit(2000)
            .to_arrow()
        )
    except Exception as e:
        logger.warning(f"[outcomes] proposal scan failed: {e}")
        return []
    out: list[dict[str, Any]] = []
    for i in range(arr.num_rows):
        pid = arr.column("id")[i].as_py()
        rtid = arr.column("resulting_task_id")[i].as_py()
        if not (pid and rtid):
            continue
        if _outcome_already_exists(outcomes_tbl=outcomes_tbl, proposal_id=pid):
            continue
        # Look up the resulting task
        try:
            tarr = (
                tasks_tbl.search()
                .where(f"id = '{rtid}'")
                .select(["status", "updated_ts", "outputs"])
                .limit(1)
                .to_arrow()
            )
            if tarr.num_rows == 0:
                continue
            status = tarr.column("status")[0].as_py()
            updated = float(tarr.column("updated_ts")[0].as_py() or 0)
            outputs = tarr.column("outputs")[0].as_py() or []
        except Exception as e:
            logger.warning(f"[outcomes] task lookup failed for {rtid}: {e}")
            continue
        if status not in ("done", "cancelled"):
            continue
        if updated <= 0 or updated > cutoff:
            # Either no timestamp, or too recent.  Wait until ≥30d old.
            continue
        out.append({
            "proposal_id":       pid,
            "resulting_task_id": rtid,
            "task_status":       status,
            "task_outputs":      outputs,
            "tenant_id":         arr.column("tenant_id")[i].as_py() or "",
            "user_id":           arr.column("user_id")[i].as_py() or "",
            "signal_type":       arr.column("signal_type")[i].as_py() or "",
            "impact_score":      float(arr.column("impact_score")[i].as_py() or 0.0),
        })
    return out


async def compute_one(
    *,
    proposals_tbl: Any,
    tasks_tbl: Any,
    outcomes_tbl: Any,
    audit_tbl: Any,
    dream_tbl: Any,
    eligible: dict[str, Any],
) -> dict[str, Any]:
    """Compute + write one outcome row.  Returns a summary dict.

    Phase 7 M2 — routes through `record_outcome_with_narrative` so
    numeric metrics live in the ledger and the TCMM observation
    carries the (currently empty) narrative slot for cross-ref.
    Future work will populate `regret_text` / `what_went_wrong` from
    Critic-prose review output; today it's the cross-ref shell.
    """
    rtid = eligible["resulting_task_id"]
    pid  = eligible["proposal_id"]
    cost, tokens = _compute_task_cost(audit_tbl=audit_tbl, task_id=rtid)
    value = _compute_value_realized(
        dream_tbl=dream_tbl,
        task_outputs=eligible["task_outputs"],
        task_id=rtid,
    )
    regret = cost / max(value, VALUE_REALIZED_EPSILON)
    from ..memory.phase_7_writers import record_outcome_with_narrative
    oid, _tcmm_obs_id = await record_outcome_with_narrative(
        proposal_id=pid,
        resulting_task_id=rtid,
        tenant_id=eligible["tenant_id"],
        user_id=eligible["user_id"],
        task_status=eligible["task_status"],
        task_cost_usd=cost,
        value_realized=value,
        regret_score=regret,
        objective_deltas={"tokens": tokens} if tokens else None,
        # Narrative slots empty for the automated regret path; Critic
        # may upgrade later with what_went_wrong / lessons_learned.
        regret_text=None,
        what_went_wrong=None,
        lessons_learned=None,
    )
    summary = {
        "outcome_id":        oid,
        "proposal_id":       pid,
        "resulting_task_id": rtid,
        "task_cost_usd":     cost,
        "value_realized":    value,
        "regret_score":      regret,
    }
    return summary


async def run_one_cycle(*, db_path: str = "/tcmm-data/veilguard/tcmm.db") -> dict[str, Any]:
    """Async one-shot pass — find + write all eligible outcomes.

    Phase 7 wired-async — awaits `compute_one` which uses the Phase 7
    M2 split-writer for the TCMM cross-ref.  Safe to call from any
    async context; the background worker awaits this in a loop.
    """
    try:
        import lancedb
    except ImportError:
        return {"eligible": 0, "written": 0, "error": "lancedb missing"}
    try:
        db = lancedb.connect(db_path)
    except Exception as e:
        return {"eligible": 0, "written": 0, "error": f"db connect: {e}"}
    # All five tables — caller errors are fatal-per-table not fatal-per-cycle
    try:
        proposals_tbl = db.open_table("task_proposals")
        tasks_tbl     = db.open_table("agent_tasks")
        outcomes_tbl  = db.open_table("proposal_outcomes")
        audit_tbl     = db.open_table("pii_audit")
    except Exception as e:
        return {"eligible": 0, "written": 0, "error": f"open tables: {e}"}
    # dream_archive is optional — value_realized goes to 0 without it
    dream_tbl = None
    try:
        dream_tbl = db.open_table("dream_archive")
    except Exception as e:
        logger.debug(f"[outcomes] dream_archive unavailable, value_realized=0: {e}")
    eligible = find_eligible_proposals(
        proposals_tbl=proposals_tbl,
        tasks_tbl=tasks_tbl,
        outcomes_tbl=outcomes_tbl,
    )
    written = 0
    summaries = []
    for e in eligible:
        try:
            s = await compute_one(
                proposals_tbl=proposals_tbl,
                tasks_tbl=tasks_tbl,
                outcomes_tbl=outcomes_tbl,
                audit_tbl=audit_tbl,
                dream_tbl=dream_tbl,
                eligible=e,
            )
            summaries.append(s)
            written += 1
        except Exception as exc:
            logger.exception(
                f"[outcomes] compute_one failed for proposal "
                f"{e['proposal_id']}: {exc}"
            )
    out = {
        "eligible":  len(eligible),
        "written":   written,
        "summaries": summaries,
    }
    if written:
        logger.info(f"[outcomes] wrote {written} regret rows")
    return out


# ── Async background worker ────────────────────────────────────────────


class OutcomesWorker:
    """Periodic regret-row writer.  Idempotent + safe to run forever."""

    def __init__(
        self,
        *,
        interval_seconds: float = DEFAULT_INTERVAL_S,
        db_path: str = "/tcmm-data/veilguard/tcmm.db",
    ):
        self.interval_seconds = float(interval_seconds)
        self.db_path = db_path
        self._stop_evt = asyncio.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    async def run(self) -> None:
        logger.info(
            f"[outcomes] worker starting; interval={self.interval_seconds:.0f}s "
            f"age_threshold={OUTCOME_AGE_THRESHOLD_S/86400:.0f}d"
        )
        try:
            while not self._stop_evt.is_set():
                try:
                    await run_one_cycle(db_path=self.db_path)
                except Exception as e:
                    logger.exception(f"[outcomes] cycle error: {e}")
                try:
                    await asyncio.wait_for(
                        self._stop_evt.wait(), timeout=self.interval_seconds,
                    )
                except asyncio.TimeoutError:
                    pass
        finally:
            logger.info("[outcomes] worker stopped")


__all__ = [
    "DEFAULT_INTERVAL_S",
    "OUTCOME_AGE_THRESHOLD_S",
    "VALUE_REALIZED_EPSILON",
    "find_eligible_proposals",
    "compute_one",
    "run_one_cycle",
    "OutcomesWorker",
]
