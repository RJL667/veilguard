"""Phase 3 — DreamScanner.

Polls `dream_archive.lance` periodically, extracts generative-signal
nodes (information_gap, contradiction_arc, reflective_heuristic,
recurring_ritual, stance_arc), converts each into a proposal via the
existing scoring + briefing + assignee pipeline, and persists to
`task_proposals.lance` via `ledger.proposals.create_proposal()`.

This is the agent-runtime-side answer to spec §3.7.0's
"proposal emission hook".  It does NOT touch dream_engine.py — it
reads dream_archive that the TCMM process writes, and writes
task_proposals that agent-runtime owns.  Both processes share the
LanceDB store at `/tcmm-data/veilguard/tcmm.db`.

Cursor semantics:
  - Per (tenant_id, user_id), persist a `last_scanned_aid` in
    extras_json of a sentinel row OR in a small JSON file under
    /tmp/scanner_cursor/<tenant>.json.  Today we use the in-memory
    dict on the worker — cursor resets on container restart but
    dedup-on-(signal_type, signal_node_ids) at `create_proposal()`
    prevents double-emission: same node_ids on a pending/deferred row
    bumps `recurrence_count` instead of inserting a duplicate
    (shipped 2026-05-28 — see `[PROPOSAL_DEDUP_2026_05_28]` in
    `ledger.proposals.create_proposal`).

Scoring source:
  - dream_archive rows don't carry the precise scoring inputs (e.g.
    gap_breadth, source_severity); we derive what we can from the
    fields that ARE present (block_class, knowledge_class, claims,
    timestamps, source_block_ids).  Heuristic mappings live in
    ``_derive_payload`` — refine empirically as real dream output
    accumulates.

  - Fields the scanner reads (per dream_archive schema):
      block_class       → maps to signal_type
      claims            → for contradiction detection
      source_block_ids  → for signal_node_ids field
      heat              → proxy for downstream_pressure / centrality
      density_score     → proxy for cluster cohesion
      lineage           → for recurrence calculations
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Optional

from .scoring import (
    SignalPayload,
    final_score as _final_score,
    SIGNAL_INFORMATION_GAP,
    SIGNAL_CONTRADICTION_ARC,
    SIGNAL_REFLECTIVE_HEURISTIC,
    SIGNAL_RECURRING_RITUAL,
    SIGNAL_STANCE_ARC,
    ALL_SIGNAL_TYPES,
)
from .briefs import render_brief, render_deliverable_spec
from .assignees import default_assignee
from .constitution_bridge import objectives_to_dict, evaluate_constraints
from . import is_emergency_lane
from ..ledger import proposals as _props

logger = logging.getLogger(__name__)


# Block-class strings dream emits → spec signal types.  Empirical;
# refine when we see real dream output.  Anything not in this map is
# skipped (non-generative per §3.7.1: causal_arc, semantic_principle,
# narrative_arc, identity_*, motif_node, belief_attribution,
# concept_node).
BLOCK_CLASS_TO_SIGNAL: dict[str, str] = {
    "INFORMATION_GAP":      SIGNAL_INFORMATION_GAP,
    "INFO_GAP":             SIGNAL_INFORMATION_GAP,
    "information_gap":      SIGNAL_INFORMATION_GAP,
    "CONTRADICTION_ARC":    SIGNAL_CONTRADICTION_ARC,
    "contradiction_arc":    SIGNAL_CONTRADICTION_ARC,
    "REFLECTIVE_HEURISTIC": SIGNAL_REFLECTIVE_HEURISTIC,
    "reflective_heuristic": SIGNAL_REFLECTIVE_HEURISTIC,
    "RECURRING_RITUAL":     SIGNAL_RECURRING_RITUAL,
    "recurring_ritual":     SIGNAL_RECURRING_RITUAL,
    "STANCE_ARC":           SIGNAL_STANCE_ARC,
    "stance_arc":           SIGNAL_STANCE_ARC,
    # The 2 deferred-emitter classes — agent-runtime-side scanner
    # writes these into dream_archive; DreamScanner picks them up.
    "LOW_STABILITY_CLUSTER":  "low_stability_cluster",
    "low_stability_cluster":  "low_stability_cluster",
    "STALE_SUPERSESSION_CHAIN": "stale_supersession_chain",
    "stale_supersession_chain": "stale_supersession_chain",
}


def _derive_payload(row: dict[str, Any], signal_type: str) -> SignalPayload:
    """Heuristic mapping of dream_archive row fields → SignalPayload.

    The dream graph carries summary stats (heat, density_score, claims,
    source_block_ids) that we treat as proxies for the scoring inputs
    in §3.7.2.  These mappings are starting points — once real dream
    output starts flowing, calibrate against actual proposal outcomes.

    For fields we can't derive, leave None — the scoring functions
    then zero out the candidate, which is the correct behaviour (drop
    rather than guess wildly).
    """
    heat = float(row.get("heat") or 0.0)
    density = float(row.get("density_score") or 0.0)
    claims = row.get("claims") or []
    source_ids = row.get("source_block_ids") or []
    timestamp = float(row.get("timestamp") or 0.0)

    if signal_type == SIGNAL_INFORMATION_GAP:
        # gap_breadth from density (inverted: low density = wide gap)
        # downstream_pressure from heat × source-blocks-count
        return SignalPayload(
            gap_breadth=max(0.0, 1.0 - density) if density > 0 else None,
            downstream_pressure=heat * (len(source_ids) or 1),
        )
    if signal_type == SIGNAL_CONTRADICTION_ARC:
        # source_kind extraction from origin field (dream may emit
        # "USER"/"AGENT"/"INFERRED" labels; if not, default INFERRED)
        origin = (row.get("origin") or "").upper()
        # Conservative default: INFERRED×INFERRED (lowest severity)
        return SignalPayload(
            source_kind_a=origin or "INFERRED",
            source_kind_b=origin or "INFERRED",
            claim_centrality=heat,  # bridge_score proxy
        )
    if signal_type in (SIGNAL_REFLECTIVE_HEURISTIC, SIGNAL_RECURRING_RITUAL):
        # recurrence ~ how many source_blocks rolled into this node
        # success_rate ~ density_score (clean clusters → high success)
        # token_savings_potential ~ heat × claim_count
        return SignalPayload(
            recurrence=float(len(source_ids)),
            success_rate=min(1.0, density) if density > 0 else None,
            token_savings_potential=heat * max(1, len(claims)),
        )
    if signal_type == SIGNAL_STANCE_ARC:
        # polarity_distance and claim_stake — dream doesn't expose
        # these directly today; conservative defaults
        return SignalPayload(
            polarity_distance=heat,
            claim_stake=density,
        )
    # The 2 emitter-injected signals (low_stability_cluster +
    # stale_supersession_chain) carry their payload in the
    # synthetic dream row's heat / source_block_ids / age fields.
    if signal_type == "low_stability_cluster":
        return SignalPayload(
            failure_count=float(len(source_ids) or 1),
            cluster_recall_frequency=heat or 1.0,
        )
    if signal_type == "stale_supersession_chain":
        age_days = max(1.0, (time.time() - timestamp) / 86400.0) if timestamp else 30.0
        return SignalPayload(
            age_days=age_days,
            recall_count=float(len(source_ids) or 1),
            topic_currency_index=heat or 0.5,
        )
    # Unknown / unmapped — return empty payload (scoring will zero)
    return SignalPayload()


def _derive_brief_vars(row: dict[str, Any], signal_type: str) -> dict[str, Any]:
    """Pull brief template variables out of the dream row.

    The text field on a dream node is human-readable; topics + claims
    are the salient extractions.  Brief templates per §3.7.5 want
    things like ``gap_topic``, ``claim_a_text``, ``pattern_id`` —
    we map best-effort.
    """
    text = (row.get("text") or "")[:240]
    topics = row.get("topics") or []
    claims = row.get("claims") or []
    aid = row.get("aid")

    if signal_type == SIGNAL_INFORMATION_GAP:
        return {
            "gap_topic": (topics[0] if topics else text[:80] or "uncategorised"),
            "target_namespace": "team_knowledge",
        }
    if signal_type == SIGNAL_CONTRADICTION_ARC:
        return {
            "claim_a_text": (claims[0] if len(claims) >= 1 else text[:80]),
            "claim_b_text": (claims[1] if len(claims) >= 2 else "(pair pending)"),
        }
    if signal_type == SIGNAL_RECURRING_RITUAL:
        return {"pattern_id": f"dream-aid-{aid}"}
    if signal_type == SIGNAL_REFLECTIVE_HEURISTIC:
        return {"pattern_desc": text[:120] or f"reflective dream node {aid}"}
    if signal_type == SIGNAL_STANCE_ARC:
        return {
            "agent_a": "?",
            "agent_b": "?",
            "claim_text": (claims[0] if claims else text[:120]),
        }
    if signal_type == "low_stability_cluster":
        return {
            "cluster_desc": (topics[0] if topics else text[:80]) or f"dream-aid-{aid}",
            "fail_count":   len(claims) or len(row.get("source_block_ids") or []),
        }
    if signal_type == "stale_supersession_chain":
        ts = float(row.get("timestamp") or 0)
        age = int((time.time() - ts) / 86400.0) if ts else 60
        return {
            "chain_topic":  (topics[0] if topics else text[:80]) or f"dream-aid-{aid}",
            "days_ago":     age,
        }
    return {}


class DreamScanner:
    """Polls dream_archive and emits proposals."""

    def __init__(
        self,
        *,
        interval_seconds: float = 600.0,
        max_per_cycle: int = 10,
        per_signal_cap: int = 3,
        deterministic_top_n: int = 3,
        rank_pass_enabled: Optional[bool] = None,
        rank_pass_model: Optional[str] = None,
        db_path: Optional[str] = None,
        constitution: Optional[dict[str, Any]] = None,
    ):
        self.interval_seconds = float(interval_seconds)
        self.max_per_cycle = int(max_per_cycle)
        self.per_signal_cap = int(per_signal_cap)
        # §3.7.5: ranks 1..N get template briefs (no LLM); N+1..max go
        # through the Haiku rank pass.  Default N=3 per spec.
        self.deterministic_top_n = int(deterministic_top_n)
        # Rank pass is OPT-IN via env so a misconfigured Anthropic
        # bearer doesn't spam-call the API on every cycle.
        if rank_pass_enabled is None:
            rank_pass_enabled = os.environ.get(
                "VEILGUARD_PROPOSAL_RANK_PASS_ENABLED", "0",
            ).lower() in ("1", "true", "yes")
        self.rank_pass_enabled = bool(rank_pass_enabled)
        from .rank import DEFAULT_RANK_MODEL as _DEFAULT_MODEL
        self.rank_pass_model = rank_pass_model or os.environ.get(
            "VEILGUARD_PROPOSAL_RANK_PASS_MODEL", _DEFAULT_MODEL,
        )
        self.db_path = db_path or os.environ.get(
            "VEILGUARD_AUDIT_DB_PATH", "/tcmm-data/veilguard/tcmm.db"
        )
        self.constitution = constitution or {}
        self._stop_evt = asyncio.Event()
        # cursor per (tenant_id, user_id) — last_aid scanned
        self._cursor: dict[tuple[str, str], int] = {}
        # in-memory dedup so back-to-back scans on the same node don't
        # spawn duplicates (real dedup belongs in create_proposal but
        # this is the lowest-friction interim guard)
        self._seen_keys: set[tuple[str, tuple]] = set()
        # [PHASE_7_5_EVENT_TRIGGER_2026_05_28] Surviving Q22 piece —
        # event-triggered early-fire.  External callers (importance-sum
        # watcher, sidebar "scan now" button, dream cycle completion
        # hook) set this event to break the interval_seconds wait and
        # run `_scan_once()` immediately on the next loop iteration.
        # Cheap: the wait_for in `run()` returns early on event-set,
        # then we clear and resume the normal timed cadence.
        self._wake_evt = asyncio.Event()
        # Accumulated importance score since the last scan.  Callers
        # add to this via `notify_importance()`; when it crosses
        # `importance_threshold`, the scanner fires.  Reset to 0 after
        # each scan.
        self._importance_accum: float = 0.0
        self.importance_threshold: float = float(os.environ.get(
            "VEILGUARD_DREAM_IMPORTANCE_THRESHOLD", "5.0"
        ))
        # [DREAM_BACKPRESSURE_2026_06_01] Hard cap on the pending-proposal
        # queue. The dream cycle re-derives clusters with churning
        # signal_node_ids, so create_proposal's node-id dedup misses and the
        # queue ballooned to 345 rows (18 real clusters x dozens of dupes)
        # in a 37h window — re-running the rank-pass LLM every cycle for
        # work nobody drains. When pending >= this cap, _scan_once SKIPS the
        # whole cycle (no scan, no scoring, no LLM rank pass, no emit) until
        # a human approves/shelves some. Bounds the queue AND stops the idle
        # LLM burn. Tunable via env.
        self.max_pending_backpressure = int(os.environ.get(
            "VEILGUARD_DREAM_MAX_PENDING", "25"
        ))

    def notify_importance(self, score: float) -> None:
        """Accumulate importance contribution toward the early-fire
        threshold.  If accumulated importance crosses
        `importance_threshold`, wake the scanner immediately.

        Used by hooks in the observation hot-path (TCMM observe wrapper,
        new high-impact proposal emit) to keep the dream graph fresh
        without paying the full interval_seconds wait when a burst of
        meaningful signal lands.
        """
        try:
            self._importance_accum += float(score or 0.0)
        except (TypeError, ValueError):
            return
        if self._importance_accum >= self.importance_threshold:
            self._wake_evt.set()

    def fire_now(self, reason: str = "manual") -> None:
        """External trigger — wake the scanner immediately regardless
        of importance accumulation.  Used by the sidebar "scan now"
        button and dream cycle completion hooks."""
        logger.info(f"[dream_scanner] fire_now: reason={reason!r}")
        self._wake_evt.set()

    def stop(self) -> None:
        self._stop_evt.set()

    async def _scan_once(self) -> dict[str, Any]:
        """One scan of dream_archive → emit proposals.

        Async because the optional rank pass (§3.7.5, ranks 4-10) is an
        async LLM call.  The lance read + scoring + write are still
        synchronous; only the rank pass yields.

        Returns:
            {
              "rows_scanned":   int,
              "candidates":     int,
              "emitted":        int,
              "skipped_dedup":  int,
              "skipped_emit":   int,
              "rank_passed":    int   # candidates that went through Haiku
              "by_signal":      {signal_type: int_emitted},
            }
        """
        try:
            import lancedb
        except ImportError:
            logger.warning("[dream_scanner] lancedb not installed")
            return {"rows_scanned": 0, "candidates": 0, "emitted": 0,
                    "skipped_dedup": 0, "by_signal": {}}
        try:
            db = lancedb.connect(self.db_path)
            tbl = db.open_table("dream_archive")
        except Exception as e:
            logger.debug(f"[dream_scanner] dream_archive unavailable: {e}")
            return {"rows_scanned": 0, "candidates": 0, "emitted": 0,
                    "skipped_dedup": 0, "by_signal": {}}

        # [DREAM_BACKPRESSURE_2026_06_01] Skip the ENTIRE cycle if the
        # pending queue is already at/over the cap. No point scanning,
        # scoring, paying the Haiku rank pass, or emitting more proposals
        # onto a backlog nobody is draining — that's exactly how 345 stale
        # rows accumulated. The cap releases automatically once the user
        # approves/shelves enough to drop pending back under the limit.
        try:
            _pending = db.open_table("task_proposals").count_rows(
                "status = 'pending'"
            )
            if _pending >= self.max_pending_backpressure:
                logger.info(
                    f"[dream_scanner] backpressure: {_pending} pending >= cap "
                    f"{self.max_pending_backpressure}; skipping cycle (no "
                    f"scan/scoring/LLM/emit until the queue drains)"
                )
                return {"rows_scanned": 0, "candidates": 0, "emitted": 0,
                        "skipped_dedup": 0, "skipped_backpressure": _pending,
                        "by_signal": {}}
        except Exception as _bpe:
            logger.debug(f"[dream_scanner] backpressure check skipped: {_bpe}")

        # Scan recent rows — limit large to catch anything we missed
        # if the cursor reset (container restart).  Filtering happens
        # in Python because Lance where-clause doesn't support our
        # signal-class filter cleanly with the current schema.
        try:
            arr = tbl.search().limit(2000).to_arrow()
        except Exception as e:
            logger.warning(f"[dream_scanner] scan failed: {e}")
            return {"rows_scanned": 0, "candidates": 0, "emitted": 0,
                    "skipped_dedup": 0, "by_signal": {}}

        rows_scanned = arr.num_rows
        candidates: list[tuple[str, dict, str]] = []  # (signal_type, row, key_tag)
        # [PHASE_3_PROACTIVE_CONFIG_2026_05_27] Skip tenants whose
        # proactive stream is disabled or paused.  Cached per cycle so
        # we don't re-read the table for every dream row.
        from ..ledger.proactive_config import get_or_default as _get_cfg
        _tenant_skip_cache: dict[tuple[str, str], bool] = {}
        def _tenant_skipped(tid: str, uid: str) -> bool:
            key = (tid, uid)
            if key in _tenant_skip_cache:
                return _tenant_skip_cache[key]
            try:
                cfg = _get_cfg(tid, uid)
                skip = not cfg.stream_active
            except Exception:
                # Conservative: on lookup failure DON'T skip (the
                # default is enabled; we'd rather over-emit than
                # under-emit when the config table is unreadable).
                skip = False
            _tenant_skip_cache[key] = skip
            return skip

        for i in range(arr.num_rows):
            row = {c: arr.column(c)[i].as_py() for c in arr.column_names}
            bc = row.get("block_class") or ""
            sig = BLOCK_CLASS_TO_SIGNAL.get(bc)
            if not sig or sig not in ALL_SIGNAL_TYPES:
                continue
            tenant_id = row.get("user_id") or ""  # dream namespaces on user_id
            user_id = tenant_id  # single-tenant-per-user posture today
            # Skip if this tenant's proactive stream is gated off
            if _tenant_skipped(tenant_id, user_id):
                continue
            aid = row.get("aid") or 0
            cursor_key = (tenant_id, user_id)
            last_aid = self._cursor.get(cursor_key, 0)
            if aid <= last_aid:
                continue
            source_ids = tuple(sorted(row.get("source_block_ids") or []))
            dedup_key = (sig, source_ids)
            if dedup_key in self._seen_keys:
                continue
            candidates.append((sig, row, dedup_key))

        # Score, cap per-type, take top max_per_cycle.
        # Per-tenant alignment vectors override the static defaults
        # when the recalibration worker has written rows for them
        # (Phase 3 §3.7.2 weekly recal).  Cached per cycle.
        objs = objectives_to_dict(self.constitution)
        from .recalibration import get_alignment_for_tenant as _get_alignment
        _alignment_cache: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
        def _alignment_for(tid: str, uid: str) -> dict[str, dict[str, float]]:
            key = (tid, uid)
            if key not in _alignment_cache:
                try:
                    _alignment_cache[key] = _get_alignment(tenant_id=tid, user_id=uid)
                except Exception:
                    _alignment_cache[key] = None  # fall back to defaults
            return _alignment_cache[key]

        scored: list[tuple[float, str, dict, dict, list]] = []
        for sig, row, key in candidates:
            payload = _derive_payload(row, sig)
            ctx = {
                "target_namespace": "team_knowledge",
                "source_kind":      row.get("origin") or "INFERRED",
            }
            violations = evaluate_constraints(self.constitution, ctx)
            tenant_id_cand = row.get("user_id") or ""
            final, _ = _final_score(
                signal_type=sig,
                payload=payload,
                constitution_objectives=objs,
                constraint_violations=violations,
                alignment_vectors=_alignment_for(tenant_id_cand, tenant_id_cand),
            )
            if final <= 0:
                continue
            scored.append((final, sig, row, payload.__dict__, violations))

        scored.sort(key=lambda x: x[0], reverse=True)

        per_type_count: dict[str, int] = {}
        capped: list[tuple[float, str, dict, dict, list]] = []
        for s in scored:
            if len(capped) >= self.max_per_cycle:
                break
            sig = s[1]
            if per_type_count.get(sig, 0) >= self.per_signal_cap:
                continue
            capped.append(s)
            per_type_count[sig] = per_type_count.get(sig, 0) + 1

        # [PHASE_3_RANK_PASS_2026_05_27]  Per spec §3.7.5:
        #   Ranks 1..deterministic_top_n → template brief + default
        #     assignee (no LLM call).
        #   Ranks deterministic_top_n+1..max_per_cycle → ONE Haiku rank
        #     pass that refines briefs + assignees + alignments.
        # Build template-side defaults for every candidate first so the
        # rank-pass call site has a single source of truth.
        defaulted: list[dict[str, Any]] = []
        for final, sig, row, payload_dict, violations in capped:
            default_brief = render_brief(sig, _derive_brief_vars(row, sig))
            default_assignee_val = default_assignee(sig, payload_dict)
            defaulted.append({
                "final":              final,
                "sig":                sig,
                "row":                row,
                "payload_dict":       payload_dict,
                "violations":         violations,
                "default_brief":      default_brief,
                "default_assignee":   default_assignee_val,
            })

        # Optional rank pass over the ranks-4..N slice
        rank_passed = 0
        refined_by_key: dict[tuple, dict[str, Any]] = {}
        ranks_for_llm = defaulted[self.deterministic_top_n:]
        if self.rank_pass_enabled and ranks_for_llm:
            try:
                from .rank import rank_candidates
                rank_input = [
                    {
                        "signal_type":      d["sig"],
                        "signal_node_ids":  d["row"].get("source_block_ids") or [],
                        "impact_score":     d["final"],
                        "default_brief":    d["default_brief"],
                        "default_assignee": d["default_assignee"],
                        "default_align":    float(objs.get(d["sig"], 0.0)),
                        "payload_summary":  (d["row"].get("text") or "")[:200],
                    }
                    for d in ranks_for_llm
                ]
                refined = await rank_candidates(
                    candidates=rank_input,
                    constitution_objectives=objs,
                    model=self.rank_pass_model,
                )
                rank_passed = len(refined)
                for r in refined:
                    key = tuple(sorted(int(n) for n in (r.get("signal_node_ids") or [])))
                    refined_by_key[key] = r
            except Exception as e:
                logger.exception(
                    "[dream_scanner] rank pass failed (using defaults for "
                    "ranks %d+): %s",
                    self.deterministic_top_n + 1, e,
                )

        # Emit — top-N use template, ranks N+1..end use refined-if-available
        emitted_by_signal: dict[str, int] = {}
        skipped = 0
        for idx, d in enumerate(defaulted):
            sig = d["sig"]
            row = d["row"]
            tenant_id = row.get("user_id") or ""
            user_id = tenant_id
            final = d["final"]
            payload_dict = d["payload_dict"]
            violations = d["violations"]

            brief = d["default_brief"]
            assignee = d["default_assignee"]
            objective_align = float(objs.get(sig, 0.0))
            rank_pass_rationale = ""
            llm_dropped = False

            # Did rank pass cover this candidate?
            node_key = tuple(sorted(int(n) for n in (row.get("source_block_ids") or [])))
            if idx >= self.deterministic_top_n and node_key in refined_by_key:
                r = refined_by_key[node_key]
                if r.get("drop"):
                    llm_dropped = True
                else:
                    brief = r.get("refined_brief") or brief
                    assignee = r.get("refined_assignee") or assignee
                    objective_align = float(r.get("refined_objective_alignment") or objective_align)
                    rank_pass_rationale = r.get("rationale") or ""

            if not brief:
                skipped += 1
                continue
            if llm_dropped:
                # Spec §3.7.5: drops are deferred (decay applies), not
                # discarded — but we still don't emit them this cycle.
                # Skip create_proposal so they don't show as `pending`.
                skipped += 1
                continue

            spec = render_deliverable_spec(sig)
            # Reconstruct SignalPayload for emergency-lane check
            sp = SignalPayload(**{
                k: v for k, v in payload_dict.items()
                if k in SignalPayload.__dataclass_fields__
            })
            emerg = is_emergency_lane(sig, sp)
            rationale_prefix = "USER_USER " if emerg else ""
            rationale_tail = (
                f" — rank_pass: {rank_pass_rationale}"
                if rank_pass_rationale else
                (" (rank-pass refined)" if idx >= self.deterministic_top_n and node_key in refined_by_key else "")
            )

            try:
                # Phase 7 M3 — sanctioned split-writer: ledger row first,
                # then TCMM observation with cross-ref.  Async-aware.
                from ..memory.phase_7_writers import record_proposal_with_content
                await record_proposal_with_content(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    signal_type=sig,
                    signal_node_ids=row.get("source_block_ids") or [],
                    impact_score=final,
                    proposed_brief=brief,
                    proposed_assignee=assignee,
                    proposed_deliverable_spec=spec,
                    rationale=(
                        f"{rationale_prefix}{sig}: final_score={final:.3f} "
                        f"(auto-emit from dream aid={row.get('aid')}){rationale_tail}"
                    ),
                    objective_alignment=objective_align,
                    constraint_violations=violations,
                    emergency_lane=emerg,
                )
                # Update cursor + seen-keys
                key = (sig, tuple(sorted(row.get("source_block_ids") or [])))
                self._seen_keys.add(key)
                self._cursor[(tenant_id, user_id)] = max(
                    self._cursor.get((tenant_id, user_id), 0),
                    int(row.get("aid") or 0),
                )
                emitted_by_signal[sig] = emitted_by_signal.get(sig, 0) + 1
            except Exception as e:
                logger.exception(
                    f"[dream_scanner] create_proposal failed: {e}"
                )
                skipped += 1

        out = {
            "rows_scanned":  rows_scanned,
            "candidates":    len(candidates),
            "emitted":       sum(emitted_by_signal.values()),
            "skipped_dedup": len(candidates) - len(capped),
            "skipped_emit":  skipped,
            "rank_passed":   rank_passed,
            "by_signal":     emitted_by_signal,
        }
        if out["emitted"] > 0 or rank_passed > 0:
            logger.info(f"[dream_scanner] cycle: {out}")
        else:
            logger.debug(f"[dream_scanner] cycle: {out}")
        return out

    async def run(self) -> None:
        logger.info(
            f"[dream_scanner] starting; interval={self.interval_seconds:.0f}s "
            f"max_per_cycle={self.max_per_cycle} per_signal_cap={self.per_signal_cap} "
            f"importance_threshold={self.importance_threshold:.1f} "
            f"rank_pass_enabled={self.rank_pass_enabled} model={self.rank_pass_model}"
        )
        try:
            while not self._stop_evt.is_set():
                try:
                    await self._scan_once()
                    # Reset importance accumulator after a successful
                    # cycle so the next early-fire requires fresh signal.
                    self._importance_accum = 0.0
                except Exception as e:
                    logger.exception(f"[dream_scanner] scan error: {e}")
                # Race the timed-interval wait against the wake event.
                # Whichever fires first ends the wait — wake_evt for
                # importance-burst / fire_now, timeout for the regular
                # cadence.  stop_evt always wins (loop exits on next
                # iteration check).
                self._wake_evt.clear()
                wake_task = asyncio.ensure_future(self._wake_evt.wait())
                stop_task = asyncio.ensure_future(self._stop_evt.wait())
                try:
                    done, pending = await asyncio.wait(
                        {wake_task, stop_task},
                        timeout=self.interval_seconds,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for p in pending:
                        p.cancel()
                    if wake_task in done:
                        logger.debug(
                            "[dream_scanner] early-fire (importance burst / fire_now)"
                        )
                except Exception as e:
                    logger.exception(f"[dream_scanner] wait error: {e}")
        finally:
            logger.info("[dream_scanner] stopped")


__all__ = ["DreamScanner", "BLOCK_CLASS_TO_SIGNAL"]
