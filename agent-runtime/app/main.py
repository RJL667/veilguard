"""FastAPI entry — agent-runtime service.

Exposes:
  POST /agent/query     main entry; runs SDK query() with our middleware
  GET  /health          liveness probe (docker-compose healthcheck)
  GET  /agents          list registered personas (for sidebar UI)
  GET  /constitution    return parsed constitution (read-only)

Wiring:
  pii-proxy receives an Anthropic-bound request from LibreChat.
  If the request is "agent-runtime-enabled" (header or env flag), proxy
  forwards to http://agent-runtime:5000/agent/query.  Otherwise, proxy
  goes direct to api.anthropic.com (current behaviour).

  This service does NOT redact PII itself — that's pii-proxy's job and
  happens before the request reaches us.  We assume incoming content
  is already redacted.  We DO send audit rows to the same Lance table
  pii-proxy uses, so cost roll-up + replay queries work uniformly.
"""

import logging
import os
import sys

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__, config
from .personas.loader import load_personas, PersonaRegistry
from .runtime import run_agent_query
from .constitution.loader import load_constitution, ConstitutionError
from .workers import inbox_poller
from .a2a import router as _a2a_router
from .a2a_external import router as _a2a_external_router
from .ledger.schemas import _register_external_tables
_register_external_tables()

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agent-runtime")


# ── App + state ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Veilguard agent-runtime",
    version=__version__,
    description=(
        "Embedded Claude Agent SDK runtime. Owns the LLM-tool-LLM loop "
        "for Anthropic-bound conversations. PII proxy continues to handle "
        "redaction + multi-provider routing + audit."
    ),
)

# Phase 4 — A2A internal transport (AgentCard, /agents/<aid>/messages,
# /agents/<aid>/tasks/<id>).  Internal-only auth via X-Internal-Secret.
app.include_router(_a2a_router)
# Phase 5 — A2A external (X-API-Key + rate-limit + allow-list).
app.include_router(_a2a_external_router)

_persona_registry: PersonaRegistry | None = None
_constitution: dict | None = None


@app.on_event("startup")
async def _startup() -> None:
    """Validate config + load personas + parse constitution.

    Fail fast and loud if anything's missing — better than discovering
    via a 500 on the first real request.
    """
    # Wire the events broadcaster to this event loop so background
    # workers (LifecycleWorker, DreamScanner, etc.) running outside the
    # FastAPI loop can still publish SSE events to subscribers.
    import asyncio as _asyncio
    from . import events as _events
    _events.attach_main_loop(_asyncio.get_running_loop())

    errs = config.validate()
    if errs:
        for e in errs:
            logger.error(f"[startup] {e}")
        # Don't abort — health check should still respond so ops sees the
        # container is alive but in bad state.  /agent/query will 503.
        return

    # Phase 6 + 7 — proactively open every known ledger table so schema
    # migrations apply at startup (not on first real request).  Without
    # this, `agent_task_heartbeats` is never created until something
    # records a heartbeat, and `tcmm_obs_id` columns on task_proposals /
    # proposal_outcomes don't materialise until those tables get a
    # write.  Touching them here forces _migrate_phase_6_0 + lazy-create.
    try:
        from .ledger.store import LedgerStore
        from .ledger.schemas import TABLE_SCHEMAS
        _store = LedgerStore.get()
        for _tname in TABLE_SCHEMAS.keys():
            try:
                _store.table(_tname)
            except Exception as _te:
                logger.warning(f"[startup] ledger table {_tname} init failed: {_te}")
        logger.info(
            f"[startup] ledger init: {len(TABLE_SCHEMAS)} tables open; "
            f"Phase 6+7 migrations applied"
        )
    except Exception as e:
        logger.exception(f"[startup] ledger table init failed: {e}")

    global _persona_registry, _constitution

    try:
        _persona_registry = load_personas(config.AGENTS_DIR)
        logger.info(
            f"[startup] loaded {len(_persona_registry)} personas: "
            f"{', '.join(p.agent_id for p in _persona_registry.all())}"
        )
    except Exception as e:
        logger.exception(f"[startup] persona load failed: {e}")
        _persona_registry = PersonaRegistry({})
    # Expose for A2A router (clean injection over module globals)
    app.state.persona_registry = _persona_registry

    try:
        if config.CONSTITUTION_PATH.exists():
            _constitution = load_constitution(config.CONSTITUTION_PATH)
            objectives = _constitution.get("objectives", [])
            constraints = _constitution.get("constraints", [])
            logger.info(
                f"[startup] constitution loaded — "
                f"{len(objectives)} objectives, {len(constraints)} constraints"
            )
        else:
            logger.warning(
                f"[startup] CONSTITUTION.md not found at {config.CONSTITUTION_PATH} — "
                "proposals will run without objective alignment"
            )
            _constitution = None
    except ConstitutionError as e:
        logger.error(f"[startup] constitution parse failed: {e}")
        _constitution = None

    # Start the IC inbox poller (background asyncio task).
    # ICs (Researcher, Builder, critic-claim, critic-prose) receive
    # assigned Tasks via this loop, not via chat turns.
    # [INBOX_POLLER_OPT_OUT_2026_05_29]  In local dev a single user
    # turn was racing 3+ background researcher dispatches for the same
    # Anthropic OAuth client + thread pool, pushing wall-clock per
    # warm Director turn from ~1.5s to 5-12s.  Honour an env opt-out
    # so dev runs can disable background work and reclaim the foreground
    # latency budget.  Prod sets this to "false" (or leaves unset).
    _inbox_disabled = os.environ.get(
        "VEILGUARD_INBOX_POLLER_DISABLED", ""
    ).strip().lower() in ("1", "true", "yes")
    if _inbox_disabled:
        logger.info(
            "[startup] inbox poller DISABLED via "
            "VEILGUARD_INBOX_POLLER_DISABLED=1 — Director foreground "
            "turns will not compete with researcher/critic dispatches"
        )
    elif _persona_registry and len(_persona_registry) > 0:
        try:
            await inbox_poller.start(_persona_registry)
            logger.info("[startup] inbox poller started")
        except Exception as e:
            logger.exception(f"[startup] inbox poller failed to start: {e}")

    # [PHASE_3_LIFECYCLE_WORKER_2026_05_27]  Per spec §3.7.4.
    # Proposals decay 0.9× per cycle, auto-shelve below floor,
    # auto-expire after 7d TTL.  Interval is 1h by default — fine for
    # session-length scales; production VM can tune via env var.
    try:
        import asyncio as _asyncio
        from .proposals.lifecycle import LifecycleWorker
        _lifecycle_interval = float(os.environ.get(
            "VEILGUARD_PROPOSAL_LIFECYCLE_INTERVAL_S", "3600"
        ))
        global _lifecycle_worker
        _lifecycle_worker = LifecycleWorker(interval_seconds=_lifecycle_interval)
        _lifecycle_task = _asyncio.create_task(_lifecycle_worker.run())
        # Stash on app.state so the shutdown hook can stop it cleanly.
        app.state.lifecycle_worker = _lifecycle_worker
        app.state.lifecycle_task = _lifecycle_task
        logger.info(
            f"[startup] proposals.LifecycleWorker started "
            f"(interval={_lifecycle_interval:.0f}s)"
        )
    except Exception as e:
        logger.exception(f"[startup] LifecycleWorker failed to start: {e}")

    # [PHASE_3_DREAM_SCANNER_2026_05_27]  Per spec §3.7.0 — the
    # proposal-emission hook.  Instead of modifying dream_engine.py
    # (which lives in the TCMM tree), we scan the shared
    # `dream_archive.lance` table directly from agent-runtime on a
    # configurable cadence.  Cross-process safe because both TCMM
    # (writer) and agent-runtime (reader) hit the same on-disk Lance
    # store at /tcmm-data.  Opt-out via
    # VEILGUARD_DREAM_SCANNER_DISABLED=1 to keep the door open for
    # the canonical in-process hook if we choose to add it later.
    try:
        if os.environ.get("VEILGUARD_DREAM_SCANNER_DISABLED", "") not in ("1", "true", "yes"):
            from .proposals.dream_scanner import DreamScanner
            _scanner_interval = float(os.environ.get(
                "VEILGUARD_DREAM_SCANNER_INTERVAL_S", "600",
            ))
            scanner = DreamScanner(
                interval_seconds=_scanner_interval,
                constitution=_constitution or {},
            )
            scanner_task = _asyncio.create_task(scanner.run())
            app.state.dream_scanner = scanner
            app.state.dream_scanner_task = scanner_task
            logger.info(
                f"[startup] proposals.DreamScanner started "
                f"(interval={_scanner_interval:.0f}s)"
            )
        else:
            logger.info(
                "[startup] proposals.DreamScanner disabled via "
                "VEILGUARD_DREAM_SCANNER_DISABLED"
            )
    except Exception as e:
        logger.exception(f"[startup] DreamScanner failed to start: {e}")

    # [PHASE_4_CONSTITUTION_AMENDMENTS_2026_05_27]  Per spec §3.10 —
    # weekly scan for amendment-eligible lessons (confidence ≥ 0.75 AND
    # reinforcement ≥ 5).  Emits one task_proposal per such lesson so
    # the user reviews + applies the edit manually.  Idempotent on
    # lesson_id.
    try:
        if os.environ.get("VEILGUARD_CONSTITUTION_WORKER_DISABLED", "") not in ("1", "true", "yes"):
            from .proposals.constitution_amendments import ConstitutionAmendmentWorker
            _const_interval = float(os.environ.get(
                "VEILGUARD_CONSTITUTION_AMENDMENT_INTERVAL_S", str(7 * 24 * 3600),
            ))
            const_worker = ConstitutionAmendmentWorker(
                interval_seconds=_const_interval,
            )
            const_task = _asyncio.create_task(const_worker.run())
            app.state.constitution_amendment_worker = const_worker
            app.state.constitution_amendment_task = const_task
            logger.info(
                f"[startup] proposals.ConstitutionAmendmentWorker started "
                f"(interval={_const_interval:.0f}s)"
            )
        else:
            logger.info(
                "[startup] proposals.ConstitutionAmendmentWorker disabled via "
                "VEILGUARD_CONSTITUTION_WORKER_DISABLED"
            )
    except Exception as e:
        logger.exception(
            f"[startup] ConstitutionAmendmentWorker failed to start: {e}"
        )

    # [PHASE_3_SIGNAL_EMITTERS_2026_05_27]  Per spec §3.7.1 — the 2
    # missing emitter signals (low_stability_cluster +
    # stale_supersession_chain).  Implemented agent-runtime-side
    # scanning archive.lance → injecting synthetic dream_archive rows
    # that DreamScanner picks up next cycle.  Avoids dream-engine
    # surgery while delivering the spec contract.
    try:
        if os.environ.get("VEILGUARD_SIGNAL_EMITTERS_DISABLED", "") not in ("1", "true", "yes"):
            from .proposals.signal_emitters import SignalEmitterWorker
            _emitter_interval = float(os.environ.get(
                "VEILGUARD_SIGNAL_EMITTERS_INTERVAL_S", "86400",
            ))
            emitter_worker = SignalEmitterWorker(interval_seconds=_emitter_interval)
            emitter_task = _asyncio.create_task(emitter_worker.run())
            app.state.signal_emitter_worker = emitter_worker
            app.state.signal_emitter_task = emitter_task
            logger.info(
                f"[startup] proposals.SignalEmitterWorker started "
                f"(interval={_emitter_interval:.0f}s)"
            )
        else:
            logger.info(
                "[startup] proposals.SignalEmitterWorker disabled via "
                "VEILGUARD_SIGNAL_EMITTERS_DISABLED"
            )
    except Exception as e:
        logger.exception(f"[startup] SignalEmitterWorker failed to start: {e}")

    # [PHASE_3_LESSON_PROMOTION_2026_05_27]  Per spec §3.9.2 + §3.10 —
    # daily scan over dream_archive for reflective_heuristic /
    # recurring_ritual nodes that meet promotion criteria
    # (confidence ≥ 0.50, reinforcement ≥ 2, ≥2 distinct agents).
    # Writes them to org_memory as institutional lessons with
    # 180d expiry + 90d review_after.  Idempotent on promoted_from.
    try:
        if os.environ.get("VEILGUARD_LESSONS_WORKER_DISABLED", "") not in ("1", "true", "yes"):
            from .proposals.lessons import LessonPromotionWorker
            _lessons_interval = float(os.environ.get(
                "VEILGUARD_LESSONS_INTERVAL_S", "86400",
            ))
            lessons_worker = LessonPromotionWorker(interval_seconds=_lessons_interval)
            lessons_task = _asyncio.create_task(lessons_worker.run())
            app.state.lessons_worker = lessons_worker
            app.state.lessons_task = lessons_task
            logger.info(
                f"[startup] proposals.LessonPromotionWorker started "
                f"(interval={_lessons_interval:.0f}s)"
            )
        else:
            logger.info(
                "[startup] proposals.LessonPromotionWorker disabled via "
                "VEILGUARD_LESSONS_WORKER_DISABLED"
            )
    except Exception as e:
        logger.exception(f"[startup] LessonPromotionWorker failed to start: {e}")

    # [PHASE_3_RECALIBRATION_2026_05_27]  Per spec §3.7.2 end —
    # weekly weight recalibration.  Reads recent proposal_outcomes,
    # aggregates regret per (tenant, signal_type), nudges
    # DEFAULT_ALIGNMENT_VECTORS for that tenant in alignment_weights.
    # Idempotent — running with no new data leaves weights unchanged.
    try:
        if os.environ.get("VEILGUARD_RECAL_WORKER_DISABLED", "") not in ("1", "true", "yes"):
            from .proposals.recalibration import RecalibrationWorker
            _recal_interval = float(os.environ.get(
                "VEILGUARD_RECAL_INTERVAL_S", str(7 * 24 * 3600),  # 1 week
            ))
            recal_worker = RecalibrationWorker(interval_seconds=_recal_interval)
            recal_task = _asyncio.create_task(recal_worker.run())
            app.state.recal_worker = recal_worker
            app.state.recal_task = recal_task
            logger.info(
                f"[startup] proposals.RecalibrationWorker started "
                f"(interval={_recal_interval:.0f}s)"
            )
        else:
            logger.info(
                "[startup] proposals.RecalibrationWorker disabled via "
                "VEILGUARD_RECAL_WORKER_DISABLED"
            )
    except Exception as e:
        logger.exception(f"[startup] RecalibrationWorker failed to start: {e}")

    # [PHASE_3_DRIFT_WATCHDOG_2026_05_27]  Per spec §3.7.3 — daily
    # check for tenants whose proposal volume spikes >3× the trailing
    # 30d median.  Auto-pauses via proactive_config.pause(reason=
    # 'signal_quality_drift') so the user reviews before the queue
    # floods.  Idempotent — already-paused tenants don't re-pause.
    try:
        if os.environ.get("VEILGUARD_DRIFT_WATCHDOG_DISABLED", "") not in ("1", "true", "yes"):
            from .proposals.drift_watchdog import DriftWatchdog
            _drift_interval = float(os.environ.get(
                "VEILGUARD_DRIFT_INTERVAL_S", "86400",  # 1 day
            ))
            drift_watchdog = DriftWatchdog(interval_seconds=_drift_interval)
            drift_task = _asyncio.create_task(drift_watchdog.run())
            app.state.drift_watchdog = drift_watchdog
            app.state.drift_task = drift_task
            logger.info(
                f"[startup] proposals.DriftWatchdog started "
                f"(interval={_drift_interval:.0f}s)"
            )
        else:
            logger.info(
                "[startup] proposals.DriftWatchdog disabled via "
                "VEILGUARD_DRIFT_WATCHDOG_DISABLED"
            )
    except Exception as e:
        logger.exception(f"[startup] DriftWatchdog failed to start: {e}")

    # [PHASE_3_OUTCOMES_WORKER_2026_05_27]  Per spec §3.7.7 — daily
    # scan over `task_proposals` looking for approved proposals
    # whose linked Task is in a terminal state ≥30 days old.  Writes
    # one row per to `proposal_outcomes.lance` with regret_score =
    # task_cost / max(value_realized, ε).  Idempotent — already-
    # written outcomes get skipped.  Weekly recalibration (not yet
    # built) reads from this table.
    try:
        if os.environ.get("VEILGUARD_OUTCOMES_WORKER_DISABLED", "") not in ("1", "true", "yes"):
            from .proposals.outcomes import OutcomesWorker
            _outcomes_interval = float(os.environ.get(
                "VEILGUARD_OUTCOMES_INTERVAL_S", "86400",  # 1 day
            ))
            outcomes_worker = OutcomesWorker(interval_seconds=_outcomes_interval)
            outcomes_task = _asyncio.create_task(outcomes_worker.run())
            app.state.outcomes_worker = outcomes_worker
            app.state.outcomes_task = outcomes_task
            logger.info(
                f"[startup] proposals.OutcomesWorker started "
                f"(interval={_outcomes_interval:.0f}s)"
            )
        else:
            logger.info(
                "[startup] proposals.OutcomesWorker disabled via "
                "VEILGUARD_OUTCOMES_WORKER_DISABLED"
            )
    except Exception as e:
        logger.exception(f"[startup] OutcomesWorker failed to start: {e}")

    # [PHASE_7_5_STALL_DETECTOR_2026_05_28]  Magentic-One Progress
    # Ledger pattern.  Periodically scans in_progress tasks for stuck
    # rows (lease expired beyond grace / heartbeat silent / progress
    # ledger frozen) and emits `task_stalled` comments so a reviewing
    # agent or the sidebar can intervene.  Default 5min — sub-grace so
    # a stall lands within one cycle.  Disable via env override.
    try:
        if os.environ.get("VEILGUARD_STALL_DETECTOR_DISABLED", "") not in ("1", "true", "yes"):
            from .proposals.stall_detector import StallDetectorWorker
            _stall_interval = float(os.environ.get(
                "VEILGUARD_STALL_DETECTOR_INTERVAL_S", "300",  # 5 min
            ))
            stall_worker = StallDetectorWorker(interval_seconds=_stall_interval)
            stall_task = _asyncio.create_task(stall_worker.run())
            app.state.stall_worker = stall_worker
            app.state.stall_task = stall_task
            logger.info(
                f"[startup] proposals.StallDetectorWorker started "
                f"(interval={_stall_interval:.0f}s)"
            )
        else:
            logger.info(
                "[startup] proposals.StallDetectorWorker disabled via "
                "VEILGUARD_STALL_DETECTOR_DISABLED"
            )
    except Exception as e:
        logger.exception(f"[startup] StallDetectorWorker failed to start: {e}")


@app.on_event("shutdown")
async def _shutdown() -> None:
    """Stop background workers cleanly so in-flight work finishes."""
    try:
        await inbox_poller.stop()
        logger.info("[shutdown] inbox poller stopped")
    except Exception as e:
        logger.warning(f"[shutdown] inbox poller stop failed: {e}")
    # Stop the proposals lifecycle worker so its loop exits cleanly.
    try:
        worker = getattr(app.state, "lifecycle_worker", None)
        task = getattr(app.state, "lifecycle_task", None)
        if worker is not None:
            worker.stop()
        if task is not None:
            try:
                await task
            except Exception:
                pass
        logger.info("[shutdown] LifecycleWorker stopped")
    except Exception as e:
        logger.warning(f"[shutdown] LifecycleWorker stop failed: {e}")
    # Stop the DreamScanner.
    try:
        scanner = getattr(app.state, "dream_scanner", None)
        stask = getattr(app.state, "dream_scanner_task", None)
        if scanner is not None:
            scanner.stop()
        if stask is not None:
            try:
                await stask
            except Exception:
                pass
        logger.info("[shutdown] DreamScanner stopped")
    except Exception as e:
        logger.warning(f"[shutdown] DreamScanner stop failed: {e}")
    # Stop the ConstitutionAmendmentWorker.
    try:
        cw = getattr(app.state, "constitution_amendment_worker", None)
        ct = getattr(app.state, "constitution_amendment_task", None)
        if cw is not None:
            cw.stop()
        if ct is not None:
            try:
                await ct
            except Exception:
                pass
        logger.info("[shutdown] ConstitutionAmendmentWorker stopped")
    except Exception as e:
        logger.warning(f"[shutdown] ConstitutionAmendmentWorker stop failed: {e}")
    # Stop the SignalEmitterWorker.
    try:
        sew = getattr(app.state, "signal_emitter_worker", None)
        st = getattr(app.state, "signal_emitter_task", None)
        if sew is not None:
            sew.stop()
        if st is not None:
            try:
                await st
            except Exception:
                pass
        logger.info("[shutdown] SignalEmitterWorker stopped")
    except Exception as e:
        logger.warning(f"[shutdown] SignalEmitterWorker stop failed: {e}")
    # Stop the LessonPromotionWorker.
    try:
        lw = getattr(app.state, "lessons_worker", None)
        lt = getattr(app.state, "lessons_task", None)
        if lw is not None:
            lw.stop()
        if lt is not None:
            try:
                await lt
            except Exception:
                pass
        logger.info("[shutdown] LessonPromotionWorker stopped")
    except Exception as e:
        logger.warning(f"[shutdown] LessonPromotionWorker stop failed: {e}")
    # Stop the RecalibrationWorker.
    try:
        rw = getattr(app.state, "recal_worker", None)
        rt = getattr(app.state, "recal_task", None)
        if rw is not None:
            rw.stop()
        if rt is not None:
            try:
                await rt
            except Exception:
                pass
        logger.info("[shutdown] RecalibrationWorker stopped")
    except Exception as e:
        logger.warning(f"[shutdown] RecalibrationWorker stop failed: {e}")
    # Stop the DriftWatchdog.
    try:
        dw = getattr(app.state, "drift_watchdog", None)
        dt = getattr(app.state, "drift_task", None)
        if dw is not None:
            dw.stop()
        if dt is not None:
            try:
                await dt
            except Exception:
                pass
        logger.info("[shutdown] DriftWatchdog stopped")
    except Exception as e:
        logger.warning(f"[shutdown] DriftWatchdog stop failed: {e}")
    # Stop the OutcomesWorker.
    try:
        ow = getattr(app.state, "outcomes_worker", None)
        ot = getattr(app.state, "outcomes_task", None)
        if ow is not None:
            ow.stop()
        if ot is not None:
            try:
                await ot
            except Exception:
                pass
        logger.info("[shutdown] OutcomesWorker stopped")
    except Exception as e:
        logger.warning(f"[shutdown] OutcomesWorker stop failed: {e}")
    # Stop the StallDetectorWorker (Phase 7.5).
    try:
        sw = getattr(app.state, "stall_worker", None)
        st = getattr(app.state, "stall_task", None)
        if sw is not None:
            sw.stop()
        if st is not None:
            try:
                await st
            except Exception:
                pass
        logger.info("[shutdown] StallDetectorWorker stopped")
    except Exception as e:
        logger.warning(f"[shutdown] StallDetectorWorker stop failed: {e}")


# ── Routes ───────────────────────────────────────────────────────────────


@app.get("/audit/tcmm_source_kinds")
async def audit_tcmm_source_kinds(request: Request) -> JSONResponse:
    """AC-P7.6 runtime probe — scan TCMM `archive` for observations
    written by agent-runtime and assert each one carries a recognised
    per-writer marker prefix in its text body.

    Why text-prefix and not `source` column: TCMM ingest_turn stores its
    own `source` tag (`atomic_write` for assistant_text origin) and does
    NOT preserve `metadata.source` from the item payload.  So the only
    reliable per-row classification is the deterministic prefix every
    Phase 7 writer emits:

        record_proposal_with_content      → "[proposal:<id>]"
        record_outcome_with_narrative     → "[outcome:<id>]"
        promote_lesson_to_team_knowledge  → "[lesson]"
        record_discussion_comment         → "[task:<id>]"

    The allow-list mirrors AC-P7.6 §7.3:
        {lesson, proposal, outcome_narrative, discussion, agent_observation}

    Map prefix → source_kind for the audit:
        "[proposal:"  → "proposal"
        "[outcome:"   → "outcome_narrative"
        "[lesson]"    → "lesson"
        "[task:"      → "discussion" (record_discussion_comment) or
                        "agent_observation" (record_discussion_comment
                        note=True; we can't distinguish here, so map to
                        "discussion" which is the dominant case)
        anything else → flagged as a violation

    Returns:
        {
          "scanned":         int,  # rows examined (agent-runtime only)
          "by_source_kind":  {kind: count, ...},
          "violations":      [{aid, namespace, text_head}, ...],
          "violation_count": int,
          "allow_list":      [str, ...],
        }

    Intended to be polled by a daily cron.  Cheap: one Lance scan
    bounded by `?limit=N` (default 5000).
    """
    try:
        from .ledger.store import LedgerStore
        tbl = LedgerStore.get()._db.open_table("archive")
    except Exception as e:
        return JSONResponse(
            {"error": f"archive open failed: {e}",
             "scanned": 0, "violations": []},
            status_code=503,
        )

    try:
        limit = max(1, min(50000, int(request.query_params.get("limit", "5000"))))
    except ValueError:
        limit = 5000

    allow_list = {
        "lesson", "proposal", "outcome_narrative",
        "discussion", "agent_observation",
    }

    try:
        arr = tbl.search().limit(limit).to_arrow().to_pylist()
    except Exception as e:
        return JSONResponse(
            {"error": f"archive scan failed: {e}",
             "scanned": 0, "violations": []},
            status_code=503,
        )

    def _classify(text: str) -> str | None:
        t = (text or "").lstrip()
        if t.startswith("[proposal:"):
            return "proposal"
        if t.startswith("[outcome:"):
            return "outcome_narrative"
        if t.startswith("[lesson]"):
            return "lesson"
        if t.startswith("[task:"):
            return "discussion"
        return None

    # Pre-Phase-7 test fixtures live under `agent/test/*` namespaces
    # (e.g. `agent/test/emit-low-stab`).  They predate the writer
    # prefix convention and would otherwise pollute violation counts
    # indefinitely.  Operators can re-include them with `?include_test=1`.
    include_test = (request.query_params.get("include_test") or "").lower() in ("1", "true", "yes")
    test_fixture_skipped = 0

    violations: list[dict] = []
    by_source_kind: dict[str, int] = {}
    scanned = 0
    for row in arr:
        eb = (row.get("extracted_by") or "")
        ns = (row.get("namespace") or "")
        # Only agent-runtime rows.  Heuristic mirrors what the Phase 7
        # writers emit: extracted_by `agent:*` and/or namespace `agent_*`.
        if not (eb.startswith("agent:") or ns.startswith("agent_") or ns.startswith("agent/")):
            continue
        # Filter pre-Phase-7 test fixtures (legacy seed data) unless the
        # caller explicitly asked to include them.
        if not include_test and (
            ns.startswith("agent/test/")
            or eb == "agent:test"
        ):
            test_fixture_skipped += 1
            continue
        scanned += 1
        text = row.get("text") or ""
        kind = _classify(text)
        if kind is None:
            # Unclassifiable agent-runtime row.  Could be from a legacy
            # writer (pre-Phase-7 record_episode) or a future writer
            # whose prefix isn't registered.  Cap text_head length.
            violations.append({
                "aid":       row.get("aid"),
                "namespace": ns,
                "text_head": text[:60],
            })
            by_source_kind["<unclassified>"] = by_source_kind.get("<unclassified>", 0) + 1
            continue
        if kind not in allow_list:
            # Defensive: a Phase 7 writer added a prefix that wasn't
            # blessed.  Shouldn't be reachable today.
            violations.append({
                "aid":       row.get("aid"),
                "namespace": ns,
                "text_head": text[:60],
                "kind":      kind,
            })
        by_source_kind[kind] = by_source_kind.get(kind, 0) + 1

    body = {
        "scanned":               scanned,
        "allow_list":            sorted(allow_list),
        "by_source_kind":        by_source_kind,
        "violations":            violations[:50],
        "violation_count":       len(violations),
        "test_fixture_skipped":  test_fixture_skipped,
        "include_test_fixtures": include_test,
    }
    code = 200 if not violations else 422
    return JSONResponse(body, status_code=code)


@app.get("/teams")
async def list_teams_endpoint(request: Request) -> JSONResponse:
    """Phase 7.5 — enumerate all teams for (tenant, user).  Sidebar
    queries this to render the Teams panel."""
    qp = request.query_params
    tenant_id = qp.get("tenant_id") or ""
    user_id   = qp.get("user_id")   or ""
    if not (tenant_id and user_id):
        raise HTTPException(status_code=400, detail="tenant_id + user_id required")
    try:
        limit = max(1, min(500, int(qp.get("limit", "200"))))
    except ValueError:
        limit = 200
    from .ledger import teams as _teams
    try:
        rows = _teams.list_teams(
            tenant_id=tenant_id, user_id=user_id, limit=limit,
        )
    except Exception as e:
        return JSONResponse({"error": str(e), "teams": []}, status_code=200)
    return JSONResponse({
        "tenant_id": tenant_id, "user_id": user_id,
        "count": len(rows), "teams": rows,
    })


@app.get("/teams/{team_id}/cost")
async def team_cost_endpoint(team_id: str, request: Request) -> JSONResponse:
    """Phase 7.5 — live cost rollup for a team.  Sums every
    `agent_tasks.cost_attributed_usd` where `team_id` matches, refreshes
    `agent_teams.cost_attributed_cached_usd`, and returns the figures
    along with whether the team has crossed its budget envelope."""
    qp = request.query_params
    tenant_id = qp.get("tenant_id") or ""
    user_id   = qp.get("user_id")   or ""
    if not (tenant_id and user_id):
        raise HTTPException(status_code=400, detail="tenant_id + user_id required")
    from .ledger import teams as _teams
    team = _teams.get_team(team_id, tenant_id, user_id)
    if team is None:
        raise HTTPException(status_code=404, detail=f"team {team_id!r} not found")
    attributed = _teams.team_cost_attributed(
        team_id=team_id, tenant_id=tenant_id, user_id=user_id,
        refresh_cache=True,
    )
    exceeded, _, ceiling = _teams.budget_exceeded(
        team_id=team_id, tenant_id=tenant_id, user_id=user_id,
    )
    return JSONResponse({
        "team_id":        team_id,
        "name":           team.get("name"),
        "attributed_usd": round(attributed, 4),
        "budget_usd":     team.get("budget_usd"),
        "budget_cap":     team.get("budget_cap"),
        "ceiling_usd":    round(ceiling, 4),
        "exceeded":       exceeded,
        "pct_consumed":   (
            round(100.0 * attributed / ceiling, 1) if ceiling > 0 else None
        ),
        "lead_agent_id":     team.get("lead_agent_id"),
        "member_agent_ids":  team.get("member_agent_ids") or [],
    })


@app.get("/health")
async def health() -> JSONResponse:
    """Liveness + readiness probe.

    Returns 200 even if startup partially failed (e.g. CONSTITUTION.md
    missing).  Returns 503 only if the process is in a state that
    cannot serve /agent/query at all.
    """
    cfg_errs = config.validate()
    ready = not cfg_errs and _persona_registry is not None
    body = {
        "service": "agent-runtime",
        "version": __version__,
        "ready": ready,
        "personas_loaded": (
            len(_persona_registry) if _persona_registry else 0
        ),
        "constitution_loaded": _constitution is not None,
        "config_errors": cfg_errs,
    }
    code = 200 if ready else 503
    return JSONResponse(body, status_code=code)


@app.get("/agents")
async def list_agents() -> JSONResponse:
    """Return the registered persona set.  Used by sidebar Agents view."""
    if not _persona_registry:
        return JSONResponse({"agents": []})
    return JSONResponse({
        "agents": [
            {
                "agent_id": p.agent_id,
                "role": p.role,
                "manager_id": p.manager_id,
                "team_id": p.team_id,
                "model": p.model,
                "tool_count": len(p.tools),
            }
            for p in _persona_registry.all()
        ]
    })


@app.get("/proposals")
async def list_proposals(request: Request) -> JSONResponse:
    """Phase 3 — `task_proposals` queue view (read-only sidebar).

    Per spec §3.7.5: shows pending + deferred proposals sorted by
    decay_score DESC.  Caller MUST provide tenant_id + user_id query
    params; this is tenant-scoped data and we don't have a route-level
    auth context to derive them.

    Optional filters:
      - status: comma-separated subset of pending|deferred|approved|shelved|expired
      - limit:  default 50, max 200
      - include_escalated: when "1", surfaces a separate `escalated`
                           array of recurrence ≥ 5 candidates per §3.7.4

    The endpoint is dependency-tolerant: if Lance isn't reachable we
    return empty arrays + an error key rather than 500-ing the
    sidebar.
    """
    qp = request.query_params
    tenant_id = qp.get("tenant_id") or ""
    user_id   = qp.get("user_id")   or ""
    if not tenant_id or not user_id:
        return JSONResponse(
            {"error": "tenant_id and user_id query params required",
             "queue": [], "escalated": []},
            status_code=400,
        )

    # Parse limit + status filter
    try:
        limit = max(1, min(200, int(qp.get("limit", "50"))))
    except ValueError:
        limit = 50
    status_filter_raw = qp.get("status", "pending,deferred")
    statuses = [s.strip() for s in status_filter_raw.split(",") if s.strip()]
    valid = {"pending", "approved", "deferred", "shelved", "expired"}
    statuses = [s for s in statuses if s in valid]
    if not statuses:
        statuses = ["pending", "deferred"]
    include_escalated = qp.get("include_escalated", "") in ("1", "true", "yes")

    from .ledger.store import LedgerStore, ns_filter
    try:
        tbl = LedgerStore.get().table("task_proposals")
    except Exception as e:
        return JSONResponse(
            {"error": f"ledger unavailable: {e}",
             "queue": [], "escalated": []},
            status_code=200,  # don't 500 the sidebar
        )
    where = (
        f"{ns_filter(tenant_id, user_id)} "
        f"AND status IN ({', '.join(repr(s) for s in statuses)})"
    )
    try:
        arr = tbl.search().where(where).limit(limit).to_arrow()
    except Exception as e:
        return JSONResponse(
            {"error": f"scan failed: {e}",
             "queue": [], "escalated": []},
            status_code=200,
        )
    queue = [
        {col: arr.column(col)[i].as_py() for col in arr.column_names}
        for i in range(arr.num_rows)
    ]
    # Sort by decay_score DESC — newest-impactful at the top per §3.7.6
    queue.sort(key=lambda r: r.get("decay_score", 0.0) or 0.0, reverse=True)

    escalated: list[dict] = []
    if include_escalated:
        from .proposals.lifecycle import escalated_proposals
        escalated = escalated_proposals(
            tenant_id=tenant_id, user_id=user_id, limit=limit,
        )

    # [PHASE_3_EMERGENCY_LANE_2026_05_27]  Per spec §3.7.3 —
    # USER×USER contradiction_arc proposals are split out for the
    # sidebar's emergency tab.  They share storage with the regular
    # queue but the UI surfaces them with visual prominence.  We
    # infer emergency-lane status from `signal_type` +
    # `signal_node_ids`-adjacent metadata stored at proposal-emit
    # time (today: heuristic on signal_type alone; future: dedicated
    # `emergency_lane: bool` typed column once we ship a small Lance
    # migration).
    emergency: list[dict] = []
    remainder: list[dict] = []
    for q in queue:
        # Heuristic: contradiction_arc with both source_kind labels =
        # USER lives on the proposal under extras_json or the
        # rationale.  Today we mark proposal.rationale with a
        # standard prefix when emergency; absence = not emergency.
        rationale = (q.get("rationale") or "")
        sig_type = q.get("signal_type") or ""
        is_emerg = (
            sig_type == "contradiction_arc"
            and "USER_USER" in rationale
        )
        if is_emerg:
            emergency.append(q)
        else:
            remainder.append(q)

    return JSONResponse({
        "tenant_id":  tenant_id,
        "user_id":    user_id,
        "statuses":   statuses,
        "queue":      remainder,
        "emergency":  emergency,
        "escalated":  escalated,
        "count":      len(remainder),
        "emergency_count": len(emergency),
    })


@app.post("/proposals/emit")
async def proposal_emit(request: Request) -> JSONResponse:
    """Phase 3 — emit one proposal from a structured signal payload.

    This is the HTTP boundary the (future) dream-cycle hook calls
    when it identifies a generative signal worth surfacing.  Today
    it's also the entry point for admin tools / tests that seed
    proposals into the queue.

    Body:
      {
        "tenant_id": str,
        "user_id":   str,
        "signal_type": one of CONFIRMED_SIGNAL_TYPES,
        "signal_node_ids": list[int],
        "payload": dict   # subset of SignalPayload fields for this signal
        "brief_vars": dict  # variables for the brief template (gap_topic, etc.)
        "constraint_context": dict  # optional — passes to constitution evaluator
      }

    Behaviour:
      1. Build SignalPayload from `payload`
      2. Score via final_score() using loaded constitution objectives
         + evaluate_constraints() for the constraint_gate
      3. Build brief via render_brief() + deliverable_spec
      4. Resolve assignee via default_assignee()
      5. Mark `rationale` with `USER_USER` prefix if emergency-lane
      6. Persist via create_proposal()
      7. Return the new proposal + final_score breakdown

    Returns:
      {"proposal": {...}, "breakdown": {...}, "emergency_lane": bool}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")

    tenant_id = body.get("tenant_id") or ""
    user_id   = body.get("user_id")   or ""
    signal_type = body.get("signal_type") or ""
    signal_node_ids = body.get("signal_node_ids") or []
    payload_raw = body.get("payload") or {}
    brief_vars = body.get("brief_vars") or {}
    constraint_ctx = body.get("constraint_context") or {}

    if not (tenant_id and user_id and signal_type):
        raise HTTPException(
            status_code=400,
            detail="tenant_id, user_id, signal_type all required",
        )

    from .proposals.scoring import (
        SignalPayload,
        final_score as _final_score,
        ALL_SIGNAL_TYPES,
    )
    from .proposals.briefs import render_brief, render_deliverable_spec
    from .proposals.assignees import default_assignee
    from .proposals.constitution_bridge import (
        objectives_to_dict,
        evaluate_constraints,
    )
    from .proposals import is_emergency_lane
    from .ledger import proposals as _props

    if signal_type not in ALL_SIGNAL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"unknown signal_type: {signal_type!r}",
        )

    # Build the SignalPayload — only pass fields the dataclass knows about
    try:
        payload = SignalPayload(**{
            k: v for k, v in payload_raw.items()
            if k in SignalPayload.__dataclass_fields__
        })
    except TypeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"invalid payload fields: {e}",
        )

    # Constitution → scoring inputs
    objs = objectives_to_dict(_constitution or {})
    violations = evaluate_constraints(_constitution or {}, constraint_ctx)

    final, breakdown = _final_score(
        signal_type=signal_type,
        payload=payload,
        constitution_objectives=objs,
        constraint_violations=violations,
    )

    # Build brief + assignee
    brief = render_brief(signal_type, brief_vars)
    spec = render_deliverable_spec(signal_type)
    payload_dict = {f: getattr(payload, f) for f in SignalPayload.__dataclass_fields__}
    assignee = default_assignee(signal_type, payload_dict)

    emergency = is_emergency_lane(signal_type, payload)
    rationale_prefix = "USER_USER " if emergency else ""
    rationale = (
        f"{rationale_prefix}{signal_type}: final_score={final:.3f}"
        + (f"; violations={violations}" if violations else "")
    )

    # If brief is empty (unknown signal_type) the proposal is unusable.
    if not brief:
        raise HTTPException(
            status_code=400,
            detail=f"no brief template for signal_type {signal_type!r}",
        )

    # Phase 7 M3 — write via sanctioned split-writer so the TCMM
    # observation gets created with `extras.proposal_id` cross-ref and
    # the ledger row gets `tcmm_obs_id` back.  Falls back gracefully if
    # TCMM is unreachable: ledger row commits, tcmm_obs_id stays NULL.
    from .memory.phase_7_writers import record_proposal_with_content
    pid, _tcmm_obs_id = await record_proposal_with_content(
        tenant_id=tenant_id,
        user_id=user_id,
        signal_type=signal_type,
        signal_node_ids=list(signal_node_ids),
        impact_score=final,
        proposed_brief=brief,
        proposed_assignee=assignee,
        proposed_deliverable_spec=spec,
        rationale=rationale,
        objective_alignment=breakdown.get("objective_alignment"),
        constraint_violations=violations,
        constitution_version=(_constitution or {}).get("version"),
        emergency_lane=emergency,
    )
    created = _props.get_proposal(
        proposal_id=pid, tenant_id=tenant_id, user_id=user_id,
    )
    return JSONResponse({
        "proposal":        created,
        "breakdown":       breakdown,
        "emergency_lane":  emergency,
        "violations":      violations,
    })


@app.post("/proposals/{proposal_id}/convert")
async def proposal_convert(proposal_id: str, request: Request) -> JSONResponse:
    """Phase 3/4 — one-click "approve → create Task" flow.

    Spec §4.1 Director tool: `convert_proposal(proposal_id)` creates a
    Task on user approval and links it back to the proposal.  This
    replaces the two-step (create_task + decision) flow that's awkward
    from the sidebar.

    Body: {tenant_id, user_id}.  Reads the proposal's
    proposed_brief / proposed_assignee / proposed_deliverable_spec
    and uses them as the Task brief.

    Returns: {task_id, proposal_id, proposal: {…updated proposal row}}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    tenant_id = body.get("tenant_id") or ""
    user_id   = body.get("user_id")   or ""
    if not (tenant_id and user_id):
        raise HTTPException(status_code=400, detail="tenant_id + user_id required")
    from .ledger import proposals as _props
    from .ledger import tasks as _tasks

    p = _props.get_proposal(
        proposal_id=proposal_id, tenant_id=tenant_id, user_id=user_id,
    )
    if p is None:
        raise HTTPException(status_code=404, detail="proposal not found")
    if p.get("status") in ("approved", "shelved", "expired"):
        raise HTTPException(
            status_code=409,
            detail=f"proposal is already {p['status']!r}",
        )

    # Create the Task with proposal-derived defaults.
    brief = p.get("proposed_brief") or "(no brief)"
    assignee = p.get("proposed_assignee") or "researcher"
    spec = p.get("proposed_deliverable_spec") or ""
    try:
        new_tid = _tasks.create_task(
            tenant_id=tenant_id,
            user_id=user_id,
            owner_id=assignee,
            assigner_id="director",
            brief=brief,
            deliverable_spec=spec,
            inputs=[],
            origin="proactive",
            pattern="A",
            constitution_version=p.get("constitution_version"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"task create: {e}")

    # Link proposal → resulting task
    _props.approve_proposal(
        proposal_id=proposal_id, tenant_id=tenant_id, user_id=user_id,
        resulting_task_id=new_tid,
    )
    updated = _props.get_proposal(
        proposal_id=proposal_id, tenant_id=tenant_id, user_id=user_id,
    )
    return JSONResponse({
        "task_id":     new_tid,
        "proposal_id": proposal_id,
        "proposal":    updated,
    })


@app.post("/proposals/{proposal_id}/decision")
async def proposal_decision(proposal_id: str, request: Request) -> JSONResponse:
    """Phase 3 — apply a user/Director decision to a single proposal.

    Body: ``{tenant_id, user_id, action, ...kwargs}`` where action is
    one of:
      * ``approve``  — proposal becomes a Task.  Caller must include
        ``resulting_task_id`` (the create_task call is OUT of band;
        this endpoint only records the link).
      * ``shelve``   — user dismissed.  Caller must include
        ``reason: str`` recorded as `shelf_reason`.
      * ``defer``    — proposal stays alive, decays per cycle.  No
        kwargs.

    Returns the updated proposal row.  404 if the proposal isn't
    found in this tenant; 400 on bad action / missing kwargs.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    tenant_id = body.get("tenant_id") or ""
    user_id   = body.get("user_id")   or ""
    action    = body.get("action")    or ""
    if not tenant_id or not user_id:
        raise HTTPException(status_code=400, detail="tenant_id + user_id required")
    if action not in ("approve", "shelve", "defer"):
        raise HTTPException(
            status_code=400,
            detail=f"action must be approve|shelve|defer, got {action!r}",
        )

    from .ledger import proposals as _props
    # Confirm exists + tenant ownership
    existing = _props.get_proposal(
        proposal_id=proposal_id, tenant_id=tenant_id, user_id=user_id,
    )
    if existing is None:
        raise HTTPException(status_code=404, detail="proposal not found")

    if action == "approve":
        rid = body.get("resulting_task_id") or ""
        if not rid:
            raise HTTPException(
                status_code=400,
                detail="approve requires resulting_task_id (call create_task first)",
            )
        _props.approve_proposal(
            proposal_id=proposal_id, tenant_id=tenant_id, user_id=user_id,
            resulting_task_id=rid,
        )
    elif action == "shelve":
        reason = body.get("reason") or "shelved_by_user"
        _props.shelve_proposal(
            proposal_id=proposal_id, tenant_id=tenant_id, user_id=user_id,
            reason=reason,
        )
    elif action == "defer":
        _props.defer_proposal(
            proposal_id=proposal_id, tenant_id=tenant_id, user_id=user_id,
        )

    updated = _props.get_proposal(
        proposal_id=proposal_id, tenant_id=tenant_id, user_id=user_id,
    )
    return JSONResponse({"proposal": updated})


@app.get("/replay/{task_id}")
async def replay_get(task_id: str, request: Request) -> JSONResponse:
    """Phase 4 — Replay v1 (§Phase 4).

    Returns the full decision-path payload for ``task_id``: task,
    ancestors, siblings, comments, decision points (review_decision
    pivots), the proposal that spawned it, its outcome row, and
    counterfactual proposals from the same signal cluster.
    """
    qp = request.query_params
    tenant_id = qp.get("tenant_id") or ""
    user_id   = qp.get("user_id")   or ""
    if not (tenant_id and user_id):
        raise HTTPException(status_code=400, detail="tenant_id + user_id required")
    from .replay import replay_task
    payload = replay_task(task_id=task_id, tenant_id=tenant_id, user_id=user_id)
    if payload.get("task") is None:
        raise HTTPException(status_code=404, detail=payload.get("error", "not found"))
    return JSONResponse(payload)


@app.post("/documents/snapshot")
async def documents_snapshot(request: Request) -> JSONResponse:
    """Phase 4 — snapshot a file before in-place edit (§Phase 4).

    Body: {target: path, by_agent_id?, note?}
    Returns the audit record (snapshot_path, is_repo_tracked, etc.)
    Callers persist this into agent_tasks.extras_json.document_history.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    target = body.get("target") or ""
    if not target:
        raise HTTPException(status_code=400, detail="target required")
    from .documents import snapshot_before_edit
    rec = snapshot_before_edit(
        target,
        by_agent_id=body.get("by_agent_id") or "",
        note=body.get("note") or "",
    )
    return JSONResponse({"snapshot": rec})


@app.get("/documents/history")
async def documents_history(request: Request) -> JSONResponse:
    """List `.history/<name>/*.snap` snapshots for a non-repo file."""
    target = request.query_params.get("target") or ""
    if not target:
        raise HTTPException(status_code=400, detail="target required")
    from .documents import list_history
    return JSONResponse({"target": target, "snapshots": list_history(target)})


@app.get("/documents/authority")
async def documents_authority(request: Request) -> JSONResponse:
    """Return the Task that holds merge authority for a path."""
    qp = request.query_params
    target = qp.get("target") or ""
    tenant_id = qp.get("tenant_id") or ""
    user_id   = qp.get("user_id")   or ""
    if not (target and tenant_id and user_id):
        raise HTTPException(
            status_code=400,
            detail="target + tenant_id + user_id required",
        )
    from .documents import merge_authority_holder
    rec = merge_authority_holder(
        target_path=target, tenant_id=tenant_id, user_id=user_id,
    )
    return JSONResponse({"target": target, "authority": rec})


@app.get("/tasks/{task_id}/ledger")
async def task_ledger_get(task_id: str, request: Request) -> JSONResponse:
    """Phase 4 — read the Magentic-One-style task_ledger +
    progress_ledger for a task (§Phase 4 §3.10)."""
    qp = request.query_params
    tenant_id = qp.get("tenant_id") or ""
    user_id   = qp.get("user_id")   or ""
    if not (tenant_id and user_id):
        raise HTTPException(status_code=400, detail="tenant_id + user_id required")
    from .ledger.task_progress import read_ledgers
    return JSONResponse(read_ledgers(
        task_id=task_id, tenant_id=tenant_id, user_id=user_id,
    ))


@app.post("/tasks/{task_id}/ledger")
async def task_ledger_set(task_id: str, request: Request) -> JSONResponse:
    """Phase 4 — overwrite task_ledger (facts / guesses / plan).

    Body: {tenant_id, user_id, facts: [str], guesses: [str],
           plan: [str], by_agent_id?: str}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    tenant_id = body.get("tenant_id") or ""
    user_id   = body.get("user_id")   or ""
    if not (tenant_id and user_id):
        raise HTTPException(status_code=400, detail="tenant_id + user_id required")
    from .ledger.task_progress import write_task_ledger
    try:
        out = write_task_ledger(
            task_id=task_id, tenant_id=tenant_id, user_id=user_id,
            facts=body.get("facts") or [],
            guesses=body.get("guesses") or [],
            plan=body.get("plan") or [],
            by_agent_id=body.get("by_agent_id") or "",
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return JSONResponse({"task_ledger": out})


@app.post("/tasks/{task_id}/progress")
async def task_progress_append(task_id: str, request: Request) -> JSONResponse:
    """Phase 4 — append one entry to progress_ledger.

    Body: {tenant_id, user_id, step, observation, plan_change?, by_agent_id?}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    tenant_id = body.get("tenant_id") or ""
    user_id   = body.get("user_id")   or ""
    if not (tenant_id and user_id):
        raise HTTPException(status_code=400, detail="tenant_id + user_id required")
    step = body.get("step") or ""
    obs  = body.get("observation") or ""
    if not (step and obs):
        raise HTTPException(status_code=400, detail="step + observation required")
    from .ledger.task_progress import append_progress
    try:
        out = append_progress(
            task_id=task_id, tenant_id=tenant_id, user_id=user_id,
            step=step, observation=obs,
            plan_change=body.get("plan_change"),
            by_agent_id=body.get("by_agent_id") or "",
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return JSONResponse({"progress_ledger": out})


@app.get("/lessons/review_queue")
async def lessons_review_queue(request: Request) -> JSONResponse:
    """Phase 3/4 — sidebar "Lessons due for review" queue (§3.10).

    Post-M1 cutover: reads lessons from TCMM archive (via
    `memory.lessons_reader.find_lessons_due_for_review`) instead of
    the dropped org_memory Lance table.  Users decide keep / amend /
    retire per row.
    """
    qp = request.query_params
    tenant_id = qp.get("tenant_id") or ""
    user_id   = qp.get("user_id")   or ""
    if not (tenant_id and user_id):
        raise HTTPException(status_code=400, detail="tenant_id + user_id required")
    try:
        limit = max(1, min(500, int(qp.get("limit", "100"))))
    except ValueError:
        limit = 100
    from .memory.lessons_reader import find_lessons_due_for_review
    try:
        rows = find_lessons_due_for_review(
            tenant_id=tenant_id, user_id=user_id, limit=limit,
        )
    except Exception as e:
        return JSONResponse({"error": f"reader: {e}", "lessons": []}, status_code=200)
    return JSONResponse({
        "tenant_id":   tenant_id,
        "user_id":     user_id,
        "count":       len(rows),
        "lessons":     rows,
    })


@app.get("/lessons/amendment_candidates")
async def lessons_amendment_candidates(request: Request) -> JSONResponse:
    """Phase 4 — surfaces lessons eligible for constitution-amendment
    promotion (§3.10): confidence ≥ 0.75 AND reinforcement_count ≥ 5.

    Post-M1 cutover: reads from TCMM via
    `memory.lessons_reader.find_amendment_eligible_lessons`.  User
    reviews these on the sidebar; on accept, the constitution file
    gets amended (manual today; future: amend tool with diff review).
    Read-only.
    """
    qp = request.query_params
    tenant_id = qp.get("tenant_id") or ""
    user_id   = qp.get("user_id")   or ""
    if not (tenant_id and user_id):
        raise HTTPException(status_code=400, detail="tenant_id + user_id required")
    from .memory.lessons_reader import find_amendment_eligible_lessons
    from .proposals.lessons import (
        CONST_AMENDMENT_CONFIDENCE_MIN,
        CONST_AMENDMENT_REINFORCEMENT_MIN,
    )
    try:
        rows = find_amendment_eligible_lessons(
            tenant_id=tenant_id, user_id=user_id,
            confidence_min=CONST_AMENDMENT_CONFIDENCE_MIN,
            reinforcement_min=CONST_AMENDMENT_REINFORCEMENT_MIN,
            limit=200,
        )
    except Exception as e:
        return JSONResponse(
            {"error": f"reader: {e}", "candidates": []},
            status_code=200,
        )
    rows.sort(key=lambda r: r.get("confidence", 0), reverse=True)
    return JSONResponse({
        "tenant_id":            tenant_id,
        "user_id":              user_id,
        "thresholds":           {
            "confidence":      CONST_AMENDMENT_CONFIDENCE_MIN,
            "reinforcement":   CONST_AMENDMENT_REINFORCEMENT_MIN,
        },
        "count":                len(rows),
        "candidates":           rows,
    })


@app.post("/lessons/{lesson_id}/decision")
async def lesson_decision(lesson_id: str, request: Request) -> JSONResponse:
    """User decides on a lesson up for review (§3.10).

    Body: {tenant_id, user_id, action: keep|amend|retire, note?: str}
      * keep:   refresh review_after by 90d, keep status=active
      * amend:  caller-side updates trigger/rule, then we call this
                with action=keep + note explaining what changed
      * retire: status=retired
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    tenant_id = body.get("tenant_id") or ""
    user_id   = body.get("user_id")   or ""
    action    = body.get("action")    or ""
    if not (tenant_id and user_id):
        raise HTTPException(status_code=400, detail="tenant_id + user_id required")
    if action not in ("keep", "retire"):
        raise HTTPException(
            status_code=400,
            detail="action must be keep|retire (use keep after editing for amend)",
        )
    # Post-M1 cutover: org_memory is gone.  We resolve the lesson via
    # the TCMM-archive reader, then append a `[lesson_kept]` or
    # `[lesson_retired]` observation rather than mutating in place.
    # The reader applies last-write-wins (retirement) and re-derives
    # review_after from the count of keep events.
    from .memory.lessons_reader import list_lessons
    import time as _t

    candidates = list_lessons(
        tenant_id=tenant_id, user_id=user_id, status="all", limit=1000,
    )
    target = next((L for L in candidates if L["id"] == lesson_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail="lesson not found")

    now = _t.time()
    marker = "[lesson_retired]" if action == "retire" else "[lesson_kept]"
    body_text = (
        f"{marker} trigger={target['trigger']}\n"
        f"note={(body.get('note') or '')[:300]}\n"
        f"ts={now:.0f}"
    )
    try:
        from .middleware.tcmm import observe_agent_output
        await observe_agent_output(
            conversation_id=f"team/knowledge/{tenant_id}",
            user_id=user_id,
            agent_id="critic-prose",
            text=body_text,
            source="lesson",
            channel="team_knowledge",  # [CHANNEL_2026_05_29] spec §6.4
        )
    except Exception as e:
        return JSONResponse(
            {"error": f"observation failed: {e}", "lesson": target},
            status_code=502,
        )

    # Reflect new status in the returned lesson row.
    target["status"] = "retired" if action == "retire" else "active"
    if action == "keep":
        target["review_after"] = now + 90 * 86400.0

    # [LESSON_SSE_2026_05_28]  Surface the user decision to the sidebar
    # Lessons tab — closes the §3.10.5 spec TODO for
    # lesson_status_changed.  Broadcast AFTER the TCMM observation has
    # been written (failure already 502'd above).
    try:
        from .events import broadcast
        broadcast({
            "type":       "lesson_status_changed",
            "tenant_id":  tenant_id,
            "user_id":    user_id,
            "id":         lesson_id,
            "action":     action,
            "new_status": target["status"],
            "trigger":    (target.get("trigger") or "")[:200],
        })
    except Exception as _bcast_err:
        # Sidebar broadcast is best-effort — never fail the decision
        # write because the SSE pipe hiccupped.
        logger.debug(f"[lessons] status_changed broadcast failed: {_bcast_err}")

    return JSONResponse({"lesson": target})


@app.get("/work_items")
async def work_items_view(request: Request) -> JSONResponse:
    """Phase 4 — `work_items_v` UNION view (§3.10.2).

    Materialises the unified "active work + pending proposals +
    lessons due for review" feed the sidebar shows.  Returns a
    sorted list combining:
      * agent_tasks (status in open|accepted|in_progress|review)
      * task_proposals (status in pending|deferred)
      * org_memory lessons due for review

    Each row carries a `_kind` discriminator (`task` | `proposal` |
    `lesson`) so the frontend can render the right card.
    """
    qp = request.query_params
    tenant_id = qp.get("tenant_id") or ""
    user_id   = qp.get("user_id")   or ""
    if not (tenant_id and user_id):
        raise HTTPException(status_code=400, detail="tenant_id + user_id required")
    try:
        limit = max(1, min(500, int(qp.get("limit", "200"))))
    except ValueError:
        limit = 200

    # [WORK_ITEMS_KIND_FILTER_2026_05_28]  Optional `kind=` query param
    # narrows the UNION to a single source — saves the Lance scan on
    # the others.  Comma-separated subset of {task, proposal, lesson};
    # absent / empty / "all" → fetch all kinds (legacy behaviour).
    # Closes the open-work bullet in §3.10.5 (clients no longer need
    # to fetch + filter post-hoc).
    valid_kinds = {"task", "proposal", "lesson"}
    kind_raw = (qp.get("kind") or "").strip().lower()
    if kind_raw in ("", "all"):
        kinds_filter: set[str] = set(valid_kinds)
    else:
        kinds_filter = {k.strip() for k in kind_raw.split(",") if k.strip() in valid_kinds}
        if not kinds_filter:
            raise HTTPException(
                status_code=400,
                detail=f"kind must be a subset of {sorted(valid_kinds)} (or 'all')",
            )

    # [WORK_ITEMS_TASK_STATUS_2026_05_29]  Optional `task_status=` query param
    # lets the sidebar switch between the active work queue and a done/all
    # audit view WITHOUT changing the proposal/lesson semantics.
    #   absent / "active" → open|accepted|in_progress|review  (default queue)
    #   "done"            → done
    #   "cancelled"       → cancelled
    #   "all"             → no status filter (every task status)
    #   else              → comma-separated explicit subset of the valid set
    # Only affects the `task` source; proposals/lessons keep their own
    # pending/review feed.
    _VALID_TASK_STATUSES = {
        "open", "accepted", "in_progress", "review", "done", "cancelled",
    }
    _ACTIVE_TASK_STATUSES = ("open", "accepted", "in_progress", "review")
    ts_raw = (qp.get("task_status") or "").strip().lower()
    if ts_raw in ("", "active"):
        task_statuses: tuple[str, ...] | None = _ACTIVE_TASK_STATUSES
    elif ts_raw == "all":
        task_statuses = None  # no status filter
    else:
        _parsed = tuple(
            s.strip() for s in ts_raw.split(",")
            if s.strip() in _VALID_TASK_STATUSES
        )
        task_statuses = _parsed or _ACTIVE_TASK_STATUSES

    from .ledger.store import LedgerStore, ns_filter
    try:
        store = LedgerStore.get()
    except Exception as e:
        return JSONResponse(
            {"error": f"ledger: {e}", "items": []}, status_code=200,
        )

    items: list[dict] = []
    # tasks
    if "task" in kinds_filter:
        try:
            tbl = store.table("agent_tasks")
            if task_statuses is None:
                _task_where = ns_filter(tenant_id, user_id)
            else:
                _in = ",".join(f"'{s}'" for s in task_statuses)
                _task_where = (
                    f"{ns_filter(tenant_id, user_id)} AND status IN ({_in})"
                )
            arr = (
                tbl.search()
                .where(_task_where)
                .limit(limit)
                .to_arrow()
            )
            for i in range(arr.num_rows):
                row = {c: arr.column(c)[i].as_py() for c in arr.column_names}
                row["_kind"] = "task"
                row["_sort_ts"] = row.get("updated_ts", 0.0)
                items.append(row)
        except Exception as e:
            logger.warning(f"[work_items] tasks scan failed: {e}")
    # proposals
    if "proposal" in kinds_filter:
        try:
            tbl = store.table("task_proposals")
            arr = (
                tbl.search()
                .where(
                    f"tenant_id = '{tenant_id}' AND user_id = '{user_id}' "
                    f"AND status IN ('pending','deferred')"
                )
                .limit(limit)
                .to_arrow()
            )
            for i in range(arr.num_rows):
                row = {c: arr.column(c)[i].as_py() for c in arr.column_names}
                row["_kind"] = "proposal"
                row["_sort_ts"] = row.get("decay_score", 0.0)
                items.append(row)
        except Exception as e:
            logger.warning(f"[work_items] proposals scan failed: {e}")
    # lessons due for review — post-M1 cutover, reads from TCMM
    if "lesson" in kinds_filter:
        try:
            from .memory.lessons_reader import find_lessons_due_for_review
            rows = find_lessons_due_for_review(
                tenant_id=tenant_id, user_id=user_id, limit=limit,
            )
            for r in rows:
                r["_kind"] = "lesson"
                r["_sort_ts"] = r.get("review_after", 0.0)
                items.append(r)
        except Exception as e:
            logger.warning(f"[work_items] lessons scan failed: {e}")

    # Sort: tasks by recency, proposals by decay_score DESC, lessons by
    # how-overdue.  Cross-kind, just use _sort_ts DESC for now — UI
    # can re-group by _kind.
    items.sort(key=lambda r: r.get("_sort_ts", 0.0) or 0.0, reverse=True)
    items = items[:limit]
    counts = {"task": 0, "proposal": 0, "lesson": 0}
    for r in items:
        counts[r["_kind"]] = counts.get(r["_kind"], 0) + 1
    return JSONResponse({
        "tenant_id": tenant_id, "user_id": user_id,
        "kinds":     sorted(kinds_filter),
        "counts":    counts,
        "items":     items,
        "total":     len(items),
    })


@app.get("/events")
async def events_stream(request: Request) -> StreamingResponse:
    """Phase 3/4 — SSE push for sidebar live updates.

    The Work Queue sidebar opens an EventSource on this endpoint after
    its initial /work_items GET.  We then push one SSE message per
    ledger mutation in the (tenant_id, user_id) namespace:

      * proposal_created / proposal_status_changed
      * task_created / task_status_changed / task_comment_added
      * lesson_created / lesson_status_changed
        (since 2026-05-28; see `[LESSON_SSE_2026_05_28]` markers)

    Heartbeat every 25s to keep proxies from idling the connection.

    Per-connection asyncio.Queue in app/events.py.  In-process only —
    we'll move to Redis pub/sub when agent-runtime is sharded.
    """
    from .events import event_stream
    qp = request.query_params
    tenant_id = qp.get("tenant_id") or ""
    user_id   = qp.get("user_id")   or ""
    if not (tenant_id and user_id):
        raise HTTPException(status_code=400, detail="tenant_id + user_id required")
    return StreamingResponse(
        event_stream(tenant_id, user_id),
        media_type="text/event-stream",
        headers={
            # Disable proxy buffering — required for SSE through reverse
            # proxies / Docker networks.  Without this, nginx/Caddy may
            # buffer the entire response and the client sees nothing
            # until the connection ends.
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection":     "keep-alive",
        },
    )


@app.get("/proposals/{proposal_id}")
async def proposal_detail(proposal_id: str, request: Request) -> JSONResponse:
    """Phase 3/4 — full proposal detail for sidebar drill-in.

    Returns the proposal row plus a scoring-breakdown summary, the
    related dream node ids, and (if the proposal has been approved) the
    resulting Task's brief + status so the drill-in can deep-link.

    Lance reads run in a thread executor — the FastAPI event loop is
    shared with inbox_poller streaming, which can starve sync I/O
    inside this handler for tens of seconds.  Offloading lets the loop
    keep servicing SSE events while we wait for the disk.  Fixed an
    ECONNABORTED → "HTTP 502" the sidebar showed when a user clicked a
    proposal while another task was mid-stream (2026-05-27).
    """
    import asyncio as _asyncio
    qp = request.query_params
    tenant_id = qp.get("tenant_id") or ""
    user_id   = qp.get("user_id")   or ""
    if not (tenant_id and user_id):
        raise HTTPException(status_code=400, detail="tenant_id + user_id required")
    from .ledger.proposals import get_proposal
    from .ledger.tasks import get_task
    p = await _asyncio.to_thread(get_proposal, proposal_id, tenant_id, user_id)
    if p is None:
        raise HTTPException(status_code=404, detail="proposal not found")
    resulting = None
    if p.get("resulting_task_id"):
        resulting = await _asyncio.to_thread(
            get_task, p["resulting_task_id"], tenant_id, user_id,
        )
    return JSONResponse({"proposal": p, "resulting_task": resulting})


@app.get("/tasks/{task_id}")
async def task_detail(task_id: str, request: Request) -> JSONResponse:
    """Phase 3/4 — full task detail with comment chain + outputs.

    Returns: task row, ordered comment chain (status changes + agent
    notes), outputs list, parent ref, children refs, and optional
    `conversation_id` from extras_json for deep-link into LibreChat.

    Comment chain is what the drill-in renders as the agent's activity
    feed — every status transition + IC's accept/submit + Critic's
    decision shows up here.

    Lance reads run in a thread executor (same reason as proposal_detail).
    """
    import asyncio as _asyncio
    qp = request.query_params
    tenant_id = qp.get("tenant_id") or ""
    user_id   = qp.get("user_id")   or ""
    if not (tenant_id and user_id):
        raise HTTPException(status_code=400, detail="tenant_id + user_id required")
    from .ledger.tasks import get_task
    from .ledger.comments import list_comments
    from .ledger.store import LedgerStore, ns_filter
    task = await _asyncio.to_thread(get_task, task_id, tenant_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    cmts = await _asyncio.to_thread(
        list_comments, task_id=task_id, tenant_id=tenant_id, user_id=user_id,
    )
    # Children for the lineage tree.
    def _scan_children() -> list[dict]:
        try:
            ttbl = LedgerStore.get().table("agent_tasks")
            ca = (
                ttbl.search()
                .where(f"{ns_filter(tenant_id, user_id)} AND parent_id = '{task_id}'")
                .limit(50)
                .to_arrow()
            )
            return [
                {c: ca.column(c)[i].as_py() for c in ("id", "brief", "status", "owner_id")}
                for i in range(ca.num_rows)
            ]
        except Exception as e:
            logger.debug(f"[task_detail] children scan failed: {e}")
            return []
    children = await _asyncio.to_thread(_scan_children)
    # Extract conversation_id from extras_json if the IC stamped one
    # (agents call observe(conversation_id=...) when they pick up the
    # task, and the wrapper stamps the same value on the task row).
    conv_id = None
    extras = task.get("extras_json")
    if extras:
        try:
            import json as _json
            ex = _json.loads(extras)
            conv_id = ex.get("conversation_id")
        except Exception:
            pass
    return JSONResponse({
        "task":            task,
        "comments":        cmts,
        "children":        children,
        "conversation_id": conv_id,
    })


@app.get("/proactive_config")
async def proactive_config_get(request: Request) -> JSONResponse:
    """Phase 3 — read per-tenant proactive-stream config (§3.7.3).

    Missing rows return synthesised defaults (is_default=True in
    payload).  Workers + sidebar use this to decide whether to scan /
    surface for a given tenant.
    """
    qp = request.query_params
    tenant_id = qp.get("tenant_id") or ""
    user_id   = qp.get("user_id")   or ""
    if not (tenant_id and user_id):
        raise HTTPException(status_code=400, detail="tenant_id + user_id required")
    from .ledger.proactive_config import get_or_default
    cfg = get_or_default(tenant_id, user_id)
    return JSONResponse(cfg.to_dict())


@app.post("/proactive_config")
async def proactive_config_set(request: Request) -> JSONResponse:
    """Upsert per-tenant proactive-stream config.  Partial updates OK.

    Body shape:
      {
        "tenant_id": str, "user_id": str,
        "proactive_stream_enabled": bool?,
        "proactive_cycles_per_day": int?,
        "proactive_approval_cap_per_day": int?,
        "cost_ceiling_per_tenant_per_day_usd": float?,
        "paused": bool?,
        "paused_reason": str?
      }
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    tenant_id = body.get("tenant_id") or ""
    user_id   = body.get("user_id")   or ""
    if not (tenant_id and user_id):
        raise HTTPException(status_code=400, detail="tenant_id + user_id required")
    # Distinguish "key absent" (leave unchanged) from "key present with
    # null" (explicitly clear) — pass only the keys the caller sent.
    from .ledger.proactive_config import upsert
    kwargs = {
        "tenant_id": tenant_id,
        "user_id":   user_id,
    }
    for k in (
        "proactive_stream_enabled",
        "proactive_cycles_per_day",
        "proactive_approval_cap_per_day",
        "cost_ceiling_per_tenant_per_day_usd",
        "paused",
        "paused_reason",
    ):
        if k in body:
            kwargs[k] = body[k]
    cfg = upsert(**kwargs)
    return JSONResponse(cfg.to_dict())


@app.get("/constitution")
async def get_constitution() -> JSONResponse:
    """Return the parsed constitution (read-only).  Used for replay/audit
    UIs that need to show what objectives a decision was scored against.
    """
    if not _constitution:
        raise HTTPException(status_code=404, detail="constitution not loaded")
    return JSONResponse(_constitution)


@app.post("/agent/query")
async def agent_query(req: Request) -> StreamingResponse:
    """Main entry.  Runs SDK query() with all middleware wired in.

    Expected JSON body:
      {
        "conversation_id": "conv-...",     # cid
        "user_id": "user-...",
        "tenant_id": "tenant-...",
        "agent_id": "director",            # which persona drives this turn
        "messages": [...],                  # Anthropic-shape messages
        "stream": true                       # SSE if true, JSON if false
      }

    Returns SSE events (text/event-stream):
      data: {"type": "text_delta", "text": "..."}
      data: {"type": "tool_call", "tool": "...", "id": "..."}
      data: {"type": "tool_result", "id": "...", "content": "..."}
      data: {"type": "status_ping", "text": "..."}    # Director status
      data: {"type": "subagent_start", "agent_id": "..."}
      data: {"type": "subagent_end", "agent_id": "...", "summary": "..."}
      data: {"type": "final_result", "result": "..."}
      data: {"type": "usage", "tokens_in": N, "cache_read": M, ...}
    """
    if not _persona_registry:
        raise HTTPException(
            status_code=503,
            detail="persona registry not loaded; check /health",
        )

    try:
        body = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")

    required = ("conversation_id", "user_id", "tenant_id", "agent_id", "messages")
    missing = [k for k in required if k not in body]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"missing required fields: {missing}",
        )

    agent_id = body["agent_id"]
    persona = _persona_registry.get(agent_id)
    if persona is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown agent_id: {agent_id}",
        )

    # Dispatch into runtime.py — which owns all the middleware composition.
    # Streaming is the default; non-streaming returns the same generator
    # collected into one JSON response.
    stream = body.get("stream", True)
    if stream:
        # [STREAM_BYTES_FIX_2026_05_28] run_agent_query yields dicts.
        # StreamingResponse needs bytes/str.  Without wrapping the
        # async-gen, FastAPI silently closes the connection after 0
        # bytes — which manifests in LibreChat as "An error occurred
        # while processing the request: terminated".  Fix: encode each
        # event as SSE `data: <json>\n\n` text frame.
        import json as _json

        async def _sse_wrap():
            try:
                async for ev in run_agent_query(
                    persona=persona,
                    conversation_id=body["conversation_id"],
                    user_id=body["user_id"],
                    tenant_id=body["tenant_id"],
                    messages=body["messages"],
                    registry=_persona_registry,
                    constitution=_constitution,
                    client_tools=body.get("tools") or [],
                ):
                    if not isinstance(ev, dict):
                        continue
                    line = "data: " + _json.dumps(ev, default=str) + "\n\n"
                    yield line.encode("utf-8")
            except Exception as e:
                # Surface as a final SSE error event so LibreChat
                # renders a real message instead of the cryptic
                # "terminated" string.
                err = {"type": "error", "code": "stream_error",
                       "message": f"{type(e).__name__}: {e}"}
                yield ("data: " + _json.dumps(err) + "\n\n").encode("utf-8")

        return StreamingResponse(
            _sse_wrap(),
            media_type="text/event-stream",
        )

    # Non-streaming fallback (for tests, debugging) — collect into list.
    import time as _qt
    _q0 = _qt.perf_counter(); _q_rs = _q_re = None
    events = []
    async for ev in run_agent_query(
        persona=persona,
        conversation_id=body["conversation_id"],
        user_id=body["user_id"],
        tenant_id=body["tenant_id"],
        messages=body["messages"],
        registry=_persona_registry,
        constitution=_constitution,
        client_tools=body.get("tools") or [],
    ):
        if isinstance(ev, dict):
            if ev.get("type") == "run_start" and _q_rs is None:
                _q_rs = _qt.perf_counter()
            elif ev.get("type") == "run_end":
                _q_re = _qt.perf_counter()
        events.append(ev)
    _q_end = _qt.perf_counter()
    _ms = lambda a, b: (b - a) * 1000 if (a is not None and b is not None) else -1
    logger.info(
        "[QUERY-PERF] entry->run_start=%.0fms run_start->run_end=%.0fms "
        "run_end->return=%.0fms total=%.0fms",
        _ms(_q0, _q_rs), _ms(_q_rs, _q_re), _ms(_q_re, _q_end), _ms(_q0, _q_end),
    )
    return JSONResponse({"events": events})
