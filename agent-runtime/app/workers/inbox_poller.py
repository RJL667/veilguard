"""IC inbox polling loop — picks up assigned Tasks and dispatches.

The reactive stream goes user → Director → final_synthesis with Director
spawning subagents inline via the SDK's `Agent` tool.  That's fine for
foreground turns.

For BACKGROUND tasks (Director Pattern D, or proactive-stream-approved
Tasks an IC needs to pick up later), an IC isn't sitting in a chat turn
waiting.  Instead, this poller watches `agent_tasks` for newly-assigned
work and dispatches it through `runtime.run_agent_query()` with the
assigned IC's persona.

Architecture:
  - One poller per agent-runtime process; polls all tenants
  - Per-task lease via the `lease_owner` / `lease_until` columns to
    prevent double-dispatch if multiple agent-runtime replicas run
  - Idempotent: re-dispatching a task whose status moved on is a no-op
  - Bounded concurrency (N in-flight dispatches max)

Trade-off: Lance doesn't have a native event/notification system, so
this is poll-based.  Interval default = 5s; tunable via env.  A
production replacement would use the dream-cycle worker pattern (TCMM
already has polling workers; we could reuse the harness).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import TYPE_CHECKING, Optional

from ..config import LEDGER_DB_PATH
from ..middleware import tenant
from ..personas.loader import PersonaRegistry, PersonaSpec
from ..ledger.store import LedgerStore, ns_filter

if TYPE_CHECKING:
    pass

logger = logging.getLogger("agent-runtime.workers.inbox_poller")


# ── Configuration ───────────────────────────────────────────────────────

POLL_INTERVAL_S = 5.0

# [LEDGER_AUTO_COMPACT_2026_05_31] The poller scans agent_task_heartbeats /
# agent_tasks / task_proposals / task_comments every cycle. Each row write
# (especially heartbeats — one per worker per beat) appends a Lance fragment;
# left unchecked these tables reach 1000s of fragments and a single
# `.to_arrow()` scan blocks the event loop for 2-3s, which QUEUES inbound
# /agent/query requests behind it (measured 2026-05-31: warm chat turns
# ballooned from ~2s to 4-5s purely from this contention, even though the
# Anthropic call + render + redact were all fast). Mirror TCMM's
# semantic_polling_worker `_maybe_compact`: every interval, ONE pass of
# compact_files()+cleanup_old_versions() per table, run in a thread so it
# never blocks the loop. Disable with AGENT_LEDGER_AUTO_COMPACT=0.
_LEDGER_AUTO_COMPACT = os.environ.get("AGENT_LEDGER_AUTO_COMPACT", "1") != "0"
_LEDGER_COMPACT_INTERVAL_S = float(
    os.environ.get("AGENT_LEDGER_COMPACT_INTERVAL_S", "300")
)
_LEDGER_COMPACT_TABLES = (
    "agent_task_heartbeats",
    "agent_tasks",
    "task_proposals",
    "task_comments",
)

# [DISPATCH_TIMING_ENV_2026_05_29] Both the lease and the dispatch timeout
# are env-tunable. The prod defaults (lease 300s / timeout 270s) assume fast
# Lance; the local Windows bind-mount runs Lance ~10x slower, so a single
# researcher dispatch + TCMM recall + critic review can blow past 270s and
# the task gets WRONGLY force-cancelled (caught live 2026-05-29: a cost
# analysis reached `review` then was killed by the 270s timeout). Bump
# VEILGUARD_LEASE_DURATION_S / VEILGUARD_DISPATCH_TIMEOUT_S on slow envs.
# INVARIANT: dispatch timeout MUST stay < lease duration so the timeout
# fires before the lease expires (else the poller re-claims a still-running
# task → double dispatch).
LEASE_DURATION_S = float(os.environ.get("VEILGUARD_LEASE_DURATION_S", "300"))

# Phase 6.2 — per-persona concurrency caps (replaces the previous global
# dispatch cap which serialized everything).  Each cap is independent:
# researcher at 8/8 must NOT starve a builder claim (AC-16
# starvation-independence test).  Old global is gone from source
# (AC-14 regex-absence enforces).
PERSONA_CAPS: dict[str, int] = {
    "researcher":        8,
    "builder":           6,
    "critic-claim":      4,
    "critic-prose":      4,
    "phishing-analyst":  2,
    "threat-analyst":    2,
    "report-writer":     2,
}

# Total concurrency is the sum of per-persona caps; used by the global
# semaphore as a hard upper bound.  Per-persona enforcement happens at
# claim time (see `_can_claim_for_persona`).
_TOTAL_CONCURRENT_DISPATCHES = sum(PERSONA_CAPS.values())

# Hard ceiling on how long a single _dispatch is allowed to run before
# we tear it down and cancel the task.  Smaller than LEASE_DURATION_S
# so the timeout fires BEFORE the lease expires (avoids the case where
# the poller re-claims the same task while the previous dispatch is
# still running).  Caught a real bug 2026-05-25 where critic-claim
# entered a read_file→LLM loop on a stale deliverable and spun for 11+
# minutes without emitting review_decision; _in_flight retained the
# task ID, blocking re-claim even after lease expiry.  Now we force-
# cancel and emit a `dispatch_timeout` comment so the operator can
# see what hung.
DISPATCH_TIMEOUT_S = float(
    os.environ.get("VEILGUARD_DISPATCH_TIMEOUT_S", "270")
)
# Enforce the invariant even if the env vars are set inconsistently: a
# dispatch must never outlive its lease.
if DISPATCH_TIMEOUT_S >= LEASE_DURATION_S:
    DISPATCH_TIMEOUT_S = max(30.0, LEASE_DURATION_S - 30.0)

# Only these personas get inbox polling.  Director runs reactive only,
# never picks up background tasks.  Consultants are pull-only.
ELIGIBLE_OWNERS = frozenset({
    "researcher",
    "builder",
    "critic-claim",
    "critic-prose",
})


# ── Poller ──────────────────────────────────────────────────────────────


class InboxPoller:
    """Background asyncio task watching agent_tasks for IC assignments.

    Lifecycle:
      poller = InboxPoller(registry)
      task = asyncio.create_task(poller.run())
      ...
      poller.stop()
      await task

    Behaviour:
      Every POLL_INTERVAL_S seconds, scan agent_tasks for rows where:
        - owner_id IN ELIGIBLE_OWNERS
        - status = 'open'
        - lease_until IS NULL OR lease_until < now()

      For each match, atomically claim by setting lease_owner + lease_until
      via Lance update with a compare-and-swap clause.  If we win the
      claim, dispatch via runtime.run_agent_query() in a background task.
    """

    def __init__(self, registry: PersonaRegistry):
        self._registry = registry
        self._stop_evt = asyncio.Event()
        # Phase 6.2 — total-cap semaphore is the sum of per-persona caps;
        # actual per-persona limits enforced in `_can_claim_for_persona`.
        self._semaphore = asyncio.Semaphore(_TOTAL_CONCURRENT_DISPATCHES)
        self._worker_id = f"worker-{uuid.uuid4().hex[:8]}"
        self._in_flight: set[str] = set()      # task_ids currently dispatching
        self._in_flight_by_persona: dict[str, int] = {}  # Phase 6.2 — per-persona counter
        self._claimed_history: set[str] = set()  # all task_ids ever claimed (telemetry)
        self._startup_sweep_done = False       # F2 — gate the one-shot orphan sweep
        self._last_ledger_compact_ts = 0.0     # [LEDGER_AUTO_COMPACT_2026_05_31]

    def _can_claim_for_persona(self, persona: str) -> bool:
        """Phase 6.2 — per-persona cap check.  Returns True iff there's
        free slot for this persona.  Independence guarantee (AC-16):
        researcher at 8/8 must NOT starve a builder."""
        cap = PERSONA_CAPS.get(persona, 0)
        in_flight = self._in_flight_by_persona.get(persona, 0)
        return in_flight < cap

    def _acquire_persona_slot(self, persona: str) -> None:
        self._in_flight_by_persona[persona] = (
            self._in_flight_by_persona.get(persona, 0) + 1
        )

    def _release_persona_slot(self, persona: str) -> None:
        cur = self._in_flight_by_persona.get(persona, 0)
        self._in_flight_by_persona[persona] = max(0, cur - 1)

    def stop(self) -> None:
        self._stop_evt.set()

    async def run(self) -> None:
        """Main poll loop.  Returns when stop() is called."""
        logger.info(
            f"[inbox_poller] starting; worker_id={self._worker_id}, "
            f"interval={POLL_INTERVAL_S}s, "
            f"per_persona_caps={PERSONA_CAPS}, "
            f"total_concurrent={_TOTAL_CONCURRENT_DISPATCHES}"
        )
        try:
            while not self._stop_evt.is_set():
                try:
                    # [F2_RESTART_ORPHAN_SWEEP_2026_05_27]  On the very
                    # first poll after process start, look for tasks
                    # that were ``in_progress`` or ``review`` when the
                    # previous process died — their ``lease_until`` is
                    # in the past AND no current worker has them in
                    # ``_in_flight``.  Without this, an agent-runtime
                    # restart leaves tasks dangling indefinitely: F2's
                    # ``_force_cancel_on_timeout`` only fires for tasks
                    # the CURRENT worker dispatched.  Caught 2026-05-27
                    # by an orphan "CHILD B" task that sat
                    # ``in_progress`` for 31 minutes across a restart.
                    if not self._startup_sweep_done:
                        try:
                            await self._sweep_restart_orphans()
                        except Exception as e:
                            logger.exception(
                                f"[inbox_poller] startup sweep error: {e}"
                            )
                        self._startup_sweep_done = True
                    # Phase 6.3 — heartbeat sweep runs every poll cycle.
                    # Auto-reclaim tasks whose last heartbeat is older
                    # than lease_ttl_s (default 5min).  Cheap: one Lance
                    # scan, picks the freshest row per task_id.
                    try:
                        await self._sweep_stale_heartbeats()
                    except Exception as e:
                        logger.debug(
                            f"[inbox_poller] heartbeat sweep error: {e}"
                        )
                    # [LEDGER_AUTO_COMPACT_2026_05_31] Keep the ledger tables
                    # de-fragmented so the per-cycle scans above stay sub-second
                    # and never block the event loop / queue chat requests.
                    try:
                        await self._maybe_compact_ledger_tables()
                    except Exception as e:
                        logger.debug(
                            f"[inbox_poller] ledger compaction error: {e}"
                        )
                    await self._poll_once()
                except Exception as e:
                    logger.exception(f"[inbox_poller] poll error: {e}")
                try:
                    await asyncio.wait_for(
                        self._stop_evt.wait(), timeout=POLL_INTERVAL_S
                    )
                except asyncio.TimeoutError:
                    pass  # normal — keep polling
        finally:
            logger.info(
                f"[inbox_poller] stopping; "
                f"in_flight={len(self._in_flight)}"
            )

    async def _sweep_restart_orphans(self) -> None:
        """One-shot startup pass: cancel tasks left dangling from a
        previous process.  Criteria:
          * status in ('in_progress','review') — actively being worked
            on at process death.
          * lease_until < now - 60  — the previous worker's lease has
            been expired for at least a minute, so we're safe assuming
            it really did die (vs. just being a couple seconds behind).
          * updated_ts < now - 270  — more than the F2 dispatch
            timeout has elapsed; if it WERE still alive we'd have
            seen an update.

        Tasks matching are flipped to ``cancelled`` and stamped with a
        ``restart_orphan`` blocker comment so the audit log shows why
        they died.
        """
        try:
            store = LedgerStore.get(str(LEDGER_DB_PATH))
            tbl = store._db.open_table("agent_tasks")
        except Exception as e:
            logger.debug(f"[inbox_poller] sweep: ledger not ready: {e}")
            return
        now = time.time()
        cutoff_lease = now - 60.0
        cutoff_update = now - 270.0
        where = (
            f"status IN ('in_progress','review') "
            f"AND lease_until < {cutoff_lease} "
            f"AND updated_ts < {cutoff_update}"
        )
        try:
            arr = tbl.search().where(where).limit(50).to_arrow()
        except Exception as e:
            logger.warning(f"[inbox_poller] sweep scan failed: {e}")
            return
        if arr.num_rows == 0:
            logger.info(
                "[inbox_poller] startup sweep: no restart orphans"
            )
            return
        logger.warning(
            f"[inbox_poller] startup sweep: found {arr.num_rows} restart "
            f"orphan(s); cancelling"
        )
        from ..ledger import comments as _cm
        for i in range(arr.num_rows):
            tid = arr.column("id")[i].as_py()
            owner = arr.column("owner_id")[i].as_py()
            tenant_id = arr.column("tenant_id")[i].as_py()
            user_id = arr.column("user_id")[i].as_py()
            try:
                tbl.update(
                    where=f"id = '{tid}'",
                    values={
                        "status": "cancelled",
                        "updated_ts": now,
                        "lease_until": 0.0,
                        "lease_owner": "",
                    },
                )
                _cm.add_comment(
                    task_id=tid,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    author_id="inbox-poller",
                    kind="blocker_raised",
                    body=(
                        f"restart_orphan: task was {arr.column('status')[i].as_py()} "
                        f"when the previous agent-runtime process died "
                        f"(lease expired > 60s ago, no update in > 270s); "
                        f"cancelled by startup sweep on worker {self._worker_id}."
                    ),
                )
                logger.warning(
                    f"[inbox_poller]   restart-orphan cancelled: {tid} "
                    f"(owner={owner})"
                )
            except Exception as e:
                logger.exception(
                    f"[inbox_poller] sweep: failed to cancel {tid}: {e}"
                )

    async def _maybe_compact_ledger_tables(self) -> None:
        """Periodically de-fragment the ledger tables this poller scans.

        Every ``_LEDGER_COMPACT_INTERVAL_S`` (default 5min), run one
        ``compact_files()`` + ``cleanup_old_versions()`` pass per table.
        Row-per-write growth (heartbeats especially) otherwise drives the
        fragment count into the thousands, at which point the per-cycle
        ``.to_arrow()`` scans in ``_sweep_stale_heartbeats`` / ``_poll_once``
        take seconds and block the event loop — directly inflating chat-turn
        latency (measured 2026-05-31). The compaction itself runs in a worker
        thread via ``asyncio.to_thread`` so even a slow pass never stalls the
        loop. Online + MVCC-safe (Lance handles concurrent readers/writers;
        a lost commit race just retries next interval). Cheap no-op between
        intervals: one timestamp compare.
        """
        if not _LEDGER_AUTO_COMPACT:
            return
        now = time.time()
        if now - self._last_ledger_compact_ts < _LEDGER_COMPACT_INTERVAL_S:
            return
        self._last_ledger_compact_ts = now
        await asyncio.to_thread(self._compact_ledger_tables_sync)

    def _compact_ledger_tables_sync(self) -> None:
        """Blocking compaction body — always called inside a thread.

        Uses lancedb's native ``Table.optimize(cleanup_older_than=...)``
        which compacts fragments AND prunes superseded versions in one call,
        with NO pylance dependency (the agent-runtime image ships lancedb but
        not pylance, so the lower-level ``to_lance().optimize.compact_files()``
        path raises ImportError here — verified 2026-05-31).
        """
        try:
            store = LedgerStore.get(str(LEDGER_DB_PATH))
        except Exception:
            return
        import datetime as _dt
        for tname in _LEDGER_COMPACT_TABLES:
            try:
                tbl = store._db.open_table(tname)
            except Exception:
                continue  # table not created yet — nothing to compact
            try:
                t0 = time.time()
                # Aggressive retention (0s) is safe — we don't use Lance
                # time-travel on the ledger tables.
                tbl.optimize(cleanup_older_than=_dt.timedelta(seconds=0))
                logger.info(
                    "[inbox_poller] auto-compacted %s in %.2fs",
                    tname, time.time() - t0,
                )
            except Exception as e:
                # Commit race against an active writer ("Retryable commit
                # conflict") — fine, retry next interval. Never let it kill
                # the poll loop.
                logger.debug(
                    "[inbox_poller] compact %s skipped: %s", tname, e
                )

    async def _sweep_stale_heartbeats(self) -> None:
        """Phase 6.3 — find tasks whose last heartbeat is older than the
        lease TTL and force-cancel them with an audit comment.

        Stale = (now - last_beat_at) > lease_ttl_s.  Workers who fail
        to beat (crash, OOM, hang) leak their claims; without this
        sweep the task sits `in_progress` forever and downstream work
        blocks.  Phase 6.3 closes the dead-worker hole that Phase 6.0.1
        turn cap doesn't cover (turn cap counts LLM turns; dead workers
        accrue zero turns so the cap never fires).
        """
        try:
            store = LedgerStore.get(str(LEDGER_DB_PATH))
        except Exception:
            return
        try:
            hb_tbl = store._db.open_table("agent_task_heartbeats")
        except Exception:
            # Table missing on first boot — schema gets created on
            # first write.  Not fatal.
            return
        try:
            tasks_tbl = store._db.open_table("agent_tasks")
        except Exception:
            return
        now = time.time()
        from ..runtime_health import is_stale, HeartbeatRow
        # Scan heartbeats, build a {task_id: latest_row} map.
        try:
            hb_arr = hb_tbl.to_arrow()
        except Exception:
            return
        latest_by_task: dict[str, dict] = {}
        for i in range(hb_arr.num_rows):
            tid = hb_arr.column("task_id")[i].as_py()
            ts = hb_arr.column("last_beat_at")[i].as_py() or 0.0
            cur = latest_by_task.get(tid)
            if cur is None or (cur["last_beat_at"] or 0.0) < ts:
                latest_by_task[tid] = {
                    "task_id": tid,
                    "worker_id": hb_arr.column("worker_id")[i].as_py(),
                    "tenant_id": hb_arr.column("tenant_id")[i].as_py(),
                    "user_id":   hb_arr.column("user_id")[i].as_py(),
                    "last_beat_at": ts,
                    "lease_ttl_s":  hb_arr.column("lease_ttl_s")[i].as_py() or 300.0,
                }
        if not latest_by_task:
            return
        # For each task in `in_progress` / `accepted`, check freshness.
        active_statuses = ("in_progress", "accepted", "review", "blocked")
        try:
            tasks_arr = (
                tasks_tbl.search()
                .where(
                    f"status IN ({', '.join(repr(s) for s in active_statuses)})"
                )
                .to_arrow()
            )
        except Exception:
            return
        reclaimed = 0
        from ..ledger import comments as _cm
        from ..ledger.store import ns_filter
        for i in range(tasks_arr.num_rows):
            tid = tasks_arr.column("id")[i].as_py()
            if tid in self._in_flight:
                continue  # our own worker; don't sweep
            beat = latest_by_task.get(tid)
            if not beat:
                continue  # no heartbeat at all — could be pre-Phase-6.3 task
            row = HeartbeatRow(
                task_id=beat["task_id"],
                worker_id=beat["worker_id"],
                last_beat_at=beat["last_beat_at"],
                lease_ttl_s=beat["lease_ttl_s"],
            )
            if not is_stale(row, now=now):
                continue
            # Stale — reclaim with an audit comment.
            elapsed = now - beat["last_beat_at"]
            try:
                tasks_tbl.update(
                    where=(
                        f"{ns_filter(beat['tenant_id'], beat['user_id'])} "
                        f"AND id = '{tid}'"
                    ),
                    values={
                        "status":      "cancelled",
                        "updated_ts":  now,
                        "lease_owner": "",
                        "lease_until": 0.0,
                    },
                )
                _cm.add_comment(
                    task_id=tid,
                    tenant_id=beat["tenant_id"],
                    user_id=beat["user_id"],
                    author_id="inbox-poller",
                    kind="blocker_raised",
                    body=(
                        f"lease_expired: worker {beat['worker_id']!r} did "
                        f"not heartbeat for {elapsed:.1f}s (lease_ttl="
                        f"{beat['lease_ttl_s']:.0f}s); reclaimed by "
                        f"{self._worker_id}."
                    ),
                )
                reclaimed += 1
            except Exception as e:
                logger.warning(
                    f"[inbox_poller] heartbeat-sweep reclaim failed "
                    f"for {tid}: {e}"
                )
        if reclaimed:
            logger.info(
                f"[inbox_poller] heartbeat sweep: reclaimed {reclaimed} "
                f"orphaned task(s) with stale lease"
            )

    async def _poll_once(self) -> None:
        """One poll cycle: find claimable tasks, dispatch them.

        Phase 6.7 — checks APR circuit breaker before claiming new work.
        If APR has been below the floor for the rolling window, halts
        new dispatches and waits for operator unblock via apr_resume().
        In-flight tasks continue their current turn unmolested.
        """
        # Phase 6.7 — APR circuit breaker.
        try:
            from ..runtime_health import apr_should_pause_dispatch, apr_snapshot
            if apr_should_pause_dispatch():
                snap = apr_snapshot()
                logger.warning(
                    f"[inbox_poller] APR circuit breaker active "
                    f"(apr={snap.get('apr')!r}, n_samples={snap.get('n_samples')}); "
                    f"skipping dispatch this cycle. "
                    f"Operator must call apr_resume() to unblock."
                )
                return
        except Exception as e:
            logger.debug(f"[inbox_poller] APR check failed (continuing): {e}")
        try:
            store = LedgerStore.get(str(LEDGER_DB_PATH))
        except Exception as e:
            # On dev / first-boot the Lance dir may not exist yet.  Wait
            # for the next interval.
            logger.debug(f"[inbox_poller] ledger not ready: {e}")
            return

        # IMPORTANT: bypass LedgerStore's cached table handle.  Lance's
        # table handle holds a fixed dataset version — rows inserted by
        # OTHER processes (e.g. the create_task tool called from a
        # different container, or our injection test script) won't be
        # visible until we re-open.  Cached handle is fine for in-
        # process writes (Lance refreshes on update) but breaks cross-
        # process visibility.  Re-opening every 5 sec is cheap.
        try:
            tbl = store._db.open_table("agent_tasks")
        except Exception as e:
            logger.debug(f"[inbox_poller] re-open agent_tasks failed: {e}")
            return
        eligible_owners_sql = ", ".join(f"'{o}'" for o in ELIGIBLE_OWNERS)
        now = time.time()

        # Find candidate tasks.  We don't filter by tenant here — one
        # poller serves all tenants in a process.  The dispatch step
        # re-establishes per-task tenant context.
        #
        # Lance where-clause parser doesn't support `IS NULL`.  We use
        # sentinel values: tasks at rest have lease_until=0 (created
        # by ledger.tasks.create_task).  An active lease has
        # lease_until > now.  Once a lease expires (now > lease_until),
        # the task becomes claimable again.
        # Pollable states:
        #   open       — fresh assignment, never claimed
        #   review     — submit_for_review reassigned to critic, awaiting pickup
        #   in_progress — STALE in_progress (lease expired = original dispatch
        #                  died OR critic returned changes_requested and the
        #                  task transitioned to in_progress with owner restored
        #                  to the original IC who needs to be re-dispatched).
        #
        # The `lease_until < now` predicate is the safety net for
        # in_progress: a task currently being worked on has lease_until
        # in the future, so it's invisible to this query.  Only stale
        # in_progress (where the lease has expired OR was reset to 0
        # by review_decision changes_requested) gets re-claimed.
        where = (
            f"owner_id IN ({eligible_owners_sql}) "
            f"AND status IN ('open', 'review', 'in_progress') "
            f"AND lease_until < {now}"
        )

        try:
            arr = tbl.search().where(where).limit(20).to_arrow()
        except Exception as e:
            logger.warning(f"[inbox_poller] scan failed (where={where!r}): {e}")
            return

        if arr.num_rows == 0:
            logger.debug(f"[inbox_poller] no claimable tasks; where={where!r}")
            return

        logger.info(
            f"[inbox_poller] found {arr.num_rows} claimable task(s); "
            f"attempting claims as {self._worker_id}"
        )

        for i in range(arr.num_rows):
            task_id = arr.column("id")[i].as_py()
            if task_id in self._in_flight:
                continue  # already dispatching

            # Phase 6.2 — per-persona cap check.  If this persona is
            # saturated, skip and let a different persona's row through;
            # this is what enforces starvation-independence (AC-16).
            row_owner = arr.column("owner_id")[i].as_py() or ""
            if not self._can_claim_for_persona(row_owner):
                logger.debug(
                    f"[inbox_poller] persona cap reached for "
                    f"{row_owner!r}; skipping {task_id} this cycle"
                )
                continue

            # [F7_DEPENDENCY_AWARE_CLAIM_2026_05_26] If this task's
            # `inputs` reference upstream task IDs that are NOT YET in a
            # terminal state, hold off on claiming.  Caught v4: Director
            # pre-created a critic-prose task whose inputs included the
            # builder's task ID.  Inbox poller claimed and dispatched the
            # critic-prose task while the builder was still in_progress;
            # critic-prose found the builder's outputs=[] and got stuck
            # in a file-not-found retry loop, eventually F2 timed out
            # the whole slot.  Skipping claim until upstream is `done`
            # OR `cancelled` removes that race entirely.
            #
            # `inputs` may also contain literal paths or memory refs
            # (e.g. `memory:1`, `team/drafts/foo.md`) — those don't
            # block claiming.
            status_now = arr.column("status")[i].as_py()
            if status_now == "open":
                raw_inputs = arr.column("inputs")[i].as_py() or []
                task_ref_inputs = [
                    s for s in (str(x) for x in raw_inputs)
                    if s.startswith("task-")
                ]
                if task_ref_inputs:
                    unmet: list[str] = []
                    for ref in task_ref_inputs:
                        try:
                            ref_arr = tbl.search().where(
                                f"id = '{ref}'"
                            ).limit(1).to_arrow()
                            if ref_arr.num_rows == 0:
                                # Upstream missing — treat as unmet (don't
                                # claim a child of a phantom parent).
                                unmet.append(f"{ref}:missing")
                                continue
                            ref_status = ref_arr.column("status")[0].as_py()
                            if ref_status not in ("done", "cancelled"):
                                unmet.append(f"{ref}:{ref_status}")
                        except Exception as e:
                            # On Lance read failure, conservatively defer.
                            _msg = str(e)[:40]
                            unmet.append(f"{ref}:err({_msg})")
                    if unmet:
                        logger.debug(
                            f"[inbox_poller] holding {task_id}: "
                            f"upstream inputs not terminal: {unmet}"
                        )
                        continue

            # [PHASE_7_5_DEPENDS_ON_2026_05_28] Cross-lineage dep check.
            # Distinct from `inputs`-task-refs above: `inputs` model
            # producer-consumer artifact handoff (Builder uses
            # Researcher's draft path); `depends_on` models pure DAG
            # ordering (don't start B until A is done — used for fan-in
            # synchronisation where multiple siblings must converge).
            try:
                row_deps = arr.column("depends_on")[i].as_py() or []
            except KeyError:
                # Pre-Phase-7.5 rows on a not-yet-migrated table — no
                # depends_on column.  Treat as zero deps.
                row_deps = []
            if row_deps:
                tenant_for_check = arr.column("tenant_id")[i].as_py() or ""
                user_for_check   = arr.column("user_id")[i].as_py() or ""
                from ..ledger.tasks import deps_satisfied, deps_dead
                ready, pending = deps_satisfied(
                    task={"depends_on": row_deps},
                    tenant_id=tenant_for_check, user_id=user_for_check,
                )
                if not ready:
                    # [DEPS_DEAD_CASCADE_2026_05_29] If a dep was CANCELLED it
                    # can never reach `done`, so this dependent can never run.
                    # Cascade-cancel it instead of holding forever — that
                    # drains the synthesis subtask + lets its coordinator
                    # parent reach a terminal state rather than hang OPEN.
                    dead = deps_dead(
                        task={"depends_on": row_deps},
                        tenant_id=tenant_for_check, user_id=user_for_check,
                    )
                    if dead:
                        logger.info(
                            f"[inbox_poller] cascade-cancelling {task_id}: "
                            f"depends_on cancelled dep(s): {dead}"
                        )
                        self._cascade_cancel_dead_dep(
                            task_id=task_id,
                            tenant_id=tenant_for_check,
                            user_id=user_for_check,
                            dead_deps=dead,
                        )
                        continue
                    logger.debug(
                        f"[inbox_poller] holding {task_id}: "
                        f"depends_on pending: {pending}"
                    )
                    continue

            claimed = self._try_claim(tbl, task_id)
            logger.info(f"[inbox_poller] task {task_id}: claim={'OK' if claimed else 'FAIL'}")
            if not claimed:
                continue
            self._claimed_history.add(task_id)
            row = {
                col: arr.column(col)[i].as_py() for col in arr.column_names
            }
            asyncio.create_task(self._dispatch_with_semaphore(row))

    def _try_claim(self, tbl, task_id: str) -> bool:
        """Atomically claim a task by setting lease_owner + lease_until.

        Compare-and-swap pattern: only update if lease_owner is still
        null OR lease_until is in the past.  If our update affected 0
        rows, another worker beat us.

        LanceDB doesn't return affected-row count in a uniform way
        across versions; we use a re-read pattern instead: update, then
        re-query and check who owns the lease.  Slightly racy but
        sufficient for low-contention inbox polling.
        """
        now = time.time()
        try:
            # Claim guard: row must still be in a pollable state (matches
            # the _poll_once scan predicate) AND have a stale lease.
            # Don't narrow to status='open' here — that would silently
            # fail to claim review-state OR re-iterated tasks the scan
            # handed us.
            tbl.update(
                where=(
                    f"id = '{task_id}' "
                    f"AND status IN ('open', 'review', 'in_progress') "
                    f"AND lease_until < {now}"
                ),
                values={
                    "lease_owner": self._worker_id,
                    "lease_until": now + LEASE_DURATION_S,
                    "updated_ts": now,
                },
            )
        except Exception as e:
            logger.warning(f"[inbox_poller] claim {task_id} failed: {e}")
            return False

        # Verify we got the lease.
        try:
            arr = (
                tbl.search()
                .where(f"id = '{task_id}'")
                .limit(1)
                .to_arrow()
            )
            if arr.num_rows == 0:
                return False
            owner = arr.column("lease_owner")[0].as_py()
            return owner == self._worker_id
        except Exception:
            return False

    async def _dispatch_with_semaphore(self, task_row: dict) -> None:
        async with self._semaphore:
            task_id = task_row["id"]
            persona = task_row.get("owner_id") or ""
            self._in_flight.add(task_id)
            self._acquire_persona_slot(persona)  # Phase 6.2
            try:
                # Wrap _dispatch in a hard timeout so a spinning agent
                # can't leak this slot.  On TimeoutError we force-
                # cancel the task and emit a structured comment so the
                # behavior is auditable; otherwise we'd just leak the
                # _in_flight slot until process restart.
                await asyncio.wait_for(
                    self._dispatch(task_row),
                    timeout=DISPATCH_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"[inbox_poller] dispatch timeout after "
                    f"{DISPATCH_TIMEOUT_S:.0f}s on {task_id}; "
                    f"force-cancelling"
                )
                self._force_cancel_on_timeout(task_row)
            except Exception as e:
                logger.exception(
                    f"[inbox_poller] dispatch failed for {task_id}: {e}"
                )
            finally:
                self._in_flight.discard(task_id)
                self._release_persona_slot(persona)  # Phase 6.2

    def _force_cancel_on_turn_cap(self, task_row: dict, max_turns: int) -> None:
        """Cancel a task that exhausted its per-dispatch turn budget.

        [PER_DISPATCH_MAX_TURNS_2026_05_27] Sibling of
        ``_force_cancel_on_timeout``.  The two paths cancel for
        different reasons:
          - timeout: wall-clock 270s exceeded (asyncio.wait_for raised)
          - turn_cap: per-IC max_turns reached (loop exited cleanly with
            ``code=max_iterations`` event)
        The audit comment distinguishes them so the operator can tell
        which gate fired when triaging a non-converging task.
        """
        from ..ledger import comments as cm
        task_id = task_row["id"]
        tenant_id = task_row.get("tenant_id", "")
        user_id = task_row.get("user_id", tenant_id)
        owner_id = task_row.get("owner_id", "")
        try:
            cm.add_comment(
                task_id=task_id,
                tenant_id=tenant_id,
                user_id=user_id,
                author_id="inbox-poller",
                kind="blocker_raised",
                body=(
                    f"max_turns_exceeded: agent {owner_id!r} did not "
                    f"converge within {max_turns} LLM turns; task "
                    f"force-cancelled."
                ),
            )
        except Exception as e:
            logger.warning(
                f"[inbox_poller] could not write turn-cap comment for "
                f"{task_id}: {e}"
            )
        try:
            store = LedgerStore.get(str(LEDGER_DB_PATH))
            tbl = store._db.open_table("agent_tasks")
            tbl.update(
                where=f"id = '{task_id}'",
                values={
                    "status": "cancelled",
                    "updated_ts": time.time(),
                    "lease_owner": "",
                    "lease_until": 0.0,
                },
            )
        except Exception as e:
            logger.error(
                f"[inbox_poller] turn-cap update failed for {task_id}: {e}"
            )

    def _force_cancel_on_timeout(self, task_row: dict) -> None:
        """Cancel a hung task and emit an audit comment.

        Runs synchronously in the asyncio thread — direct Lance writes
        are fast (<50ms) and avoid the complexity of spawning another
        task during shutdown of this one.
        """
        from ..ledger import tasks as tm, comments as cm
        task_id = task_row["id"]
        tenant_id = task_row.get("tenant_id", "")
        user_id = task_row.get("user_id", tenant_id)
        owner_id = task_row.get("owner_id", "")

        # [TIMEOUT_NO_CLOBBER_2026_05_29] The timeout fired, but re-read the
        # task's CURRENT status before clobbering it. If the IC already
        # moved it past in_progress (submitted for review, or it finished),
        # the wall-clock timeout fired LATE — cancelling now would destroy a
        # task that's actually done/under review (caught live 2026-05-29: a
        # cost analysis reached `review` then the 270s timeout cancelled it).
        # Only force-cancel work that's genuinely still mid-dispatch.
        try:
            _cur = tm.get_task(task_id, tenant_id, user_id)
        except Exception:
            _cur = None
        _cur_status = (_cur or {}).get("status")
        if _cur_status not in ("accepted", "in_progress", "blocked"):
            logger.info(
                f"[inbox_poller] dispatch timeout fired for {task_id} but "
                f"status={_cur_status!r} (already advanced past dispatch) — "
                f"NOT force-cancelling"
            )
            return

        try:
            cm.add_comment(
                task_id=task_id,
                tenant_id=tenant_id,
                user_id=user_id,
                author_id="inbox-poller",
                kind="blocker_raised",
                body=(
                    f"dispatch_timeout: agent {owner_id!r} did not complete "
                    f"within {DISPATCH_TIMEOUT_S:.0f}s; task force-cancelled."
                ),
            )
        except Exception as e:
            logger.warning(
                f"[inbox_poller] could not write timeout comment for "
                f"{task_id}: {e}"
            )
        try:
            store = LedgerStore.get(str(LEDGER_DB_PATH))
            tbl = store._db.open_table("agent_tasks")
            tbl.update(
                where=f"id = '{task_id}'",
                values={
                    "status": "cancelled",
                    "updated_ts": time.time(),
                    "lease_owner": "",
                    "lease_until": 0.0,
                },
            )
        except Exception as e:
            logger.error(
                f"[inbox_poller] force-cancel update failed for {task_id}: {e}"
            )

    def _cascade_cancel_dead_dep(
        self, *, task_id: str, tenant_id: str, user_id: str, dead_deps: list,
    ) -> None:
        """[DEPS_DEAD_CASCADE_2026_05_29] Cancel a task whose depends_on
        includes a cancelled dep (it can never become ready).

        Goes through the proper update_status transition so the
        parent-autoclose hook fires — which lets the coordinator parent
        reach a terminal state once all its children are terminal, instead
        of hanging OPEN behind a permanently-blocked synthesis subtask.
        """
        from ..ledger import tasks as tm, comments as cm
        try:
            cm.add_comment(
                task_id=task_id, tenant_id=tenant_id, user_id=user_id,
                author_id="inbox-poller", kind="blocker_raised",
                body=(
                    f"dependency_cancelled: depends_on {dead_deps} was "
                    f"cancelled and can never reach 'done'; cascade-"
                    f"cancelling this dependent task."
                ),
            )
        except Exception as e:
            logger.warning(
                f"[inbox_poller] dead-dep comment failed for {task_id}: {e}"
            )
        try:
            tm.update_status(
                task_id=task_id, tenant_id=tenant_id, user_id=user_id,
                new_status="cancelled", actor_agent_id="inbox-poller",
            )
        except Exception as e:
            logger.error(
                f"[inbox_poller] cascade-cancel failed for {task_id}: {e}"
            )

    async def _dispatch(self, task_row: dict) -> None:
        """Run the assigned IC's persona on this task.

        We do NOT stream events back anywhere — the IC's response (status
        transitions, comments, attached outputs, submit_for_review) is
        the side-effect that matters.  Per-event audit + ledger writes
        already happen via the tools the IC calls.

        On completion: the lease auto-expires; the task's status reflects
        what the IC did (in_progress / blocked / review / cancelled).
        """
        from ..runtime import run_agent_query  # avoid circular import on module load

        task_id = task_row["id"]
        owner_id = task_row["owner_id"]
        tenant_id = task_row["tenant_id"]
        user_id = task_row["user_id"]
        status = task_row.get("status", "open")
        brief = task_row.get("brief", "")
        spec = task_row.get("deliverable_spec", "")
        outputs = task_row.get("outputs") or []

        persona = self._registry.get(owner_id)
        if persona is None:
            logger.warning(
                f"[inbox_poller] task {task_id} assigned to unknown agent {owner_id!r}; "
                "leasing held for re-route via Director"
            )
            return

        # Construct the task-context prompt the IC sees on wake-up.
        # Includes brief + deliverable_spec + any task.inputs the IC
        # should read.  This is the "task as primary primitive" surface
        # per spec §3.3.
        #
        # [INPUT_OUTPUT_RESOLUTION_2026_05_26] When an input is a task_id
        # (starts with "task-"), resolve its outputs[] so the dispatched
        # IC sees the EXACT artifact paths the upstream produced — not
        # just the upstream task ID.  Caught v3 (F6): builder wrote to
        # an off-spec path, critic-prose only saw the brief's spec path
        # (which didn't match what builder used), couldn't find the
        # file, raised 4 file-not-found blockers in a row, hit F2.
        # Resolving inputs upfront eliminates that round-trip ambiguity
        # — the critic reads the actual path the upstream IC committed.
        # [F17_CANCELLED_INPUT_SKIP_2026_05_27] When an input task is in
        # ``cancelled`` state, mark it SKIP in the dispatch prompt and
        # explicitly tell the agent NOT to attempt to read or review
        # that artifact.  v8 task-d002 (multi-input critic) spent its
        # entire 270s budget emitting 0 tool calls because one of its
        # two researcher inputs was cancelled mid-iteration; the critic
        # had no signal about which inputs were live and which were
        # dead.  Also count surviving (non-cancelled) inputs so the
        # critic knows how much real work to do.
        inputs_str = ""
        _raw_inputs = task_row.get("inputs") or []
        _live_input_count = 0
        _cancelled_input_count = 0
        if _raw_inputs:
            from ..ledger.store import LedgerStore
            from ..config import LEDGER_DB_PATH
            _store = LedgerStore.get(str(LEDGER_DB_PATH))
            _tbl_resolve = _store._db.open_table("agent_tasks")
            _lines: list[str] = []
            for _i in _raw_inputs:
                _s = str(_i)
                if _s.startswith("task-"):
                    try:
                        _arr = _tbl_resolve.search().where(
                            f"id = '{_s}'"
                        ).limit(1).to_arrow()
                        if _arr.num_rows > 0:
                            _upstream_outputs = _arr.column("outputs")[0].as_py() or []
                            _upstream_status = _arr.column("status")[0].as_py() or "?"
                            _upstream_owner = _arr.column("owner_id")[0].as_py() or "?"
                            # F17 — cancelled upstream means the work is
                            # dead.  Don't lead the agent into reading a
                            # potentially partial / never-finished file.
                            if _upstream_status == "cancelled":
                                _cancelled_input_count += 1
                                _lines.append(
                                    f"  - {_s}  (status=cancelled, "
                                    f"by={_upstream_owner})  **SKIP** — "
                                    f"upstream task was cancelled, do not "
                                    f"read or review its artifact"
                                )
                            elif _upstream_outputs:
                                _live_input_count += 1
                                _paths = ", ".join(repr(p) for p in _upstream_outputs)
                                _lines.append(
                                    f"  - {_s}  (status={_upstream_status}, "
                                    f"by={_upstream_owner})  outputs: {_paths}"
                                )
                            else:
                                _live_input_count += 1
                                _lines.append(
                                    f"  - {_s}  (status={_upstream_status}, "
                                    f"by={_upstream_owner})  outputs: [] — task has not "
                                    f"attached deliverables yet"
                                )
                        else:
                            _lines.append(f"  - {_s}  (not found)")
                    except Exception as _re:
                        logger.warning(
                            f"[inbox_poller] resolve input {_s}: {_re}"
                        )
                        _lines.append(f"  - {_s}  (resolve failed)")
                else:
                    # Treat as a literal path / opaque identifier.
                    _live_input_count += 1
                    _lines.append(f"  - {_s}")
            _footer = ""
            if _cancelled_input_count and _live_input_count:
                _footer = (
                    f"\n\n  ({_cancelled_input_count} of {len(_raw_inputs)} "
                    f"input(s) were cancelled upstream — review only the "
                    f"{_live_input_count} live input(s) above.)"
                )
            elif _cancelled_input_count and not _live_input_count:
                _footer = (
                    f"\n\n  (All {_cancelled_input_count} input(s) were "
                    f"cancelled upstream.  There is no live artifact to "
                    f"review — close this task with review_decision "
                    f"declined and a one-line note 'all upstream inputs "
                    f"cancelled'.)"
                )
            inputs_str = (
                "\n\nInputs — upstream artifacts (READ FROM THESE EXACT "
                "PATHS, not from the brief's example paths):\n"
                + "\n".join(_lines)
                + _footer
            )

        if status == "review":
            # Critic dispatch — Phase 6.1 fresh-context discipline.
            # The critic receives ONLY (spec, acceptance_criteria,
            # artifact paths).  Producer's chain-of-thought, intermediate
            # comments, and submit_for_review narrative are NOT inlined.
            # The critic reads artifact CONTENT via tool calls (read_file,
            # output_path_* executors); the prompt tells WHAT to check,
            # not WHAT the producer thought.
            #
            # Source: agent-runtime/app/acceptance/critic_prompt.py
            # Tests:  agent-runtime/tests/critic/test_fresh_context.py
            #         (AC-21 prompt content, AC-22 structural grep,
            #          AC-23 negative-fixture catches a deliberate leak)
            from ..acceptance.critic_prompt import build_critic_user_message
            # [CRITIC_NAMEERROR_FIX_2026_05_29]  Was `inputs=inputs` — but
            # the dispatch function never defines a bare `inputs`; it has
            # `_raw_inputs` (raw list) + `inputs_str` (formatted).  The
            # NameError crashed EVERY critic dispatch before the LLM was
            # even called, leaving the lease held → the watchdog reaped
            # the task 300s later as "lease_expired".  That's why review
            # tasks never completed.  Pass the raw input list.
            user_message = build_critic_user_message(
                task_id=task_id,
                brief=brief,
                deliverable_spec=spec,
                acceptance_criteria=task_row.get("acceptance_criteria") or [],
                outputs=outputs,
                inputs=[str(x) for x in (_raw_inputs or [])],
            )
        elif status == "in_progress":
            # Iteration dispatch: this IC ALREADY accepted this task at
            # some earlier turn (status moved past 'open'), but a critic
            # has now returned changes_requested.  The owner_id was
            # restored to this IC and the lease reset so we'd re-claim.
            #
            # [F16_INLINE_CRITIC_FEEDBACK_2026_05_27] Pull the latest
            # review_decision body out of the comment chain and embed
            # it directly in the user message.  Previously the iteration
            # prompt told the IC to "call get_task to read the
            # review_decision" — but ICs don't have ``get_task`` in
            # their toolset, so they'd either hallucinate the call
            # (failing silently) or burn turns trying to inspect the
            # artifact for clues.  v8 task-f1e8 spun 270s+ this way.
            # Inlining the feedback text + the artifact path makes the
            # iteration prompt self-sufficient.
            from ..ledger import comments as _cm
            _critic_feedback = ""
            try:
                _cmts = _cm.list_comments(
                    task_id=task_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                )
                # Walk back to the most recent review_decision (the
                # one that bounced this task into in_progress).  We
                # only inline that one — earlier rounds are noise.
                for c in reversed(_cmts or []):
                    if c.get("kind") == "review_decision":
                        _critic_feedback = (c.get("body") or "").strip()
                        break
            except Exception as _fe:
                logger.warning(
                    f"[inbox_poller] iteration: could not load critic "
                    f"feedback for {task_id}: {_fe}"
                )
            _feedback_block = (
                f"\n\n**Critic's review_decision (most recent):**\n"
                f"{_critic_feedback}\n"
                if _critic_feedback else
                "\n\n(Critic feedback unavailable — proceed by re-reading "
                "the artifact and applying common-sense improvements; "
                "do NOT call get_task, it is not in your toolset.)\n"
            )
            _outputs_block = (
                f"\n**Your current artifact(s):**\n"
                + "\n".join(f"  - {o}" for o in outputs)
                + "\n"
                if outputs else ""
            )
            user_message = (
                f"Task {task_id} has been returned for revision.\n\n"
                f"Brief: {brief}\n\n"
                f"Deliverable spec: {spec}"
                f"{inputs_str}"
                f"{_outputs_block}"
                f"{_feedback_block}\n"
                f"Apply the requested changes to the artifact (use "
                f"read_file → edit_file or write_file → attach_output "
                f"if path changed), then submit_for_review again with "
                f"the SAME target as before.  Do NOT call accept_task "
                f"(task is already accepted).  Do NOT call get_task — "
                f"the critic's feedback is inlined above; that's all "
                f"the context you need."
            )
        else:
            # status == 'open' — fresh assignment.
            # [IC_ANTIFLAIL_2026_05_29]  Everything the IC needs is in
            # THIS message.  Without an explicit guard the IC (esp. when
            # a tool like web_search is unavailable) invents task-state
            # files — caught live: researcher looping on
            # read_file("tasks/current_task.json"), read_file("tasks/current").
            # Also tell it how to behave when a tool returns UNAVAILABLE:
            # complete from reasoning or raise a blocker — never loop.
            user_message = (
                f"You have been assigned Task **{task_id}** (authoritative "
                f"id — everything you need is in this message; do NOT call "
                f"read_file/get_task to look up your task or 'task-state' "
                f"files — there are none).\n\n"
                f"Brief: {brief}\n\n"
                f"Deliverable spec: {spec}"
                f"{inputs_str}\n\n"
                f"Steps:\n"
                f"  1. accept_task to start.\n"
                f"  2. Do the work. Write your deliverable with write_file "
                f"to the exact path in the spec (it's a server-side team "
                f"workspace path like team/drafts/...).\n"
                f"  3. attach_output(path) for the artifact.\n"
                f"  4. submit_for_review with the appropriate target.\n\n"
                f"Tool-failure rule: if a tool result says UNAVAILABLE or "
                f"errors (e.g. web_search when the research bridge is "
                f"offline), do NOT retry it and do NOT invent alternative "
                f"files to read. Either complete the deliverable from the "
                f"knowledge you already have (note the limitation in the "
                f"artifact), or call add_comment(kind=blocker_raised, "
                f"body=<what's blocked>) and submit_for_review with what "
                f"you have. Never loop on a failing tool."
            )

        # Use a sub-cid so the agent's calls are scoped properly and
        # the audit / approval gate can tell this is background.
        sub_cid = self._make_sub_cid(parent_cid=task_id[:7], kind="ibx")

        # [PER_DISPATCH_MAX_TURNS_2026_05_27] Per-role turn caps.
        # ENABLED 2026-05-27 after a researcher burned 100+ stream events
        # (`task-6becb4f4e1dc`) looping on observe() with TCMM down.
        # Plumbing was in place from the start; activation was deferred
        # while we observed natural critic behaviour.  Empirically the
        # non-convergence cases were ALWAYS dithering, never productive
        # past ~12-15 turns — so 25 (IC) / 10 (critic) is a safe cap.
        # If hit, runtime emits `max_iterations` event and
        # `_force_cancel_on_turn_cap` writes an audit comment so the
        # operator can see which gate fired vs the wall-clock timeout.
        _CRITIC_MAX_TURNS = 10
        _IC_MAX_TURNS = 25
        _ENABLE_TURN_CAP = True
        _max_turns_for_role = (
            (_CRITIC_MAX_TURNS if owner_id.startswith("critic-") else _IC_MAX_TURNS)
            if _ENABLE_TURN_CAP else None
        )

        logger.info(
            f"[inbox_poller] dispatching task {task_id} → {owner_id} "
            f"(tenant={tenant_id}, sub_cid={sub_cid}, "
            f"max_turns={_max_turns_for_role or 'default'})"
        )

        _hit_max_turns = False
        # [LIVE_STREAM_TO_SIDEBAR_2026_05_27] Broadcast every agent event
        # to SSE subscribers tagged with this task_id, so the sidebar's
        # open task DetailSheet can render the LLM's thinking +
        # tool-calls + token stream + final result in real time.  These
        # events do NOT trigger a /tasks/{id} refetch on the client — the
        # client appends them straight to a per-task live buffer.  Only
        # actual ledger writes (status_change, comment_added) hit the
        # detail refetch path.
        from ..events import broadcast as _bcast
        # Coalesce text_delta tokens into bigger batches so we don't
        # fire a network event per character — every 250ms flush.
        _text_buf: list[str] = []
        _last_flush = time.monotonic()
        def _flush_text() -> None:
            nonlocal _text_buf, _last_flush
            if not _text_buf:
                return
            txt = "".join(_text_buf)
            _text_buf = []
            _last_flush = time.monotonic()
            try:
                _bcast({
                    "type":      "task_stream_delta",
                    "tenant_id": tenant_id, "user_id": user_id,
                    "task_id":   task_id,
                    "agent_id":  owner_id,
                    "text":      txt,
                })
            except Exception:
                pass

        # [PHASE_7_5_TEAM_ID_DISPATCH_2026_05_28] Propagate team_id from
        # the task row into the dispatch tenant context so any tool the
        # IC calls (cost rollup, team_cost_report, team-scoped recall)
        # sees its team without having to re-read the task row.
        _team_id_for_dispatch = task_row.get("team_id")
        try:
            async for _ev in run_agent_query(
                persona=persona,
                conversation_id=sub_cid,
                user_id=user_id,
                tenant_id=tenant_id,
                messages=[{"role": "user", "content": user_message}],
                registry=self._registry,
                constitution=None,  # ICs don't directly use constitution
                parent_cid=task_id,
                task_id=task_id,
                team_id=_team_id_for_dispatch,
                max_turns=_max_turns_for_role,
            ):
                if not isinstance(_ev, dict):
                    continue
                etype = _ev.get("type") or ""
                if _ev.get("code") == "max_iterations":
                    _hit_max_turns = True
                # Phase 6.3 — heartbeat at every assistant turn boundary
                # so the inbox-poller's sweep can detect a dead worker
                # (one that crashes mid-stream).  Cheap: ~one Lance
                # insert per LLM turn.
                if etype in ("assistant_text", "tool_call", "final_result"):
                    try:
                        from ..memory.writers import record_heartbeat
                        record_heartbeat(
                            task_id=task_id,
                            worker_id=self._worker_id,
                            tenant_id=tenant_id,
                            user_id=user_id,
                        )
                    except Exception:
                        pass
                # Stream events to the sidebar.
                if etype == "text_delta":
                    _text_buf.append(_ev.get("text", "") or "")
                    if time.monotonic() - _last_flush > 0.25:
                        _flush_text()
                elif etype == "assistant_text":
                    _flush_text()
                    # [EMPTY_BUBBLE_FIX_2026_05_29]  A tool-only LLM turn
                    # (no prose, just a tool_use) still emits an
                    # assistant_text event with empty text — which the
                    # sidebar rendered as a blank "researcher:" bubble.
                    # Skip the broadcast when there's nothing to show.
                    _atext = (_ev.get("text", "") or "").strip()
                    if _atext:
                        try:
                            _bcast({
                                "type":      "task_stream_message",
                                "tenant_id": tenant_id, "user_id": user_id,
                                "task_id":   task_id,
                                "agent_id":  owner_id,
                                "text":      _atext,
                            })
                        except Exception:
                            pass
                elif etype == "tool_call":
                    _flush_text()
                    try:
                        _bcast({
                            "type":      "task_stream_tool_call",
                            "tenant_id": tenant_id, "user_id": user_id,
                            "task_id":   task_id,
                            "agent_id":  owner_id,
                            "tool_name": _ev.get("name", "") or "",
                            "tool_id":   _ev.get("id", "") or "",
                        })
                    except Exception:
                        pass
                elif etype == "tool_dispatch":
                    # Lower priority — already implied by tool_call;
                    # included here only when the dispatch differs from
                    # the call (e.g. fan-out).  Skip to reduce noise.
                    pass
                elif etype == "tool_result":
                    # [TOOL_RESULT_VISIBILITY_2026_05_29]  Surface tool
                    # success/failure to the sidebar.  Previously this
                    # event was DROPPED (no branch), so the UI showed
                    # `tool_call read_file` then jumped straight to the
                    # next assistant turn — the user couldn't see that
                    # the call had errored (file-not-found, tool
                    # unavailable, etc.), which is what made the agent
                    # loops look mysterious.  Now each result emits a
                    # task_stream_tool_result with is_error + a short
                    # preview of the result/error text.
                    _flush_text()
                    try:
                        _bcast({
                            "type":      "task_stream_tool_result",
                            "tenant_id": tenant_id, "user_id": user_id,
                            "task_id":   task_id,
                            "agent_id":  owner_id,
                            "tool_name": _ev.get("name", "") or "",
                            "tool_id":   _ev.get("id", "") or "",
                            "is_error":  bool(_ev.get("is_error", False)),
                            "degraded":  bool(_ev.get("degraded", False)),
                            "preview":   (_ev.get("result_preview", "") or "")[:500],
                        })
                    except Exception:
                        pass
                elif etype == "final_result":
                    _flush_text()
                    # [EMPTY_BUBBLE_FIX_2026_05_29]  Per-LLM-turn the
                    # Agent emits a final_result even when the turn was
                    # tool-only (empty result) — rendered as a blank
                    # "✓ final_result:" bubble.  Only broadcast when
                    # there's actual final text.
                    _fr = (_ev.get("result", "") or "").strip()
                    if _fr:
                        try:
                            _bcast({
                                "type":      "task_stream_final",
                                "tenant_id": tenant_id, "user_id": user_id,
                                "task_id":   task_id,
                                "agent_id":  owner_id,
                                "result":    _fr[:2000],
                            })
                        except Exception:
                            pass
                elif etype == "error":
                    _flush_text()
                    try:
                        _bcast({
                            "type":      "task_stream_error",
                            "tenant_id": tenant_id, "user_id": user_id,
                            "task_id":   task_id,
                            "agent_id":  owner_id,
                            "code":      _ev.get("code", "") or "",
                            "message":   (_ev.get("message", "") or "")[:500],
                        })
                    except Exception:
                        pass
        except Exception as e:
            logger.exception(
                f"[inbox_poller] task {task_id} dispatch raised: {e}"
            )
        finally:
            _flush_text()

        # [PER_DISPATCH_MAX_TURNS_2026_05_27] If the agent loop exited
        # cleanly because it hit max_turns, the task is still in its
        # mid-flight status (review / in_progress) with no
        # review_decision and no submit.  Force-cancel with an audit
        # comment that distinguishes "turn cap" from "wall-clock
        # timeout" — the operator can see which gate fired.
        if _hit_max_turns:
            logger.warning(
                f"[inbox_poller] task {task_id} ({owner_id}) hit max_turns "
                f"cap of {_max_turns_for_role}; force-cancelling"
            )
            self._force_cancel_on_turn_cap(task_row, _max_turns_for_role)

    @staticmethod
    def _make_sub_cid(parent_cid: str, kind: str) -> str:
        """Match the sub-<parent7>-<kind3>-<uuid8> convention from
        sub-agents service spawn_scope.
        """
        return f"sub-{parent_cid[:7]}-{kind[:3]}-{uuid.uuid4().hex[:8]}"


# ── Module-level lifecycle helpers ──────────────────────────────────────


_singleton: Optional[InboxPoller] = None
_runner_task: Optional[asyncio.Task] = None


async def start(registry: PersonaRegistry) -> None:
    """Start the singleton poller.  Idempotent."""
    global _singleton, _runner_task
    if _singleton is not None:
        logger.warning("[inbox_poller] already running; start() is idempotent")
        return
    _singleton = InboxPoller(registry)
    _runner_task = asyncio.create_task(_singleton.run())


async def stop() -> None:
    """Stop the poller cleanly.  Awaits in-flight dispatches."""
    global _singleton, _runner_task
    if _singleton is None:
        return
    _singleton.stop()
    if _runner_task is not None:
        try:
            await asyncio.wait_for(_runner_task, timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("[inbox_poller] stop timed out; cancelling")
            _runner_task.cancel()
    _singleton = None
    _runner_task = None


__all__ = [
    "InboxPoller",
    "ELIGIBLE_OWNERS",
    "start",
    "stop",
]
