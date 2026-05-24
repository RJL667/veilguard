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
MAX_CONCURRENT_DISPATCHES = 4
LEASE_DURATION_S = 300.0  # 5 min; long enough for most agent turns

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
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_DISPATCHES)
        self._worker_id = f"worker-{uuid.uuid4().hex[:8]}"
        self._in_flight: set[str] = set()      # task_ids currently dispatching
        self._claimed_history: set[str] = set()  # all task_ids ever claimed (telemetry)

    def stop(self) -> None:
        self._stop_evt.set()

    async def run(self) -> None:
        """Main poll loop.  Returns when stop() is called."""
        logger.info(
            f"[inbox_poller] starting; worker_id={self._worker_id}, "
            f"interval={POLL_INTERVAL_S}s, max_concurrent={MAX_CONCURRENT_DISPATCHES}"
        )
        try:
            while not self._stop_evt.is_set():
                try:
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

    async def _poll_once(self) -> None:
        """One poll cycle: find claimable tasks, dispatch them."""
        try:
            store = LedgerStore.get(str(LEDGER_DB_PATH))
        except Exception as e:
            # On dev / first-boot the Lance dir may not exist yet.  Wait
            # for the next interval.
            logger.debug(f"[inbox_poller] ledger not ready: {e}")
            return

        tbl = store.table("agent_tasks")
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
        where = (
            f"owner_id IN ({eligible_owners_sql}) "
            f"AND status = 'open' "
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
            tbl.update(
                where=(
                    f"id = '{task_id}' "
                    f"AND status = 'open' "
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
            self._in_flight.add(task_row["id"])
            try:
                await self._dispatch(task_row)
            except Exception as e:
                logger.exception(
                    f"[inbox_poller] dispatch failed for {task_row['id']}: {e}"
                )
            finally:
                self._in_flight.discard(task_row["id"])

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
        brief = task_row.get("brief", "")
        spec = task_row.get("deliverable_spec", "")

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
        inputs_str = ""
        if task_row.get("inputs"):
            inputs_str = "\n\nInputs (paths or upstream task_ids):\n" + "\n".join(
                f"  - {i}" for i in task_row["inputs"]
            )

        user_message = (
            f"You have been assigned Task {task_id}.\n\n"
            f"Brief: {brief}\n\n"
            f"Deliverable spec: {spec}"
            f"{inputs_str}\n\n"
            f"Call accept_task to start, then do the work, then "
            f"submit_for_review with the appropriate target."
        )

        # Use a sub-cid so the agent's calls are scoped properly and
        # the audit / approval gate can tell this is background.
        sub_cid = self._make_sub_cid(parent_cid=task_id[:7], kind="ibx")

        logger.info(
            f"[inbox_poller] dispatching task {task_id} → {owner_id} "
            f"(tenant={tenant_id}, sub_cid={sub_cid})"
        )

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
            ):
                # We don't stream IC events anywhere; side-effects are
                # what counts.  Could log here for debugging.
                pass
        except Exception as e:
            logger.exception(
                f"[inbox_poller] task {task_id} dispatch raised: {e}"
            )

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
