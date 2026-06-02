# Tool consolidation manifest + fork-A spec (2026-05-31)

Actionable plan from tonight's 4 code investigations. **Nothing here is executed**
— renames/deletions reshape your live tool surface and need your A/B call. In the
morning, pick a fork (or cherry-pick rows marked ✅ safe-now) and I'll execute.

## The fork decision (drives everything)

You run two agentic stacks: the **OLD** `sub-agents` host service (spawn/start,
orchestration, daemons, schedule, Set-B teams) and the **NEW** agent-runtime
ledger (Director + ICs). A migration to fold OLD→NEW exists but **stalled at
Phase B-2** (`core/tasks/__init__.py` calls the old representations "five
disconnected task representations" it replaces; `SchedulerExecutor` is built but
unwired; `DIRECTOR_RUN` task-kind is dead).

- **Fork A — finish the migration.** Effort: high. Outcome: one stack, durable +
  tenant-scoped. Retire daemons/schedule/async-family in favour of the unified
  `core/tasks` SchedulerExecutor + ledger.
- **Fork B — keep OLD as the deliberate "no-Director execution layer."** Effort:
  low. Outcome: two stacks long-term, but tidy (rename collisions, collapse
  redundant ergonomics). Accepts the duplication.

## Per-tool manifest

Verdict key: KEEP · MERGE (fold into another) · RENAME · CUT (after dep) · ✅=safe to do without the fork decision.

| Tool(s) | file:line | Verdict | Notes / depends-on |
|---|---|---|---|
| **Teams — Set A** create_team / list_teams / team_cost_report | `agent-runtime/app/tools/ledger_mcp.py:1210,1267,1296` (`ledger/teams.py`) | **KEEP** | Canonical: durable Lance `agent_teams`, budget envelope, Director-only |
| **Teams — Set B** team_create / team_assign / team_status / team_delete | `services/sub-agents/tools/teams.py:18,34,72,88` (in-RAM `core/state.py:134`) | **RENAME → crew_*** ✅ | Live-exec helper; only the NAME collides with Set A. Rename kills the confusion. Update `tools/register.py`, `tool_search.py`, any persona refs |
| **async family** start_task/check_task/get_result/start_parallel_tasks/wait_for_tasks/smart_task | `services/sub-agents/tools/tasks.py:99,196,210,248,410,309` | **MERGE → 3** ✅ | All funnel into one `_run_background_worker` (tasks.py:67). Collapse to `run`(async|sync, 1|N) + `check` + `result`. smart_task = start_task+inline-wait; start_parallel_tasks = N-fan-out. Pure ergonomics |
| **spawn family** spawn_agent / spawn_agentic / parallel_agents | `services/sub-agents/tools/agents.py:40,251,74` | **KEEP (Fork B)** / CUT (Fork A) | The no-Director execution path. Inline `call_llm`. Director never calls them |
| **orchestration** coordinate / pipeline / review_loop | `services/sub-agents/tools/agents.py:106,187,225` | **KEEP** | Distinct shapes: coordinate ⊃ parallel; pipeline=sequential; review_loop=iterate. Director reimplements natively via ledger depends_on/critic — so redundant only under Fork A |
| **daemons** start_daemon/stop/list/log/wait | `services/sub-agents/tools/daemons.py:101,157,176,186,199` (in-RAM `core/state.py:121`) | **CUT after B-2.5** | Replacement exists: `core/tasks/executors/scheduler.py` (`SchedulerExecutor`, persistent). Don't delete until start_daemon is routed through it + daemon_wait has a unified equivalent |
| **schedule** schedule_task/run/list/pause/resume/delete | `services/sub-agents/tools/schedules.py:68,80,89,100,109,119` (in-RAM `core/state.py:45`) | **MERGE → SchedulerExecutor** | Least-integrated cluster. Unified Task already has `interval_seconds`/`next_run_at` (`core/tasks/model.py:116-117`, kind SCHEDULED :54). Fixed-interval only, no persistence today |
| **clipboard** clipboard_copy/paste/list | (sub-agents tools) | **CUT** | Desktop metaphor; use files/ledger/scratchpad. Low usage |
| **set_permission_level** | (client_admin) | **MOVE off agent surface** | Agent raising its own approval ceiling is a privilege-escalation smell. `get_` is fine |
| **get_notifications** | (sub-agents) | **CUT** | Retrieving toasts you sent is near-useless |
| **agent messaging** agent_send/inbox/broadcast | (sub-agents) | **REVIEW** | Likely redundant with ledger + typed TCMM channels; confirm before cut |
| **tool_search catalog** daemon_wait gap | `services/sub-agents/tools/tool_search.py` | **DONE** ✅ | Added 2026-05-31 (this loop) |

## Fork-A execution order (dependency-sorted)

1. **Wire `SchedulerExecutor`** as the real engine for `start_daemon` + `schedule_task`
   (route their bodies to `TaskDispatcher` with `kind=SCHEDULED`, `interval_seconds`).
   Files: `daemons.py`, `schedules.py`, `core/tasks/executors/scheduler.py`,
   `core/tasks/dispatcher.py:57`. (This is "Phase B-2.5" referenced in
   `executors/sub_agent.py:14-18`.)
2. **Port `daemon_wait`** to poll the unified store instead of `state.daemons`.
3. **Wire `DIRECTOR_RUN`** (`core/tasks/model.py:57`) so the scheduler can fire
   agent-runtime Director invocations — currently a dead enum with no producer/consumer.
4. **Collapse the async family** (tasks.py) to `run`/`check`/`result`.
5. **Retire** daemons + schedule + Set-B teams + clipboard; update `register.py`,
   `tool_search.py`, personas.
6. **Re-test** `tests/test_all.py` + add coverage for the new SCHEDULED path.

## Fork-B execution order (minimal)

1. Rename Set-B teams → `crew_*` ✅
2. Collapse async family → run/check/result ✅
3. Cut clipboard + get_notifications; move set_permission_level off-surface
4. Update `register.py` + `tool_search.py`; re-run `test_all.py`

## Safe-to-execute-now subset (no fork decision needed)
- ✅ Rename Set-B teams → `crew_*` (kills the collision either way)
- ✅ Collapse the async family → run/check/result (one engine, pure ergonomics)
- ✅ tool_search catalog (done)
These are surface changes, so I still want your one-word go before touching them.
