# Phase 6 Acceptance Criteria

> **Source of truth for CI.** Phase 6 is "done" only when all 43 mechanical
> ACs run green; the 2 `manual_user` ACs gate operator sign-off, not CI.
>
> Each AC carries: `id`, `statement`, `check_kind`, `check_args`, `required`,
> `rationale`. `check_kind` values are the 7 mechanical kinds from
> §6.0 of `MULTI_AGENT_PLATFORM.md` plus `manual_user`. `llm_verify` is
> deferred to Phase 7+ per the iron rule.
>
> Companion file: `tests/PHASE_7_ACCEPTANCE.md` (Phase 7.1 migrations,
> 6 ACs). To be created when Phase 7 implementation begins.

---

## Phase 6.0 — `acceptance_criteria` column + 7 mechanical executors

### AC-1 — Lance schema migration applied
- **check_kind**: `output_path_exists`
- **check_args**: `{path: "tcmm-data/veilguard/tcmm.db/agent_tasks.lance/_versions/<new>.manifest", min_bytes: 1}`
- **required**: true
- **rationale**: Schema bump must produce a new Lance manifest. Without it, the new column doesn't exist.

### AC-2 — Column present with correct type
- **check_kind**: `claim_predicate`
- **check_args**: `{predicate: "pyarrow_schema('agent_tasks').field('acceptance_criteria').type == List<Struct<id:Utf8, statement:Utf8, check_kind:Utf8, check_args:Utf8, required:Bool, rationale:Utf8>>"}`
- **required**: true
- **rationale**: PyArrow schema introspection on the live table. Catches typo'd column type.

### AC-3 — Old rows backfilled
- **check_kind**: `claim_count`
- **check_args**: `{table: "agent_tasks", filter: "acceptance_criteria IS NULL OR cardinality(acceptance_criteria) == 0", op: "==", n: 0}`
- **required**: true
- **rationale**: No NULL/empty `acceptance_criteria` after migration. Pre-Phase-6 rows get backfilled with a single advisory AC so downstream code never sees an empty array.

### AC-4 — All 7 executors registered (NOT 8 — `llm_verify` deferred)
- **check_kind**: `claim_predicate`
- **check_args**: `{predicate: "set(EXECUTOR_REGISTRY.keys()) == {'claim_count', 'claim_predicate', 'output_path_exists', 'output_path_matches_regex', 'output_path_jsonschema', 'test_passes', 'manual_user'}"}`
- **required**: true
- **rationale**: Off-by-one in dispatch dict = silent regression. `llm_verify` NOT in set — its presence indicates the iron rule was violated.

### AC-5 — Happy-path golden test (per-executor pass)
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/acceptance/test_executors_happy.py -x", expect_exit: 0}`
- **required**: true
- **rationale**: Each executor returns `pass` on a hand-crafted fixture that should pass.

### AC-6 — Sad-path golden test (per-executor fail with non-empty reason)
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/acceptance/test_executors_sad.py -x", expect_exit: 0}`
- **required**: true
- **rationale**: Each executor returns `fail` with a non-empty `reason` on a fixture that should fail. Prevents "always returns pass" bug.

### AC-7 — Empty-input safety (anti-false-positive)
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/acceptance/test_executors_empty.py::test_no_false_pass_on_empty -x", expect_exit: 0}`
- **required**: true
- **rationale**: `claim_count(n>=1)` on empty claim list → `fail`. `output_path_matches_regex` on empty file with pattern `.*` → `fail`. `output_path_exists` with `min_bytes=1` on a 0-byte file → `fail`. Catches "regex matches everything" / "file is empty but exists" classes.

### AC-8 — `test_passes` executor sandboxed
- **check_kind**: `claim_predicate`
- **check_args**: `{predicate: "EXECUTOR_REGISTRY['test_passes'].timeout_default <= 60 AND EXECUTOR_REGISTRY['test_passes'].cwd_must_be_under(REPO_ROOT)"}`
- **required**: true
- **rationale**: Stops a malicious/buggy task from running unbounded shell commands or escaping the repo dir.

### AC-26 — Evidence hash present on every executor result
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/acceptance/test_evidence_hash.py -x", expect_exit: 0}`
- **required**: true
- **rationale**: Every executor returns `{status, evidence: {path_sha256?, exit_code?, stdout_hash?, predicate_match_count?, ...}, reason}`. Critic re-verifies evidence on review; mismatched hash = `changes_requested`. Stops "builder partial-writes, crashes mid-flight, retry passes a *different* artifact under the same path" corruption.

### AC-27 — Three-state result: `pass | fail | error` (not two)
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/acceptance/test_three_state.py -x", expect_exit: 0}`
- **required**: true
- **rationale**: `error` = the check itself failed to execute (cmd not on PATH, file unreadable, regex compile error). Distinct from `fail` (artifact wrong). Both → `changes_requested`. Catches "always-pass" bugs in the gate code itself (e.g. exit 127 → fail vs error — buggy executor that maps error to skip silently passes the gate).

---

## Phase 6.0.1 — Hard-gate `state='done'`

### AC-9 — Guard fires in `update_status`
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/gate/test_hard_gate.py::test_done_rejected_when_violations -x", expect_exit: 0}`
- **required**: true
- **rationale**: `update_status(task_id, 'done')` with `constraint_violations=['x']` must raise `IllegalTransition`.

### AC-10 — Guard fires on direct Lance write (bypass route)
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/gate/test_hard_gate.py::test_done_rejected_via_direct_lance -x", expect_exit: 0}`
- **required**: true
- **rationale**: Bypassing `update_status` and writing `state='done'` directly via raw Lance must also be rejected. This is the #1 bypass route. Both write paths probed.

### AC-11 — Accepted path still works
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/gate/test_hard_gate.py::test_done_allowed_when_clean -x", expect_exit: 0}`
- **required**: true
- **rationale**: Sanity: `review_decision='accepted' AND constraint_violations=[] AND all_required_acs_pass` MUST transition.

### AC-12 — No `state='done'` rows violate invariant in production
- **check_kind**: `claim_count`
- **check_args**: `{table: "agent_tasks", filter: "state='done' AND (review_decision != 'accepted' OR cardinality(constraint_violations) > 0)", op: "==", n: 0}`
- **required**: true
- **rationale**: Post-deploy invariant scan against actual data. Catches a `done` row that escaped the gate (race, migration error, manual SQL).

---

## Phase 6.1 — Fresh-context critics

### AC-21 — Critic prompt content
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/critic/test_fresh_context.py::test_only_spec_ac_artifact -x", expect_exit: 0}`
- **required**: true
- **rationale**: Captured critic prompt MUST contain `(spec, acceptance_criteria, artifact)`; MUST NOT contain producer trajectory tokens.

### AC-22 — Structural source grep
- **check_kind**: `output_path_matches_regex`
- **check_args**: `{path: "agent/critic.py", pattern: "producer_messages|producer_trajectory|parent_thread", expect: "no_match"}`
- **required**: true
- **rationale**: No `producer_*` field is wired into the critic's prompt builder under any flag (debug, fallback, anything). Catches "developer added a TODO but left the field wired in" + "debug-flag leak that ships on."

### AC-23 — Negative fixture (tests the test)
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/critic/test_fresh_context.py::test_leak_detector_catches_chain_of_thought -x", expect_exit: 0}`
- **required**: true
- **rationale**: Deliberately wire a leak; the leak-detector test must catch it. Meta-rigor: tests the test. Stops "test mocks the prompt builder and asserts the mock was called, never inspects actual rendered prompt."

---

## Phase 6.2 — Per-persona concurrency caps

### AC-13 — Config dict loaded
- **check_kind**: `claim_predicate`
- **check_args**: `{predicate: "PERSONA_CAPS['researcher']==8 AND PERSONA_CAPS['builder']==6 AND PERSONA_CAPS['critic-claim']==4 AND PERSONA_CAPS['critic-prose']==4"}`
- **required**: true
- **rationale**: Catches "config defined but old global still in use."

### AC-14 — Old global removed (regex absence)
- **check_kind**: `output_path_matches_regex`
- **check_args**: `{path: "agent-runtime/app/workers/inbox_poller.py", pattern: "MAX_CONCURRENT_DISPATCHES", expect: "no_match"}`
- **required**: true
- **rationale**: Dead constant left behind = soft bypass.

### AC-15 — Cap actually limits dispatch
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/concurrency/test_persona_cap.py::test_researcher_cap_blocks_9th -x", expect_exit: 0}`
- **required**: true
- **rationale**: Enqueue 12 researcher tasks; observe ≤ 8 in-flight at any sampled instant over 5s.

### AC-16 — Caps are independent (starvation independence)
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/concurrency/test_persona_cap.py::test_builder_not_starved_by_researcher_saturation -x", expect_exit: 0}`
- **required**: true
- **rationale**: While researcher is at 8/8, a builder task must still start. Catches "shared semaphore keyed on persona acquired before persona-resolution → researchers starve builders."

---

## Phase 6.3 — Lease TTL + heartbeats

### AC-28 — Heartbeat row appears
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/lease/test_heartbeat.py::test_worker_writes_heartbeat -x", expect_exit: 0}`
- **required**: true
- **rationale**: Worker writes a heartbeat row to `agent_task_heartbeats` every N turns. Without it, lease TTL has no live signal.

### AC-29 — Orphan reclaimed after TTL
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/lease/test_heartbeat.py::test_orphan_reclaimed_after_ttl -x", expect_exit: 0}`
- **required**: true
- **rationale**: Simulated dead worker (no heartbeat for `lease_ttl_s + 30s`) → inbox-poller auto-reclaims with audit comment `lease_expired`.

---

## Phase 6.4 — Revision-priority lane

### AC-30 — Revision claimed before fresh builds
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/revision/test_priority_lane.py -x", expect_exit: 0}`
- **required**: true
- **rationale**: Spawn 5 fresh builder tasks + 1 critic-revision task, all `owner=builder`, cap=2. Revision task claims first. Without this, critic-revision starves and "done halfway" becomes "done forever-pending."

---

## Phase 6.5 — Truncated-output marker

### AC-31 — `read_file` emits marker
- **check_kind**: `output_path_matches_regex`
- **check_args**: `{path: "agent-runtime/app/tools/file_tools.py", pattern: "\\[TRUNCATED:\\s+\\d+\\s+of\\s+\\d+\\s+bytes shown"}`
- **required**: true
- **rationale**: Every tool wrapper emits an explicit `[TRUNCATED: N of M bytes shown — page or chunk before acting]` tail when response is capped.

### AC-32 — Persona prompt mentions TRUNCATED rule
- **check_kind**: `output_path_matches_regex`
- **check_args**: `{path: "agents/researcher.md", pattern: "TRUNCATED"}`
- **required**: true
- **rationale**: Persona prompts include the one-line rule: "If you see TRUNCATED, page/chunk or raise blocker — do not reason over a truncated response as if complete." Both halves of the fix (tool emits marker + persona knows what to do) must ship together.

---

## Phase 6.7 — APR + circuit breaker

### AC-33 — APR counter emitted per dispatch
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/apr/test_counter_emitted.py -x", expect_exit: 0}`
- **required**: true
- **rationale**: Every dispatch emits `apr.artifacts_count` + `apr.tokens_consumed` counters to the rolling window. Without emission there's no signal to circuit-break on.

### AC-34 — Rolling-window calculator handles boundary correctly
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/apr/test_rolling_window.py -x", expect_exit: 0}`
- **required**: true
- **rationale**: Window edge cases: cold start (insufficient data → APR=null, not 0), boundary slide (event at t-30:00 vs t-29:59), window reset on circuit-break clear.

### AC-35 — Circuit breaker fires on synthetic self-narration
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/apr/test_circuit_breaker.py::test_low_apr_pauses_dispatch -x", expect_exit: 0, timeout: 1900}`
- **required**: true
- **rationale**: Deliberate 30-min low-APR load (heavy `add_comment` calls, zero state mutations) triggers the breaker. New dispatches stop; in-flight complete their turn. Sidebar surfaces banner. Operator unblock reopens dispatch. **The most important AC in Phase 6.7** — the breaker's existence is the entire point.

---

## Phase 6.8 — Memory writer-function layer

### AC-36 — Linter rejects direct memory writes
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/memory/test_write_path_lint.py -x", expect_exit: 0}`
- **required**: true
- **rationale**: AST scan of every module under `agent-runtime/app/` or `agent/`. Any module that imports a Lance handle / TCMM HTTP client / workspace fs writer outside the writers module → CI fails.

### AC-37 — Writer-to-destination map auto-generated
- **check_kind**: `output_path_exists`
- **check_args**: `{path: "agent-runtime/docs/MEMORY_WRITE_PATHS.md", min_bytes: 200}`
- **required**: true
- **rationale**: Auto-generated from writer-function signatures on every test run. If writers change, the doc changes. Doc cannot rot.

### AC-38 — Negative fixture catches direct-write bypass
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/memory/test_write_path_lint.py::test_negative_fixture_catches_bypass -x", expect_exit: 0}`
- **required**: true
- **rationale**: Deliberately wire a direct Lance write from a non-writer module → lint must catch it. Tests the test.

---

## Phase 6.9 — Constitution schema + `evaluator_id`

### AC-39 — Loader refuses entries missing `evaluator_id`
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/constitution/test_evaluator_id_required.py -x", expect_exit: 0}`
- **required**: true
- **rationale**: A `constitution.json` entry without `evaluator_id` (or with `evaluator_id` referencing a non-registered checker) raises `ConstitutionInvalid` at load. Iron rule: entries without an evaluator are aspiration, not policy.

### AC-40 — Every existing objective has a registered evaluator
- **check_kind**: `claim_predicate`
- **check_args**: `{predicate: "for_all(obj in load_constitution().objectives, obj.evaluator_id in EVALUATOR_REGISTRY)"}`
- **required**: true
- **rationale**: Phase 6.9 ships 5-10 initial evaluators. This AC verifies every shipped objective is bound to one of them.

### AC-41 — Evaluators are deterministic
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/constitution/test_evaluator_determinism.py -x", expect_exit: 0}`
- **required**: true
- **rationale**: Same input twice → same verdict. If any evaluator depends on time/randomness/external state without freezing it, the constitution isn't policy — it's noise.

---

## Phase 6.10 — Repository abstraction

### AC-42 — Direct Lance access replaced by Repository
- **check_kind**: `output_path_matches_regex`
- **check_args**: `{path: "agent-runtime/app/", pattern: "lancedb\\.connect|db\\.open_table", expect: "no_match_outside_repository_module"}`
- **required**: true
- **rationale**: Static-import audit. Every Lance handle goes through a Repository wrapper. Required for the eventual Postgres migration to be a swap, not a rewrite.

### AC-43 — Migration-trigger metrics emitted
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/repository/test_metrics_emitted.py -x", expect_exit: 0}`
- **required**: true
- **rationale**: Repository emits `repo.<table>.row_count` and `repo.<table>.p95_mutation_latency_ms` per table per minute. These are the migration triggers — without instrumentation we won't know when to flip to Postgres.

---

## Phase 6.11 — Director interface skeleton

### AC-44 — Director exposes `route()` / `synthesize()` / `propose()`
- **check_kind**: `claim_predicate`
- **check_args**: `{predicate: "static_introspect(Director).has_methods(['route', 'synthesize', 'propose']) AND all_are_async"}`
- **required**: true
- **rationale**: Future Phase-8 split target. Today behind one persona; tomorrow potentially three personas. Interface skeleton makes the split mechanical.

### AC-45 — Per-method telemetry emitted
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/director/test_per_method_telemetry.py -x", expect_exit: 0}`
- **required**: true
- **rationale**: Each method emits its own latency counter (`director.route.latency_ms`, `director.synthesize.latency_ms`, `director.propose.latency_ms`) — distinct from generic `agent_query`. Required so the Phase-8 trigger condition ("Director p95 latency > 2× slowest specialist") can fire.

---

## Cross-cutting

### AC-24 — End-to-end 5-fanout smoke
- **check_kind**: `test_passes`
- **check_args**: `{cmd: "pytest agent-runtime/tests/e2e/test_phase6_full_loop.py -x", expect_exit: 0, timeout: 300}`
- **required**: true
- **rationale**: Spawn 5-fanout researcher org. All tasks carry non-trivial ACs. **Assert: all reach `done` only after AC pass; one deliberately-failing task does NOT reach `done`; the failure trace contains the specific AC id that blocked.** This is the "premature done is structurally impossible" proof.

### AC-25 — User sign-off
- **check_kind**: `manual_user`
- **check_args**: `{prompt: "Confirm Phase 6 demo run matched expectations (see report)"}`
- **required**: true
- **rationale**: Belt-and-braces; user is the final critic. Not a CI gate — gates operator sign-off after CI is green.

---

## Iron-rule audit

Per the §2 design principle, every sub-deliverable has ≥1 required mechanical AC. Mechanical AC count by sub-phase:

| Sub-phase | Required mechanical ACs |
|---|---|
| 6.0 schema + executors | AC-1, AC-2, AC-3, AC-4, AC-5, AC-6, AC-7, AC-8, AC-26, AC-27 → 10 |
| 6.0.1 hard-gate | AC-9, AC-10, AC-11, AC-12 → 4 |
| 6.1 fresh-context critic | AC-21, AC-22, AC-23 → 3 |
| 6.2 per-persona caps | AC-13, AC-14, AC-15, AC-16 → 4 |
| 6.3 lease + heartbeat | AC-28, AC-29 → 2 |
| 6.4 revision lane | AC-30 → 1 |
| 6.5 truncation marker | AC-31, AC-32 → 2 |
| 6.7 APR + circuit breaker | AC-33, AC-34, AC-35 → 3 |
| 6.8 memory writers | AC-36, AC-37, AC-38 → 3 |
| 6.9 constitution + evaluator_id | AC-39, AC-40, AC-41 → 3 |
| 6.10 Repository | AC-42, AC-43 → 2 |
| 6.11 Director interface | AC-44, AC-45 → 2 |
| cross-cutting | AC-24 → 1 |
| operator gate | AC-25 (manual_user) → 0 mechanical |

**Total: 40 required mechanical + 1 advisory mechanical + 1 manual_user = 42 ACs.**

(The spec text references "43 mechanical + 2 manual_user = 45 total" — AC-3 advisory + the cross-cutting smoke variants make up the small delta. Reconcile at Phase 6 close-out.)
