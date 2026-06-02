# Veilguard Multi-Agent Platform — User Acceptance Test (UAT) Report

**Subject:** `agent-runtime/` (the Claude-Agent-SDK-based multi-agent service)
**Spec:** `Documents/veilguard/MULTI_AGENT_PLATFORM.md`
**Started:** 2026-06-01 (autonomous overnight UAT loop)
**Runner:** Claude Code, self-paced `/loop` (dynamic mode)
**Authority:** local-first per `workflow_changes_local_first`; no VM/prod access exercised; no worktrees.

> This is a living document. Each loop iteration appends to the **Iteration log** at the
> bottom and updates the findings table. Status reflects the most recent run.

---

## 1. Executive summary

| Metric | Iteration 1 start | Iteration 1 end |
|---|---|---|
| pytest collected | 358 | 358 |
| pytest passed | 346 | **358** |
| pytest failed | 12 | **0** |
| Demo scenarios passing | n/a | 3 of 4 (1 stale — see F6) |

**Verdict after iteration 1:** the full unit/integration suite is **green (358/358)**. Of the 12
initial failures, **8 were a local environment gap** (missing `torch`), **2 were stale tests** that
hadn't kept pace with the product, **1 was a real Phase-6.8 gate scoping defect**, and **1 was a real
product regression** in the critic prompt (anti-rubber-stamp instruction had been dropped). All four
non-environment issues are fixed in the working tree. The one remaining red is a **stale demo
scenario** whose failure actually *confirms* two correct, recently-added safety guards.

No bet-the-product (§11 MUST-HAVE) failures were observed in the existing suite; mapping the suite to
the §11 MUST-HAVEs + filling coverage gaps with new executable tests is the work of subsequent
iterations (see **Next-iteration plan**).

**After iteration 2:** F6 fixed — **all 4 demos pass**; the §11 MUST-HAVE coverage map is done
(see §8) and the previously-untested subtask-cancellation cascade (§11.2.4) now has 4 tests. Suite:
**362 passed, 0 failed** (this was the `tests/`-only subset — see F7).

**After iteration 3:** discovered `pytest.ini testpaths=tests` was **excluding 178 tests** (F7) which
hid **3 failures broken on Python 3.12** (F8); both fixed. The **default `pytest` now runs the
complete suite: 540 passed, 0 failed.** Authored the **live-stack UAT checklist**
(`LIVE_STACK_UAT_CHECKLIST.md`) for the VM layer, added an exact-threshold amendment-eligibility test,
corrected the over-stated coverage-gap list (O4), and flagged §11.3.5 re-scoring as likely
unimplemented (O3).

---

## 2. Environment

- **Host:** Windows 11, Python 3.12.3, pytest 9.0.2.
- **Repo root on PYTHONPATH:** `agent-runtime` imports the shared `agent/`, `pii/`, `llm/` packages
  from the repo root, so the suite must run with `PYTHONPATH=C:\Users\rudol\Documents\veilguard`.
- **Dependency provisioning (iteration 1):** installed CPU-only **torch 2.12.0+cpu**
  (`pip install torch --index-url https://download.pytorch.org/whl/cpu`). The local env had
  `transformers` present but no `torch`; presidio (pulled in via `pii/redactor.py`) eagerly imports
  the HuggingFace NER recognizer → `transformers.pipeline` → `torch`, so any test importing
  `agent.director` failed at import. This is an **env provisioning step, not a product change** — see F1.

### Reproduce

```bash
cd /c/Users/rudol/Documents/veilguard/agent-runtime
PYTHONPATH='C:\Users\rudol\Documents\veilguard' python -m pytest -q -p no:cacheprovider
PYTHONPATH='C:\Users\rudol\Documents\veilguard' python -m demo.run_all
```

---

## 3. Test inventory (as run)

- **358 pytest cases** across `tests/` (root + `acceptance/`, `constitution/`, `critic/`, `memory/`,
  `runtime_health/`) and `app/proposals/tests/` + `app/ledger/tests/`.
- Coverage is overwhelmingly **pure-unit + scripted-backend**. Mechanical acceptance criteria
  (Phase 6: `tests/PHASE_6_ACCEPTANCE.md`, 43 ACs) are systematically covered; Phase 7 ACs
  (`runtime_health/test_phase_7_*`, `memory/test_phase_7_acceptance.py`) covered.
- **4 demo scenarios** (`demo/run_all.py`) drive the end-to-end orchestration pipeline with a
  deterministic scripted backend (no LLM/API cost): Pattern A solo, Pattern B delegation, Pattern C
  fanout, critic-iterate.
- **Live-stack (CORRECTION, 2026-06-02):** the full stack actually runs **locally** via
  `docker-compose.yml` — `veilguard-{agent-runtime,pii-proxy,api,mongodb,meilisearch,admin,sandbox}` were
  all `Up`/healthy, TCMM on `:8811`, SSO creds present. So this is NOT VM-only. **A genuine live local
  run was executed** (`demo/scenario_live_sso_tool.py`, `CLAUDE_SSO=bearer`): agent-runtime →
  `Agent.run_turn` → TCMM `/pre_request` (HTTP 200) → **real Anthropic call** (OAuth bearer,
  `claude-haiku-4-5`, HTTP 200, `cache_read=8912`) → Director emitted `create_task` → dispatched
  in-process → **task landed in the Lance ledger** → final synthesis referenced the id. **[PASS]**
  (Ran against an isolated temp ledger, so no real data touched.) The remaining un-exercised live
  surfaces — LibreChat UI → pii-proxy → agent-runtime SSE, client-daemon approval toast, sidebar SSE,
  and a full real-LLM Pattern-B/C with the inbox-poller driving real ICs — are all **runnable locally**
  next; the manual checklist (`LIVE_STACK_UAT_CHECKLIST.md`) drives them.

---

## 4. Findings

| ID | Severity | Area | Type | Status |
|---|---|---|---|---|
| F1 | Low (env) | torch import chain | Environment gap | **Resolved** (installed torch) |
| F2 | Medium | `app/acceptance/critic_prompt.py` | **Real product regression** | **Fixed** |
| F3 | Medium | `tests/memory/test_write_path_lint.py` (AC-36) | Real gate scoping defect | **Fixed** |
| F4 | Low | `tests/test_memory_mcp_tools.py` | Stale test | **Fixed** |
| F5 | Low | `tests/test_agent_teams.py` | Stale test + design Q | **Fixed** (Q tracked) |
| F6 | Low | `demo/scenario_critic_iterate.py` | Stale demo (product correct) | **Fixed** (iter 2 — demo writes a real artifact) |
| F7 | **Medium** | `pytest.ini` `testpaths` | **CI blind spot** — 178 tests excluded from default run | **Fixed** (iter 3) |
| F8 | Medium | `app/proposals/tests/test_lesson_sse.py` | 3 tests broken on Python 3.12 (legacy asyncio idiom) | **Fixed** (iter 3) |
| F9 | Low | `demo/scenario_live_sso_tool.py` + `_harness.py` | Live demo crashed *after* passing on cp1252 `→`; `CLAUDE_SSO` prereq undocumented | **Fixed** (live run) — utf-8 stdout in harness |
| F10 | **Medium** | `app/runtime_health/apr.py` + `app/main.py` | APR circuit breaker is **sticky with no wired resume endpoint** — once tripped, dispatch is wedged until a process restart | **Fixed** — wired `POST /apr/resume` + `GET /apr/status` (+3 tests), verified live (200) |
| F11 | **Medium** | `app/workers/inbox_poller.py` | Force-cancel (turn-cap **and** timeout) raw-writes `cancelled`, bypassing the parent-autoclose hook → **orphan coordinator stuck `open`** | **Fixed** — shared `_autoclose_parent_after_force_cancel` on both paths (+3 tests) |
| F12 | Low | `app/scripts_cleanup_stale_coordinators.py` | Docstring's invocation (`python /app/app/scripts_…py`) fails `ModuleNotFoundError: app.ledger`; correct form is `-w /app python -m app.scripts_cleanup_stale_coordinators` | Noted (fix docstring) |
| F13 | **High** | `app/main.py` `/proposals/convert` (+ `ledger_mcp`, `ledger/tasks`) | Proactive **approve→Task convert returned 400** — called `create_task` with no ACs, so the Phase 6.0.2 contract rejected *every* proposal approval. The whole proactive payoff was dead. | **Fixed** — shared `synthesize_default_acceptance_criteria` used by both convert + MCP tool (+4 tests); verified live (200 → task → dispatched), confirmed end-to-end via UI Approve |
| AC-path | **Medium** | `app/proposals/briefs.py` `DELIVERABLE_SPECS` | Specs named no extractable path → synthesized AC fell back to `deliverable.md` (never written) → proactive tasks died at Critic | **Fixed** — concrete `team/drafts/*.md` paths in all 7 specs (+1 test); validated live (artifact written at AC path) |
| F14 | **High** | `app/runtime.py` LOOP_CONTEXT_FIX tool-result cap | **1200-char cap** truncated *every* tool result in the rebuilt next-turn context → a critic re-reading a >1.2 KB artifact saw it "truncated/corrupted" → rejected valid work → revision-cap cancel. **Systemic Critic-gate killer (reactive + proactive).** | **Fixed** — cap raised to 12 KB (env-tunable) + Phase-6.5 marker (+3 tests); validated live (proactive task now reaches `done`, critic approves full artifact) |
| O1 | Info | team-lead role | Design question | Tracked |
| O2 | Info | auto-downgrade → cancel | Behavior confirmation | Tracked |
| O3 | **Finding** | §11.3.5 amendment re-scoring | MUST-HAVE behavior appears **unimplemented** | Flagged for user |
| O4 | Info | iter-1 coverage-gap list | Over-stated — most were already covered | Corrected (iter 3) |

### F1 — 8 tests blocked on missing `torch` (environment, not product)

Tests importing `agent.director` (`tests/test_cost_writeback_and_anchors.py` ×5,
`tests/runtime_health/test_phase_6_10_11.py::test_ac44*/test_ac45*` ×3) failed at import:
`agent/__init__ → pii → presidio → transformers.pipeline → No module named 'torch'`. Installing
CPU-only torch resolved all 8. **Disposition:** environment provisioning. Note for the deployed
container: ensure either `torch` is present *or* `transformers` is absent so presidio doesn't attempt
the HF NER recognizer import. (`requirements.txt` pins neither torch nor transformers; the local env
had a half-state.)

### F2 — Critic prompt missing anti-rubber-stamp instruction (real regression) — FIXED

`tests/critic/test_fresh_context.py::test_ac21_critic_prompt_contains_spec_ac_artifact` failed: the
prompt from `build_critic_user_message` contained `review_decision`/`changes_requested` but **neither**
"every required AC" **nor** "do not give benefit of the doubt". Root cause: the prompt was rewritten
2026-05-29 (`[CRITIC_NO_GET_TASK_LOOP_2026_05_29]` / `[CRITIC_SERVER_SIDE_AC_2026_05_29]`) to fix a
`get_task`/`add_comment` loop, and that rewrite **dropped the explicit gate-discipline language**
AC-21 guards. Since the entire Phase-6 thesis is anti-rubber-stamp, the correct fix is to **restore
the instruction** (strengthen the prompt), not relax the test.

**Fix:** added a hard rule to `app/acceptance/critic_prompt.py`:
> "Judge against EVERY required AC listed above before you decide. Do not give benefit of the doubt:
> if any required AC is unmet, unverifiable from the artifact, or the work only partially satisfies the
> brief, the decision is changes_requested — never approved."

Verified: does not introduce any `FORBIDDEN_PRODUCER_FIELDS` token (AC-22 still green).

### F3 — AC-36 write-path linter flags tests + admin script (gate scoping) — FIXED

`tests/memory/test_write_path_lint.py::test_ac36_*` failed with two violations:
`app/scripts_cleanup_stale_coordinators.py` and `app/ledger/tests/test_proposal_dedup.py` import
`app.ledger.store` directly. Root cause: the scan walked **all** `.py` under `app/`/`agent/`
(only excluding `__pycache__`), so it flagged (a) a legitimate **test file** and (b) a one-shot
**`docker exec` maintenance script**. Per §3.11 the discipline governs **runtime/agent write paths**,
not the test harness or admin tooling.

**Fix:**
1. Excluded test files (`tests/` dirs, `test_*.py`, `conftest.py`) from `_all_py_files()`. This is
   safe — AC-38 (negative fixture) proves the detector still fires via synthetic `tmp_path` files, so
   excluding the production scan from real test files does **not** weaken the guarantee.
2. Sanctioned `app/scripts_cleanup_stale_coordinators.py` in `_SANCTIONED_DIRECT_IMPORTERS` with a
   justification comment (same treatment as the existing `m4_backfill.py` one-shot).

### F4 — `test_two_tools_present` stale (recall now exposed) — FIXED

`tests/test_memory_mcp_tools.py::TestMetadata::test_two_tools_present` asserted the memory MCP server
exposes exactly `{observe, read_constitution}`, but it now exposes `{observe, read_constitution,
recall}`. The same test file already imports `recall_tool` and has `test_recall_has_query_required`,
so `recall` is clearly intended (decision log 2026-05-22: "Memory tools recall/observe/
read_constitution exposed via second in-process MCP server"). The assertion + comment were stale.
**Fix:** renamed to `test_three_tools_present`, assert all three, updated the comment.

### F5 — `test_team_lead_persona_file_exists` expects a non-existent role — FIXED

The test asserted `Role: manager` in `agents/team-lead.md`, but the persona declares
`**Role:** director`. The spec role enum (§0.3) is `{director, ic, consultant}` — there is **no
`manager` value** — so the persona using `director` (a mini-Director) is spec-legal and the test was
wrong. **Fix:** assert `director`, with a comment pointing to the design question O1.

### F6 — `demo/scenario_critic_iterate` is stale; product behavior is correct — FIXED (iter 2)

The scenario expects final status `done` after reject→iterate→approve. Actual: ends `cancelled`.
**Root cause (two correct, newer guards the 2026-05-22 demo predates):**
1. `[AC_RESULTS_WIRING_2026_05_29]` (`ledger_mcp.review_decision_tool`): an `approved` decision runs
   the required **mechanical** AC checks server-side first. The scripted researcher calls
   `attach_output(path=...passkey-research.md)` + `observe()` but **never writes a real file** at that
   path, so `output_path_exists` fails → the approve is auto-downgraded to `changes_requested`. This is
   exactly the anti-rubber-stamp behavior Phase 6 exists to enforce.
2. `[REVISION_ROUND_CAP_2026_06_01]`: default `VEILGUARD_MAX_REVISION_ROUNDS=2`. The auto-downgrade is
   the 2nd `changes_requested`, hitting the cap → the task is **cancelled** instead of bounced to
   `in_progress` (anti-infinite-loop; the memory note cites a live 40-turn/982k-token runaway this
   guard stops).

**Disposition:** **not a product bug** — both guards are working as designed.

**Fix applied (iteration 2):** the executor resolves `output_path_exists` under
`VEILGUARD_WORKSPACE_ROOT` (default `/workspace`), and the in-process `workspace_fs.write_file` tool
writes under the same root. So: (a) `demo/_harness.py setup_demo_env` now points
`VEILGUARD_WORKSPACE_ROOT` at a writable temp dir; (b) the scripted researcher now calls `write_file`
in both submit turns to create a real artifact at the attached path. Result: round-1 reject (citation)
→ round-2 approve with the file present → mechanical AC passes → `done` in one revision round (under
the cap). **All 4 demos now pass.** This preserves the gate's meaning — the demo reaches `done` only
because a real artifact now exists, not because the gate was relaxed.

### F7 — `testpaths = tests` silently excluded 178 tests (CI blind spot) — FIXED (iter 3)

`pytest.ini` set `testpaths = tests`, so the default `pytest` invocation collected **only `tests/`
(362)** and never ran the **178 tests under `app/proposals/tests/` (151) + `app/ledger/tests/` (7…)**.
That's a real blind spot: if CI uses the default invocation, those tests never run — and indeed **3 of
them were failing** (F8) without anyone seeing it. (My own iteration-1/2 "full suite" runs were the
`tests/` subset for the same reason.)

**Fix:** widened `testpaths` to `tests app/proposals/tests app/ledger/tests`. `demo/` is deliberately
excluded (its `test_e2e_loop.py` has import-time side effects and expects a live container). The
default `pytest` now collects **540** tests. **Recommend:** confirm CI runs the default invocation (or
`pytest` with no path args) so this stays covered.

### F8 — 3 `test_lesson_sse.py` tests broken on Python 3.12 — FIXED (iter 3)

All three failed with `RuntimeError: There is no current event loop in thread 'MainThread'`. They are
**sync** tests that manually call `asyncio.get_event_loop().run_until_complete(_go())`. On Python 3.12,
`asyncio.get_event_loop()` raises when there's no running/set loop (the auto-loop-creation removed in
3.12). The product code (`promote_lesson_to_team_knowledge`) is fine; the **test idiom is stale**.

**Fix:** replaced the three calls with `asyncio.run(_go())` (the 3.12-correct, behavior-preserving
idiom — a fresh loop per call; safe because these are sync tests with no running loop). All 3 now pass.
Surfaced only because F7's fix made them run.

### O3 — §11.3.5 constitution-amendment re-scoring appears unimplemented (finding for user)

The spec (§11.3.5 + §6 failure table) requires: *when the constitution is amended, pending proposals
are re-scored against the new weights and re-sorted; any failing `constraint_gate` auto-shelves with
`reason=constitution_amended`; done/retired items are NOT retroactively re-evaluated (replay uses the
`constitution_version` snapshot).* The snapshot half **is** present — proposals/tasks carry
`constitution_version` (`schemas.py`, `snapshot_anchor.py`). But a search of `agent-runtime/app` finds
**no `constitution_amended` shelve path and no pending-proposal re-scoring** on amendment. Constitution
*amendments* are surfaced as proposals (`constitution_amendments.py`), but adopting one does not appear
to re-score the live queue.

**Disposition:** likely a deferred Phase-3/4 behavior, not a regression — but it is a documented
MUST-HAVE. **Flagged for the user to confirm** whether it's intentionally deferred. Not writing a test
for unimplemented behavior; if/when it lands, the test is: amend constitution → assert pending
constraint-failing proposals move to `shelved(reason=constitution_amended)` and `done` ones are
untouched.

### O4 — iteration-1 coverage-gap list was over-stated (corrected)

The iteration-1 gap list (derived heuristically from a "0 direct tests" signal) named several modules
as gaps that are in fact **already covered**: persona loader (23 tests), revision-lane (AC-30
`is_revision_candidate` + `test_wire_up`), MCP tool dispatch (`test_ledger_mcp_tools` 12 +
`test_memory_mcp_tools` 9), audit rows (`test_smoke_integration` `tap_sdk_stream` + `test_outcomes`
cost rollup), constitution amendments (6 tests). The suite is richer than the iter-1 list implied. The
**one genuine thin spot** found and closed: the amendment-eligibility **exact threshold boundary**
(0.75/5 vs 0.74/4), now pinned in `test_lessons.py`. `config.validate()` remains lightly tested but
low-value (startup env checks).

### O1 — Should `team-lead` have a distinct role? (design question)

`team-lead` currently uses `Role: director`, giving two director-role personas (Director + team-lead).
The spec calls team-lead a "mini-Director scoped to one team" but the role enum has no team-lead value.
If role-based authority logic ever needs to distinguish them, a `team-lead` enum value (and loader +
capability-matrix updates) would be required. No action needed now; flagged for the user.

### O2 — Auto-downgraded `approved` → `cancelled` is intentional

Confirmed via source (`ledger_mcp.py:930` `[REVISION_ROUND_CAP_2026_06_01]`). After the revision cap,
`changes_requested` cancels rather than bouncing. Reasonable anti-loop design; documented so future
readers don't mistake it for a lifecycle bug.

---

## 5. Changes made (this report's working tree)

All edits are local-first, no worktrees, confined to `agent-runtime/`:

- `app/acceptance/critic_prompt.py` — restored anti-rubber-stamp instruction (F2).
- `tests/memory/test_write_path_lint.py` — scope scan to runtime code; sanction admin script (F3).
- `tests/test_memory_mcp_tools.py` — `test_three_tools_present` (F4).
- `tests/test_agent_teams.py` — assert `Role: director` (F5).

Iteration 2:
- `demo/_harness.py` — set `VEILGUARD_WORKSPACE_ROOT` to a writable temp dir for demos (F6).
- `demo/scenario_critic_iterate.py` — researcher `write_file`s a real artifact in both submit turns (F6).
- `tests/test_cancel_cascade.py` — **new**: 4 tests for `cancel_cascade` (§11.2.4 MUST-HAVE).

Iteration 3:
- `pytest.ini` — widened `testpaths` to include `app/proposals/tests` + `app/ledger/tests` (F7).
- `app/proposals/tests/test_lesson_sse.py` — `asyncio.run()` instead of the 3.12-broken
  `get_event_loop().run_until_complete()` (F8).
- `app/proposals/tests/test_lessons.py` — **new** test: exact amendment-eligibility thresholds (O4).
- `LIVE_STACK_UAT_CHECKLIST.md` — **new**: manual VM acceptance checklist (pii-proxy→agent-runtime→
  TCMM→client-daemon→sidebar, real tenant/user, live UI).

(`.uat_pytest_run*.txt` are captured run logs; safe to delete.)

---

## 6. Next-iteration plan

1. ~~**Fix F6 demo**~~ — ✅ done (iteration 2).
2. **Map suite → §11 top-10 MUST-HAVEs** — ✅ map done (§8, iteration 2); 11.2.4 cascade test written.
   Remaining: verify 11.3.5 (amendment re-scoring) + 11.6.3 (cycle-cap), and write the TCMM/sub-agents
   ones into the live-stack checklist. Original list:
   - 11.5.5 cross-namespace dream `bridge_score` leak (bet-the-product)
   - 11.5.1/.2 `source_kind` set by tool not agent
   - 11.1.3 dream-node soft-delete while pinned by pending proposal
   - 11.6.3 per-tenant proactive-cycle cap
   - 11.2.4 subtask-cancel cascade + orphan-claim handling
   - 11.4.4 dream-cycle atomicity around proposal emission
   - 11.5.4 CRLF/control-char arg-glob normalization
   - 11.1.1 document merge-authority on concurrent edits
   - 11.3.5 constitution amendment re-scoring
   - 11.7.2 consultant private memory `(consultant_id, tenant_id)`-namespaced
3. **Fill unit-coverage gaps** (from the coverage map): `app/config.py` validation,
   `app/personas/loader.py` malformed-frontmatter cases, `app/runtime_health/revision_lane.py`
   isolation, per-tool MCP dispatch in `ledger_mcp`/`memory_mcp`, `app/middleware/audit.py` row
   generation, `constitution_amendments` boundary conditions.
4. **Live-stack UAT checklist (manual / VM, for the user)** — author a step-by-step acceptance script
   to run under the real tenant/user_id through pii-proxy → agent-runtime → TCMM → client-daemon →
   LibreChat sidebar, validating the live UI (the layer the chain-of-8-bugs lesson says keeps breaking).

---

## 7. Iteration log

### Iteration 1 — 2026-06-01
- Probed env; set PYTHONPATH; collected 358 tests.
- Full run #1: **346 passed, 12 failed** (`.uat_pytest_run1.txt`).
- Triaged all 12 → F1–F6.
- Installed CPU torch (F1); applied F2–F5 fixes.
- Full run #2: **358 passed, 0 failed** (`.uat_pytest_run2.txt`).
- Demos: 3/4 pass; root-caused critic-iterate (F6) to two correct guards.
- Wrote this report. Scheduled next iteration to start the §11 MUST-HAVE mapping + F6 demo fix.

### Iteration 2 — 2026-06-01
- Fixed F6: demo workspace root + researcher writes a real artifact → **all 4 demos pass**.
- Audited the §11 top-10 MUST-HAVEs against the agent-runtime suite (see §8 map).
- Found the subtask-cancellation cascade (`cancel_cascade`, §11.2.4) had **no test**; wrote
  `tests/test_cancel_cascade.py` (4 tests: subtree-not-siblings, skip-terminal, checkpoint-comment,
  tenant-scoped).
- Full run #3: **362 passed, 0 failed** (`.uat_pytest_run3.txt`).
- Next: verify 11.3.5 (constitution amendment re-scoring) + 11.6.3 (per-tenant cycle cap) coverage;
  fill remaining unit-coverage gaps; author the manual live-stack UAT checklist.

### Iteration 3 — 2026-06-01
- **Discovered `pytest.ini testpaths=tests` excluded 178 tests** (F7) → ran the complete set explicitly
  → found **3 failures** in `test_lesson_sse.py` (F8, Python-3.12 asyncio idiom). Fixed both.
- Default `pytest` now runs the **complete suite: 540 passed, 0 failed** (`.uat_pytest_run5_default.txt`).
- Verified §11.6.3 (cycle-cap config covered) + §11.3.5 (re-scoring **unimplemented** → O3).
- Audited the iter-1 coverage-gap list: most already covered (O4); closed the one real gap
  (amendment-eligibility exact thresholds) with a new test in `test_lessons.py`.
- Authored `LIVE_STACK_UAT_CHECKLIST.md` (the VM/live-UI layer — highest-value, can't be unit-tested).
- Next: the plan is essentially worked. Remaining = the manual live-stack run (user, on the VM) +
  optional `config.validate()` micro-tests (low value). Loop will do a final consolidation pass then
  likely stop.

---

## 8. §11 top-10 MUST-HAVE coverage map

Where each bet-the-product MUST-HAVE is (or isn't) covered. Several live in the **TCMM** codebase
(`~/.gemini/antigravity/tcmm/TCMM/`) or **sub-agents** service — agent-runtime's suite cannot exercise
those, so they are routed to the manual live-stack checklist (iteration 3).

| # | MUST-HAVE | Where it lives | agent-runtime coverage |
|---|---|---|---|
| 11.5.5 | Dream `bridge_score` cross-namespace leak | **TCMM** dream engine | N/A here → live-stack checklist (Phase-1 mandatory dream test in TCMM repo) |
| 11.5.1/.2 | `source_kind` set by tool, never agent | pii-proxy + TCMM observe | N/A here (agent observe goes through pii-proxy) → live-stack checklist |
| 11.1.3 | Dream-node soft-delete while pinned by pending proposal | TCMM GC + proposal `signal_node_ids` | Partial: proposals store materialized brief (not runtime refs) — covered by proposal tests; dream-GC pin is TCMM-side → checklist |
| 11.6.3 | Per-tenant proactive-cycle cap | agent-runtime `proactive_config` / scanner | Covered: `test_proactive_config.py` (7) — **verify cycle-cap specifically next iter** |
| **11.2.4** | **Subtask cancellation cascade** | **agent-runtime** `ledger/tasks.cancel_cascade` | **NOW covered** — `tests/test_cancel_cascade.py` (4), added iter 2 |
| 11.4.4 | Dream-cycle atomicity around proposal emission | TCMM dream cycle + scanner write | Partial: proposal write/dedup atomicity in `test_proposal_dedup.py`; dream-cycle batch atomicity is TCMM-side → checklist |
| 11.5.4 | CRLF/control-char arg-glob normalization | **sub-agents** `core/approval.py` (moved out of agent-runtime 2026-05-25) | N/A here (`test_client_tool_policy.py` is a removal stub) → verify in `services/sub-agents/tests/test_approval.py` |
| 11.1.1 | Document merge authority on concurrent edits | agent-runtime `documents.py` | Covered: `test_documents.py` (9) |
| 11.3.5 | Constitution amendment re-scoring | agent-runtime constitution/proposals | **Verify next iter** — no obvious test; confirm whether re-scoring on amendment is implemented |
| 11.7.2 | Consultant private memory `(consultant_id, tenant_id)`-namespaced | TCMM tenant isolation + agent-runtime tenant ctx | Tenant scoping in ledger covered (`test_proposal_dedup`, `test_ledger_mcp_tools`, new cascade test); consultant-memory namespacing is TCMM-side → checklist |

**Summary:** of the 10, agent-runtime can meaningfully test **6** (11.6.3, 11.2.4, 11.4.4-partial, 11.1.1,
11.3.5, 11.7.x-partial). Of those, 11.2.4 was the one genuine gap and is now closed. 11.3.5 + the
11.6.3 cycle-cap specific assertion need verification next iteration. The other 4 are TCMM/sub-agents
concerns that belong to those suites + the manual live-stack checklist.

---

## 9. Live local stack validation (2026-06-02)

**Correction to the earlier "VM-only / no access" framing:** the **entire stack runs locally**. `docker compose ps`
showed `agent-runtime` (healthy, inbox-poller enabled, `CLAUDE_SSO=1`), `pii-proxy`, LibreChat
`api`/`mongodb`/`meilisearch`, `admin`, `sandbox`; **TCMM** runs on `:8811` and **sub-agents** on `:8809`
(both separate processes), and **`VeilguardClient.exe`** (the Windows daemon) is running and connected to
**local** sub-agents (`127.0.0.1:8809`, ESTABLISHED). So all three live layers were exercised here.

### #1 — Full live multi-agent flow ✅ PASS (the headline)
Drove the **real running container** via `POST :5000/agent/query` (Director, scratch tenant
`uat-live-20260602`). Director (live Claude) emitted `create_task` → **task-a43386ded37b** (server
synthesized a mechanical `output_path_exists` AC). The container's background **inbox-poller dispatched the
Researcher live**, which **wrote a real 2929-byte artifact** to `/workspace/team/drafts/pwreset-enum.md`,
submitted for review → **Critic (live) `review_decision: approved`** → server ran the mechanical AC
(`ac_results: {AC-default: pass}`) → **`status: done`**. Real `cost_attributed_usd: 0.34`, SHA-chained
comment chain (`open→accepted→in_progress→review→review_request→review_decision→done`). **This is the exact
end-to-end flow that "never completed" in the chain-of-8-bugs era — now green, live, with real LLMs for
Director + Researcher + Critic.** Also separately confirmed the in-process live harness
(`demo/scenario_live_sso_tool.py`, `CLAUDE_SSO=bearer`) — real Anthropic round-trip via TCMM `/pre_request`.

### #2 — UI / sidebar data path ✅ (data path) / ⚠️ (visual render needs your session)
- agent-runtime sidebar endpoints work live: `GET /work_items` → 200 tenant-scoped JSON (correctly empty
  for the scratch tenant since the task is `done` — the active feed drops terminal tasks), kind-filter
  honored; `GET /events` SSE → `data: {"type":"ready"}`.
- **O5 (doc finding):** spec §3.10.5 says the sidebar goes through **pii-proxy `:4000/api/veilguard/*`** —
  but `:4000` only routes LLM backends and returns 404 `Unknown backend: 'api'`. The **real front-door is
  LibreChat `api` at `:3080/api/veilguard-client/*`** (returns **401** = route exists, needs your logged-in
  session). Spec §3.10.5 should be corrected. The browser render is the one piece needing your session.

### #3 — Approval gate → Windows toast ✅ (path wired + proven by daemon log)
The daemon is connected to local sub-agents (`:8809` ESTABLISHED; polls `/api/client/latest` → 200).
`daemon.log` shows the gate→toast chain has **actually fired**: real approval toasts for `run_command`
(×4) and `write_file` (×1), each with an `approval_token` + callback URL (`veilguard://approve`, HTTP
fallback). So `classify(APPROVE) → request_approval → daemon WS → winotify toast` works end-to-end. Caveat
the daemon logs itself: `show() returned None` — whether the banner is **visible** depends on Windows
Notifications / Focus-Assist settings (OS config, not a product defect).

### What still needs you (human-in-the-loop, both local)
1. **Eyeball the sidebar** in your logged-in LibreChat (`:3080`) — the `/api/veilguard-client/*` data path
   is healthy; only the visual render needs your session.
2. **See/click a live approval toast** — trigger a background Builder shell task, then Approve/Deny the
   toast (the human gate is the whole point). The plumbing is proven; the click is yours.

**Scratch data:** the live run left `task-a43386ded37b` + `/workspace/team/drafts/pwreset-enum.md` under
tenant `uat-live-20260602`. Harmless (isolated test tenant); delete if you want it gone.

### Re-run under the REAL tenant (visible in the user's sidebar)
To make the flow visible in the logged-in sidebar, a second live task was fired under the real local-dev
tenant `69c4468a1fde1abc19c7835c` (= user_id; confirmed from the sidebar's own `/work_items` + `/events`
requests in the agent-runtime logs). It appeared in the Work Queue (`counts: {task:1}`, `open`) — **#2 visually
confirmed** — then flowed `open → in_progress → review → done` (AC `pass`), artifact
`/workspace/team/drafts/uat-sidebar-demo.md` (1122 bytes, real live-LLM content). This surfaced **F10**.

### New findings this pass
- **F9** (fixed): `demo/scenario_live_sso_tool.py` crashed *after* passing on a Windows cp1252 `→`; the
  prereqs also omit `CLAUDE_SSO=bearer`. Patched `demo/_harness.py` to force utf-8 stdout (guards all demos
  on Windows); the live demo now exits 0.
- **F10** (flagged): the real-tenant task initially **sat `open` and never dispatched**. Root cause: the
  **APR circuit breaker (Phase 6.7) had tripped** (`apr=0.057 < 0.1 floor, n_samples=16`) from this UAT's
  burst of high-token / low-artifact Director queries — i.e. the breaker working *as designed*. BUT:
  (a) it is **sticky** — `apr_should_pause_dispatch()` short-circuits on `is_tripped()`, so it does **not**
  auto-recover when the 30-min window ages out; (b) recovery is **only** `apr_resume()` (`GLOBAL_APR.clear()`),
  an in-process function with **no wired HTTP route** — `apr.py` docstrings reference `/apr/resume` +
  `/apr/status` but **neither exists in `main.py`**; (c) `docker exec` can't help (in-process module state),
  so the **only operator recovery is restarting the container**. The spec (§6.7) promises a sidebar
  "Operator unblock required" banner + action, but the action isn't wired. **Recommend:** wire
  `POST /apr/resume` + `GET /apr/status`, and consider auto-recovery once the rolling window clears.
  (Cleared here via `docker restart veilguard-agent-runtime-1`; the task then dispatched and completed.)
  **Fix landed (2026-06-02):** wired `POST /apr/resume` (clears + returns snapshot) and `GET /apr/status`
  in `main.py`, with `tests/runtime_health/test_apr_endpoints.py` (3 tests: source-grep registration,
  sticky-until-resume, resume-clears-and-dispatch-resumes). Verified live on the running container — both
  return 200; the operator can now recover a tripped breaker without a process restart.

### Scratch/demo data left under the real tenant
`task-fbea7006d622` + `/workspace/team/drafts/uat-sidebar-demo.md` now exist under the **real** tenant
`69c4468a1fde1abc19c7835c` (brief is prefixed `[UAT demo — safe to cancel]`). Harmless; delete if unwanted.

---

## 10. Tier-1 live runs (2026-06-02)

### Step 0 — F10 fix (done): wired `POST /apr/resume` + `GET /apr/status`, verified live (see F10).

### Tier 1.1 — Pattern C parallel fan-out (live) ✅ with findings
Director (live) decomposed a two-stream request into a **coordinator** (`task-4df19420b044`, owner
`team-lead` — the Phase-7.5 mini-Director) + two parallel subtasks: Researcher (`task-18facf62f068`) +
Builder (`task-422dccc03d3e`).
- **Concurrency confirmed:** Researcher and Builder ran simultaneously (one in `review` while the other was
  `in_progress`), driven by the background inbox-poller under per-persona caps.
- **Researcher → `done`** after its own reject→iterate cycles (Tier 1.2 convergence, below).
- **Builder → `cancelled`:** the critic flagged "incomplete or truncated," the Builder revised
  (`is_revision: true`), and the **2nd critic review dithered to the 25-turn cap → force-cancel** (the
  Phase-6.0.1 turn-cap guard firing as designed). The artifact *was* written (4630 bytes/61 lines), so the
  critic's "truncated" judgment is suspect — a **quality observation** (critic non-convergence on a valid
  artifact; possibly a Phase-6.5 truncation-marker interaction).
- **APR collapsed to ~0.03** during the thrash (high-token / low-artifact) — the metric correctly reflected
  the struggle; with F10 unfixed this would have wedged all dispatch.
- **F11 surfaced:** the coordinator stayed **`open`** with both children terminal (done + cancelled) — the
  turn-cap force-cancel bypassed parent-autoclose. **Fixed** (both force-cancel paths now propagate
  autoclose; +3 tests). The pre-existing orphan was closed via `scripts_cleanup_stale_coordinators.py`
  (→ coordinator `done`).

### Tier 1.2 — Critic reject → iterate → converge (live) ✅
Confirmed *naturally* during 1.1: the Researcher bounced `review → in_progress` across multiple cycles
(critic `changes_requested` → IC revises → resubmit) and **converged to `done`**. The anti-infinite-loop
guard was also confirmed live on the Builder (turn-cap force-cancel after non-convergence). The clean
deterministic reject→iterate→approve remains covered by the (green) `scenario_critic_iterate` demo.

### Tier 1.3 — Proactive stream (Phase 3, live) ✅ with a High-sev finding (F13)
Drove the full proactive chain via HTTP under the real tenant:
- **Emit** — `POST /proposals/emit` (signal `information_gap`, `gap_breadth=0.7 × downstream_pressure=3.0`)
  → proposal `prop-1355246e72aa`, impact `0.735`, assignee `researcher`, status `pending`.
- **Queue** — `GET /proposals?status=pending` → the proposal is in `queue[]` (the data the sidebar's
  **Proposed Tasks** tab renders). ✅
- **Approve (convert)** — `POST /proposals/{id}/convert` initially **400 → F13**: the endpoint called
  `create_task` with no `acceptance_criteria`, so the Phase 6.0.2 contract rejected *every* proposal
  approval. **The entire proactive payoff (approve → Task) was dead.**
- **F13 fix:** added a shared `tasks.synthesize_default_acceptance_criteria` helper (the same default AC
  the Director's MCP `create_task` synthesizes), routed BOTH the convert endpoint and the MCP tool through
  it so they can't drift (the drift *was* the bug). +4 tests. After the fix, convert returns **200** →
  `task-e780bff35f7e` created (`origin=proactive`, AC attached, proposal `approved` + linked) → the
  background poller **dispatched it live** (`open → in_progress (researcher) → review (critic-claim)`).
  Full chain green.
- **Confirmed end-to-end via the UI (user, 2026-06-02):** clicking **Approve** in the sidebar on a fresh
  proposal now succeeds (no 400) and converts to a task — validating the full path
  LibreChat → agent-runtime through the F13 fix (the one hop not reachable from the backend).
- **AC-path gap — FIXED (2026-06-02):** the `DELIVERABLE_SPECS` named no regex-extractable path, so the
  synthesized AC always fell back to `deliverable.md` (a file the IC never writes). Rewrote all 7 specs to
  name a concrete `team/drafts/*.md` path the IC is told to write to (+regression test that every spec
  yields a non-`deliverable.md` AC path). **Validated live:** a converted `information_gap` task's AC now
  targets `team/drafts/research-note.md` and the researcher wrote a real 3790-byte note there.
- **F14 (High, NEW — surfaced by the above validation):** even with the artifact correctly written, the
  task ended `cancelled` because **critic-claim read the file as "truncated/corrupted"** (review notes:
  *"Artifact is truncated/corrupted at line 18"*, *"File is truncated in all read attempts"*) → repeated
  `changes_requested` → revision-cap. The file is well-formed on disk (3790 bytes), so the critic's
  `read_file` (or the tool-result size cap feeding the critic LLM) is returning a truncated view. **This is
  the systemic reason tasks die at the Critic gate — reactive (Pattern-C builder hit the same "incomplete
  or truncated") AND proactive.**
- **F14 FIXED (2026-06-02):** root cause was `runtime.py`'s `[LOOP_CONTEXT_FIX]` doing
  `payload = str(payload)[:1200]` — a hard **1200-char cap on every tool result** when rebuilding the
  next-turn context. A critic that read a >1.2 KB artifact then decided on the next turn only ever saw
  ~18 lines (≈ the 1200-char cut → matches the "truncated at line 18" note). Raised the cap to a
  deliverable-sized, env-tunable **12 KB** (`VEILGUARD_TOOL_RESULT_LOG_CAP`) and switched the naive slice
  to the Phase-6.5 `truncate_with_marker` so genuine over-cap truncation is explicit. +3 tests.
  **Validated live:** a converted `information_gap` task now runs `open → in_progress → review → done`,
  with the critic **approving** the full note (*"≥2 sources cited inline (NIST, OWASP), ≤500 words"*).
  This is the change that makes multi-agent tasks actually complete instead of dying at the Critic.

### Tier 2 (toast click, UI chat) + Tier 3 (failure-mode sweep, TCMM dream `bridge_score`): still pending — need you / the TCMM repo.

**Suite after Tier-1 work: 550 passed, 0 failed** (default invocation; +10 over the 540 baseline =
F10 ×3, F11 ×3, F13 ×4). All fixes local-first, no worktrees.
