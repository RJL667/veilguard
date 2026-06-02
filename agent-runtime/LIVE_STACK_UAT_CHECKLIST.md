# Veilguard Multi-Agent — Live-Stack UAT Checklist (manual, run on the VM)

**Companion to:** `agent-runtime/UAT_REPORT.md` (which covers the automated suite).
**Why this exists:** the pytest suite + scripted demos validate logic in isolation. They do **not**
exercise the wired stack (pii-proxy → agent-runtime → TCMM → client-daemon → LibreChat sidebar) under
a **real tenant/user_id** with the **live UI**. Per the hard-won lesson in
`architecture_multiagent_flow_fixes`: *the chain-of-8-bugs all passed component checks but the live
flow never completed; always test under the real tenant/user_id and validate the live UI, not just
ledger rows.* This checklist is that test.

**Run as:** the operator, on the prod/staging VM, signed into LibreChat as a real user, with the
Windows client-daemon paired. **Do not** run under `default`/`ptest` namespaces — use the real
tenant/user. Record the actual `tenant_id` + `user_id` you used at the top of your run.

> Convention: each check is **PRECONDITION → STEPS → EXPECT → PASS/FAIL**. Mark ❏→✅/❌ and note the
> observed value. A check that can only be verified in ledger rows AND the UI must pass in **both**.

---

## 0. Pre-flight (environment)

| # | Check | How | Expect |
|---|---|---|---|
| 0.1 | Containers up | `docker compose ps` | `agent-runtime`, `pii-proxy`, `api`, `tcmm-service` all `Up` |
| 0.2 | Routing flag on | `docker exec veilguard-agent-runtime-1 printenv AGENT_RUNTIME_ENABLED` + allowlist | `true`; your user in `AGENT_RUNTIME_USER_ALLOWLIST` (or empty = all) |
| 0.3 | Internal secret set | both `pii-proxy` + `api` have `VEILGUARD_INTERNAL_SECRET` | non-empty, identical (else 503 cascade — see `architecture_internal_secret_503`) |
| 0.4 | Lance dir ownership | `ls -ln /home/rudol/veilguard/tcmm-data/.../tcmm.db` | owned by `rudol`, NOT root (else recall silently returns 0 — `architecture_lance_index_perms`) |
| 0.5 | `.env` clean | `file /home/rudol/veilguard/.env` | no `CRLF`; no cp1252 bytes (`workflow_env_crlf_gotcha`) |
| 0.6 | Workspace root | `printenv VEILGUARD_WORKSPACE_ROOT` in agent-runtime | a real writable mounted dir (AC `output_path_exists` resolves here) |
| 0.7 | Daemon paired + capability handshake | sidebar Daemons tab shows daemon connected | `approval_gate: true` advertised; not "missing approval_gate" warning |

---

## 1. Cost prerequisite — cache stability (Phase 0.1)

| # | Check | Steps | Expect |
|---|---|---|---|
| 1.1 | Warm cache on sibling sub-calls | Run a Pattern C fanout (check 4.3), then query `pii_audit` for the turn's rows | `cache_read` ≫ `cache_create` on the 2nd+ sub-calls of the same parent; not a fresh `cache_create` per sub-call |
| 1.2 | Cold-start sanity | First message of a NEW conversation | one `cache_create` (cold), subsequent turns `cache_read` |

**Why:** §10.5.4 — cache-stable prefix is the single highest-ROI item ($24/tenant/day). A regression
here shows up as `cache_create` on every sub-call.

---

## 2. The chain-of-8-bugs regression gate (run FIRST — these are the known live-only failures)

These map 1:1 to `architecture_multiagent_flow_fixes`. If any fails, stop and triage — the rest of the
checklist depends on them.

| # | Check | Steps | Expect |
|---|---|---|---|
| 2.1 | read/write_file hit server workspace, not a dead daemon | Builder writes an artifact during a task | file appears under `VEILGUARD_WORKSPACE_ROOT`; no "TOOL UNAVAILABLE" |
| 2.2 | IC keeps task brief across turns | delegate a multi-turn task | IC's later turns still reference the original brief (no `[LOOP_CONTEXT_FIX]` regression) |
| 2.3 | Offline web tools soft-degrade | run a research task with web unavailable | task continues `degraded=True`, does NOT loop on `isError` |
| 2.4 | Critic review completes (no NameError, no add_comment loop) | submit a deliverable for review | critic emits exactly one `review_decision`, transitions status; does NOT loop on add_comment to max_turns |
| 2.5 | `ac_results` written + read by hard-gate | a task with a mechanical AC reaches review | `review_decision(approved)` runs server-side AC check; `done` only if AC passes (else auto-downgrade — see 5.x) |
| 2.6 | **Full lifecycle completes under the REAL tenant** | one delegated task end-to-end | `open → in_progress → review → done` actually reaches `done`; verify in BOTH the sidebar AND `agent_tasks` |
| 2.7 | Tested in the real namespace | confirm `tenant_id`/`user_id` on the rows | match your live user, NOT `default`/`ptest` |

---

## 3. TCMM memory integration (live)

| # | Check | Steps | Expect |
|---|---|---|---|
| 3.1 | observe persists to archive synchronously | IC observes a finding | block appears in TCMM archive immediately (NLP/embedding is background — `architecture_ingest_archive_first`); NOT blocked by AIStudio 429 / GPU-absent |
| 3.2 | recall returns hits | Director/IC recall on a topic just observed | ≥1 hit; if 0, check `_indices` ownership (0.4) before suspecting logic |
| 3.3 | provenance stamped | inspect a recalled block | `author=agent:<aid>` / `extracted_by` set; user turns `source_kind=USER` |
| 3.4 | channel stamping | observe to team_knowledge vs agent_private | `channel` column set; visibility filter honors it (`TCMM_CHANNEL_ARCHITECTURE.md`) |

---

## 4. Interaction patterns (live, via chat)

| # | Pattern | User message | Expect |
|---|---|---|---|
| 4.1 | A — Solo | "what's our exposure to <recent CVE>?" | Director answers from recall in one turn; no delegation; sidebar shows no new task |
| 4.2 | B — Delegation | "write a script that does X" | Director creates+assigns a task to Builder; sidebar Work Items shows it `in_progress`; Director returns the deliverable when done |
| 4.3 | C — Fanout | "research and prototype Y" | ≥2 parallel tasks (researcher + builder), Critic review, Director consolidation; sidebar shows the tree |
| 4.4 | D — Background | "monitor the threat feed nightly" | a scheduled/recurring task; published findings surface in the sidebar without a chat turn |
| 4.5 | Mid-task status vs final | during 4.2/4.3 | UI distinguishes "researching now…" pings from the final synthesis response |
| 4.6 | Cross-turn continuity | close the browser mid-task, reopen | sidebar still shows in-flight task; optional Windows toast on completion |

---

## 5. Done-gating + critic (Phase 6, live)

| # | Check | Steps | Expect |
|---|---|---|---|
| 5.1 | No-artifact approve is blocked | force a critic `approved` on a task whose artifact is missing | server auto-downgrades to `changes_requested` (`[AC_RESULTS_WIRING]`); task does NOT reach `done` |
| 5.2 | Revision round cap | drive an unsatisfiable task | after `VEILGUARD_MAX_REVISION_ROUNDS` (default 2) rejections it is **cancelled**, not looped forever (`[REVISION_ROUND_CAP]`) |
| 5.3 | Fresh-context critic | inspect the critic's dispatch prompt (logs) | contains spec + AC + artifact paths; NO producer trajectory / chain-of-thought |
| 5.4 | Turn cap | a dithering IC | caps at the per-dispatch turn limit, emits `max_iterations`, force-cancels with a distinct audit comment |
| 5.5 | Truncation marker | a tool result that gets truncated | result ends with `[TRUNCATED: N of M bytes]`; agent pages/chunks instead of reasoning over a prefix |

---

## 6. Approval gate (client-daemon, live — Windows)

| # | Check | Steps | Expect |
|---|---|---|---|
| 6.1 | Background shell → toast | a background/IC `run_command` on the user box | Windows toast appears with [Approve]/[Deny]/[Always allow]; NOT auto-executed |
| 6.2 | Foreground Director bypass | Director (user in chat) reads a file | ALLOW without toast (foreground privilege) |
| 6.3 | Deny path | deny the toast | tool returns an error; agent fails the task gracefully or raises `blocker_raised` |
| 6.4 | Timeout | ignore the toast > timeout (default 120s) | auto-DENY with reason `approval_timeout`; logged to `client_tool_approvals` |
| 6.5 | TOCTOU / arg binding | (if testable) approve `git status`, attempt to execute a different command | rejected — `approval_token` binds tool+arg_hash (§3.8.5) |
| 6.6 | CRLF arg bypass | a tool arg containing `\r\n; rm -rf ...` matching a bypass glob | DENY (control chars rejected before glob match — §11.5.4); verify in `services/sub-agents/tests/test_approval.py` too |
| 6.7 | Offline daemon fail-closed | stop the daemon, trigger a background tool | DENY + sidebar "daemon missing approval_gate" — never silent allow |
| 6.8 | Audit | after 6.1–6.4 | each decision is a row in `client_tool_approvals` with the canonical server timestamp |

---

## 7. Proactive stream + decision-ledger sidebar (Phase 3/4, live)

| # | Check | Steps | Expect |
|---|---|---|---|
| 7.1 | Proposal emission | let a dream cycle / `run_proposal_pass` run | rows appear in `task_proposals`; surface in sidebar "Proposed Tasks" |
| 7.2 | Approve → task | click [Approve] on a proposal | proposal `status=approved`, `resulting_task_id` set; new row in `agent_tasks`; idempotent on double-click |
| 7.3 | Defer / Shelve | click [Defer]/[Shelve] | status transitions; decay applies; recurrence escalates after 5 surfaces |
| 7.4 | Emergency lane | (if reproducible) a USER×USER contradiction | surfaces directly, visually emphasised, bypassing Director pre-eval |
| 7.5 | Per-tenant cap | observe cadence | proposals/day bounded by `proactive_cycles_per_day`; aggregate cost ≤ `cost_ceiling_per_tenant_per_day_usd` |
| 7.6 | SSE push | open the sidebar, mutate a task in another tab | tab updates live via `EventSource`; falls back to 10s polling if SSE drops |
| 7.7 | Work Items / Lessons tabs | open each tab | Work Items unions tasks+proposals+lessons; Lessons review queue reads from TCMM (post-M1); endpoints < 400ms warm |
| 7.8 | Auto-pause on drift | (if reproducible) flood proposals | `signal_quality_drift` alert + auto-pause when > 3× trailing-30d median |

---

## 8. Multi-tenant isolation (bet-the-product)

| # | Check | Steps | Expect |
|---|---|---|---|
| 8.1 | Cross-tenant proposals disjoint | two tenants with identical claim text | disjoint proposals; no leakage (§11.7.1) |
| 8.2 | Dream bridge_score doesn't cross namespaces | observe overlapping topics in two namespaces | no `bridge_score`/`concept_gravity` arc crosses the boundary (§11.5.5 — **the Phase-1 mandatory dream test; run it in the TCMM repo**) |
| 8.3 | Consultant memory per-tenant | invoke a consultant under tenant A then B | private memories are `(consultant_id, tenant_id)`-scoped; no bleed (§11.7.2) |

---

## 9. Sign-off

- [ ] All §2 chain-of-8 checks pass under the real tenant/user (hard gate — do not sign off otherwise).
- [ ] Patterns A–D each completed end-to-end in the live UI.
- [ ] Approval gate toast + deny + timeout + audit all observed on the real Windows box.
- [ ] Proactive proposal → approve → task observed in the sidebar.
- [ ] No cross-tenant leakage on the §8 spot-checks.

**Operator:** ____________  **Date:** ____________  **tenant_id/user_id used:** ____________

> Items §8.2 (dream bridge_score) and §6.6 (CRLF arg) are owned by the **TCMM** and **sub-agents**
> codebases respectively — verify them in those repos' own suites in addition to the live spot-check.
> They are listed here so the operator confirms the wired behavior, not just the unit behavior.
