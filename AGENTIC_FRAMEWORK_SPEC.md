# Veilguard Agentic Framework — As-Built Specification

**Status:** As-built reference. Describes the system **as implemented in code**, derived from a full deep read of `agent-runtime/`, `agent/`, and `agents/` on **2026-06-08**.
**Companion docs:** [`MULTI_AGENT_PLATFORM.md`](MULTI_AGENT_PLATFORM.md) is the original *design + phase-plan + decision log* (forward-looking; dated 2026-05-21). This document is the *engineering reference* for what actually runs. TCMM has its own docs; this spec treats TCMM as an external dependency reached over HTTP.
**Source-of-truth note:** all edits are local-first (`C:\Users\rudol\Documents\veilguard\`), then deployed. `file:line` anchors are against the tree as read on the date above and will drift as code changes — treat them as starting points, not guarantees.

---

## 0. What this is

The agentic framework turns Veilguard from "single assistant + ephemeral tool calls" into a **small corporate-structured organization of agents**: one **Director** orchestrates standing ICs (**Researcher**, **Builder**, **critic-claim**, **critic-prose**), pulls in on-demand **consultants** (phishing-analyst, threat-analyst, report-writer), and optionally groups them into budgeted **teams** under a **team-lead**. Every cross-agent interaction collapses to one primitive — the **Task** — persisted in a **decision ledger**. Work arrives on two streams:

- **Reactive:** user → Director → Tasks → ICs → Critic → done.
- **Proactive:** TCMM dream signal → proposal → Director rank → user approval → Task → outcome → recalibration.

The framework lives in three places under `C:\Users\rudol\Documents\veilguard\`:

| Tree | Role |
|---|---|
| `agent-runtime/` | The FastAPI service (port 5000): HTTP API, the turn loop, the ledger, the background workers, the tools, the proactive stream. |
| `agent/` | The LLM **harness** — the `Agent` base class (`run_turn` pipeline), the persona-specific subclasses, the shadow tool, the preamble. Imported by both agent-runtime (background) and the pii-proxy ChatAgent path (reactive). |
| `agents/` | The **persona** definitions (`*.md` bold-KV files) + `CONSTITUTION.md` + `PROMPTS.md` (prompt drafts). |

**Where it sits in the platform:** LibreChat (3080) → pii-proxy (4000, redacts + routes) → **agent-runtime (5000)** when the request is Anthropic-bound and multi-agent-aware. agent-runtime calls TCMM (8811) over HTTP for memory render/recall/ingest, dispatches client tools to the sub-agents service (8809), and persists state to the shared Lance/Postgres DB at `/tcmm-data/veilguard/tcmm.db` (co-located with TCMM's archive and the `pii_audit` table).

---

## 1. The runtime service (`agent-runtime/app/`)

### 1.1 Service & boot

- **Framework:** FastAPI; ASGI app `app` built in `app/main.py:48-56`, served by uvicorn against `app.main:app`.
- **Port:** `config.PORT` = env `AGENT_RUNTIME_PORT`, default **5000** (`config.py:41`).
- **Routers mounted at import:** internal A2A (`app/a2a.py`) and external A2A (`app/a2a_external.py`, prefix `/external/a2a`) (`main.py:60-62`). `_register_external_tables()` runs at module load (`main.py:36`); `tool_dispatcher._build_registry()` runs at its import (`tool_dispatcher.py:91`).

**Startup sequence** — `@app.on_event("startup")` (`main.py:68-428`), in order:
1. `events.attach_main_loop(...)` so off-loop workers can publish to the `/events` SSE stream (`main.py:78-80`).
2. `config.validate()` — if it returns any error, **log and return early** (`main.py:82-88`): `/health` stays up to report the bad state, but **no personas, no constitution, and no background workers start**.
3. Open every ledger table (`main.py:96-110`) — forces Phase-6/7 schema migrations at boot rather than first-write.
4. `load_personas(config.AGENTS_DIR)` → `_persona_registry` (`main.py:114-124`).
5. `load_constitution(config.CONSTITUTION_PATH)` → `_constitution` (`main.py:126-143`).
6. Start the background workers (§7), each individually env-gated and try/except-wrapped (`main.py:145-428`).

**Shutdown** (`main.py:431-564`) stops the inbox poller and every worker cleanly.

### 1.2 HTTP API

Two auth schemes: **`X-Internal-Secret`** (must equal `config.VEILGUARD_INTERNAL_SECRET`; **bypassed if that env var is unset** — dev convenience) on the mutating internal routes; and JWT/mTLS/API-key on the public `/external/a2a/*` routes. Everything else has no route-level auth and is tenant-scoped by data filter only — the service is designed to sit behind the pii-proxy on a private network.

**Core**

| Method | Path | Auth | Purpose | Anchor |
|---|---|---|---|---|
| POST | `/agent/query` | — | **Main entry.** Body `{conversation_id, user_id, tenant_id, agent_id, messages, stream?=true, tools?, workspace_block?}`. SSE `text/event-stream` (default) or `{events:[…]}` if `stream:false`. Required fields → else 400; unknown `agent_id` → 404; registry unloaded → 503. | `main.py:2101-2224` |
| GET | `/health` | — | `{service, version, ready, personas_loaded, constitution_loaded, config_errors}`; 200 if ready else 503. | `main.py:780-801` |
| GET | `/agents` | — | Persona list for the sidebar. | `main.py:804-821` |
| GET | `/constitution` | — | Parsed constitution; 404 if unloaded. | `main.py:2060-2067` |

**Proposals (proactive stream)** — `GET /proposals`, `POST /proposals/emit`, `POST /proposals/{id}/convert` (one-click approve→Task; synthesizes default ACs), `POST /proposals/{id}/decision` (approve/shelve/defer; **approve is link-only**, expects a `resulting_task_id`), `GET /proposals/{id}` (`main.py:824-1163, 1320-1385, 1897-1928`).

**Tasks / teams / delegation / workspace** — `POST /delegate` (the **single sanctioned chat→org bridge**: requires `X-Internal-Secret`, **requires the Postgres ledger backend** else 503, and coerces `owner_hint` to a dispatchable IC so the task can't hang OPEN), `POST/GET /teams`, `GET /teams/{id}/cost`, `GET /workspace/file`, `GET /tasks/{id}` (+ `/ledger`, `/progress` Magentic-One sub-ledgers) (`main.py:716-777, 1166-1534, 1931-1995`).

**Lessons / work-items / events / governance** — `GET /lessons/review_queue` + `/amendment_candidates` + `POST /lessons/{id}/decision` (lessons now read from **TCMM archive**, not a Lance table); `GET /work_items` (unioned task+proposal+lesson feed); `GET /events` (per-connection SSE with 25 s heartbeat); `GET /replay/{task_id}`; `/documents/{snapshot,history,authority}`; `GET/POST /proactive_config`; `GET /apr/status` + `POST /apr/resume` (the APR circuit breaker); `GET /audit/tcmm_source_kinds` (AC-P7.6 provenance probe) (`main.py:1388-2098`).

**A2A internal** (`a2a.py`) — `GET /.well-known/agents[/{id}/agent-card.json]` (public), `POST /agents/{id}/messages` (secret; → a task comment), `GET /agents/{id}/{tasks/{tid},inbox}` (secret). **A2A external** (`a2a_external.py`, `/external/a2a/*`) — admin key issuance/revoke + message/task/inbox endpoints authed by JWT→mTLS→API-key with per-key allow-list and a 60 s sliding-window rate limit (429 + `Retry-After`).

### 1.3 The turn loop — `run_agent_query` (`runtime.py:267-815`)

An async generator that yields SSE-shaped event dicts. The control flow:

1. **Background detection + tenant context** — `is_background = is_subagent_cid(cid) or bool(parent_cid)`; the whole turn runs inside `tenant.set_tenant_context(...)` (contextvars; isolates concurrent requests and fan-outs).
2. **Backend resolution** — `BACKEND` env, default `live`; aliases `sdk`/`sso` → `live`. Only `live` and `scripted` are valid (`runtime.py:85-90`). `live` drives the `agent.Agent` harness directly; `scripted` uses canned turns for tests.
3. **System blocks** — in `live` mode TCMM does its own rendering inside `Agent.run_turn`, so the runtime passes `system_blocks=[]`; it always prepends the **OAuth-attribution magic block** `"You are a Claude agent, built on Anthropic's Claude Agent SDK."` (required so OAuth-bearer calls to non-Haiku models don't 429 on Anthropic's byte-exact attribution gate), then the persona system prompt.
4. **The LLM↔tool↔LLM loop** (`runtime.py:501-733`, capped at `_MAX_LOOP_ITERATIONS = 50` or a per-IC `max_turns`): call `live_agent.run_turn(...)`, tap the stream into a `TurnUsage` accumulator, forward events; extract `tool_use` blocks; for each, emit `tool_dispatch`, `await tool_dispatcher.dispatch(...)`, emit `tool_result`, append the tool-result block; **rebuild a self-contained plain-text user message every turn** (`LOOP_CONTEXT_FIX`) because the harness adapter takes a single user string, not a message array — without re-stating the brief + a 12-action log, an IC loses its task on turn 2. Stop on: no tool_uses, `stop_reason == "end_turn"`, `iteration >= max`, or a non-transient backend error (transient errors retry the same iteration twice).
5. **Audit + completion** — one FROM_LLM summary row via `audit.record_turn`; emit a `usage` event; feed APR token counters; if `task_id` is set, compute the turn's USD cost and `increment_cost_attributed` so team budgets stay live; emit `run_end`.

### 1.4 Middleware (`app/middleware/`) — composition helpers, not ASGI middleware

- **`tenant.py`** — the `TenantContext` frozen dataclass (`conversation_id, user_id, tenant_id, agent_id, parent_cid?, is_background, task_id?, team_id?`) in a `ContextVar`; `is_subagent_cid` / `parent_cid_from_sub` parse the `sub-<parent>-<kind>-<uuid>` cid shape.
- **`tcmm.py`** — `get_system_prefix` (scripted-mode only) returns TCMM blocks **unmodified** (TCMM owns cache_control). Memoization key `f"{parent_cid}:{agent_id}"`, TTL **300 s**. **Degraded mode**: any TCMM error returns `([], "")` so the agent runs context-free rather than 503-cascading. `observe_agent_output` ingests agent output into the `agent/{id}/observations/{user}` namespace and **returns True only if `added > 0`** (guards silent ingest drops).
- **`audit.py`** — `TurnUsage` with `tokens_input_total = new + cache_create + cache_read` (the spec-normalized total). `tap_sdk_stream` mutates usage from each message; `record_turn` writes one FROM_LLM row, `record_event` mirrors per-round TO_LLM/FROM_LLM rows. Same `pii_audit` table the proxy uses.
- **`cache_control.py`** — `normalize_cache_control` (≤4 markers, place one on the last block). **Dormant** — not called in the live path; both runtime and tcmm pass TCMM bytes through untouched. Standby defense only.

### 1.5 Tool dispatch & the trust boundary (`tool_dispatcher.py`)

`_build_registry()` indexes handlers from `ledger_mcp`, `memory_mcp`, and **`workspace_fs` (registered last, overriding any same-named client tool)** — handling both the SDK `.tool` decorator shape and a fallback `_mcp_tool_meta` shim (the SDK was removed 2026-05-25; the shim is the live path).

`dispatch(tool_name, tool_input)` has two paths — **this is the trust boundary**:
- **Path 1 — in-process (trusted):** ledger/memory/workspace-fs tools touch only the org's own Lance ledger and the mounted `/workspace`. Errors return as `isError` envelopes the LLM sees next turn.
- **Path 2 — remote (untrusted):** client tools (`file_write`, `run_command`, web/shell/fs) POST to the sub-agents service (`{SUB_AGENTS_URL}/api/agent_runtime/dispatch_tool`), which owns the per-user daemon WebSocket bridge **and the canonical approval gate**. agent-runtime runs **no** approval gate itself (the old SDK PreToolUse hooks are dead). **Graceful degradation** when sub-agents is down: read-only research tools return a *soft success* (`degraded=True`) so the LLM proceeds and marks claims `[unverified]`; mutation/host tools return `isError` with "raise a blocker, do not retry."

Name-mangling is normalized: `mcp__server__tool` prefixes are stripped; LibreChat's `<tool>_mcp_<server>` suffix form is **forced to Path 2** (it belongs to the user's interactive daemon, not the container filesystem).

### 1.6 Config & env (`config.py` + direct `os.environ` reads)

Centralized constants: `AGENT_RUNTIME_PORT`(5000), `LOG_LEVEL`(INFO), `ANTHROPIC_API_KEY`(""; live uses OAuth bearer), `TCMM_ENABLED`(True), `TCMM_URL`(`http://host.docker.internal:8811`), `TCMM_RENDER_TIMEOUT_S`(30), `SUB_AGENTS_URL`(`…:8809`), `VEILGUARD_INTERNAL_SECRET`(""→bypass), `AGENTS_DIR`(`/app/agents`), `LEDGER_DB_PATH`/`AUDIT_DB_PATH`(`/tcmm-data/veilguard/tcmm.db`), `CACHE_TTL_DEFAULT`("1h"), `APPROVAL_TIMEOUT_S`(120), `APPROVAL_FAIL_CLOSED`(True). `validate()` makes a missing API key fatal only for non-`live` backends.

Direct reads: `BACKEND`(live), `LEDGER_BACKEND`/`TCMM_STORAGE`(→postgres for the PG ledger), `VEILGUARD_TOOL_RESULT_LOG_CAP`(12000), `AUDIT_MAX_CONTENT_CHARS`(1_000_000), plus the worker cadence/disable vars (§7) and the inbox-poller lease/timeout/compaction vars (§6).

> **Dead/stale flags:** the legacy Claude-Agent-SDK functions in `runtime.py:100-148` are uncalled stubs; `middleware/cache_control.py` is dormant; `app/hooks/approval_gate.py` and `app/policy/client_tool_policy.py` **no longer exist** (the README still lists them); the `org_memory` Lance table is **retired** (lessons live in TCMM). See §13.

---

## 2. The agent harness (`agent/`)

`Agent(ABC)` (`base.py:185`) is the single LLM pipeline every actor inherits. It is instantiated **per turn** (cheap); all per-conversation state lives in TCMM and the PII session store, never on the instance. `adapter_cls` is injectable so tests swap a `ScriptedAdapter` while keeping render/redact/rehydrate real.

### 2.1 `run_turn` — the pipeline (`base.py:337-1013`)

One LLM call per invocation; the caller drives the external tool loop. `TurnContext(conversation_id, user_id, tenant_id, parent_cid, is_background, workspace_block)` is threaded through every step so redactor + TCMM + adapter key off the same ids.

- **Preamble** — mint `run_id`; build `sid = SessionId(tenant_id, conversation_id)`; emit `run_start`.
- **STEP 0 — user ingest:** *deferred to STEP 5* (firing it at turn start contended with the next render; TCMM serializes per session).
- **STEP 0.5 — `prepare_session`:** once per `(conversation_id, agent_id)`, **blocking**, pins the preamble + persona prompt + tool definitions into TCMM's **immutable tier** (idempotent; fast no-op if TCMM is down).
- **STEP 1 — TCMM render:** `render_structured(conv, user, task_query=raw_user_msg, pii_sid=sid.canonical())`. Result cached in a process-level `_RENDER_CACHE` keyed `(conversation_id, user_id, agent_id)` with a **90 s TTL**, invalidated after this turn's deferred ingest so the just-said turn surfaces next time. The `_MAGIC_PREFIX` block is guaranteed at index 0. **Cache-control ownership:** if TCMM provided *any* `cache_control`, the harness adds none of its own (mixing 1h-then-5m markers triggers an Anthropic 400). A Haiku-cache-floor inline-preamble fill runs when TCMM owns no cache and the prompt is under ~12 000 chars.
- **STEP 2 — PII redaction:** off-loop via `asyncio.to_thread`. `redact_memory_blocks(blocks, sid)` (keyed by live-block AID so cache-marked bytes stay stable) + `redact_messages(messages, sid)` (every turn). **Fail-closed:** a real Presidio failure → `redaction_unavailable` error event + `run_end(error)`; the request is refused, never shipped raw.
- **STEP 3 — adapter:** tools ship as Anthropic's **native `tools` field**, sourced from the immutable pins (deduped by name); the adapter is constructed with the redacted system blocks and sent **only the latest redacted user message** (prior turns live in TCMM). Optional true streaming (`VEILGUARD_STREAM=1`) rehydrates deltas through a 48-char boundary buffer.
- **STEP 3.5 — `intercept_response`:** strips the `tcmm_record_turn` shadow tool *before* rehydrate so its metadata never hits the redactor; captures `flag_obj`.
- **STEP 4 — rehydrate:** `rehydrate_blocks` / `rehydrate_text` restore REF_* tokens; emit `assistant`, the TO_LLM/FROM_LLM audit events (with normalized total input tokens), per-block `assistant_text`/`tool_call`, `final_result`, `usage`.
- **STEP 5 — deferred ingest:** a single ordered background coroutine ingests the user message then the assistant message (with `flag_obj`), then pops the render cache. Tracked in `_BACKGROUND_TASKS` so asyncio doesn't GC it.

### 2.2 Subclass knobs & the registry

Subclasses override **knobs, never the pipeline**: `tools()`, `model()` (default `persona.model_for("reactive")`), `include_memory()`, `preamble()`, `prepare_session()`, `prepare_tools()`, `intercept_response()`. `agent_for(persona)` (`registry.py`) maps persona-id (`critic-claim`→`CriticClaimAgent`, `critic-prose`→`CriticProseAgent`) then role (`director`→`DirectorAgent`, `ic`→`ICAgent`, `consultant`→`ConsultantAgent`, `chat`→`ChatAgent`), defaulting to `ICAgent`.

- **`DirectorAgent`** — overrides no knobs; adds `route`/`synthesize`/`propose` methods (currently thin stubs that emit replay anchors, the interface for a future Phase-8 split into Router/Synthesis personas) and per-method p95 latency telemetry. Uses its persona's mapped model (`reactive`/`rank_pass`/`synthesis`).
- **`ICAgent` / `ConsultantAgent`** — empty bodies; differ from each other only by persona file.
- **`CriticClaimAgent` / `CriticProseAgent`** — empty bodies + `_FRESH_CONTEXT = True` (the dispatch prompt carries only spec+AC+artifact; enforced by the inbox poller, see §5.6).
- **`ChatAgent`** — the LibreChat-facing path; synthesizes a minimal `PersonaSpec` from the request body, overrides `tools()` (client MCP schemas), `model()`, `include_memory()` (False for side-channel calls), pins the client `system` text as a distinct TCMM kind. **Side-channel detection** (`is_side_channel`): a last-user-message prefix matching title/summary requests forces `include_memory=False` so a 5-word title doesn't drag 20–70 KB of memory.

### 2.3 The shadow tool — `tcmm_record_turn` (`shadow_tool.py`)

Injected as the model's **last** tool, intercepted before the user sees the response. Required input fields: `knowledge_class` ∈ `{derived, novel, mixed}`; `used` (map of memory-block-id → relevance 0–1, drives heat); `epoch_complete` (bool); `emit_class` ∈ the **14-value `EPISODIC_CLASS_SET`** (`FACT, DECISION, INSIGHT, PROCEDURE, STATE, INTENT, DERIVED_FACT, ARTIFACT, AGENT_NOTE, CHATTER, ACK, QUERY, TRANSIENT_DATA, EXECUTION_LOG`) — this enum **must** match `core/episodic_ontology.py` or classification silently defers to the dead NLP path and `block_class` stays NULL. Optional `redaction_audit` (`over_redacted` REF tokens that are public, `missed` literal PII) feeds the PII allow/deny-list tuning loop (writes `pii_deny_list` for misses, `redaction_suggestions` for over-redaction candidates).

---

## 3. Personas (`agents/`)

### 3.1 Format, parse, registry (`agent/persona.py`)

A persona is a bold-KV Markdown file. Header lines match `**Key:** value`; the body after `## System Prompt` is the prompt. `**Model:**` is **required** (its absence means "not a persona" — that's how `CONSTITUTION.md`/`PROMPTS.md`/`README.md` are skipped, along with the `^[a-z][a-z0-9-]*\.md$` filename gate). `**Tools:**` supports `group(a, b), group2(c)` syntax → both `tool_groups` and a flat allow-list. `**Model:**` supports `role=model, role2=model2` → a `model_map`.

`PersonaSpec` (frozen) fields: `agent_id, role (director|ic|consultant), manager_id?, team_id?, model, model_map, tools (flat allow-list), tool_groups, system_prompt, display_name, source_path?, content_sha256, schema_version`. `model_for(role)` returns `model_map[role]` or the scalar. **`content_sha256`** (over canonical fields, excluding path/mtime) is the prompt-cache memoization key. `PersonaRegistry` is loaded once at startup — **no hot-reload** (restart to pick up edits); duplicate `agent_id`s: first wins.

**The flat `tools` list is the allow-list:** `Agent._default_tool_schemas` emits a schema only if its name is in `persona.tools`, so a tool absent from a persona's `**Tools:**` line is never advertised to that agent's LLM. (`ChatAgent` bypasses this — it returns client tools directly.)

### 3.2 Persona inventory

| agent_id | role | model (or map) | tool groups granted | mandate |
|---|---|---|---|---|
| **director** | director | reactive=`haiku-4-5`, rank_pass=`haiku-4-5`, synthesis=`sonnet-4-6` | orchestration (create_task, assign_task, consult, final_synthesis); proactive (list/convert/defer/shelve_proposal, list_lessons_for_review, rank_proposals, surface_org_memory_candidate); teams (create_team, list_teams, team_cost_report); strategy (read_constitution) | Tier-0 orchestrator on both streams; routes/judges, **never executes**; forms teams; enforces constitution + cost ceilings. No `observe`, no fs/shell/web. |
| **researcher** | ic | `sonnet-4-6` | web (web_search, web_fetch); filesystem (write_file, read_file); memory (observe); task (accept_task, add_comment, attach_output, submit_for_review) | Open-ended investigation, web fan-out, cross-checked cited deliverables. |
| **builder** | ic | `sonnet-4-6` | filesystem (read_file, write_file, edit_file); shell (run_command); memory (observe); task (…) | The **only** team agent with shell; every shell/fs-write is approval-gated. |
| **critic-claim** | ic→`CriticClaimAgent` | `haiku-4-5` | task (add_comment, review_decision); filesystem (read_file) | Fast inline **structural** arbiter of typed-claim promotion (enums, bitemporal, source_kind, citations); pass/fail <10 s. |
| **critic-prose** | ic→`CriticProseAgent` | `sonnet-4-6` | task (add_comment, review_decision); filesystem (read_file) | Async **semantic** PR-style reviewer of deliverables/blackboard; approve / changes_requested / declined, once. |
| **team-lead** | director | `sonnet-4-6` | task (create_task, assign_task, add_comment, get_task, inbox); memory (recall, observe); team (get_team, team_cost_attributed) | Mini-Director scoped to one team; routes within team, owns its review queue, escalates cross-team/budget/constitution up. |
| **phishing-analyst** | consultant | `sonnet-4-6` | filesystem; web (browse_url, google_search); code-exec | Consultant: triage suspicious email/URL/attachment; LOW–CRITICAL verdict + IOCs. |
| **threat-analyst** | consultant | `opus-4-6` | web; filesystem; code-exec (python, bash) | Consultant: infra/domain deep-dive, MITRE ATT&CK mapping. |
| **report-writer** | consultant | `sonnet-4-6` | filesystem; code-exec | Consultant: professional security reports from findings. |

The three consultants use the legacy bare format (no `**Role:**` etc.) so they auto-derive `role=consultant`, `manager=None`. Model IDs are documented verbatim from the persona files. `CONSTITUTION.md` (§11) and `PROMPTS.md` (prompt drafts, **not** loaded by code) live alongside.

---

## 4. The decision ledger (`agent-runtime/app/ledger/`)

One conceptual model: every entity is a row with a shared skeleton, persisted to Lance (or Postgres). `schemas.py` defines all tables; `TABLE_SCHEMAS` is the authoritative created/migrated set.

### 4.1 Shared skeleton (`schemas.py:40-53`)

Prepended to `agent_tasks`, `task_proposals`, `proposal_outcomes`, `agent_teams` (and the retired `org_memory`): `id, kind, status, parent_id?, lineage_chain[], tenant_id, user_id, created_by_agent_id?, created_ts, updated_ts, cost_attributed_usd?`. This common shape is what makes the unioned `/work_items` feed possible.

### 4.2 Tables

**`agent_tasks`** — the Task. Skeleton + `owner_id` (∈ `VALID_OWNER_IDS`), `assigner_id?`, `brief`, `deliverable_spec`, `inputs[]`, `outputs[]`, `due_ts?`, `trace_ref?` (TCMM cid), `comments_head_hash?` (tamper anchor), `origin` (foreground|background), `pattern` (A|B|C|D), `constitution_version?`, `lease_owner`/`lease_until` (sentinels `""`/`0.0` — Lance has no `IS NULL`), **`acceptance_criteria` list<struct> (non-null, may be `[]`)**, `team_id?`, `depends_on[]` (DAG), `extras_json` (holds `ac_results`, `constraint_violations`, the Magentic-One ledgers). The AC struct: `{id, statement, check_kind, check_args (JSON string), required, rationale?}`.

**`task_comments`** — append-only SHA-256 chain: `id, task_id, tenant_id, user_id, author_id, kind, body, ts, prev_hash?, self_hash, extras_json?`.

**`task_proposals`** — proactive candidates. Skeleton + `signal_type`, `signal_node_ids[]`, `impact_score`, `decay_score`, `objective_alignment?`, `constraint_violations[]`, `proposed_brief`, `proposed_assignee`, `proposed_deliverable_spec?`, `rationale?`, `recurrence_count`, `first/last_surfaced_ts`, `director_decision_ts?`, `shelf_reason?`, `resulting_task_id?`, `emergency_lane?`, `tcmm_obs_id?` (cross-ref, not a mirror).

**`proposal_outcomes`** — `proposal_id`, `resulting_task_id?`, `task_status`, `task_cost_usd`, `value_realized`, `regret_score`, `objective_deltas_json?`, `computed_at_ts`, `tcmm_obs_id?`.

**`client_tool_approvals`** (append-only audit) — `id, ts, ts_local?, tenant_id, user_id, agent_id, conversation_id?, parent_cid?, tool, args_sha256, args_preview?, origin, decision (allow|deny|approve|timeout|auto_foreground), reason?, latency_ms?, bypass_rule_id?, approval_token (TOCTOU), extras_json?`.

**`client_tool_bypass`** — user "always allow" rules: `id, user_id, agent_id?` (null=any), `tool, arg_glob, created_ts, expires_at?, created_via, active, extras_json?`. Deliberately in Lance, not TCMM ("security policy shouldn't blend with memory blocks").

**`tenant_proactive_config`** — per `(tenant,user)`: `proactive_stream_enabled`(T), `proactive_cycles_per_day`(12), `proactive_approval_cap_per_day`(20), `cost_ceiling_per_tenant_per_day_usd`(5.0), `paused`(F)+reason/at.

**`alignment_weights`** — per `(tenant, signal_type, objective_id)`: `weight`, `default_weight`, `last_regret_avg?`, `last_recalibrated_ts`, `recalibration_count`.

**`agent_teams`** — skeleton + `name`, `lead_agent_id`, `member_agent_ids[]`, `budget_usd`, `budget_cap?`, `cost_attributed_cached_usd?` (denormalized snapshot; source of truth is `sum(agent_tasks.cost_attributed_usd WHERE team_id=…)`).

**`agent_task_heartbeats`** — per `(task_id, worker_id)` beat: `last_beat_at`, `lease_ttl_s`.

**`org_memory` — RETIRED** (`[M1_CUTOVER_2026_05_28]`): not in `TABLE_SCHEMAS`, no table created; institutional lessons now live in TCMM `archive`, read via `memory.lessons_reader`. Schema retained for archaeology only.

### 4.3 The Task state machine (`tasks.py:37-45`)

```python
_TRANSITIONS = {
    "open":        {"accepted", "cancelled"},
    "accepted":    {"in_progress", "cancelled"},
    "in_progress": {"review", "blocked", "done", "cancelled"},
    "blocked":     {"in_progress", "cancelled"},
    "review":      {"done", "in_progress", "cancelled"},
    "done":        set(),       # terminal
    "cancelled":   set(),       # terminal
}
```

All transitions route through `update_status`, which: validates the transition (else `IllegalTransition`), applies the **hard-gate** only for `→ done`, writes status, broadcasts a `task_status_changed` SSE event, bumps APR, **emits a `status_change` chain comment**, and runs parent autoclose. `in_progress → review` (via `submit_for_review`) reassigns `owner_id` to the critic and resets the lease.

**The Phase-6.0.1 hard-gate** (`tasks.py:384`) — a `→ done` transition requires **all three**: (1) the latest `review_decision` comment verdict is `accepted`; (2) `extras.constraint_violations` is empty; (3) every `required` AC has `extras.ac_results[id] == "pass"` (an AC with no recorded result counts as **fail**).

**Coordinator autoclose** (`tasks.py:566-754`) — `director`/`team-lead`-owned coordinators own zero ACs and are never dispatched by the poller, so when all children are terminal: all-cancelled → drain (cancel coordinator); ≥1 done → re-run the coordinator's own mechanical ACs and close to `done` only if they genuinely pass. **Cancel cascade** (`tasks.py:877`) BFS-cancels a task and all lineage descendants.

### 4.4 The comment hash chain (`comments.py`)

Allowed kinds: `comment, status_change, review_request, review_decision, blocker_raised, blocker_cleared`. Each row's `self_hash = sha256(json{id, task_id, author_id, kind, body, ts(6dp), prev_hash})` with `sort_keys`; `prev_hash` = the prior row's `self_hash`; the task's `comments_head_hash` tracks the head. `verify_chain` walks oldest→newest recomputing hashes and checking links; any mutation of a prior comment breaks that row and every downstream link (plus a `HEAD_MISMATCH` sentinel if the head pointer is wrong). Append-then-update-head is non-atomic (Lance has no cross-table txns) but an orphaned comment is **detectable**, not silent.

### 4.5 Task CRUD highlights (`tasks.py`)

`VALID_OWNER_IDS` = the 9 personas; `_COORDINATOR_OWNERS` = `{director, team-lead}`; critics can never *own* a producing task (enforced in `create_task` — `[PROPOSAL_ASSIGNEE_NO_CRITIC]`). `create_task` inserts at `open`, enforces the **Phase-6.0.2 acceptance contract** (non-empty ACs incl. ≥1 required mechanical, unless `_phase_6_legacy_exempt`), validates `depends_on` (no self-loop, no unknown ids) and `team_id` budget. `synthesize_default_acceptance_criteria` (shared by the Director tool and `/proposals/convert`) gives coordinators `[]` and workers a single `output_path_exists` AC (path extracted from `deliverable_spec`, else a unique `team/drafts/deliverable-<hex8>.md`). `deps_satisfied`/`deps_dead`/`verify_depends_on_acyclic` drive the DAG; `increment_cost_attributed` rolls turn cost into the task (never raises — billing must not block the agent).

### 4.6 Teams & budget (`teams.py`)

`create_team` (budget ≥ 0, cap ≥ 1.0). `team_cost_attributed` sums live task costs (authoritative; writes back the cache). `budget_exceeded` tests `attributed + additional ≥ budget_usd × budget_cap`. A task opts in via `team_id`; `create_task` rejects if the team is missing/inactive/over-budget. The budget is a **soft envelope** that blocks at the boundary — it does not auto-shelve.

### 4.7 Store layer (`store.py` / `pg_store.py`)

Backend selected by `LEDGER_BACKEND`→`TCMM_STORAGE`: `postgres`/`postgresql` → `PgLedgerStore`, else LanceDB. The **Lance store never caches table handles** — it re-opens on every access so cross-process writes (sibling poller, `docker exec` injection) are visible; idempotent additive migrations add any missing nullable column (never drops). **`PgLedgerStore`** is a drop-in shim emulating the Lance fluent API over Postgres (scalars→typed columns, lists/structs→JSONB, `merge_insert`→`ON CONFLICT`, real `pyarrow.Table` from `to_arrow`), buying real transactions — no merge_insert clobber, no lease races. `migrate_ledger.py` is the one-shot Lance→PG copy.

---

## 5. Background dispatch — the inbox poller (`workers/inbox_poller.py`)

One poller per process, polling **all tenants**. Interval **`POLL_INTERVAL_S = 5.0`** (hardcoded, *not* env-tunable despite the docstring). Each cycle: one-shot startup orphan sweep → stale-heartbeat sweep → ledger compaction (≤ every 300 s) → `_poll_once`.

- **Lease claim (CAS):** candidate scan is `owner_id IN (ELIGIBLE_OWNERS) AND status IN ('open','review','in_progress') AND lease_until < now` (limit 20). Claim = conditional `tbl.update(... lease_owner=worker_id, lease_until=now+LEASE)` then **re-read to confirm** (Lance gives no uniform affected-row count — "update-then-re-read, slightly racy, sufficient for low contention").
- **`ELIGIBLE_OWNERS`** = `{researcher, builder, critic-claim, critic-prose}` — Director is reactive-only; consultants are pull-only.
- **Concurrency caps (hardcoded):** researcher 8, builder 6, critic-* 4, consultants 2 each; total **28** = the global semaphore. Per-persona caps enforced independently of the semaphore (a saturated persona can't starve another).
- **Lease vs timeout:** `VEILGUARD_LEASE_DURATION_S`(300) and `VEILGUARD_DISPATCH_TIMEOUT_S`(270), with a hard invariant `timeout < lease` (clamped) so the timeout fires before the lease frees the row → no double dispatch.
- **Cancel paths** (each writes a distinct `blocker_raised` comment, re-runs parent autoclose): wall-clock timeout (`dispatch_timeout`, but **re-reads status first** so a late timeout on a now-`review` task is a no-op), turn-cap (`_CRITIC_MAX_TURNS=10` / `_IC_MAX_TURNS=25`, hardcoded), generic exception.
- **Guards:** startup orphan sweep (cancel `in_progress`/`review` left by a dead process); heartbeat sweep (force-cancel tasks whose freshest beat is older than `lease_ttl_s`); **blocker retry cap** (`BLOCKER_RETRY_CAP`=2 → terminal cancel to stop refusal loops); dependency-aware claim (`inputs` task-refs + `depends_on` must be `done`; a cancelled dep cascade-cancels the dependent); the **APR circuit breaker** (skip all new dispatch when `apr_should_pause_dispatch()` until operator `apr_resume()`).
- **Auto-compaction** (`AGENT_LEDGER_AUTO_COMPACT`=1, every `AGENT_LEDGER_COMPACT_INTERVAL_S`=300, in a worker thread): `tbl.optimize(cleanup_older_than=0)` over heartbeats/tasks/proposals/comments — heartbeats append one fragment per write; unchecked, a scan blocks the loop 2–3 s and inflates chat latency.
- **Dispatch internals:** three user-message shapes by status (`open` fresh assignment; `review` critic dispatch via `acceptance.critic_prompt.build_critic_user_message` — fresh-context; `in_progress` iteration inlining the latest `review_decision`); each turn boundary writes a heartbeat and streams `task_stream_*` SSE; sub-cid `sub-<parent7>-<kind3>-<uuid8>`.

---

## 6. Agent-facing tools (`app/tools/`)

The LLM sees a tool only if its name is in the persona allow-list (§3.1). Tenant scope comes from contextvars, never args. All three modules + `__init__` are the entire `tools/` surface.

**`ledger_mcp.py`** (server `veilguard_ledger`): `create_task`, `assign_task`, `convert_proposal`, `shelve_proposal`, `defer_proposal`, `list_proposals`, `list_lessons_for_review` (Director/critic-prose); `accept_task`, `add_comment` (kind ∈ comment/blocker_raised/blocker_cleared), `attach_output` (**stat-before-attach** via a daemon read probe), `submit_for_review` (ICs); `review_decision` (critics — runs mechanical ACs server-side, see §7); `get_task`, `inbox`, `create_team`/`list_teams`/`team_cost_report`. (Persona-named tools `consult`, `final_synthesis`, `rank_proposals`, `surface_org_memory_candidate`, `get_team`, `team_cost_attributed` are orchestration-level, wired elsewhere.)

**`memory_mcp.py`** (server `veilguard_memory`): `recall` (HTTP to TCMM `/recall`, scope auto/conv/agent/team_knowledge/blackboard), `observe` (writes the agent's private TCMM memory; **provenance hard-coded by the wrapper** — the agent can never set `source_kind`; non-persist returns an explicit "do NOT retry" error), `read_constitution` (returns parsed governance, not a file read).

**`workspace_fs.py`** — `read_file`/`write_file`/`edit_file`/`list_directory` running **in-process, server-side**, registered in Path 1 so they **override** the client-daemon versions (the fix for critics looping on `TOOL UNAVAILABLE` when the daemon was offline). Every path is sandboxed under `VEILGUARD_WORKSPACE_ROOT` (`/workspace`); any `..` escape is refused.

---

## 7. Acceptance criteria & critics (`app/acceptance/`)

**Check kinds** (`executors.py:61-71`, 9 total): `claim_count, claim_predicate, output_path_exists, output_path_matches_regex, output_path_jsonschema, test_passes, word_count_range, manual_user, llm_verify`. **`MECHANICAL_CHECK_KINDS`** = those 7 minus `manual_user` and `llm_verify`.

**The iron rule:** every task's required ACs must include ≥1 mechanical kind (enforced at `create_task`); `manual_user` can only be advisory; an `llm_verify` AC must be **paired** with ≥1 mechanical required AC — the LLM judge may *fail* what mechanics passed, never *clear* what mechanics didn't.

**3-state `CheckResult`:** `pass` (meets it), `fail` (doesn't), `error` (the check itself couldn't run — cmd missing, file unreadable, timeout, malformed AC). Both `fail` and `error` block the gate; `evidence` carries sha256/counts/exit-codes for re-verification (catches a builder partial-write-then-retry under the same path). Notable executors: `output_path_exists` (exists AND ≥ min_bytes; empty file → fail even if a regex would match), `test_passes` (the only side-effecting one — sandboxed under repo root, exit 127 / "not recognized" → `error`), `word_count_range` (deterministic — replaced a non-deterministic LLM word count), `manual_user` (never auto-passes; returns `error` + `awaiting_user=True` so the gate blocks until a human answers).

**How critics run them — fresh-context discipline:** the critic has **no executor tool**. `critic_prompt.build_critic_user_message` restricts the dispatch to `(task_id, brief, deliverable_spec, acceptance_criteria, outputs, inputs)` — producer chain-of-thought/trajectory are structurally excluded (`FORBIDDEN_PRODUCER_FIELDS`, AC-22). The mechanical checks run **server-side inside `review_decision`** (`ledger_mcp._run_required_ac_checks`), which persists verdicts to `extras.ac_results` (the missing link that lets the hard-gate pass), **auto-downgrades** an `approved` verdict to `changes_requested` if a required mechanical AC fails (no rubber-stamping), and cancels an unsatisfiable IC↔critic loop after `VEILGUARD_MAX_REVISION_ROUNDS`(2). Decision→status: `{approved:done, changes_requested:in_progress, declined:cancelled}`.

**`llm_verify`** (`llm_judge.py`): default model **`claude-haiku-4-5`**, `temperature=0.0`, rubric + artifact in isolated XML tags, "be conservative — when in doubt, fail." Artifact capped at 50 KB. A broken/unimportable judge → `error` or a conservative `fail` (never a spurious pass).

---

## 8. The proactive stream (dream-as-scheduler, `app/proposals/`)

### 8.1 Lifecycle

TCMM dream cycle writes signal-bearing `dream_archive` rows → `SignalEmitterWorker` (daily) also injects two synthetic signals → **`DreamScanner`** (hourly, with importance/`fire_now` early-fire) reads them: backpressure gate (skip the whole cycle if ≥25 pending), tenant gate (skip if proactive stream off), cursor + node-id dedup, score, drop `final ≤ 0`, sort, cap (10/cycle, 3/signal-type), ranks 1–3 get deterministic template briefs, ranks 4–10 get one Haiku rank pass (opt-in) → emit via the split-writer (ledger row + TCMM cross-ref) at `status=pending`. The user approves (→ Task) / defers (decays) / shelves; `LifecycleWorker` (hourly) decays deferred ×0.9, auto-shelves below floor 0.05, expires at 7 d TTL, flags recurrence ≥5. On task completion (≥30 d), `OutcomesWorker` writes `regret_score`; `RecalibrationWorker` (weekly) nudges `alignment_weights`, feeding back into future scoring.

### 8.2 Signal taxonomy

**5 confirmed** (dream-emitted) + **2 deferred** (agent-runtime-synthesized) + 1 synthesized governance signal. All current generative signals default to the **researcher** (critics are coerced away — they can't own producing work; builder is reserved for code/shell):

| signal | impact formula | default alignment seed | special |
|---|---|---|---|
| `information_gap` | gap_breadth × downstream_pressure | toil .5 / sec .3 / agency .2 | → research note |
| `contradiction_arc` | source_severity × claim_centrality | sec .5 / agency .4 / toil .1 | **USER×USER → emergency lane** (severity 10.0; bypasses caps + pre-eval) |
| `reflective_heuristic` | recurrence × success_rate × token_savings | toil .7 / sec .2 / agency .1 | also a lesson-promotion source |
| `recurring_ritual` | (= reflective_heuristic) | toil .8 / sec .1 / agency .1 | → skill def |
| `stance_arc` | polarity_distance × claim_stake | agency .5 / sec .4 / toil .1 | multi-reviewer committee |
| `low_stability_cluster` *(deferred)* | failure_count × recall_frequency | sec .6 / agency .3 / toil .1 | emitted from archive density |
| `stale_supersession_chain` *(deferred)* | age_days × recall_count × topic_currency | sec .5 / toil .4 / agency .1 | emitted from archive age |
| `constitution_amendment` *(synthesized)* | fixed 10.0 | — | bypasses scoring; user-applied edit only |

### 8.3 Scoring (`scoring.py`)

```
final_score = signal_impact × objective_alignment × constraint_gate
  signal_impact      ∈ [0,∞)   per-signal formula (above)
  objective_alignment∈ [0,1]   dot(signal seed vector, constitution objective weights)
  constraint_gate    ∈ {0,1}   0 if any constitution constraint violated, else 1
```

**Multiplicative** — any weak/zero factor zeroes the candidate; a missing payload field yields 0 (explicit), not a crash. Caps: 10 candidates/cycle, 3/signal-type, deterministic top-3, pending backpressure 25 (`VEILGUARD_DREAM_MAX_PENDING`), daily approval cap 20 (enforced at approval time), $5/tenant/day ceiling.

### 8.4 Workers (cadence · default · disable flag)

| Worker | env interval | default | disable flag | job |
|---|---|---|---|---|
| **DreamScanner** | `VEILGUARD_DREAM_SCANNER_INTERVAL_S` | 1 h | `…_DREAM_SCANNER_DISABLED` | scan dream_archive → emit proposals (rank pass opt-in via `…_PROPOSAL_RANK_PASS_ENABLED`=0) |
| **LifecycleWorker** | `…_PROPOSAL_LIFECYCLE_INTERVAL_S` | 1 h | *(none — always on)* | decay ×0.9 / shelf < 0.05 / expire 7 d / flag recurrence ≥5 |
| **SignalEmitterWorker** | `…_SIGNAL_EMITTERS_INTERVAL_S` | 24 h | `…_SIGNAL_EMITTERS_DISABLED` | inject low_stability (density < 0.30, ≥5 claims) + stale_chain (age > 60 d, ≥3 claims) |
| **OutcomesWorker** | `…_OUTCOMES_INTERVAL_S` | 24 h | `…_OUTCOMES_WORKER_DISABLED` | terminal-≥30 d proposals → `regret = cost / max(value, 0.1)` |
| **RecalibrationWorker** | `…_RECAL_INTERVAL_S` | 7 d | `…_RECAL_WORKER_DISABLED` | 28 d regret window, ≥5 outcomes, nudge ±0.05 (clamp ±0.10), floor 0.05 |
| **DriftWatchdog** | `…_DRIFT_INTERVAL_S` | 24 h | `…_DRIFT_WATCHDOG_DISABLED` | today > 3× trailing-30 d median (median ≥1) → auto-pause stream |
| **LessonPromotionWorker** | `…_LESSONS_INTERVAL_S` | 24 h | `…_LESSONS_WORKER_DISABLED` | dream heuristic/ritual → TCMM team_knowledge (conf ≥.50, reinforce ≥2, ≥2 agents) |
| **ConstitutionAmendmentWorker** | `…_CONSTITUTION_AMENDMENT_INTERVAL_S` | 7 d | `…_CONSTITUTION_WORKER_DISABLED` | lessons (conf ≥.75, reinforce ≥5) → amendment proposal (user-applied only) |
| **StallDetectorWorker** | `…_STALL_DETECTOR_INTERVAL_S` | 5 min | `…_STALL_DETECTOR_DISABLED` | lease-expired / heartbeat-silent (15 m) / progress-frozen (30 m) → `task_stalled` comment |

### 8.5 Rank pass & constitution bridge

**`rank.py`** — one Haiku structured-output call per cycle/tenant over ranks 4–10 (≤7 candidates), ~\$0.001/cycle, forced through a `rank_proposals` tool (validate-before-return). Output per candidate: `refined_brief/assignee/objective_alignment/rationale/drop`; `drop=True` defers (not discards). Robust: empty→no call, adapter error→passthrough, invalid assignee→default (critics re-coerced to researcher). **`constitution_bridge.py`** — `objectives_to_dict` flattens objectives for the dot product; `evaluate_constraints` runs wired evaluators (`cost_ceiling_per_task` $5 hardcoded, `no_hidden_automation` (proposal-time no-op), `preserve_provenance`); permissive-by-default (unevaluated constraint ≠ violation), but an evaluator that *raises* is conservatively a violation.

---

## 9. Approval gate (canonical, in `services/sub-agents/`)

agent-runtime runs no gate itself; Path-2 dispatch hits the gate at `services/sub-agents/core/approval.py`.

- **Capability matrix** (`policy.py`): `Decision ∈ {ALLOW, APPROVE, DENY}`. Exec tools → APPROVE (dangerous-pattern match → DENY); write tools → APPROVE (protected path → DENY); read-only → ALLOW (secret path → APPROVE); other client tool → APPROVE (conservative); non-client (cloud/ledger/memory) → ALLOW.
- **Levels** (`client_settings.py`): `auto | confirm | strict`, default **confirm**, default timeout **60 s** (conv override → user default → default). `auto` collapses APPROVE→silent; `confirm` toasts on APPROVE; `strict` toasts on APPROVE and ALLOW.
- **`gate()`** is **fail-closed**: `system:sandbox` is pre-approved; an unimportable/raising policy forces a toast even at `auto`; a needed toast with no daemon bridge → deny; timeout → `timeout`/deny; "always for this conv" writes an `auto`-level conv override. Every decision is audited (`direction="APPROVAL"`) and a shadow `AWAITING_USER` Task row is opened/closed.
- **TOCTOU `approval_token`** binds the grant to `args_sha256` in the append-only `client_tool_approvals`; subsequent dispatches echo it and the gate re-verifies it isn't revoked. `client_tool_bypass` holds user "always allow" rules.

---

## 10. A2A transport (`app/a2a.py`)

Internal inter-agent transport. `X-Internal-Secret` (bypassed if unset) on the message POST + task GETs; the two discovery GETs are public. **AgentCard** is the Google A2A draft shape (`schemaVersion, name, url, version, authentication, capabilities, skills` derived 1:1 from `persona.tools`) plus Veilguard extensions (`agent_id, role, manager_id, team_id, model`). `POST /agents/{id}/messages` maps a message to a durable `task_comments` row. (`a2a_external.py` adds the OAuth/mTLS/API-key public surface with rate limiting.)

---

## 11. Memory write discipline & governance

**Write discipline** (`app/memory/`): every memory mutation must flow through a typed writer; a lint test (AC-36) rejects any other module importing Lance/TCMM/workspace-fs directly, and `WRITER_DESTINATIONS` regenerates `docs/MEMORY_WRITE_PATHS.md` (AC-37) so docs can't drift. Phase-6.8 writers: `record_episode`, `record_heartbeat`, `promote_to_semantic`, `log_decision` (the one site touching both `agent_tasks` and `task_comments`), `enqueue_dream_input`, `record_approval`, `attach_artifact` (+ deprecated `update_org_memory`). Phase-7 **split-writers** (the only sanctioned dual-write sites — ledger row + TCMM observation cross-ref): `record_outcome_with_narrative`, `record_proposal_with_content`, `promote_lesson_to_team_knowledge`, `record_discussion_comment`.

**Constitution** (`agents/CONSTITUTION.md`, loaded by `app/constitution/loader.py`): user-authored, ~10 entries. **Objectives** (weighted, sum 1.0): `reduce_toil` .40, `improve_security` .30, `preserve_user_agency` .30. **Constraints** (boolean vetoes): `no_hidden_automation` (Tasks > $0.50 surface first), `cost_ceiling_per_task` ($5), `preserve_provenance`, `no_autonomous_client_daemon_access`. **Metrics:** time_saved, knowledge_reuse, regret. Amendment is always user-gated (evidence accrues in lessons → eligible at confidence ≥0.75 + reinforcement ≥5 → surfaced as a proposal → user approves → `constitution_version` bumps). Read by the Director at startup; steers every proposal's `objective_alignment` + `constraint_gate`.

**APR (Artifact Production Ratio, Phase 6.7):** state mutations (task create, status change, state-machine comments) are counted against narration; a sustained drop trips a circuit breaker that pauses the inbox poller's new dispatch until operator `/apr/resume`. The governing principle: *Tasks transform state; conversations are scaffolding.*

---

## 12. Cost model

Per-turn cost is derived from `pii_audit` token columns × a per-model rate card (`outcomes.py`, USD/M tokens): Opus 5/75/0.50/10 (output a placeholder, unconfirmed), Sonnet-4-x 3/15/0.30/6, Haiku-4-5 0.80/4/0.08/1.60 (in/out/cache-read/cache-write); unknown model → Sonnet rates. Token accounting normalizes to **total input** (`input + cache_creation + cache_read`); cache hit rate = `cache_read / total`. Cost rolls from the turn into the task (`increment_cost_attributed`) into the team rollup.

---

## 13. Implementation status & known drift

**Built (Phases 0–7.5 largely landed):** the agent-runtime service + harness; the 9 personas + constitution loader; the full decision ledger (Lance + Postgres shim) with the state machine, hard-gate, and SHA-chained comments; the inbox poller with leases/caps/timeouts/sweeps/compaction; the acceptance-criteria system + fresh-context critics + server-side AC execution + llm_verify; the full proactive stream (scanner + 9 workers + scoring + rank pass + constitution bridge/amendments); teams + budgets; A2A (internal + external); the approval gate + TOCTOU + bypass; memory write discipline + split-writers; APR breaker; Magentic-One task/progress ledgers.

**Deferred / partial:** `value_realized` is a v1 proxy (any reference counts; agent-id-scoped recall is v2); recalibration spreads one mean-regret across all objectives (per-objective deltas are v2); the Director `route/synthesize/propose` methods are interface stubs for a future Phase-8 split; `constitution_bridge` thresholds are partly hardcoded ($5).

**Drift to fix (doc-vs-code):**
- `agent-runtime/README.md` is stale — lists `app/hooks/approval_gate.py` and `app/policy/client_tool_policy.py` (deleted), "8 personas", "BACKEND=sdk default", and SDK troubleshooting that no longer applies.
- The inbox-poller docstring claims the poll interval is env-tunable; `POLL_INTERVAL_S = 5.0` is hardcoded. `PERSONA_CAPS`, turn caps (10/25), and `ELIGIBLE_OWNERS` are all hardcoded.
- The legacy SDK functions in `runtime.py:100-148` and `middleware/cache_control.py` are dead/dormant; `org_memory` is retired (some docstrings still reference it).
- `MULTI_AGENT_PLATFORM.md` is the **design/phase-plan**, written forward-looking ("what Phase X ships") and dated 2026-05-21 — this as-built spec supersedes its status claims (its decision log and rationale remain valuable history).

---

## 14. File index (where to look)

| Concern | Path |
|---|---|
| Service entry, routes, startup, workers wiring | `agent-runtime/app/main.py` |
| Turn loop | `agent-runtime/app/runtime.py` |
| Config/env | `agent-runtime/app/config.py` |
| Middleware | `agent-runtime/app/middleware/{tenant,tcmm,audit,cache_control}.py` |
| Tool dispatch + trust boundary | `agent-runtime/app/tool_dispatcher.py` |
| The harness pipeline | `agent/base.py` |
| Persona classes / registry | `agent/{director,ic,chat,critic,consultant,registry}.py` |
| Persona loader / spec | `agent/persona.py` (≡ `agent-runtime/app/personas/loader.py`) |
| Shadow tool | `agent/shadow_tool.py` |
| Persona definitions | `agents/*.md`, `agents/CONSTITUTION.md`, `agents/PROMPTS.md` |
| Ledger schemas / Task / comments / proposals / teams / store | `agent-runtime/app/ledger/{schemas,tasks,comments,proposals,teams,store,pg_store}.py` |
| Background dispatch | `agent-runtime/app/workers/inbox_poller.py` |
| Agent tools | `agent-runtime/app/tools/{ledger_mcp,memory_mcp,workspace_fs}.py` |
| Acceptance criteria / critics / judge | `agent-runtime/app/acceptance/{executors,critic_prompt,llm_judge}.py` |
| Proactive stream | `agent-runtime/app/proposals/*.py` |
| Memory writers | `agent-runtime/app/memory/{writers,phase_7_writers,lessons_reader}.py` |
| A2A | `agent-runtime/app/a2a.py`, `a2a_external.py`, `a2a_auth.py` |
| Approval gate (canonical) | `services/sub-agents/core/{approval,policy,client_settings}.py`, `utils/{safety,tool_location}.py` |
| Constitution loader | `agent-runtime/app/constitution/loader.py` |
| Design + decision log (origin doc) | `MULTI_AGENT_PLATFORM.md` |

---

*Generated from a deep code analysis on 2026-06-08. When code and this spec disagree, the code wins — fix the spec.*
