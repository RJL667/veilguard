# Postgres cutover — UAT report (autonomous overnight loop, 2026-06-03)

Stack: local, on Postgres (`tcmm-pg` :5433). LLM via OAuth-bearer SSO (claude-haiku-4-5 / opus-4-7).
Endpoints: LibreChat :3080 · pii-proxy :4000 · agent-runtime :5000 · tcmm-service :8811 · admin :8820.

Legend: ✅ pass · ⚠️ partial/known-issue · ❌ fail · ⏳ pending

## Coverage checklist
- [x] Port completeness — no live Lance left (verified earlier sweep; re-confirm in loop)
- [x] Agentic smoke — Director direct turn on PG via /agent/query
- [ ] Agentic full flow — Director→Researcher→Critic→done on PG (multi-agent, ledger + memory + audit land in PG)
- [x] Perf — ingest/recall benchmarks + **fixed the 30s full-stack hang (connection exhaustion)**
- [ ] Endpoint coverage — proxy redaction (/), tcmm recall (/pre_request,/post_response,/ingest_turn), dashboard APIs, ledger ops, a2a
- [ ] Local Lance→PG data migration (optional)
- [ ] Commit + final summary

## Iteration log

### Iter 1 — agentic smoke on Postgres ✅
`POST :5000/agent/query` (agent_id=director, stream=false, "Reply with exactly: PONG").
- Director answered **PONG** live (model=claude-haiku-4-5, backend=live, 1 iteration, **no 429**).
- usage: tokens_input_total=9276 (new 374, cache_create=8902), output=120.
- `pii_audit` TO_LLM row landed in tcmm-pg ✅ (FROM_LLM follows on the 2s batch flush).
- Confirms runtime → LLM → audit → Postgres end-to-end.
- Note: use a non-`UID` shell var name (UID is readonly in bash) for the tenant id in later curls.

### Iter 2 — diagnosed + fixed the "30s on the full stack" ✅ (TCMM commit 1b67e3a)
ROOT CAUSE = **Postgres connection exhaustion**, NOT recall/ingest latency.
- Each PG provider (storage+vector+sparse+dream, ×N) opened + HELD an idle connection for life;
  the TCMM session pool accumulated ~90 idle conns → `max_connections=100` exceeded →
  `FATAL: sorry, too many clients already` → new conns failed/retried/hung → the 30s.
- Evidence: `pg_stat_activity` = 101 total, 95 idle; tcmm-service log full of "too many clients".
  Killing the pre-pool service dropped conns 101→10 (it held ~90).
- FIX: shared per-process `ThreadedConnectionPool` (transparent `_PooledConn`/`_PooledCursorCtx`
  proxy — borrows per `with cursor()` block, returns immediately; semaphore-gated so borrows block
  not raise). Bounded at `TCMM_PG_POOL_MAX` (default 16) per process. provider/recall/worker 3/3 green.
- AFTER: **12 concurrent cold sessions → ~2s, peak 11 connections, 0 errors** (was hang+errors).
- Benchmark (live full stack, fixed): recall `/pre_request` cold 0.36s, warm ~0.25s; ingest
  `/ingest_turn` ~0.21s/turn; agentic Director turn ~2.2s (LLM-bound).
