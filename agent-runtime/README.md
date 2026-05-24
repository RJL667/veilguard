# agent-runtime

Veilguard's embedded Claude Agent SDK runtime. Owns the LLM-tool-LLM loop
for Anthropic-bound conversations; multi-tenant + provenance-aware via
middleware around the SDK.

**This README is the wake-up runbook.** If you're reading this fresh,
go through the sections in order — verify, then deploy.

---

## What's in here

```
agent-runtime/
├── Dockerfile
├── requirements.txt
├── pytest.ini
├── README.md                        ← this file
├── app/
│   ├── main.py                      ← FastAPI entry; starts inbox poller on startup
│   ├── config.py                    ← env-var consts; validate() called on startup
│   ├── runtime.py                   ← composes middleware around SDK query()
│   ├── middleware/
│   │   ├── tenant.py                ← per-request contextvars (cid, user_id, etc.)
│   │   ├── tcmm.py                  ← TCMM render middleware + memoization
│   │   ├── cache_control.py         ← SDK-bug defense for cache_control
│   │   └── audit.py                 ← wraps SDK stream, writes pii_audit rows
│   ├── hooks/
│   │   └── approval_gate.py         ← PreToolUse hook implementing §3.8
│   ├── policy/
│   │   └── client_tool_policy.py    ← capability matrix + CRLF rejection
│   ├── personas/
│   │   └── loader.py                ← bold-Markdown KV → AgentDefinition
│   ├── constitution/
│   │   ├── loader.py                ← parses agents/CONSTITUTION.md
│   │   └── scorer.py                ← final_score = impact × align × gate
│   ├── ledger/
│   │   ├── schemas.py               ← PyArrow schemas for 7 new Lance tables
│   │   ├── store.py                 ← lazy lancedb connection + tenant filters
│   │   ├── tasks.py                 ← agent_tasks CRUD + cascade cancel
│   │   ├── comments.py              ← append-only chain with prev_hash
│   │   └── proposals.py             ← task_proposals queue + decay/expiry
│   ├── tools/                        ← Phase 2 — MCP tool surface for agents
│   │   ├── ledger_mcp.py             ← 10 ledger tools (create_task, etc.)
│   │   └── memory_mcp.py             ← recall / observe / read_constitution
│   ├── workers/                      ← Phase 2 — background loops
│   │   └── inbox_poller.py           ← IC inbox dispatcher
│   └── utils/
│       └── text_sanitize.py         ← two-mode CRLF + cp1252 rescue
├── scripts/
│   └── spike_cache_validation.py    ← LIVE-API spike (user runs once)
└── tests/                            ← 150 unit + integration tests
```

---

## Step 0 — what was built and what's still on you

**Done (no human-in-the-loop needed):**
- All harness modules
- All 7 decision-ledger Lance schemas + CRUD
- 5 new agent personas (director, researcher, builder, critic-claim,
  critic-prose) in `Documents/veilguard/agents/`
- Docker-compose entry for the new service
- 121 unit + integration tests (all passing locally on Windows)

**Still on you (after Phase 2 ship 2026-05-22):**
1. Validate Phase 0.1 against the real Anthropic API (one-shot spike script)
2. Build the container + deploy to VM
3. Set `AGENT_RUNTIME_ENABLED=true` in `.env` to flip on pii-proxy routing
   (Phase 2 already wrote the forwarding code; just toggle on)
4. Build + ship client-daemon v0.3.0 (Windows toast code is written;
   needs `python -m PyInstaller` + Inno Setup + `publish_release.py`
   per `workflow_daemon_release` memory)

**Done in Phase 2 ship (no longer on your list):**
- 10 ledger MCP tools + 3 memory MCP tools exposed via in-process
  `create_sdk_mcp_server` — agents can now actually call
  `create_task` / `submit_for_review` / `recall` / `observe` / etc.
- IC inbox polling loop (`app/workers/inbox_poller.py`) — runs on
  startup, picks up `status=open` Tasks for researcher/builder/critic-*,
  dispatches via the runtime
- Windows toast handler in client-daemon for `request_approval` WS
  method (winotify + sqlite WAL queue, v0.3.0)
- pii-proxy routing patch — Anthropic-bound calls forward to
  agent-runtime when enabled; graceful 503 fallback if agent-runtime
  is down
- Capability handshake (daemon advertises `approval_gate: true|false`
  so agent-runtime can fail-closed on older daemons)
- 29 new tests covering MCP tool dispatch, inbox poller lifecycle,
  memory tools (150 total, all passing)

---

## Step 1 — verify locally (5 minutes)

Install deps and run the unit + integration tests. This is the safety
net that catches regressions before deploy.

```bash
cd C:\Users\rudol\Documents\veilguard\agent-runtime
pip install -r requirements.txt
pytest tests/
```

Expected output: `151 passed`.

If something fails: the test names tell you which module is broken. Don't
deploy until tests are green.

---

## Step 1.5 — run the demos (no API credit needed)

Four end-to-end demo scenarios exercise the orchestration pipeline with
a scripted backend (canned LLM responses; no Anthropic credit). These
prove that real tools fire, the real Lance store writes, the real inbox
poller dispatches, and the comment chain accumulates correctly.

```bash
python -m demo.run_all
```

Expected summary at the end:

```
# Running demo.scenario_pattern_a_solo
  [PASS] solo Director path works; no spurious tasks created.
# Running demo.scenario_pattern_b_delegation
  [PASS] Director delegated; ledger has the task; inbox poller picked it up.
# Running demo.scenario_pattern_c_fanout
  [PASS] Both ICs ran in parallel; both reached review.
# Running demo.scenario_critic_iterate
  [PASS] Critic iterate flow worked end-to-end.
  Overall: ALL PASS
```

Run any single scenario for detailed output:

```bash
python -m demo.scenario_pattern_b_delegation
```

Each scenario prints the event stream + the final ledger state including
the comment chain. The Critic iterate scenario shows ~10 chain entries
covering: 6 status_changes (open -> accepted -> in_progress -> review ->
in_progress -> review -> done) + 2 review_decisions (changes_requested +
approved). Real proof the orchestration works on the actual Lance store.

**The demos use ScriptedBackend.** When `ANTHROPIC_API_KEY` is set and
`BACKEND=sdk` (default), production runs use the real SDK against the
Anthropic API. The demos set `BACKEND=scripted` to drive the same
runtime pipeline with predetermined LLM responses. The SDK is the only
piece you can't exercise locally without credit; everything else
(ledger, inbox poller, hooks, audit, MCP tools) runs for real.

---

## Step 1.6 — live SSO demo (uses your Claude Max subscription, no API charges)

The four scripted demos prove the orchestration works. The fifth demo
proves an actual LLM can drive the pipeline using your Claude Code SSO
token (no Anthropic API credit needed).

```bash
# Must be run from your NORMAL Windows terminal (not a sandboxed shell)
# so the Python subprocess can reach Claude's stored OAuth token.
cd C:\Users\rudol\Documents\veilguard\agent-runtime
python -m demo.scenario_live_sso_tool
```

What it does:
- Sets `BACKEND=sso` so agent-runtime uses the SsoBackend
- SsoBackend subprocesses `claude -p --output-format stream-json --verbose ...`
- Director persona is asked to "list contents of agent-runtime/app/ using Bash"
- Real Claude (Sonnet/Opus, via your subscription) decides to use Bash
- Claude CLI executes the Bash command on your Windows machine
- Output flows back through agent-runtime's audit + event stream

Expected success output:
```
[PASS] real LLM invoked a real tool via Claude SSO.
       the full agent-runtime pipeline (TCMM disabled, ledger
       tables created, audit captured, backend = sso CLI
       subprocess) ran end-to-end.
```

If you see `[AUTH] Claude CLI returned 401`: your terminal subprocess
isn't reading the keychain. Run `claude` from the same terminal first to
confirm you're logged in. If that works but the demo doesn't, you may be
in a nested shell (WSL, Docker exec) that the keychain doesn't bridge to.

### Extending the live demo to use the client-daemon

The live demo above uses Claude CLI's built-in Bash tool — runs on YOUR
machine because the CLI is local. To extend to the production-shape
"agent uses client-daemon to do FS/shell on user's machine" pattern:

1. Start the daemon locally pointed at a local sub-agents stub (or the
   real VM):
   ```bash
   cd C:\Users\rudol\Documents\veilguard\services\client-daemon
   python veilguard_client.py --config config.yaml
   ```
2. Configure agent-runtime's MCP server list (`app/runtime.py:_build_mcp_server_config`)
   to include the sub-agents WS endpoint at `http://localhost:8809/mcp`
   (default).
3. Have the persona's `tool_allow_list` include daemon-prefixed tools
   (e.g. `mcp__client_daemon__run_command`).
4. Re-run the demo. Now Director's "list directory" request routes:
   Claude SSO → tool_use → MCP → sub-agents → WS → client-daemon → your
   Windows shell. The approval gate hook (Phase 0.3 wiring) intercepts.

The wire is built end-to-end — the live demo above proves the SSO leg
and the wiring proves the rest. The remaining piece is local-vs-VM
daemon setup, which is environmental, not architectural.

---

## Step 2 — run the cache spike (10 minutes, ~$2 of Anthropic credit)

The ONE architecturally-risky bet: does the Claude Agent SDK preserve
TCMM's cache_control markers when it relays the system prefix to
Anthropic? TCMM owns marker placement per-provider (it already does
this for the pii-proxy → direct-Anthropic path). agent-runtime passes
TCMM bytes through unmodified; if the SDK rewrites the markers, the
cache breaks.

Run against your actual TCMM service + a real conversation:

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Against a running TCMM (recommended):
python scripts/spike_cache_validation.py \
    --tcmm-url http://localhost:8811 \
    --conversation-id <real-conv-id> \
    --user-id <real-user-id> \
    --calls 5

# Or against a recorded fixture (offline):
python scripts/spike_cache_validation.py --fixture sample_render.json --calls 5
```

The spike will print TCMM's block layout first (how many blocks, where
the cache_control markers are) so you can spot weird shapes before
spending credit. Then it sends 5 SDK calls and reports hit rates.

Expected outcome:
- Call 1: `cache_create ≈ TCMM-prefix-tokens`, `cache_read ≈ 0` (warming)
- Calls 2-5: `cache_create ≈ 0`, `cache_read ≈ prefix-tokens`, hit_rate 95%+
- Final verdict line: `[VERDICT] PASS — SDK preserves TCMM's cache_control markers`

If you see `[VERDICT] FAIL`:
- The SDK is rewriting markers between our pass-through and the API.
- Options (in increasing complexity):
  1. Re-introduce `normalize_cache_control()` defensively after TCMM
     in `app/middleware/tcmm.py` — re-place the marker on the last
     block agent-runtime sends, hoping the SDK respects an explicit
     placement
  2. Patch claude-agent-sdk locally (depends on version + maintainer)
  3. Bypass the SDK for the LLM call: use a raw Anthropic client
     inside runtime.py; keep the SDK only for subagent + MCP machinery

**Do not deploy until this spike passes.** The whole multi-agent
economics depends on cache hits (~$24/tenant/day savings per spec §10.5).

---

## Step 3 — build + deploy the container

```bash
cd C:\Users\rudol\Documents\veilguard
# Pull latest VM state if anything has changed since dev:
gcloud compute scp --recurse veilguard-prod-jnb:~/veilguard/. ./vm-pull-current --zone=africa-south1-a

# Push the new agent-runtime + updated compose:
gcloud compute scp --recurse ./agent-runtime veilguard-prod-jnb:~/veilguard/ --zone=africa-south1-a
gcloud compute scp ./docker-compose.yml veilguard-prod-jnb:~/veilguard/ --zone=africa-south1-a

# SSH and bring up:
gcloud compute ssh veilguard-prod-jnb --zone=africa-south1-a
cd ~/veilguard
docker compose build agent-runtime
docker compose up -d --no-deps agent-runtime

# Verify:
docker compose logs -f agent-runtime
curl http://localhost:5000/health
curl http://localhost:5000/agents    # should list 8 personas
```

If `/health` returns 503: check `config_errors` in the response body.
Most common: `ANTHROPIC_API_KEY` unset, or `agents/` directory not mounted.

**Permission sanity check** (per memory `architecture_lance_index_perms`):
the agent-runtime process MUST run as the same UID that owns
`/home/rudol/veilguard/tcmm-data/`. Container runs as root by default;
either:
- Run with `--user $(id -u rudol):$(id -g rudol)` in compose, OR
- Pre-create new Lance tables as `rudol` user, OR
- Add `chown -R rudol:rudol tcmm-data/` to a healthcheck script.

Watch the agent-runtime logs on first request: if you see "WARNING — Lance
dir owned by uid=X but process running as uid=0", fix before continuing.

---

## Step 4 — wire pii-proxy to forward Anthropic calls

The agent-runtime is now running but pii-proxy doesn't know about it.
Add the routing decision in `agent-proxy/app/main.py`:

```python
# (sketch — actual edit goes in the anthropic route handler)
if os.environ.get("AGENT_RUNTIME_ENABLED", "false").lower() == "true":
    # Forward Anthropic-bound requests to agent-runtime
    agent_runtime_url = os.environ.get(
        "AGENT_RUNTIME_URL", "http://agent-runtime:5000"
    )
    response = await client.post(
        f"{agent_runtime_url}/agent/query",
        json={
            "conversation_id": conv_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "agent_id": "director",  # default; LibreChat picker can override
            "messages": body["messages"],
            "stream": True,
        },
    )
    return StreamingResponse(response.aiter_bytes(), media_type="text/event-stream")
```

Set `AGENT_RUNTIME_ENABLED=true` in `.env` (and `AGENT_RUNTIME_URL=http://agent-runtime:5000`).

For Phase 0 you can start with this turned ON for one tenant only (filter by `user_id`) so the blast radius is bounded.

---

## Step 5 — validate end-to-end

Send a message from LibreChat. You should see in `agent-runtime` logs:

```
[startup] loaded 8 personas: builder, critic-claim, critic-prose, ...
[tcmm] cache MISS parent_cid=conv-abcdef agent=director blocks=N
[approval] decision=allow tool=... agent=director user=...
[audit] writer ready — /tcmm-data/veilguard/tcmm.db::pii_audit
```

And in the `pii_audit` LanceDB table:

```bash
docker exec -it agent-runtime python -c "
import lancedb
db = lancedb.connect('/tcmm-data/veilguard/tcmm.db')
print(db.open_table('pii_audit').to_arrow().to_pandas().tail(5))
"
```

You should see rows with `direction=FROM_LLM`, `cache_read > 0` (after
the second turn), and the `extra` JSON column carrying `agent_id`,
`tenant_id`, `tool_calls`, etc.

---

## Step 6 — what's safe to defer (and what isn't)

**Safe to defer (these work as stubs/placeholders):**
- Windows-toast approval surface in client-daemon. Currently stubs to
  DENY for background-origin calls (`APPROVAL_FAIL_CLOSED=true`). Means
  Builder can't run shell commands until you wire the real toast. That's
  a real product limitation but doesn't block deploy of the runtime.
- Dream-as-scheduler proposal hook. The scoring + Lance tables are
  built; the actual hook into TCMM dream's `run_cycle()` is a separate
  patch into the TCMM tree (Phase 3 in the spec).
- `org_memory` Lance writes from reflective_heuristic promotion. Schema
  + CRUD exist; the trigger is wired in Phase 3.
- A2A endpoint exposure. Phase 4.

**Do NOT defer:**
- The cache-validation spike. If cache doesn't work, the multi-agent
  cost story (§10.5) is invalid and the whole bet is wrong.
- Lance permission sanity. One root-owned `_indices/` directory kills
  recall silently. We log a warning on startup; act on it.
- The CRLF/UTF-8 sanitizer in CONSTITUTION.md load. Already wired and
  tested (test_text_sanitize.py); just don't disable it.

---

## Troubleshooting

**`/health` returns 503 with `personas_loaded: 0`:**
- `agents/` dir not mounted or empty. Check `docker compose exec agent-runtime ls /app/agents`.

**`/agent/query` returns "claude_agent_sdk not importable":**
- Container built but SDK install failed silently. Check `requirements.txt`
  pin matches what's available on PyPI.

**Sub-agent calls have low cache hit rate but spike passed:**
- TCMM render is producing different bytes per call. Check
  `app/middleware/tcmm.py` memoization — `parent_cid` extraction may be
  wrong for your conversation ID format.

**Approval gate fires on tools you expected to be free:**
- Check `app/policy/client_tool_policy.py` — the tool may be in
  `CLIENT_TOOLS` even though you didn't expect. Add to `SAFE_READ_TOOLS`
  if appropriate, or rename the tool to not match the MCP prefix.

**Critic agents not getting triggered:**
- The MCP server for critic invocations needs to route `submit_for_review`
  → status=review → Critic polls. Phase 2 wiring; in Phase 0 the
  AgentDefinitions exist but the inbox-polling loop is a TODO.

---

## What the spec says vs what this ships

- Spec §3.7 dream-as-scheduler: scoring + Lance tables built; **hook into
  TCMM dream's `run_cycle()` is unbuilt** (Phase 3).
- Spec §3.8 approval gate: capability matrix + hook built; **Windows
  toast is stubbed** (real UI in client-daemon Phase 0.3.5).
- Spec §3.9 strategy layer: Constitution loader + scorer built;
  **org_memory promotion trigger is unbuilt** (Phase 3).
- Spec §3.10 decision ledger: all 7 tables built; **`work_items_v` UNION
  view is unbuilt** (Phase 4).
- Spec §4 personas: all 5 markdown files written; **MCP tool registration
  for the new agent-runtime tools (`create_task`, `assign_task`, etc.)
  is unbuilt** — they're stubs in `app/ledger/tasks.py` and need MCP
  exposure (Phase 2).

The harness is the bedrock; everything else builds on it.

---

## Next session — Phase 2 picks up here

1. Wire `create_task`, `assign_task`, `convert_proposal`, etc. as MCP
   tools the SDK can dispatch.
2. Build the inbox-polling loop for ICs (Researcher/Builder/Critics
   need to pick up `status=review` tasks).
3. Implement the Windows toast handler in `client-daemon/veilguard_client.py`.
4. Wire dream-cycle hook for proposal emission (Phase 3).
5. Sidebar UI for the decision ledger (Phase 2-3 / fork-patch work).

See `MULTI_AGENT_PLATFORM.md` §5 Phase plan for the full sequence.
