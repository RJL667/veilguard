# Veilguard Multi-Agent Platform — Design

**Status:** Draft. Architectural plan, not an implementation spec.
**Date:** 2026-05-21
**Scope:** Extend Veilguard from "single assistant + ephemeral sub-agent calls" into a small **corporate-structured organization of agents**: a Director coordinating Researcher / Builder / Critic ICs, with on-demand domain consultants (existing security personas). Agents have org-chart relationships, persistent private memory, an inbox for async messages, first-class Task records, shared documents in workspace, and a safety gate on dangerous tool calls.

This document was distilled from a multi-round research panel (state-of-the-art survey, workspace patterns, codebase mapping, critic pass, edge-case audit, adversarial security review, latency/cost modeling, onboarding review) and reshaped per the design intent of a corporate structure with inter-agent communication and shared work. Open questions and risks are kept explicit at the bottom rather than papered over.

---

## TL;DR (read this first if you've never seen the doc before)

**What Veilguard is today:** a self-hosted multi-tenant AI assistant with a PII-redacting LLM proxy, a memory layer called TCMM (LanceDB-backed, with a cognitive-architecture-style "dream" knowledge-graph builder), a LibreChat-fork UI, a sub-agents Python service exposing MCP tools, and a Windows client-daemon that gives users a local workspace.

**What we're adding:** a small *organization* of named, persistent agents. One Director (orchestrator), three+1 ICs (Researcher, Builder, critic-claim, critic-prose), plus existing security personas as on-demand consultants. Agents have org-chart relationships, private/team/blackboard memory channels, async Tasks as the primary work primitive, and a Constitution + Org Memory + Regret strategy layer that steers a dream-driven proactive stream.

**What's load-bearing in this doc:**
- §3.10 — every entity (Task / proposal / outcome / lesson) is a row in a unified **decision ledger** with shared lifecycle: `proposed → accepted → executed → evaluated → institutionalized`.
- §3.7 — the **proactive stream**: dream emits signals → Director ranks them → user approves → Tasks → outcomes feed back into dream.
- §3.9 — the **strategy layer**: a user-authored Constitution at the top, system-proposed Org Memory lessons that expire if unreinforced, and per-proposal Regret feeding weekly recalibration.
- §11 — concrete edge-case catalog with top-10 MUST-haves before Phase 2/3 ships.

**System diagram (today + target overlaid):**

```
                                  USER
                                    │
                                    ▼
                          ┌────────────────────────┐
                          │   LibreChat UI (fork)  │
                          │   [DO NOT TOUCH ─ fork │
                          │    patches die on      │
                          │    deploy]             │
                          └──────────┬─────────────┘
                                     │ HTTP
                                     ▼
            ┌────────────────────────────────────────────────┐
            │             PII Proxy (vm-pull/)                │
            │  redacts PII + injects cid + audits all calls   │
            │  + injects ~260k-token TCMM blob as sys prefix  │
            │  ◀ Phase 0.1: cache-stable prefix for sub-calls │
            └─┬──────────┬─────────────┬────────────────┬────┘
              │          │             │                │
              ▼          ▼             ▼                ▼
        ┌─────────┐  ┌─────────┐  ┌─────────┐    ┌──────────┐
        │ Anthropic│ │ OpenAI  │  │  xAI    │    │  TCMM    │
        │  Claude  │ │   GPT   │  │  Grok   │    │ (memory) │
        └─────────┘  └─────────┘  └─────────┘    └────┬─────┘
                                                       │
                       ◀ Phase 0.2 + 3: agent_id      ▼
                          flow through observe/recall  ┌──────────┐
                                                       │  dream   │
                                                       │ (knowledge│
                                                       │  graph)   │
                                                       └────┬─────┘
                                                            │
                                                            ▼
                          ┌──────────────────────────────────────────┐
                          │  Sub-agents service (MCP tools, Python)  │
                          │  - existing: handle_tool dispatch        │
                          │  - existing: daemons.py (background)     │
                          │  ◀ Phase 0.0-0.3: agent registry, gate   │
                          │  ◀ Phase 1+: Director / ICs / consultants│
                          │  ◀ Phase 3: dream-as-scheduler           │
                          └────────────────┬─────────────────────────┘
                                           │ WebSocket
                                           ▼
                          ┌──────────────────────────────────────────┐
                          │     Client-daemon (user's Windows box)   │
                          │  filesystem + shell + browser tools      │
                          │  ◀ Phase 0.0.4: capability handshake     │
                          │  ◀ Phase 0.3: approval toast (winotify)  │
                          └──────────────────────────────────────────┘

  Sidebar Daemons tab (/api/veilguard-client/daemons):
    safe surface for all new UI — no LibreChat fork patches (mostly)
```

**5 things you must read before Phase 1:**
1. **§3.10 decision-ledger framing** — explains why we treat Tasks/proposals/lessons/outcomes uniformly.
2. **§10 worked example** — a real Task traced end-to-end. Skip ahead if §3-§4 feels abstract.
3. **§3.7.0 dream hook correction** — the most-likely-to-be-implemented-wrong piece.
4. **§4 The org** — the 5 agent roles with their tools/memory/triggers.
5. **§11 edge case catalog** — top-10 MUST-haves before Phase 2/3.

---

## Glossary

| Term | Meaning |
|---|---|
| **TCMM** | "Tagged Conversational Memory Manager" — LanceDB-backed memory layer at `.gemini/antigravity/tcmm/TCMM/`. Has `observe()` / `recall()` / `render()` API. Per-tenant + per-user isolation enforced by `_ns_filter`/`_user_filter` in lance.py. |
| **dream module** | A multi-thousand-LOC cognitive-architecture inside TCMM (`core/dream/`). Builds a knowledge graph with typed claims, contradiction_arcs, stance_arcs, reflective_heuristics, etc. The multi-agent platform integrates with it; does NOT replace it. |
| **pii-proxy** | LLM gateway at `vm-pull/main_prod_.py`. All Anthropic/OpenAI/xAI calls pass through it. Injects cid + the ~260k-token TCMM blob as system prefix. **VM is source-of-truth.** |
| **sub-agents service** | Python FastAPI service at `Documents/veilguard/services/sub-agents/`. Exposes MCP tools to LibreChat. Where most new code in this spec lands. |
| **client-daemon** | Per-user Windows binary at `Documents/veilguard/services/client-daemon/`. Filesystem + shell + browser access. Ships via `build.bat` + Inno Setup + auto-update poll (~30 min). |
| **LibreChat fork** | UI lives at `Documents/veilguard/librechat-src/`. Fork patches die on deploy unless baked into Dockerfile. Treated as fragile. |
| **Sidebar Daemons tab** | At `/api/veilguard-client/daemons`. The "safe" UI surface — end-to-end controlled, no fork rebase risk. Primary surface for multi-agent UI. |
| **typed_claim** | A dream-module data structure: S/P/O triple with bitemporal validity, polarity, modality, source_kind, extracted_by. Defined in `core/dream/typed_claims.py`. |
| **agent_id** | Unique identifier per agent within a tenant. Added in Phase 0.2. Flows through `extracted_by` field on typed_claims. |
| **Task** | The primary work primitive. One Lance table `agent_tasks`. Has owner_id, status, brief, deliverable_spec, inputs, outputs, comments. |
| **Proposal** | A candidate Task surfaced by dream-as-scheduler. Lives in `task_proposals` table until user approves → becomes Task. |
| **Constitution** | User-authored `agents/CONSTITUTION.md`. Top-level steering: objectives + constraints + metrics. Steers proposal ranking. |
| **Org Memory** | Lance table `org_memory`. Institutional lessons (rules about how the org operates) with expiry + decay. |
| **Decision ledger** | The unified conceptual model: every entity (Task/proposal/lesson/outcome) shares a lifecycle skeleton. See §3.10. |
| **Critic-claim** | One of the two split Critic agents. Haiku, inline structural arbiter on typed_claims promotion. |
| **Critic-prose** | The other split Critic. Sonnet, async PR-style review of artifacts (markdown drafts, code). |
| **Pattern A/B/C/D** | Four reactive interaction patterns: Solo / Single delegation / Parallel fanout / Background ongoing. See §3.6. |

---

## 0a. Architecture pivot — agent-runtime + Claude Agent SDK (2026-05-22)

After the SDK fit audit (panel round, see decision log entry 2026-05-22), the spec's Phase 0 model collapsed significantly:

**Decision:** stand up a new sibling service `agent-runtime/` that embeds the Claude Agent SDK as the LLM-tool-LLM loop. PII proxy keeps doing multi-provider routing, redaction, and audit; when a request is Anthropic-bound, the proxy forwards to agent-runtime instead of going direct to api.anthropic.com.

**Shape:**

```
LibreChat (UI + auth + MongoDB)
        ↓
pii-proxy (Anthropic | OpenAI | xAI | Gemini routing + redaction + audit)
        ↓ (if Anthropic + multi-agent-aware)
agent-runtime  ←  this is the new service
   - embeds Claude Agent SDK
   - middleware: TCMM render, cache_control normalize, tenant context, audit
   - hooks: approval gate (PreToolUse)
   - subagents: AgentDefinition per persona
   - state: agent_tasks + task_proposals + org_memory Lance tables
        ↓
Anthropic API
```

**What the SDK gives us for free** (so this collapses from spec):
- Subagent spawning + isolated context windows — was §3.7.5 / §4 design problem; now `agents={…AgentDefinition…}` per call
- Prompt caching with cache_control markers — was Phase 0.1 hand-waved; now SDK auto-places, we just normalize (Phase 0.1 shrinks to ~80 LOC of defensive wrapping)
- MCP client (connects to our sub-agents + client-daemon as tool servers) — was Phase 4 work; now `mcp_servers={…}` per call
- PreToolUse hook for the approval gate — was a 30-LOC patch at agentic.py:122; now `HookMatcher(matcher="^mcp__client_daemon__", hooks=[approval_gate_hook])`
- Streaming + intermediate tool-call events — was UI plumbing problem; now `include_partial_messages=True`

**What we still build ourselves** (spec drives this; SDK doesn't cover):
- Tenant context (cid, user_id, tenant_id, agent_id) — SDK is tenant-blind by design; we own routing
- TCMM render middleware — fetches the 260k-token blob, normalizes cache_control, memoizes per parent_cid
- Audit wrapper — taps the SDK's message stream to capture tokens + write to existing `pii_audit` Lance table
- Approval gate logic — capability matrix from §3.8 (PreToolUse hook calls our `classify()`)
- Persona loader — bold-Markdown KV → SDK's `AgentDefinition`
- Constitution loader + scorer
- Decision-ledger Lance tables (agent_tasks, task_comments append-only chain, task_proposals, proposal_outcomes, org_memory, client_tool_approvals, client_tool_bypass)
- All the multi-agent state primitives the spec describes

**Built as of 2026-05-22:** the entire harness above + 5 persona markdown files (Director, Researcher, Builder, critic-claim, critic-prose) + decision-ledger schemas + tasks/comments/proposals CRUD. 121 unit + integration tests passing. Live cache-validation spike script ready for user to run with a real API key.

**See:** `agent-runtime/` directory in the repo + `agent-runtime/README.md` runbook for what to do next.

---

## 0. Prerequisites (read this BEFORE any code change)

A panel of 5 audit agents surfaced foundational issues with earlier spec drafts. This section is the operational discipline + source-of-truth layer that must land BEFORE Phase 0.1 ships, or every later phase inherits avoidable rework.

### 0.1 Source-of-truth table per artifact

| Artifact | Local path | VM path | Authority | scp direction | Restart |
|---|---|---|---|---|---|
| `CONSTITUTION.md` | `Documents\veilguard\agents\CONSTITUTION.md` | `/home/rudol/veilguard/agents/CONSTITUTION.md` | **VM after first deploy** (user amends in place) | local→VM once; then VM→local periodically | sub-agents |
| `agents/*.md` persona files | `Documents\veilguard\agents\` | `/home/rudol/veilguard/agents/` | Local-authoritative | local→VM | sub-agents |
| `services/sub-agents/` Python | `Documents\veilguard\services\sub-agents\` | `/home/rudol/veilguard/services/sub-agents/` | Local-authoritative | local→VM | sub-agents |
| `client-daemon/veilguard_client.py` | `Documents\veilguard\services\client-daemon\` | NA (installer ships to user Windows) | Local-authoritative | `build.bat` → `publish_release.py` → VM `/downloads/` → user auto-update (~30 min poll) | user Windows machine |
| TCMM `core/`, `api/`, `adapters/` | `.gemini\antigravity\tcmm\TCMM\` (stale) | `/home/rudol/veilguard/tcmm-service/` | **VM-authoritative** (drift confirmed; per `architecture_tcmm_service_source.md`) | **VM→local FIRST, diff, merge, push back** | tcmm-service |
| pii-proxy `vm-pull/main_prod_.py` | `Documents\veilguard\vm-pull\main_prod_.py` | `/home/rudol/veilguard/` | **VM-authoritative** | VM→local FIRST | pii-proxy + api containers |
| New Lance tables (`agent_tasks`, `task_proposals`, `proposal_outcomes`, `org_memory`, `client_tool_approvals`, `client_tool_bypass`) | NA | `/home/rudol/veilguard/tcmm-data/veilguard/tcmm.db/<table>.lance/` | VM data — **must be created by service running as `rudol`, NEVER root** | NA | NA |
| `pii_audit.task_id` column add | NA | VM-side LanceDB | VM data | NA | pii-proxy (after schema add) |
| Sidebar UI (Agents view, Proposed Tasks, Lessons, Decision Ledger) | `Documents\veilguard\librechat-src\` + `deploy/librechat-patches/` | `/home/rudol/veilguard/librechat-src/` | **Both must match — fork patch territory** | Bake into `Dockerfile.librechat`; image rebuild | docker compose up -d --no-deps api |

**Standing rules:**
- For VM-authoritative files: ALWAYS `gcloud compute scp` VM→local FIRST, diff, merge edits IN the VM copy, then push back. The "Claude wipes VM" pattern is documented multiple times in memory; this discipline prevents recurrence.
- For local-authoritative files: scp local→VM is safe ONLY if no on-VM editing convention exists for that file.
- For "both must match" (fork patches): never bind-mount dist directory; bake all changes into `Dockerfile.librechat` and rebuild image.

### 0.2 CRLF / UTF-8 sanitization (mandatory at config-load boundaries)

Veilguard's history shows CRLF + Windows-1252 smart-quotes crashing parsers twice (`.env` × 2). Two new user-editable artifacts inherit the same risk: `CONSTITUTION.md` and `agents/*.md` frontmatter.

**Required at every config-load entry point** in sub-agents service startup + on file change. **Two modes** (implementation audit caught that one sanitizer for both env values and markdown text corrupts prose):

```python
# Mode "env" — lossy ASCII rescue, for .env values and config keys
# Mode "text" — Unicode-preserving rescue, for markdown body / prompts

_CP1252_RESCUE_ENV = {  # ASCII targets for env values
    0x91:"'", 0x92:"'", 0x93:'"', 0x94:'"',
    0x96:"-", 0x97:"--", 0x85:"...", 0x82:"'", 0xA0:" ",
}
_CP1252_RESCUE_TEXT = {  # proper Unicode codepoints for prose
    0x91:"‘", 0x92:"’", 0x93:"“", 0x94:"”",
    0x96:"–", 0x97:"—", 0x85:"…", 0x82:"‚", 0xA0:" ",
}

def sanitize_bytes(raw: bytes, mode: Literal["env","text"]) -> tuple[str, list[Issue]]:
    if raw.startswith(b"\xef\xbb\xbf"): raw = raw[3:]   # strip BOM
    rescue = _CP1252_RESCUE_ENV if mode == "env" else _CP1252_RESCUE_TEXT
    try:
        s = raw.decode("utf-8")
    except UnicodeDecodeError:
        ba = bytearray(raw)
        for i,b in enumerate(ba):
            if b in rescue:
                ba[i:i+1] = rescue[b].encode("utf-8")
        s = bytes(ba).decode("utf-8","replace")
    s = s.replace("\r\n","\n").replace("\r","\n")
    s = s.replace("\x00","")  # NUL never legal
    return s, issues_diff(raw, s)
```

**Mode usage rules:**
- `.env`, sub-agents config files, daemon config → `mode="env"` (em-dash → `--`, lossy but safe)
- `CONSTITUTION.md` frontmatter values → `mode="env"` (constraint rule strings are config)
- `CONSTITUTION.md` description fields + `agents/*.md` system-prompt body → `mode="text"` (preserves prose Unicode)

**Detection vs correction:** sanitize functions return `(clean_text, issues_list)`. `write_back=True` only on deploy preflight (with a loud "wrote N substitutions" log line); CI uses detect-only and FAILS on any issue. Two engineers will diverge on default; spec makes it explicit.

Failure path (no sanitization): smart-quote in `CONSTITUTION.md` description → yaml parse error at startup → Director init fails → sidebar shows "Waiting for connection..." → 503 cascade across `/api/client/status`. Same shape as the documented `VEILGUARD_INTERNAL_SECRET` cascade.

**Acceptance test for Phase 0.0:** ship a CONSTITUTION.md with intentional CRLF + smart-quote contamination, verify Director loads cleanly with a logged warning, verify the in-memory parsed structure has clean values.

### 0.3 agents/*.md frontmatter parser (does not exist today)

Touchpoint audit confirmed: existing `agents/*.md` files are prose-only. No YAML frontmatter parser exists. Spec assumes `agent_id`, `role`, `manager_id`, `team_id`, `tool_allow_list` are machine-readable — they're not, yet.

**Phase 0.0 deliverable:** add a frontmatter parser to sub-agents service startup. **Format: bold-Markdown KV** (matches the 3 existing personas):

```markdown
# Phishing Analyst

**Model:** claude-sonnet-4-6
**Tools:** filesystem (read_file, list_directory), web (browse_url)
**Role:** consultant
**Agent ID:** phishing-analyst
**Manager:** none
**Team:** none

## System Prompt
<rest of file is markdown body, becomes System Prompt>
```

Extended frontmatter keys (added by Phase 0.0):
- `Agent ID:` — unique per tenant. Defaults to filename stem if absent.
- `Role:` — `director | ic | consultant`. Defaults to `consultant` for files without this key.
- `Manager:` — agent_id of manager. Nullable / "none" for director and consultants.
- `Team:` — team_id. Nullable / "none" for consultants.
- `Model:` — model identifier. Required.
- `Tools:` — grouped (`group (tool1, tool2)`) OR flat comma-list. Parser handles both.
- `Schema Version:` — int, default 1, for forward-compat.

**Parser grammar:** `^\*\*(Key Name):\*\*\s*(value)$` matched line-by-line in the header region (before `## System Prompt`). Body after `## System Prompt` is the system prompt verbatim.

**`PROMPTS.md` and `CONSTITUTION.md` exclusion:** both live in the same `agents/` dir but are not personas. Parser whitelists files matching `^[a-z][-a-z0-9]+\.md$` (lowercase + hyphens) and requires a `**Model:**` line to count. Others are skipped.

**Backward compatibility:** the 3 existing security personas (`phishing-analyst`, `threat-analyst`, `report-writer`) currently lack `Role:`/`Manager:`/`Team:`/`Agent ID:`. Parser defaults: `Role=consultant`, `Manager=none`, `Team=none`, `Agent ID=<filename stem>`. Migration to explicit values is non-blocking.

**Director's dual-model schema** (resolves a previously open question): the `Model:` key supports two forms:
- Scalar: `**Model:** claude-sonnet-4-7` (single model, all calls)
- Mapped: `**Model:** reactive=claude-sonnet-4-7, rank_pass=claude-haiku-4-5` (per-tool model selection)

Director uses the mapped form. ICs and consultants use scalar form by default.

### 0.4 Capability handshake (sub-agents ↔ client-daemon)

Approval gate (§3.8) calls `bridge.request_approval()` — **a method that doesn't exist today**. Once added, deployment ordering matters: sub-agents service ships before user daemons auto-update. Without a capability handshake, every client-tool call hangs at the new gate calling a method old daemons don't implement.

**Phase 0.0 deliverable: bidirectional capability handshake** (implementation audit found that one-way isn't enough — Phase 0.3 requires the daemon to *know* the server enforces the gate, so it doesn't proxy commands that bypass the approval policy locally):

**Daemon → server** on connect:
```json
{
  "capabilities": {
    "schema_version": 1,
    "daemon_version": "0.3.0",
    "tools": ["run_command","read_file","write_file","list_directory",...],
    "features": {
      "approval_gate": true,
      "streaming": false,
      "utf8_mode": true,
      "max_payload_bytes": 50000
    },
    "limits": {"request_timeout_s": 60, "concurrent_tools": 4}
  }
}
```

**Server → daemon** in auth-ack:
```json
{
  "status": "authenticated",
  "client_id": "...",
  "server_capabilities": {
    "schema_version": 1,
    "features": {
      "approval_gate_enforced": true,
      "audit_logged": true
    }
  }
}
```

**Graceful degradation:**
- Old daemon (no `capabilities` key): server synthesizes baseline (`tools=run_command/read_file/write_file/list_directory`, `features={}`). Background-origin calls → DENY with sidebar warning "Daemon vN missing approval_gate; update recommended."
- Server doesn't advertise `approval_gate_enforced=true`: daemon refuses to honor background commands and surfaces "Server does not enforce approval gate" notification. Defense-in-depth.

**Error code discipline:** capability-mismatch rejection uses code `-32003` with message that does NOT contain the substrings `"invalid token"` or `"missing user_id"` — those trigger the existing `CredentialsRevokedError` heuristic in `veilguard_client.py:818-823` which wipes daemon config and forces re-pairing. Subtle and would be a regression if missed.

**Pre-flight `execute_remote`:** new check — if `tool not in bridge.capabilities.tools`, return synthetic error WITHOUT a network round-trip. Saves latency on every tool call against tools the daemon doesn't support.

### 0.5 What's built today vs what Phase 0 ships

To be built in Phase 0 (not present in the codebase today): `bridge.request_approval()`, `utils/client_tool_policy.py`, `_agent_filter()` in lance.py, all six new Lance tables (`pii_audit`, `client_tool_approvals`, `client_tool_bypass`, `agent_tasks`, `task_proposals`, `proposal_outcomes`, `org_memory`), `agent_id` parameter on `add_new_block()`, `extracted_by` field on `ObserveRequest`, agents/*.md frontmatter parser, `run_proposal_pass()` mini-cycle, dream-cycle scheduler.

Phase tables in §5 mark each as a deliverable. This subsection exists so a reader doesn't assume any of these names refer to existing code.

### 0.6 Canonical line numbers (anchor)

| Symbol | Path | Line |
|---|---|---|
| `get_child_conversation_id()` | `services/sub-agents/core/request_ctx.py` | 143 |
| `add_new_block()` | TCMM `core/tcmm_core.py` | 1871 |
| `ObserveRequest` class | TCMM `api/models.py` | 11 |
| `RecallRequest` class | TCMM `api/models.py` | 50 |
| `handle_tool()` | `services/sub-agents/core/agentic.py` | 122 |
| `run_cycle()` | TCMM `core/dream/dream_engine.py` | 6312 (returns ~6537) |
| Reflective stage 52f | TCMM `core/dream/dream_engine.py` | 12082+ |
| `_ns_filter`/`_user_filter`/`_session_filter` | TCMM `core/providers/lance.py` | 1498-1535 |
| `extras_json` field | TCMM `core/providers/lance.py` | 220 |
| `extracted_by` field | TCMM `core/dream/typed_claims.py` | 183 |

**Dream hook point reminder:** wrap `run_cycle()` END (after line 12082+). Wrapping the inner `run_dream_cycle()` at line 5993 misses the reflective signals because they're emitted later in the cycle.

---

## 1. Today's state (what already exists)

The platform is further along than a greenfield design would suggest:

- **PII proxy** at the LLM boundary (`vm-pull/main_prod_.py`). Multi-provider. Owns conversation ID injection, audit logging to LanceDB `pii_audit`, and TCMM render integration. **Source-of-truth = VM** (the `vm-pull/` prefix says so).
- **TCMM** at `C:\Users\rudol\.gemini\antigravity\tcmm\TCMM\`. LanceDB-backed tiered memory with `observe()` / `recall()` / `render()`. Tenant + user isolation at `_ns_filter` / `_user_filter` in [lance.py:1498-1535](C:/Users/rudol/.gemini/antigravity/tcmm/TCMM/core/providers/lance.py). 45+ columns plus an `extras_json` forward-compat field at [lance.py:220](C:/Users/rudol/.gemini/antigravity/tcmm/TCMM/core/providers/lance.py:220). **VM is source-of-truth** — local tree drifts; pull VM→local before editing.
- **Sub-agents service** at `services/sub-agents/`. FastAPI exposing MCP tools. Today supports ephemeral role-personas (ROLES dict: analyst, threat, writer, coder, researcher, critic, verify, security, explore, planner) and background daemons via [start_daemon](C:/Users/rudol/Documents/veilguard/services/sub-agents/tools/daemons.py).
- **Persona convention.** Three deployed agents as markdown files: [phishing-analyst.md](agents/phishing-analyst.md), [threat-analyst.md](agents/threat-analyst.md), [report-writer.md](agents/report-writer.md). Each declares Model, Tools, System Prompt, workflow, output format. This is the seed of the registry.
- **Skills library** at `services/sub-agents/skills/` (phishing-triage, incident-response, threat-intel). Reusable procedural knowledge agents invoke.
- **Workspace area** at `Documents\veilguard\workspace\`. Persistent shared scratch.
- **Client-daemon** at `services/client-daemon/veilguard_client.py`. Runs on the user's Windows machine, exposes FS/shell tools over WebSocket.
- **LibreChat fork** as UI — treated as fragile. **No multi-agent UI work touches the fork.**
- **Sidebar Daemons tab** at `/api/veilguard-client/daemons`. End-to-end controlled, no rebase risk. Primary UI surface for the org.

### What's actually missing

1. **No agent identity in the data model.** No `agent_id` or `author` column in TCMM. No way to ask "what did the Critic say last week" or "is this user-authored or agent-authored."
2. **No org structure.** Agents have no notion of who reports to whom, what team they're on, or what their role is. Today they're stateless role prompts.
3. **No inter-agent messaging.** No inbox, no peer DMs, no task records. The only "coordination" is implicit — multiple sub-agent calls landing in the same conversation namespace.
4. **No first-class Task primitive.** Work isn't tracked as an entity with brief / deliverable / status / deadline. It's just function calls returning text.
5. **No shared document workflow.** Workspace files exist but there's no check-in / check-out, no review gate, no versioning model for multi-agent edits.
6. **No tool-call origin distinction.** Background scheduled agents calling `shell_exec` on the user's Windows box use the same code path as foreground user calls. No human gate.
7. **No cache-stable conversation id for sub-agent calls.** Each sub-call cold-creates the prompt cache. With an org of 5 agents, this is multiplicatively expensive. Must be fixed before multi-agent UX ships.

---

## 2. Design principles

Non-negotiables extracted from the critic pass and existing-regression memory.

- **Do not touch the LibreChat fork.** Every multi-agent surface lives in the sidebar Daemons tab or a Windows-native surface from the client-daemon.
- **Provenance is mandatory.** Every memory block carries `author`. Agent-authored content must be distinguishable from user-authored at recall time.
- **No autonomous client-daemon access.** Background agents calling FS / shell / browser tools must pass through a human-approval gate. Prompt injection from any fetched page is otherwise an RCE channel into Windows.
- **No interactive debate loops.** Critic reviews artifacts asynchronously like a PR reviewer, not as a back-and-forth interlocutor. Per ICLR 2025 critical review, multi-agent debate does not beat single-agent test-time compute on interactive turns.
- **Async by default.** Inter-agent communication is inbox-based, not synchronous request-response. Sync delegation exists as a tool when the orchestrator needs a blocking sub-call, but the default coordination shape is async — like a real organization.
- **Asymmetric authority.** The Director can delegate; ICs can publish to the team channel but not the blackboard; the Critic alone can promote to the org-wide blackboard. Authority is encoded in the role, not in agent prompts.
- **Recall fanout must not multiply the existing /pre_request bottleneck.** Cache-stable cid lands before any multi-agent UX.
- **Reuse existing primitives.** Agent identity = extension of `agents/*.md`. Persistent agents = renaming the daemons mechanism + adding identity. Workspace = a dir convention. Inter-agent messaging = A2A protocol (don't build a bespoke RPC).
- **Honor existing user memory + project constraints.** The `feedback_no_worktrees` memory entry says: "work directly in canonical paths; never `isolation:"worktree"`." Earlier drafts violated this with a worktree-per-agent mandate; corrected in §3.5. The general rule: if a memory entry contradicts a spec choice, the memory wins unless the spec explicitly justifies an exception.
- **`source_kind` is set by the tool, never by the agent.** Agent prose cannot influence its own provenance class. `fetch()` always stamps `source_kind=TOOL_RESULT`. User-conversation observe always stamps `source_kind=USER` (verified via `x-user-id` header at pii-proxy). Agent observe is hard-coded to a per-agent class (e.g., `AGENT_RESEARCHER` or `extracted_by=agent:<aid>` with `source_kind=INFERRED`). Any attempt to overwrite from agent prose is rejected server-side. This closes the prompt-injection provenance-laundering hole.

---

## 3. Architecture

### 3.1 The organization

```
                            User
                             │
                             ▼
                  ┌──────────────────────┐
                  │      Director         │
                  │   (orchestrator,      │
                  │    dual-stream,       │
                  │    task-router)       │
                  └──────────┬───────────┘
                             │ delegates
       ┌─────────────────────┼─────────────────────┐
       ▼                     ▼                     ▼
 ┌──────────┐         ┌──────────┐          ┌──────────────────┐
 │Researcher│         │ Builder  │          │  Critic (split)  │
 │ (analyst,│         │  (eng,   │          ├──────────────────┤
 │  fanout) │         │  tools,  │          │ critic-claim     │
 │          │         │ shell)   │          │  (Haiku, inline) │
 │          │         │          │          │ critic-prose     │
 │          │         │          │          │  (Sonnet, async) │
 └────┬─────┘         └────┬─────┘          └────┬─────────────┘
      │                    │                     │
      └────────────────────┼─────────────────────┘
                           │ peer DMs, shared work
                           ▼
                  ┌─────────────────────────────┐
                  │  Team channel (typed)       │ ← TCMM team/<tid>/
                  │   events/  knowledge/ drafts│   {events,knowledge,drafts}/
                  │  + workspace                │ ← agents/team/*
                  └────────┬────────────────────┘
                           │ critic-claim + critic-prose gated publish
                           ▼
                  ┌─────────────────┐
                  │   Blackboard    │ ← TCMM blackboard/
                  │   (org-wide,    │ ← agents/published/*
                  │   read-by-all)  │
                  └─────────────────┘

  Consultants (pulled in on demand by Director or peers):
    ├─ Phishing Analyst        (existing persona)
    ├─ Threat Analyst          (existing persona)
    └─ Report Writer           (existing persona)
```

**IC count = 4** (Researcher, Builder, critic-claim, critic-prose). The Critic split (§4.4) means two distinct agent processes with different models, latencies, and trigger conditions. The org chart treats them as one box for readability but they are operationally separate.

**Reading the chart:**
- **Director** is the only agent the user talks to directly by default. It decomposes user requests, decides whether the work is solo or needs delegation, and routes accordingly. Like a manager fielding inbound work.
- **Researcher / Builder / Critic** are the standing IC team. They can be invoked by the Director, by each other (peer messaging), or run autonomously on scheduled tasks.
- **Consultants** are existing domain agents (`phishing-analyst`, `threat-analyst`, `report-writer`). They join the org as on-demand specialists — invoked when their domain comes up, not standing on the team. The platform-level personas and the domain personas coexist.

### 3.2 Agent model (extended)

An agent is a persistent, named entity with these bindings:

| Binding | Purpose | Storage |
|---|---|---|
| `agent_id` | Unique identifier | Frontmatter in `agents/<id>.md` |
| `role` | One of: `director`, `ic`, `consultant` | Frontmatter |
| `manager_id` | Whom this agent reports to (nullable for Director) | Frontmatter |
| `team_id` | Which team channel this agent shares (nullable for consultants) | Frontmatter |
| System prompt | Role behavior | Markdown body |
| Tool allow-list | What this agent can call | Frontmatter |
| Private memory | Per-agent episodic TCMM (**Director: none**) | TCMM `agent/<aid>/` |
| Team memory | Typed channels shared with teammates | TCMM `team/<tid>/{events,knowledge,drafts}/` |
| Workspace | Per-agent + team-shared dirs | `agents/<aid>/`, `agents/team/<tid>/` |
| Inbox | Derived view (`tasks WHERE owner_id=me AND status IN (...)`) — no separate table | `agent_tasks` |

### 3.3 The Task ledger — single primary primitive

Everything that moves between agents flows through **one** entity: the Task. Messages collapse to task comments. Documents collapse to task artifacts. Inboxes are a query, not a table.

```
task:
  id: uuid
  parent_id: nullable                # subtask decomposition
  owner_id: agent_id                 # who's responsible
  assigner_id: nullable              # who assigned (null = self-initiated)
  status: open | in_progress | blocked | review | done | cancelled
  brief: str                         # what to do
  deliverable_spec: str              # what "done" looks like
  inputs: [refs]                     # task_ids, artifact paths, blackboard refs
  outputs: [paths]                   # workspace file paths produced
  cost: {tokens_in, tokens_out, cache_read, cache_write, usd}
  due_ts: nullable
  created_ts, updated_ts: float
  trace_ref: nullable                # TCMM trace cid for the run
  comments: [
    { id, author_id, ts, kind, body }
  ]
```

`comment.kind` ∈ `{comment, status_change, review_request, review_decision, blocker_raised, blocker_cleared}`. That's how peer communication, status updates, and Critic reviews are recorded — no separate message table, no separate review table.

**Inbox = view, not table.** `SELECT * FROM tasks WHERE owner_id = ? AND status IN ('open', 'in_progress', 'blocked', 'review') ORDER BY priority`. Surfaced in the sidebar. Zero new schema.

**Documents = task outputs.** A draft report at `agents/team/core/drafts/report.md` is the `outputs[0]` of task `t-...`. Versioning lives in the file system (append-only `.history/` or git, depending on location). Multiple tasks can reference the same document via `inputs`/`outputs` — that's the document's "ownership chain."

**Why one table beats three:** every cross-agent action has a Task it belongs to. There's no DM that isn't related to work; there's no document that isn't an artifact of some task. Forcing the Task hierarchy as the spine gives us a free audit log, a free dependency graph, a free cost ledger, and a natural cancellation/cascade semantics.

Persisted in a single new Lance table `agent_tasks` alongside (not inside) TCMM. Reason for separation: TCMM is for unstructured memory recall; Tasks have rigid schema, deterministic lookups, lifecycle state. Conflating them poisons recall quality.

### 3.4 Knowledge layer = the dream module (existing)

Veilguard already has a cognitive architecture. The multi-agent platform does not add a "knowledge layer"; it adds an *agent layer that participates in the existing one*.

The dream module at `.gemini/antigravity/tcmm/TCMM/core/dream/` is a multi-thousand-LOC system that:

- Extracts **typed claims** (S/P/O with bitemporal validity, predicate classes, polarity, 10 modalities, 6 source kinds) — [typed_claims.py](C:/Users/rudol/.gemini/antigravity/tcmm/TCMM/core/dream/typed_claims.py)
- Builds a **graph of derived primitives** (`DERIVED_TYPES` at [dream_engine.py:301](C:/Users/rudol/.gemini/antigravity/tcmm/TCMM/core/dream/dream_engine.py)): identity nodes, concept nodes, narrative arcs, causal arcs, contradiction arcs, stance arcs, temporal arcs, belief_attribution, semantic_principle, information_gap, reasoning_trace, reflective_heuristic, motif_node, recurring_ritual, task_sequence_arc, action_event
- Runs **canonicalization, contradiction detection, supersession, bridge detection, concept gravity, selection pressure** in a `run_dream_cycle` orchestration
- Validates synthesis output (repetition loops, overlap confidence) and tracks per-block stability via success/failure ratios
- Supports multi-backend synthesis (Claude / Vertex / cloud Ollama / local GGUF Qwen/Phi/Gemma) with a two-tier light/main model split

**Implication for agents:**

- Agents `observe()` typed_claims into their private namespace with the existing `extracted_by` field (line 183) carrying `agent:<aid>`. **No new `source_kind` enum values needed** — agents are data, not types.
- The dream cycle ingests those claims. Derived primitives emerge from the graph normally.
- Knowledge currency, decay, supersession are *already* solved by bitemporal `valid_until` + `supersedes` chain. The 7-day stale-dep rule in §10 is a cruder version of what dream natively handles for claims; it stays as the *Task-level* heuristic but doesn't apply to knowledge claims themselves.

**`extracted_by` is not wired end-to-end today** (Phase 0.2 must complete the plumbing):
- `ObserveRequest` in `api/models.py:11` has NO `extracted_by` field. Only `ai_studio_nlp_adapter.py` writes it today (hard-coded values `nlp_v2` / `spacy_fallback` at lines 1267, 1407).
- Phase 0.2 extends `ObserveRequest` with optional `extracted_by: Optional[str]` field, plumbs through `add_new_block` → block extras → TypedClaim. Agent identity arrives via HTTP header `x-agent-id` at the pii-proxy / sub-agents boundary; gets validated against the agent registry before being threaded into `extracted_by`.

**Cross-agent contradiction — design choice required:**

`canonical_triple_hash` today is computed from S/P/O only ([typed_claims.py:239](C:/Users/rudol/.gemini/antigravity/tcmm/TCMM/core/dream/typed_claims.py:239)) — it does NOT include `extracted_by`. So two agents observing the same triple (e.g., Researcher and Builder both claim "passkeys are adopted by Chase") will collapse into one claim via posterior aggregation BEFORE contradiction detection fires. Polarity-flip detection requires both rows to survive distinctly.

**Decision (Phase 0.2):** partition posterior aggregation by `extracted_by` BEFORE merge. Each agent's view of a triple is stored separately; aggregation across agents happens at recall/render time, not at write time. This preserves dream's existing dedup discipline for same-agent-different-utterance dedup while keeping cross-agent claims independently visible to `contradiction_arc` detection.

**Alternative considered, not adopted:** include `extracted_by` in the hash itself. Rejected because it would double-write same-agent reinforcing observations and break the existing `posterior_alpha`/`posterior_beta` aggregation.

**Cross-namespace contamination fix:**

Open Q #10 in earlier drafts asked whether dream's bridge_score / concept_gravity / supersession cross-pollinate two agents' findings within the same namespace. **Audit confirmed: yes, uncontrolled.** `dream_archive.values()` loops at multiple sites (`dream_engine.py:777, 1223, 1462, 1759, 1853, 2093, 2329, 2418, 2548, 2780, 2833...`) have NO namespace filter.

**Phase 1 acceptance test (mandatory):** observe claims in two distinct namespaces with overlapping topics; assert that bridge_score arcs and concept_gravity merges never cross namespace boundaries. If they do, add per-agent-or-per-namespace filtering to the relevant `dream_archive.values()` loops. **This must ship as a Phase 1 *test*, not a Phase 4 cleanup** — cross-tenant leak is the bet-the-product failure mode.

**The "blackboard" is a view, not a write target.** "Recall from blackboard" = read dream-compiled state (identity anchors + semantic principles + Critic-approved claim views). There's no separate write target — promotion happens because the dream cycle eats the claim and emits derived structure. The earlier `blackboard/published/` write channel collapses to a *filter* over dream output.

**Director invokes existing dream signals.** Dream's `information_gap` detection becomes an auto-Task-creation trigger (assigned to Researcher). `reflective_heuristic` becomes a proposal stream Director can surface as Skill Crystallization. `stance_arc` between two agents' claims auto-elevates the artifact to a multi-reviewer Committee review.

### 3.4.1 TCMM scoping — typed channels (within the dream-backed substrate)

```
TCMM namespace tree (after Phase 2):

tenant/<tid>/
  user/<uid>/
    conv/<cid>/                  conversation memory (existing)
    agent/<aid>/                 private episodic per agent
      observations/
      reflections/
    team/<tid>/
      events/                    append-only log of what happened
      knowledge/                  curated facts the team agrees on (gated)
      drafts/                    WIP scratch (free-write, low recall weight)
    blackboard/                  org-wide published artifacts
      published/                 promoted by Critic
      traces/                    selected agent-run logs
```

**Why the team/ channel is typed.** A single shared write-anywhere channel becomes a junk drawer; recall quality collapses. Three subchannels with explicit semantics avoid this:

- `team/events/` — **append-only**. Any team member writes. Status transitions, task lifecycle events, "I just published X." Recall weighting: low (this is for accountability and short-term context, not knowledge retrieval). Auto-decayed after N days.
- `team/knowledge/` — **Critic-gated writes**. The team's curated facts ("our threat model assumes X", "we've decided to standardize on Y library"). Functions like a mini-blackboard scoped to the team. High recall weight.
- `team/drafts/` — **free write, low recall weight**. WIP scratch — Researcher sketching an outline, Builder noting a half-finished approach. Recall returns these only when explicitly asked for; doesn't pollute default recall.

**Write rules summary:**
- `agent/<aid>/`: only this agent writes and recalls. Personal notebook.
- `team/events/`: any team member writes; team members recall (low weight).
- `team/knowledge/`: Critic-gated; team members recall (high weight).
- `team/drafts/`: any team member writes; recall on explicit request only.
- `blackboard/`: Critic-gated; org-wide recall.

This mirrors A2A's "no direct memory access; share via task artifacts" discipline. Memory pollution is bounded because Critic gates both `team/knowledge/` and the blackboard.

**`author` field** on every block: `user` | `agent:<id>` | `system`. Recall returns include author; renderer can weight or filter.

#### 3.4.2 Naming conventions (anchor — referenced throughout doc)

Two parallel identifier systems exist; do not mix them:

| Concept | Storage namespace (slash form) | Enum / API identifier (underscore form) |
|---|---|---|
| Team curated knowledge | `team/knowledge/` | `team_knowledge` |
| Org-wide published artifacts | `blackboard/` | `org_blackboard` |
| User-scoped deliverables | n/a (per-user; written to recipient's namespace) | `user_deliverable` |
| Team event log | `team/events/` | `team_events` |
| Team WIP scratch | `team/drafts/` | `team_drafts` |
| Per-agent private | `agent/<aid>/` | `agent_private` |

**Rule:** the slash form is for paths and namespace strings. The underscore form is for enum values, function arguments, and JSON serialization. The submission-target enum (used by `submit_for_review(target=...)`) is `{team_knowledge, user_deliverable, org_blackboard}`. Any occurrence of `team/knowledge|user_deliverable` in spec text is an error — should be `team_knowledge|user_deliverable`.

### 3.5 Workspace

**Per-agent dir** on the VM at `/home/rudol/veilguard/agents/<aid>/`:
- `MEMORY.md` — curated long-term notes
- `SCRATCHPAD.md` — current-task TODO checklist
- `daily/YYYY-MM-DD.md` — append-only run log (Devin-style wake-up)
- `artifacts/` — non-text outputs

**Team-shared dir** at `/home/rudol/veilguard/agents/team/<tid>/`:
- `briefs/` — incoming work briefs from Director
- `drafts/` — in-progress shared documents
- `published/` — Critic-approved, blackboard-mirrored

**Document collaboration (canonical paths, no worktrees):** when two ICs work on the same draft, the **Task that owns the document is the merge authority** — the Task whose `outputs[]` first listed the path. Other Tasks editing the same path are auto-converted to subtasks of the owner Task at `attach_output()` time. Critic reviews the owner Task; other Tasks open subtasks for changes. Drafts are versioned via append-only `.history/` sidecars in canonical paths, OR via native git when the directory is under repo control. (Worktrees are explicitly out per `feedback_no_worktrees`.)

**Atomic writes:** `attach_output()` writes to a temp file then atomically renames. `outputs[]` is only appended once the write completes. Director's `final_synthesis` requires every referenced subtask to be `status=done`, not `in_progress`/`review` — explicit assertion, not implicit.

**Client-daemon as a tool, not a home.** The org lives on the VM. When Builder needs the user's filesystem or browser, it calls a client-daemon tool — gated by the approval system (§3.8).

### 3.6 Interaction patterns

Four canonical flows. The Director picks the pattern based on the inbound request.

#### Pattern A — Solo (default)
```
User → Director: "what's our exposure to CVE-2026-...?"
Director: handles directly, no delegation. Returns.
```
Costs one agent turn. Most interactions are this.

#### Pattern B — Single delegation (sync, blocking)
```
User → Director: "write a script that does X"
Director.assign(Task) → Builder.inbox
Builder picks up, executes, returns deliverable
Director reviews, returns to user
```
Sync from the user's perspective. Director awaits Builder's deliverable before responding.

#### Pattern C — Parallel fanout
```
User → Director: "research and prototype Y"
Director.assign(Task_research) → Researcher.inbox
Director.assign(Task_prototype) → Builder.inbox  (depends on Task_research deliverable)
Researcher completes → posts to team channel
Builder picks up Researcher's findings, builds prototype
Critic reviews prototype before publish
Director consolidates, returns to user
```
Async between ICs, sync from user's perspective. Cache cost amortized because each IC uses cache-stable prefix.

#### Pattern D — Background ongoing task
```
Director: "monitor the threat feed nightly"
Director.assign(Task, schedule=cron) → Researcher.inbox  (recurring)
Researcher runs nightly, publishes findings to team channel
Critic reviews; published items go to blackboard
User reads via sidebar Daemons tab
```
Async from user's perspective. The scheduled-daemon mechanism that already exists.

**No interactive debate.** Critic reviews are async PR-style — the IC submits a deliverable, Critic returns approve/changes-requested with notes, IC iterates. No turn-based dialogue. No fixed round count. Critic can decline review (out of scope, lacks context) — that's a status transition, not a dialogue.

### 3.7 Dream-as-scheduler — the proactive stream

The system has two operational streams:

- **Reactive** (§3.6): user request → Director decomposes → Tasks → consolidation → response
- **Proactive** (this section): dream signal → proposal → Director pre-evaluates → user approves → Task → IC work → consolidation back into dream

The proactive stream is what turns Veilguard from "a tool the user reaches for" into "a system that surfaces work the user didn't know to ask for." It's also the resolution of the earlier "should Director be an agent" question — under the proactive stream, judgment about which dream signals are worth surfacing is genuinely LLM-shaped work that pure routing wouldn't make.

#### 3.7.0 Hook point + scheduler — net-new infrastructure

**Proposal emission hook:** wrap the END of `run_cycle()` at `dream_engine.py:6312-6537`, after the final reflective stage at line 12082+. (Do NOT wrap `run_dream_cycle()` at line 5993 — that's an inner method which returns BEFORE reflective signals exist, so proposals would see a half-populated graph.)

**Scheduler:** no scheduler exists today. Dream runs on-demand via `POST /dream`. Full cycles take minutes (Phase-B parallel block alone ~9-10 min fresh-cache). Phase 3 deliverables:
- Scheduler (cron or background thread) triggering `run_cycle()` at the chosen cadence
- `run_proposal_pass()` mini-cycle emitting proposals against the most recent fully-committed graph state WITHOUT a fresh dream cycle (for hourly proposal cadence vs daily full cycles)

Both are net-new infrastructure, not wiring of existing code.

#### 3.7.1 Generative signal taxonomy

Not every dream primitive should propose a Task. The split:

**Generative signals — 5 confirmed to exist in dream today:**

| Signal | Default Task shape | Default assignee |
|---|---|---|
| `information_gap` | Fill the gap with research | Researcher |
| `contradiction_arc` | Investigate / adjudicate the conflict | Critic or Researcher (per source_kind mix) |
| `reflective_heuristic` | Review the detected pattern; decide if skillify | Critic (prose) |
| `recurring_ritual` | Codify procedure into a skill | Critic (prose) |
| `stance_arc` (between agents) | Multi-reviewer committee review | Critic (claim) + relevant consultant |

**Generative signals — 2 require new emitter implementation in dream itself (Phase 3 task):**

| Signal | Status | Implementation |
|---|---|---|
| `low_stability_cluster` | Not emitted as a node type today (only `_reason="low_stability_low_rc"` strings in persistence.py) | New emitter stage in dream_engine.py |
| `stale_supersession_chain` | Supersession state exists per-claim, no aggregator emits chain-level signals | New aggregator stage |

Until these emitters ship, proposal generation runs on 5 signal types, not 7.

**Non-generative** (shape recall, don't propose action): `causal_arc`, `semantic_principle`, `narrative_arc`, `identity_*`, `motif_node`, `belief_attribution`, `concept_node`.

#### 3.7.2 Impact scoring

**Multiplicative composition** (any single weak factor zeroes out the candidate; act on things that matter to real work, not on dream's most interesting findings in the abstract):

```
final_score = signal_impact × objective_alignment × constraint_gate

  constraint_gate = 0 if any constitution constraint is violated, else 1
  objective_alignment = dot(signal_default_alignment_vector, constitution.objectives)
                        for top-3 path (no LLM)
                      = Director's LLM rank-pass output × constitution.objectives
                        for ranks 4-10
```

**Per-signal `signal_impact` formulas:**

```
information_gap:        gap_breadth × downstream_pressure
contradiction_arc:      source_severity × claim_centrality
                        (source_severity: USER×USER=10, USER×INFERRED=5,
                         AGENT×AGENT=3, AGENT×INFERRED=1)
                        (claim_centrality = bridge_score)
reflective_heuristic:   recurrence × success_rate × token_savings_potential
recurring_ritual:       same as above (procedure flavor)
stance_arc:             polarity_distance × claim_stake
low_stability:          failure_count × cluster_recall_frequency
stale_chain:            age_days × recall_count × topic_currency_index
```

**Default alignment vectors per signal type** (static config; gives top-3 candidates a constitution-aware score without an LLM call):

```
information_gap:        {reduce_toil: 0.5, improve_security: 0.3, preserve_user_agency: 0.2}
contradiction_arc:      {improve_security: 0.5, preserve_user_agency: 0.4, reduce_toil: 0.1}
reflective_heuristic:   {reduce_toil: 0.7, improve_security: 0.2, preserve_user_agency: 0.1}
recurring_ritual:       {reduce_toil: 0.8, ...}
stance_arc:             {preserve_user_agency: 0.5, improve_security: 0.4, ...}
low_stability:          {improve_security: 0.6, preserve_user_agency: 0.3, ...}
stale_chain:            {improve_security: 0.5, reduce_toil: 0.4, ...}
```

These are seed values; Director's LLM rank-pass refines them per-candidate for ranks 4-10.

All weights are pure-function and unit-testable. Recalibrated weekly via outcome tracker (§3.7.7) using `regret_score` (not just approval_rate) — signal types whose approvals tend to regret get downweighted faster than signal types whose approvals tend to reuse.

#### 3.7.3 Budget mechanics

Three stacking caps:

- **Per-cycle:** top 10 candidates by impact_score after de-dup
- **Per-signal-type cap:** max 3 of any one signal type per cycle (prevents single-signal floods)
- **Per-day approval cap:** Director approves max 20 proposals/day → Tasks; tunable per tenant

**User-driven Tasks bypass all caps.** User authority is supreme; the proposal queue is additional work, not replacement.

**Emergency lane:** USER×USER `contradiction_arc` skips caps and surfaces immediately to the user, not via Director's pre-eval. Critical signals shouldn't queue behind low-impact info_gaps.

#### 3.7.4 Dedup, recurrence, carry-over

Candidate keyed on `(signal_type, sorted(signal_node_ids))`. Same key re-triggering increments `recurrence_count` and refreshes `last_surfaced_ts`; no duplicate row created.

- **Decay:** deferred candidates' impact decays 0.9× per cycle until either re-triggered (impact recomputed fresh) or below floor → auto-shelved with reason `aged_unaddressed`.
- **TTL:** 7 days. Stale candidates auto-expire with reason `expired_ttl`.
- **Recurrence escalation:** if a candidate re-surfaces 5+ times without approval, Director presents it as a forced choice — approve, shelve-with-reason, or recalibrate scoring. Prevents silent infinite loops.

#### 3.7.5 Pre-evaluation flow — deterministic top-3 + single LLM rank pass

```
Dream cycle (every N hours)
       │
       ▼
Generate candidates → score with §3.7.2 formula
                      (signal_impact × default_alignment × constraint_gate)
                      de-dup, per-type-cap of 3
       │
       ▼
Sort by final_score, take top 10
       │
       ├─── Top 3 (deterministic, no LLM):
       │        - Template-generated brief per signal_type
       │        - Default assignee from signal_type → role mapping
       │        - Surfaced directly
       │
       └─── Ranks 4-10 (ONE Haiku rank pass, not per-candidate):
                - Single structured-output call ranks all 7
                - Returns: top 2-3 worth surfacing, with refined briefs +
                  rationale + per-candidate objective_alignment refinements
                - Below cutoff: deferred (decay applies)
       │
       ▼
Surface ~3-6 candidates per cycle to sidebar "Proposed Tasks" tab:
   - signal type + final score
   - brief (template OR Haiku-drafted)
   - suggested assignee
   - 1-line rationale + which objective(s) this serves
   - recurrence count
   - quick actions: [Approve] [Defer] [Shelve]
       │
       ▼
User reviews asynchronously (no chat turn needed)
       │
       ▼
Approved → real Task created in agent_tasks
           proposal.status=approved
           proposal.resulting_task_id = new Task's id
```

**Cost:** 1 Haiku call per cycle (rank pass over 7 candidates) instead of 10 calls. ~$0.001/cycle, ~70% reduction. Less cognition stacked.

**Brief templates per signal_type** are static config:
```
information_gap:    "Research {gap_topic}. Cite sources. ≤500 words. Target: {team_knowledge|user_deliverable}."
contradiction_arc:  "Investigate conflict between {claim_a_text} and {claim_b_text}. Determine which supersedes; cite evidence."
recurring_ritual:   "Codify procedure {pattern_id} as a skill. Trigger conditions + steps + 2 examples."
reflective_heuristic: "Review pattern: {pattern_desc}. Propose org-memory lesson or skill."
stance_arc:         "Multi-reviewer review: agents {agent_a} and {agent_b} disagree on {claim_text}."
low_stability:      "Re-investigate cluster: {cluster_desc} ({fail_count} failed claims)."
stale_chain:        "Refresh assumptions: {chain_topic} last touched {days_ago}d ago."
```

The pre-evaluation keeps Director-as-agent (judgment on the marginal candidates is genuinely LLM work) while removing the per-candidate LLM cost on the obvious top-3 and the not-worth-surfacing tail.

#### 3.7.6 Data model

New Lance table `task_proposals` — **separate from `agent_tasks`** (different lifecycle, schema, access pattern; conflating would fill `agent_tasks` with dead-letter rows):

```
proposal:
  id: uuid
  signal_type: enum
  signal_node_ids: [int]              # dream graph nodes that triggered
  impact_score: float
  decay_score: float                  # impact × 0.9^cycles_since_surfaced
  proposed_brief: str                 # Director's LLM draft
  proposed_assignee: agent_id
  proposed_deliverable_spec: str
  rationale: str                      # one-line "why this matters"
  recurrence_count: int
  first_surfaced_ts: float
  last_surfaced_ts: float
  status: pending | approved | deferred | shelved | expired
  director_decision_ts: nullable
  shelf_reason: nullable
  resulting_task_id: nullable         # if approved
```

Indexed on `status`, `signal_type`, `(status, decay_score DESC)` for the queue view.

#### 3.7.7 Outcome tracker (self-calibration via regret)

After each Task spawned from a proposal completes, log to `proposal_outcomes`:

```
outcome:
  proposal_id, resulting_task_id, task_status (done | cancelled | failed)
  task_cost                    # rolled up from pii_audit via trace_ref
  value_realized               # recall_count on task outputs + downstream tasks
                               # that consumed them (settles at 30d)
  regret_score                 # v1: task_cost / max(value_realized, ε)
                               # (computed at 30d after completion)
  objective_deltas             # per-objective measured delta from constitution.metrics
  computed_at_ts: float
```

**Weekly recalibration** uses `regret_score` and `objective_deltas`, not just approval rate. Signal types whose approvals tend to **regret** get downweighted faster than signal types whose approvals tend to **reuse**. The default alignment vectors in §3.7.2 also get adjusted: if `information_gap` proposals consistently produce work that improves measurable security (per `improve_security` metric), the default alignment vector for `information_gap` shifts toward security.

**Org-memory feed-in:** high-regret approvals + recurring failure patterns are the evidence source for org-memory lesson proposals (§3.9.2). "Critic-skip publishes had high regret 7 times → propose lesson: 'require Critic'."

---

### 3.8 Approval gate

The choke point already exists: `handle_tool()` at [agentic.py:122](C:/Users/rudol/Documents/veilguard/services/sub-agents/core/agentic.py:122). Both foreground and background calls land here. Origin tag also exists: `get_child_conversation_id()` at [request_ctx.py:36](C:/Users/rudol/Documents/veilguard/services/sub-agents/core/request_ctx.py) is empty for foreground, non-empty (cid `sub-<parent7>-<kind3>-<uuid8>`) for background.

```
LLM tool_use block
       │
       ▼
handle_tool(name, args)               [agentic.py:122]
       │
       ▼
is_client_tool(name)?                  no  → handle locally
       │ yes
       ▼
get_child_conversation_id()
       │
       ├─ "" (foreground/Director)    → bridge.execute_remote()
       │
       └─ "sub-..." (any IC or
              background)
              │
              ▼
       classify(name, args, agent_id)  [new: utils/client_tool_policy.py]
              │
              ├─ ALLOW    → bridge.execute_remote()
              ├─ DENY     → return error (audit)
              └─ APPROVE  → bridge.request_approval()
                              │
                              ▼
                       Windows toast on user's machine
                              │
                       [Approve] [Deny] [Always allow]
                              │
                              ▼
                       Decision logged to `client_tool_approvals`
                              │
                              ▼
                       ALLOW → execute_remote()
                       DENY  → return error
```

**Capability matrix** (default policy for non-foreground/non-Director calls):

| Tool | Director (foreground) | Builder (delegated) | Other ICs |
|---|---|---|---|
| `read_file`, `search_files`, `grep` | ALLOW | ALLOW (path allow-list) | ALLOW (read-only) |
| `host_file_read` | ALLOW | APPROVE | APPROVE |
| `write_file`, `edit_file` | ALLOW | APPROVE | DENY |
| `host_file_write` | ALLOW | DENY | DENY |
| `run_command`, `run_cmd`, `run_powershell` | ALLOW | APPROVE (every call) | DENY |
| `run_docker`, `run_git` | ALLOW | APPROVE | DENY |

The Director gets foreground privileges because the user is in the chat. ICs get less authority because they may be running async. Researcher / Critic don't need shell at all — denied. Builder has shell but always gated.

**Approval surface = Windows toast from the client-daemon.** The threat is on the user's box, the user may not have LibreChat open, the daemon already has notification permission. Sidebar Daemons tab gets a "Pending approvals" badge as a mirror, not the primary surface.

**Bypass rules** stored in `client_tool_bypass` Lance table. Keyed on `(user_id, agent_id, tool, arg_glob)`, optional `expires_at`. Audit in `client_tool_approvals`.

**Arg sanitization at the policy layer** (corrects an edge case from audit): reject any tool arg containing control chars (0x00-0x1f except tab/newline) BEFORE glob matching. Memory `workflow_env_crlf_gotcha` documents two CRLF incidents on .env values; same shape would bite here as bypass-glob evasion (`run_command("ls\r\n; rm -rf /")` matches `ls*` if not normalized). Add a unit test asserting CRLF args trigger DENY even with a matching bypass rule.

**Daemon capability handshake** (Phase 0.0 deliverable per §0.4): daemon WS handshake includes `supported_features: [...]` on connect. If `request_approval` not in features list, gate falls back to **DENY for background-origin calls** with sidebar warning "Daemon vN does not support approval gate; update recommended." Foreground (Director) calls bypass the gate as today. Fail-closed default prevents the deployment-ordering hazard where sub-agents ships approval logic before user daemons auto-update (~30-min poll window).

**Approval timeout**: 120s default, configurable per tool (`run_command` = 120s, `write_file` = 300s). After timeout: decision auto-DENY with reason `approval_timeout`, logged to `client_tool_approvals`. The agent's `execute_remote` future resolves with the deny error; agent sees it as a normal tool failure and either fails the Task gracefully or raises `blocker_raised`. Without a timeout, an offline daemon would wedge background work indefinitely.

---

### 3.8.5 Trust-boundary discipline (server-of-record per field)

Adversarial-review finding: the spec correctly identifies the choke points (approval gate, observe, lineage_chain, task_id, comments) but earlier drafts didn't lock down *who* is authoritative for the bytes on each side of those choke points. **Every "client supplies field X" is an authority leak.** This subsection makes the trust boundaries explicit.

**Rule:** for every field on every cross-trust-boundary write, declare who is server-of-record and reject any client supplying it.

| Field | Cross-boundary write | Server-of-record | Reject if client supplies |
|---|---|---|---|
| `task_id` (on `pii_audit` rows) | Agent LLM call → pii-proxy | pii-proxy, derived from `cid → agent_tasks` join | YES — `x-task-id` header is ignored if present |
| `lineage_chain` | Any decision-ledger insert | Server (computed as `parent.lineage_chain + [parent.id]` at insert time) | YES — client-supplied lineage_chain is rejected with 400 |
| `source_kind=USER` | observe() | pii-proxy (verifies `x-user-id` header matches authenticated session) | YES — agent observes with `source_kind=USER` are rejected |
| `source_kind=USER_PARAPHRASE` (new enum value) | observe() | sub-agents service (set when an agent observes a paraphrased user statement) | n/a — internally set |
| Approval-bound `arg` bytes | `request_approval` → execute_remote | Server-issued `approval_token` binds tool + arg_hash + nonce; daemon re-verifies hash at execute time | TOCTOU defense: client cannot substitute arg between approval and execution |
| `approval_id` | client-daemon WS reply | Server-generated HMAC(vm_secret, agent_id ‖ tool ‖ arg_hash ‖ nonce) | YES — client cannot pre-issue ids |
| `created_at` on `client_tool_approvals` | daemon offline → online replay | Server stamps `received_at_vm` (canonical); daemon's `decided_at_local` recorded separately; >5min drift flags row | n/a — daemon-supplied timestamp is informational only |
| `comments[]` on Tasks | IC adds comment | **Stored in separate append-only `task_comments` Lance table with `prev_hash` chaining**; Task row holds head hash only. Mutating the chain = audit alarm. | n/a — chain is append-only by construction |
| `reinforcement_count` on lessons | dream cycle promotes a reflective_heuristic | Server, requires evidence from ≥2 distinct `extracted_by` values (cross-agent or USER); single-agent reinforcement logged but does NOT advance the counter | YES — direct increment requests rejected |
| `extracted_by` (`agent:<aid>`) on typed_claims | observe() | pii-proxy validates `x-agent-id` header against the per-tenant agent registry before threading into `extracted_by` | YES — unknown agent_id → treated as user-direct (no agent provenance) |

**`source_kind=USER_PARAPHRASE` — 7th enum value (added in Phase 0.2):**

Without this distinction, an agent that summarizes a user statement back into conversation can manufacture a "USER" claim that contradicts the original, triggering the emergency lane (§3.7.3) on manipulated content.

Rules:
- `source_kind=USER` requires verbatim observation through the user-conv pipeline with `x-user-id` validated.
- Agent-summarized user statements get `source_kind=USER_PARAPHRASE`.
- USER×USER emergency lane requires BOTH claims to be verbatim `source_kind=USER`. USER×USER_PARAPHRASE goes through normal proposal ranking.

**Why this discipline matters:** without server-of-record fields, the audit log is repudiable, the approval gate is TOCTOU-vulnerable, lineage_chain replay is forgeable, and the emergency lane is exploitable. Each of these closes a class of attack, not just one specific bug.

---

### 3.9 Strategy layer — the compass above the proactive stream

The dream-as-scheduler loop (§3.7) is a closed adaptive system: dream proposes → Director ranks → user approves → Task runs → outcome tracker recalibrates weights. Without an external compass, it self-tunes toward "what got approved" — a proxy goal that drifts to local optima and status-quo bias. The strategy layer is what gives the loop direction.

Three components, top-down:

#### 3.9.1 Constitution — user-authored direction

Top-level direction. Lives at `agents/CONSTITUTION.md` (markdown + frontmatter). Approximately 10 entries total, deliberately small.

```
constitution:
  objectives:                  # Weighted; sum to 1.0
    - id: reduce_toil
      weight: 0.40
      description: "Eliminate repetitive work for the user."
    - id: improve_security
      weight: 0.30
      description: "Strengthen the user's security posture."
    - id: preserve_user_agency
      weight: 0.30
      description: "Surface decisions, don't make them silently."

  constraints:                 # Boolean gates — no weight-buying
    - no_hidden_automation:
        rule: "Tasks above $0.50 budget must surface to user before execution."
    - cost_ceiling_per_task:
        rule: "No single Task may exceed $5 without explicit user approval."
    - preserve_provenance:
        rule: "Published claims require source_kind ≠ INFERRED unless Critic-promoted."
    - no_autonomous_client_daemon_access:
        rule: "Background-origin client_daemon tool calls require approval per §3.8."

  metrics:                     # How outcomes get scored
    - time_saved: minutes_per_week_saved_via_skills
    - knowledge_reuse: recall_count_per_published_claim
    - regret: weighted_avg_regret_score_per_proposal
```

**Rules of the road:**
- **User-authored.** System can *propose amendments* via reflective_heuristic; adoption requires user approval. Auto-adopted policy is exactly what §8 Non-goals forbids.
- **Objectives** are weighted vectors (sum to 1.0). Steer proposal ranking via `objective_alignment` factor.
- **Constraints** are categorical vetoes. No proposal may violate a constraint regardless of objective alignment — boolean gates, not soft preferences.
- **Metrics** define how outcomes get scored for recalibration. Each metric must be measurable from existing tables (`pii_audit`, `agent_tasks`, dream graph).

#### 3.9.2 Org Memory — system-proposed, user-approved lessons

Institutional lessons. Distinct from:
- **Skills** (how-tos — `services/sub-agents/skills/`)
- **Blackboard** (factual knowledge — dream's derived primitives)
- **Constitution** (top-level steering — ~10 entries)

Org Memory entries are **rules about how the organization operates**, learned from experience.

```
lesson:
  id: lesson_<uuid>
  trigger: "prototypes published without Critic review failed N times"
  rule: "prototypes require Critic before publish"
  confidence: 0.81                      # from underlying reflective_heuristic
  evidence: [task_ids ...]              # tasks that contributed
  promoted_from: reflective_heuristic_node_id
  status: proposed | accepted | retired | expired
  reinforcement_count: 0                # bumps each time evidence accumulates
  last_reinforced_ts: float             # resets confidence decay timer
  created_ts, updated_ts: float

  # Lifecycle — prevents organizational fossilization
  expires_at: float                     # default = created_ts + 180d
  review_after: float                   # default = created_ts + 90d (soft review trigger)
  confidence_decay_per_week: 0.02       # active erosion if not reinforced
  reviews: [                            # history of explicit reviews
    { ts, decision: keep | amend | retire, reviewed_by: user | critic, note }
  ]
```

**Decay mechanics:** confidence erodes linearly at `confidence_decay_per_week` until either (a) a new reflective_heuristic reinforces the lesson (resets `last_reinforced_ts`, restores confidence) or (b) a scheduled review by user/Critic decides `keep | amend | retire`. At `expires_at`, status auto-transitions to `expired` with reason `unreinforced`. Expired lessons are kept for audit but no longer influence proposal scoring.

Why this matters: without expiry, lessons become organizational scar tissue. A rule learned during one phase ossifies and steers proposals long after its premises have shifted. Real-org dysfunctional bureaucracy emerges exactly this way; explicit decay + scheduled review is the architectural answer.

Storage cap: ~100 entries. Curated, not accumulated. New Lance table `org_memory` indexed on `(status, confidence DESC)`.

**Promotion pipeline:** `reflective_heuristic` → org-memory candidate (surfaced via the proactive stream, like any proposal) → user approval → accepted entry. After N reinforcements *and* a user-initiated review, a high-confidence accepted lesson can be proposed as a **constitution amendment** (new objective, new constraint, or weight adjustment).

This makes the system's evolution **traceable**: every rule it operates by has a story (evidence chain) attached.

#### 3.9.3 Regret as a first-class primitive

The outcome tracker (§3.7.7) gains a per-proposal `regret_score`. Two regret types — very different cost profiles:

**v1 — `accepted_but_low_value` (ships Phase 3):**
- Proposal approved → Task ran → cost N → outputs not recalled within 30d window
- `regret_score = task_cost / max(value_realized, ε)` where `value_realized` = recall_count on the task's outputs + downstream tasks that consumed them
- Easy: all data already in `agent_tasks` + dream recall logs

**v2 — `rejected_but_should_have_done` (Phase 4+):**
- When a contradiction explodes / user expresses frustration / a task fails, search recently-shelved proposals for ones that could have prevented it
- Requires graph traversal across `causal_arc` + `claim_id` connectivity
- Hard problem; not blocking on v1

Weekly recalibration uses `regret_score`, not just approval_rate. Signal types whose approvals tend to regret get downweighted faster than signal types whose approvals tend to reuse. Regret data is also the **evidence source for org-memory promotion**: "Critic-skip publishes had high regret 7 times → proposed lesson: 'require Critic'."

#### 3.9.4 Composed flow — the full loop with compass

```
                            ┌─────────────────────┐
                            │   Constitution      │  user-authored
                            │   (objectives,      │  ~10 entries
                            │    constraints,     │
                            │    metrics)         │
                            └──────────┬──────────┘
                                       │ steers
                                       ▼
Dream signals ─────► Score: signal_impact × objective_alignment
                            gated by: constraints (boolean veto)
       ▲                              │
       │                              ▼
       │                       Top 3: deterministic, template brief
       │                       Ranks 4-10: ONE Haiku rank pass
       │                              │
       │                              ▼
       │                       Surface to user
       │                              │
       │                              ▼
       │                       Approved → Task → IC work
       │                              │
       │                              ▼
       │                       Outcome + regret scored against metrics
       │                              │
       │                              ▼
       │                       Weekly recalibration of weights
       │                              │
       │                              ▼
       │                       Reflective_heuristic on patterns
       │                              │
       │                              ▼
       │                       Org Memory candidate
       │                              │
       │                              ▼ user-approved
       │                       Org Memory entry
       │                              │
       │                              ▼ after N reinforcements + user review
       └────────── feedback ─── Constitution amendment proposal
                                      │
                                      ▼ user-approved
                              Updated Constitution
```

The compass at the top. The proactive stream in the middle. Outcome → regret → reflection → org memory → constitution amendment as the long arc of organizational learning. Every layer is user-gated; nothing is auto-adopted.

---

### 3.10 The decision ledger — unified framing across §3.3, §3.7, §3.9

Step back from the per-table view. Everything we've designed — Tasks (§3.3), proposals (§3.7), outcomes (§3.7.7), lessons (§3.9.2) — moves through a single lifecycle skeleton:

```
proposed → accepted → in_progress → review → done | cancelled | retired | institutionalized
```

That's not coincidence; it's the structural fact that Veilguard's data model is **a decision ledger**, not a task system. Every row is a decision made (or pending) by some actor (user, Director, Critic, dream-cycle worker, IC).

#### 3.10.1 Shared lifecycle skeleton

Every decision-ledger entity carries:

```
shared_skeleton:
  id: uuid
  kind: enum                    # proposal | task | lesson | outcome (extensible)
  status: enum                  # see unified status enum below
  parent_id: nullable           # subordination / decomposition
  lineage_chain: [uuid]         # ancestor chain for replay + audit
  created_ts, updated_ts: float
  created_by_agent_id: nullable # which agent or user originated
  cost_attributed: float        # accumulated USD on this decision's behalf
```

**Unified status enum** (across all kinds; not every kind uses every state):

```
proposed | accepted | in_progress | blocked | review |
done | cancelled | retired | institutionalized | expired
```

#### 3.10.2 Physical vs logical — Option B (separate tables, UNION view)

We keep separate Lance tables (`agent_tasks`, `task_proposals`, `proposal_outcomes`, `org_memory`). Reasons:

- **Write-path hygiene** — different services own different kinds. Dream-cycle writes proposals (~10/cycle, high churn). Agent loop writes tasks (low throughput, long-lived, growing comment lists). Tracker job writes outcomes (write-once-per-completion). User-approval handler writes lessons (~100 total, slow-mutating). Mixing these in one table creates contention and fragmentation on slow rows.
- **Schema width** — `outputs`/`comments` (tasks), `signal_node_ids`/`recurrence` (proposals), `evidence`/`rule`/`confidence` (lessons), `regret_score`/`value_realized` (outcomes) become mostly-NULL columns under full collapse. Lance handles sparse schemas, but pa.schema bloat hurts scan latency.
- **Lance fragmentation is the recurring perf killer** (per existing memory). Highest-churn kind sets fragmentation cadence for the whole table if collapsed.

**Logical view is unified** via `work_items_v`:

```sql
work_items_v =
  SELECT id, kind='proposal', status, parent_id, lineage_chain,
         created_ts, updated_ts, created_by_agent_id, cost_attributed
    FROM task_proposals
  UNION ALL
  SELECT id, kind='task', status, parent_id, lineage_chain,
         created_ts, updated_ts, created_by_agent_id, cost_attributed
    FROM agent_tasks
  UNION ALL
  SELECT id, kind='lesson', status, parent_id, lineage_chain,
         created_ts, updated_ts, created_by_agent_id, cost_attributed
    FROM org_memory
  UNION ALL
  SELECT id, kind='outcome', status, parent_id, lineage_chain,
         created_ts, updated_ts, created_by_agent_id, cost_attributed
    FROM proposal_outcomes;
```

This gives lineage-across-kinds queries (`WHERE topic LIKE ... ORDER BY created_ts`) and unified audit + cost-roll-up without the physical-table costs. Kind-specific queries hit the underlying tables directly for performance.

**Caching is mandatory** (audit feedback). A naive UNION ALL across 4 Lance tables means 4 full scans per refresh; per memory `workflow_admin_dashboard_caching`, even single-table pii_audit scans were the dashboard's perf killer at ~480ms. The sidebar refreshes the decision-ledger view often.

**Required caching pattern** (Phase 4 ship requirement, not deferrable):
- The view is computed lazily (no materialization) for kind-specific filter queries — those hit the underlying table directly.
- For the unified sidebar feed, materialize an **incremental projection** `work_items_recent` containing only rows from last 30d (covers all displayed lifecycle states).
- Refresh on each underlying-table write via the existing `_windowed_rows()` 8s TTL cache pattern from the admin dashboard.
- Per-tenant cache key (no cross-tenant invalidation).

Without this, dashboard P99 climbs past 5s by month 3 and "FTS INDEX: pending" reappears across the new tables.

**Maintenance cron extension (mandatory):** the existing weekly `optimize_indices()` + `compact_files()` cron must be extended to cover the four new Lance tables from day 1. Memory `architecture_lance_maintenance` documents that `task_proposals` will have the worst fragmentation profile (~10 writes/cycle, high decay/recurrence updates). Same anti-pattern that produced 813 fragments on `sparse_archive` in 2 days will hit `task_proposals` in week 1 without the cron extension.

#### 3.10.3 When to collapse to Option A instead

Switch to a single `work_items` Lance table only if you find yourself wanting **atomic cross-kind writes** — e.g., "approve this proposal AND create the resulting task AND close the proposal" in one transaction. UNION-ALL view can't do that.

Until that pattern shows up, Option B gives most of the framing benefit at lower physical cost. Option A remains reachable later via a one-time migration that backfills `work_items` from the four extant tables and switches reads/writes over.

#### 3.10.4 What this changes about the UI

The sidebar surface is conceptually **the decision ledger**, not four separate panels:

- **Pending decisions** — proposals awaiting user review + lessons due for review
- **Active work** — `in_progress` tasks across the org
- **Recent decisions** — `done` / `retired` / `institutionalized` from the last N days
- **Filter by kind** for power users

All on the existing Daemons-tab infrastructure. No LibreChat fork changes.

The deeper benefit: a user looking at "what is this system doing about topic X" gets a unified timeline across proposals→tasks→outcomes→lessons. That's the lineage view, free from the shared skeleton.

---

## 4. The org — concrete agents

### 4.1 Director (dual-stream orchestrator — deliberately thin)
- **Role:** `director` · **Manager:** none · **Team:** `core`
- **Job:** run TWO streams:
  - **Reactive** (user-in-chat): user request → decide solo vs delegated → decompose into Tasks → consolidate → respond. Direct `create_task` allowed; user is in the loop implicitly.
  - **Proactive** (dream-driven): consume `task_proposals` queue → pre-evaluate each candidate (Haiku, structured output) → publish recommendations to the sidebar → on user approval, convert proposal to Task. **Director does NOT create Tasks directly from dream signals — every proactive Task goes through user approval.**
- **Tools:**
  - Reactive: `create_task`, `assign_task`, `consult(consultant_id)`, `recall` (read-only across blackboard + owned tasks), `final_synthesis`
  - Proactive: `rank_proposals(candidate_ids)` → single LLM rank pass (Haiku) over ranks 4-10; `convert_proposal(proposal_id)` → creates Task on user approval; `shelve_proposal(proposal_id, reason)`
  - Strategy: `read_constitution()` (at startup + on file change); `surface_org_memory_candidate(reflective_heuristic_id)` when a reflective_heuristic pattern recurs
  - **No `observe`. No client-daemon tools. No web. No shell.**
- **Memory:** none of its own (no `observe`). Director's "memory" IS the task + proposal tables. **Recall scope:** `conv/<current_cid>/` (current conversation), `team/knowledge/` (Critic-promoted team facts), and `blackboard/` (org-wide). Read-only — Director writes no memory blocks. The conv-scoped recall is what makes Pattern A viable for "what did the user just tell you" follow-ups.
- **Workspace:** none. Director doesn't produce artifacts; ICs do.
- **Triggered by:**
  - Reactive: the user (default chat target)
  - Proactive: each dream cycle's proposal emission (~every N hours)
- **Why thin + dual-stream:** the proactive stream's pre-eval is genuinely LLM-shaped work — judgment about salience, ranking, brief drafting. Pure routing wouldn't make these calls. Without execution tools, observe, or memory, Director remains a judgment + routing layer, not an accidental monolith.
- **Surfacing model:** Director's outputs (proposals queued, status pings during active tasks, final synthesis responses) all flow into the **decision ledger** sidebar (§3.10) — pending decisions, active work, recent decisions, lessons due for review — not into separate UI panels. From the user's perspective, the org has one queue, kind-filterable.

### 4.2 Researcher (analyst IC)
- **Role:** `ic` · **Manager:** `director` · **Team:** `core`
- **Job:** open-ended investigation, web fanout, synthesis, cross-checking sources
- **Tools:** `web_search`, `fetch`, `recall` (org-wide), `observe` (private + team), `dm(agent_id)`, `submit_for_review`
- **Memory:** private + team + blackboard read
- **Workspace:** `agents/researcher/` + team
- **Triggered by:** Director task, peer DM, scheduled daemon

### 4.3 Builder (engineer IC)
- **Role:** `ic` · **Manager:** `director` · **Team:** `core`
- **Job:** writes code, runs tools, executes in workspace, builds prototypes
- **Tools:** `file_*`, `shell` (gated), `client_daemon.*` (gated), `recall`, `observe`, `dm`, `submit_for_review`
- **Memory:** private + team + blackboard read
- **Workspace:** `agents/builder/` + canonical team paths (NO worktrees — per §3.5; honors `feedback_no_worktrees` memory). Parallel edits resolved via "first Task whose outputs[] listed the path is merge authority."
- **Triggered by:** Director task only — **never auto-invoked by peers** (highest-privilege agent, kept under direct manager control)

### 4.4 Critic — two roles (split)

The Critic role is split into two agents because prose review (Sonnet/Opus, ~30s tolerable) and typed-claim validation (Haiku, inline, must run on every team-knowledge observe) are incompatible workloads. Bundling forces either over-spending or shallow review.

### 4.4a Critic (claim) — structural arbiter, inline, cheap

- **Role:** `ic` · **Manager:** `director` · **Team:** `core` · **Model:** Haiku
- **Job:** Validate typed_claims as they're submitted for promotion to `team/knowledge/` or higher. Runs inline (sync, ~1-3s per batch). Structural-only — does NOT make semantic judgments about claim correctness.
- **Validation surface:** structural via existing [validate_claim()](C:/Users/rudol/.gemini/antigravity/tcmm/TCMM/core/dream/typed_claims.py:685) + synthesis via [validate_synthesis()](C:/Users/rudol/.gemini/antigravity/tcmm/TCMM/core/dream/dream_validation.py:15) (repetition loops, overlap confidence ≥30%), then schema-conformance checks: source_kind correctness, polarity correctness, predicate_class correctness, no `source_kind=INFERRED` with higher confidence than its weakest source, citation presence on factual claims, bitemporal sanity (valid_from ≤ valid_until).
- **Tools:** `recall` (read-only, blackboard scope), `validate_claims(claim_ids)`, `flag(claim_id, reason)`. **No write tools, no shell, no memory observe.**
- **Triggered by:** every `submit_for_review` targeting `team_knowledge` or `org_blackboard`. SLA: <10s p99 (cheap path).
- **Decision output:** `pass` (claims structurally valid) / `fail(reasons)` (block promotion; return to IC).

### 4.4b Critic (prose) — semantic reviewer, async, expensive

- **Role:** `ic` · **Manager:** `director` · **Team:** `core` · **Model:** Sonnet (or Opus for high-stakes)
- **Job:** PR-style semantic review of artifacts. The *unit of review* is the artifact itself (markdown draft, code, etc.) plus the claims it produced. Decides whether work is fit for promotion to its declared target. Runs async; can take minutes.
- **Validation surface:** post-claim-Critic. By the time Critic (prose) sees an artifact, Critic (claim) has already passed it structurally. Prose Critic evaluates: scope adherence (does the artifact match the brief?), citation quality, conflict with blackboard knowledge, scope creep, drift from deliverable_spec.
- **Tools:** `recall` (org-wide), `score(artifact)`, `request_changes(notes)`, `approve(target)`, `decline(reason)`.
- **Triggered by:** Critic (claim) `pass` AND target is `org_blackboard` or `user_deliverable`. For `team_knowledge` target, Critic (claim) pass alone is sufficient (no prose review — keeps team-channel throughput high).
- **SLA:** 5 min foreground / 1 hr background before auto-decline with reason `critic_timeout`.
- **Memory:** read-everywhere, write-nowhere except own private namespace.
- **Workspace:** none.

**Asymmetric authority preserved:** neither Critic can write to anything except its own private namespace; together they gate every promotion from private → shared → blackboard. The split removes the single-Critic-replica bottleneck (Critic-claim is cheap and parallelizable; Critic-prose is the slower path but only runs on artifacts that already cleared the cheap gate).

**Failure modes mitigated:**
- Critic-claim DOWN → block all promotions (fail closed). User sees "Critic-claim unavailable; promotions queued."
- Critic-prose DOWN → `team_knowledge` promotions continue; `org_blackboard` and `user_deliverable` queue with visible backlog indicator.

### 4.5 Consultants (existing personas as on-demand specialists)
- `phishing-analyst`, `threat-analyst`, `report-writer` — kept as defined in `agents/*.md` today
- **Role:** `consultant` · **Manager:** none · **Team:** none
- **Triggered by:** Director's `consult(consultant_id, brief)` call or peer DM
- Consultants don't have an inbox in the same way — they're invoked per-request, deliver a response, and their findings go through the requester's submit-for-review flow
- Over time, consultants can be promoted to team ICs if usage warrants the org-chart commitment

**Asymmetric authority** summary: Director has user-facing privilege; Builder has the dangerous tools but only under direct managerial assignment; Critic has read-everywhere but write-nowhere; Researcher does the heavy recall work because its private namespace growing large is acceptable.

---

## 5. Phase plan

### Phase 0.0 — operational prerequisites (must ship FIRST)

Foundational discipline + missing scaffolding. Without these, Phases 0.1-0.3 inherit avoidable rework.

| # | Change | Where | Effort | Validates with |
|---|---|---|---|---|
| 0.0.1 | Source-of-truth + deploy-order documentation | This doc §0.1 (already landed) | trivial | Manual review |
| 0.0.2 | `agents/*.md` frontmatter parser + schema | sub-agents startup; new module | ~150 LOC | Existing 3 personas load with auto-default `role=consultant`; new agents validate `agent_id` uniqueness per tenant |
| 0.0.3 | CRLF/UTF-8 sanitization on config-load boundaries | `safe_load_yaml_frontmatter()` helper used by Director init + agent registry | ~50 LOC | Test fixture with CRLF + smart-quotes loads cleanly with logged warning |
| 0.0.4 | Daemon WS capability handshake | client-daemon `veilguard_client.py` + sub-agents bridge | ~80 LOC | Old daemon connects → sub-agents logs "missing feature: request_approval" + falls back to DENY for background |
| 0.0.5 | Lance maintenance cron extension to cover new tables | maintenance cron (`/home/rudol/veilguard/maintenance/`) | trivial | `optimize_indices()` runs cleanly against new tables before they ship |

### Phase 0.1 — cache-stable cid (cost prerequisite for everything else)

| # | Change | Where | Effort | Validates with |
|---|---|---|---|---|
| 0.1.1 | Cache-stable system prefix for sub-agent calls | `vm-pull/main_prod_.py` (pii-proxy) — **VM is source-of-truth; pull first** | ~80 LOC (was estimated 50; design constraint surfaced) | `pii_audit.cache_create` vs `cache_read` ratio before/after on a 1-day window |

**Design constraint surfaced by audit** (not in earlier drafts): Anthropic's prompt cache key is content-derived from the prefix bytes, NOT from the cid. So "cache-stable cid" is a misnomer — the actual requirement is **byte-identical system prefix across sibling sub-calls of the same parent**. The cid serves only as the *memoization key* on the pii-proxy side for the rendered prefix.

Concrete shape:
```python
# In pii-proxy, when incoming cid starts with 'sub-':
parent_cid = cid.split('-', 2)[1]
# Memoize on (parent_cid, tcmm_version) — NOT on the full sub-cid
system_blocks = render_cache.get((parent_cid, tcmm_version))
if system_blocks is None:
    system_blocks = tcmm.render(parent_cid)
    render_cache.set((parent_cid, tcmm_version), system_blocks)
system_blocks[-1]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
```

**Critical:** the per-call user message and the per-call tool result are NOT in the cached prefix — they vary per call. Only the system+tools prefix is byte-stable. If Director or any agent injects per-call context (task brief, current step) into the system prefix, the prefix de-stabilizes and the cache miss returns. Phase 0.1 must enforce a clean prefix/per-call boundary at the pii-proxy.

**Dual-read window for legacy cids:** in-flight conversations at deploy time will have old-format cids. For 1h after deploy, fall back to legacy behavior for cids that don't match new format. Otherwise the first message of every active chat post-deploy is a cold cache.

### Phase 0.2 — agent_id wiring end-to-end (full plumbing, not just extras_json)

The earlier draft said "ship `agent_id` in `extras_json`, no schema migration." Audit found this insufficient: `extras_json` is a string column, so filter pushdown breaks for any per-agent recall. At 100 tenants × 1M typed_claims, recall scans become Python-side filtering — the regression Veilguard already lived through once (project_tcmm_perf_wins). Phase 0.2 plumbs it properly.

| # | Change | Where | Effort | Validates with |
|---|---|---|---|---|
| 0.2.1 | Extend `ObserveRequest` model with `extracted_by`, `agent_id` (optional) | [api/models.py:11](C:/Users/rudol/.gemini/antigravity/tcmm/TCMM/api/models.py) — **VM-authoritative** | ~20 LOC | New observe with `extracted_by=agent:researcher` writes block with field populated end-to-end |
| 0.2.2 | Thread `extracted_by` through `add_new_block()` at [tcmm_core.py:1871](C:/Users/rudol/.gemini/antigravity/tcmm/TCMM/core/tcmm_core.py) into block extras AND typed_claim | tcmm_core + adapters | ~50 LOC | Round-trip: observe → recall → claim object has `extracted_by` |
| 0.2.3 | Add `extracted_by` typed column to archive Lance schema (NOT just extras_json) | [lance.py:105-220](C:/Users/rudol/.gemini/antigravity/tcmm/TCMM/core/providers/lance.py) + idempotent migration via [migrate_schema.py](C:/Users/rudol/.gemini/antigravity/tcmm/tools/migrate_schema.py) | ~60 LOC | Existing rows backfill NULL; new writes populate column; Lance filter pushdown works |
| 0.2.4 | Add `_agent_filter()` to lance.py alongside existing `_ns_filter`/`_user_filter`/`_session_filter` | [lance.py:1498-1535](C:/Users/rudol/.gemini/antigravity/tcmm/TCMM/core/providers/lance.py) | ~30 LOC | Recall with `agent_id` filter scans Lance-side, not Python-side |
| 0.2.5 | Pii-proxy stamps `x-agent-id` header validation against agent registry | pii-proxy | ~40 LOC | Bogus agent_id rejected with 403; unknown agent_id treated as user-direct (no agent provenance) |
| 0.2.6 | Partition posterior aggregation by `extracted_by` BEFORE merge | dream module — `confidence_update.py` or equivalent | ~80 LOC | Two agents observe same triple; both rows survive distinctly; contradiction_arc detection sees both |

**Total Phase 0.2:** ~280 LOC, not the 200 estimated earlier. Wider scope but cleaner foundation.

### Phase 0.3 — approval gate (smallest change, biggest safety)

| # | Change | Where | Effort | Validates with |
|---|---|---|---|---|
| 0.3.1 | Approval gate at choke point | [agentic.py:122](C:/Users/rudol/Documents/veilguard/services/sub-agents/core/agentic.py) | ~30 LOC | Background daemon's `run_command` triggers toast; decision logs to `client_tool_approvals` |
| 0.3.2 | New `utils/client_tool_policy.py` with capability matrix + classify() | new file | ~100 LOC | Unit tests: foreground=ALLOW, background+shell=APPROVE, CRLF arg=DENY |
| 0.3.3 | New `bridge.request_approval()` method | [client_bridge.py:215](C:/Users/rudol/Documents/veilguard/services/sub-agents/core/client_bridge.py) | ~60 LOC | WS round-trip; idempotent on approval_id; daemon offline → DENY+log |
| 0.3.4 | Daemon WS handler for `request_approval` + Windows toast (`winotify` or `winrt.windows.ui.notifications`) | client-daemon `veilguard_client.py` | ~120 LOC | Toast renders; user click round-trips back to sub-agents; SQLite WAL queues decisions during network partition |
| 0.3.5 | `client_tool_approvals` + `client_tool_bypass` Lance tables (NEW — don't exist today) | TCMM (or sub-agents-local) Lance instance | ~40 LOC | Audit query works; bypass rules respected on subsequent calls |

### Phase 1 — agent identity (smallest viable org)

The minimum to have a named, persistent Director that uses TCMM-scoped private memory and is governed by the approval gate. **No Tasks, no team channels, no A2A yet.** Validates the identity layer in isolation.

- **Agent registry.** Extend `agents/*.md` frontmatter with `agent_id`, `role`, `manager_id`, `team_id`, `tool_allow_list`. Sub-agents service reads at startup; exposes via `/api/veilguard-client/agents`.
- **Director deployed as the standing user-facing agent.** ICs from §4 are *defined* in markdown but only invocable as sync sub-calls (today's pattern) — they have no inbox yet because there are no Tasks yet.
- **Private agent memory.** `agent/<aid>/` namespace works against the `agent_id` field added in Phase 0.2; Director's recalls hit blackboard only (Director has no observe).
- **Complete the approval gate** (Phase 0.3 finishes; Windows toast + audit table + bypass table all live).
- **Sidebar Agents view (read-only).** Shows registered agents, last-active timestamp, the org chart, recent approvals. No Task UI yet (Tasks don't exist).
- **Constitution stub.** `agents/CONSTITUTION.md` lands as a user-authored file; sub-agents service reads it at startup. No scoring uses it yet (Tasks don't exist), but it's available for Phase 2/3 to reference. Provides early surface for the user to declare objectives + constraints before the system starts proposing work.

Phase 1 ships an org *structure* without org *dynamics*. Validates the identity and approval bones before adding coordination on top.

### Phase 2 — Tasks, team memory, Critic publish

The coordination layer.

- **Task ledger.** Single new Lance table `agent_tasks` with schema per §3.3. Indexed on `owner_id`, `status`, `parent_id`. Inbox is a derived query, not a table.
- **Task tools for Director:** `create_task`, `assign_task`. Task tools for ICs: `accept_task`, `update_status`, `add_comment`, `submit_for_review`, `attach_output`.
- **Typed team channels.** `team/<tid>/events/`, `team/<tid>/knowledge/`, `team/<tid>/drafts/` added as TCMM scopes; recall filters honor the typing (different weights, `drafts/` only on explicit request).
- **Critic publish workflow.** `submit_for_review` on a Task transitions it to `review`; Critic's `review_decision` comment either approves (artifact promoted to `team/knowledge/` or blackboard, depending on target) or requests changes (Task returns to `in_progress`). PR-style, async, no debate.
- **Schema promotion (cleanup):** `agent_id` + `author` from `extras_json` to typed Lance columns via [migrate_schema.py](C:/Users/rudol/.gemini/antigravity/tcmm/tools/migrate_schema.py).
- **Cost tracking on Tasks:** each Task accumulates `cost.{tokens_in, tokens_out, cache_read, cache_write, usd}` from `pii_audit` rows tagged with its trace. Enables per-task budgets.
- **Sidebar Tasks view.** Per-agent inbox count, current task, recent comments. Still reuses `/api/veilguard-client/daemons` infrastructure — no LibreChat fork changes.

### Phase 3 — proactive stream (dream-as-scheduler)

Veilguard pivots from reactive-only to dual-stream. Full design in §3.7. Phase deliverables:

- **`task_proposals` Lance table** with schema per §3.7.6, indexed on `(status, decay_score DESC)` for queue scans.
- **Dream-cycle proposal hook.** Wrap the END of `run_cycle()` at [dream_engine.py:6312-6537](C:/Users/rudol/.gemini/antigravity/tcmm/TCMM/core/dream/dream_engine.py:6312) — AFTER the final reflective stage at line 12082+ (where `reflective_heuristic` emission happens). **NOT** `run_dream_cycle()` at line 5993 — that's an inner method that returns before reflective signals exist. Per §3.7.0 correction. Add a `run_proposal_pass()` mini-cycle for faster proposal cadence than full compaction. Identifies the **5 confirmed + 2 deferred** generative signal types (§3.7.1), scores them via §3.7.2 formulas, emits top-10 (capped per-type at 3).
- **Impact-score module** — pure functions per signal type, unit-testable. Weights stored as a small config table for runtime recalibration.
- **Director's pre-evaluator.** SINGLE Haiku rank pass over candidates ranked 4-N (top 3 use deterministic template briefs, no LLM). Structured output `{ranked_top_N, briefs[], assignee_picks[], rationales[]}`. Budget **~$0.001 per cycle per tenant** (one call, not per-candidate). Per §3.7.5.
- **Sidebar "Proposed Tasks" tab.** Reuses Daemons-tab infrastructure (no LibreChat fork changes). Shows queue with quick actions [Approve / Defer / Shelve]. Aged-unaddressed and recurrence-escalated proposals get visual emphasis.
- **Proposal lifecycle workers.** Per-cycle decay 0.9×, 7d TTL expiry, recurrence escalation at 5+ surfaces, auto-shelve on aged-unaddressed.
- **`proposal_outcomes` table.** Tracks proposal → Task → completion → value-realized chain. Per-proposal cost + downstream-reuse signal.
- **Weekly recalibration job.** Approval rates and downstream-value rates from `proposal_outcomes` feed back into impact-score weights. Signal types producing noise get auto-downweighted; signal types producing reused knowledge get auto-boosted.
- **Emergency lane.** USER×USER `contradiction_arc` bypasses Director pre-eval and surfaces directly to the user as a "needs your attention" sidebar item.
- **Strategy layer wiring** (full §3.9):
  - Constitution becomes load-bearing: every proposal's `final_score = signal_impact × objective_alignment × constraint_gate`. Constraint violations auto-decline; objective alignment shapes ranking.
  - `org_memory` Lance table (schema per §3.9.2) — org-memory candidates surface via the same proactive stream as Task proposals, just with a different table on the back end.
  - Regret v1 — `proposal_outcomes` gains `regret_score = task_cost / max(value_realized, ε)`, computed at 30d after task completion. Weekly recalibration uses regret + objective_deltas, not just approval rate.
- **Per-tenant proactive-stream config** (promoted from open question to ship requirement):
  - `proactive_stream_enabled: bool` (default true) — tenant can disable entirely; dream still runs, proposals still scored for audit, nothing surfaces
  - `proactive_cycles_per_day: int` (default 12 = every 2h) — bounded cycle cadence per tenant; prevents the $24/day-at-100-tenants surprise
  - `proactive_approval_cap_per_day: int` (default 20) — caps Director approvals/day; user-driven Tasks bypass
  - Constitution constraint `cost_ceiling_per_tenant_per_day` (default $5/day) — aggregate-level cost cap separate from per-task ceiling
- **Auto-pause on signal-quality drift** — if approved+pending+shelved proposals/day exceeds 3× the trailing-30d median for that tenant, recalibration job emits `signal_quality_drift` alert and auto-pauses the proactive stream pending user review. Prevents misconfigured constitution weights from flooding the queue silently.
- **Two missing signal emitters** (deliverable in dream, not just hook): implement `low_stability_cluster` and `stale_supersession_chain` as first-class dream emitter stages. Until these ship, proposal generation runs on 5 signal types, not 7 (§3.7.1).
- **Dream cycle scheduler** (net-new infrastructure):
  - `run_proposal_pass()` mini-cycle that emits proposals against the most recent fully-committed graph state WITHOUT running a fresh dream cycle. Runs at the per-tenant `proactive_cycles_per_day` cadence.
  - Full `run_cycle()` invocations remain on-demand or on a slower schedule (daily). Mini-cycle and full cycle don't conflict — mini-cycle is read-only against dream state.
  - Hook point: end of `run_cycle()` (line 6312+), AFTER stage 52f. NOT at `run_dream_cycle()` line 5993 — that was the wrong hook per audit.

Phase 3 is the architectural pivot that makes Veilguard a *system that surfaces work* rather than a tool that waits — with a constitution-shaped compass on top.

### Phase 4 — A2A (internal) + documents + lesson lifecycle

The protocol, collaboration, and organizational-learning-maintenance layer.

- **A2A as internal inter-agent transport.** Each agent exposes a minimal A2A endpoint internally — AgentCard at `/.well-known/agents/<aid>/agent-card.json`, `POST /agents/<aid>/messages` (maps to task comments), `GET /agents/<aid>/tasks/<id>`. Internal-only auth (API key bound to tenant). ~300-400 LOC. Replaces the bespoke `assign_task` RPC from Phase 2 with a standard-conformant transport.
- **Document workflow.** Task outputs in workspace get versioning: append-only `.history/` for files not under git, native git for files in repo-tracked dirs. Canonical paths only (no worktrees, per §3.5). Conflict resolution = the Task whose `outputs[]` first listed the path holds merge authority; peer Tasks open subtasks for changes.
- **Magentic-One-style ledger** on Tasks: structured `task_ledger` (facts/guesses/plan) + `progress_ledger` (per-step self-reflection). Persisted in the Task row's extras.
- **Lesson lifecycle management** (the anti-fossilization layer):
  - `expires_at` / `review_after` / `confidence_decay_per_week` enforced by a scheduled job (daily). Lessons past `expires_at` without recent reinforcement auto-transition to `expired`.
  - **"Lessons due for review" queue** in the decision-ledger sidebar — siblings to the proposed-tasks queue. Surfaces lessons whose `review_after` has passed, presented to user for `keep | amend | retire`.
  - Reinforcement signal — when a new reflective_heuristic node matches an existing lesson's trigger pattern, `last_reinforced_ts` resets and confidence is restored (up to original cap).
  - **Constitution-amendment proposals** — high-confidence, multiply-reinforced lessons (`reinforcement_count >= 5 AND confidence >= 0.75`) become eligible for promotion to constitution amendment, surfaced in the same review queue with distinct visual treatment.
- **`work_items_v` UNION view** (§3.10.2) materialized as a Lance view or a query helper. Powers the unified decision-ledger sidebar.
- **Replay v1** — counterfactual path reconstruction using bitemporal + lineage_chain. Shows alternate decision paths, not byte-exact LLM outputs.

### Phase 5 — A2A (external)

- A2A endpoints exposed externally. API-key + (later) OAuth/mTLS.
- External agents (e.g. a customer's CrewAI deployment) can delegate Tasks into Veilguard's org. Inverse: Veilguard's Director can delegate to external A2A endpoints.
- Small effort once Phase 3 internal A2A exists — mostly auth, rate-limiting, and tenant-scoped allow-lists.

---

## 6. Failure modes & guards

| Risk | Severity | Mitigation |
|---|---|---|
| Recall fanout multiplies /pre_request bottleneck and cold-cache cost | Critical | Phase 0.1 cache-stable cid; per-agent recall is the same Lance scan + agent_id filter, not extra scans |
| Memory pollution — agents corrupt user-authored facts | High → Low | `extracted_by=agent:<aid>` on every typed_claim; dream's existing `contradiction_arc` flags same-triple polarity flips automatically; Critic gates which claims enter the dream cycle. Largely solved by existing dream module; multi-agent only needs to thread agent_id through. |
| Claim contradiction triage — two agents make incompatible claims | Medium | Dream emits `contradiction_arc` automatically; Director sees it as a Task-creation signal (resolve via Researcher re-investigation or Critic adjudication) |
| Stance-arc escalation loop — agents disagree repeatedly | Medium | After N stance_arcs on related claims, dream's existing `reflective_heuristic` triggers; Director escalates to Committee (multi-reviewer quorum) |
| Org-chart loops (A asks B asks A) | Medium | Task lineage tracked via `parent_id`; cycle check at assignment time; max depth = 3 |
| Inbox unbounded growth | Medium | Per-agent open-task cap (e.g. 100); Director can't assign past cap; surfaced in sidebar |
| Director becomes a monolith (memory + tools + execution + review) | High | Director is deliberately thin — no observe, no client-daemon, no execution tools; recall against blackboard only; "memory" IS the Task table |
| Team channel becomes a junk drawer (recall quality collapses) | High | `team/` split into `events/` (low weight, decayed), `knowledge/` (Critic-gated, high weight), `drafts/` (recall on explicit request only) |
| Builder RCE via prompt injection from Researcher's fetched content | Critical | Phase 0.3 approval gate; `host_file_write` denied for all delegated calls; shell calls require user approval per-call |
| Critic becomes a bottleneck | Medium | Critic reviews are async; Director can override Critic-pending with explicit user consent; Critic has SLA (e.g. 5 min for foreground, 1 hr for background) before auto-decline |
| LibreChat fork patches die on deploy | High | All multi-agent UI lives in sidebar Daemons tab or Windows toast. Fork not touched. |
| AIStudio NLP rate-limit storm under fanout | High | Existing circuit breaker in `ai_studio_nlp_adapter.py`. Multi-agent does not increase per-turn NLP fanout; only adds parallel turns. |
| Schema migration risk | Low | v0 ships in `extras_json` (no migration). v1 uses existing idempotent migration framework. |
| Cost runaway under deep delegation | High | Director must explicitly choose Pattern B/C; Pattern A is default. Per-task cost ceiling configurable. Cache-stable cid keeps amortized cost low. |
| Proactive-stream echo chamber (proposal → task → claims → new proposal on same topic → loop) | High | Dedup on `(signal_type, sorted(signal_node_ids))`; recurrence escalation forces user choice after 5+ surfaces; impact-score decay 0.9× per cycle for unaddressed candidates |
| Proposal inflation toward one assignee (dream learns one IC's tasks get approved more → biases candidate generation) | Medium | Per-assignee fairness factor in ranking; alert if one assignee receives >70% of approvals over 30d |
| Adversarial signal injection via TCMM poisoning (bad claim enters → dream proposes work on bad premise → action taken) | Medium | Candidates derived from low-trust source_kind get reduced impact weights; USER×USER required for emergency-lane bypass; pre-eval Haiku sees source_kind explicitly |
| Pre-eval cost runaway (10 candidates × cycles × tenants × LLM calls) | Medium | Haiku-only for pre-eval; structured output ~500 tokens; per-tenant cycle cap; sampling fallback if tenant count grows large |
| Director drowns despite caps (queue grows faster than approvals) | Medium | Per-day approval cap caps the queue indirectly; aged-unaddressed auto-shelve after 7d; user can shelve in bulk |
| Signal-quality drift (a signal type that worked early starts producing noise) | Low | Outcome tracker monitors approval rates per signal_type; weekly recalibration auto-downweights bad signals; <20% approval over 100 candidates triggers manual review |
| Cross-namespace dream contamination — agents' findings cross-pollinate via bridge_score / concept_gravity | **Critical** | Phase 1 mandatory integration test: observe in two namespaces, assert bridge_score never crosses. Mitigate via per-agent-or-per-namespace filtering on `dream_archive.values()` loops in dream_engine.py. Bet-the-product severity. |
| Lance table permission inheritance (new tables created by root → silent 0-result recall) | High | Mandate `sudo -u rudol` for ALL table creation. Service unit explicitly runs as `rudol`. Add startup check: `assert os.geteuid() == os.getuid_of('rudol')` before table init. Memory documents the 2026-05-21 jnb migration that took recall to 0; six new tables = six opportunities to re-hit it. |
| CRLF/UTF-8 corruption in user-editable CONSTITUTION.md or agents/*.md | High | Phase 0.0.3 sanitization at every config-load boundary. Acceptance test with contaminated fixture. Without it: yaml parse error → Director init fails → sidebar 503 cascade. Same shape as documented .env gotchas. |
| Daemon-version skew during approval-gate rollout | High | Phase 0.0.4 capability handshake. Fail-closed default (DENY on missing feature). Sidebar warning surfaces stale daemon versions. ~30-min auto-update poll means rollout window where the gate could wedge background work. |
| Dream node soft-delete while pinned by pending proposal | High | Dream GC must be soft-delete-only for nodes referenced by a `pending` proposal. Proposals carry materialized brief text + `signal_node_ids` as audit trail, NOT as runtime refs after creation. Add `pinned_by_proposal_ids` back-pointer on dream nodes. |
| Subtask completes after parent cancelled (orphan claims) | High | Cancelling a parent immediately sets all descendant `open`/`blocked` to `cancelled(reason=parent_cancelled)`; `in_progress` get a flag to checkpoint+stop; any claims already observed are tagged `parent_task_cancelled=true` in extras_json — Critic's promotion gate rejects them by default. |
| Constitution amendment re-scoring policy | High | When constitution updates, pending proposals are re-scored against new weights and re-sorted; any failing `constraint_gate` auto-shelve with `reason=constitution_amended` + diff pointer. Done items NOT retroactively re-evaluated — replay uses `constitution_version` snapshot stored on the proposal. |
| Constitution self-justifying loop — metrics calibrated by the same loop they steer | Medium (acknowledged, no clean fix) | Recalibration weights bounded ±20%/week; quarterly user review of metric definitions surfaced as sidebar "metric calibration" prompt. Bounds drift; doesn't fully solve self-justification. |
| Pattern C cache amortization is additive across agents, not multiplicative | Medium | 3 agents = 3 distinct cache-create costs per fresh-cache turn at 1h TTL. ~3× cache_write per Pattern C cold start; subsequent calls within TTL get full benefit. Acceptable tradeoff for parallel work. |

---

## 7. Open questions

To resolve before Phase 1:

1. **Director model.** Same model as the foreground assistant today, or a different (cheaper? smarter?) model? Cost vs quality tradeoff.
2. **Team membership at multi-tenant scale.** Is `team_id = core` global, or per-tenant? Almost certainly per-tenant — confirm and design the table accordingly.
3. **Consultant promotion.** What triggers a consultant moving to IC status — usage count, tenant-admin action, both?
4. **Critic SLA defaults.** 5 min foreground / 1 hr background — calibrate against real Critic latency once measured.
5. **Cache TTL choice.** Default 1h for sub-calls (math says 1h pays after ~0.83 extra hits); 5m for the user-foreground turn where Director's system prompt is the only stable prefix. Confirm.
6. **Workspace location for the user's Windows box.** Server-side is decided. Does Builder also get a "local workspace" mirror at `Documents\veilguard\workspace\<aid>\` as a sync target, or do all writes go through the approval gate per-call?
7. **A2A external auth model.** Phase 3 question. API key sufficient, or OAuth required? Depends on whether external agents from different orgs/tenants need access.
8. **Critic output schema.** `score()` returns what — single float, structured `{score, blockers, suggestions}`? Decide once before Phase 1.
9. **Inbox surfacing.** Does the user see all agent inboxes, or only Director's queue? Probably only Director's by default + drill-in to others.
10. **Does dream currently respect `agent_id` / namespace scoping?** Today nodes carry `namespace` + `user_id`; dream's derived primitives are scoped per-namespace. Need to verify bridge_score and concept_gravity don't cross-pollinate two agents' findings within the same namespace uncontrollably.
11. **Dream cycle cadence vs Task lifecycle.** Dreams run on a schedule (need to confirm cadence — every N blocks? hourly?). Does a Critic-approved artifact trigger an immediate mini-cycle, or wait for the next batch? Critic-approval-to-recall latency depends on this.
12. **`extraction_confidence` across the agent → Critic boundary.** If Researcher emits a claim with `extraction_confidence=0.7` and Critic approves, does the confidence get a boost (Critic-verified), stay the same, or aggregate? Need a rule before Phase 2.
13. **Per-agent reflective_heuristic scoping.** Dream already produces `reflective_heuristic` nodes (meta-cognition about patterns). With multi-agent, should each agent have its own reflective layer, or do all heuristics live at the org level? The latter is simpler but loses per-agent specialization signal.
14. **Dream cycle cadence vs proposal cadence.** Full dream cycles are heavy. The proposal-emission pass may want faster cadence (hourly) than full compaction (daily). Need a `run_proposal_pass()` mini-cycle, or a way to emit proposals without re-running the full graph build.
15. **USER×USER emergency-lane definition.** What exactly counts as USER×USER for emergency-lane bypass? Both claims have source_kind=USER and incompatible polarity on the same canonical_triple_hash. Decide formal criteria before Phase 3.
16. **Stance_arc auto-escalation threshold.** At what polarity_distance does a stance_arc trigger committee review vs single-Critic? Default 0.7; calibrate against real data once Phase 2 ships.
17. **User raw-queue visibility.** Two-tier UI ("raw queue" vs "Director-recommended") or single-tier? Single-tier is simpler; raw queue can be a power-user toggle later.
18. **`team/events/` channel under dream-as-scheduler.** Most events flow from Task lifecycle, and dream already tracks what happened. The events sub-channel may be redundant. Reconsider during Phase 3.
19. **`extracted_by` partition exact location in dream's posterior aggregation.** Phase 0.2.6 says partition BEFORE merge; the exact code location in `confidence_update.py` (or equivalent) needs depth-reading before Phase 0.2 implementation.
20. **`run_proposal_pass()` mini-cycle scope.** Phase 3 net-new infrastructure. What stages does it run? Just proposal-emission against existing graph, or also refresh `decay_score` on existing proposals?

---

## 8. Non-goals

- Not a debate framework. Not interactive actor-critic loops. Not mixture-of-agents.
- Not microVM-isolated agent sandboxes (Fly Sprites-style). A plain dir on the VM is the v1 workspace. Revisit when blast radius outweighs the cost.
- Not auto-skill-acquisition / recursive self-improvement.
- Not a replacement for the existing LibreChat-driven chat UX. Multi-agent extends it via the sidebar.
- Not unlimited agent counts. Org is small (Director + 4 ICs = Researcher + Builder + critic-claim + critic-prose; consultants on demand). Adding more agents requires a deliberate design pass — the goal is a competent team, not a swarm.
- Not autonomous Director. The user is always upstream of Director on the reactive stream; on the proactive stream Director proposes Tasks but cannot create them without explicit user approval. Director never initiates chat turns with the user except via scheduled-report digests.
- Not a claim-market / epistemic-economics layer. Letting agents bid budget on claims conflates effort with truth. Use `reinforcement_count` + downstream-value tracking instead.
- Not autonomous policy adoption. Reflective_heuristic and recurring_ritual signals propose skills and rules, but adoption is always human-gated (proposal → user approval → adoption). Drift from auto-adopted policy is the failure mode automated organizational learning hits.

---

## 9. Decision log

| Date | Decision | Reasoning |
|---|---|---|
| 2026-05-21 | Doc location: `Documents\veilguard\MULTI_AGENT_PLATFORM.md` | Single canonical plan at repo root; `agents/` is for per-agent definitions |
| 2026-05-21 | v0 uses `extras_json` not Lance schema columns for `agent_id` / `author` | Zero-downtime ship path; promote in Phase 2 once shape is stable |
| 2026-05-21 | Approval surface = Windows toast, not LibreChat banner | Fork patches die on deploy; user may not have LibreChat open; client-daemon has notification permission |
| 2026-05-21 | No interactive debate loops; Critic is async PR-reviewer | ICLR 2025: multi-agent debate doesn't beat single-agent test-time compute on interactive turns |
| 2026-05-21 | 1h default cache TTL for sub-agent calls | Breakeven after ~0.83 extra hits; multi-agent fanout clears that easily |
| 2026-05-21 | Corporate structure: Director + 3 ICs + consultants; not flat swarm | Aligns authority with privilege; matches user's mental model; well-trodden production pattern (CrewAI hierarchical, Magentic-One, Manus) |
| 2026-05-21 | Inter-agent messaging = A2A | Standard exists, ~300 LOC for internal use, future-proofs external interop |
| 2026-05-21 | Tasks live in a dedicated Lance table, not in TCMM | Different lifecycle, schema, lookup patterns; conflating them pollutes recall |
| 2026-05-21 | Existing security personas (`phishing-analyst`, etc.) become `consultant` role | Preserves existing work; lets them rejoin the org without rework |
| 2026-05-21 | **Single primary primitive: Task.** Messages = task comments. Documents = task artifacts. Inbox = a query. | Three primitives bloat Phase 1, duplicate audit paths, and create ambiguity about which entity owns what. Collapsing to Task gives free audit, free dependency graph, free cost ledger. |
| 2026-05-21 | **Director is deliberately thin** — no observe, no client-daemon, no execution tools | Avoids the accidental monolith. Director's value is routing, not doing. Director's "memory" IS the Task table. |
| 2026-05-21 | **Phase 1 compressed** — registry + Director + private memory + approval gate only. Tasks deferred to Phase 2. A2A deferred to Phase 3. | Previous Phase 1 smuggled the coordination layer into the identity layer. Separating them keeps each phase debuggable in isolation. |
| 2026-05-21 | **`team/` channel typed** into `events/` / `knowledge/` / `drafts/` with distinct write rules and recall weights | A single shared write-anywhere channel becomes a junk drawer; typing avoids it. |
| 2026-05-21 | **Consultants read context from `task.inputs`, not `recall()`** | Consultants aren't on team `core`; cross-team recall would leak. Explicit input paths are cleaner and auditable. Surfaced in worked-example trace §10. |
| 2026-05-21 | **Submission target enum has three values:** `team_knowledge` / `user_deliverable` / `org_blackboard` | Trace revealed a third case: artifacts produced FOR a specific user, not for org-wide reuse. All three Critic-gated, but only `org_blackboard` writes to global TCMM. |
| 2026-05-21 | **IC picks the lowest-impact promotion target** that satisfies the task | Encoded in IC system prompt, not data model. Avoids over-promotion of one-off work to org-wide knowledge. |
| 2026-05-21 | **Stale-dep TTL = 7 days default** | If an IC accepts a task whose deps completed >7d ago, raise `blocker_raised(reason=stale_dep)` and Director decides whether to re-run. Surfaced in worked-example trace §10. |
| 2026-05-21 | **Add `task_id` column to `pii_audit`** (Phase 2) | Without it, per-task cost roll-up is impossible. One column, default NULL, written by pii-proxy when the request carries a task_id header. |
| 2026-05-21 | **Cross-turn task notifications via sidebar + optional toast** | User must be able to leave the chat and return; sidebar Tasks view shows in-flight; client-daemon toast on user-owned task transitions to `done`. Reuses approval-gate notification surface. |
| 2026-05-21 | **Director mid-task status messages are distinct from `final_synthesis`** | UI must distinguish ephemeral "researching now..." pings from the final deliverable response. Same conversation, different affordance. |
| 2026-05-21 | **Knowledge layer = existing dream module; multi-agent does not invent one** | Reading dream_engine.py + typed_claims.py revealed a full cognitive architecture with causal_arc, contradiction_arc, stance_arc, belief_attribution, semantic_principle, information_gap, reflective_heuristic, motif_node, recurring_ritual. Most of my proposed "Knowledge Compiler" moonshot already exists. |
| 2026-05-21 | **Agent provenance via `extracted_by`, not new `source_kind` values** | `extracted_by` field on TypedClaim (line 183) already exists for forward-compat. Threading `agent:<aid>` through it requires zero schema or enum churn. Agents are data, not types. |
| 2026-05-21 | **Critic's review unit = typed_claims, not the prose artifact** | Critic gates which claims enter the dream cycle. Validation reuses existing `validate_claim()` + `validate_synthesis()` structural checks + semantic judgment. Rejected claims stay in IC's private namespace and die in compaction. |
| 2026-05-21 | **Blackboard = view over dream-compiled state, not a write target** | Promotion happens because dream eats the claim and emits derived structure. No separate `blackboard/published/` write channel; "recall from blackboard" filters dream output. |
| 2026-05-21 | **Director invokes existing dream signals as workflow triggers** | `information_gap` → auto-create Task for Researcher. `reflective_heuristic` → Skill Crystallization proposal. `stance_arc` between agents → multi-reviewer Committee escalation. These signals exist already; multi-agent just wires them to Task creation. |
| 2026-05-21 | **Dual-stream Director: reactive (user) + proactive (dream-as-scheduler)** | Pre-evaluating dream-proposed Tasks is genuinely LLM-shaped work. Resolves earlier "should Director be an agent" question — yes, because pure routing wouldn't make these calls. Director gains a second stream of work but stays thin (no execution, no observe, no memory). |
| 2026-05-21 | **Generative-signal taxonomy: 7 dream node types propose Tasks** | `information_gap` / `contradiction_arc` / `reflective_heuristic` / `recurring_ritual` / `stance_arc` / low_stability_cluster / stale_supersession_chain. Other dream primitives (causal_arc, semantic_principle, narrative_arc, identity_*, motif_node, belief_attribution, concept_node) shape recall but don't propose action. |
| 2026-05-21 | **Proposal budget: 10/cycle, 3/signal-type, 20/day approvals** | Three stacking caps prevent flood. User-driven Tasks bypass all caps (user authority supreme). Tunable per tenant. |
| 2026-05-21 | **Impact scoring is multiplicative, not additive** | Any single weak factor zeroes the candidate. Want to act on things that matter to real work, not on dream's most interesting findings in the abstract. |
| 2026-05-21 | **Emergency lane for USER×USER contradictions** | Skip Director pre-eval, skip caps; surface directly to user. Critical signals shouldn't queue behind low-impact info_gaps. |
| 2026-05-21 | **`task_proposals` table separate from `agent_tasks`** | Different lifecycle (most never become Tasks), different schema (signal_node_ids, recurrence, decay), different access pattern (queue scans + bulk decay). Conflating would fill `agent_tasks` with dead-letter rows. |
| 2026-05-21 | **Director pre-evaluator uses Haiku, not Opus** | Structured output ~500 tokens, ~$0.001/call. Salience judgment doesn't need Opus-tier reasoning; downweight false positives via outcome tracker, not via expensive model. |
| 2026-05-21 | **Outcome tracker recalibrates impact-score weights weekly** | Per-signal-type approval rates + downstream-value rates feed back into weights. Signal types producing noise auto-downweighted; signal types producing reused knowledge auto-boosted. Self-calibrating loop. |
| 2026-05-21 | **Push back on Claim Markets moonshot** | Effort ≠ truth. Letting agents spend to back claims conflates "I spent more compute" with "I'm more correct"; budget-rich agents dominate epistemics. Use existing `reinforcement_count` (dream_topology.py:94) + downstream-value tracking instead. Empirical, not economic. |
| 2026-05-21 | **Per-agent LoRA preferred over manual Skill Forking** | `dream_to_lora.py` already exists. Per-agent fine-tune emerging from accumulated typed_claims is reachable in Phase 4+. Skill Forking as manual config tweak is a stepping stone; LoRA-from-claims is the destination. |
| 2026-05-21 | **Replay v1 = path-not-bytes** | Counterfactual replay shows alternative decision paths, not byte-exact LLM outputs. Anthropic responses aren't deterministic; caching prompts for byte-exact replay is a separate (deferred) moonshot. Bitemporal already gives "what did the system believe at time T" for free. |
| 2026-05-21 | **Semantic GC entropy field deferred to Phase 4+** | Dream already has selection pressure + supersession + stability scoring. Explicit `entropy` field becomes mandatory once Veilguard runs for months and namespace size matters. Until then, existing decay mechanisms suffice. |
| 2026-05-21 | **Strategy layer: Constitution + Org Memory + Regret** | Closed adaptive loop needs a compass. Constitution = user-authored top (objectives, constraints, metrics). Org Memory = system-proposed lessons user-approved as institutional rules. Regret = per-proposal score driving recalibration. Together they steer the proactive stream away from local-optima drift. |
| 2026-05-21 | **Constitution is user-authored, never auto-adopted** | Constitution amendments may be proposed via reflective_heuristic + org memory promotion, but adoption is always user-gated. Auto-adopted policy is explicitly forbidden in §8 Non-goals. |
| 2026-05-21 | **Org Memory is a 100-entry curated table, not accumulated** | Distinct from skills (how-tos), blackboard (facts), and constitution (steering). Org Memory entries are *rules about how the organization operates*, with an evidence chain tracing back to reflective_heuristic + tasks that contributed. |
| 2026-05-21 | **Regret v1 = `accepted_but_low_value` only; v2 deferred to Phase 4+** | Easy regret (approved + ran + never recalled) ships in Phase 3 with full instrumentation. Hard regret (rejected_but_should_have_done) requires causal_arc graph traversal across shelved proposals; defer until v1 data shows v2 is worth the complexity. |
| 2026-05-21 | **Deterministic top-3 + single Haiku rank pass on ranks 4-10** | Per-candidate Haiku pre-eval was excess cognition. Top 3 use template briefs + signal-type default alignment vectors (no LLM). Ranks 4-10 get ONE Haiku rank pass that picks 2-3 worth surfacing with refined briefs. ~70% pre-eval cost reduction. Less "dream → Director → dream → Director" stacking. |
| 2026-05-21 | **Default objective-alignment vectors per signal type** | Static config; gives top-3 candidates a constitution-aware score without an LLM call. Seed values per signal type; recalibrated weekly from `objective_deltas` measured against constitution.metrics. Signal types that consistently advance an objective shift their default alignment toward it. |
| 2026-05-21 | **Lessons have explicit expiry + confidence decay + scheduled review** | Without expiry, lessons become organizational scar tissue — rules learned in one phase ossify and steer proposals after their premises have shifted. `expires_at` (default 180d), `review_after` (default 90d), `confidence_decay_per_week` (0.02) erode unreinforced lessons. Reinforcement via new matching reflective_heuristic resets the timer. |
| 2026-05-21 | **Decision-ledger framing adopted** (§3.10) | What we built is a decision ledger, not a task system. Tasks / proposals / outcomes / lessons all move through a shared lifecycle skeleton (proposed → accepted → executed → evaluated → institutionalized). Unified status enum across all kinds. Lineage_chain field on every row for cross-kind ancestry queries. |
| 2026-05-21 | **Option B for physical storage: separate tables + `work_items_v` UNION view** | Full collapse (Option A) into one `work_items` table has real costs: write-path contention across high-churn (proposals) vs slow-mutating (lessons) kinds; sparse schema bloat; Lance fragmentation acceleration on slow rows. Keep four tables, expose unified view for lineage/audit/UI queries. |
| 2026-05-21 | **Option A reversibility noted, not adopted** | If atomic cross-kind writes become a requirement (e.g., approve-proposal-and-create-task-atomically), switch to single `work_items` table via one-time migration backfilling from the four extant tables. UNION view can't do atomic cross-kind writes. Until that pattern shows up, Option B is sufficient. |
| 2026-05-21 | **Sidebar surface is conceptually "the decision ledger"** | Pending decisions (proposals + lesson reviews) + active work (in_progress tasks) + recent decisions + filter-by-kind. Same Daemons-tab infrastructure. No LibreChat fork changes. User sees one queue, kind-filterable. |
| 2026-05-21 | **Two audit panels applied (10 agents total): touchpoint, dream-integration, edge-case, critic, integration strategist (round 1); implementation simulator, consistency auditor, adversarial security, latency/cost modeler, onboarding reviewer (round 2).** | Round 1 reshaped Phase 0/3, split Critic, corrected dream hook, dropped 2 fake signal types. Round 2 added trust-boundary discipline, fixed 12+ doc contradictions, added cost model + glossary + system diagram. Specific corrections recorded as individual log rows below. |
| 2026-05-21 | **Dream hook point: `run_cycle()` end (line 6312-6537, after stage 12082+), NOT `run_dream_cycle()` line 5993** | The inner method returns before reflective signals exist. Wrapping it would emit proposals against a half-populated graph. |
| 2026-05-21 | **Signal taxonomy: 5 confirmed + 2 deferred (`low_stability_cluster`, `stale_supersession_chain` require new dream emitter stages)** | Earlier "7 signals" claim was incorrect — those two don't exist as emittable node types today. |
| 2026-05-21 | **Critic split into critic-claim (Haiku, inline) + critic-prose (Sonnet, async)** | Bundling forced either over-spend or shallow review. Critic-claim runs structural validation on every team-knowledge promotion (~1-3s); critic-prose runs semantic review only on `org_blackboard`/`user_deliverable` (minutes). Throughput bottleneck disappears. |
| 2026-05-21 | **Director recall scope expanded to include `conv` + `team/knowledge`** | Earlier "blackboard only" scope made Pattern A fictional — every conv-context question forced delegation. Read-only conv-scoped recall lets Director answer ~80% of turns solo, restoring cache-amortization economics. |
| 2026-05-21 | **`source_kind` is set by the tool, never the agent** | Prompt-injection provenance-laundering hole. Hard-coded: `fetch()` always stamps `TOOL_RESULT`, user-conv observe always stamps `USER` (verified via `x-user-id` header), agent observe stamps per-agent class. Server-side rejection of any agent-prose attempt to overwrite. |
| 2026-05-21 | **Phase 0.0 added — operational prerequisites must ship FIRST** | Source-of-truth table, agents/*.md frontmatter parser (doesn't exist today!), CRLF/UTF-8 sanitization helper, daemon WS capability handshake, Lance maintenance cron extension. Without these, Phases 0.1-0.3 inherit avoidable rework. |
| 2026-05-21 | **Phase 0.2 expanded from `extras_json` ride-along to full typed-column plumbing** | extras_json is a string column → filter pushdown breaks → Python-side filtering at scale = the perf regression Veilguard already lived through. Phase 0.2 ships full ObserveRequest + add_new_block + Lance typed column + _agent_filter end-to-end. ~280 LOC (was ~200). |
| 2026-05-21 | **`canonical_triple_hash` policy: partition aggregation by `extracted_by` BEFORE merge** | Two agents observing the same triple were collapsing into one claim via posterior aggregation before contradiction detection. Partition-before-merge preserves both views; aggregation across agents happens at recall/render time. Alternative (hash includes extracted_by) rejected because it doubles same-agent reinforcement. |
| 2026-05-21 | **Cross-namespace dream contamination — Phase 1 integration test is mandatory** | `dream_archive.values()` loops at 11+ sites have no namespace filter. bridge_score / concept_gravity will cross-pollinate. Phase 1 test: observe in two namespaces with overlap, assert no cross-namespace arcs. Bet-the-product severity if missed. |
| 2026-05-21 | **Worktree mandate removed; canonical paths only** | Violated documented user memory `feedback_no_worktrees`. Replaced with "first Task whose outputs[] listed the path is merge authority" + atomic writes via temp-then-rename. Honors standing user instruction. |
| 2026-05-21 | **Sidebar UI explicitly acknowledged as fork-patch work** | Earlier draft claimed "no LibreChat fork changes" but `useVeilguardAPI.ts` IS in `deploy/librechat-patches/`. New sidebar tabs (Agents, Proposed Tasks, Lessons, Decision Ledger) ship via Dockerfile bake — never bind-mount dist (the 2026-05-18 regression pattern). |
| 2026-05-21 | **Daemon capability handshake with fail-closed default** | `bridge.request_approval()` doesn't exist today. Phase 0.0.4 ships the handshake so deployment ordering (sub-agents before user daemons auto-update) doesn't wedge background work. Missing-feature → DENY for background, sidebar warning. |
| 2026-05-21 | **Approval-gate arg sanitization rejects control chars BEFORE glob match** | CRLF-encoded args bypass arg_glob today. Memory documents two CRLF incidents already; this would be the third. Normalize before policy lookup; unit test asserts CRLF triggers DENY even with matching bypass. |
| 2026-05-21 | **Approval timeout = 120s default → auto-DENY with reason `approval_timeout`** | Without timeout, offline daemon wedges background work indefinitely. Per-tool override possible. Agent sees timeout as normal tool failure → raises blocker_raised or fails Task gracefully. |
| 2026-05-21 | **Cache-stable cid implementation clarified — byte-stable PREFIX, not stable cid** | Anthropic cache key is content-derived from prefix bytes; cid is only the memoization key on pii-proxy. Phase 0.1 enforces clean prefix/per-call boundary at pii-proxy. Dual-read window for legacy cids during 1h post-deploy. |
| 2026-05-21 | **UNION view caching is mandatory, not optional** | Audit: 4-table UNION at sidebar refresh cadence = perf killer (same pattern as old admin dashboard). Materialize `work_items_recent` (last 30d), refresh via existing `_windowed_rows()` 8s TTL pattern. Phase 4 ship requirement. |
| 2026-05-21 | **Lance maintenance cron extended to new tables from day 1** | `task_proposals` has worst fragmentation profile (~10 writes/cycle, decay updates, recurrence bumps). Memory shows 813 fragments on `sparse_archive` in 2 days under similar pattern. Extend cron BEFORE first table created. |
| 2026-05-21 | **Lance tables created as `rudol` user, never root** | 2026-05-21 jnb migration took recall to 0 via root-owned `_indices/`. Six new tables = six places to repeat the failure. Startup assert: `os.geteuid() == os.getuid_of('rudol')` before table init. |
| 2026-05-21 | **Subtask cancellation cascade: parent cancel → descendants cancel/checkpoint; orphan claims tagged + rejected by Critic** | Without cascade, in-progress subtasks keep writing to team channels after parent context is gone. Tag any orphan claims `parent_task_cancelled=true`; Critic promotion gate rejects them by default. |
| 2026-05-21 | **Constitution amendment re-scoring policy: pending re-evaluated; done/retired NOT** | Done items use `constitution_version` snapshot stored on the proposal/task for replay. Pending re-scored against new weights; failing constraint_gate auto-shelve with `reason=constitution_amended`. Avoids retroactive invalidation of the audit log. |
| 2026-05-21 | **Consultant private memory is `(consultant_id, tenant_id)`-namespaced** | Multi-tenant correctness. Single namespace = cross-tenant leak via consultant's private memory. Consultant definition (markdown) is shared; memory is per-tenant. Extends existing TCMM tenant isolation invariant. |
| 2026-05-21 | **Auto-pause proactive stream on signal-quality drift (3× trailing-30d median)** | Misconfigured constitution weights would silently flood the queue. Auto-pause + `signal_quality_drift` alert pending user review. Self-limiting. |
| 2026-05-21 | **Two missing signal emitters added as Phase 3 dream-side deliverables** | `low_stability_cluster` + `stale_supersession_chain` must be implemented in dream (new emitter stages). Until then, 5-signal taxonomy. Phase 3 explicitly includes the implementation, not just the hook. |
| 2026-05-21 | **No autonomous Director — proactive Tasks always user-approved (reaffirmed under audit pressure)** | Anchoring-trap critique (user blesses bad amendments by trusting evidence chain): partial mitigation via constraint-relaxation amendments requiring 7d cool-off, asymmetric friction matching asymmetric blast radius. Constraint-tightening amendments don't need cool-off. |
| 2026-05-21 | **Frontmatter format = bold-Markdown KV, NOT YAML** | Implementation simulator caught existing personas use `**Model:** X` not YAML `---`. Earlier draft assumed YAML; would have rejected existing files or forced premature migration. Parser uses `^\*\*(Key):\*\*\s*(value)$` line-grammar; YAML fenced support deferred. |
| 2026-05-21 | **Sanitizer has TWO modes: env (lossy ASCII rescue) + text (Unicode-preserving)** | Em-dash → "--" is correct for .env values; corrupts markdown prose. One sanitizer for both was an audit miss. Mode is required parameter on all sanitize calls; default fails loudly. |
| 2026-05-21 | **Bidirectional capability handshake (daemon↔server, not one-way)** | Phase 0.3 requires daemon to know server enforces the gate so it doesn't proxy commands that bypass policy locally. Server advertises `approval_gate_enforced=true`; daemon refuses tools without it. Defense in depth. |
| 2026-05-21 | **Critic split honored in org chart: 4 ICs, not 3** | §3.1 chart, §8 Non-goals, decision log all said "Director + 3 ICs"; §4.4 split made it 4 (critic-claim + critic-prose). All references updated. |
| 2026-05-21 | **Server-of-record discipline applied per field across trust boundaries (§3.8.5)** | Adversarial audit: every "client supplies field X" is an authority leak. Made server-of-record explicit for `task_id`, `lineage_chain`, `source_kind`, approval_token bytes, approval_id, comments[] chain, reinforcement_count, extracted_by. Each closes a class of attack. |
| 2026-05-21 | **TOCTOU defense on approval gate via approval_token bound to arg_hash** | Without this, agent could approve `git status` and execute `rm -rf /`. Server-issued HMAC-bound token; daemon re-verifies arg hash before exec; single-use. Wire-protocol change required in Phase 0.3, not patchable later. |
| 2026-05-21 | **`lineage_chain` is server-computed, never client-supplied** | Forgeable lineage breaks replay/audit. Server computes at insert time from `parent.lineage_chain + [parent.id]`. Entries are `(kind, uuid)` tuples to prevent cross-kind UUID forgery. |
| 2026-05-21 | **`source_kind=USER_PARAPHRASE` added as 7th enum value** | Agent-summarized user statements were forgeable as `source_kind=USER` and triggered emergency lane. Verbatim USER ≠ paraphrased; emergency lane requires both verbatim. |
| 2026-05-21 | **`task_id` on `pii_audit` derived server-side from `cid → agent_tasks` join, NOT from header** | Header-trusted = forgeable. Per-task cost attribution must be derivable from authenticated session state, not client-supplied identifiers. |
| 2026-05-21 | **`comments[]` on Tasks moved to separate `task_comments` Lance table with `prev_hash` chaining** | Mutable comments destroy review-decision audit integrity. Append-only chain; Task row holds head hash; mutation = audit alarm. |
| 2026-05-21 | **`reinforcement_count` requires evidence from ≥2 distinct `extracted_by` values** | Single-agent self-reinforcement gamed the org-memory promotion path. Cross-agent OR USER evidence required to advance counter. |
| 2026-05-21 | **Naming conventions anchor section (§3.4.2)** | Slash form (`team/knowledge/`) for namespaces, underscore form (`team_knowledge`) for enums/API values. Brief template at §3.7.5 had mixed form; corrected. |
| 2026-05-21 | **TL;DR + glossary + system diagram added at top of doc** | Onboarding review: cold reader gave up at §0 because all nouns were undefined. New 5-things-to-read pointer + glossary of 14 terms + ASCII system diagram. Doc grows but doubles legibility for new readers. |
| 2026-05-21 | **Cost model section added (§10.5) with concrete numbers** | Latency/cost audit produced: Pattern A=$0.155, B=$0.42, C=$0.665 warm; $11.7/tenant/day active; Phase 0.1 ROI=$24/tenant/day=$72k/mo at 100 tenants. Highest single-LOC ROI in the spec. |
| 2026-05-21 | **Director's `Model:` key supports `reactive=X, rank_pass=Y` mapped form** | Resolves the previously-open dual-model question (Sonnet reactive + Haiku rank-pass). Mapped form is per-tool. ICs and consultants use scalar form by default. Frontmatter parser handles both. |
| 2026-05-22 | **Use Claude Agent SDK as the agent-runtime loop** | Audit panel verified the SDK supports: subagent spawning, prompt caching, MCP client, PreToolUse hook for approval gate, streaming, per-subagent model selection. The 3 gaps (tenant context, post-LLM hook, cache_control auto-placement bug) are mechanical workarounds we handle in middleware. Net effect: ~30% of Phase 0 collapses; agent-runtime becomes a new sibling service to pii-proxy. |
| 2026-05-22 | **agent-runtime is a new sibling service, not folded into pii-proxy** | Separate Docker container (port 5000). Keeps pii-proxy's redaction/multi-provider concern isolated. Pii-proxy routes Anthropic-bound calls to agent-runtime; everything else (OpenAI, xAI, Gemini) continues going direct. Provider neutrality preserved at the routing layer. |
| 2026-05-22 | **Phase 0 implementation shipped: harness + agent personas + ledger schemas + 121 tests passing** | Built end-to-end on 2026-05-22 in one autonomous build session. What landed: agent-runtime FastAPI scaffold, TCMM render middleware (with byte-stable parent_cid memoization), audit wrapper, cache_control normalizer (SDK-bug defense), tenant contextvars, PreToolUse approval gate hook + capability matrix + CRLF/control-char arg sanitization, persona loader (bold-Markdown KV per §0.3), text_sanitize (2-mode), constitution loader + scorer, 5 new persona files (director / researcher / builder / critic-claim / critic-prose), all 7 decision-ledger Lance schemas + CRUD for tasks / append-only comment chain / proposals, docker-compose entry. Tests: 121 unit + integration. Live-API spike script ready. Real CONSTITUTION.md + all 8 personas (3 existing + 5 new) verified loadable. |
| 2026-05-22 | **Phase 2 shipped: MCP tools + inbox poller + Windows toast + pii-proxy routing. 150 tests passing.** | Built immediately after Phase 0. Director's 10 ledger tools (create_task / assign_task / convert_proposal / shelve_proposal / accept_task / add_comment / attach_output / submit_for_review / get_task / inbox) exposed via in-process MCP server using `create_sdk_mcp_server`. Memory tools (recall / observe / read_constitution) exposed via second in-process MCP server. IC inbox poller (`workers/inbox_poller.py`) — background asyncio loop that watches `agent_tasks` for status=open + owner_id ∈ {researcher, builder, critic-claim, critic-prose}, atomically claims via lease columns, dispatches via `runtime.run_agent_query()`. agent_tasks schema gained `lease_owner` + `lease_until` columns for distributed-safe claiming. Client-daemon bumped to v0.3.0: `winotify`-based Windows toast for `request_approval` WS method, local SQLite WAL for offline approval queue, capability advertisement in auth handshake (`approval_gate: true|false`). Pii-proxy gained `_handle_agent_runtime_request` early-route: when `AGENT_RUNTIME_ENABLED=true` AND user in `AGENT_RUNTIME_USER_ALLOWLIST` (empty = all), Anthropic-bound chat requests forward to agent-runtime over httpx with SSE pass-through. Per-user allowlist for safe rollout. Graceful 503 fallback on agent-runtime unreachable. installer.iss `MyAppVersion` bumped in lockstep (per `workflow_daemon_release` discipline). |
| 2026-05-22 | **Cache framing corrected: TCMM owns cache_control placement; agent-runtime passes through unmodified.** | Initial draft had `normalize_cache_control()` running on TCMM's `/render` output — fighting TCMM's per-provider renderer (which already places markers correctly, per memory `project_tcmm_renderer_architecture`). Stripped from the hot path. `app/middleware/tcmm.py` now passes TCMM bytes through verbatim; persona system_prompt appended as a SEPARATE final block (post-cache-point) so persona changes don't invalidate TCMM's cache slot. `normalize_cache_control()` kept as a defensive utility for health checks but no longer called on TCMM output. Regression test pins the pass-through behaviour. Spike script (`scripts/spike_cache_validation.py`) rewritten to call real TCMM `/render` and verify the SDK preserves TCMM's markers end-to-end. The real question being answered: "does the SDK molest TCMM's careful work" not "can we produce a stable prefix" (TCMM already does the latter). 151 tests passing. |
| 2026-05-22 | **Backend abstraction added: SdkBackend / ScriptedBackend / SsoBackend.** | `app/backends/` defines an `LLMBackend` interface so agent-runtime can swap the agent-loop engine. **SdkBackend** wraps `claude_agent_sdk.query()` (production, needs ANTHROPIC_API_KEY). **ScriptedBackend** returns canned tool_use / text responses keyed on `(persona_id, turn_index)` — lets us drive demos + tests without LLM cost. **SsoBackend** is a stub that documents the wire shape for routing through TCMM `/generate` (user's Claude Max subscription via Claude CLI subprocess pool); implementation pending until the route is exposed in TCMM. Runtime selects via `BACKEND=sdk|scripted|sso` env var. SDK backend keeps its internal agent loop; scripted/sso use runtime.py's external dispatch loop via `app/tool_dispatcher.py`. |
| 2026-05-22 | **Lance `IS NULL` not supported in where clauses — use sentinel values.** | Inbox poller's claim filter `lease_owner IS NULL OR lease_until < now` returned 0 rows because LanceDB's SQL parser doesn't accept `IS NULL`. Switched to sentinel: tasks at rest have `lease_owner=""` and `lease_until=0.0`; available filter is just `lease_until < now`. `ledger/tasks.py:create_task` updated to set sentinels. Pinned by tests + demos. |
| 2026-05-22 | **`review_decision` MCP tool added for Critic.** | Atomically adds a `review_decision` comment AND transitions task status. `decision=approved` → review→done; `decision=changes_requested` → review→in_progress (IC iterates); `decision=declined` → review→cancelled. Tool count went from 10 → 11. Single-call pattern avoids the comment/status race. |
| 2026-05-22 | **4 demo scenarios run end-to-end, all pass.** | `demo/scenario_pattern_a_solo.py` (Director answers from recall), `demo/scenario_pattern_b_delegation.py` (Director → Researcher full lifecycle), `demo/scenario_pattern_c_fanout.py` (Director → Researcher + Builder parallel), `demo/scenario_critic_iterate.py` (Researcher → Critic reject → Researcher iterate → Critic approve). `demo/run_all.py` orchestrates. Demonstrates: ledger writes, inbox poller dispatch, parallel task claiming via lease, comment chain with 6+ status_changes and 2+ review_decisions, final status=done. All four scenarios use the scripted backend (no LLM cost) and prove the full orchestration pipeline works on the actual Lance store. |

---

## 10. Worked example — trace one Task end-to-end

A concrete trace of how a real user request flows through the org. Picked to exercise: Director routing, parallel fanout, consultant invocation, Critic review with changes, dependency unblock, cost accounting.

**T+0 — User → Director (foreground chat):**
> "Research the current state of WebAuthn passkey adoption among US consumer banks, then draft a one-page memo I can send to my security team."

**T+1 — Director reasons:**
- Recalls blackboard. Finds 3 stale notes on passkeys from 2 months ago — not enough.
- Pattern A (solo) rejected: needs fresh web research + a polished writeup.
- Picks Pattern C (parallel fanout). Decomposes:
- `create_task(t-001, brief="WebAuthn passkey memo (coordinator)", deliverable_spec="user-facing memo + research backing", assignee=director)` — parent
- `create_task(t-001-a, parent=t-001, brief="research passkey adoption among top 10 US consumer banks by deposits; cite sources; 200-400 words", deliverable_spec="markdown at team/drafts/passkey-research.md", assignee=researcher)` — open
- `create_task(t-001-b, parent=t-001, brief="given t-001-a output, write 1-page exec memo for security team", deliverable_spec="markdown at team/drafts/passkey-memo.md, ~400 words, sections TLDR/Trend/Risks/Recommendation", assignee=report-writer (consultant), deps=[t-001-a])` — blocked
- Director responds to user: "Researching now. Memo follows once the findings are clean."

**T+2 — Researcher's view (`status=open AND owner_id=researcher`) surfaces t-001-a:**
- `accept_task(t-001-a)` → in_progress
- Runs `web_search` + `fetch`; observes findings to `agent/researcher/observations/` (author=`agent:researcher`)
- Writes `team/drafts/passkey-research.md` with inline citations
- `attach_output("team/drafts/passkey-research.md")`
- `submit_for_review(target=team/knowledge)` — promotion target chosen because findings should be reusable by future tasks
- Status → review; Critic notified via task comment

**T+3 — Critic picks up review:**
- Recalls blackboard for context, reads draft
- Finds: one claim about Chase has no citation
- `add_comment(kind=review_decision, body=changes_requested, notes="Para 3 Chase claim needs source")`
- Task → in_progress; Researcher notified

**T+4 — Researcher iterates:**
- Reads review_decision, fixes the citation
- `submit_for_review(target=team/knowledge)` again

**T+5 — Critic approves:**
- `add_comment(kind=review_decision, body=approved)`
- Artifact promoted: TCMM block written to `team/knowledge/` (author=`agent:researcher`, promoted_by=`agent:critic`)
- File moved to `team/published/passkey-research.md`
- Task → done

**T+6 — Dependency cleared, t-001-b auto-unblocks:**
- `t-001-b` status: blocked → open
- Report-Writer is a `consultant` (not on team `core`) — does NOT have recall against `team/knowledge/`. Instead, the Task's `inputs` field is auto-populated with the artifact path from the deps: `inputs=[team/published/passkey-research.md]`.
- Report-Writer accepts, reads `inputs[0]`, writes `team/drafts/passkey-memo.md`
- `attach_output(...)`, `submit_for_review(target=user_deliverable)` — this memo is for THIS user, not org-wide knowledge

**T+7 — Critic reviews the memo:**
- Approves (assume)
- Task → done. No promotion to team/knowledge or blackboard — this is a user-scoped artifact.

**T+8 — Director sees all subtasks done:**
- t-001 status → review (Director's own consolidation step)
- Director calls `final_synthesis(task_ids=[t-001, t-001-a, t-001-b])`
- Composes user response with the memo body + a citations footer
- t-001 → done
- Costs roll up: `t-001.cost = sum(t-001-a.cost, t-001-b.cost, director_synth.cost)`. Stamped from `pii_audit` via trace_ref.

**T+9 — User receives the memo + cost summary in the sidebar.**

---

### Gaps surfaced by the trace

These didn't appear in the abstract design and need explicit treatment:

1. **Consultant recall scope.** Report-Writer is not on team `core`, can't `recall(team/knowledge)`. Resolution: **consultants read their context from `task.inputs` (explicit artifact paths), not from recall.** The Task carrying them upstream context is the only channel. Add to consultant role spec.

2. **Submission target enum needs three values, not two.** The earlier design implied `team/knowledge` or `blackboard`. The trace reveals a third: **`user_deliverable`** — artifacts produced FOR this user, not for org-wide reuse. Targets: `{team_knowledge, user_deliverable, org_blackboard}`. Critic gates all three, but only the last writes to blackboard TCMM.

3. **Critic SLA / queue starvation.** With one Critic, a backlog blocks everything. Resolution options: (a) per-tenant Critic-2 replica spun up automatically when queue > N, (b) Director can auto-decline-and-resubmit-to-user after timeout. Decide before Phase 2.

4. **Cross-turn task continuity.** If user closes the browser between T+1 and T+8, how do they know t-001 finished? Resolution: sidebar Tasks view shows in-flight tasks per user; optional Windows toast via client-daemon when a user-owned task transitions to `done`. Reuses the approval-gate notification surface.

5. **Stale-dep risk.** If t-001-b doesn't pick up for two weeks (Report-Writer unavailable), the research may be stale. Resolution: Task carries `dep_freshness_ttl` (default 7d); on accept, if any dep is older than ttl, the assignee raises `blocker_raised(reason=stale_dep)` and Director decides whether to re-run.

6. **Promotion target choice is in the IC's prompt, not the Task schema.** Researcher decided `team/knowledge` in T+2; the design didn't say who chooses. Resolution: **the submitting IC picks the lowest-impact target that satisfies the task** — encoded in IC system prompts, not data model.

7. **Cost roll-up depends on `trace_ref` linking Tasks to pii_audit rows.** Today `pii_audit` has `conversation_id` but no `task_id`. Phase 2 must add `task_id` to `pii_audit` writes from the pii-proxy (one column, optional, default NULL). Otherwise cost per task is unknowable.

8. **Director responding mid-task** ("Researching now...") is a SECOND foreground turn, not part of the synthesis. Means Director can emit status messages without owning a Task. That's fine — it's a free comment, not a deliverable — but the UI must distinguish "Director status ping" from "Director final response."

These eight items are added to §7 Open Questions and (where decided) to §9 Decision Log.

---

## 10.5 Cost + latency model

Quantitative model produced by a dedicated cost/latency audit. Assumptions: TCMM blob = 260k tokens (memory-confirmed); Opus 4.7 $5/M in, $25/M out, $10/M cache write 1h, $0.50/M cache read; Sonnet ~half that; Haiku ~10× cheaper; pii-proxy overhead ~80ms p50 / 250ms p95; Anthropic TTFT Opus 3-10s, Sonnet 2-5s, Haiku 1-3s on large prefixes.

### 10.5.1 Per-pattern unit economics (warm-cache, Phase 0.1 landed)

| Pattern | $/turn | Latency p50 | Latency p95 | Latency p99 |
|---|---|---|---|---|
| **A — Solo Director (Opus)** | $0.155 | 6.5s | 9s | 13s |
| **B — Director + 1 IC sync (Opus + Sonnet)** | $0.42 | 12s | 18s | 25s |
| **C — Parallel fanout (3 sub-calls + Critic-claim)** | $0.665 | 13s | 20s | 30s |

**Cold cache (first call against TCMM blob):** Pattern A = $2.625, Pattern C = ~$8.5 (5 cache writes at Opus rate). This is why Phase 0.1 cache-stable cid is the highest-ROI item.

### 10.5.2 Per-tenant daily costs

| Tenant state | $/day |
|---|---|
| Idle (proactive cycles running, no user activity) | $0.26 |
| Active (40 reactive turns/day, 60/30/10 A/B/C mix) | $11.7 |

**Composition at active:** $11.44 reactive + $0.26 proactive ≈ $11.7/day.

### 10.5.3 Scale projection

| Tenants | $/mo active mix | Lance growth/mo | What breaks first |
|---|---|---|---|
| 1 | $351 | 150 MB | Nothing |
| 10 | $3,500 | 1.5 GB | Nothing |
| 100 | $35,000 | 15 GB | Lance scan on `task_proposals` (~200 tenants) |
| 1000 | $351,000 | 150 GB | Anthropic rate limits at ~300 concurrent tenants |

**Hard ceilings:**
1. **Anthropic rate limits** — Tier-4 workspace caps ~4k RPM Opus. Multi-workspace sharding required at ~300 concurrent tenants.
2. **Lance scan on `task_proposals`** — 1000 tenants × 12 cycles × 10 candidates × 30d = 3.6M rows, ~5 GB hot. Fragments faster than `sparse_archive` did. Mitigation: per-tenant table partitioning + weekly `optimize_indices()`.
3. **TCMM cold-fanout serialization** — pre-Phase-0.1 each sub-agent's recall pays 5-7s cold-start under pool lock. Hits at ~50 concurrent users. Phase 0.1 eliminates this.

### 10.5.4 Phase 0.1 ROI (single tenant before/after cache-stable cid)

- **Before Phase 0.1:** ~10 sub-agent cold-create calls/day × $2.60 Opus cache_write = **$26/day wasted** + main thread $6.20 → ~$32/day/tenant
- **After Phase 0.1:** sub-agents share parent cache → 10 × $0.13 read = $1.30 + main $6.20 → **~$7.5/day/tenant**
- **Savings: $24/tenant/day = $720/mo per tenant = $72k/mo at 100 tenants**

**Phase 0.1 has the largest ROI per LOC of any phase in the spec.** ~80 LOC; pays back the entire engineering investment for the multi-agent platform within the first 10 tenants.

### 10.5.5 Latency tradeoffs forcing redesign

1. **Pattern C fanout vs cache-stability** — at 100+ tenants the 3 parallel sub-calls each pay an Opus cache-write if cid isn't sub-key-cache-stable. Tradeoff: serialize the fanout (cheap, +6s latency) vs implement per-agent persistent cids (Phase 0.1++, complex, keeps parallelism). Cost gap $0.665 → $8.5/turn cold.
2. **Render-per-call vs render-once-per-conv** — render is 150ms but underlying TCMM recall is 3.3s warm. At Pattern C wallclock 13s, recall is 25% of budget. Forces caching `/render` output per cid for the turn (proxy-level memoization).
3. **Haiku rank-pass vs local-Qwen rank-pass** — at 1000 tenants × 12 cycles, Haiku costs $260/day floor. Local Qwen-3B = ~$0 but requires GPU on prod VM (currently CPU per `architecture_nlp_backend`). Decision point at ~500 tenants.

### 10.5.6 Phase 0 minimum cost projection

Single existing tenant, active mix, post-Phase-0:
- Phase 0.0 ships → no LLM cost impact (operational setup only)
- Phase 0.1 ships → $32/day → $7.5/day (the saving above)
- Phase 0.2 ships → +$0 (schema additions; observe() calls are unchanged in count)
- Phase 0.3 ships → +<$0.10/day for capability handshake + approval audit writes

**Net Phase 0 result: $7.5/day/tenant baseline.** Multi-agent UX builds on top of this.

---

## 11. Edge case catalog

Distilled from a dedicated edge-case audit pass. Organized by category. Each scenario has a design answer (extending the spec, not contradicting it). The top-10 marked **MUST** are mandatory before Phase 2/3 ships — late discovery forces schema migration, retroactive recomputation, security postmortem, or tenant-isolation cleanup.

### 11.1 Concurrent operations

**11.1.1 — Two ICs editing the same team draft simultaneously.** First-Task-whose-`outputs[]`-listed-the-path is merge authority; other Tasks editing the same path auto-convert to subtasks at `attach_output()` time. Critic reviews the owner Task. Sequential PR-style merges against it.

**11.1.2 — Researcher observes a claim contradicting one Critic is currently reviewing.** `validate_claims()` takes a `claim_set_version` (monotonic per IC namespace). `approve_publish()` rejected if version moved; Critic must re-review the diff.

**11.1.3 — MUST: Dream node GC vs pending proposal references.** Soft-delete only for nodes referenced by pending proposals. Promoted Task carries materialized brief text, not runtime refs. Add `pinned_by_proposal_ids: [uuid]` back-pointer on dream nodes.

**11.1.4 — Director synthesis while Builder is still mid-write.** `attach_output()` writes atomically (temp file + rename); `outputs[]` only appended on completion. Director's `final_synthesis` asserts all referenced subtask statuses are `done`, not `in_progress`/`review`.

### 11.2 Lifecycle ambiguity

**11.2.1 — Task done, claims later contradicted.** Closed Tasks stay closed; the contradiction emits a NEW proposal (`signal_type=contradiction_arc`) referencing both the new claim and the original Task's published artifact. Add `superseded_by_task_id` extras column on completed Tasks. Immutable audit log; no zombie state machines.

**11.2.2 — User defers proposal 6 times.** At the 5th defer, the proposal auto-shelves with `reason=user_repeatedly_deferred`; the underlying signal_type gets a per-tenant penalty in recalibration. Forcing infinite re-promotion is itself a UX flaw.

**11.2.3 — Expired lesson exactly matches new reflective_heuristic.** Never auto-revive. Require human re-acceptance via a "lesson revival" proposal kind. Auto-revival would re-create the organizational scar tissue the expiry mechanism prevents. Mark the new proposal's `lineage_chain` with the expired lesson's id.

**11.2.4 — MUST: Subtask cancellation cascade.** Parent cancel → descendant `open`/`blocked` → `cancelled(reason=parent_cancelled)`; `in_progress` get checkpoint+stop flag. Orphan claims tagged `parent_task_cancelled=true`; critic-claim rejects by default.

**11.2.5 — Constitution amendment proposed while related lessons expiring.** Presenting an amendment for approval pins (`expires_at=null`) every lesson in its evidence chain until the amendment is approved/shelved/expired. Without pinning, a stale amendment could land on evidence that no longer exists.

### 11.3 User behavior

**11.3.1 — Browser closed mid-foreground task.** Director writes `synthesis_draft` Task comment as it streams; on browser-reopen, sidebar shows "Director was synthesizing — resume / restart / discard." Reactive stream cannot rely on the chat being open at the synthesis moment.

**11.3.2 — Two tabs both approve the same proposal.** `convert_proposal()` is idempotent via `proposal.id` + status transition CAS (`UPDATE WHERE status='pending'`); the second click sees `status='approved'` and is a no-op surfacing "already approved → task t-XXX". Daemons-tab WS push invalidates the queue on every transition.

**11.3.3 — User approves then rescinds Task before IC accepts.** Task in `open` can be rescinded → `cancelled(reason=user_rescinded)`. Once IC has called `accept_task()` (status `in_progress`), rescinding requires a second confirm and incurs IC's compute-so-far on the cost ledger.

**11.3.4 — User ignores queue for 30 days.** After tenant-wide 14-day no-touch, proactive stream auto-pauses (still scoring/storing dream output, not surfacing). Sidebar shows banner "proactive stream paused — N proposals queued, click to resume." Prevents 200 stale forced-choice notifications.

**11.3.5 — MUST: Constitution amendment re-scoring policy.** Pending proposals re-scored against new constitution; constraint-failing ones auto-shelve. Approved/done items NOT retroactively re-evaluated — replay uses `constitution_version` snapshot stored on the proposal.

**11.3.6 — User opts out of proactive stream.** Tenant-level flag `proactive_stream_enabled=false`. Dream still runs (knowledge layer), proposals still get scored and written to `task_proposals` for audit, nothing surfaces. Re-enabling shows only proposals from last 7d (older TTL-expired).

### 11.4 Failure modes

**11.4.1 — pii-proxy restart with 3 in-flight proposals.** `task_proposals` in Lance (durable). In-flight Haiku rank-pass dies; rank-pass is idempotent via `rank_pass_lease_until` column — if a worker dies, lease expires after 60s, another worker picks up. Top-3 deterministic path needs no recovery (pure function).

**11.4.2 — Network partition during approval toast.** Daemon offline → user click can't reach VM. Client-daemon queues approval decisions to local SQLite WAL; on reconnect, replays in order with idempotency key. VM's `request_approval` carries `approval_id` as dedup key. Sidebar mirror is fallback view but daemon is canonical user-presence proof.

**11.4.3 — LLM timeout during Director rank pass.** 30s timeout + retry-once; on second failure, ranks 4-10 auto-deferred (not shelved — they'll re-rank next cycle via decay). Surface "rank pass degraded — only top 3 shown this cycle" badge. Never silently drop candidates.

**11.4.4 — MUST: Dream cycle atomicity around proposal emission.** Proposals written only on atomic Lance batch commit; mid-cycle crash = no proposals written; next cycle re-runs from last checkpoint.

**11.4.5 — Lance fragmentation during proposal write storm.** `task_proposals` added to weekly `optimize_indices()` + `compact_files()` maintenance cron from day 1. Per-tenant cycle cap bounds churn rate. Monitor fragment count via dashboard; alert at >100 fragments.

**11.4.6 — AIStudio NLP rate-limit storm during dream cycle.** Proposal emission is async with respect to NLP-dependent passes — if NLP rate-limited, dream still emits proposals using most recent cached typed_claims rather than blocking. Proposal cadence decouples from NLP throughput. Circuit breaker (existing) handles the storm.

### 11.5 Security

**11.5.1 / 11.5.2 — Source_kind forgery via prompt injection (RESOLVED in §3.8.5):** `source_kind` is tool-set, never agent-set. See trust-boundary table.

**11.5.3 — Constitution amendment poisoned via reflective_heuristic.** Amendment approval UI shows full evidence chain (lineage_chain rendered as tree). Constraint-RELAXATION amendments require 7-day cool-off; constraint-TIGHTENING don't. Asymmetric friction.

**11.5.4 — CRLF/control-char arg bypass (RESOLVED in §3.8):** policy layer rejects control chars before glob match.

**11.5.5 — MUST: Cross-tenant leak via dream's bridge_score.** Phase 1 integration test (mandatory): observe claims in two namespaces with overlapping concepts; assert bridge_score arcs never cross namespace boundaries. Bet-the-product severity. Mitigation in §3.4 (`dream_archive.values()` loops need per-namespace filtering).

### 11.6 Cost

**11.6.1 — Misconfigured objective weight floods proposals.** Per-tenant proposal-rate monitor; if approved+pending+shelved proposals/day exceeds 3× trailing-30d median, recalibration emits `signal_quality_drift` alert and auto-pauses proactive stream pending user review. Constitution editor UI enforces `sum(weights) == 1.0 ± 0.01`.

**11.6.2 — Regret-score gaming.** `value_realized` excludes recalls where `recaller.agent_id == producer.agent_id`; only cross-agent + user recalls count. Add diversity factor: `value_realized × (unique_recallers / total_recalls)`. Prevents closed-loop self-reinforcement.

**11.6.3 — MUST: Per-tenant cycle cap.** Resolved by Phase 3: `proactive_cycles_per_day` (default 12) + `cost_ceiling_per_tenant_per_day` Constitution constraint.

**11.6.4 — `work_items_v` UNION view at scale.** Lazy view for kind-specific filter queries; for unified sidebar feed, served from incremental projection (`work_items_recent` last 30d, refreshed on each underlying write via existing `_windowed_rows()` 8s cache pattern). UNION cost concern is real; cache pattern is proven.

### 11.7 Multi-tenant

**11.7.1 — Tenant A reads dream signal scoped to tenant B.** Direct extension of 11.5.5. `task_proposals` has `(tenant_id, user_id)` columns matching TCMM's discipline; dream proposal emission iterates per-tenant; `signal_node_ids` are tenant-local primary keys. Integration test: two tenants with identical claim text → disjoint proposals.

**11.7.2 — MUST: Consultant private memory is `(consultant_id, tenant_id)`-namespaced.** Single namespace would leak across tenants. Definition (markdown) shared; memory per-tenant.

**11.7.3 — Parent org pushes baseline constitution to child tenants.** Constitution gets nullable `parent_constitution_id`; child constitution is an overlay (adds objectives, can tighten but not loosen parent constraints). Constraint loosening attempt = compile-time error in constitution editor. Schema field reserved in Phase 3 to avoid migration when Phase 5 ships full multi-tenant.

**11.7.4 — Tenant's lesson promoted to constitution amendment — cross-pollinate?** Never auto-cross-pollinate. Tenant-admin can publish a *lesson template* to a registry; other tenants opt in by importing → fresh lesson in their org_memory with `imported_from: tenant_id+lesson_id` lineage. Manual import preserves user agency.

### Top 10 MUST-HAVES (panel-ranked, by "force-redesign-if-late" severity)

1. **11.5.5** — Dream bridge_score cross-namespace test as Phase 1. Bet-the-product.
2. **11.5.1 / 11.5.2** — `source_kind` hard-coded by tool, never agent. Forgeable provenance breaks the entire strategy layer.
3. **11.1.3** — Dream node soft-delete while pinned by pending proposal. Late discovery = NPE on stale `signal_node_ids`.
4. **11.6.3** — Per-tenant proactive-cycle cap. $1200/mo surprise + recalibration math depends on cadence.
5. **11.2.4** — Subtask cancellation cascade + orphan-claim handling. Recall pollution before team notices.
6. **11.4.4** — Dream cycle atomicity around proposal emission. Partial state = proposals referencing nonexistent nodes.
7. **11.5.4** — CRLF/control-char normalization in arg_glob policy. Documented gotcha pattern; predictable third occurrence.
8. **11.1.1** — Document merge authority on concurrent edits. Without "first-output Task wins," concurrent ICs corrupt drafts.
9. **11.3.5** — Constitution amendment re-scoring policy. First amendment after Phase 3 either silently invalidates audit log or floods sidebar.
10. **11.7.2** — Consultant private memory is `(consultant_id, tenant_id)`-namespaced. Phase 5 assumes this; if Phase 3 ships single-namespace, migration silently splits histories.

---

## 12. References

Research panel transcripts that produced this design are in the conversation log (2026-05-21). Key external sources:

- [A2A Protocol v0.3 Specification](https://a2a-protocol.org/v0.3.0/specification/)
- [Claude API — Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [Anthropic Engineering — Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)
- [Magentic-One dual ledger architecture](https://www.microsoft.com/en-us/research/articles/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/)
- [CrewAI hierarchical teams blueprint](https://sparkco.ai/blog/implementing-crewai-in-hierarchical-teams-a-2025-blueprint)
- [LangGraph supervisor vs swarm tradeoffs](https://focused.io/lab/multi-agent-orchestration-in-langgraph-supervisor-vs-swarm-tradeoffs-and-architecture)
- [Letta v1 agent loop rearchitecture](https://www.letta.com/blog/letta-v1-agent)
- [Blackboard multi-agent system (arXiv 2507.01701)](https://arxiv.org/html/2507.01701v1)
- [Multi-agent debate critical review (ICLR 2025)](https://d2jud02ci9yv69.cloudfront.net/2025-04-28-mad-159/blog/mad/)
- [Devin 2.0 agent-native design](https://medium.com/@takafumi.endo/agent-native-development-a-deep-dive-into-devin-2-0s-technical-design-3451587d23c0)
- [Persistent sandbox platforms — Northflank](https://northflank.com/blog/persistent-sandboxes)
- [AI Agent Memory in 2026 — DEV](https://dev.to/max_quimby/ai-agent-memory-in-2026-auto-dream-context-files-and-what-actually-works-39m8)
