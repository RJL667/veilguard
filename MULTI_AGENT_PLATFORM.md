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
- **Tasks transform state; conversations are scaffolding.** *(Added 2026-05-27 in response to external review.)* The dominant workload of this system must be agents producing observable state mutations — ledger writes, memory writes, artifact creation, approval responses, dream node updates — not agents narrating to each other. Any feature whose primary observable is "more messages exchanged between agents" gets architectural skepticism. The canonical failure mode of public multi-agent platforms (Auto-GPT, BabyAGI) is collapse into expensive self-narration: ~70-80% of tokens spent on "I should think about what I just thought about," near-zero artifact production. **Operationalised** by the Artifact Production Ratio (APR) metric and a Phase 6 circuit breaker (§3 — memory write-path discipline + Phase 6.7) that pauses agent-to-agent calls when APR drops below a healthy band. The principle is observable, not aspirational: if APR < 0.1 sustained over 30 minutes, the system is talking to itself and gets halted automatically.
- **TCMM is the knowledge-graph substrate; the ledger is the state machine.** *(Added 2026-05-27 after the 2-iteration TCMM-unification panel.)* Content that contributes to the knowledge graph — agent reasoning, lessons, outcome narratives, proposal rationales, observed claims — lives in TCMM, where it earns concept_gravity, bridge_score, and contradiction detection. Operational state — task status, leases, hash-chained audit, alignment weights, secrets, approval records, settings — lives in the ledger, where it gets exact lookups, atomic transitions, hash-chain integrity, and never-compress guarantees. **TCMM observations may reference ledger PKs (cross-ref pointers in `extras.entity_id`) but must never mirror ledger fields** (status, decided_at, cost, approval state). Never dual-write operational state to TCMM — the sync overhead doesn't earn a recall benefit you can't get from FTS over the ledger's content columns. Migrate only knowledge-graph contributors. This rule aligns Veilguard with the universal pattern across Magentic-One (Task Ledger vs working memory), MemGPT/Letta (main/recall/archival hierarchy), Cognition Devin ("decisions an agent commits to must be reconstructible exactly — RAG is unsafe for that"), Anthropic Multi-Agent Research (Jun 2025 lead-plan vs subagent findings), and the entire event-sourcing literature (Kafka, Datomic, EventStore — one-way projection, never bidirectional).
- **Every subsystem must justify itself via measurable operational leverage.** *(Added 2026-05-27 in response to an external "conceptual overfitting" critique.)* APR (Phase 6.7) measures the system's overall artifact-vs-narration discipline. This principle is the per-subsystem counterpart: any subsystem — `proposal_taxonomies`, `stance_arcs`, `reflective_heuristics`, `regret_weighting`, future additions — must be defensible against a concrete operational question: *"What measurable outcome moves when this subsystem is on vs off?"* If the answer is qualitative ("it makes recall richer") or unfalsifiable ("the dream graph would be less expressive"), the subsystem is decoration, not infrastructure. Existing subsystems get audited under this rule at the Phase 6 post-mortem; new subsystems must pass it before merging. The architecture has reached a level of philosophical elegance where preserving abstractions because they're beautiful is now the dominant failure mode — this principle is the guardrail. Mechanical enforcement: every subsystem in `agent-runtime/app/` is named in a `tests/SUBSYSTEM_ROI.md` companion file with its specific metric, its on/off measurement, and the threshold below which it's a candidate for removal.

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

> **AMENDED 2026-05-29 — authoritative: [TCMM_CHANNEL_ARCHITECTURE.md](TCMM_CHANNEL_ARCHITECTURE.md).** The typed-channel design below is refined:
> - **Channel is a first-class block COLUMN + a read-time visibility filter + a recall weight — NOT a substrate partition.** The dream graph stays unified per user; channels are stamped (`channel` on raw blocks, `_channels` owner-qualified token-set on dream nodes) and filtered at the single `get_archive_entry` chokepoint via a swappable `subset` (conservative default) / `intersection` (permissive) policy.
> - **One dream substrate per collaboration unit (v1 = per `user_id`; tenant = `user_id` is the hard wall).** The slash-form "namespace tree" below is now a set of recall-time VIEWS over that unified per-user graph, selected by `channel` — not separate stores.
> - **Substrate-wall reality:** there are **~231** dream cross-block loops (not the ~30 estimated) and dream nodes carry no `user_id`, so per-loop guards are infeasible. The protection is the **single-user-per-instance invariant** enforced at `bulk_warm_user_archive` (tripwire), audited by `tests/test_channel_substrate_wall.py`.
> - **Promotion = re-author** a fresh clean-lineage block (Critic-gated), never flag-flip. **Channel enum** (reconciled with §3.4.2 / §3.11 submission targets): `agent_private, conv, team_drafts, team_events, team_knowledge, user_deliverable, org_blackboard`.

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
| `low_stability_cluster` | **Shipped 2026-05-27 → verified end-to-end 2026-05-28.** Implemented agent-runtime-side in `agent-runtime/app/proposals/signal_emitters.py` as a scanner over the `archive` table that injects synthetic `LOW_STABILITY_CLUSTER` rows into `dream_archive`. DreamScanner picks them up next cycle. Avoids dream-engine surgery. | `emit_low_stability_clusters()` |
| `stale_supersession_chain` | **Shipped 2026-05-27 → verified end-to-end 2026-05-28.** Same pattern — scans archive for topic groups whose max(`timestamp`) is older than `STALE_CHAIN_AGE_DAYS` (default 60d) and injects `STALE_SUPERSESSION_CHAIN` synthetic dream rows. | `emit_stale_supersession_chains()` |

End-to-end live verification — see `scripts/verify_signal_emitters.py`. Run from inside the agent-runtime container; seeds the local `archive` with two synthetic clusters (one low-density, one 70 days old), fires `run_one_cycle()`, asserts:
1. Empty-archive guard short-circuits without log noise on a 0-row archive.
2. `emit_low_stability_clusters` writes exactly 1 LOW_STABILITY_CLUSTER row to `dream_archive` for the seeded low-density cluster.
3. `emit_stale_supersession_chains` writes exactly 1 STALE_SUPERSESSION_CHAIN row for the seeded stale cluster.
4. Re-running the cycle on unchanged data emits 0 (idempotency — both emitters check `dream_archive` for an existing same-topic synthetic node per user_id).
5. `DreamScanner._scan_once()` picks up both synthetic rows, computes `signal_impact × objective_alignment > 0`, and writes a `task_proposal` row per user-signal combination (verified: TOTAL +19 over a scan with 19 candidate dream rows when per-signal cap raised; per-user delta +2 = 1 low_stab + 1 stale_chain).

**Latent bug surfaced + fixed during verify run (2026-05-28).** The emitter's `_scan_topics()` called `tbl.search().select(["aid", …])` directly. On an empty `archive` table (local dev or a freshly-provisioned tenant), Lance raises `SchemaError: No field named aid` and the worker logged a WARNING every 24h cycle indefinitely. Patched: short-circuit if `tbl.count_rows() == 0` or the schema lacks `aid`/`topics` fields. See `[EMPTY_ARCHIVE_GUARD_2026_05_28]` comment in `signal_emitters.py:_scan_topics`.

Phase 3 proposal generation now runs on all 7 signal types as intended; the original "5 signal types, not 7" footnote is retired.

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

Candidate keyed on `(tenant_id, user_id, signal_type, sorted(signal_node_ids))`. Same key re-triggering increments `recurrence_count`, refreshes `last_surfaced_ts`, and floors `decay_score` to `max(existing_decay, new_impact_score)`; no duplicate row created.

**Implementation state.** Shipped 2026-05-28 — `ledger.proposals.create_proposal` performs the dedup-check at write time. Until 2026-05-28 the spec contract was correct but the writer didn't enforce it (every re-emission was a new row); the dream_scanner's in-memory `_seen_keys` provided a single-process workaround but reset on container restart, allowing the audit table to fill with structurally-identical rows. The Phase-3 dedup now runs in the storage layer: on a SHELVED status the dedup releases (a user-shelved signal that re-emerges deserves a new surfacing). Cross-tenant collisions are impossible — the key includes `(tenant_id, user_id)`. Anchor: `[PROPOSAL_DEDUP_2026_05_28]` in `agent-runtime/app/ledger/proposals.py:create_proposal`. Regression tests at `app/ledger/tests/test_proposal_dedup.py`.

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

#### 3.10.5 Sidebar surface — concrete endpoints (Phase 3/4 ship state)

*Added 2026-05-28 — closes out the sidebar-tab specification.* The §3.10.4 description is the conceptual view; this subsection pins down the wire contract so the LibreChat fork's `useVeilguardAPI.ts` hook and the four sidebar tabs are implementable without re-reading the code.

**Auth.** All endpoints expect `tenant_id` + `user_id` as query params (GET) or top-level JSON keys (POST). `VEILGUARD_INTERNAL_SECRET` is checked via the proxy injection middleware — sidebar requests go LibreChat → `pii-proxy:4000/api/veilguard/*` → `agent-runtime:5000/*`. The proxy injects the secret header; the sidebar JS never sees it. Missing tenant/user → 400. Missing secret → 503 cascade (see `architecture_internal_secret_503` memory).

**Push model.** All four tabs open one `EventSource` to `GET /events?tenant_id=…&user_id=…` after their initial GET. Server pushes one SSE message per ledger mutation in that namespace (`proposal_created`, `proposal_status_changed`, `task_created`, `task_status_changed`, `task_comment_added`, `lesson_status_changed`). Heartbeat every 25s. **Polling is the fallback** when EventSource fails — back off to 10s GETs.

| Tab | Initial GET | What it shows | Row actions | Mutator endpoints |
|---|---|---|---|---|
| **Proposed Tasks** | `GET /proposals?tenant_id=…&user_id=…&status=pending,deferred&include_escalated=1` | `queue[]` sorted by `decay_score DESC`, plus `emergency[]` (USER×USER contradictions, visual emphasis), plus `escalated[]` (recurrence ≥ 5) | [Approve], [Defer], [Shelve], [Open drill-in] | `POST /proposals/{id}/convert` (Approve), `POST /proposals/{id}/decision` body `{action: defer\|shelve, until?: ts, reason?: str}`, `GET /proposals/{id}` (drill-in) |
| **Work Items** | `GET /work_items?tenant_id=…&user_id=…[&kind=task,proposal,lesson]` | UNION of `agent_tasks` (status in open/accepted/in_progress/review), `task_proposals` (pending/deferred), and `lessons` due-for-review. Each row tagged `_kind` ∈ {task, proposal, lesson}. Sorted cross-kind by `_sort_ts DESC`; UI re-groups. Optional `?kind=` narrows the UNION server-side (`task`, `proposal`, `lesson`, comma-separated, or `all` / omit for full UNION). | [Open drill-in] dispatches per kind | `GET /tasks/{id}` (task drill-in), `GET /proposals/{id}` (proposal drill-in), no direct mutator — drill-in screens own actions |
| **Lessons / Review Queue** | `GET /lessons/review_queue?tenant_id=…&user_id=…` | Lessons whose `review_after` has passed (post-M1 cutover: reads from TCMM archive via `memory.lessons_reader.find_lessons_due_for_review`, NOT the dropped `org_memory` Lance table) | [Keep], [Amend], [Retire] | `POST /lessons/{id}/decision` body `{action: keep\|retire, note?: str}`. "Amend" is client-side: edit trigger/rule in modal, then POST `keep` with `note` describing the diff |
| **Amendment Candidates** *(sub-tab of Lessons)* | `GET /lessons/amendment_candidates?tenant_id=…&user_id=…` | Lessons with `confidence ≥ 0.75` AND `reinforcement_count ≥ 5`. Sorted by confidence DESC. Threshold values returned in response for UI to display. | [Promote to constitution amendment] (manual today; future: amend tool with diff review) | No server mutator today — promotion is a manual operator workflow that edits `CONSTITUTION.md` and writes a `[lesson_amended]` observation to TCMM |

**Caching contract.** Per memory `workflow_admin_dashboard_caching`, sidebar reads MUST go through the 8s TTL `_windowed_rows()` cache pattern on the agent-runtime side — naïve `t.to_arrow()` scans on each refresh murdered the admin dashboard at ~480ms/scan. The cache key is `(endpoint, tenant_id, user_id, rounded_timestamp)`. The unified `/work_items` UNION specifically gets the `work_items_recent` 30-day projection per §3.10.2.

**Latency budget.** Sidebar GET round-trip should be < 400ms warm (≤ 250ms agent-runtime + ≤ 150ms transit). Drill-in GET (`/proposals/{id}`, `/tasks/{id}`) runs Lance reads in `asyncio.to_thread` per the 2026-05-27 fix — without that, an in-flight SSE stream on the same event loop starves the read and the sidebar shows HTTP 502.

**Dependency-tolerant errors.** All sidebar endpoints return `200` with `{error: "…", items: []}` when Lance is unreachable, NOT 500. The sidebar shouldn't go red over a transient ledger hiccup; it should render a "memory layer reconnecting" banner over an empty list. The single exception: a missing `tenant_id`/`user_id` IS 400 (it's a contract violation, not a backend issue).

**Open work** (not blocking Phase 4 ship, listed so the closeout is honest):
- ~~`lesson_created` / `lesson_status_changed` SSE events are TODO~~ — shipped 2026-05-28. `lesson_created` broadcasts from `phase_7_writers.promote_lesson_to_team_knowledge()` after the TCMM write persists. `lesson_status_changed` broadcasts from `main.lesson_decision()` after the observation lands. Both gated on the upstream write success so a failed write doesn't generate a phantom event. Verified end-to-end + cross-namespace isolation via `scripts/verify_lesson_sse.py`. Anchors: `[LESSON_SSE_2026_05_28]` comment in both call sites.
- `task_status_changed` events fire today but the EventSource reconnection logic in `useVeilguardAPI.ts` doesn't dedupe events received during reconnect window — possible double-render. Idempotent-render fix on the client side, not server.
- ~~Cross-kind filter (`/work_items?kind=proposal`) not implemented~~ — shipped 2026-05-28. `?kind=` accepts a comma-separated subset of `{task, proposal, lesson}` (or omit / `all` for the legacy UNION). Bad-kind returns 400 with a helpful error; missing query param defaults to all kinds (no breaking change). Saves one Lance scan per filtered-out kind. Anchor: `[WORK_ITEMS_KIND_FILTER_2026_05_28]` in `main.py:work_items_view`.

### 3.11 Memory write-path discipline (Phase 6 sub-deliverable)

> **AMENDED 2026-05-29 (channel correction).** Every TCMM writer now stamps a `channel` (top-level ingest item field, persisted to the `channel` column); the `_WriterDest` map + regenerated `MEMORY_WRITE_PATHS.md` carry a `channel` column (`agent_private` / `team_knowledge` / `team_events` / `n/a` for ledger-only). Routing is centralised in `observe_agent_output(..., channel=)`. See [TCMM_CHANNEL_ARCHITECTURE.md](TCMM_CHANNEL_ARCHITECTURE.md) §6.

*Added 2026-05-27 in response to external review.* Veilguard's memory topology has ~13 distinct destinations (7 agent-runtime ledger tables + 6 TCMM-side typed channels). The reviewer's concern: as agents start "choosing" memory destinations, entropy rises fast — developers and LLMs both lose track of "where should this thing live?" The fix is not documentation; it's narrowing the write surface.

**Rule.** Agent code does not write directly to a memory destination. Every memory mutation flows through one of 5-7 typed *writer functions* that decide destination internally via declarative rules. Linter blocks direct writes; tests verify each writer's destination contract.

**The writer functions** (target signatures; final names to be confirmed in implementation):

| Writer | Destination(s) it owns | Routing rule |
|---|---|---|
| `record_episode(...)` | TCMM `agent/<aid>/observations/<user_id>`, conversation memory | Default for raw observations during a turn |
| `promote_to_semantic(...)` | TCMM `team/knowledge/`, `org/blackboard/` | Critic-gated; carries `extracted_by`, source_kind, decision_ledger ref |
| `log_decision(...)` | `agent_tasks`, `task_comments` (SHA-chained), `proposal_outcomes` | Status transitions + review_decisions; never bypassed |
| `update_org_memory(...)` | `org_memory`, `reflective_heuristics` | Promoted lessons, dream-derived heuristics |
| `enqueue_dream_input(...)` | `dream_archive` (TCMM-side) | High-importance events crossing TCMM's importance threshold |
| `record_approval(...)` | `client_tool_approvals`, `client_tool_bypass` | Approval gate decisions only |
| `attach_artifact(...)` | `agent_tasks.outputs[]`, workspace path under task ownership | Builder outputs; respects workspace isolation (Phase 7) |

**Enforcement.** Lint rule rejects any module under `agent-runtime/app/` or `agent/` that imports a memory-destination client (Lance table handle, TCMM HTTP client, workspace fs writer) outside the writer-function module. Negative-fixture test catches violations.

**Documentation.** The flat `(writer × destination × trigger × transformation)` table is auto-generated from writer function signatures + their destination-routing rules — not hand-maintained. Generated artefact lives at `agent-runtime/docs/MEMORY_WRITE_PATHS.md` and is rebuilt on every Phase-6+ test run. If it falls out of sync with the writers, the test fails — the table cannot rot.

**Note on TCMM vs agent-runtime destinations.** Six of the thirteen destinations (the typed TCMM channels: `agent/<aid>/observations/`, `team/{events,knowledge,drafts}/`, `blackboard/`, derived dream-graph nodes) live in a separate codebase at `~/.gemini/antigravity/tcmm/TCMM/`. Writer functions calling into those destinations cross a process boundary (HTTP to `:8811`) — the writer-function discipline still applies on the agent-runtime side, but the TCMM-side write semantics are governed by TCMM's own `/ingest_turn` contract. See `architecture_tcmm_canonical_path` memory note.

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

- A2A endpoints exposed externally. API-key shipped in 5.0; OAuth + mTLS shipped in 5.1 (this subsection).
- External agents (e.g. a customer's CrewAI deployment) can delegate Tasks into Veilguard's org. Inverse: Veilguard's Director can delegate to external A2A endpoints.
- Small effort once Phase 3 internal A2A exists — mostly auth, rate-limiting, and tenant-scoped allow-lists.

#### Phase 5.1 — OAuth/mTLS auth layer for `/external/a2a/*`

*Added 2026-05-28 — closes out the external A2A auth specification.* The base Phase-5 transport (`agent-runtime/app/a2a_external.py`) ships API-key auth bound to a tenant via the Lance-backed `a2a_external_keys` registry. Phase 5.1 layers two additional auth methods on top — OAuth/JWT and mTLS — so customers can plug Veilguard into an existing identity infrastructure without provisioning per-tenant static keys. Implementation lives in `agent-runtime/app/a2a_auth.py:authenticate()`; auth methods are tried in fixed precedence and any one succeeding produces the same tenant-context row shape that the API-key path returns. Downstream call-sites (`/messages`, `/tasks`, `/inbox`) don't know which method authenticated.

**Auth precedence at request time** (strict order — first success wins):

1. **`Authorization: Bearer <jwt>`** → JWT verified via JWKS.
2. **`X-Client-Cert-Subject: CN=…`** → mTLS cert subject mapped to tenant.
3. **`X-API-Key: …`** → existing API-key registry lookup.

All three end at the same `a2a_external_keys` row — the row carries the rate-limit + agent allow-list config that downstream gates enforce. JWT/mTLS are *identity binders*; the registry row is the *authorization carrier*. This keeps "who are you" and "what can you do" cleanly separated.

**JWT path.** Operator sets `VEILGUARD_A2A_JWT_JWKS_URL` (PUBLIC keys only — server NEVER holds signing keys), `VEILGUARD_A2A_JWT_AUD`, `VEILGUARD_A2A_JWT_ISS`. The token `aud` claim MUST match the configured audience, `iss` MUST match the issuer, and the tenant-claim (default `sub`, override via `VEILGUARD_A2A_JWT_TENANT_CLAIM`) is used to look up the registry row by `tenant_id`. JWKS is cached 5 min in-process to avoid hitting the IdP on every request. Algorithms accepted: `RS256`, `ES256`, `PS256` — no `HS*` (symmetric → we'd need a shared secret, defeats the JWKS model) and no `none`. PyJWT is the implementation; the codebase carries a `VEILGUARD_A2A_JWT_INSECURE_UNSIGNED=1` dev override that decodes header+payload without signature check — this MUST NEVER be set in prod and the code emits a `WARNING` every call when it is.

**mTLS path.** TLS terminates at the reverse proxy (Caddy / nginx / GCP LB). The proxy sets `X-Client-Cert-Subject: CN=acme-corp,O=Acme,C=ZA` after validating the client cert chain. agent-runtime parses the header, extracts `CN`, and looks up `a2a_external_keys WHERE label = '<CN>' AND status = 'active'`. The TLS layer already verified the chain — we don't need to re-validate the cert here. Operator convention: when issuing an mTLS-bound peer access, set the registry row's `label` field to the CN. Cert rotation = issue new cert with the same CN; revocation = flip `status = 'revoked'`. The TLS terminator config is OUT OF SCOPE for this doc (it's standard nginx/Caddy mTLS) — but it MUST set the header before the request reaches agent-runtime or all mTLS requests 401.

**Defensive defaults — opt-in per gate.** Operator unset → that gate returns `None` and falls through. So a fresh deployment with neither `VEILGUARD_A2A_JWT_JWKS_URL` nor `VEILGUARD_A2A_MTLS_ENABLED` set runs **API-key only** with no behavioural change from Phase 5.0. Turn on what you need; leave the rest off.

**Rejected handshake — what the peer sees.** Any combination of "bearer present but JWT verify fails" / "mTLS header present but CN unknown" / "API key present but hash mismatch or revoked" → `401 Unauthorized` with `{error: "auth_failed"}`. We do NOT distinguish "no JWT presented" from "JWT presented but invalid" in the response (info-leak prevention). We DO log the failed method server-side so an operator can correlate spikes. Rate-limit failures are `429`, not `401` — auth succeeded, quota didn't.

**Audit trail.** Every authenticated request appends to `client_tool_approvals` (today reused; future: dedicated `a2a_external_audit` table) with `auth_method ∈ {jwt, mtls, api_key}`, `tenant_id`, `target_agent_id`, `request_ts`, `outcome`. Failed auths log to the same table with `outcome=auth_failed` so a misconfigured peer is visible without spelunking server logs.

**What this changes vs Phase 5.0.** Nothing user-facing on the rate-limit / allow-list / quota path. Same `a2a_external_keys` row, same enforcement, same `/external/a2a/*` URL surface. The only delta is HOW the caller's identity gets bound to that row. Adding either gate doesn't break existing API-key consumers; removing the API-key gate (operator decision when they're confident in JWT/mTLS coverage) is a separate config flip not addressed here.

**Anchors.**
- `agent-runtime/app/a2a_auth.py` — `verify_jwt()`, `parse_mtls_subject()`, `mtls_to_tenant()`, `authenticate()`
- `agent-runtime/app/a2a_external.py` — base API-key flow + `_resolve_key()` (line 81)
- Env-config block: `_jwt_enabled()`, `_mtls_enabled()`, `_expected_aud()`, `_expected_iss()`, `_tenant_claim_name()` (lines 56-74)
- Schema: `a2a_external_keys_schema()` — `label` field reused as mTLS CN-binding (line 64)

### Phase 6 — Hierarchical decomposition + acceptance-criteria gating + scaling tiers

The "make critics actually enforce completion + scale past 10 agents" phase. Produced 2026-05-27 by a 2-iteration research panel (Researcher + Evaluator + Critic, then Researcher + Critic adversarial second pass). Each iteration ran in parallel, finished independently, then was synthesised under the framework's own acceptance criteria — i.e. **the plan passes its own gate**.

The animating failure mode this phase closes: *"the LLM declared done halfway, no one noticed, the work shipped."* Every existing mitigation at lower phases — Critic personas, `review_decision` enum, `constraint_violations`, deliverable_spec freeform string — is prompt-level. Phase 6 makes done-gating **structural**: row-level constraints, mechanically-checkable acceptance criteria, server-side critic invariants, and explicit lease semantics so dead workers don't hold work hostage. This is the phase that justifies the org's existence past the demo-toy size.

#### Phase 6.0 — Acceptance criteria framework (the foundation)

Single most important change in this phase. Without it, every other tier-1 deliverable degrades to prompt engineering.

**New column on `agent_tasks`:**

```
acceptance_criteria: List[Struct{
  id          string  not null    # "AC-1", monotonic within task
  statement   string  not null    # human-readable assertion
  check_kind  string  not null    # enum, see below
  check_args  string  not null    # JSON, schema depends on check_kind
  required    bool    not null    # default true
  rationale   string              # ≤ 280 chars audit-trail
}>  not null                      # array MUST be non-empty
```

**`check_kind` enum (7 mechanical, 1 escape hatch — see §6.0.3 for the deferral of `llm_verify` to Phase 7):**

| kind | check_args | semantics | result on empty / missing |
|---|---|---|---|
| `claim_count` | `{predicate: str, op: ">="|"=="|"<=", n: int}` | Count claims in task's chain matching predicate, compare to `n` | `fail` if predicate matches nothing — never silently `pass` |
| `claim_predicate` | `{predicate: str, must_exist: bool}` | ≥1 claim must match JSONPath predicate | `fail` if no claim matches |
| `output_path_exists` | `{path: str, min_bytes: int = 1}` | File at path exists in task's output dir AND is ≥ min_bytes | rejects empty / stub files |
| `output_path_matches_regex` | `{path: str, pattern: str, flags: str = ""}` | File exists AND contents match regex | empty file with regex `.*` returns `fail`, not `pass` (anti-false-positive) |
| `output_path_jsonschema` | `{path: str, schema: dict}` | File parses as JSON and validates against JSONSchema | parse failure → `error`, not `fail` |
| `test_passes` | `{cmd: str, cwd: str, timeout_s: int = 60, expect_exit: int = 0}` | Run command, assert exit code; stdout/stderr captured into evidence | timeout → `error`, exit 127 (cmd not found) → `error`, not `fail` |
| `manual_user` | `{question: str}` | Gates to user; critic emits `undecidable`; Director must escalate | never auto-passes; explicit human gate |

**Iron rule:** every required AC's `check_kind ∉ {manual_user}` AND `≠ llm_verify` (deferred). At least one required AC per task must be mechanical. Director-side `create_task` rejects tasks that violate this rule. **LLM-only verification is forbidden as sole gate** — this is the anti-rubber-stamp guarantee.

**Three-state executor result** (not two): `pass | fail | error`. `error` means the check itself failed to execute (cmd not on PATH, file unreadable, regex compile error). `error` blocks the gate the same way `fail` does. Distinction matters for diagnostics: `fail` means "the artifact is wrong"; `error` means "we couldn't verify." Both → `changes_requested`.

**Evidence hashes.** Every executor returns:
```
{status: pass|fail|error, evidence: {path_sha256?, exit_code?, stdout_hash?, predicate_match_count?, ...}, reason: str}
```
The critic re-verifies evidence on review and rejects if mismatched (catches "builder partial-writes, crashes mid-flight, retry passes a *different* artifact under the same path" — silent corruption that AC-pass-on-file-existence alone misses).

#### Phase 6.0.1 — Hard-gate on `state='done'`

Atomic with Phase 6.0 — shipping the gate without the criteria framework gives false confidence (criticisms become prose-only rubber stamps, hard-gate fires on every reviewed task).

Lance / programmatic guard:
```
state transition to 'done' REQUIRES:
  review_decision = 'accepted'
  AND constraint_violations = []
  AND (all required AC results are status='pass')
```

Enforced at **two write paths**: (a) `ledger.tasks.update_status`, (b) any direct `observe()`/Lance write that mutates `state`. The duplicate-enforcement is deliberate: AC-10 in the Phase 6 test plan probes the second path specifically because it's the most common bypass route ("we fixed `update_status` but forgot the raw write").

#### Phase 6.0.2 — Director-side acceptance contract

`create_task` validates at insertion:

1. `acceptance_criteria` must be non-empty.
2. At least one required AC must be mechanical (`check_kind ∉ {manual_user, llm_verify}`).
3. If `expected_artifact != null`, ≥1 required AC must be of kind `output_path_exists` / `_matches_regex` / `_jsonschema` / `test_passes` whose `check_args.path` (or test target) is consistent with `expected_artifact`.
4. If `expected_artifact == null` (claim-only tasks: research, analysis), ≥1 required AC must be `claim_count` or `claim_predicate`. No "vacuous research task" with no observable.

These rules force Director to articulate "done" before dispatch — moves the judgment from worker self-assessment to mechanical post-condition. The Director's most common failure mode (vague brief, "use your judgment", trust the worker) becomes a schema violation.

#### Phase 6.0.3 — Deferrals (explicitly out of scope for Phase 6)

To keep tier-1 atomic and review-friendly, the following are **not** in Phase 6 even though the AC framework names them:

- **`llm_verify` check_kind.** Reintroduces the rubber-stamp risk Phase 6 exists to eliminate (LLM judging LLM, possibly same model family). Director must express ACs in the 7 mechanical kinds. Anything that can't be expressed mechanically isn't done. Re-evaluate in a future phase only after the mechanical kinds are proven insufficient for some real artifact class.
- **Explicit `depends_on TEXT[]` column.** The existing implicit DAG via `agent_tasks.inputs[]` (`F7_DEPENDENCY_AWARE_CLAIM_2026_05_26` at `app/workers/inbox_poller.py:307`) handles 5-10 agent producer→consumer ordering. Two representations create a consistency obligation. Promote to tier-2 (30 agents) where cross-lineage fan-in matters.

#### Phase 6.1 — Fresh-context critics

Critic dispatch path builds the LLM prompt from `(spec, acceptance_criteria, artifact)` only — explicitly excludes the producer's trajectory / chain-of-thought / per-turn comments. This is the **correctness foundation** that must ship before per-persona concurrency caps; otherwise scaling critic throughput just scales rubber-stamping.

Implementation surface: `agent/critic.py` prompt assembly. Structural enforcement: a regex-absence check in source guarantees no `producer_messages | producer_trajectory | parent_thread` field is wired into the critic's prompt builder under any flag (debug, fallback, anything). Negative-fixture unit test deliberately wires a leak; the test must catch it.

#### Phase 6.2 — Per-persona concurrency caps

Replace global `MAX_CONCURRENT_DISPATCHES = 4` with per-persona budgets:

```
PERSONA_CAPS = {
  "researcher":   8,
  "builder":      6,
  "critic-claim": 4,
  "critic-prose": 4,
  "phishing-analyst": 2,
  "threat-analyst":   2,
  "report-writer":    2,
}
```

Independence guarantee: a researcher saturated at 8/8 MUST NOT starve a builder claim. (Single shared semaphore keyed on persona acquired before persona-resolution is the wrong implementation — see AC-16 in the test plan.) The old global constant `MAX_CONCURRENT_DISPATCHES` must be **removed from the source**, not just bypassed — AC-14 in the test plan does a structural source grep (`regex absence`) to prove dead-code didn't ship.

#### Phase 6.3 — Lease TTL + heartbeats

Highest-risk-but-easy-to-miss change in Phase 6. The turn cap (`_ENABLE_TURN_CAP = True`, `app/workers/inbox_poller.py:786`) catches **live dithering** but does nothing for **dead workers** holding stale claims — turns aren't accruing on the corpse. At 5-10 agents this surfaces as either (a) orphan claims that sit forever or (b) duplicate workers re-claiming and double-writing on the same workspace path.

New table:
```
agent_task_heartbeats: List[Struct{
  task_id     string  not null
  worker_id   string  not null
  last_beat_at f64    not null
  lease_ttl_s  i64    not null    # default 300
}>
```

Worker writes a heartbeat row at every N turns (N = 1 is fine — heartbeats are cheap). `inbox_poller` startup sweep + per-cycle sweep auto-reclaim any task where `state IN ('open','accepted','in_progress','blocked')` AND `now() - last_beat_at > lease_ttl_s`. Reclaim writes an audit comment (`lease_expired: reclaimed from worker {worker_id} after {elapsed}s of no heartbeat`).

This is the architectural answer to the failure mode the existing `_force_cancel_on_timeout` (line 494) was a *symptom-treater* for. Phase 6.3 makes it root-cause-correct.

#### Phase 6.4 — Revision-priority lane

When a critic returns `changes_requested`, the IC's respawned revision task gets `is_revision = True`. Inbox-poller's claim SQL prefers `is_revision=True` rows when its persona's cap has open slots AND a revision is waiting. Otherwise: critic-revision sits behind fresh Director-emitted builds, never gets re-attempted, "done halfway" becomes "done forever-pending."

Single boolean column on `agent_tasks`. ~15 LOC of claim SQL change.

#### Phase 6.5 — Truncated-output marker (sibling-of-observe-silent fix)

The agent-loop bug we just closed (2026-05-27, `tcmm.py:317-326` + `memory_mcp.py:212-237`) has a sibling: tool outputs that get **truncated for response-size limits** look identical to complete outputs from the LLM's point of view. `read_file` / `web_search` / `inbox` results capped → agent reasons over a prefix → same epistemic failure class as observe-silent.

Every tool wrapper emits an explicit tail: `[TRUNCATED: <N> of <M> bytes shown — page or chunk before acting]`. Persona prompts get a one-line rule:
> If a tool result ends with `[TRUNCATED: ...]`, the response is incomplete. Either (a) call the tool again with pagination args, or (b) raise a `blocker_raised` comment and `submit_for_review` with what you have. **Do not reason over a truncated response as if complete.**

~10 LOC per wrapper + a 1-line persona-prompt edit.

#### Phase 6.7 — APR (Artifact Production Ratio) metric + circuit breaker

*Added 2026-05-27 after iter-3 panel.* Operationalises the §2 design principle "Tasks transform state; conversations are scaffolding" — without instrumentation that principle is decoration.

```
APR = artifacts_per_window / (LLM_tokens_per_window / 1000)
```

**"Artifact"** counts only state-mutation events: status_change, attach_output, observe (when persisted), approval response, dream node update, decision-ledger entry. **Does NOT count**: agent-to-agent messages, intermediate reasoning, internal critique narration.

**Healthy band** (calibrated from Magentic-One / ChatDev telemetry analogs): **0.5–2.0 artifacts per 1k tokens** at the system level.

**Collapse signal**: APR sustained **< 0.1 over a rolling 30-minute window**. At that ratio the system is talking to itself (the Auto-GPT / BabyAGI failure mode the strategic principle exists to prevent).

**Circuit breaker.** When APR < 0.1 for > 30 min:
1. inbox_poller stops dispatching new tasks.
2. In-flight tasks complete their current turn and are paused (not cancelled).
3. Sidebar surfaces a banner: "Veilguard halted: artifact production below floor (APR=<N>). Operator unblock required."
4. Director receives an `apr_circuit_breaker` event in its inbox.
5. Operator either acks (resumes dispatch) or cancels the offending task subtree.

**Implementation surface.** ~150 LOC: APR rolling-window calculator + Prometheus-style counter emission + inbox_poller circuit-check before dispatch + sidebar surfacing.

#### Phase 6.8 — Memory writer-function layer

*Added 2026-05-27 after iter-3 panel.* Implements §3.11 (Memory write-path discipline). Defines the 5-7 typed writer functions, the linter rule, and the auto-generated docs.

| Deliverable | LOC |
|---|---|
| Writer-function module under `agent-runtime/app/memory/writers.py` with `record_episode`, `promote_to_semantic`, `log_decision`, `update_org_memory`, `enqueue_dream_input`, `record_approval`, `attach_artifact` | ~250 |
| Linter rule (`tests/test_memory_write_paths.py`) — fails any module under `agent-runtime/app/` or `agent/` that imports Lance handles / TCMM HTTP client / workspace writers outside the writers module | ~80 |
| Auto-generated docs (`agent-runtime/docs/MEMORY_WRITE_PATHS.md`) — built from writer-function introspection on every test run | ~70 |

~400 LOC total. The linter is the hard gate: if a future agent writes to memory outside a writer function, CI breaks.

#### Phase 6.9 — Constitution schema with mandatory `evaluator_id`

*Added 2026-05-27 after iter-3 panel.* Converts the constitution from "natural-language governance sludge" (reviewer's phrase) into typed policy with deterministic checkers.

**Mandatory schema for every `constitution.json` entry:**

```jsonc
{
  "id":                 "string, unique within file",
  "kind":               "objective | constraint | default",
  "rank":               "int, conflict-resolution priority (lower = higher priority)",
  "metric_name":        "string",
  "comparison_op":      "<= | >= | == | in | contains",
  "threshold":          "number | string | list",
  "evidence_source":    "string — which audit log / eval / counter produces the metric",
  "evaluator_id":       "string — pointer to a deterministic check function in evaluators/ registry",
  "applicability":      "{ when this rule fires — task_kind, owner_id, source_kind, etc. }",
  "action_on_violation": "block | warn | log | reflect"
}
```

**`evaluator_id` is the iron-rule field.** Entries without an evaluator are **not policy — they're aspiration. Loader refuses to load them.** This is the discipline that prevents the constitution from drifting into vague natural-language sludge.

Phase 6.9 scope: schema definition + loader refuses-to-load behavior + 5-10 evaluators wired (the ones the existing constitution.json already implicitly relies on: approval-rate threshold, cost-ceiling, fairness-factor, source-kind-trust, USER×USER emergency lane).

Phase 7 follow-on: expand evaluator registry to cover every constitutional objective; deprecate entries that can't be evaluator-bound.

~200 LOC for schema + validator + initial evaluators.

#### Phase 6.10 — Repository abstraction with table tagging

*Added 2026-05-27 after iter-3 panel.* Prepares for the eventual LanceDB → PostgreSQL migration for `mutable_transactional` tables without forcing it now.

```python
class Repository(Protocol):
    table_name: str
    table_kind: Literal["mutable_transactional", "append_analytical", "vector"]
    backend:    Literal["lance", "postgres", "lance+postgres"]
    # CRUD methods abstracted; today all backends return "lance"

class TasksRepository(Repository):    table_kind = "mutable_transactional"
class CommentsRepository(Repository): table_kind = "append_analytical"  # SHA-chained, never updated
class ProposalsRepository(Repository):table_kind = "mutable_transactional"
class ApprovalsRepository(Repository):table_kind = "append_analytical"
class OrgMemoryRepository(Repository):table_kind = "mutable_transactional"
class DreamArchiveRepository(Repository): table_kind = "vector"           # TCMM-side
```

All current code is routed through Repository wrappers. Implementation stays Lance for now.

**Migration triggers** (instrumented as Prometheus metrics):
- `tasks` row count > 500K, OR
- `agent_tasks` p95 mutation latency > 100 ms, OR
- Any operation requires cross-row transactional guarantee.

When a trigger fires, the migration is a Repository-level swap (Postgres for `mutable_transactional`, Lance for `append_analytical`/`vector`). Two weeks of focused work, no agent-runtime code changes.

~300 LOC for Repository protocol + concrete adapters + metric instrumentation.

#### Phase 6.11 — Director interface skeleton: `route()` / `synthesize()` / `propose()`

*Added 2026-05-27 after iter-3 panel.* Bakes the future Phase-8 Router/Synthesis split as method-level interfaces now — without splitting the persona — so the eventual split is mechanical.

```python
class Director:
    async def route(self, signal: WorkSignal) -> RoutingDecision:
        """Pure routing: which persona owns this? Sub-1k-token call. Today: Sonnet. Future: Haiku."""

    async def synthesize(self, task: Task, child_outputs: list[ChildOutput]) -> FinalDeliverable:
        """Final synthesis: collect, summarise, decide. ≤8k-token call. Today: Sonnet. Future: Sonnet/Opus."""

    async def propose(self, ranked_candidates: list[Proposal]) -> list[ApprovedTask]:
        """Proactive-stream surface: select N from ranked queue for approval. Mid-weight call. Today: Sonnet."""
```

Today all three live behind the same persona prompt + model. Phase 8 swap target: `route` moves to Haiku, `synthesize` stays Sonnet, `propose` stays Sonnet, and the calls become independent dispatches.

**Trigger to actually split** (instrumented):
- Director prompt > 8k structured tokens per turn (p95), OR
- Director p95 latency > 2× the slowest specialist persona, OR
- Synthesis-correctness errors appear in eval suite (sub-agents' findings lost in Director synthesis).

~150 LOC for interface scaffolding + per-method telemetry. No behavior change today.

#### Phase 6.6 — Acceptance criteria for Phase 6 itself (the meta-gate)

The whole phase passes its own framework. **45 ACs total** (after iter-3 expansion: 32 original + 13 new for sub-phases 6.7–6.11), 43 mechanical, 2 `manual_user`. Each Phase-6 sub-deliverable has ≥1 required mechanical AC. Distribution:

| Sub | ACs | Coverage |
|---|---|---|
| 6.0 schema + executors | AC-1, AC-2, AC-3, AC-4 (7 not 8 executors registered), AC-5, AC-6, AC-7 (empty-input no-false-pass), AC-8 (sandbox), **AC-26 (evidence hash present)**, **AC-27 (3-state error blocks gate)** | migration + executor correctness + adversarial sad paths |
| 6.0.1 hard-gate | AC-9, **AC-10 (direct-observe bypass)**, AC-11, AC-12 (post-deploy invariant scan) | both write paths probed |
| 6.0.2 Director contract | embedded in AC-9 / AC-12 | rejection-at-insert validated |
| 6.1 fresh-context critic | AC-21 (prompt content), **AC-22 (structural source grep)**, **AC-23 (negative-fixture / tests the test)** | catches debug-flag leaks |
| 6.2 per-persona caps | AC-13, **AC-14 (regex-absence of dead global)**, AC-15, **AC-16 (starvation independence)** | catches stub-shipped config-only PRs |
| 6.3 lease + heartbeat | **AC-28 (heartbeat row appears)**, **AC-29 (orphan reclaimed after TTL)** | proves auto-reclaim fires |
| 6.4 revision lane | **AC-30 (revision claimed before fresh builds when both available)** | proves priority logic |
| 6.5 truncation marker | **AC-31 (read_file emits marker)**, **AC-32 (persona prompt mentions TRUNCATED)** | both halves of the fix |
| **6.7 APR + circuit breaker** | **AC-33** (APR counter emitted per dispatch), **AC-34** (rolling-window calculator handles boundary correctly), **AC-35** (synthetic-self-narration fixture triggers circuit breaker; deliberate 30-min low-APR load gets paused) | proves operational backstop fires |
| **6.8 Memory writers** | **AC-36** (linter test fails when a module imports Lance handle outside writers.py), **AC-37** (writer-to-destination map auto-generated; every writer routes to its declared destination only), **AC-38** (negative fixture: try to write to org_memory from a non-writer module → blocked) | catches direct-write bypass shipping |
| **6.9 Constitution schema + evaluator_id** | **AC-39** (loader refuses constitution entries missing `evaluator_id`), **AC-40** (every existing constitutional objective has a registered evaluator on load), **AC-41** (evaluator output is deterministic — same input twice → same verdict) | constitution converted from prose to typed policy |
| **6.10 Repository abstraction** | **AC-42** (every direct Lance table access in `agent-runtime/app/` is replaced by a Repository call; static-import audit), **AC-43** (migration-trigger metrics emitted — `repo.<name>.row_count`, `repo.<name>.p95_mutation_latency_ms`) | swap-readiness without forcing migration |
| **6.11 Director interface skeleton** | **AC-44** (Director exposes `route()`, `synthesize()`, `propose()` as separate awaitables; static-introspection check), **AC-45** (per-method telemetry emitted — `director.route.latency_ms`, etc., distinct from generic `agent_query`) | future split is mechanical |
| cross | AC-24 (end-to-end 5-fanout smoke), AC-25 (user sign-off) | demo asymmetry: flip gate off, prove unhardened path "ships done"; flip gate on, prove it doesn't |

The AC list is held in a companion file `agent-runtime/tests/PHASE_6_ACCEPTANCE.md` (created during Phase 6 implementation, not yet in this commit). It is the **release gate** — Phase 6 is "done" only if all 43 mechanical ACs run green in CI.

#### Phase 6 — totals & adoption ladder

**~1775 LOC total** (after iter-3 expansion), ordered:

| # | Sub | LOC | Ships with | Unlocks |
|---|---|---|---|---|
| 1 | 6.0 + 6.0.1 + 6.0.2 atomic | ~330 | each other (one PR) | the anti-rubber-stamp gate |
| 2 | 6.1 fresh-context critic | ~80 | standalone | correctness foundation for scaling |
| 3 | 6.8 memory writer-function layer | ~400 | with 6.0 (touches same write paths) | narrow write surface; lint blocks bypass |
| 4 | 6.9 constitution schema + evaluator_id | ~200 | standalone | typed governance; loader refuses untyped policy |
| 5 | 6.10 Repository abstraction | ~300 | standalone (no behavior change today) | swap-readiness for Postgres migration |
| 6 | 6.11 Director route/synth/propose interface | ~150 | standalone (no behavior change today) | Phase 8 Director split becomes mechanical |
| 7 | 6.2 per-persona caps | ~100 | after 6.1 (per iter-2 sequence) | real fanout throughput, safe because critic is fresh |
| 8 | 6.3 lease + heartbeats | ~40 | standalone | dead-worker recovery |
| 9 | 6.4 revision lane | ~15 | standalone | prevents critic-revision starvation |
| 10 | 6.5 truncation marker | ~10 + prompts | standalone | closes sibling-of-observe-silent class |
| 11 | 6.7 APR + circuit breaker | ~150 | last (depends on writers + lease + AC) | operationalises "tasks transform state, not conversations" |

**After Phase 6: 5-10 agents in parallel, structurally gated against premature-done, no orphan claims, narrow write surface, typed constitution, swap-ready storage, baked-in Director split interface, conversation-collapse circuit breaker.**

**Phase 7 (tier-2, ~30 agents)** — not detailed here; spawned naturally by Phase 6 success:
- `agent_teams` table + `agent_tasks.team_id` (aggregate cost rollup; per-team budget enforcement)
- New `team-lead` persona (mini-Director scoped to one team)
- Explicit `depends_on TEXT[]` column (now genuinely needed for cross-lineage fan-in)
- Stall detector worker (Magentic-One Progress Ledger pattern)
- Re-introduce `llm_verify` for artifact classes mechanical checks can't cover

**Phase 8 (tier-3, ~90 agents)** — aspirational; real precedent is the Anthropic C-compiler run (16 parallel Claudes, $20k, GCC as oracle), not the mythologised 90-agent OS-build:
- Manager-of-managers (`agent_teams.parent_team_id`, capped depth 3)
- Per-artifact-class oracle gates (non-LLM verifiers wherever possible)
- Token-bucket throttler in front of Anthropic (provider-rate-limit defense is ours, not theirs)
- `director_plans` table for Director memory-file pattern (Anthropic Research pattern when Director's own context exceeds ~150k tokens)
- Per-team kill switch
- **Runtime replay determinism** (new — added 2026-05-27 after external "you are building an OS, not an assistant" critique). Distinct from Phase 4's Replay v1, which reconstructs lineage trees for counterfactual UI. This is the harder version: given a complete snapshot of `(Task graph, approvals, tool outputs, Constitution version, TCMM memory snapshot at time T, alignment_weights version, dream-graph signals at T)`, can we reproduce the exact decision the Director or any agent made? Required because Phase 6 + 7 introduce subsystems that mutate system behavior over time (regret-weighted recalibration of `alignment_weights`, institutional memory consolidation in `org_memory`, dream-derived heuristics surfacing as proposals). Without runtime replay, the OS becomes unauditable: you can see *that* a proposal surfaced but not *why* — that's existential when autonomous proposal generation + self-recalibration are both active. **Trigger condition**: before any feature that mutates system behavior over time ships beyond Phase 6.9, runtime replay must produce deterministic reproduction of at least one full decision trace (proposal → critic decision → outcome → recalibration delta). Scope: extends Phase 4 Replay v1 from lineage-tree view to runtime determinism by adding (a) snapshot anchors at every Director decision point (anchor = content hash of all input substrates), (b) deterministic re-execution against a frozen snapshot, (c) reproduction tolerance bounded by LLM non-determinism — anchor allows comparing reasoning ENVELOPES, not exact tokens. ~600 LOC + 1-week per-anchor test scaffolding.

#### Phase 6 — design provenance

Produced by a 3-iteration adversarial research panel:

- **Iteration 1** (parallel): Researcher (Plan agent) literature review + tier ladder; Evaluator (Explore agent) codebase audit producing the 10-dimension gap matrix; Critic (general-purpose agent) acceptance-criteria framework. Reports converged on "row-level done-gating via mechanical ACs" as the highest-leverage tier-1 change.
- **Iteration 2** (parallel, adversarial against iter-1 plan): Researcher hunting tier-1 blind spots produced 3 new failure modes (evidence hashes, lease/heartbeat, revision lane) + 2 sequence corrections (atomic ACs+hard-gate, fresh-context-before-caps); Critic applied iter-1's own AC framework to the tier-1 plan, produced 25 ACs (later expanded to 32 with iter-2 Researcher additions), ran adversarial mapping of 10 "ships-looks-done-isn't-done" bypass patterns, all caught by existing ACs.
- **Iteration 3** (parallel, against external expert critique): A skilled external reviewer produced a detailed critique identifying 5 risks (Director cognitive overload, memory topology complexity, dream-cycle latency, constitution operational fuzziness, LanceDB long-term ceiling) + 1 strategic recommendation ("agents transform state, not conversations") + 1 prioritisation recommendation (approval gate / provenance / cache / ledger / replay before autonomy expansion). The panel re-fired against the critique: Researcher (Plan) ran prior-art research on each risk with concrete Veilguard responses; Evaluator (Explore) verified the critique's factual claims against the actual codebase (found ~3 stale claims — `rank_proposals` in persona frontmatter but not in `_ALL_TOOLS`, TCMM channels conflated with agent-runtime ledger tables, dream-cycle timing confused with DreamScanner poll interval); Critic gated the proposed spec additions against its own iter-1 AC framework, verdict `changes_requested` with 4 specific upgrades required. Iter-3 produced 6 new Phase 6 sub-deliverables (6.7–6.11 above, plus §3.11 memory write-path discipline) converting acknowledgment-only spec additions into concrete operational commitments. Phase 6 grew from ~575 to ~1775 LOC.

All three iterations are preserved in the agent-runtime audit log as cancelled tasks under tenant `69c4468a1fde1abc19c7835c` (the local dev user). Transcripts in `C:\Users\rudol\AppData\Local\Temp\claude\C--Users-rudol--veilguard\f5ed0511-8d78-4c80-8c49-0ccfffa96b3f\tasks\`.

**Properties the external review missed (surfaced by iter-3 Evaluator) and therefore not yet credited in §3:** SHA-chained `task_comments` (append-only anti-tampering integrity), explicit server-of-record trust-boundary discipline per field (§3.8.5), Magentic-One-style task_ledger + progress_ledger via `agent-runtime/app/ledger/task_progress.py` (Phase 4), per-tenant proactive-config pause (user can halt the proactive stream per-tenant), atomic lease semantics on `agent_tasks` (prevents concurrent dispatch by multiple workers). These are not new work — they're existing strengths the critique didn't account for, and they strengthen the Phase 6 design's foundation.

### Phase 7 — Close the TCMM-ledger duplication (tier-2 prerequisite)

*Added 2026-05-27 after a 2-iteration TCMM-unification panel (iter-4: 3 agents discovered the duplication; iter-5: 3 agents gated the boundary rule + 4 migrations).* Phase 7 was originally a sketch of "~30 agent scaling" deliverables. This sub-section is the first concrete Phase 7 work and a prerequisite for the team-leader work below: **before scaling to 30 agents, fix the memory topology so the existing ~10-table sprawl doesn't multiply.**

The boundary rule was added as a §2 design principle (above): TCMM holds knowledge-graph contributors; the ledger holds the state machine; observations may reference ledger PKs but never mirror ledger fields. This phase migrates the four destinations that violate that rule today.

#### Phase 7.1 — Four migrations

| # | Migration | Type | Cross-ref |
|---|---|---|---|
| **M1** | `org_memory` Lance table → TCMM `team/knowledge/<team_id>/` lane=`org_critic` with hard-include floor | full (delete the Lance table post-cutover) | none — TCMM-side only |
| **M2** | `proposal_outcomes`: numeric columns (`task_cost_usd`, `value_realized`, `regret_score`, `succeeded`, `computed_at_ts`) stay in ledger. **Narrative observations born in TCMM** (`source_kind="outcome_narrative"`) via `dream/prediction_record.py`. `objective_deltas_json` migrates with the narrative. | not-quite-a-split — narrative was never in ledger; this adds a new TCMM observation type alongside existing ledger work | bidirectional: ledger gains `tcmm_obs_id` column; TCMM observation carries `extras.outcome_id` |
| **M3** | `task_proposals`: migrate `proposed_brief` + `rationale` to TCMM `archive[source_kind="proposal"]`. Status index (`id`, `status`, `decided_at`, `decay_score`, `signal_type`, `signal_node_ids`, `impact_score`, `objective_alignment`, `constraint_violations`, `recurrence_count`, all timestamps, `shelf_reason`, `resulting_task_id`, `emergency_lane`, `proposed_deliverable_spec`, `constitution_version`) stays in ledger as thin Lance projection. | column-level split | bidirectional: ledger gains `tcmm_obs_id` column; TCMM observation carries `extras.proposal_id` |
| **M4** | `task_comments[kind=discussion / note]` → TCMM `agent/<aid>/observations/` (`source_kind="discussion"`). SHA-chained kinds (`status_change`, `review_decision`, `blocker_raised`, `blocker_cleared`) stay in ledger with chain. | split-by-kind | one-directional: ledger chain is immutable source of truth; TCMM observations are derivative; no back-ref needed |

The `tcmm_obs_id ↔ entity_id` cross-ref protocol is the load-bearing piece: ledger row stores `tcmm_obs_id` for cascade tracking; TCMM observation stores `extras.<entity>_id` for joining back. Even after TCMM's dream cycle compresses the observation into a summary node, the ledger row's `tcmm_obs_id` becomes a "this used to point at observation X, which may now be part of summary Y" trail — graceful degradation, not orphan corruption.

For M2 and M3, the writer functions (`record_outcome`, `record_proposal`) are the only sanctioned cross-ref sites — they write the ledger row, get its PK, post the TCMM observation with the PK in `extras_json`, then update the ledger row with `tcmm_obs_id`. All other call sites are blocked by the Phase 6.8 writer-function lint.

#### Phase 7.2 — Migration order

1. **M1** first — lowest risk, highest payoff. `org_memory` is the most clearly-misplaced table; the migration closes the Phase-3 incompleteness where `lessons.py` writes Lance but never observes to TCMM, and unblocks Critic-promoted lessons earning concept_gravity / bridge_score in the dream graph. ~80 LOC + 1-week parallel-write soak.
2. **M2** — adds the cross-ref pattern that M3 reuses. ~150 LOC (split writer + ledger schema column add + initial `prediction_record` integration).
3. **M3** — reuses M2's cross-ref pattern. ~200 LOC (split writer + schema column changes + status-index-only query path).
4. **M4** — must land **after** Phase 6.8 writer-function discipline. Without 6.8's lint, agents writing through the old `add_comment(kind=...)` path would route both kinds to one destination. ~150 LOC + careful migration of historical `task_comments[kind=discussion]` rows to TCMM observations.

Total Phase 7.1+7.2: **~580 LOC** spread across 4 PRs. Each PR ships independently and is gated by its own AC subset.

#### Phase 7.3 — Acceptance criteria (6 mechanical gates)

| AC | check_kind | What it verifies |
|---|---|---|
| **AC-P7.1** | `output_path_matches_regex` | The boundary rule text is present in `MULTI_AGENT_PLATFORM.md` §2 (`"TCMM is the knowledge-graph substrate; the ledger is the state machine"`). |
| **AC-P7.2** | `claim_predicate` | `lance_table_exists('org_memory') == False` post-M1 cutover. |
| **AC-P7.3** | `claim_predicate` | Post-M2/M3, ledger schemas for `proposal_outcomes` and `task_proposals` have no free-text content columns; only numeric/enum/FK/timestamp columns plus `tcmm_obs_id`. |
| **AC-P7.4** | `test_passes` | Status-index queries on `task_proposals` (e.g. `SELECT * WHERE status='pending'`) execute with zero TCMM HTTP calls — assert via mocked TCMM client `call_count == 0`. Operational hot path stays cold on TCMM. |
| **AC-P7.5** | `test_passes` | SHA-chain integrity over remaining `task_comments` kinds (`status_change`, `review_decision`, `blocker_raised`, `blocker_cleared`) verifies end-to-end after M4. The chain was not broken by removing the discussion/note rows. |
| **AC-P7.6** | `test_passes` | Runtime probe scans TCMM `archive` for observations written by `agent-runtime` and asserts every `source_kind` is in the allow-list `{lesson, proposal, outcome_narrative, discussion, agent_observation}`. Unknown source_kind from agent-runtime → fail. |

#### Phase 7.4 — What we explicitly rejected

Iter-5's Critic proposed 4 additional amendments that were folded back out after a forcing-function call:

- **M5 `alignment_weights[narrative]` migration** — premature. The recalibration narrative isn't a feature today; add the migration when the feature ships, not preemptively.
- **"Free text bound to a never-compress guarantee" clause** in the rule — edge case for a feature (operator justification on secret rotation) that doesn't exist. The main rule already covers it via "hash-chained audit lives in the ledger."
- **"References between TCMM rows are graph edges" clause** in the rule — true but obvious; TCMM-internal refs are TCMM's whole point. Not worth a clause.
- **`# boundary: split-writer` annotation discipline + fail-closed enum + 13 of the 21 ACs** — over-engineered. With 3 split writers (M2, M3 — and M5 when it eventually lands), the dual-write scanner hard-codes their function names. AC-P7.6 is the runtime backstop that catches violations without AST-introspection tooling we don't have.

These rejections are themselves logged in the decision log so future drift back into the amendments has a forcing function to argue against.

#### Phase 7.5 — Remaining Phase 7 work (sketched only)

After Phase 7.1-7.4 close the duplication, the remaining tier-2 work spawns naturally:
- `agent_teams` table + `agent_tasks.team_id` (per-team budget enforcement; aggregate cost rollup)
- New `team-lead` persona (mini-Director scoped to one team)
- Explicit `depends_on TEXT[]` column (cross-lineage fan-in)
- Stall detector worker (Magentic-One Progress Ledger pattern)
- Re-introduce `llm_verify` for artifact classes mechanical checks can't cover
- ~~Tiered micro-dreams resolving Q22~~ **Retired 2026-05-28** — see Q22 below; live benchmark showed 100-block cycle at 112s vs spec's "9-10 min" premise, killing the architectural justification

These ride on top of the cleaner memory topology Phase 7.1 establishes.

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
| **Agent declares done halfway (LLM rubber-stamps own output)** | **Critical** | **Phase 6.0 acceptance-criteria framework + 6.0.1 hard-gate.** `state='done'` is a Lance/programmatic constraint: required ACs all `pass`, `review_decision='accepted'`, `constraint_violations=[]`. ≥1 required AC per task is mechanical (`check_kind ∉ {manual_user, llm_verify}`). Director-side `create_task` rejects no-mechanical-AC tasks. Critics walk ACs one-by-one with evidence hashes; undecidable required AC ≠ pass. **Iron rule: LLM-only verification is forbidden as sole gate.** |
| Critic rubber-stamps by inheriting producer's trajectory blind spots | Critical | Phase 6.1 fresh-context critics. Critic dispatch builds prompt from `(spec, AC, artifact)` only. Structural source-grep guarantees no `producer_messages | producer_trajectory | parent_thread` field is wired in under any flag. Negative-fixture test catches debug-flag leaks. |
| Global concurrency cap serializes 5+ agent fanouts; one persona starves another | High | Phase 6.2 per-persona caps (`{researcher: 8, builder: 6, critic-*: 4}`). Independence guarantee: researcher saturation MUST NOT starve a builder claim. Old `MAX_CONCURRENT_DISPATCHES` constant removed from source (not just bypassed) — regex-absence AC catches dead-code shipping. |
| Dead worker holds stale claim; turn cap doesn't catch corpse | High | Phase 6.3 lease TTL + heartbeats. `agent_task_heartbeats` table; worker beats every turn; `inbox_poller` auto-reclaims tasks with `now() - last_beat_at > lease_ttl_s` and writes `lease_expired` audit comment. |
| Critic-revision starves behind fresh Director builds — "done halfway" → "done forever-pending" | High | Phase 6.4 `is_revision BOOLEAN` flag + priority claim. Inbox-poller's claim SQL prefers revisions when persona cap has open slots. |
| Truncated tool output reasoned over as complete (sibling of observe-silent failure) | High | Phase 6.5 explicit `[TRUNCATED: N of M bytes]` tail on every tool wrapper. Persona prompt rule: "If you see TRUNCATED, page/chunk or raise blocker — do not reason over a truncated response as if complete." |
| `acceptance_criteria` executor itself fails silently (cmd not on PATH, regex compile error) | High | Phase 6.0 three-state executor result: `pass | fail | error`. `error` blocks the gate the same as `fail`. Catches "always-pass" bugs in the gate code itself. |
| Builder partial-writes artifact, crashes, retry writes different content under same path; `output_path_exists` AC still passes | High | Phase 6.0 evidence hashes. Every executor returns `{status, evidence: {path_sha256?, exit_code?, stdout_hash?}}`. Critic re-verifies evidence at review time; mismatched hash → `changes_requested`. |

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

Q21. **Director role-splitting — Router/Synthesis/Strategy.** *(External reviewer raised; iter-3 panel resolved.)* Should Director eventually split into Router-Director (Haiku-class, pure routing, <1k tokens) + Synthesis-Director (Sonnet-class, owns final response + ledger + strategy)? **Decision: yes, but bake the interface in Phase 6.11 without splitting personas; resolve the behavioral split in Phase 8 when trigger conditions fire.** Triggers: Director prompt > 8k structured tokens per turn (p95), OR Director p95 latency > 2× slowest specialist, OR synthesis-correctness errors appear in eval (sub-agent findings lost in Director synthesis). **Resolved-by:** Phase 8 (behavioral split); interface skeleton resolved in Phase 6.11.

Q22. **Dream-cycle freshness pattern.** *(External reviewer raised; iter-3 panel proposed tiered micro-dreams; **retired 2026-05-28 by live benchmark.**)* Original premise — "TCMM's 9-10 min `run_cycle()` is the largest architectural pressure point" — is **obsolete**. Live trace (2026-05-28, 100-block cold-cache cycle on `bench_F_phase3_group`): `cycle_wall=112.3s` (~1.9 min, 46 LLM calls @ 0% cache hit, top stage `_run_causal_counterfactual_52h`=11s × 2 calls). With warm cache + larger workloads this stays sub-3-min. The proposed Q22 architecture (hot overlay graph + Pinot-style lambda merge at retrieval) is sized for a 10-min cycle and doesn't pay for itself at <2 min. **Decision: retire Q22.** No micro-dream / overlay work. If future benchmarks regress past 5-min the question reopens. The only surviving piece — event-triggered cycle firing on importance-sum threshold — becomes a small standalone optimization candidate inside the existing dream worker, not a new substrate.

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
| 2026-05-27 | **`observe()` no longer silently swallows TCMM-unreachable failures.** | `middleware/tcmm.py:observe_agent_output` returned `None` for both success and HTTP failure → `tools/memory_mcp.py:observe_tool` reported `_ok({"observed": True})` to the LLM even when nothing was written. LLM then re-checked task state, saw outputs[] empty, looped re-running observe(). Verified live: `task-6becb4f4e1dc` burned 100+ stream events / ~25 LLM turns before user force-cancelled. Fix: `observe_agent_output` now returns `bool` (True = persisted, False = any failure). `observe_tool` reads the bool and returns `_err("TOOL UNAVAILABLE: TCMM did not persist this observation. Do NOT retry observe()...")` with explicit fallback guidance (continue in-context OR submit_for_review + blocker_raised). Sibling failure mode: tool outputs silently truncated → reasoned over as complete. Phase 6.5 closes that. |
| 2026-05-27 | **`_ENABLE_TURN_CAP` enabled (was False).** | `workers/inbox_poller.py:786`. IC tasks cap at 25 LLM turns, critics at 10. Hit cap → runtime emits `max_iterations` event → `_force_cancel_on_turn_cap` writes audit comment distinguishing "turn cap" from wall-clock "dispatch_timeout". Activation was empirically deferred per the original plan; the 2026-05-27 demo loop justified flipping the switch — every observed non-convergence was dithering, not productive past ~12-15 turns. Cap is per-dispatch, not per-task lifetime — a task that legitimately needs multiple critic rounds gets fresh budgets each turn. |
| 2026-05-27 | **Phase 6 designed: row-level done-gate + acceptance-criteria framework.** | Produced by a 2-iteration adversarial research panel (Researcher + Evaluator + Critic, then Researcher + Critic re-applying their own framework). Iter-1 converged on "row-level mechanical AC gating" as the highest-leverage tier-1 change. Iter-2 added 3 missing failure modes (evidence hashes, lease/heartbeat, revision-priority lane), 2 sibling classes of the agent-loop bug (truncation, stale-claim), 2 sequence corrections (ACs+hard-gate must ship atomic; fresh-context-critics before per-persona caps), and dropped 2 deliverables to tier-2 (explicit `depends_on` column, `llm_verify` check_kind). Final Phase 6 = 6 sub-phases, ~575 LOC, 32 ACs (31 mechanical, 1 manual_user). Critic iter-2 verdict on the plan: `accepted` (every sub-deliverable has ≥1 required mechanical AC; iron rule satisfied). Implementation gated by AC-1..AC-32 in CI. |
| 2026-05-27 | **`llm_verify` check_kind deferred from Phase 6 to Phase 7.** | Reintroduces the rubber-stamp risk Phase 6 exists to eliminate (LLM judging LLM, possibly same model family). Director must express tier-1 ACs in 7 mechanical kinds only. If something can't be expressed mechanically, by Phase-6 definition it isn't done. Re-evaluate only after the 7 mechanical kinds are proven insufficient for some real artifact class. |
| 2026-05-27 | **Explicit `depends_on TEXT[]` deferred from Phase 6 to Phase 7.** | The existing implicit DAG via `agent_tasks.inputs[]` (F7_DEPENDENCY_AWARE_CLAIM_2026_05_26 at `app/workers/inbox_poller.py:307`) handles 5-10 agent producer→consumer ordering. A second explicit edge representation creates a consistency obligation between the two (`depends_on` and `inputs[]` could disagree → potential cycle one source sees but the other doesn't). Promote to Phase 7 where cross-lineage fan-in from sibling tasks not sharing an `inputs` edge becomes genuinely necessary. |
| 2026-05-27 | **`acceptance_criteria` + hard-gate `state='done'` must ship atomically.** | Hard-gate without structured ACs makes critics rubber-stamp on prose-only `deliverable_spec` — *false confidence is worse than no gate*. ACs without hard-gate means the framework is advisory; nothing prevents an agent from declaring done with a failing AC. Iter-2 Researcher caught this sequence bug. Treated as one ~330 LOC change covering 6.0 + 6.0.1 + 6.0.2 (schema migration + 7 executors with 3-state + evidence hashes + Lance constraint + Director-side `create_task` validation). |
| 2026-05-27 | **Fresh-context critics must ship before per-persona concurrency caps.** | Per-persona caps double critic throughput. If the critic still inherits producer trajectory (Phase 6.1 not yet shipped), that's doubled rubber-stamping. Fresh-context is the *correctness* prerequisite for the *throughput* upgrade. Iter-2 Researcher: "Fresh-context is the single biggest leverage point for the 'declared done halfway' failure — the critic is currently being convinced by the same trajectory that convinced the builder." |
| 2026-05-27 | **New §2 design principle: "Tasks transform state; conversations are scaffolding."** *(owner: Director persona contract; revisit: 2026-08-27 against APR telemetry)* | External reviewer's strategic recommendation: don't let the system become "agents talking to agents" as the dominant workload. Iter-3 panel converted it from acknowledgment to operational commitment via Phase 6.7 APR (Artifact Production Ratio) metric + circuit breaker. Healthy band 0.5-2.0 artifacts per 1k LLM tokens; sustained APR < 0.1 over 30-min window auto-pauses agent-to-agent dispatch. The principle is now observable, not aspirational. Calibrated against Magentic-One / ChatDev telemetry analogs. |
| 2026-05-27 | **§3.11 — Memory write-path discipline via typed writer functions, not flat documentation table.** *(owner: agent-runtime/app/memory/writers.py; revisit: when 14th destination considered)* | Iter-3 Researcher rejected the originally-proposed flat `(writer × destination × trigger × transformation)` table as "a read projection, not a write governance tool — rots the moment a 14th destination appears." Replaced with 5-7 typed writer functions (`record_episode`, `promote_to_semantic`, `log_decision`, `update_org_memory`, `enqueue_dream_input`, `record_approval`, `attach_artifact`). Hard linter rule rejects direct memory writes from any module under `agent-runtime/app/` or `agent/` outside the writers module. Flat table is auto-generated from writer signatures — cannot drift. Pattern source: MemGPT's narrow API surface (4 functions cover all writes) + ATLAS write/read separation. |
| 2026-05-27 | **Phase 6.11 — Director exposes `route()` / `synthesize()` / `propose()` as separate methods, behind one persona for now.** *(owner: agent/director.py; revisit: when trigger conditions fire — see Q21)* | Bakes the future Phase-8 Router/Synthesis split as a method-level interface now. Trigger conditions for the actual split: Director prompt > 8k structured tokens per turn (p95), OR Director p95 latency > 2× slowest specialist, OR synthesis-correctness errors in eval. Pattern source: Anthropic's lead-researcher pattern (Jun 2025) — the lead's token consumption became ~80% of total cost; baking the interface now is cheap insurance against the same outcome here. ~150 LOC. No behavior change today. |
| 2026-05-27 | **Phase 6.10 — Repository abstraction with `mutable_transactional | append_analytical | vector` tagging.** *(owner: agent-runtime/app/storage/repository.py; revisit: when any trigger metric fires)* | External reviewer flagged LanceDB as eventually-wrong for operational tables. Iter-3 Researcher: don't migrate now, architect for it. All Lance access routes through Repository wrappers. Migration triggers instrumented as Prometheus metrics: `tasks` row count > 500K, OR `agent_tasks` p95 mutation latency > 100 ms, OR any operation requires cross-row transactional guarantee. When triggered: 2-week swap of `mutable_transactional` tables to PostgreSQL primary; `append_analytical` + `vector` stay on Lance. Pattern source: Notion CQRS, Pinecone → Postgres migrations (2023-2024 postmortems), Weaviate hybrid at >100M objects. ~300 LOC for the wrappers + metric emission. |
| 2026-05-27 | **Phase 6.9 — Constitution schema with mandatory `evaluator_id`; loader refuses to load untyped entries.** *(owner: agent-runtime/app/constitution/loader.py; revisit: Phase 7 evaluator-registry expansion)* | External reviewer: "Constitution layer may become operationally fuzzy. Drift into vague natural-language governance sludge." Iter-3 Researcher response: every constitution.json entry must have (`metric_name`, `comparison_op`, `threshold`, `evidence_source`, `evaluator_id`). **`evaluator_id` is non-negotiable** — entries without an evaluator are not policy, they're aspiration; loader refuses to load them. Pattern source: Anthropic Responsible Scaling Policy (capability × eval × threshold × action tuple), OpenAI Model Spec (explicit conflict-resolution rank), NIST AI RMF as the negative example (typing without evaluators is theater). Phase 6.9: schema + loader + 5-10 evaluators wired. Phase 7: expand evaluator registry to cover every constitutional objective. ~200 LOC. |
| 2026-05-27 | **Phase 6.7 — APR (Artifact Production Ratio) metric + circuit breaker** operationalises the new §2 design principle. | See "New §2 design principle" entry above. The circuit breaker is the load-bearing piece: Veilguard now has a structural answer to the question "is this system actually doing work?" beyond looking at logs. |
| 2026-05-27 | **Existing strengths the external reviewer missed (surfaced by iter-3 Evaluator).** | Reviewer wrote a thorough critique but read some spec text as deployed reality (`rank_proposals` in `director.md` frontmatter is not in `_ALL_TOOLS` — it's a Phase 3 deliverable not yet wired), and conflated systems (the 13 memory destinations span agent-runtime AND TCMM, which is a separate codebase at `~/.gemini/antigravity/tcmm/`; the 9-10 min cycle is TCMM's `run_cycle()`, separate from agent-runtime's DreamScanner 600s poll). Strengths the critique missed: (1) SHA-chained `task_comments` (anti-tampering), (2) explicit server-of-record trust-boundary per field (§3.8.5), (3) Magentic-One-style `task_progress.py` ledger (Phase 4 already shipped), (4) per-tenant proactive-config pause, (5) atomic lease semantics on `agent_tasks`. These existing properties strengthen the Phase 6 foundation. |
| 2026-05-27 | **New §2 design principle: "TCMM is the knowledge-graph substrate; the ledger is the state machine."** *(owner: §2 + Phase 7.1; revisit: when M5 alignment_rationale or M6 secret-justification features are proposed)* | Produced by a 2-iteration TCMM-unification panel (iter-4 + iter-5). Iter-4 surfaced the critical finding that `org_memory` is **accidentally duplicated** vs TCMM's `team/knowledge/` channel — Phase-3 incompleteness, not design choice. The lessons module writes Lance but never observes to TCMM; the dream cycle reads TCMM's dream_archive but writes Lance org_memory; the loop never closes. The boundary rule (TCMM = knowledge-graph contributors, ledger = state machine, observations may reference ledger PKs but never mirror ledger fields, never dual-write operational state) aligns Veilguard with the universal production pattern: Magentic-One Task Ledger vs working memory, MemGPT/Letta main/recall/archival, Cognition Devin ("decisions must be reconstructible exactly — RAG is unsafe"), Anthropic Multi-Agent Research (Jun 2025: lead plan vs subagent findings), and event-sourcing's one-way projection (Kafka, Datomic, EventStore). Resolves into Phase 7.1 with 4 migrations and 6 ACs. |
| 2026-05-27 | **Phase 7.1 — 4 migrations close the TCMM-ledger duplication.** *(owner: agent-runtime/app/memory/writers.py + lessons.py + outcomes.py + proposals.py + comments.py; revisit: after each migration's AC subset passes CI)* | M1 `org_memory` → TCMM `team/knowledge/<team_id>/` (lane=org_critic, hard-include, ~80 LOC) — first because lowest risk and closes the duplication. M2 `proposal_outcomes` keeps numeric in ledger; narrative observations born in TCMM via `dream/prediction_record.py` with `tcmm_obs_id ↔ outcome_id` cross-ref (~150 LOC) — adds the cross-ref pattern M3 reuses. M3 `task_proposals` migrates `proposed_brief` + `rationale` to TCMM `archive[source_kind=proposal]`; status index stays in ledger (~200 LOC). M4 `task_comments[kind=discussion/note]` → TCMM `agent/<aid>/observations/`; SHA-chained kinds stay (~150 LOC) — must land after Phase 6.8 writer-function discipline so the lint enforces the routing split. Total ~580 LOC. |
| 2026-05-27 | **Rejected iter-5 Critic amendments (explicit forcing-function entry to prevent drift back).** *(owner: Phase 7 reviewer; revisit: only if a real feature surfaces that motivates the rejected amendment)* | Iter-5 Critic proposed 5 additional amendments to the Phase 7 plan. After a user-driven forcing function ("the panel is finding work for itself"), four were rejected: **(a) M5 `alignment_weights[narrative]` migration** — premature; recalibration narrative isn't a feature today; add when the feature ships. **(b) "Free text bound to never-compress guarantee" clause in §2 rule** — edge case for operator-justification on secret rotation, which doesn't exist; main rule already covers via "hash-chained audit lives in ledger." **(c) "References between TCMM rows are graph edges" clause in §2 rule** — true but obvious; TCMM-internal refs are TCMM's whole point. **(d) `# boundary: split-writer` annotation discipline + fail-closed enum + 13 of the 21 proposed ACs** — over-engineered; with 3 split writers (M2, M3, future M5) the dual-write scanner hard-codes function names; AC-P7.6 is the runtime backstop. Only the "reference vs mirror" addition from iter-5 Researcher + the cross-ref protocol from iter-4 Critic survived into the rule + Phase 7.1. The rejected items are documented here so future iteration must explicitly argue against this entry. |
| 2026-05-27 | **Evaluation lens: Veilguard is an organizational OS, not an assistant.** *(owner: any reviewer; revisit: never — this is foundational)* | Surfaced by an external review of the post-iter-3 spec: "The architecture now reads less like 'chatbot with helpers' and more like 'event-sourced cognitive operating system with semantic memory.'" Going forward, every feature is evaluated against OS properties — **replayability, authority boundaries, isolation, observability, deterministic state transitions, scheduler behavior, memory pressure, organizational throughput** — not chatbot properties (conversation quality, instruction-following, persona consistency). The chatbot properties still matter for the foreground UX (LibreChat is a chatbot), but the agent-runtime platform is graded against OS criteria. This reframing justifies: (a) Phase 6's structural completion gates over prompt engineering, (b) Phase 7's strict separation of state machine from knowledge substrate, (c) Phase 8's runtime replay determinism as a first-class requirement before self-mutating behavior ships, (d) the rejection of conversation-volume metrics in favor of APR. The single sentence that captures the lens: **"Tasks transform state through hash-chained, replayable, authority-scoped transitions; memory accumulates as a separate semantic graph; conversations are scaffolding for both."** |
| 2026-05-27 | **Subsystem-ROI design principle added to §2.** *(owner: every PR proposing a new subsystem; revisit: each Phase post-mortem)* | Companion to APR. APR measures the system; this principle requires per-subsystem ROI. Every subsystem in `agent-runtime/app/` (proposal_taxonomies, stance_arcs, reflective_heuristics, regret_weighting, future additions) must be defensible against: "What measurable outcome moves when this subsystem is on vs off?" Qualitative answers ("richer recall") = decoration, not infrastructure. Existing subsystems audited at Phase 6 post-mortem; new subsystems must pass before merging. Mechanically enforced via a `tests/SUBSYSTEM_ROI.md` companion file naming each subsystem + its metric + on/off measurement + removal threshold. The architecture is now elegant enough that preserving abstractions because they're beautiful is the dominant failure mode — this principle is the guardrail. |
| 2026-05-27 | **Phase 8 runtime replay determinism added as a Phase-8 sub-deliverable.** *(owner: Phase 8 implementation; revisit: before any self-mutating feature ships beyond Phase 6.9)* | Distinct from Phase 4 Replay v1 (lineage-tree counterfactual UI). Runtime replay determinism: given a complete snapshot at time T, can we reproduce the exact decision an agent made? Trigger condition: must produce deterministic reproduction of at least one full decision trace (proposal → critic decision → outcome → recalibration delta) before regret-weighted recalibration or institutional memory consolidation feature ships beyond Phase 6.9. Without it, autonomous proposal generation + self-recalibration produce an OS where you can see *that* a decision was made but not *why* — unauditable, which is existential for an organizational-OS framing. Scope: snapshot anchors at every Director decision point (content hash of all input substrates), deterministic re-execution against frozen snapshots, reproduction tolerance bounded by LLM non-determinism (compare reasoning envelopes, not exact tokens). ~600 LOC. |
| 2026-05-28 | **Phase 7 wire-up landed — all four split-writers active in production callsites.** *(owner: agent-runtime/app/proposals/{dream_scanner,outcomes,lessons,constitution_amendments}.py + app/main.py; revisit: when TCMM comes back online to verify tcmm_obs_id cross-refs populate)* | Phase 7.1's writers were defined 2026-05-27 but the production code paths still called the low-level `_props.create_proposal` / `write_lesson` / `_write_outcome` helpers, so `tcmm_obs_id` columns stayed NULL and no TCMM observations were created from agent-runtime. **2026-05-28**: migrated all five production callsites — `dream_scanner._scan_once`, `main.py:POST /proposals`, `constitution_amendments.propose_one`, `outcomes.compute_one`, `lessons.promote_one` — to await the Phase 7 split-writers (`record_proposal_with_content`, `record_outcome_with_narrative`, `promote_lesson_to_team_knowledge`). This required cascading async through three `run_one_cycle` worker functions; their tests were updated to match. Schema migrations also extended: `task_proposals` + `proposal_outcomes` now gain `tcmm_obs_id` via the new `_migrate_add_tcmm_obs_id` recreate-merge helper (the Lance SQL-DEFAULT parser rejects the cast-from-NULL form on current versions), and `main.py:_startup` now eagerly opens every table in `TABLE_SCHEMAS` so migrations apply on container restart rather than lazily on first write. Live verification (with TCMM down): dream_scanner emitted 6 proposals through the new path, ledger commits succeeded, `tcmm_obs_id=NULL` per spec's documented fallback. When TCMM is up, cross-refs will populate automatically with no further code change. Wire-up locked in against regression by `tests/runtime_health/test_phase_7_wire_up.py` (15 source-grep tests; AST-rejects any direct `_props.create_proposal` call in wired modules). |
| 2026-05-28 | **Phase 7 M1 final cutover landed — org_memory Lance table dropped.** *(owner: agent-runtime/app/memory/lessons_reader.py + main.py lesson endpoints + constitution_amendments.run_one_cycle; revisit: when retirement/decay/reinforcement features re-appear)* | The 1-week parallel-write soak was deemed unnecessary because production had zero lessons in org_memory (only test seeds existed). Cutover steps: (a) one-shot backfill migrated existing org_memory rows to TCMM via the M1 split-writer (1 candidate / 1 migrated); (b) `app/memory/lessons_reader.py` shipped, scanning TCMM `archive` Lance file directly for rows starting with `[lesson]` prefix and filtering by `user_id` column (tenant isolation per memory `architecture_tcmm_api.md`); (c) reader collapses multiple-observations-per-trigger into a single row with `reinforcement_count` = distinct extracted_by tags; (d) 5 main.py endpoints (review_queue, amendment_candidates, lesson_decision, work_items_view) migrated to TCMM reader; (e) `lesson_decision` now writes append-only `[lesson_kept]`/`[lesson_retired]` markers instead of mutating Lance rows in place; (f) `constitution_amendments.run_one_cycle` reads from TCMM via `find_amendment_eligible_lessons`; (g) legacy `write_lesson` call removed from `promote_lesson_to_team_knowledge` (TCMM-only); (h) `org_memory` dropped from `TABLE_SCHEMAS` + `REPOSITORY_REGISTRY` + on-disk (`db.drop_table`); (i) `update_org_memory` deprecated to asyncio-wrapped shim around Phase 7 writer for back-compat. Bonus bug caught: middleware hardcodes namespace `agent/<agent_id>/observations/<user_id>` ignoring caller's `team/knowledge/<team_id>` conv_id (`[F4c_LESSON_NAMESPACE_2026_05_28]`) — reader works around it via text-prefix + user_id filter. AC-P7.2 flipped from "transitional" to strict assertion: `'org_memory' not in TABLE_SCHEMAS`. 440 tests green. |
| 2026-05-28 | **Q22 retired (no micro-dreams).** *(owner: spec § Q22; revisit: only if benchmarks regress past 5-min cycle)* | Original premise — "TCMM's 9-10 min `run_cycle()` is the largest architectural pressure point" — is obsolete. Live trace from `bench_F_phase3_group` (100-block cold-cache cycle, `--no-cache`, AI Studio backend): `cycle_wall=112.3s` (~1.9 min), 46 LLM calls @ 0% hit rate, top stage `_run_causal_counterfactual_52h`=11s × 2 calls, full harness 151s incl. init. With warm cache + larger workloads this stays sub-3-min. The proposed Q22 architecture (hot overlay graph + Pinot-style lambda merge at retrieval) is sized for a 10-min cycle — at <2-min the operational complexity (two graph layers + query-time merge + cycle-fold coordination) doesn't pay for itself. **Decision: retire Q22 entirely.** No micro-dream / overlay work scheduled. If a benchmark ever crosses 5-min wall-time on representative workloads the question reopens. The only surviving piece worth implementing later is event-triggered cycle firing (importance-sum threshold → fire `run_cycle` early instead of waiting for poll) — that's ~50 LOC inside the existing dream worker, not a new substrate. |
| 2026-05-28 | **Phase 7.5 `llm_verify` check_kind landed under the paired-mechanical rule.** *(owner: app/acceptance/executors.py + tests/acceptance/test_llm_verify.py; revisit: when Critic dispatch wires `ctx['llm_judge']` to a real LLM adapter)* | Phase 6 deliberately deferred `llm_verify` per the iron rule (LLM-only verification can't be a sole gate — rubber-stamp risk). Phase 7.5 added it back under the **paired-mechanical rule**: `llm_verify` is in `CHECK_KINDS` (executable) but NOT in `MECHANICAL_CHECK_KINDS`, so the Director-side `create_task` validator (`ledger/tasks.py:130-143`) still requires ≥1 required AC of a mechanical kind. `llm_verify` becomes a *quality lift* — it can fail tasks the mechanical layer let through (catching prose quality, factual grounding, subjective ACs), but it cannot clear what the mechanical layer hasn't cleared. Executor takes `{rubric, target_path | target_text, model?, allow_large?}` and a `ctx['llm_judge']` callable; returns 3-state CheckResult with evidence carrying `rubric_sha + artifact_sha + verdict + confidence`. Cost guards: 4KB rubric cap, 50KB artifact cap (override via `allow_large=True`). Sandbox: `target_path` resolves via `_safe_path` so workspace escape attempts hit `error`, not pwn. Missing-judge / bad-verdict / judge-raised conditions all surface as `error` (block gate with diagnostic). 20 unit tests + iron-rule integration tests pass; existing AC-4 test updated to reflect the 8-kind registry (was 7). Critic dispatch must inject `ctx['llm_judge']` from a real adapter — left as a wiring TODO since the test stubs prove the contract end-to-end. |
| 2026-05-28 | **Phase 7.5 `depends_on TEXT[]` landed — cross-lineage DAG dependencies.** *(owner: app/ledger/schemas.py + tasks.py + workers/inbox_poller.py; revisit: when the Director-tool surface gains a `verify_depends_on_acyclic` exposure for plan-time validation)* | `parent_id` + `lineage_chain` model only single-edge subtask trees (producer-of-X chain).  Real Director workflows have fan-in: Researcher A + Researcher B → Builder consumes both → Critic reviews → Director synthesises.  Phase 7.5 added a `depends_on: list<string>` column to `agent_tasks` so tasks can declare cross-lineage preconditions explicitly.  Helpers `deps_satisfied(task) -> (ready, pending_dep_ids)` and `verify_depends_on_acyclic(task_id, depends_on, ...) -> (ok, cycle_path)` live in `ledger/tasks.py`.  Inbox-poller (`_poll_once`) gained a pre-claim guard at `[PHASE_7_5_DEPENDS_ON_2026_05_28]` that calls `deps_satisfied` before `_try_claim` and skips blocked tasks with a debug log naming the pending dep_ids — closes the race window where a Builder gets dispatched before its Researcher prerequisite completes.  `create_task` validates that every dep id exists in the same (tenant, user) scope and rejects self-loops at the boundary; full cycle detection is exposed via `verify_depends_on_acyclic` for Director plan-time checks.  Generic missing-column migration extended to also run on `agent_tasks` (the previous dispatcher routed agent_tasks ONLY to the struct-aware migrator, so additive nullable columns like `depends_on` and `emergency_lane` were silently missed until a forced restart).  13 unit tests + live end-to-end smoke (task B blocked on task A, cycle detected, phantom dep rejected).  Live verified: `agent_tasks` table grew to 105 rows post-migration with `depends_on` column added in-place. |
| 2026-05-28 | **Phase 7.5 `agent_teams` table + `team-lead` persona landed.** *(owner: app/ledger/teams.py + app/ledger/schemas.py + app/ledger/tasks.py + agents/team-lead.md; revisit: when Director gets a `create_team` MCP tool + cost rollup endpoint)* | The "organisational scaling primitive" the spec called out for Phase 7.5.  New `agent_teams` Lance table carries `(name, lead_agent_id, member_agent_ids, budget_usd, budget_cap, cost_attributed_cached_usd)`.  `agent_tasks` gains an optional `team_id` column.  CRUD module `app/ledger/teams.py` provides `create_team`, `get_team`, `list_teams`, `add_member`, `team_cost_attributed`, `budget_exceeded`.  `create_task` got a `team_id` param + three-stage guard at `[PHASE_7_5_TEAM_ID_2026_05_28]`: unknown team → ValueError; team status≠active → ValueError; cost rollup ≥ budget × cap → ValueError with the actual cost figures so Director can route the message back to the user.  `budget_cap` defaults to 1.0 (hard ceiling) and is operator-tunable per team (e.g. 1.2 = 20% slack).  `team-lead` persona added (`agents/team-lead.md`): mini-Director scoped to one team, owns within-team routing + the team's review queue, escalates cross-team / cross-budget / cross-constitution decisions to Director.  Persona registered in `VALID_OWNER_IDS`.  16 unit tests + live end-to-end smoke: team created with $50/cap=1.0, task assigned, budget check returned $0/$50 exceeded=False; forced $60 spend → budget_exceeded fired with proper figures; new `create_task` then rejected with `"team has already crossed its budget envelope (attributed=$60.00 >= ceiling=$50.00)"`.  Schema migration ran live: `agent_teams` table created from scratch + `team_id` column added to `agent_tasks` (107 rows backfilled with NULL via the generic missing-nullable-column helper). |
| 2026-05-28 | **Phase 7.5 follow-up sweep — six small finishers landed in one turn.** *(owner: many; revisit: with the focused Phase 8 replay-harness session)* | **(1) Phase 8.0 snapshot anchor primitive** (`app/replay/snapshot_anchor.py` ~250 LOC + 18 tests): `SnapshotAnchor` dataclass + `compute_anchor()` pure-function + `record_anchor()` persistence to `agent_tasks.extras_json.snapshot_anchors`.  Hash composes 7 inputs (task_graph, approvals, tool_outputs, constitution_version, tcmm_snapshot_ref, alignment_weights_ver, dream_graph_signals_ref) + decision_point + schema version — ts and extras deliberately excluded.  **Honest scope cap**: the replay harness (frozen-snapshot reconstruction + reasoning-envelope comparison) is the multi-session piece that's been explicitly deferred to a focused follow-up; this lands only the foundation primitive other parts will compose on.  **(2) `llm_judge` adapter wired into Critic dispatch** — `app/acceptance/llm_judge.py` provides `default_llm_judge` that lazy-imports the Anthropic backend; `run_check` auto-injects it for `llm_verify` when no `ctx['llm_judge']` is provided.  Backend-unavailable RAISES so the executor catches it as `error` (preserves the existing "missing judge → error" contract).  **(3) Director MCP `create_team` / `list_teams` / `team_cost_report` tools** added to `ledger_mcp.py` (3 new tools, tool count 14 → 17).  HTTP endpoints `/teams` and `/teams/<id>/cost` added to `main.py` — sidebar can now render the teams panel and the cost-consumed gauge.  **(4) Event-triggered dream cycle firing** — `DreamScanner` gained `notify_importance(score)` and `fire_now(reason)` methods + an `_wake_evt` raced against the `interval_seconds` timeout in `run()`.  Importance threshold is env-tunable via `VEILGUARD_DREAM_IMPORTANCE_THRESHOLD` (default 5.0).  Surviving piece of the retired Q22.  **(5) Lessons.py maintenance cleanup** — `find_lessons_due_for_review`, `expire_stale_lessons`, `apply_confidence_decay`, `reinforce_lesson`, `run_maintenance_cycle` collapsed into deprecation stubs that warn once and return no-op summaries (was 245 LOC of dead code referencing the dropped `lessons_tbl`).  **(6) AC-P7.6 audit endpoint test-fixture filter** — `agent/test/*` rows now skip by default; operator can re-include via `?include_test=1`; response carries `test_fixture_skipped` count.  Closes the 19-violation noise the M4 backfill exposed.  **All six landed in one turn**; tests touched: +18 (snapshot_anchor), +3 (ledger_mcp tool count), no broken tests. |
| 2026-05-28 | **Phase 7.5 closure — cost write-back + Director persona teams guidance + team_id propagation + snapshot-anchor decision-point wiring.** *(owner: app/ledger/tasks.py + app/runtime.py + app/middleware/tenant.py + agent/director.py + agents/director.md; revisit: with the focused Phase 8 replay-harness session)* | Four sub-deliverables shipped to close all remaining Phase 7.5 loose ends in one session: **(A) Cost write-back** — `app/ledger/tasks.py` gained `increment_cost_attributed(task_id, tenant_id, user_id, delta_usd)` which read-then-writes the per-task cost; `runtime.run_agent_query`'s turn-end hook calls it with `_cost_from_tokens(model, in, out, cc, cr)` whenever `task_id` is set.  Closes the team budget loop — `team_cost_attributed()` now sees live spend without a sweeper.  Guarded behind `if task_id:` so `/agent/query` chat turns aren't billed.  **(B) Director persona teams guidance** — `agents/director.md` got a dedicated "TEAMS (Phase 7.5 — organisational scaling)" section: when to form a team (sustained multi-task initiatives, not one-offs), team-budget discipline (`team_cost_report` before big decisions, warn at 80%, stop at 100%), team-lead delegation rule (route inbound to team-lead once the team has one; don't be the within-team router for a team that has its own lead).  **(C) `team_id` propagation** — `TenantContext` gained optional `task_id + team_id` fields; `set_tenant_context` and `run_agent_query` plumb them through; inbox-poller `_dispatch` reads `task_row['team_id']` and threads it to the dispatch context.  Downstream tool calls now see the team without re-reading the task row.  **(D) Snapshot anchor wiring at Director decision points** — `DirectorAgent.route / synthesize / propose` each call a new `_emit_anchor(decision_point, signal, task_id?)` helper that builds a `SnapshotAnchor` from the live `TenantContext` and persists via `record_anchor`.  Best-effort by design: no tenant context, replay package missing, or anchor write failure all silently skip so the Director decision is never blocked.  Anchor decision points (`director.route`, `director.synthesize`, `director.propose`) are aligned 1:1 with `DIRECTOR_METHOD_LATENCY_MS` buckets so a replay match has a corresponding latency measurement.  **Test totals**: +5 (team_id propagation) + +5 (cost write-back) + +5 (anchor wiring) = 15 net-new.  **Phase 7.5 declared closed**; the only remaining Phase 8 work is the replay harness itself (frozen-snapshot reconstruction + reasoning-envelope comparison + 1-week per-anchor test scaffolding per spec) which is honestly a focused session, not a part-of-a-sweep deliverable. |

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
