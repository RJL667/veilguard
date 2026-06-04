# Veilguard Connector Architecture

> Status: **design** (2026-06-03). Extends the working `services/connectors/`
> framework (Connector ABC + hint fan-out + shadow blocks + provenance
> envelope + SharePoint). North star: **Perplexity Computer**-class breadth
> (dozens of apps, read **and** write, custom remote MCP connectors) without
> losing Veilguard's structural differentiators — proactive TCMM recall and
> the PII-proxy egress boundary.

---

## 1. Where we are today

We already have more than a stub. The live framework:

| Layer | File | What it does |
|---|---|---|
| Type contract | `_base/types.py` | `Capability`, `Ref`, `UserContext`, `Snippet`, `Content`, `Chunk`, `RecallHit` |
| Connector ABC | `_base/base.py` | `hint / search / read / list / get_permissions / chunk / calibrate / healthcheck` |
| Recall fan-out | `_base/recall.py` | `gather_hints()` — parallel `hint()` with per-connector deadline + circuit breaker |
| Registry | `_base/registry.py` | process-wide `ConnectorRegistry`, `all_hint_capable()` |
| Shadow render | `_base/shadow.py` | `<shadow>` block format + the system-prompt fragment that teaches the LLM to read them |
| Provenance | `_base/envelope.py` | `wrap_with_provenance()` / `parse_envelope()` — `_veilguard` sidecar the PII proxy strips on egress |
| Parsing | `_base/parsing.py` | LlamaIndex `parse_to_text()` + `chunk_text()` (shared by all connectors) |
| Credentials | `_base/credentials.py` | `CredentialResolver` (Static / Http), `OAuthToken`, `ReauthenticationRequiredError` |
| First connector | `sharepoint/` | `connector.py` + `graph.py` + `server.py` (FastMCP/SSE, header-scoped identity) |
| Wired live | `tcmm-service/server.py` | `_augment_with_connector_hints()` fans out hints, renders shadow blocks into the recall path (env-gated) |

**This is the load-bearing insight:** the recall→shadow→dereference→ingest
loop already runs end-to-end. "Extending connectors" is therefore **(a) breadth**
(many sources), **(b) a few missing capabilities** (write, non-OAuth auth,
custom remote MCP, per-tenant/user enablement), and **(c) governance for
outward actions** — not a rewrite.

### What makes us different from Perplexity (keep this)

Perplexity Computer is **reactive**: the LLM decides to call a connector tool,
the tool returns JSON, the model synthesizes. Veilguard is reactive **plus
proactive**: `hint()` fans out at recall time and stages `<shadow>` candidates
into context *before* the model asks, and `read()` results auto-ingest into
TCMM so the second mention of a doc is a warm memory hit, not a re-fetch. That
hint/shadow/TCMM-fusion path is the moat. Every new connector should light up
**both** paths.

---

## 2. The Perplexity Computer model (what we're matching)

From research (sources at bottom):

- **Two connector classes.** *Prebuilt* (Gmail, Gcal, Outlook, Google Drive,
  OneDrive, Dropbox, Box, Notion, Linear, GitHub, Slack, Atlassian, Asana,
  Snowflake, Databricks, Shopify…) and *Custom remote MCP* (user supplies an
  MCP server URL + auth and Perplexity discovers its tools).
- **Read and write.** Connectors don't just search — they *act*: send email,
  create calendar invites, create/update Notion docs, assign Linear tickets,
  post to Slack.
- **Three auth modes.** OAuth 2.0 (user tools), API key (internal services),
  service account (enterprise / shared).
- **Scoping + admin control.** Connectors are Individual (private) or
  Organization (shared). Enterprise admins toggle which connector types members
  may add and can *require per-member auth*.
- **Permission-preserving.** Connectors respect the source's native ACLs — you
  only ever see what you're already authorized to see.
- **Transport.** Remote, internet-reachable MCP servers over Streamable
  HTTP / SSE; OAuth handled by the host, session maintained after authorize.
- **Governance baked in.** Audit logging, data masking, rate limiting,
  retention/staleness policy per source.

We have parity on permission-preservation, transport, and (uniquely) proactive
recall. We're **behind on**: breadth, write, non-OAuth auth, custom remote MCP,
per-tenant/user catalog + enablement, and write-action governance.

---

## 3. Target architecture

Five layers. The bottom three exist; the top two are the extension.

```
┌─────────────────────────────────────────────────────────────────────┐
│ 5. EXPERIENCE      sidebar "Connectors" tab · catalog · connect /     │
│                    disconnect · per-tenant admin toggles              │
├─────────────────────────────────────────────────────────────────────┤
│ 4. CONTROL PLANE   Catalog (connector types) · Tenant policy ·        │
│                    User connections + creds · Enablement resolver     │
├─────────────────────────────────────────────────────────────────────┤
│ 3. RECALL / ACTION recall: gather_hints → fuse → ACL filter → shadow  │
│    (EXISTS)         action: MCP tool call → read (ingest) / write     │
├─────────────────────────────────────────────────────────────────────┤
│ 2. CONNECTOR        Connector ABC · native impls · RemoteMCPConnector │
│    (EXTEND)         · declarative manifest loader                     │
├─────────────────────────────────────────────────────────────────────┤
│ 1. PRIMITIVES       types · auth · parsing · envelope · registry      │
│    (EXISTS, extend) + AuthSpec, write Content, ActionResult           │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.1 Layer 1 — primitive additions

**Auth becomes polymorphic.** Today the only path is per-user OAuth bearer.
Generalize so a connector *declares* what it needs and the resolver returns the
right credential:

```python
class AuthKind(str, Enum):
    OAUTH2   = "oauth2"      # per-user, 3-legged (Gmail, Slack, GitHub…)
    API_KEY  = "api_key"     # per-user or per-tenant secret
    SERVICE  = "service"     # tenant-wide service account (Snowflake, internal)
    NONE     = "none"        # open / public MCP server

@dataclass(frozen=True)
class AuthSpec:
    kind: AuthKind
    scopes: list[str] = field(default_factory=list)   # OAuth scopes
    # OAuth endpoints for the connection lifecycle (initiate/callback/refresh):
    authorize_url: str | None = None
    token_url: str | None = None
    # API_KEY / SERVICE: where the secret is keyed (env, vault ref, per-user store)
    secret_ref: str | None = None

class Credential:                # what a resolver returns, any kind
    kind: AuthKind
    bearer: str | None           # OAuth access token or API key as bearer
    headers: dict[str, str]      # arbitrary auth headers (x-api-key, etc.)
```

`CredentialResolver.get_oauth_token()` widens to
`resolve(user_ctx, connector, auth: AuthSpec) -> Credential`. Existing OAuth
behavior is the `OAUTH2` branch; `StaticCredentialResolver` and
`HttpCredentialResolver` keep working.

**Write becomes first-class.** Add to the type contract:

```python
@dataclass
class Action:                    # a write request the LLM wants to perform
    verb: str                    # "send_email", "create_doc", "assign_issue"
    args: dict[str, Any]
    confirm_token: str | None    # set after user confirms an outward action

@dataclass
class ActionResult:
    ok: bool
    ref: Ref | None              # ref to the created/modified resource
    summary: str                 # human-readable result for the LLM/user
    extra: dict[str, Any] = field(default_factory=dict)
```

### 3.2 Layer 2 — connector surface

Add `write()` to the ABC (default `NotImplementedError`, gated by
`Capability.WRITE`) and a class-level `auth: AuthSpec`:

```python
class Connector(ABC):
    name: str
    version: str
    auth: AuthSpec                       # NEW — declares auth requirement
    capabilities: set[Capability]
    # existing: hint / search / read / list / get_permissions / chunk / calibrate
    async def write(self, action: Action, user_ctx: UserContext) -> ActionResult:
        raise NotImplementedError(f"{self.name}: WRITE not supported")
```

Two ways to add a connector:

**(a) Native connector** — a Python subclass (the SharePoint pattern). Best for
sources with rich search, ACL APIs, or odd parsing (Graph, Slack, Gmail).
Highest quality `hint()`.

**(b) Declarative + Remote-MCP connector** — the scaling spine, and how we get
to "hundreds of apps" without hundreds of bespoke modules:

- **`RemoteMCPConnector`** wraps *any* remote MCP server URL. On register it
  does an MCP `tools/list`, exposes those tools straight through, and
  synthesizes `hint()` by calling a designated search tool (declared in the
  manifest, e.g. `hint_tool: "search_messages"`) — so even a third-party MCP
  server lights up our proactive recall path. This is direct parity with
  Perplexity "add a custom connector by URL."
- **Manifest loader** — a connector can be defined as data
  (`connectors/<name>/manifest.yaml`): `name`, `icon`, `auth` (kind + scopes +
  OAuth URLs), `capabilities`, optional `hint_tool`, `chunk` strategy, and
  either `mcp_url` (remote) or `python_class` (native). Adding a connector
  becomes config + (optionally) a thin client, not a from-scratch subclass.

```yaml
# connectors/gmail/manifest.yaml
name: gmail
icon: gmail.svg
auth:
  kind: oauth2
  scopes: [https://www.googleapis.com/auth/gmail.modify]
  authorize_url: https://accounts.google.com/o/oauth2/v2/auth
  token_url: https://oauth2.googleapis.com/token
capabilities: [hint, search, read, write, permissions]
hint_tool: search_messages
chunk: email          # one message/thread per chunk
impl: python_class:gmail.connector:GmailConnector
```

### 3.3 Layer 3 — recall & action (mostly exists)

- **Recall** is done: `gather_hints()` → calibrate → sort → (TCMM fuse + ACL
  filter) → `render_shadow_blocks()`. One change: fan-out must iterate the
  **per-user enabled** connector set (§3.4), not the flat process registry.
- **Action (write)** is the new flow. An MCP write tool →
  `connector.write(action)` → on success, auto-ingest the created resource
  (same read-through path) so it's immediately recallable, and emit an
  `ActionResult` the LLM relays. Writes pass through the **confirmation gate**
  (§5).

### 3.4 Layer 4 — control plane (the biggest new piece)

The flat process-wide registry can't express "tenant A allows Slack + Gmail;
user U connected Slack but not Gmail." Introduce three stores (Postgres —
aligns with the TCMM/ledger PG migration already in flight):

- **`connector_catalog`** — the app store: every connector *type* available in
  the build (from manifests). Static-ish.
- **`tenant_connector_policy`** — `(tenant_id, connector, enabled,
  allow_custom_mcp, require_member_auth, write_allowed)`. Admin-controlled →
  Perplexity's org toggles.
- **`user_connection`** — `(tenant_id, user_id, connector, status, scopes,
  cred_ref, connected_at)`. One row per user×connector they've authorized.

**`EnablementResolver.active_for(user_ctx) -> list[Connector]`** intersects all
three (catalog ∩ tenant policy ∩ user connections) and returns instantiated
connectors. Recall fan-out and the MCP tool router both ask it — so a user only
ever fires hints / sees tools for connectors they've actually connected, in a
tenant that allows them. Replaces `registry.all_hint_capable()` at the call
sites; the in-process registry stays as the type/instance cache.

### 3.5 Layer 5 — experience

A sidebar **"Connectors"** tab (reuse the existing Veilguard sidebar
front-door, `:3080/api/veilguard-client/*`):

- **Catalog grid** — available connectors (from `connector_catalog` filtered by
  tenant policy), each with connect/disconnect, status, last-sync.
- **Connect flow** — OAuth: redirect to provider, callback lands the token in
  LibreChat's `MCPTokenStorage` (the `HttpCredentialResolver` endpoint contract
  already documented in `credentials.py`). API key / service: a secret form.
- **Custom connector** — "Add MCP server": URL + auth kind + name + icon →
  writes a `user_connection` (or tenant-level) backed by `RemoteMCPConnector`.
- **Admin** — per-tenant toggles: enable connector types, allow custom MCP,
  require member auth, allow writes.

---

## 4. End-to-end lifecycle

```
ADD        admin enables "slack" for tenant  ──▶ tenant_connector_policy
CONNECT    user clicks Connect Slack ─OAuth─▶ token in MCPTokenStorage
                                            ─▶ user_connection(status=active)
RECALL     turn N: gather_hints(active_for(user)) ─▶ Slack.hint(prompt)
                 ─▶ calibrate ─▶ TCMM fuse + ACL filter ─▶ <shadow> in context
DEREF      LLM calls slack.read(ref) ─▶ Content ─▶ wrap_with_provenance
                 ─▶ PII proxy strips _veilguard on egress (LLM sees text only)
                 ─▶ auto-ingest: chunk → embed(raw) → upsert (tenant, tool_ref, etag)
RECALL+1   turn N+1: same content surfaces as a warm TCMM hit (is_live_hint=False)
ACT        LLM calls slack.write(post_message) ─▶ CONFIRM GATE ─▶ Slack API
                 ─▶ ActionResult ─▶ auto-ingest the new message ─▶ relay to user
```

ACL flows the whole way: `get_permissions()` tags `entry.acl` at ingest; recall
filters `user.principals ∩ entry.acl` so cached content can never leak across
the permission boundary even if the source ACL drifts.

---

## 5. Write path & confirmation gate (in scope from day one)

Reads are already safe (ACL filter + PII-proxy egress redaction over raw
storage). **Writes are outward-facing, model-initiated actions** — the highest-
risk surface in the whole system — so the write path is specified in full here,
not deferred. Decision (2026-06-03): build write into the first connector.

### 5.1 Write classification (server-side, authoritative)

Every write *verb* a connector exposes declares a `WriteClass`. This — **not the
model, not the tool annotation** — decides whether the confirmation gate fires:

| WriteClass | Meaning | Examples | Gate |
|---|---|---|---|
| `INTERNAL` | additive, reversible, stays in the user's own space | create private doc, save draft, upload to own Drive | auto-allow if tenant `write_allowed`; audited |
| `OUTWARD` | leaves the user's control / visible to others / hard to retract | send email, post Slack message, share a file, assign a ticket | **always confirm** |
| `DESTRUCTIVE` | deletes / overwrites / revokes | trash/delete file, overwrite doc body, remove access | **confirm + destructive styling**; tenant may forbid entirely (`allow_destructive`) |

These map onto the **MCP tool annotations** we advertise on each write tool
(`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`) so
annotation-aware clients (Claude, ChatGPT) also prompt. But per the MCP
project's own guidance, **annotations are hints, not guarantees** — a server can
lie. Veilguard therefore treats annotations as UX sugar and **enforces the gate
server-side** off the authoritative `WriteClass`. The model cannot bypass the
gate by mislabeling a tool.

### 5.2 Two-phase propose → confirm

```
1. PROPOSE   LLM calls gdrive.share_file(file_id, "alice@x.com", "writer")
             tool handler classifies OUTWARD → does NOT execute.
             Returns {status:"confirmation_required", confirm_id, summary,
                      diff, expires_at}.  No side effect yet.
2. SURFACE   Veilguard renders a confirmation card (sidebar/inline):
             "Share 'Q1 Plan' with alice@x.com as editor?"  [Confirm] [Cancel]
             The confirm token is minted by the UI/server — NEVER by the model.
3. EXECUTE   On Confirm → POST /connectors/confirm {confirm_id}
             → server runs the held action with the user's credential
             → ActionResult → auto-ingest the result → audit.
             On cancel/expiry → proposal dropped, nothing happens.
```

This is the structural defense against **prompt injection via shadow blocks**: a
poisoned external candidate can talk the model into *proposing* an email, but the
action cannot fire without an out-of-band human confirm. (This is exactly the
failure mode behind the Comet hidden-MCP-API takeover — see sources.)

### 5.3 Idempotency, audit, governance knobs

- **Idempotency** — each proposal carries key `(user, connector, verb,
  hash(args))`. Idempotent verbs (`idempotentHint:true`, e.g. set-permission)
  replay safely; non-idempotent verbs (send/append) get single-use confirm
  tokens so a double-click can't double-send.
- **Auto-ingest after write** — a successful create/update flows through the
  same read-through path so the new resource is recallable next turn.
- **`connector_audit`** — `(ts, tenant, user, connector, verb, write_class,
  ref, status, confirm_id)`, sibling to `pii_audit`. Every read and write;
  provenance only, no raw payloads (masking).
- **Tenant policy** — `write_allowed` (master, default **off**),
  `allow_destructive` (default off), `auto_confirm_internal` (default on),
  per-(user, connector) rate limit (token bucket; the recall circuit breaker
  already covers hint failures).
- **Scope minimization** — request the least-privilege OAuth scope tier that
  satisfies the enabled capabilities (read-only scope unless writes are on).

---

## 6. Connector catalog (parity roadmap)

Grouped by family (shared auth/client → cheap to add once the family's first
lands). ★ = highest leverage first wave.

| Family | Connectors | Auth | Notes |
|---|---|---|---|
| Microsoft 365 | SharePoint ✅, OneDrive, Outlook mail/cal, Teams | OAuth2 (Graph) | Graph client + parsing already built — siblings are cheap |
| Google Workspace | Gmail ★, Google Drive ★, Google Calendar | OAuth2 | richest "Computer"-style value (mail + files + cal) |
| Comms | Slack ★, Discord | OAuth2 | Slack = canonical hint() showcase (messages) |
| Dev | GitHub ★, GitLab, Jira, Confluence, Linear | OAuth2 | code/issues; high write value (assign/comment) |
| Storage | Dropbox, Box, Google Drive | OAuth2 | reuse `parse_to_text` for all docs |
| Knowledge | Notion ★, Confluence | OAuth2 | read + create/update docs |
| PM | Asana, Linear, Jira | OAuth2 | write-heavy (tickets) |
| Data | Snowflake, Databricks, Postgres | Service acct | tenant-wide, no per-user OAuth |
| Custom | **any remote MCP server** | OAuth2 / API key / none | `RemoteMCPConnector` — the long tail |

**First wave (★):** Gmail, Google Drive, Slack, GitHub, Notion + the generic
`RemoteMCPConnector`. That set covers most "Computer" demos and exercises every
auth mode and the write path.

---

## 7. Build phases (re-sequenced for Drive-first + write-from-start)

- **P0 — Primitive widening.** `AuthKind`/`AuthSpec`/`Credential`, `write()` on
  the ABC, `Action`/`ActionResult`/`WriteClass`, generalize
  `CredentialResolver.resolve()`. The write/confirmation primitives from §5
  (proposal store, `confirm_id`, `connector_audit`). Backward-compatible with
  SharePoint. *(backbone — no new connector yet)*
- **P1 — Google Drive, read path + write path.** The reference connector (§8),
  end-to-end: `hint/search/read(export)/list/permissions` **and** `write` with
  `create_doc` (INTERNAL) + `share_file`/`trash_file` (OUTWARD/DESTRUCTIVE)
  routed through the confirmation gate. Proves every primitive on one connector.
- **P2 — Confirmation UX + governance.** The confirm card (sidebar/inline),
  `POST /connectors/confirm`, tenant `write_allowed`/`allow_destructive`,
  rate limiting. Makes P1's write path safe for real use.
- **P3 — Control plane.** PG tables (catalog / tenant policy / user connection),
  `EnablementResolver.active_for()`, swap recall + tool router to use it. (Until
  this lands, Drive is enabled process-wide via the existing registry.)
- **P4 — Manifest + RemoteMCPConnector.** Declarative loader; wrap any MCP URL;
  synthesize `hint()` from a declared search tool. *(unlocks the long tail)*
- **P5 — Breadth.** Slack → GitHub → Gmail → Notion, reusing the family clients
  and the now-proven read+write+gate machinery.
- **P6 — Experience.** Full sidebar Connectors tab: catalog grid,
  connect/disconnect, custom MCP, admin toggles.
- **P7 — Freshness (optional).** Webhook/poll sync for hot sources; otherwise
  read-through cache + TCMM consolidation already handle staleness.

P0–P2 deliver one connector (Drive) doing everything safely; P3+ scales breadth
on the proven backbone.

---

## 8. Reference connector: Google Drive (first build)

The concrete worked example that shakes out the framework. Drive is a good
first pick: it reuses the existing `parse_to_text()` path directly, shares
Google OAuth with Gmail/Calendar (cheap follow-ons), and is permission-
preserving *by construction* (`files.list` only ever returns files the user can
already see).

### 8.1 Auth — least-privilege scope tiers

OAuth2 (Google). The tenant's enabled capabilities pick the scope tier:

| Capability set | Scopes |
|---|---|
| read only | `drive.readonly` + `drive.metadata.readonly` (for `permissions.list`) |
| + INTERNAL writes | add `drive.file` (manage only files the app created) |
| + OUTWARD/DESTRUCTIVE on arbitrary files | add `drive` (full) — gated, off by default |

`manifest.yaml`: `authorize_url=https://accounts.google.com/o/oauth2/v2/auth`,
`token_url=https://oauth2.googleapis.com/token`, token lands in LibreChat's
`MCPTokenStorage`, fetched by `HttpCredentialResolver`.

### 8.2 Read-side methods

- **`hint(prompt)`** → `files.list(q="fullText contains '<tok>' and
  trashed=false", corpora="user", orderBy="modifiedTime desc",
  fields="files(id,name,mimeType,modifiedTime,webViewLink)", pageSize=top_k)`.
  ⚠️ `fullText contains` matches **whole tokens only** — so tokenize the prompt,
  drop stopwords, and build the query from salient terms (don't pass the raw
  sentence). Drive returns no relevance score → `calibrate()` scores by rank
  ordinal × recency. `Snippet.content` = name + description; full text deferred
  to `read()`.
- **`search(query)`** → same query, returns `Ref`s.
- **`read(ref)`** → branch on `mimeType`:
  - Google-native (`application/vnd.google-apps.document/.spreadsheet/
    .presentation`) → `files.export(fileId, mimeType=…)` — Docs→`text/plain`
    (or `.docx` for fidelity then `parse_to_text`), Sheets→`text/csv`,
    Slides→`text/plain`. **Native files have no `alt=media` download** — export
    is mandatory; this is the #1 Drive integration gotcha.
  - Binary (PDF/docx/…) → `files.get(fileId, alt=media)` → bytes →
    `parse_to_text()` (existing LlamaIndex path).
  - ACL via `permissions.list(fileId,
    fields="permissions(type,emailAddress,domain,role)")` → principals:
    `user:<email>`, `group:<email>`, `domain:<domain>`, `anyone`. `etag` =
    `headRevisionId`/`md5Checksum`. Returns `Content` with text/acl/etag/title.
- **`list(path=folderId)`** → `files.list(q="'<folderId>' in parents and
  trashed=false")`.
- **`get_permissions(ref)`** → `permissions.list` mapped as above.
- **`chunk()`** → `chunk_text()` (512/64); special-case Sheets to row-grouped
  later.

### 8.3 Write verbs — class, API, MCP annotations

| Verb | Drive API | WriteClass | readOnly / destructive / idempotent / openWorld |
|---|---|---|---|
| `create_doc` | `files.create` (+ Docs API for body) | INTERNAL | F / F / F / F |
| `upload_file` | `files.create` (media) | INTERNAL | F / F / F / F |
| `update_doc` (app-owned, revisioned) | `files.update` | INTERNAL | F / F / T / F |
| `share_file` | `permissions.create` | **OUTWARD** | F / F / T / **T** |
| `overwrite_doc` | `files.update` (content) | **DESTRUCTIVE** | F / **T** / F / F |
| `move_to_trash` | `files.update trashed=true` | **DESTRUCTIVE** | F / **T** / T / F |
| `delete_file` | `files.delete` | **DESTRUCTIVE** | F / **T** / T / F |

`create_doc`/`upload_file`/`update_doc` auto-allow when tenant `write_allowed`
(still audited + auto-ingested). `share_file` and all DESTRUCTIVE verbs route
through the §5.2 confirmation gate.

### 8.4 MCP server (`connectors/gdrive/server.py`)

FastMCP/SSE, header-scoped identity via the `_RequestContextMiddleware` already
proven in `sharepoint/server.py`. Tools:

- read: `search_drive`, `read_drive_file`, `list_drive_folder` — results wrapped
  with `wrap_with_provenance` (PII proxy strips `_veilguard` on egress).
- write: `create_drive_doc`, `share_drive_file`, `trash_drive_file` — each
  carries MCP annotations from the table above **and** returns a
  `confirmation_required` proposal (never executes inline) for OUTWARD/
  DESTRUCTIVE verbs.

Files: `gdrive/{manifest.yaml, connector.py, drive_client.py, server.py,
requirements.txt}` + `tests/test_gdrive_*.py` mirroring the SharePoint test
layout (mock the Drive client; assert the gate fires for OUTWARD/DESTRUCTIVE).

---

## 9. Decisions

**Resolved (2026-06-03):**
- ✅ **First connector = Google Drive** (§8).
- ✅ **Write in scope from the start** — full propose→confirm gate (§5), not
  deferred.
- ✅ **Scaling strategy = both** — native for the high-value sources,
  `RemoteMCPConnector` + manifest for the long tail.

**Still open:**
1. **Control-plane store** — Postgres (consistent with the TCMM/ledger PG
   migration) vs. Mongo (LibreChat-native). *Recommendation: Postgres.*
2. **Drive write scope tier** — ship INTERNAL writes only first (scope
   `drive.file`, no gate needed beyond `write_allowed`), or full
   OUTWARD/DESTRUCTIVE (scope `drive`, gate live) in the same pass?
3. **Confirm UX surface** — inline chat card vs. the existing Veilguard sidebar
   (or both). Affects the `POST /connectors/confirm` client wiring.
4. **Google OAuth app** — one shared Veilguard Google Cloud OAuth client for all
   tenants vs. per-tenant client registration (matters for the consent screen /
   verification and the scope-tier story).

---

## Sources

- Perplexity Enterprise — App Connectors: https://www.perplexity.ai/enterprise/app-connectors
- Perplexity integrations 2026: https://www.partnerfleet.io/blog/perplexity-integrations-you-should-know-about-in-2026
- Perplexity Computer + MCP (architecture, auth, custom skills): https://aibuilderhub.dev/en/blog/perplexity-computer-mcp
- MCP connectors across ChatGPT/Claude/Perplexity (auth, transport, read/write tiers): https://truthifi.com/education/mcp-connection-guide
- Comet + MCP (agentic browser, connector list): https://medium.com/@jimmisound/the-end-of-tabs-how-perplexity-comet-mcp-turn-your-browser-into-a-real-agent-14405eaa2c10
- Comet MCP security (why write/confirmation governance matters): https://labs.sqrx.com/comet-mcp-api-allows-ai-browsers-to-execute-local-commands-dec185fb524b
- MCP tool annotations — risk vocabulary, hints-not-guarantees: https://blog.modelcontextprotocol.io/posts/2026-03-16-tool-annotations/
- Google Drive API v3 — files.list search query terms: https://developers.google.com/workspace/drive/api/guides/ref-search-terms
- Google Drive API v3 — files.list reference: https://developers.google.com/workspace/drive/api/reference/rest/v3/files/list
