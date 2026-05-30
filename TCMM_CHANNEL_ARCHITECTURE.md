# TCMM Channel-Aware Shared Memory — Architecture Spec

> Status: design, pre-implementation. Companion sequencing doc: the implementation plan at
> `~/.claude/plans/eventual-percolating-meteor.md`. This spec is the authoritative *design contract*;
> the plan is the *order of operations*. On adoption, §3.4/§3.4.1/§3.4.2/§3.11 of
> [MULTI_AGENT_PLATFORM.md](MULTI_AGENT_PLATFORM.md) are amended to point here.
>
> Incorporates the coverage-panel findings (run `wkx24xzsx`, 2026-05-29): C1–C3 (critical),
> H1–H7 (high), M1–M5 (medium) are resolved inline and marked `[Cn]/[Hn]/[Mn]`.

## 1. Context — what is broken

"Shared memory" in the multi-agent platform (team_knowledge / blackboard / team_events) is broken at the architecture level:

- **Writes mis-route.** `observe_agent_output` ([agent-runtime/app/middleware/tcmm.py:296](agent-runtime/app/middleware/tcmm.py)) hardcodes the namespace to `agent/<agent_id>/observations/<user_id>` and ignores the caller's `conversation_id`. Every shared-channel writer lands data in the *writer's private notebook*.
- **There are no channels.** TCMM recall defaults to `scope="user"` (cross-namespace, `_user_filter`) — one undifferentiated per-user pool ranked by relevance. "private" vs "team" vs "blackboard" are fictions; Critic-gating doesn't actually restrict recall.
- **The reader workaround leaks the abstraction.** `lessons_reader` full-scans the archive on `user_id` + `[lesson]` prefix with a "team_id==tenant_id==user_id" assumption; the `tcmm_obs_id` cross-ref is a fabricated content hash.
- **The substrate wall is incidental.** The dream combines knowledge across a user's namespaces (the emergent identity/concept/bridge nodes are the product differentiator). It's user-bounded only because `bulk_warm_user_archive` happens to be user-filtered; the ~30 cross-block dream loops have no user guard.

**Goal:** make channels a real first-class dimension **without fragmenting the dream substrate**. One dream per collaboration unit (v1 = per `user_id`); channels = a provenance tag + read-time visibility filter + recall weight; promotion = re-author a clean shared block; recall = a bounded always-on compiled digest + on-demand scoped search.

## 2. Decisions

- **D1 — `channel` is a first-class, nullable column on TCMM archive blocks.** Not overloaded onto `extracted_by` (author-provenance, load-bearing for cross-agent contradiction detection) or `namespace` (cache key + private path; overloading re-fragments the substrate).
- **D2 — `namespace` stays per-conv/per-agent-private (cache-stable); `channel` is the recall axis.** Preserves the Anthropic/Haiku prefix-cache key.
- **D3 — One dream substrate per `user_id`; tenant = `user_id` is the hard wall** (TCMM has no `tenant_id` column). Generalizes to per-`team_id` later via a filter swap.
- **D4 — Promotion to a shared channel = re-author a fresh, clean-lineage block** (Critic-gated). Never flag-flip a private node → no private-derived synthesized text leaks; shared tier stays small/high-signal.
- **D5 — Dream nodes carry a derived channel-**set** stamp `_channels`** (the set of channel tags present across the node's lineage, mirroring how `source_block_ids` is stored), maintained by **union** forward and backfilled for existing nodes (§4). Registered as a **promoted, queryable Lance column** `[H3]`.
- **D8 — Visibility is one swappable policy** read at the recall chokepoint: `subset` (conservative, default — node visible iff *all* its lineage channels are in scope) or `intersection` (permissive — visible iff *any* are). Flippable without touching the stamp or the dream (§4.2).
- **D6 — Recall = always-on compiled digest (bounded, cache-stable, versioned) + on-demand scoped hybrid search.** Channel filters, query ranks, budget caps. Never dump a channel.
- **D7 — Substrate wall = in-loop `user_id` guards on every dream cross-block loop + single-user-per-instance assertion** (SQL filters do not reach in-memory dict iterations).

## 3. The channel model

### 3.1 Channel enum
Underscore form (enum/API/JSON), matching §3.4.2 of the platform doc:

`agent_private`, `conv`, `team_drafts`, `team_events`, `team_knowledge`, `user_deliverable`, `org_blackboard`.

`[H5]` `user_deliverable` is included — it is a Critic-gated `submit_for_review` target (§3.11 of the platform doc) and previously had no channel home. The §3.4.2 identifier table and the §3.11 submission-target enum are reconciled against this single list (P7) so they cannot drift.

### 3.2 Channel tier-groups `[H4]`
No numeric ordering is needed — channels are named **sets**, and visibility is a set test (§4.2), not a rank comparison. Every channel belongs to exactly one group (a unit test asserts the partition is **total** over §3.1's enum, incl. `conv`):

| group | channels | meaning |
|---|---|---|
| **NON_SHAREABLE** | `agent_private:<aid>`, `conv`, `team_drafts`, `__truncated__` (sentinel) | never surfaces in a *shared* scope; visible only to its owner / own-conv scope |
| **SHAREABLE_TEAM** | `team_events`, `team_knowledge`, `user_deliverable`, `org_blackboard` | recallable in a team scope |
| **DIGEST_TIERS** (⊂ SHAREABLE_TEAM) | `team_knowledge`, `user_deliverable`, `org_blackboard` | additionally eligible for the always-on digest (§5.1) |

**Two distinct gates — do not conflate** `[G2]`: *shared-visible* (recallable by a teammate) and *digest-included* are separate set tests against SHAREABLE_TEAM vs DIGEST_TIERS — so `team_events` is shareable-to-team but **not** in the digest. A unit test asserts a `team_events`-only node is shared-visible yet excluded from the digest.

### 3.3 Block-level storage `[H3]`
`channel` (raw blocks, nullable utf8) and `_channels` (dream nodes, list utf8 — the lineage channel-set) are added to `_PROMOTED_KEYS` and `_archive_schema` in [core/providers/lance.py](../.gemini/antigravity/tcmm/TCMM/core/providers/lance.py), with an idempotent `add_columns` migration mirroring the `extracted_by` migration (`lance.py:1469-1484`). Without promotion they spill to `extras_json` and cannot be used in a Lance WHERE pushdown — silently degrading the fast path to a full Python scan. Tests assert both are filter-pushdownable.

## 4. Dream-node channel-awareness — derive-and-stamp (the crux)

Dream nodes have **no channel of their own** — each is a composite of `source_block_ids` spanning channels. We do not compute channel at read time (O(nodes × sources), too slow); the dream's *synthesis* logic is unchanged. We **derive a channel-set from the source blocks and persist it on the node, mirroring how `source_block_ids` is stored**:

- **`_channels`** — the set of channel tags present anywhere in the node's lineage (e.g. `{team_knowledge, agent_private:researcher}`). `agent_private` entries carry the owner (`agent_private:<aid>`), so per-agent ownership is folded into the set — no separate owner field.

### 4.1 When the stamp is written
1. **Forward / incremental (monotonic via union).** At the `source_block_ids` merge sites ([dream_engine.py:1264/2488/3020](../.gemini/antigravity/tcmm/TCMM/core/dream/dream_engine.py)): when a block joins a node, `_channels |= {block.channel}`. Sets only grow; no recompute, no reverse index.
2. **Backfill (existing nodes) — conservative.** A one-time pass walks every `dream_archive` node, resolves `source_block_ids`, and unions their channels. The dream **destructively caps** `source_block_ids` at 30 (`dream_engine.py:2492`) / 50 (`:3024`) with no reverse index, so backfill cannot trust a complete lineage `[C2]`. When in doubt, **add the `__truncated__` sentinel** (a NON_SHAREABLE member) so the node fails any shared-scope *subset* test:
   - lineage at/near the cap (≥30/≥50) → add `__truncated__` `[C2]`;
   - a `source_block_id` that fails to resolve (deleted/superseded) → add `__truncated__` `[H6]`;
   - a `source_block_id` that is itself a **dream-node id** (`10000xxx` range; `source_block_ids` is `list(int64)` but the merge writes sorted *str* ids `[M5]`) → **union that node's `_channels`** (recurse), don't treat it as a raw block;
   - unknown/NULL source channel → add `__truncated__`.
   - **Net: any uncertainty adds a non-shareable member**, so under the conservative policy the node stays private; historical nodes re-share only via D4 re-author. (The sentinel protects the *subset* policy only — under *intersection*, one shareable child still shares the node; that's the policy's accepted cost, §4.2.)

### 4.2 How recall uses it — one chokepoint, one swappable policy
`get_archive_entry` ([core/archive.py:51-114](../.gemini/antigravity/tcmm/TCMM/core/archive.py)) is the one resolver every read path funnels through (search materialization, graph-link expansion `recall_graph._expand_*`, traverse, digest). For a raw block it reads `channel`; for a dream node it reads `_channels`; then it applies **one policy** against the requester's visible-channel set `S` (from `_recall_channel_var`, built from the §5.3 agent→channel map):

- **`subset` (conservative, default):** visible iff `node._channels ⊆ S` — *every* lineage channel is in scope. A single private/conv/drafts member (or the `__truncated__` sentinel) excludes the node from any shared scope. Safe; the shared compiled layer can be sparse (filled by D4 re-author).
- **`intersection` (permissive, opt-in):** visible iff `node._channels ∩ S ≠ ∅` — *any* lineage channel in scope. Richer shared recall, but a node fusing a private child with a shared child becomes shared-visible → private-derived synthesized text can leak **and** the curated tier can be polluted. Accepted cost; choose deliberately.

`VISIBILITY_POLICY` is a single config constant read here — flip `subset`↔`intersection` **without touching the stamp or the dream**. NULL/empty `_channels` ⇒ treat as `agent_private` (never shared). Retrieval pre-filters on the column; `get_archive_entry` re-enforces (defense in depth + covers traversal). The **digest** (§5.1) is a second, independent test: `_channels ⊆ DIGEST_TIERS` (subset) / `∩ DIGEST_TIERS ≠ ∅` (intersection).

## 5. Recall pipeline

### 5.1 Always-on compiled digest
- **Feeds:** Critic-promoted `team_knowledge` / `org_blackboard` / `user_deliverable` blocks + dream `semantic_principle` / identity-anchor nodes that pass the digest test (`_channels` vs DIGEST_TIERS, §4.2) (blackboard = a *view* over dream-compiled state, per §3.4:528, not a write target).
- **Bounded:** N principles + M facts, ranked by recall-weight × recency, deterministic id tiebreak.
- **Cache-stable + versioned `[C1]`:** `digest_version = hash(sorted(node_ids) ++ per-node rendered-text fingerprint)`. It **excludes** `reinforcement_count`, `stability`, `_last_reinforced_cycle`, and `updated_ts` — these are bumped every cycle a node is reinforced (`dream_engine.py:1106-1196 update_reinforcement` sets `isDirty` independent of membership), so an `updated_ts`-keyed hash would flip every cycle and bust the prefix cache the channel split exists to protect (D2). Recompute only when the node *set* or a node's *rendered text* changes. Rendered as a dedicated cache-control block; flag-gated flip; canary first.
- **Canary test `[C1]`:** reinforce a digest node, re-render, assert `digest_version` unchanged AND prefix-cache hit.

### 5.2 On-demand scoped recall
Re-enable the scoped `recall` tool ([agent-runtime/app/tools/memory_mcp.py:284](agent-runtime/app/tools/memory_mcp.py)). The digest is always-on context; the tool is for *explicit, deeper, scoped* pulls beyond it — channels make the tool's scope distinct from the digest (resolves the 2026-05-25 "two paths to memory" retirement reason). `auto` is gated to avoid redundant calls.

### 5.3 Canonical scope→channel table `[M4]`
ONE source of truth, in [core/recall/scope.py](../.gemini/antigravity/tcmm/TCMM/core/recall/scope.py) (or a constants module), consumed by `memory_mcp.recall_tool`, `/recall`, and `/render_structured` — they must not each invent their own mapping:

| agent / scope | channel frozenset |
|---|---|
| `director` | `{conv, team_knowledge, org_blackboard, user_deliverable}` |
| `researcher`, `builder` (ICs) | `{agent_private:self, team_knowledge, org_blackboard}` |
| `critic-prose` | `{agent_private:self, team_knowledge, org_blackboard}` |
| explicit `team_knowledge` | `{team_knowledge}` |
| explicit `blackboard` | `{org_blackboard}` |
| explicit `agent` | `{agent_private:self}` |
| `auto` | the calling persona's row above |

`observe_tool` inherits the writer default (`agent_private`); it does **not** take a channel arg (promotion is the only shared-write path, D4) `[M4]`.

### 5.4 Channel scoping lives in RECALL, not render `[C3]`
Channel filtering is a **recall** concern. The renderer's `render(task_query)` ([base_renderer.py:239](../.gemini/antigravity/tcmm/TCMM/core/renderers/base_renderer.py)) takes **only `task_query`** — no scope, no channels, no agent_id. Recall scope is established on a contextvar (`_recall_scope_var`, plus the new `_recall_channel_var`) **bound before recall runs**, then enforced in `get_archive_entry`. Consequences:
- **No `RenderBody` / `render_structured` surgery.** Render is downstream and unchanged; it inherits channel-filtered results because the recall it triggers is already scoped.
- The channel surface is the **recall layer**: `core/recall/scope.py` (contextvars + the §5.3 agent_id→channel map), `core/archive.py get_archive_entry` (the predicate), `core/providers/lance.py _channel_filter`, and the `/recall` + `/search` endpoints.
- The ONE plumb the render path needs: the request's `agent_id` must reach the scope binding so recall can pick the persona's channel set (§5.3). If `agent_id` isn't already threaded to that point, it's a small passthrough — **not** filtering logic in render.

Both memory paths flow through this single recall layer: the auto-per-turn render (render → internal recall, scope bound from `agent_id`) and the explicit re-enabled recall tool (→ `/recall` with an explicit scope). `agent_private` is just `channel='agent_private' AND extracted_by='agent:<aid>'` applied at recall.

## 6. Write routing

### 6.1 Carrier `[H1]`
`channel` rides as a **top-level item field**, mirroring `extracted_by` (not `metadata.channel`, which contradicted the code: `ingest_turn` never reads `item.metadata`, and `ObserveRequest` has no metadata field). `observe_agent_output` sets `item['channel']`; the adapter reads `item.get('channel')`.

### 6.2 Stamp site `[H2]`
`add_new_block` ([adapters/veilguard_adapter.py:1442](../.gemini/antigravity/tcmm/TCMM/adapters/veilguard_adapter.py)) takes no `channel` and is not a `**extras` passthrough. Stamp `created.channel` + `_entry['channel']` **post-creation** at `veilguard_adapter.py:1466-1479`, the same pattern `extracted_by` uses. Do **not** change `add_new_block`'s signature. Add `channel: Optional[str]` to `ObserveRequest` for the HTTP path.

### 6.3 `observe_agent_output` contract
Add `channel: str = "agent_private"` to the signature; set `"channel": channel` in the ingest payload dict (currently `tcmm.py:~309-324`, beside `extracted_by` at `~316` — the hardcoded namespace at `:296` **stays**) `[G3]`. **Keep** that hardcoded `agent/<aid>/observations/<uid>` *namespace* (persistence-correct, cache-stable — do not reopen the [F4_AGENT_SCOPED_OBSERVE] non-persistence bug); preserve `extracted_by`; keep the `added>0` gate and the empty-`agent_id` drop. Persistence becomes byte-identical; only the new column differs.
**Return type changes `bool → Optional[list[int]]`** `[G4]`: the real inserted aid(s), or `None`/`[]` when nothing persisted. The `added>0` gate is preserved — an empty result is falsy, so existing callers that did `if not persisted:` behave exactly as before; §8 callers additionally store the echoed aid instead of the sha256.

### 6.4 Writer call-sites — complete list
| writer | site | channel |
|---|---|---|
| `record_episode`, `record_discussion_comment` | writers/phase_7 | `agent_private` |
| `promote_to_semantic` | `writers.py:237` | `team_knowledge` |
| `promote_lesson_to_team_knowledge` | `phase_7_writers.py:254`; wrapper `phase_7_writers.py:225` `[M2]` | `team_knowledge` |
| `record_outcome_with_narrative`, `record_proposal_with_content` | phase_7 | `team_events` |
| `enqueue_dream_input` | `writers.py:362` | `team_events` |
| `main.py` call sites `[M3]` | grep `~1496–1510`, **all** of them | per intent (a missing channel silently defaults `agent_private` — a recall-correctness bug, not a no-op) |
| `proposals/lessons.py:312` → wrapper | `[M2]` | `team_knowledge` |

`[M1]` Add a `channel: str` field to the `_WriterDest` dataclass ([writers.py:44](agent-runtime/app/memory/writers.py)) and populate every entry **in the same commit** `[G5]` — the dataclass is frozen, so the field add breaks all ~12 `WRITER_DESTINATIONS` constructors until each is updated. **Then** regenerate `MEMORY_WRITE_PATHS.md` via `generate_docs()` (AC-36/37 require channel in the regenerated doc — it is not auto-added by the existing dataclass).

## 7. Substrate-wall hardening `[H7]`

- `bulk_warm_user_archive` ([lance.py:2073](../.gemini/antigravity/tcmm/TCMM/core/providers/lance.py)): post-load assertion every row `user_id == _uid()`; one-user-per-instance invariant on `archive`/`dream_archive`.
- **In-loop `user_id` guards on EVERY cross-block dream loop.** P5 must begin with an exhaustive grep of `self.tcmm.archive` / `self.tcmm.dream_archive` `.values()` in `dream_engine.py` — the panel found 12+ beyond the originally-listed ~7 (1790, 1884, 1992, 2124, 2221, 2360, 2395, 2579, 3129, 3168, 3398, 3581, plus §3.4:524's list). Add `if block.user_id != self.tcmm._uid(): continue` + a loud tenant assertion at each. No-op under correct single-user operation; defense-in-depth otherwise.
- **CI lint** (sibling of `test_write_path_lint.py`): an AST check that fails CI if any `dream_archive`/`archive` `.values()` iteration in `dream_engine.py` lacks an in-loop `user_id` predicate. This is the durable guard — manual enumeration rots.
- **Mandatory §3.4 leak test** (already spec-required at line 526): two `user_id`s + two channels with overlapping topics → run a dream cycle → assert no bridge/concept-gravity/identity arc crosses the user boundary AND no private block surfaces in a shared-scope recall/digest.

## 8. Promotion = re-author; cross-ref fix

Promotion writes a *fresh* clean-lineage block tagged to the shared channel (D4). The `tcmm_obs_id` cross-ref is a `sha256(body)` content hash, not a real aid, at **two separate writers** `[G3]` — `record_outcome_with_narrative` ([phase_7_writers.py:113](agent-runtime/app/memory/phase_7_writers.py)) and `record_proposal_with_content` ([:203](agent-runtime/app/memory/phase_7_writers.py)); **both** must change. Fixed by making `/ingest_turn` **echo the real inserted aid(s)** (additive response field; old callers ignore it). `observe_agent_output` returns it; split-writers store the real aid; `lessons_reader`'s `lsn-tcmm-<aid>` ids become correlatable (fixes the documented mismatch at `phase_7_writers.py:269-279`). `lessons_reader.list_lessons` ([:150](agent-runtime/app/memory/lessons_reader.py)) replaces the full-archive scan with `WHERE user_id=<uid> AND channel='team_knowledge'` (keep `[lesson]` prefix as a secondary discriminator + last-write-wins/retirement), preferring the P3 `/recall` channel scope when HTTP is up.

## 9. Migration phases (reversibility annotated)

```
P0  Confirm recall path + columns from LOCAL code ... LOCAL / read-only (no prod)
P1  channel + _channels columns,
    promotion-registration, migrations, backfill .... ADDITIVE / reversible (nullable, unread)
P2  write-routing (top-level channel; ns stable) ... ADDITIVE / reversible
P3  server-side channel scope + filters + table .... ADDITIVE / reversible
P4  digest (cache-stable hash) + scoped recall ..... ADDITIVE → CUTOVER (render-default flip one-way)
P5  substrate-wall guards + CI lint + leak test .... DEFENSIVE / reversible
P6  reader cutover + real-aid cross-ref ............ CUTOVER (needs P1 backfill first)
P7  spec amendment (§3.4 / §3.4.1 / §3.4.2 / §3.11) . DOC (incl. enum/tier-group reconciliation)
```
Everything before P4 is dark-launchable. One-way gates: the P4 render-default flip and P6 workaround removal.

## 10. Touch-point checklist (panel-verified, 26 areas)

All 26 areas from coverage run `wkx24xzsx` are owned by a phase above. The 2 prior correctness defects (dream-node backfill, digest hash) are resolved in §4.1 / §5.1; the contradictions (carrier H1, stamp-site H2, tier-group H4, render-surface C3) are resolved in §6 / §3.2 / §5.4; the under-specifications (columns H3, loop enumeration H7, scope table M4, writer doc M1, wrapper M2, main.py M3, dream-source-ids M5) in §3.3 / §7 / §5.3 / §6.4 / §4.1.

## 11. Risks & test strategy

| Risk | Severity | Mitigation / test |
|---|---|---|
| Cross-user leak via dream loops | Bet-the-product | §7 in-loop guards + CI lint + §3.4 two-user leak test |
| Private-derived text leak into shared recall | High | D4 re-author + §4.2 conservative visibility (any uncertainty ⇒ most-private) |
| Backfill under-stamps truncated/unresolvable lineage | High `[C2/H6/M5]` | §4.1 cap/resolve/dream-id rules; test node with absent private source |
| Digest cache-key instability / cost regression | High `[C1]` | §5.1 content-fingerprint hash; canary asserts stable version + cache hit |
| Render surface dangling on absent endpoint | High `[C3]` | §5.4 binding P0 decision before P3/P4 code |
| Lance schema evolution on large table | Medium | P1 idempotent `add_columns` tested on snapshot; nullable default; never tar 21GB model dirs |
| Reader cutover drops historical lessons | Medium | P6 after P1 backfill; transitional `[lesson]` OR-clause; count-parity test |
| Enum/tier-group/submission-target drift | Medium `[H5]` | single enum in §3.1; P7 reconciles §3.4.2 ↔ §3.11 |

**Verification order:** (1) unit — tier map totality, backfill conservatism (capped/unresolvable/dream-id), scope→channel mapping; (2) TCMM integration — §3.4 two-user leak test, channel-visibility test, digest determinism + stable `digest_version` under reinforcement; (3) agent-runtime↔TCMM — each writer stamps the right channel, re-enabled `recall` returns only in-scope channels, Director `auto` never returns another agent's `agent_private`; (4) cross-ref — promote a lesson → `tcmm_obs_id` resolves to a real aid; (5) cache/cost canary — one persona on the new render path 24h before global flip; (6) lint/doc — `test_write_path_lint.py` + the new dream-loop lint pass; regenerated `MEMORY_WRITE_PATHS.md` includes `channel`.

## 12. Open items

- **Channel-stratified synthesis (still open, your call).** This spec is read-time-filter-only: the dream synthesizes unchanged and the shared *compiled* layer is limited to nodes whose entire lineage is shareable (can be sparse — richness comes from D4 re-authoring + re-dreaming clean facts). The alternative — a synthesis pass restricted to shareable blocks so the team gets a rich compiled graph by construction — is a larger dream-module change, not included here. The recall mechanism is identical either way; stratified synthesis only changes *how many* nodes come back shared-visible.
- **No production access in this work.** Design and implementation are local-only; deploying to the VM is user-controlled and explicit. (Earlier drafts framed a "VM parity / SSH the VM" step — removed; C3 is resolved from local code per §5.4.)
- **Multi-user teams** (per-`team_id` substrate) deferred to a later phase; v1 locks `team_id = user_id` but the channel column + filter make the generalization a filter swap, not a re-architecture.

## Deploy discipline
All edits are **local-first** in `Documents/veilguard/` (agent-runtime) and `.gemini/antigravity/tcmm/TCMM/` (TCMM), tested locally. **I do not touch the prod VM** — deployment is user-controlled and explicit, done by you when you decide. No git worktrees. The `tcmm-service` snapshot is known-stale vs whatever is running; whoever deploys diffs first. Deploy tarballs = core/adapters/api only (never the 21GB model dirs).
