# Blazingly-Fast Redaction — Spec

**Status:** §4 unification + FROZEN block cache + fail-closed — **IMPLEMENTED 2026-05-29** (local, untested in prod). Tier-2 TCMM `_vg_*` metadata (§3.2) still pending on the TCMM side.
**Date:** 2026-05-29
**Author:** Rudolph + Claude
**Owner module:** `veilguard.pii` (`pii/redactor.py`, `pii/session_store.py`)
**Consumers:** `agent-proxy/app/main.py` (LibreChat path), `agent/base.py` (multi-agent `run_turn`)

---

## IMPLEMENTATION STATUS (2026-05-29, local-first — not yet deployed)

**ONE live redactor.** Both paths now import `veilguard.pii`. The proxy's
old uncached `agent-proxy/app/redactor.py` + in-memory `app/session.py` are
gone — `app/redactor.py` is now a thin re-export shim of `veilguard.pii`.

Landed:
- **Unification** — `main.py` imports `from pii import get_redactor,
  RedactionUnavailable, get_store`; `redact_json` lives on the shared module
  and routes the top-level Anthropic `system` block list through
  `redact_render_blocks`. `base.py` `redact_blocks`/`redact_messages` already
  resolve to the same engine.
- **FROZEN block-output cache** — `_block_cache` keyed by `(tenant, conv,
  blake2b(text))` → redacted bytes; warm byte-stable blocks replay with zero
  Presidio/Lance. `redact_blocks` delegates to `redact_render_blocks` so BOTH
  paths get it.
- **Line cache REMOVED (2026-05-29)** — the per-line delta analyzer
  (`_analyze_cached`/`_line_spans`) was deleted. Rationale: TCMM memory blocks
  are immutable, so a block is either a block-cache HIT (never re-analyzed) or
  BRAND NEW (every line is new anyway → line caching bought nothing). The model
  is now dead simple: **new block → redact in full → store; seen block →
  replay.** Verified equivalent: warm-turn latency unchanged (~86ms) under the
  per-fragment model. Tradeoff consciously accepted: a block that grows *in
  place* now re-scans in full — a case TCMM's per-fragment working tier never
  produces.
- **CLEAN-skip** — provenance-anchored exact-prefix fingerprints (magic
  prefix, Veilguard preamble) + the `_skip_pii` sentinel + future `_vg_pii`.
- **Metadata strip** — `_emit_block` removes `_skip_pii`/`_vg_*` before the
  wire (Anthropic rejects unknown fields).
- **Fail-closed RESTORED** — the 2026-05-23 proxy redactor had silently
  regressed to fail-*open* (`return text` on Presidio error). The shared
  module now raises `RedactionUnavailable`; proxy → 503, runtime → `error`
  event (`base.py` guards the `_do_redact` call).
- **String-sid compat** — `_coerce_sid` wraps the proxy's per-user
  `pii-{user}` string into `SessionId("_proxy", …)`, preserving per-user token
  scoping with no product-behavior change.
- **Hydration fast-paths** — `if "REF_" not in text: return text` short-circuit;
  `/rehydrate` endpoint now uses the Lance store (`rehydrate_any`, token-scoped)
  instead of the removed `pii_store._store` scan.

Verified locally: 14/14 in `_smoke_unified_redactor.py`, 13/13 in
`pii/tests/test_byte_stability.py`, shim import resolves to `pii.redactor`.

**Still open:**
- **Keyspace harmonization** — proxy keys per-USER (`pii-{user}` under tenant
  `_proxy`); runtime keys per-CONVERSATION (`SessionId(user_tenant, conv)`).
  They share the module + store but live in different partitions, so the same
  PII can get different tokens across the two paths. Pre-existing behavior;
  decide later whether to converge on per-conv everywhere.
- **Tier-2 `_vg_*`** from TCMM's renderer (§3.2 / §11) — the classifier
  already honors it when present; TCMM doesn't emit it yet.
- **Deploy** — local-first per workflow; push + restart `pii-proxy` and the
  agent-runtime, then validate live under the real tenant.

---

## 0. TL;DR

Redaction is slow because **we re-run spaCy NER over the entire ~20–70 KB
assembled prompt on every single turn**, even though 90%+ of those bytes are
either (a) static and provably PII-free (preamble, tool schemas, persona) or
(b) byte-stable memory fragments we already redacted on a previous turn.

The fix is to **pay redaction cost once per unique (content, session) and
never again**, by classifying every block before we touch Presidio:

| Block class | What it is | Redaction policy |
|---|---|---|
| **CLEAN** | magic prefix, Veilguard preamble, tool schemas, persona, immutable tier | **Skip** — never scanned |
| **FROZEN** | byte-stable memory fragments (frozen/stable tiers, past `[Memory index=N]` working fragments) | **Redact once, cache the redacted bytes by `(sid, content-hash)`** |
| **VOLATILE** | growing live tier, conversation tail | **Delta-redact** — analyze only new lines (already implemented) |
| **PROMPT** | the current user message | **Redact every turn** (it's new, and small) |

Hydration (REF_* → original) is already cheap (one regex pass keyed by the
session map); we only make it cheaper by caching the session map per request.

**Target:** warm-turn redaction goes from *seconds* (full NER) to **<50 ms**
(hash lookups + delta-NER of the one new fragment + the user prompt).

---

## 1. Current state (what actually happens today)

There are **two live redaction code paths**, and they do not share an
implementation. This is the single most important thing to understand before
optimizing.

### 1.1 Path A — LibreChat → pii-proxy (`agent-proxy/app/main.py`)

- Imports the **old** redactor: `from .redactor import get_redactor` →
  `agent-proxy/app/redactor.py` (no caching) + the **in-memory**
  `from .session import pii_store`.
- On every request it calls `redactor.redact_json(data, pii_session_id)`
  (`main.py:4565`) over the **entire request body** — `data["system"]` (the
  full TCMM-rendered block list) *and* `data["messages"]`.
- `redact_json` recurses into every dict; `USER_CONTENT_KEYS` includes
  `"text"`, so **every system block's `text` is scanned**, including the
  immutable preamble + tool schemas.
- The `_skip_pii` sentinel exists in `redactor.py:194` **but `main.py` never
  sets it** — so the "skip the static preamble" optimization is dead code on
  this path. → full Presidio NER over the whole prefix, every turn.

> **Headline bug:** the proxy path has *no* analyzer cache at all and scans
> ~20–70 KB of mostly-static content on every turn.

### 1.2 Path B — multi-agent runtime (`agent/base.py::run_turn`)

- Imports the **new** module: `from pii import SessionId, get_redactor`
  (`base.py:53`) → `pii/redactor.py` + Lance-backed `pii/session_store.py`.
- STEP 2 (`base.py:541-559`) calls `redactor.redact_blocks(raw_blocks, sid)`
  and `redactor.redact_messages(messages, sid)`, offloaded to a worker thread.
- This module already has the speed fixes landed **today**:
  - `[PII_DELTA_LINE_CACHE_2026_05_29]` — per-line analyzer cache
    (`redactor.py:178-241`): only NEW lines hit spaCy; cached lines are free.
  - `[PII_MEMO_2026_05_29]` — `add_mapping` memoizes `(sid, entity, value) →
    token` in a process dict (`session_store.py:151-160`), killing the
    per-span Lance `.search()` (~33 ms each → 3 s+ on a repetitive prefix).
  - `[PII_BATCH_WRITE_2026_05_29]` — buffer new tokens, one Lance `add()` per
    redaction instead of one table-version per span (`session_store.py:379`).
  - `[PII_PIPELINE_TRIM_2026_05_29]` — drop the spaCy **parser** always
    (~34% of pipeline time, unused by Presidio) and optionally the whole NER
    stack via `PII_REGEX_ONLY` (`redactor.py:112-149`).
  - `[RENDER_DISPATCH_CACHE_2026_05_29]` — cache the *raw* rendered blocks for
    a short TTL so the inner turns of one dispatch render+redact once
    (`base.py:388-408`).

### 1.3 Where the time goes (measured, from code comments)

On a realistic ~30 k-token prompt (`redactor.py:118-128`):

```
full spaCy pipeline ........ 7570 ms
  - parser (unused) ........ 4960 ms   ← dropped already
  - NER .................... 1497 ms
regex-only (spaCy off) ......  463 ms   ← ~15 ms / 1000 tok floor
```

Plus, pre-fixes:
- per-span Lance lookup: ~33 ms × N spans (memo'd away)
- one-version-per-span writes: 1.5 s manifest walk on the bind mount (batched away)
- a single IC dispatch = 6+ inner turns → **minutes** of pure redaction if each
  turn re-renders + re-redacts (render-dispatch cache addresses this *within* a
  dispatch only).

### 1.4 Gaps the current fixes do NOT close

1. **Proxy path (A) gets none of it.** It still uses the uncached old redactor.
2. **The line cache still does per-turn work on stable blocks**: it splits
   every block into lines, looks each up, re-mints tokens (memo'd), and
   re-substitutes. For a block we already fully redacted last turn, *all* of
   that is wasted — the output is byte-identical to last turn.
3. **No "clean" skip list.** Tool schemas + preamble + persona are PII-free by
   construction but still walked line-by-line.
4. **Render-dispatch cache is TTL-bound and per-dispatch.** Across dispatches /
   across normal chat turns the stable memory is redacted again.

---

## 2. Core insight & taxonomy

Redaction output for a block is a **pure function of `(block_text, sid)`**:
- Token assignment is **append-only and deterministic** within a session
  (`session_store.py` invariants A/B: same input + same SessionId → same token,
  existing tokens never change).
- Therefore: *if we have ever redacted this exact `block_text` under this exact
  `sid`, the redacted bytes are identical and can be replayed from cache with
  zero Presidio and zero token minting.*

This is stronger than the current line cache (which caches *analysis*, not
*output*, and still re-substitutes). We cache the **finished redacted string**.

The only question per block is: **"have I seen these exact bytes for this
session before?"** — a `dict[(sid, sha256(text))] → redacted_text` lookup.

Blocks fall into four classes (see table in §0). The render is structured
exactly to make this classification cheap:

- TCMM emits the **immutable tier** (pinned system prompt + tool schemas) and
  the **frozen/stable** memory tiers as byte-stable blocks
  (`anthropic_renderer.py:44-66`).
- The **working/live tier** is split **one block per fragment**, each headed
  `[Memory index=N | role=X | src=live]` (`anthropic_renderer.py:200-228`).
  **`index=N` is the stable per-fragment id** ("AID") — fragment N is immutable
  once created; each turn only appends a higher N. So fragments `0..k-1` are
  FROZEN (cache hits) and only fragment `k` is new.
- The magic prefix, inlined preamble, and persona are appended by the runtime
  (`base.py:442,477,528`) and are CLEAN by construction.

---

## 3. Design

### 3.1 The redaction decision, per block

```
for blk in assembled_blocks:
    cls = classify(blk)                      # CLEAN | FROZEN | VOLATILE
    if cls == CLEAN:
        emit(blk)                            # zero work
    elif cls == FROZEN:
        key = (sid, blk_hash)                # blk_hash = sha256(text)
        out = FROZEN_CACHE.get(key)
        if out is None:
            out = redact_text(text, sid)     # full path, once
            FROZEN_CACHE[key] = out
        emit(blk_with_text(out))             # warm: O(1)
    else:  # VOLATILE
        emit(blk_with_text(redact_text_delta(text, sid)))   # line-delta cache
```

`redact_text` and the line-delta path **already exist** in `pii/redactor.py`.
The new pieces are: **`classify()`**, the **`FROZEN_CACHE`** (block-output
cache), and **wiring the proxy onto this module**.

### 3.2 How `classify()` decides — "take the lead from TCMM"

Per the user's framing, **TCMM is the authority**: it builds the blocks, it is
inside the trust zone, and it knows which are immutable and which already
passed redaction. So TCMM should *tag* each block and the redactor should obey
the tag. Two-tier approach (ship tier 1 now, tier 2 when TCMM is updated):

**Tier 1 — content-hash classification (no TCMM change, ship immediately):**
- CLEAN: block text matches a known-static fingerprint — magic prefix,
  `_VEILGUARD_PREAMBLE_TEMPLATE[:80]` (`base.py:486`), persona system prompt,
  or any block whose text starts with a tool-schema/immutable-tier marker.
  These are matched by prefix/fingerprint, exactly like the existing
  preamble-idempotency check.
- FROZEN vs VOLATILE: a block is FROZEN if its **content hash was present in
  the previous turn's block set** for this sid. Concretely: keep a per-sid set
  of "hashes seen last turn"; a hash in that set ⇒ FROZEN (cache hit expected);
  a new hash ⇒ VOLATILE (delta-redact, then it becomes FROZEN next turn).
  This is self-validating: if a block's bytes change, its hash changes, it's
  treated as VOLATILE and re-redacted — **no stale redaction is ever emitted.**

**Tier 2 — TCMM-emitted metadata (the clean version):**
TCMM's renderer adds three optional keys to each block dict it emits:

```jsonc
{
  "type": "text",
  "text": "...",
  "cache_control": {...},          // unchanged
  "_vg_id": "frag:42",             // stable block id (Memory index, tier id, or content hash)
  "_vg_immutable": true,           // bytes will not change across turns
  "_vg_pii": "clean"               // "clean" = system/tool/preamble (never scan)
                                   // "scan"  = user-derived (must redact)
}
```

Then `classify()` is exact and free:
- `_vg_pii == "clean"` → CLEAN
- `_vg_pii == "scan" && _vg_immutable` → FROZEN, keyed by `(sid, _vg_id)`
- else → VOLATILE

> **Strip rule (mandatory):** these `_vg_*` keys MUST be removed before the
> body is sent upstream — Anthropic 400s on unknown fields. Reuse the exact
> pattern already used for `_skip_pii` (`redactor.py:194`) and
> `_veilguard` envelope stripping (`main.py:4547`). The block-emit step copies
> all keys *except* `_vg_*` / `_skip_pii`.

**Recommendation:** implement Tier 1 first (works against today's renderer,
zero coordination), keep the `_vg_*` contract as the fast path that activates
automatically when TCMM starts emitting it. The classifier checks for `_vg_*`
and falls back to content-hash heuristics when absent.

### 3.3 The FROZEN block-output cache

```python
# pii/redactor.py — new
self._block_cache: dict[tuple[SessionId, str], str] = {}   # (sid, sha256(text)) -> redacted
self._block_cache_lru: collections.OrderedDict           # bound size, LRU evict
```

- Key: `(sid.root(), blake2b(text, digest_size=16).hexdigest())`.
  Use BLAKE2b (faster than sha256, in stdlib); 128-bit is collision-safe at
  conversation scale.
- Value: the fully-redacted string.
- Hit ⇒ return immediately. **No spaCy, no line split, no `add_mapping`, no
  Lance read/write.**
- Miss ⇒ run the existing `redact_text(text, sid)` (which itself uses the line
  cache + memo + batch write), store result, return.
- **Memory bound:** LRU cap (`PII_BLOCK_CACHE_MAX`, default 2048 entries).
  Entries are whole redacted blocks (≤ tens of KB) — a few MB at the cap.
- **Eviction does not affect correctness** — a miss just recomputes.
- **Cross-process:** this cache is per-process and in-memory; that's fine
  because it's a pure accelerator over the Lance store, which is the durable,
  shared source of truth for token mappings. Two processes may each compute the
  same redacted bytes once; both get identical output (deterministic tokens).

### 3.4 VOLATILE delta path (already built — keep)

For the growing live tier and the conversation tail, keep
`_analyze_cached` (`redactor.py:178`): split on `\n`, analyze only
not-yet-seen lines in one batched `analyze()` call, reassemble full-text spans.
Warm cost ≈ NER of just the new lines (<100 ms). This is the right tool for
"block that is *mostly* the same but grew by one line".

> **Why both a block cache *and* a line cache?** The block cache wins big when a
> whole block is byte-identical (frozen tiers, past fragments). The line cache
> wins when a block changed slightly (live tier appended a line). They compose:
> block-cache miss → line-cache mostly-hits.

### 3.5 The user prompt

Always redact it (`redact_text`, full path). It is new every turn and small
(typically < 2 KB) — NER on it is single-digit ms. No caching needed; caching a
per-turn-unique string would just bloat memory.

### 3.6 Hydration (REF_* → original)

Already correct and cheap (`session_store.py:403-417`,
`_REF_TOKEN_RE.sub`): one regex pass, `\b`-bounded so `REF_PERSON_1` can't
match inside `REF_PERSON_15`. Two refinements:

1. **Cache the session rehydrate-map per response** instead of re-querying
   Lance per chunk. The streaming path calls `rehydrate_text` many times per
   response (`main.py:4823,4855,5044,5308,5383,5566`); each currently calls
   `_rehydrate_map` → a Lance scan. Build the map once at response start, reuse
   for every chunk. (Map is small — one row per distinct PII in the conv.)
2. **Reference-driven hydration** (the user's "look for references"): only
   bother building/scanning if the text contains the literal substring `REF_`.
   `if "REF_" not in text: return text` before the regex. Skips the work
   entirely on responses with no tokens (the common case for non-PII chats).

---

## 4. Unify the two paths

The duplication in §1.1/§1.2 is itself a correctness risk (the proxy and the
runtime can mint *different* tokens for the same conv, breaking cache
stability across the LibreChat-vs-agent boundary the `__init__.py` docstring
promises). **Both paths must use `veilguard.pii`.**

### 4.1 Migrate the proxy (Path A) onto `veilguard.pii`

- Replace `from .redactor import get_redactor` / `from .session import
  pii_store` in `main.py` with `from pii import get_redactor, SessionId`.
- Construct `sid = SessionId(tenant_id, conv_id)` from the request
  (`extract_pii_session_id` currently returns a string at `main.py:3916`;
  replace with a `SessionId`).
- Replace the monolithic `redactor.redact_json(data, pii_session_id)` call
  (`main.py:4565`) with a **structure-aware redaction** that knows what each
  part is, instead of blindly walking JSON:
  - `data["system"]` (list of blocks) → `redact_render_blocks(blocks, sid)`
    (the new classify+cache path, §3.1).
  - `data["messages"]` → `redact_messages(messages, sid)` (existing), which
    already skips `assistant`/`model`-authored content.
- Delete `agent-proxy/app/redactor.py` and `agent-proxy/app/session.py` once
  the proxy imports the shared module (keep a thin shim re-exporting from
  `veilguard.pii` for one release to avoid breaking any stragglers).

### 4.2 New top-level method on `PIIRedactor`

```python
def redact_render_blocks(self, blocks: list[dict], sid: SessionId) -> list[dict]:
    """Classify each block, then CLEAN→skip / FROZEN→block-cache /
    VOLATILE→line-delta.  Strips _vg_* metadata.  Preserves cache_control
    and every other structural key (byte-stability contract C)."""
```

`redact_blocks` (the current method, `redactor.py:294`) becomes the VOLATILE
fallback used internally; `redact_render_blocks` is the new entry point both
consumers call.

### 4.3 Make the runtime use it too

In `base.py:557`, swap `redactor.redact_blocks(raw_blocks, sid)` →
`redactor.redact_render_blocks(raw_blocks, sid)`. The render-dispatch cache
(`base.py:388`) stays as a coarse outer layer; the block cache makes it
unnecessary across dispatches but they compose harmlessly.

---

## 5. Correctness invariants (must hold; add to `pii/tests/`)

These extend the existing `tests/test_byte_stability.py` invariants A–D.

- **A. Determinism.** `redact_render_blocks(blocks, sid)` is byte-identical
  whether served from cache or computed cold. (Test: redact twice, compare.)
- **B. No stale redaction.** If a block's text changes by one byte, its output
  reflects the new content (hash miss ⇒ recompute). (Test: mutate a frozen
  block, assert re-redacted.)
- **C. Structural preservation.** Only `.text` is mutated; `cache_control`,
  `type`, and all non-`_vg_*` keys pass through unchanged; `_vg_*`/`_skip_pii`
  are stripped. (Test: assert key-set equality minus the meta keys.)
- **D. Token stability.** A PII value that appeared in a FROZEN block in turn 1
  keeps its REF token when it reappears in turn 5. (Guaranteed by the
  append-only store; test the cache doesn't shadow it.)
- **E. CLEAN never leaks scan cost AND never hides PII.** A block classified
  CLEAN must be provably PII-free (system/tool/preamble fingerprint or
  `_vg_pii=="clean"`). **Never** classify a user-derived block CLEAN. (Test:
  inject PII into a "clean"-fingerprinted block in a fuzz test and assert the
  fingerprint can't match user content — i.e. CLEAN matching is prefix-anchored
  to known static templates, not heuristic.)
- **F. Fail-closed.** Presidio failure on a VOLATILE/FROZEN-miss block still
  raises `RedactionUnavailable` → proxy returns 503 (`main.py:4566`).
  Cache hits never fail. Don't let the cache mask a degraded analyzer for *new*
  content.

> **Security note on the CLEAN class (E is the dangerous one).** The whole
> speed win rests on never scanning CLEAN blocks. That is only safe if CLEAN is
> determined by *provenance*, not by *content heuristics*. The clean set is:
> blocks the **system** authored (magic prefix, preamble template, persona,
> pinned tool schemas) — never anything derived from a user message or tool
> output. Tier-1 fingerprinting must anchor on the exact static templates
> (startswith on the known constant), and Tier-2 `_vg_pii=="clean"` must only
> be emitted by TCMM for system/tool/immutable blocks. When in doubt, classify
> VOLATILE (scan it) — correctness over speed.

---

## 6. Performance budget (expected)

For a warm Director turn, ~30 k-token prefix, ~5 frozen fragments + preamble +
tools + 1 new fragment + a 1 KB user prompt:

| Step | Before | After |
|---|---|---|
| CLEAN blocks (preamble+tools+persona) | full NER (~hundreds ms) | **0 ms** (skip) |
| FROZEN fragments (×5) | NER each turn | **~0 ms** (hash hits) |
| 1 new VOLATILE fragment | — | line-delta NER (~tens ms) |
| user prompt (1 KB) | included in blob | ~5 ms |
| token minting | per-span Lance | memo O(1) + 1 batched write |
| **total warm redact** | **seconds (full pipeline)** | **< 50 ms** |
| cold turn (first sight) | seconds | one full pass (parser dropped: ~2.6 s; `PII_REGEX_ONLY`: ~0.5 s) |

Hydration: unchanged hot path, minus a Lance scan per chunk (map cached) and
minus the whole pass when `REF_` absent.

---

## 7. Implementation plan (ordered, each independently shippable)

1. **Wire `_skip_pii` / unify the proxy (biggest, cheapest win).** Point
   `main.py` at `veilguard.pii`; replace `redact_json(data,...)` with
   `redact_render_blocks(data["system"])` + `redact_messages(data["messages"])`.
   *This alone gives the proxy the line cache + memo + batch write it currently
   lacks.*
2. **Add `classify()` Tier-1** (content-hash + static fingerprints) and the
   **FROZEN block-output cache** in `pii/redactor.py`. Add
   `redact_render_blocks`.
3. **Switch `base.py:557`** to `redact_render_blocks`.
4. **Hydration refinements** (`REF_` short-circuit + per-response map cache).
5. **Tier-2 TCMM metadata** (`_vg_id`/`_vg_immutable`/`_vg_pii`) in
   `anthropic_renderer.py` + the strip rule; classifier auto-upgrades.
6. **Delete** old `agent-proxy/app/redactor.py` + `session.py` (after a release
   with a re-export shim).

Steps 1–4 need **no TCMM change** and capture the bulk of the win.

---

## 8. Observability

- Reuse the redact-cache hit-rate logging pattern from the stale
  `.veilguard/redactor.py:219-225`. Emit every ~50 calls:
  `[redact] block_cache hits=… misses=… (NN%)  clean_skipped=…  volatile=…`.
- Add `redact_ms` / `block_cache_hit_pct` to the existing per-phase log
  (`main.py:5575` `[PHASE-PROXY]`) and to `pii_audit` so the admin dashboard
  can chart it next to `tcmm_pre_http` / `anthropic`.
- Alert if `block_cache_hit_pct` on warm turns drops below ~70% — that means
  blocks aren't byte-stable (a renderer regression shifting bytes, which also
  destroys Anthropic's prompt cache → the two problems share a root cause).

---

## 9. Config flags

| Env | Default | Effect |
|---|---|---|
| `PII_BLOCK_CACHE_MAX` | `2048` | LRU cap on the FROZEN block-output cache |
| `PII_REGEX_ONLY` | off | drop spaCy NER entirely (regex/checksum entities only; no PERSON) — emergency speed valve, already implemented |
| `PII_CLEAN_SKIP` | on | allow CLEAN classification (set off to force-scan everything for audits / paranoia) |
| `PII_ALLOW_LIST_PATH` | — | existing brand/tech allow-list override |

---

## 10. Risks & mitigations

- **R1 — CLEAN misclassification leaks PII.** Mitigation: provenance-anchored
  CLEAN only (§5-E); default-VOLATILE on doubt; `PII_CLEAN_SKIP=off` kill
  switch; fuzz test E.
- **R2 — Cache key collision** emits one conv's redaction into another.
  Mitigation: key includes `sid.root()` (tenant + root conv); 128-bit hash;
  collisions recompute, never cross sessions because sid is in the key.
- **R3 — Memory growth** from the block cache on a server with thousands of
  live convs. Mitigation: LRU cap (whole redacted blocks, not per-conv
  unbounded); evicting is free (recompute on miss).
- **R4 — Renderer byte-instability** silently tanks the hit rate. Mitigation:
  the §8 alert; this is the *same* invariant Anthropic prompt caching needs, so
  it's already load-bearing.
- **R5 — Two processes mint a token concurrently** for the same new PII.
  Already handled by `session_store.py` (re-query under lock, append-only). The
  block cache only ever *replays* committed mappings, so it can't introduce a
  new race.

---

## 11. Open questions for TCMM side

1. Can the renderer cheaply emit a **stable per-fragment id**? `[Memory
   index=N]` already exists in the working tier — promote it to `_vg_id` and
   add `_vg_id` to the frozen/stable/immutable tier blocks too.
2. Does TCMM already know a block is **immutable** at render time (frozen tier =
   yes; can it assert it for stable tier?) so `_vg_immutable` is trustworthy?
3. Should TCMM **pre-redact** memory at ingest so the trust-zone/LLM-boundary
   split (per `architecture_pii_boundary`) is preserved while letting render
   emit already-tokenized bytes? (Probably **no** — TCMM stores raw by design;
   redaction stays at the LLM boundary. Keeping it here means the block cache is
   the right place for the "redact once" property. Confirm this stays true.)
