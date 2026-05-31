# Redaction — handoff note (2026-05-29)

Another agent is taking redaction. Here's the full state so you don't redo the work.

## Current deployed state (WORKING, PERSON detection ON)
In-process Presidio (`pii/redactor.py`) in agent-runtime + pii-proxy, with these fixes live:
- **Dropped the spaCy dependency `parser`** unconditionally — Presidio never uses it; pure ~34% speedup, zero detection change.
- **`PII_REGEX_ONLY=0`** — spaCy NER (PERSON) is ON. (Code supports regex-only via `PII_REGEX_ONLY=1` but it is NOT enabled — the user explicitly did NOT authorize dropping PERSON.)
- **BLAS threads pinned** `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1` (compose env) — fixes catastrophic OpenBLAS oversubscription under concurrent redaction.
- **Delta line-cache** in `_analyze_cached` — caches PII analysis per line (session-independent), batches unseen lines into ONE `analyze()` call. Process-global. Byte-exact split/join on `\n`.
- **Per-dispatch render cache** (`agent/base.py _RENDER_CACHE`, 90s TTL) — memory blob rendered+redacted once per dispatch.

## Measured perf (this WSL2 Docker box, 16-core)
- Realistic 30k-token redaction: full pipeline 7570ms → parser-off 2532ms.
- Delta cache: cold (all new) 7147ms, warm (prefix cached + 2 new lines) 750ms, identical 0ms.
- Concurrency: default 16-thread BLAS = 33s/call at 4-concurrent; pinned = 15s. (Per-op single ~5-8s.)
- Per-component (30k tok): parser ~2.6s, NER ~2.3s, lemmatizer ~1.4s, regex-only floor ~463ms.

## KNOWN ISSUES / the real fix
1. **Cold pass = 7s** for a never-seen 30k-token blob. No prompt-path cache removes this.
2. **Delta cache eviction is a CLIFF**: `if len(cache) > 200_000: cache.clear()` dumps everything → periodic cold-7s spikes. Should be LRU. (Real bug, not yet fixed.)
3. **Architectural root cause:** redaction sits on the *prompt* path (re-redacts the whole assembled prompt each turn). The correct fix is **redact per memory-item at ingest/observe time, once**, and cache the redacted render by item-id — then recall returns already-redacted content and per-turn redaction = only the new user message (<100ms), regardless of eviction/re-ranking. TCMM stores raw (embeddings need raw), so keep raw + a redacted-render cache.

## redact-core (censgate/redact) spike — BLOCKED locally
Rust Presidio-equivalent + ONNX BERT NER (`dslim/bert-base-NER`, PERSON/ORG/LOC). REST: `POST /api/v1/analyze` `{text,language}` → `{results:[{entity_type,start,end,score,text}],metadata:{processing_time_ms}}`. Docker `ghcr.io/censgate/redact:full` (model baked). v0.8.3, Apache-2.0. Claims 2-10ms/text, 10-100× faster.
- **Both `:latest` and `:full` crash SIGILL (exit 132) on this WSL2/Hyper-V box** — quantized ONNX uses AVX-512-VNNI that Hyper-V advertises but doesn't faithfully execute. amd64 confirmed, `--platform linux/amd64` forced, still SIGILL. **CANNOT benchmark locally.**
- **It would run on the prod VM** (GCP n2 = Cascade/Ice Lake, has real AVX-512-VNNI). Benchmark there.
- Integration plan if adopted: use it as a **detection sidecar only** (`/api/v1/analyze`), keep our reversible REF-token + Lance rehydration layer + the 3 SA recognizers (SA ID/phone/bank) in Python regex. Reversibility: redact's anonymize is Hash(irreversible)/Encrypt(opaque) — don't use it; mint our own tokens from its spans.

## Recommended next step for redaction
Benchmark redact on the **prod VM** (speed on 30k-tok + concurrent, PERSON recall vs spaCy). If it holds, wire it as a detection sidecar. Otherwise fix the delta-cache eviction to LRU and pursue redact-at-ingest.

## FOLLOW-UP (found 2026-05-29): immutable blocks are re-NER'd per dispatch
The clean-skip fingerprints (`_clean_fingerprints`) only cover the Anthropic magic
prefix + the Veilguard preamble. The **persona system_prompt (~1880 tok)**, tool
defs, and constitution are NOT fingerprinted → they go through the block cache,
which is keyed on `(tenant_id, conv_id, hash(text))` (session-scoped). So every NEW
dispatch (new conv_id) is a cache MISS on those immutable blocks → full spaCy NER on
~2k+ tokens of our own PII-free config, every dispatch's first turn.
FIX: mark the persona/tools/constitution blocks `_skip_pii=True` (or `_vg_pii="clean"`)
in agent/base.py where they're appended — they're system-authored, provably PII-free.

## RESOLVED (2026-05-31): cross-process PII token persistence — REAL root cause
The redact-in-render fast path was blocked because the host TCMM render process
mints `REF_*` tokens but the agent-runtime container couldn't rehydrate them — the
`pii_session_mapping` table looked empty/missing cross-process. The earlier theory
("empty `create_table` never persists") was WRONG.

**Real root cause:** `pii/session_store.py` decided table existence with
`_TABLE_NAME in self._db.table_names()`. In **lancedb 0.30.2 `table_names()`
defaults to `limit=10`** and returns names alphabetically. The agent-runtime
`tcmm.db` has 15+ tables; `pii_session_mapping` sorts PAST the first 10
(`open_table` proves it's on disk with a valid 294-manifest chain), so the
membership check returned **False every time**. Consequence: every reader saw the
table as "missing" → rehydrate returned text unchanged (REF tokens leaked); and the
write path's create-guard fired and `create_table(mode="overwrite")` **wiped real
mappings**. (A temp-db round-trip "passed" only because that DB had <10 tables.)

**Fix (landed, live):** existence is now decided by trying `open_table` via a new
`_open_existing()` helper — never via `table_names()`. `_get_tbl()` uses it +
`checkout_latest()` for cross-process freshness; reads tolerate a missing table;
`flush()` uses `mode="create"` (NOT overwrite) with an append fallback so a
mis-detection can never destroy rows again. All `table_names()` membership checks
removed from the hot path.

**Verified end-to-end:**
- 2-/3-process round-trips on a fresh DB and on the real 15-table DB: mint in one
  process/container → rehydrate in another = PASS; append (not overwrite) confirmed.
- Container (agent-runtime) rehydrates tokens minted by a host process: PASS.
- After restarting agent-runtime + pii-proxy (bind-mounted code), **live traffic**
  minted 5 real tokens that persisted and read back: `PERSON×2, SA_ID_NUMBER,
  SA_BANK_ACCOUNT, SA_PHONE_NUMBER` (PERSON-NER intact, per the hard constraint).

**Same bug elsewhere (flagged, not yet fixed):** any `X in db.table_names()` check
on a >10-table DB is broken. Highest risk: `services/tcmm-service/server.py:979`
skips index creation for tables past page 1 — if `sparse_archive` sorts past 10, its
FTS index is silently never built → matches the recurring "FTS INDEX: pending" /
slow-recall symptom. Also in `setup_lance_indices.py`, `scripts/*`, sub-agents tasks
store, admin-dashboard lance_stats. Fix: pass `limit=10_000` or use try/open_table.

## NEXT (now UNBLOCKED, but a PII-sensitive decision — not yet enabled)
Enabling redact-in-render needs BOTH flags flipped **together**, or memory passes to
the LLM un-redacted:
- TCMM host: `VEILGUARD_RENDER_PII_HOOK=1` (render redacts + mints tokens), running
  the new `session_store.py`.
- agent-runtime: `VEILGUARD_REDACT_IN_RENDER=1` (trusts pre-redacted blocks, passes
  through, rehydrates after the LLM call — now reliable thanks to this fix).
Both currently OFF (production-safe). Redactor.py's `redact_memory_blocks` /
passthrough is owned by the other redaction agent — coordinate before flipping, then
benchmark a live LibreChat turn to capture the per-turn win.
