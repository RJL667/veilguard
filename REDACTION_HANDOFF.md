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
