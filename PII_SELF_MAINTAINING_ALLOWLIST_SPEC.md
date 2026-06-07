# PII Self‑Maintaining Allow‑List — Design Spec

**Status:** proposed (2026‑06‑07). Precedes implementation.
**Goal:** stop hand‑curating `pii/allow_list.txt` (whack‑a‑mole). Maintain it from a
signal we already pay for — the per‑block NLP entity/classification call — **without**
weakening PII protection.

---

## 1. The problem
The redactor (`pii/redactor.py`; Presidio + spaCy `en_core_web_lg`; `min_score=0.7`)
false‑positives single‑capitalised code / public nouns as `PERSON` at ~0.85 (e.g.
`Scheduler`, `Dispatcher`). The only correction lever today is the static
`pii/allow_list.txt` (~1800 hand‑curated terms). `min_score` can't be raised past
0.85 without dropping real names, so the allow‑list IS the lever — but curating it by
hand doesn't scale.

## 2. The design constraint — asymmetric risk
- **Over‑redaction** (`Scheduler`→token) = quality bug → revision churn. *Cheap to be wrong.*
- **Wrongly whitelisting a real name** = a real name is **never redacted again** →
  **permanent, silent PII leak.** *Catastrophic to be wrong.*

⇒ the two correction directions get **opposite postures**. Everything below follows from this.

## 3. Where it hooks — no new LLM call
The per‑block NLP contract already extracts **entities + topics**, stamps `block_class`,
and feeds the **cite‑contract**:
- `adapters/ai_studio_nlp_adapter.py` — `_EXTRACT_PROMPT`, `classify_episodic`, JSON
  `response_schema` (already supports structured output).
- consumed at `adapters/veilguard_adapter.py` (post_response / ingest path).

It runs on **RAW text INSIDE the trust zone** — so it sees ground truth the boundary
redactor (which only sees post‑redaction tokens) does not. It is the correct place to
*grade* the redactor. We extend its response schema; we do **not** add a call.

## 4. Schema additions (NLP response)
```jsonc
"entity_disposition": [
  { "text": "Scheduler", "type": "TECH",      // PERSON|ORG|PRODUCT|TECH|LOC|EVENT|OTHER
    "disposition": "public",                   // public | private | uncertain
    "reason": "code/class identifier, not a person" }
],
"redaction_audit": {                           // optional in v1
  "over_redacted": ["Scheduler"],              // tokenized but NOT PII
  "missed_pii":    ["acct 4012‑…"]             // visible PII the redactor would miss
}
```
`over_redacted` needs the NLP to know what the redactor flagged → cheap: re‑run
`redactor.redact_text()` on the raw block in‑process (deterministic) and pass the token
set into the prompt. **v1 can skip this** and derive whitelist candidates purely from
`entity_disposition` (public + would‑be‑flagged by type).

## 5. Data model — stage, never mutate live
```sql
CREATE TABLE redaction_suggestions (
  term            text,
  kind            text,        -- 'whitelist' | 'denylist'
  entity_type     text,
  disposition     text,
  support_count   int,         -- # DISTINCT blocks corroborating
  confidence      real,        -- rolling average
  first_seen      timestamptz,
  last_seen       timestamptz,
  status          text,        -- 'staged' | 'proposed' | 'applied' | 'rejected'
  sample_block_ids text[],
  PRIMARY KEY (term, kind)
);
```

## 6. Promotion logic — asymmetric
**Denylist (missed PII)** → **auto‑apply, fast.** Add the term/pattern to a must‑redact
set. Over‑redacting is acceptable; bias to redact. (Light cap to avoid runaway.)

**Whitelist (over‑redacted / public entity)** → **STAGED.** A term is eligible only if
ALL hold:
- `entity_type NOT IN (PERSON, EMAIL, PHONE, ID, …)`  ← the hard gate
- `disposition == 'public'`  (never `uncertain`)
- `support_count >= K`  (e.g. 3 distinct blocks)
- `confidence >= floor`  (e.g. 0.9)
- `term` not in a small hard‑deny seed (common occupational surnames:
  Walker, Carter, Mason, Cooper, Porter, Hunter, Fletcher, Tanner, Parker, Archer, …)

Even when eligible it becomes a **proposal**, not an auto‑apply (see §7). Rule of thumb:
**when unsure, redact.**

## 7. Governance — reuse the proposals/confirm framework
The org already runs a propose→confirm pipeline (`LifecycleWorker`,
`ConstitutionAmendmentWorker`, `RecalibrationWorker`, the propose→confirm gate). A
redaction‑policy change is exactly a proposal:
- **whitelist additions → emitted as PROPOSALS** → human/critic confirms → applied.
- **denylist additions → auto‑applied** (safe direction).

This is the safety valve for the catastrophic direction: no single block can mutate
global redaction policy; the dangerous edits get a confirm gate.

## 8. Apply mechanism
Promoted whitelist terms → appended to a `pii/allow_dynamic.txt` merged by the loader
(extend `_load_allow_list` to read both files), or appended to `pii/allow_list.txt`.
Redactor reload: today the allow‑list loads once at singleton construction → either
(a) restart both services (`/home/rudol/veilguard/pii` is mounted → `pii-proxy` +
`agent-runtime`), or (b) add an mtime hot‑reload check to `_load_allow_list`.

## 9. Rollout — measure before trusting
1. **Shadow mode (~2 weeks):** add the schema + write to `redaction_suggestions`.
   **Change nothing live.**
2. **Measure:** precision of *would‑be* whitelist promotions — how often it would have
   whitelisted something actually private. This is the leak‑risk meter.
3. **Enable:** if clean → turn on (a) auto‑denylist and (b) *proposed* whitelist.

## 10. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Permanent PII leak via a bad whitelist | type‑gate (`!=PERSON`) + K‑corroboration + confidence floor + proposal confirm gate + `uncertain→redact` + hard‑deny seed |
| Dispositioning LLM sees raw PII | it IS the NLP backend, which **already** reads raw — no new exposure; keep it on a trusted backend (Vertex/local), not a consumer endpoint |
| Global allow‑list (term never redacted anywhere) | type‑gate + hard‑deny seed; per‑tenant scoping is a later option |
| Runaway denylist over‑redacting | cap + the existing allow‑list still wins for known‑safe terms |

## 11. Files to touch
- `adapters/ai_studio_nlp_adapter.py` — extend `response_schema` + `_EXTRACT_PROMPT`.
- `adapters/veilguard_adapter.py` — capture new fields on ingest; upsert `redaction_suggestions`.
- migration — `redaction_suggestions` table.
- new promote worker (mirror a `proposals/*` worker; weekly cadence).
- `pii/redactor.py` — optional `allow_dynamic.txt` merge + mtime hot‑reload.

## 12. Open questions
- K / confidence floor values — tune from shadow‑mode data, don't guess.
- Per‑tenant vs global allow‑list (start global; revisit if a tenant‑specific public
  entity collides with another tenant's private one).
- Denylist representation — exact term vs regex vs entity‑type rule.
