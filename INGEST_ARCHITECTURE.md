# Veilguard Ingest Architecture

> Status: **design + Phase 1 building** (2026-06-12). Adds a first-class
> way to ingest **documents and files** (PDF, Excel, Word, images, and
> recursive zips) straight into TCMM archive memory. Builds directly on the
> existing `services/connectors/` spine (`parse_to_text` / `chunk_text` /
> provenance) and the Postgres archive + staged NLP worker. North star: a
> user (or an agent, or a connector) can point at a file — or a zip full of
> files — and have its content become durable, recallable knowledge, with
> full provenance and tenant isolation intact.

---

## 1. The shape of the problem

Today TCMM ingests **conversation turns**. `POST /ingest_turn` takes
`{text, origin}` items and truncates each to 8000 chars
(`veilguard_adapter.py:1502`) — correct for tool round-trips, wrong for a
200-page manual. There is no path that says *"here is a body of knowledge,
chunk it properly and remember it."*

We want two layers, split exactly along the boundary the user framed:

- **TCMM owns `text → archive`.** A bulk write primitive that chunks
  properly, keeps the chunks cohesive, and lets the background workers
  enrich them — without the turn-shaped truncation.
- **Veilguard owns `file → text`.** Read PDFs/Excel/Word/images down to
  text (with OCR fallback), walk zips recursively, and feed the result to
  the TCMM primitive.

### The load-bearing insight

**A file is a degenerate connector.** The connectors framework already runs
`read() → parse_to_text() → chunk_text() → provenance → TCMM`
(`CONNECTORS_ARCHITECTURE.md` §4). The only missing piece is an *eager bulk
write* — connectors today drip content into the recall/hint path; nothing
says "ingest this whole document into the archive now." So we build **one**
new primitive (`/ingest_text`) and a file front-end that reuses the rest of
the spine. No parallel parsing stack, no second chunker.

```
  ┌─────────────────────────────  VEILGUARD LAYER  ─────────────────────────────┐
  │  ingest_files(sources, scope)         "file → text"                          │
  │     ├─ resolve bytes  (upload | daemon-fetch | connector.read)               │
  │     ├─ dispatch by type                                                      │
  │     │     ├─ zip/tar  → recurse (depth + ratio + count guards)               │
  │     │     ├─ pdf/img  → parse_to_text → OCR fallback (ocr_client)            │
  │     │     └─ docx/xlsx/pptx/csv/… → parse_to_text (LlamaIndex)               │
  │     ├─ chunk_text()  (512/64 tok, sentence-aware)   ← the GOOD chunker       │
  │     └─ POST /ingest_text  (one call per document, N chunks)                  │
  └───────────────────────────────────┬─────────────────────────────────────────┘
                                       │  HTTP
  ┌────────────────────────────────────▼──────────────  TCMM LAYER  ─────────────┐
  │  POST /ingest_text   →  ingest_document()          "text → archive"           │
  │     ├─ fresh per-doc lineage_root  (NOT the chat tip)                         │
  │     ├─ recall-only:  NO live blocks  (never reinjected into the prompt)       │
  │     ├─ intra-doc temporal chain  (traverse walks the doc in reading order)    │
  │     ├─ channel + provenance stamped per chunk (extras_json)                   │
  │     └─ archive-first (sync rows) → NLP/embeddings background (worker-owned)   │
  └───────────────────────────────────────────────────────────────────────────--─┘
```

---

## 2. Layer A — TCMM: the `/ingest_text` write primitive

**New endpoint** `POST /ingest_text` on `services/tcmm-service/server.py`,
backed by `VeilguardTCMM.ingest_document()` (adapter) →
`TCMM.ingest_document()` (core, `core/tcmm_core.py`).

### 2.1 Why a new method, not `add_new_block` in a loop

`add_new_block` is built for conversation turns and does two things that are
**wrong** for documents:

1. **Inherits `lineage_root` from `self.live_blocks[-1]`**
   (`tcmm_core.py:2389`) — the conversation tip. A naive loop would chain
   document chunks onto the live chat *and* onto each other across
   documents. Documents need a **fresh, per-document lineage root**.
2. **Appends every block to `self.live_blocks`** (`tcmm_core.py:2508`) — the
   working set that gets reinjected into the prompt. We do **not** want 188
   chunks of a manual injected into the live context. Document chunks must
   be **recall-only**: archive rows that surface via search/traverse, never
   the live prompt.

`ingest_document` therefore writes archive rows directly via
`_ensure_archive_entry` (`core/archive.py:946`) and **does not create live
blocks at all**.

### 2.2 What `ingest_document` does, per chunk

| Step | Detail |
|---|---|
| Archive row | `_ensure_archive_entry(chunk, fp)` → new `aid` from the global PG sequence; row written with `kind="block"`, `embedding_done=semantic_done=False` |
| Cohesion | First chunk's `aid` becomes the **document lineage root**; every chunk's `lineage.root` / `lineage_root` is set to it |
| Reading order | Intra-document **temporal chain** — `temporal.prev_aid` / `next_aid` link chunk *i* to *i±1* (overrides the conversation wiring `_ensure_archive_entry` defaulted in) |
| Identity | `priority_class="DOCUMENT"`, `origin="document"`, `source` = e.g. `file:manual.pdf` or `document:<doc_id>` |
| Recallability | `recallable=True` explicitly. Document prose never matches a `NON_RECALLABLE` class (`episodic_ontology.py:72`), so it stays recallable |
| Scoping | `channel` stamped on the promoted column (NULL → agent_private at recall) |
| Provenance | `extras_json["document"]` = `{doc_id, filename, mime, sha256, title, page, ingest_batch_id, path_in_archive, chunk_index, chunk_total, …}` |
| Write-back | `self.archive[aid] = entry` — the durable write (PG proxies are stateless copies; mutate-in-place is silently lost) |

### 2.3 Enrichment is worker-owned (archive-first)

Rows land **synchronously**; the background staged worker fills the vector
(`embedding_done=False`) and runs NLP — entities/topics/claims/semantic_text
(`semantic_done=False`). On Postgres the inline embed is already skipped
(`_uses_polling_workers`, `archive.py:1105`); the embedding stage drains
~1.5k blocks/s. So a big document **returns fast** and warms over the next
minute. This is the same "archive-first, NLP-background" contract every
block already follows — no new ordering guarantees to reason about.

### 2.4 Request / response contract

```jsonc
// POST /ingest_text
{
  "conversation_id": "conv-abc",            // namespace
  "user_id": "user@…",
  "doc_id": "sha256:…",                     // optional idempotency / grouping key
  "title": "ACME Security Manual",
  "source": "file:acme_manual.pdf",         // optional; defaults to document:<doc_id>
  "channel": "team_knowledge",              // DEFAULT — durable, cross-session KB
  "chunks": ["…", "…"],                     // PREFERRED: pre-chunked by the Veilguard layer
  "text": "…",                              // OR raw text → core word-budget split (fallback)
  "provenance": { "filename": "…", "mime": "…", "sha256": "…", "pages": 212 }
}
// →
{ "doc_id": "sha256:…", "lineage_root": 4711, "chunks_added": 188,
  "requested": 188, "session_id": "conv-abc" }
```

---

## 3. Layer B — Veilguard: `ingest_files`

A sub-agent tool (so an agent can ingest mid-conversation) backed by a shared
orchestrator that reuses the connector spine. **Phase 2.**

### 3.1 Byte sources (one abstraction, three resolvers — all already exist)

| Source | Mechanism | Constraint |
|---|---|---|
| Upload | LibreChat upload / OCR shim path | — |
| Daemon fetch | `host_docs._fetch_bytes_via_daemon()` — file on the user's Windows machine | 15 MB cap; no spaces in path |
| Connector | `connector.read(ref)` bytes | per-connector |

### 3.2 Per-file pipeline (as built)

```
ingest_core.expand(name, bytes)             # recurse archives → leaf (path, bytes) + skips
  └─ for each leaf:
       _extract_text_from_bytes(name, data) # tools/host_docs.py
         ├─ pdf   → _parse_pdf_file (fitz) → OCR fallback for scanned/garbled pages
         ├─ xlsx  → openpyxl (every sheet → text)
         ├─ docx  → python-docx (headings + paragraphs + tables)
         ├─ pptx  → python-pptx (per-slide title + text frames)
         ├─ image → ocr_bytes()             # Gemini shim via pii-proxy /v1/ocr
         └─ text/csv/md/json/code → decode (utf-8 → latin-1)
       ingest_core.chunk_text(text)         # sentence-aware, ~300-word target, 50 overlap
       POST /ingest_text                    # one document, N chunks
```

**Format coverage (as-built, sub-agents host):** PDF (PyMuPDF + OCR
fallback), XLSX (openpyxl), **DOCX (python-docx)**, **PPTX (python-pptx)**,
images (Gemini OCR), and text/CSV/Markdown/code (direct decode). Still
*unsupported*: legacy binary `.doc`/`.ppt` (python-docx/-pptx read OOXML
only), EPUB/ODT/RTF — these surface in the manifest as `unsupported` (export
to PDF/DOCX).

> **Why not LlamaIndex.** The original design floated reusing the connector
> `parse_to_text` (LlamaIndex) for breadth. We didn't: the two formats that
> actually mattered — Word and PowerPoint — only need `python-docx` /
> `python-pptx`, which LlamaIndex itself calls under the hood **and which
> Veilguard already ships in `services/documents/`**. So we added those two
> small, already-vetted libs to sub-agents and mirrored the documents
> server's `read_docx`/`read_pptx` extraction (from `BytesIO`, so it works
> for daemon-fetched bytes too). Result: full Office coverage without the
> LlamaIndex framework, its ~hundreds-of-MB tree, or its startup cost. The
> chunker is the dependency-free sentence splitter in `ingest_core`.
>
> *Considered and rejected:* routing DOCX/PPTX through the `documents`
> container (which has the libs) — but that container has no client-daemon
> access to the user's machine and reads only `/workspace`, so it can't see
> the files `ingest_files` resolves. Duplicating two tiny pure-Python libs
> is simpler and works for both local and daemon-fetched files.

### 3.3 Return a manifest, not a count

```jsonc
{ "ingest_batch_id": "batch-…",
  "files": [ { "path": "manuals.zip/acme.pdf", "status": "ingested",
               "chunks": 188, "bytes": 1048576, "doc_id": "sha256:…" } ],
  "totals": { "files": 12, "chunks": 1043, "bytes": 18432000 } }
```

`status ∈ {ingested, ocr, skipped_dup, unsupported, error}`.

---

## 4. Recursion — zips (and tars for free). **Phase 3.**

```python
def _walk(name, raw, ctx, depth=0):
    if is_archive(name):                      # .zip / .tar / .tgz
        guard(depth < MAX_DEPTH)              # nested-zip bomb
        for inner_name, inner_bytes in iter_archive(raw):
            if skip(inner_name): continue     # __MACOSX, .DS_Store, dotfiles
            guard(total_uncompressed += len(inner_bytes) < MAX_TOTAL_UNCOMPRESSED)
            guard(file_count += 1 < MAX_FILES)
            _walk(f"{name}/{inner_name}", inner_bytes, ctx, depth+1)   # recurse
    else:
        ingest_one(name, raw, ctx)            # path_in_archive = full nested name
```

**Zip-bomb guards (non-negotiable for arbitrary uploads):** `MAX_DEPTH=3`,
`MAX_FILES≈500`, `MAX_TOTAL_UNCOMPRESSED≈200 MB`, per-entry compression-ratio
cap, per-file size cap (~25 MB). **Every guard trip is logged into the
manifest, never silent** — silent truncation reads as "ingested everything"
when it didn't.

---

## 5. Scope / channel — knowledge-base by default

Decision (2026-06-12): ingested files default to **`channel=team_knowledge`**
— durable, cross-session, digest-eligible the moment enrichment lands.

- **Tenant isolation still bounds it.** `team_knowledge` is "shareable within
  the tenant," but the structural `(namespace, user_id)` + shared-anchor
  scoping still walls it off from other tenants. For single-user-per-tenant
  deployments this is effectively *"this user's durable knowledge base."*
- **`conv` is the opt-in override** — `ingest_files(..., scope="conv")` for
  "just this chat, don't pollute the KB."
- **Digest-eligibility means proactive recall.** A freshly-ingested manual can
  influence the *next* turn without an explicit search. The doc-anchor node
  (Phase 5) is the mitigation so a giant zip digests as *one* anchor, not 200
  chunks.

---

## 6. Cross-cutting

- **Idempotency / dedup.** `doc_id = sha256(bytes)` is stamped now (provenance
  + grouping + future dedup). Doc-level "skip if already ingested" needs a
  provider-level filtered lookup (`source`-prefix / `extras_json.doc_id`) that
  Postgres doesn't expose cheaply yet — **deferred to a small follow-up**
  (Phase 1.5). Until then re-ingest creates new rows.
- **PII.** Raw text goes into TCMM — correct and intended: the PII proxy is the
  *LLM* egress boundary, not the TCMM boundary. OCR rides pii-proxy for
  *extraction*, not redaction.
- **Cost / perf.** A 200-page PDF = hundreds of background NLP + embedding
  jobs. Archive-first keeps the call fast, but a 50-file zip can flood the
  staged worker / trip the AI-Studio billing breaker. Add a per-batch
  throttle (Phase 3). The naive per-chunk async sparse-index thread is also a
  reason to prefer the bulk path for very large documents (Phase 1.5).
- **Doc anchor node (Phase 5).** A synthetic lineage-root block holding
  `{title, source, summary}` that surfaces first in recall; chunks hang off it
  via `traverse`. Mirrors the dream-anchor pattern — makes a 200-chunk PDF
  feel like *one* memory instead of 200 noisy hits, and is what digests
  instead of every chunk.

---

## 7. Build phases

| Phase | Scope | State |
|---|---|---|
| **1** | TCMM `/ingest_text` + `ingest_document()` (fresh lineage, recall-only, provenance, worker-owned enrichment) + pure unit tests | **DONE (local)** |
| **2** | `ingest_core` orchestrator (classify + chunk + recursive `expand`) + `ingest_files` sub-agent tool + `_extract_text_from_bytes` (fitz/openpyxl/OCR) + `utils/tcmm.ingest_text` + 16 pure tests | **DONE (local)** |
| **3** | Recursive zip/tar walk + bomb guards (depth/count/size/ratio) + manifest | **DONE — folded into Phase 2 (`ingest_core.expand`)** |
| 1.5 | Doc-level dedup (provider filtered lookup) + bulk-ingest path for very large docs | planned |
| **2.1** | DOCX + PPTX coverage via `python-docx`/`python-pptx` (mirrors `services/documents/`), no LlamaIndex | **DONE (local)** |
| 2.5 | Per-batch embedding throttle; EPUB/legacy-Office coverage | planned |
| 5 | Doc-anchor / summary node + UI "ingest to knowledge base" affordance | planned |

---

## 8. Key file references

| Purpose | File | Lines |
|---|---|---|
| New write endpoint | `services/tcmm-service/server.py` | `/ingest_text` (after `/ingest_turn` @2856) |
| Adapter wrapper | `…/TCMM/adapters/veilguard_adapter.py` | `ingest_document` (after `ingest_turn` @1473) |
| Core write primitive | `…/TCMM/core/tcmm_core.py` | `ingest_document` (after `add_new_block` @2357) |
| Archive row creation | `…/TCMM/core/archive.py` | `_ensure_archive_entry` @946 ; `_create_unified_node` @198 |
| Recallable classes | `…/TCMM/core/episodic_ontology.py` | `NON_RECALLABLE_CLASSES` @72 ; `derive_recallable` @166 |
| Orchestrator (pure) | `services/sub-agents/tools/ingest_core.py` | `kind_of` / `chunk_text` / `expand` (+ `Limits` guards) |
| `ingest_files` tool + extractor | `services/sub-agents/tools/host_docs.py` | `ingest_files` (in `register`) ; `_extract_text_from_bytes` |
| TCMM client | `services/sub-agents/utils/tcmm.py` | `ingest_text` (POST `/ingest_text`) |
| Connector chunker (not used here) | `services/connectors/_base/parsing.py` | `parse_to_text` @59 ; `chunk_text` @164 |
| OCR shim client | `services/_shared/ocr_client.py` | `ocr_bytes` @33 |
| Daemon file fetch | `services/sub-agents/tools/host_docs.py` | `_fetch_bytes_via_daemon` @158 |
| Channel scoping | `TCMM_CHANNEL_ARCHITECTURE.md` | §3 (values) §5–6 (write/recall) |
