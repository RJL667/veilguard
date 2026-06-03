# Postgres Port Plan — TCMM + Veilguard

Status: DRAFT (2026-06-03). Storage engine LanceDB → **PostgreSQL + pgvector**.
Provider spine already built and proven (`TCMM/core/providers/postgres.py`,
`TCMM/migrations/001_blocks_schema.sql`, `TCMM/_pg_slice_test.py` — 17/17 pass).

---

## 0. Decisions (locked)

| Decision | Value |
|---|---|
| Engine | PostgreSQL 16 + `pgvector` (ANN) + GIN `tsvector` (FTS) + `pgcrypto` |
| Block id (`aid`) | Globally-unique `BIGINT` from one sequence `blocks_aid_seq`. Stays an int → no 200-site string refactor, FAISS-free, short citation tokens. |
| Node type | Explicit `kind` column ('block'/'dream'/'dream_gap'/'dream_reflective'/'entity_vec') — replaces the ~75 `aid >= 10_000_000` range hacks. |
| Tenancy | `namespace` + `user_id` columns on **every** row, **including dreams** (closes the cross-tenant dream leak). |
| Record shape | Real typed columns per field; `jsonb` **columns** only for the inherently key→val link maps + residual; **never** one opaque blob. |
| Bleed fix | Stage flags are columns; workers `UPDATE … WHERE aid=?` (transactional). `__setitem__` never overwrites worker-owned columns on conflict. Clobber is structurally impossible. |
| Vectors | `pgvector` colocated with the row. **FAISS retired.** Sidecar embedding JSON retired. |
| Concurrency | One Postgres server (MVCC + row locks) replaces N processes hammering one LanceDB file. |
| Cutover | Behind `TCMM_STORAGE` flag; Lance stays a working fallback until we delete it. Reversible. |

### Why (the four problems it resolves, at once)
1. **Unique id** — Lance has no identity column; the per-(tenant,namespace) counter collides. Postgres `BIGINT` sequence is unique by construction.
2. **The token bleed** — the lost-update clobber needs a full-row read-modify-write with no transaction. Postgres column-level `UPDATE` + row locks make it impossible.
3. **Dreams not per-tenant** — confirmed isolation bug (dream rows carry no tenant, distinguished only by id range, live in a shared archive). Fixed by `namespace`+`kind` columns.
4. **"Always slow"** — Lance fragmentation/compaction stalls ("FTS INDEX: pending", stale-handle IO-not-found). Postgres autovacuum + B-tree/GIN/HNSW are incremental.

---

## 1. Complete store inventory + disposition

Everything currently lives in **one LanceDB file** `tcmm-data/veilguard/tcmm.db`, written by **three processes** (tcmm-service, agent-runtime, agent-proxy). That multi-process-single-file write pattern is itself a bleed/fragmentation contributor. Target: **one Postgres database**, two logical groups of tables, one connection pool per process.

### 1a. TCMM memory tables (owned by `core/providers/`)
| Lance table / store | holds | → Postgres |
|---|---|---|
| `archive` | memory blocks | **`blocks`** (`kind='block'`) — DONE |
| `dream_archive` | dream/consolidation nodes | **`blocks`** (`kind='dream'`, now namespaced) |
| `embeddings` (`emb_type` archive/semantic/topic) | secondary vectors | record vec → `blocks.vector`; semantic/topic → **`block_vectors(aid,emb_type,vector)`** |
| `sparse_archive` | BM25/FTS index | **`blocks.fts`** (generated `tsvector` + GIN) |
| `sparse_dream` | dream FTS | same `fts` (on `kind='dream'` rows) |
| `_sequences` | id counters | **deleted** — Postgres `blocks_aid_seq` |
| `_meta` (`_TCMM_META_TABLE`) | embedder identity / dim | **`tcmm_meta`** (one-row KV) — used for embedder-mismatch detection |
| FAISS `archive_vector_index`, `dream_vector_index`, `dream_entity_vector_index` (in `recall_indices.py`) | in-memory ANN | **retired** → pgvector HNSW |
| FAISS/BM25 sparse + claims/entity/topic indices | keyword search | **retired** → GIN `tsvector` |
| TinyDB sidecars `archive_embeddings.json`, `topic_embeddings.json`, `dream_archive_embeddings.json` | legacy local-storage vectors | **retired** (vectors live in pgvector) |

### 1b. Agentic / Ledger tables (owned by agent-runtime `LedgerStore` / `app/storage/repository.py`)
All marked `mutable_transactional` — i.e. exactly what Lance is worst at. They have their own seam (`LedgerStore.get().table(name)` + `TABLE_SCHEMAS` in `app/ledger/schemas.py`), so they port behind that abstraction the same way TCMM ports behind its provider ABC.

| Lance table | purpose | → Postgres |
|---|---|---|
| `agent_tasks` | **the Task primitive** — multi-agent work items (inbox_poller, outcomes) | `agent_tasks` |
| `agent_task_heartbeats` | task lease / liveness | `agent_task_heartbeats` |
| `task_comments` | task discussion | `task_comments` |
| `task_proposals` | proposal records | `task_proposals` |
| `proposal_outcomes` | proposal results | `proposal_outcomes` |
| `alignment_weights` | recalibration weights | `alignment_weights` |
| `a2a_external_keys` | agent-to-agent auth keys | `a2a_external_keys` |
| `tenant_proactive_config` | per-tenant config | `tenant_proactive_config` |
| `org_memory` | org-level memory | `org_memory` |
| `pii_audit` | PII + token/cost audit log (the cost ground-truth) | `pii_audit` |

> Note: agent-runtime ALSO **reads** `archive` / `dream_archive` directly (e.g. `lessons_reader.py`, `signal_emitters.py`, `proposals/*`). Those direct `db.open_table("archive")` reads are **leak sites** — they bypass both seams and must move to the provider/SQL (see W3).

### 1c. Embeddings — how they work now (the explicit question)
- **Record (archive) embedding** → `blocks.vector VECTOR(dim)`; ANN via a pgvector **HNSW** index (`vector_cosine_ops`). Replaces the FAISS `archive_vector_index`.
- **Semantic-text & topic embeddings** → **`block_vectors(aid, namespace, user_id, emb_type, vector)`**, one pgvector row per (aid, emb_type). Replaces the Lance `embeddings` table + the FAISS `dream_entity_vector_index`.
- **No FAISS, no sidecar JSON, no `vec_meta.json`.** Embedder identity (`<backend>:<model>`, dim) lives in `tcmm_meta`; a mismatch is detected on startup the same way, but there's no separate index file to drift.
- **Search** = `ORDER BY vector <=> :q LIMIT k` (cosine). Score = `1 - distance`. Cross-namespace / session variants are just different `WHERE` clauses (already implemented in `postgres.py`: `vector_search`, `search_user`).
- **Write path** = `store_embedding(aid, emb, emb_type)` (archive→`blocks.vector`, else→`block_vectors`). Embeddings are local/free (CPU) — unchanged; only their *home* moves.

### 1d. Out of scope (not LanceDB)
- LibreChat **MongoDB** (conversation history) — untouched. Only note: any user message that *references* an aid is a stale-reference risk after re-id (see §5).
- Redis / queues, if any — untouched.

---

## 2. Target architecture

- **One Postgres DB** `tcmm`. Two table groups: memory (`blocks`, `block_vectors`, `tcmm_meta`) and ledger (`agent_tasks`, …). Optionally separate schemas (`mem.`, `ledger.`) — cosmetic.
- **Two seams, two providers** behind their existing abstractions:
  - TCMM `StorageProvider` ABC → `PostgresStorageProvider` (DONE).
  - agent-runtime `LedgerStore`/repository → Postgres-backed ledger (new, mirrors the same pattern: `INSERT … ON CONFLICT`, `UPDATE … WHERE id`, `SELECT … FOR UPDATE SKIP LOCKED` for claims).
- **Connection pooling** per process (`psycopg2.pool.ThreadedConnectionPool`), replacing the per-process `_SharedLanceConnection` singleton.
- **Cross-process safety is free** — it's Postgres's core competency. The lease/claim races and the lost-update clobber both dissolve.

---

## 3. Workstreams

| WS | Scope | Size | Depends on |
|---|---|---|---|
| **W0** | `PostgresStorageProvider` + schema + slice test | DONE | — |
| **W1** | `kind` column: replace ~75 `>=10M` range checks across `heat.py`, `tcmm_core.py`, `recall/*`, `archive.py`, `backfill_channels.py`, `config.py`, `memory_search.py` | M (mechanical) | independent — ship first |
| **W2** | Dream tenant-isolation: stamp `namespace`+`kind` on dream writes (`dream/identity_system.py`, `tcmm_core.alloc_dream_id`, `dream/persistence.py`); fix `backfill_channels` cross-tenant walk | M | W1 (kind) |
| **W3** | Port TCMM **leak sites** off raw Lance/FAISS onto the provider ops: `semantic_polling_worker.py` (`_LanceWriter`/`merge_insert`/`checkout_latest`→`claim`/`update_fields`/`upsert_batch`), `recall_indices.py` (FAISS→pgvector), `recall/recall_bayes.py` + `backfill_channels.py` (direct `lancedb.connect`) | **L** (also the permanent bleed fix) | W0 |
| **W4** | Port the **Ledger** (`agent-runtime/app/storage/repository.py` + `LedgerStore`) to Postgres behind its own seam; port the direct `db.open_table("archive"/"dream_archive")` reads in `proposals/*`, `lessons_reader.py`, `signal_emitters.py` to the provider | **L** | W0 |
| **W5** | Service/API: `tcmm-service/server.py` storage switch (`"lance"`→`"postgres"`), `DATA_DIR`/`LANCE_DB_NAME`→`TCMM_DATABASE_URL`, lineage helpers, `keys()/items()` int assumptions; `veilguard-mcp` + agent-runtime `middleware/tcmm.py` response shapes (aid stays int → minimal) | M | W0,W3,W4 |
| **W6** | Deploy: add `postgres` (pgvector image) to `docker-compose.yml` (+ volume, healthcheck, `depends_on`); `.env` `TCMM_DATABASE_URL`; backups → `pg_dump`; drop `ADMIN_LANCE_DIR`/`TCMM_LANCE_DB` | S | — |
| **W7** | Data migration: Lance → Postgres for all tables, with the **re-id mapping** | M | W1,W2,schema |
| **W8** | Cutover + delete Lance | S | all |

---

## 4. Phased execution (each phase independently shippable & testable)

**Phase A — correctness fixes on Lance, no engine change yet (de-risks everything):**
1. **W1 `kind` column** — add `kind` to the Block/node + the Lance schema (additive), backfill from id-range once, replace every `>=10M` check with `kind=='dream'`. Shippable on Lance. Tests: recall/heat/dream classification unchanged.
2. **W2 dream namespacing** — stamp `namespace`/`user_id`/`kind` on dream writes; backfill existing dream rows from their source blocks' namespace. Test: the cross-tenant dream isolation test (`test_dream_cross_tenant_isolation.py`) goes green.

**Phase B — Postgres provider integration (flagged, parallel to Lance):**
3. **W0 done** → register `"postgres"` in `factory.py`.
4. **W3 leak-site port** — start with the **semantic polling worker** (claim/update_fields) → prove the real enrichment loop runs bleed-free on Postgres (the headline). Then embedding/links workers, then `recall_indices` (pgvector), then the stray `lancedb.connect` readers.
5. Bring up TCMM end-to-end on Postgres behind `TCMM_STORAGE=postgres` in a dev stack; run the existing UAT (`agent-runtime/demo/*`) against it.

**Phase C — Ledger + service + deploy:**
6. **W4 Ledger port** behind `LedgerStore`; **W6 deploy** (compose + env).
7. **W5 service/API** switch; run the live-stack UAT.

**Phase D — migrate + cut over:**
8. **W7 migration** (below), **W8 cutover** (flag flip), monitor, then delete Lance.

---

## 5. Data migration (W7)

1. **Stand up empty Postgres schema** (memory + ledger).
2. **Memory:** read every Lance row (per namespace), assign a **new global `aid`**, write a mapping `old(user,ns,aid) → new_aid`. Re-write ALL references through the map: `prev_aid`/`next_aid`, `source_block_ids`, `cited_aids`, `lineage_root`/`parents`, `semantic_links`/`entity_links`/`topic_links` keys, `cite_scores` keys. Dreams get `kind='dream'` + their source blocks' `namespace`. Vectors → `blocks.vector` / `block_vectors`.
   - *Simplification for the single prod namespace:* if all live data is one namespace, preserve existing aids and just `setval` the sequence past the max + give dream rows their kind/namespace — only cross-namespace collisions need remapping.
3. **Ledger:** straight table copy (`agent_tasks` etc.); rewrite any stored memory-aid references through the same map.
4. **Cache-bust:** the Anthropic prompt cache keys on rendered TCMM blocks → **re-render the corpus** after re-id. Note `pii_audit` rows that stored old aids (historical, low-risk).
5. **Verify:** row counts match per table; spot-check link integrity; recall returns the same top-k for a sample of queries (Lance vs Postgres).

---

## 6. Cutover & rollback (W8)

- `TCMM_STORAGE` (and a ledger equivalent) select engine per process. Default stays `lance` until the dev-stack UAT + migration verify pass.
- Flip to `postgres`, watch one enrichment cycle: **requests ≈ row count, zero re-enrichment** (the bleed is gone), recall latency, FTS/ANN correctness.
- Rollback = flip the flag back (Lance files retained read-only until we're confident, then deleted).

---

## 7. Risks & open questions

- **Re-id blast radius** — the reference rewrite is the riskiest step; mitigated by the mapping table + the single-namespace simplification + verify pass.
- **pgvector recall parity** — confirm HNSW (cosine) returns equivalent neighbours to FAISS IP; tune `m`/`ef_search`; normalize vectors to match IP-vs-cosine.
- **Ledger schema fidelity** — `TABLE_SCHEMAS` in `app/ledger/schemas.py` must map cleanly to SQL DDL; some fields may be jsonb.
- **Dim pinning** — `VECTOR(dim)` must match the embedder; set from `tcmm_meta` at deploy.
- **Two providers, one DB** — keep memory and ledger ports coordinated (shared connection config), but they can ship independently behind their separate seams.

---

## 8. Artifacts already in place
- `TCMM/core/providers/postgres.py` — provider (ABC + `scan`/`claim`/`update_fields`/`vector_search`), conflict policy preserves worker-owned columns.
- `TCMM/migrations/001_blocks_schema.sql` — canonical DDL.
- `TCMM/_pg_slice_test.py` — 17/17 on real Postgres+pgvector (unique id, typed record, bleed fix, claim, pgvector search, tenant+kind isolation).
