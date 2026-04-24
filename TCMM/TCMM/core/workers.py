from threading import Thread
import re
import json
import time
import queue
import os
from .entropy_signals import cosine as _cosine
from .config import CFG
try:
    from tinydb import TinyDB, Query
except ImportError:
    TinyDB = None
    Query = None

IDENTIFIER_REGEX = re.compile(
    r'\b(?:[A-Z]+-\d+(?:-[A-Z0-9]+)*|\d{2,}|[A-Z]{2,}-\d{2,}|[A-Z0-9]{3,}-[A-Z0-9\-]{2,})\b'
)

_TEMPORAL_RE = re.compile(
    r'^\d{1,2}:\d{2}(:\d{2})?$'       # HH:MM or HH:MM:SS
    r'|^\d{4}-\d{2}-\d{2}$'            # YYYY-MM-DD
    r'|^\d{1,2}/\d{1,2}/\d{2,4}$'      # M/D/YY or MM/DD/YYYY
)
_TEMPORAL_WORDS = frozenset({
    "today", "yesterday", "tomorrow", "now", "later",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "morning", "evening", "afternoon",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "session", "am", "pm",
})

def normalize_entities(entities):
    cleaned = []
    for e in entities:
        e = e.strip().title() # Phase 17: Standardize casing

        # Remove pure numbers
        if e.replace('.', '').isdigit():
            continue

        # Remove single characters
        if len(e) < 2:
            continue

        # Remove temporal patterns (timestamps, dates)
        if _TEMPORAL_RE.match(e.lower()):
            continue

        # Remove common temporal words
        if e.lower() in _TEMPORAL_WORDS:
            continue

        # Remove tokens that are mostly digits (e.g. "16:51:22", "2026")
        stripped = re.sub(r'[\s\-:/.]', '', e)
        if stripped and sum(c.isdigit() for c in stripped) / len(stripped) > 0.6:
            continue

        cleaned.append(e)

    return list(set(cleaned)) # Deduplicate

def looks_truncated(text: str) -> bool:
    if not text: return True
    if text.endswith("...") or text.endswith(".."): return True
    if text[-1] not in ".!?\"'": return True
    return False




from .tcmm_logging import _log_warn, log_exception, log_adapter, _log_tcmm, log_console, log_semantic, log_perf, summary, format_l2
from . import tcmm_debug as dbg

def _emit_indices_stage(tcmm):
    recall_id = getattr(tcmm, "_active_recall_id", None)
    if not recall_id:
        return
    summary.stage(recall_id, "indices", {
        "dense_dirty": bool(getattr(getattr(tcmm, "archive_vector_index", None), "_dirty", False)),
        "sparse_dirty": bool(getattr(getattr(tcmm, "archive_sparse_index", None), "_dirty", False)),
        "semantic_queue_size": getattr(tcmm, "semantic_queue", None).qsize() if hasattr(tcmm, "semantic_queue") else 0,
        "embedding_queue_size": getattr(tcmm, "embedding_queue", None).qsize() if hasattr(tcmm, "embedding_queue") else 0,
    })


def _tag_async(tcmm, msg: str) -> str:
    recall_id = getattr(tcmm, "_active_recall_id", None)
    if recall_id:
        return format_l2(recall_id, "indices", msg)
    return msg


def _log_batch(tcmm, worker, reason, batch_size, queue_size,
               accepted, queue_wait_ms=None, sem_wait_ms=0.0, embed_ms=0.0, written=0, **kwargs):
    if queue_wait_ms is None:
        queue_wait_ms = kwargs.get("waited_ms", 0.0)
    msg = (
        f"[{worker}] flush reason={reason} "
        f"batch={batch_size} queued={queue_size} "
        f"accepted={accepted} "
        f"queue_wait={queue_wait_ms:.1f}ms sem_wait={sem_wait_ms:.1f}ms embed={embed_ms:.1f}ms "
        f"written={written}"
    )
    log_perf(_tag_async(tcmm, msg))
    _emit_indices_stage(tcmm)

try:
    from src.memory.TCCM import json_utils
except ImportError:
    try:
        from .. import json_utils
    except ImportError:
        import json_utils

# Use Adapter (Remove direct imports/init of heavy ML libraries)
# import spacy
# from keybert import KeyBERT
# from sentence_transformers import SentenceTransformer
# _nlp = spacy.load(...) --> MOVED to adapters/nlp_adapter.py


# ============================================================
# Archive embedding worker
# ============================================================
# ... unchanged embedding_worker_loop ...

# ============================================================
# Semantic analysis worker (heavy lifting)
# ============================================================




def embedding_worker_loop(tcmm):
    """
    Consumes tcmm.embedding_queue
    Strict Batched Implementation (Phase 108)
    Optimized Flush Logic (Phase 135) - Uses batch_start_time to avoid premature timeouts
    """
    batch = []
    batch_start_time = None
    should_exit = False

    while True:
        # 0. Priority Gate: If semantic backlog exists, yield the Cold worker
        if not tcmm.semantic_embed_queue.empty():
            time.sleep(0.1)
            continue

        # 1. Collect Batch
        try:
            # Smart Timeout: If we have a batch, wait only for the remaining window.
            # If empty, wait for poll interval.
            if batch:
                elapsed = time.time() - batch_start_time
                timeout = max(0, CFG.EMBED_BATCH_MAX_WAIT - elapsed)
                if timeout == 0: timeout = 0 # Immediate check if time is up
            else:
                timeout = 0.1  # 100ms poll when empty
            
            item = tcmm.embedding_queue.get(timeout=timeout)
            
            if item is None:
                tcmm.embedding_queue.task_done()
                should_exit = True
            else:
                if not batch:
                    batch_start_time = time.time() # Start the clock on first item
                batch.append(item)
            
        except queue.Empty:
            pass
        except Exception as e:
            log_exception("Embedding Worker Queue Error", e)
            continue

        # 2. Check Flush Conditions
        now = time.time()
        is_full = len(batch) >= CFG.COLD_EMBED_BATCH_SIZE
        is_timeout = batch and (now - batch_start_time) >= CFG.EMBED_BATCH_MAX_WAIT
        
        # Phase 108: Flush on full, timeout, or exit signal
        if batch and (is_full or is_timeout or should_exit):
            # Phase 110: Opportunistic Drain
            while len(batch) < CFG.COLD_EMBED_BATCH_SIZE:
                try:
                    extra = tcmm.embedding_queue.get_nowait()
                    if extra is None:
                        should_exit = True
                        break
                    batch.append(extra)
                except queue.Empty:
                    break
                except Exception:
                    break

            # Process Batch
            aids = []
            texts = []
            valid_items = []
            
            # Unpack items
            for it in batch:
                try:
                    curr_aid = None
                    curr_text = None
                    
                    if isinstance(it, tuple):
                        curr_aid, curr_text = it
                    else:
                        curr_aid = it
                        curr_text = tcmm.resolve_text(curr_aid)
                    
                    if curr_aid is not None and curr_text:
                         aids.append(curr_aid)
                         texts.append(curr_text)
                         valid_items.append(it)
                except Exception:
                    pass

            # Perform Embedding
            if aids:
                flush_started = time.time()
                reason = "size" if len(batch) >= CFG.EMBED_BATCH_SIZE else "timeout"
                if should_exit: reason = "exit"
                accepted = len(texts)
                queue_size = tcmm.embedding_queue.qsize()

                if not tcmm.embedder:
                     # No adapter - clear
                     pass
                elif not hasattr(tcmm.embedder, "embed_batch"):
                    log_exception(
                        "[EmbeddingWorker] adapter missing embed_batch()",
                        RuntimeError(type(tcmm.embedder).__name__)
                    )
                else:
                    try:
                        # Yield to semantic if pending (prevents cold blocking semantic)
                        if not tcmm.semantic_embed_queue.empty():
                            time.sleep(0.01)  # 10ms - prevent busy loop
                            # We can't clear batch here without losing data or complexity.
                            # Just proceed for now, parallelization handles concurrency better anyway.
                            pass 
                        
                        # PARALLELIZATION: Cold worker runs independently.
                        embed_start = time.time()
                        # Use embed_batch for passages
                        embeddings = tcmm.embedder.embed_batch(texts)
                        embed_ms = (time.time() - embed_start) * 1000.0
                        sem_wait_ms = 0.0 # No longer relevant

                        
                        if len(embeddings) != len(aids):
                            _log_warn(
                                "[ColdEmbedWorker] embed_query_batch returned "
                                f"{len(embeddings)} vectors for {len(aids)} inputs"
                            )

                        written = 0
                        for aid, emb in zip(aids, embeddings):
                            if emb is None:
                                _log_warn(f"[ColdEmbedWorker] empty embedding for aid={aid}")
                                continue

                            entry = tcmm.get_archive_entry(aid)
                            if entry:
                                entry["embedding"] = emb
                                entry["isDirty"] = True
                                tcmm.archive[aid] = entry  # Write back to provider
                                tcmm.archive_embeddings[aid] = emb
                                tcmm.archive_vector_index.add(aid, emb)
                                written += 1
                        
                        # Calculate wait time based on when the batch started
                        queue_wait_ms = (flush_started - batch_start_time) * 1000.0 if batch_start_time else 0.0
                        
                        _log_batch(
                            tcmm,
                            worker="ColdEmbedWorker",
                            reason=reason,
                            batch_size=len(batch),
                            queue_size=queue_size,
                            accepted=accepted,
                            queue_wait_ms=queue_wait_ms,
                            sem_wait_ms=sem_wait_ms,
                            embed_ms=embed_ms,
                            written=written
                        )

                        # Checkpoint: persist embedding-enriched nodes
                        if aids:
                            try:
                                tcmm.archive.persist(aids)
                            except Exception as pe:
                                log_exception("ColdEmbedWorker persist failed", pe)

                    except Exception as e:
                        log_exception(
                            "ColdEmbedWorker batch failure (embed_query_batch)",
                            e
                        )

            # Cleanup
            for _ in batch:
                tcmm.embedding_queue.task_done()
            batch = []
            batch_start_time = None
        
        if should_exit and not batch:
             break


# ============================================================
# Semantic worker (archive backend)
# ============================================================

import time

def semantic_worker_loop(tcmm):
    """
    Consumes tcmm.semantic_queue
    Queue item: aid (int) [v2] OR (aid, text) [v1 legacy]
    Phase 15: Parallelized via Batching
    """
    while True:
        batch_items = []
        batch_data = []  # Initialize before try block to prevent UnboundLocalError
        try:
            # 1. Accumulate Batch — wait for items to build up
            item = tcmm.semantic_queue.get()
            if item is None:
                tcmm.semantic_queue.task_done()
                break
            batch_items.append(item)

            # Give the queue time to fill before processing.
            # During bulk ingest many items land in quick succession;
            # a short wait lets us scoop them into one batch instead
            # of processing one-for-one.
            _batch_max = getattr(tcmm.nlp_adapter, "DEFAULT_BATCH_SIZE", 32)
            _batch_wait = 0.15          # seconds to wait for more items
            _batch_deadline = time.time() + _batch_wait
            while len(batch_items) < _batch_max:
                _remaining = _batch_deadline - time.time()
                if _remaining <= 0:
                    break
                try:
                    next_item = tcmm.semantic_queue.get(timeout=_remaining)
                    if next_item is None:
                        tcmm.semantic_queue.put(None)
                        break
                    batch_items.append(next_item)
                except queue.Empty:
                    break
            
            # 2. Pre-process Batch (Text resolution & Context)
            batch_data = []
            texts_to_process = []
            
            for b_item in batch_items:
                start_time = time.time()
                tcmm._metrics["semantic.dequeue"] += 1

                prev_ctx = None
                if isinstance(b_item, dict):
                    aid = b_item.get("aid")
                    prev_ctx = b_item.get("prev_semantic")
                    text = tcmm.resolve_text(aid)
                elif isinstance(b_item, tuple):
                   aid, text = b_item
                else:
                   aid = b_item
                   text = tcmm.resolve_text(aid)
                
                if not text:
                   continue
                    
                text_str = str(text)
                if not text_str.strip():
                    continue

                # Contextual combined text
                entry = tcmm.get_archive_entry(aid) or {}
                prev_aid = entry.get("temporal", {}).get("prev_aid")
                if prev_aid == aid or prev_aid is None:
                    prev_aid = aid - 1 if aid > 0 else None
                
                prev_text = ""
                if prev_aid is not None:
                    prev_entry = tcmm.get_archive_entry(prev_aid)
                    if prev_entry:
                        prev_text = prev_entry.get("text", "") or ""
                
                if prev_text:
                    prev_chunk = prev_text[-500:]
                    combined_text = prev_chunk + "\n" + text_str
                    # Offset where current block text starts (for entity boundary filtering)
                    current_text_offset = len(prev_chunk) + 1
                else:
                    combined_text = text_str
                    current_text_offset = 0
                texts_to_process.append(combined_text)

                # Collect previous node's topics + entities as GLiNER hints
                _prev_topics_hint = []
                _prev_entities_hint = []
                if prev_ctx:
                    _prev_topics_hint = list(prev_ctx.get("topics", []) or [])
                    _prev_entities_hint = list(prev_ctx.get("entities", []) or [])
                elif prev_aid is not None:
                    _pn = tcmm.get_archive_entry(prev_aid)
                    if _pn:
                        _prev_topics_hint = list(_pn.get("topics", []) or [])
                        _prev_entities_hint = list(_pn.get("entities", []) or [])

                batch_data.append({
                    "aid": aid,
                    "text_str": text_str,
                    "combined_text": combined_text,
                    "current_text_offset": current_text_offset,
                    "prev_ctx": prev_ctx,
                    "start_time": start_time,
                    "topic_hints": _prev_topics_hint,
                    "entity_hints": _prev_entities_hint,
                    "role": (entry.get("origin") or "unknown").lower(),
                })

            if not batch_data:
                continue

            # 3. Parallel NLP Batch Execution
            nlp_adapter = getattr(tcmm, "nlp_adapter", None)
            if not nlp_adapter:
                 continue

            # Pass text boundary offsets for entity position filtering
            text_offsets = [bd["current_text_offset"] for bd in batch_data]
            _topic_hints = [bd["topic_hints"] for bd in batch_data]
            _entity_hints = [bd["entity_hints"] for bd in batch_data]
            _roles = [bd["role"] for bd in batch_data]
            batch_nlp_start = time.time()
            # Gemma is the only NLP path now. GLiNER + KeyBERT removed;
            # the old process_batch code still exists as a legacy fallback
            # but we no longer dispatch to it.
            if hasattr(nlp_adapter, "process_batch_gemma"):
                batch_results = nlp_adapter.process_batch_gemma(texts_to_process, roles=_roles)
            else:
                batch_results = nlp_adapter.process_batch(texts_to_process, text_offsets=text_offsets, topic_hints=_topic_hints, entity_hints=_entity_hints, roles=_roles)
            batch_nlp_elapsed = time.time() - batch_nlp_start
            # Quiet batch metric — only print every 5th batch to reduce noise
            tcmm._metrics.setdefault("nlp_batch_count", 0)
            tcmm._metrics["nlp_batch_count"] += 1
            if tcmm._metrics["nlp_batch_count"] % 5 == 0:
                print(f"  [NLP] batch {tcmm._metrics['nlp_batch_count']}: {len(texts_to_process)} texts @ {batch_nlp_elapsed/max(1,len(texts_to_process)):.2f}s/text", flush=True)

            # 4. Sequentially Post-process Results
            for data, res in zip(batch_data, batch_results):
                item_start = time.time()
                aid = data["aid"]
                text_str = data["text_str"]
                combined_text = data["combined_text"]
                prev_ctx = data["prev_ctx"]
                
                doc = res.get("doc")
                raw_ents = res.get("entities", [])
                entity_dicts = res.get("entity_dicts", [])

                # Phase 141: Strengthen Entity Extraction
                identifiers = IDENTIFIER_REGEX.findall(combined_text)
                if identifiers:
                    raw_ents.extend(identifiers)

                # Phase 140: Entity Normalization Filter
                entities = normalize_entities(raw_ents)
                
                # Topics (Pre-extracted in batch by adapter)
                topics = res.get("topics", [])

                identifier_count = len(identifiers)
                line_count = text_str.count('\n') + 1
                structure_score = 0
                if line_count >= 3: structure_score += 1
                if (len(text_str) / max(1, line_count)) < 120: structure_score += 1

                # Phase 6b: Entity Stabilisation
                if prev_ctx:
                    prev_ents = set(prev_ctx.get("entities", []) or [])
                    stabilized = []
                    for e in entities:
                        if e in prev_ents:
                            stabilized.append(e)
                            continue
                        found_match = False
                        for p in prev_ents:
                            if (p in e or e in p) and abs(len(e) - len(p)) < 8:
                                stabilized.append(p)
                                found_match = True
                                break
                        if not found_match:
                            stabilized.append(e)
                    seen = set()
                    entities = []
                    for x in stabilized:
                        if x not in seen:
                            entities.append(x)
                            seen.add(x)

                # Claims — use SVO-structured claims from NLP adapter
                claims = res.get("claims", [])
                claim_scores = res.get("claim_scores", [])
                topic_dicts = res.get("topic_dicts", [])

                # Resolve previous node context — prefer prev_ctx from queue,
                # fall back to reading the archive node via temporal link
                semantic_out = {}
                _prev_ents = set()
                _prev_topics = set()
                _prev_topic_dicts = []
                _prev_entity_dicts = []
                if prev_ctx:
                    _prev_ents = set(prev_ctx.get("entities", []) or [])
                    _prev_topics = set(prev_ctx.get("topics", []) or [])
                    _prev_topic_dicts = prev_ctx.get("topic_dicts") or []
                    _prev_entity_dicts = prev_ctx.get("entity_dicts") or []
                else:
                    # Fall back to temporal link (covers bench catchup path)
                    _entry = tcmm.get_archive_entry(aid)
                    _prev_aid = (_entry or {}).get("temporal", {}).get("prev_aid")
                    if _prev_aid is not None and _prev_aid != aid:
                        _prev_node = tcmm.get_archive_entry(_prev_aid)
                        if _prev_node:
                            _prev_ents = set(_prev_node.get("entities", []) or [])
                            _prev_topics = set(_prev_node.get("topics", []) or [])
                            _prev_topic_dicts = _prev_node.get("topic_dicts") or []
                            _prev_entity_dicts = _prev_node.get("entity_dicts") or []

                if _prev_ents or _prev_topics:
                    curr_ents = set(entities)
                    curr_topics = set(topics)
                    continued_ents = list(curr_ents & _prev_ents)
                    continued_topics = list(curr_topics & _prev_topics)
                    introduced_ents = list(curr_ents - _prev_ents)
                    introduced_topics = list(curr_topics - _prev_topics)
                    is_break = False
                    if (curr_ents and not continued_ents) and (curr_topics and not continued_topics):
                        is_break = True
                    semantic_out["flow_from_prev"] = {
                        "continued_topics": continued_topics,
                        "continued_entities": continued_ents,
                        "introduced_topics": introduced_topics,
                        "introduced_entities": introduced_ents,
                        "break": is_break
                    }

                    # No naive inheritance — GLiNER hint labels handle both
                    # topic and entity continuations from previous node.

                data = {
                    "topics": topics,
                    "topic_dicts": topic_dicts,
                    "entities": entities,
                    "entity_dicts": entity_dicts,
                    "claims": claims,
                    "claim_scores": claim_scores,
                    "semantic": semantic_out
                }

                density_score = (
                    len(claims) * 1.0
                    + len(entities) * 0.7
                    + len(topics) * 0.5
                    + identifier_count * 1.5
                    + structure_score * 0.5
                )
                MAX_TAIL = 300
                tail_text = combined_text[-MAX_TAIL:]
                semantic_out["text_tail"] = tail_text
                
                entry = tcmm.get_archive_entry(aid)
                if entry:
                    entry["density_score"] = float(density_score)
                    # Episodic classification: single writer of block_class.
                    # block_class values now follow the flat episodic ontology
                    # (FACT / DECISION / INSIGHT / ... / CHATTER / ACK ...).
                    # recallable is derived: non-recallable classes gate
                    # recall scoring; 'unclassified' is safe-default recallable=True.
                    try:
                        _ep_cls, _ep_rec = nlp_adapter.classify_episodic_recallable(combined_text)
                    except Exception as _ep_e:
                        log_exception("classify_episodic_recallable failed", _ep_e)
                        _ep_cls, _ep_rec = "unclassified", True
                    entry["block_class"] = _ep_cls
                    entry["recallable"] = _ep_rec
                    entry["isDirty"] = True
                    tcmm.archive[aid] = entry  # Write back to provider

                log_semantic(f"DENSITY aid={aid} score={density_score:.2f} claims={len(claims)} entities={len(entities)}")

                if density_score < CFG.MIN_DENSITY_SCORE:
                    tcmm._metrics["semantic.weak_nodes"] += 1
                    if entry:
                        entry["semantic_text"] = re.sub(r'^\[[\w_]+\|[^\]]*\]\s*', '', text_str)
                        entry["isDirty"] = True
                        tcmm.archive[aid] = entry  # Write back
                    tcmm.attach_semantic_overlay(aid, {"semantic": {"text_tail": tail_text}})
                    tcmm.reindex_archive_entry(aid) 
                    if tcmm.semantic_embed_queue.qsize() < getattr(tcmm, "MAX_SEMANTIC_QUEUE", 128):
                         tcmm.semantic_embed_queue.put((aid, text_str))
                    tcmm._metrics["semantic.skipped"] += 1
                    summary.increment_counter("semantic_worker", "skipped")
                    continue

                if aid not in tcmm.archive:
                    tcmm._metrics["semantic.skipped"] += 1
                    summary.increment_counter("semantic_worker", "skipped")
                    continue

                # Semantic text: grounded in raw text, not derived noise
                # Strip block header (legacy data with headers in text; no-op for clean text)
                _clean_text = re.sub(r'^\[[\w_]+\|[^\]]*\]\s*', '', text_str)
                _sents = [s.strip() for s in re.split(r'[.!?]\s+', _clean_text) if s.strip()]
                _raw_summary = '. '.join(_sents[:2])
                if len(_raw_summary) > 400:
                    _raw_summary = _raw_summary[:400]
                _ent_suffix = f" [{', '.join(entities[:6])}]" if entities else ""
                sem_text = _raw_summary + _ent_suffix
                if not sem_text.strip():
                    sem_text = text_str

                if entry:
                    entry["semantic_text"] = sem_text
                    entry["isDirty"] = True
                    tcmm.archive[aid] = entry  # Write back to provider
                    if sem_text.strip():
                         try:
                            max_q = getattr(tcmm, "MAX_SEMANTIC_QUEUE", 128)
                            if tcmm.semantic_embed_queue.qsize() < max_q:
                                tcmm.semantic_embed_queue.put((aid, sem_text))
                                tcmm.semantic_embed_wakeup.set()
                            else:
                                _log_warn(f"Semantic Embed Queue Full, dropping aid={aid}")
                         except Exception as e:
                            log_exception("Failed to enqueue semantic embedding job", e)

                def _clean_claims(xs):
                    return [re.sub(r"\s*\[Block id=\d+\]\s*", "", str(x)) for x in xs]

                if not claims:
                     tcmm.attach_semantic_overlay(aid, data)
                     tcmm._metrics["semantic.skipped"] += 1
                     continue

                # Bulk ingest mode: skip per-node link computation, defer to batch pass
                if os.environ.get("TCMM_BULK_INGEST") == "1":
                    data["semantic"]["text_tail"] = tail_text
                    tcmm.attach_semantic_overlay(aid, data)
                    tcmm.reindex_archive_entry(aid)
                    tcmm._metrics["semantic.success"] += 1
                    summary.increment_counter("semantic_worker", "processed")
                    item_elapsed = time.time() - item_start
                    # Suppressed per-item metric — tracked via semantic.success counter
                    continue

                # Phase 17: Global Semantic Linking
                links = {}
                cleaned_curr_claims = _clean_claims(claims)

                # Use user-scoped vector search for cross-namespace link candidates
                # Embedding lookup: DB providers (LanceDB) strip embeddings from the
                # JSON blob into a separate vector column. Use the embedding proxy,
                # which delegates to provider.get_embedding() — works for both.
                curr_emb = None
                try:
                    if hasattr(tcmm, "archive_embeddings"):
                        curr_emb = tcmm.archive_embeddings.get(aid)
                except Exception:
                    curr_emb = None
                if curr_emb is None and entry:
                    curr_emb = entry.get("embedding")
                id_to_vector_score = {}
                if curr_emb is not None:
                    # Search across all namespaces for this user (cross-session linking)
                    if hasattr(tcmm.archive_vector_index, 'search_user'):
                        all_ids, all_scores = tcmm.archive_vector_index.search_user(curr_emb, k=CFG.GLOBAL_LINK_CANDIDATES+1)
                    else:
                        all_ids, all_scores = tcmm.archive_vector_index.search(curr_emb, k=CFG.GLOBAL_LINK_CANDIDATES+1)
                    id_to_vector_score = {i: s for i, s in zip(all_ids, all_scores) if i != aid}
                    candidate_ids = list(id_to_vector_score.keys())
                else:
                    # Fallback to temporal locality if no embedding
                    LOCAL_RANGE = 50
                    candidate_ids = sorted([k for k in tcmm.archive.keys() if k != aid], reverse=True)[:LOCAL_RANGE]
                
                scored_candidates = []
                
                for other_id in candidate_ids:
                    # Use user-scoped lookup for cross-namespace candidates
                    other_entry = tcmm.archive.get_user(other_id) if hasattr(tcmm.archive, 'get_user') else tcmm.archive.get(other_id)
                    if not other_entry: continue
                    other_claims = other_entry.get("claims", [])
                    other_topics = other_entry.get("topics", [])
                    other_entities = other_entry.get("entities", [])
                    
                    if not other_claims and not other_topics and not other_entities:
                        continue
                    
                    ent_overlap = set(entities) & set(other_entities)
                    topic_overlap = set(topics) & set(other_topics)

                    # Additive hybrid scoring: vector + overlap signals
                    weight = 0.0

                    # Vector similarity — primary signal, always contributes
                    v_score = id_to_vector_score.get(other_id, 0.0)
                    if v_score > 0.3:
                        weight += v_score * 0.6  # Scale: 0.3→0.18, 0.7→0.42, 0.9→0.54

                    # Entity overlap bonus
                    if ent_overlap:
                        weight += min(0.3, 0.15 * len(ent_overlap))

                    # Topic overlap bonus
                    if topic_overlap:
                        weight += min(0.25, 0.12 * len(topic_overlap))

                    # Claim overlap (exact match — strongest signal)
                    if other_claims:
                        cleaned_other_claims = _clean_claims(other_claims)
                        claim_overlap = set(cleaned_curr_claims) & set(cleaned_other_claims)
                        if claim_overlap:
                            jaccard = len(claim_overlap) / max(1, len(set(cleaned_curr_claims) | set(cleaned_other_claims)))
                            weight += 0.4 * jaccard

                    final_weight = min(1.0, float(weight))
                    if final_weight >= CFG.SEMANTIC_LINK_THRESHOLD:
                         scored_candidates.append((other_id, final_weight))
                
                # Keep top-K links
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                for other_id, weight in scored_candidates[:CFG.MAX_LINKS_PER_NODE]:
                    links[other_id] = weight
                
                if len(links) == 0 and len(claims) < 3:
                     tcmm._metrics["semantic.weak_nodes"] += 1
                     data["semantic"]["text_tail"] = tail_text
                     tcmm.attach_semantic_overlay(aid, data)
                     tcmm._metrics["semantic.skipped"] += 1
                     summary.increment_counter("semantic_worker", "skipped")
                     continue

                # Phase 18: Entity Links — pure entity-overlap graph
                # Builds entity_links from candidates that share entities.
                # Separate from semantic_links so recall_graph can traverse
                # the entity graph independently (single-hop, query-aware).
                entity_links = {}
                if entities:
                    curr_ent_set = set(e.lower() for e in entities)
                    entity_scored = []
                    for other_id in candidate_ids:
                        other_entry = tcmm.archive.get_user(other_id) if hasattr(tcmm.archive, 'get_user') else tcmm.archive.get(other_id)
                        if not other_entry:
                            continue
                        other_ents = other_entry.get("entities", [])
                        if not other_ents:
                            continue
                        other_ent_set = set(e.lower() for e in other_ents)
                        overlap = curr_ent_set & other_ent_set
                        if not overlap:
                            continue
                        # Weight by overlap count, capped at 1.0
                        w = min(1.0, 0.2 * len(overlap))
                        entity_scored.append((other_id, w))
                    entity_scored.sort(key=lambda x: x[1], reverse=True)
                    for oid, w in entity_scored[:CFG.MAX_LINKS_PER_NODE]:
                        entity_links[oid] = w

                if entry:
                    entry["semantic_links"] = links
                    entry["entity_links"] = entity_links
                    entry["isDirty"] = True
                    tcmm.archive[aid] = entry  # Write back to provider

                if not links:
                    tcmm._metrics["semantic.zero_link_nodes"] += 1

                data = {
                    "topics": topics,
                    "entities": entities,
                    "entity_dicts": entity_dicts,
                    "claims": claims,
                    "semantic": semantic_out,
                    "semantic_links": links
                }
                if claims:
                    semantic_out["last_claim_text"] = claims[-1]
                tcmm.attach_semantic_overlay(aid, data)

                for other_id, weight in links.items():
                    other_node = tcmm.get_archive_entry(other_id)
                    if other_node:
                        other_node.setdefault("semantic_links", {})[aid] = weight
                        other_node["isDirty"] = True
                        # Only write back if node belongs to current namespace
                        # Cross-namespace nodes are read-only during linking
                        if other_id in tcmm.archive:
                            tcmm.archive[other_id] = other_node

                # Bidirectional entity links
                for other_id, weight in entity_links.items():
                    other_node = tcmm.get_archive_entry(other_id)
                    if other_node:
                        other_node.setdefault("entity_links", {})[aid] = weight
                        other_node["isDirty"] = True
                        # Only write back if node belongs to current namespace
                        # Cross-namespace nodes are read-only during linking
                        if other_id in tcmm.archive:
                            tcmm.archive[other_id] = other_node

                # Phase 19: Topic Links — embedding similarity graph
                topic_links = {}
                topic_emb = None
                if topics:
                    topic_str = ", ".join(topics[:5])
                    try:
                        topic_emb = tcmm.embedder.embed(topic_str)
                    except Exception:
                        topic_emb = None
                if topic_emb is not None:
                    # Re-read entry in case attach_semantic_overlay updated it
                    entry = tcmm.get_archive_entry(aid)
                    if entry:
                        entry["topic_embedding"] = topic_emb
                    tcmm.topic_embeddings[aid] = topic_emb
                    topic_scored = []
                    for other_id in candidate_ids:
                        other_topic_emb = tcmm.topic_embeddings.get(other_id)
                        if other_topic_emb is None:
                            other_entry = tcmm.archive.get_user(other_id) if hasattr(tcmm.archive, 'get_user') else tcmm.archive.get(other_id)
                            other_topic_emb = (other_entry or {}).get("topic_embedding")
                        if other_topic_emb is None:
                            continue
                        sim = _cosine(topic_emb, other_topic_emb)
                        if sim >= 0.70:
                            topic_scored.append((other_id, float(sim)))
                    topic_scored.sort(key=lambda x: x[1], reverse=True)
                    for oid, w in topic_scored[:CFG.MAX_LINKS_PER_NODE]:
                        topic_links[oid] = w

                if entry:
                    entry["topic_links"] = topic_links
                    tcmm.archive[aid] = entry  # Write back with topic links

                # Bidirectional topic links
                for other_id, weight in topic_links.items():
                    other_node = tcmm.get_archive_entry(other_id)
                    if other_node:
                        other_node.setdefault("topic_links", {})[aid] = weight
                        other_node["isDirty"] = True
                        # Only write back if node belongs to current namespace
                        # Cross-namespace nodes are read-only during linking
                        if other_id in tcmm.archive:
                            tcmm.archive[other_id] = other_node

                tcmm.reindex_archive_entry(aid)
                tcmm._metrics["semantic.success"] += 1
                summary.increment_counter("semantic_worker", "processed")

                item_elapsed = time.time() - item_start
                # Suppressed per-item metric — tracked via semantic.success counter

        except Exception as e:
            tcmm._metrics["semantic.error"] += 1
            summary.increment_counter("semantic_worker", "failed")
            import traceback as _tb
            tcmm._metrics["semantic.last_error"] = f"{e}\n{''.join(_tb.format_exception(type(e), e, e.__traceback__))}"
            log_exception("Semantic Worker Error", e)
        finally:
            # Checkpoint: persist all nodes touched by this batch
            if batch_data:
                try:
                    dirty_aids = set()
                    for bd in batch_data:
                        dirty_aids.add(bd["aid"])
                    # Also persist bidirectional link partners
                    for bd in batch_data:
                        _e = tcmm.get_archive_entry(bd["aid"])
                        if _e:
                            for partner_id in list(_e.get("semantic_links", {}).keys()) + \
                                               list(_e.get("entity_links", {}).keys()) + \
                                               list(_e.get("topic_links", {}).keys()):
                                dirty_aids.add(partner_id)
                    tcmm.archive.persist(list(dirty_aids))
                except Exception as pe:
                    log_exception("SemanticWorker persist failed", pe)

            # Phase 15 Robustness: Mark every item in the batch as done exactly once
            for _ in batch_items:
                tcmm.semantic_queue.task_done()


# ============================================================
# Semantic embedding worker (batched re-index)
# ============================================================

def semantic_embedding_worker_loop(tcmm):
    """
    Consumes tcmm.semantic_embed_queue
    Phase 109: Batch Semantic Re-indexing
    Optimized Flush Logic (Phase 135)
    """
    batch = []
    batch_start_time = None

    while True:
        try:
            # Calculate remaining time
            if batch:
                elapsed = time.time() - batch_start_time
                timeout = max(0, CFG.EMBED_BATCH_MAX_WAIT - elapsed)
            else:
                timeout = CFG.EMBED_BATCH_MAX_WAIT

            try:
                # Poll queue with calculated timeout
                if batch:
                     item = tcmm.semantic_embed_queue.get(timeout=timeout)
                else:
                     # Check wakeup signal first if empty (pseudo-priority)
                     if tcmm.semantic_embed_wakeup.is_set():
                         tcmm.semantic_embed_wakeup.clear()
                         item = None 
                         raise queue.Empty 
                     
                     item = tcmm.semantic_embed_queue.get(timeout=timeout)

                if item is None:
                    break
                
                if not batch:
                     batch_start_time = time.time()
                batch.append(item)
                
                # Drain more if available immediately
                while len(batch) < CFG.EMBED_BATCH_SIZE:
                    try:
                        extra = tcmm.semantic_embed_queue.get_nowait()
                        batch.append(extra)
                    except queue.Empty:
                        break

            except queue.Empty:
                pass  # Timeout expired

            now = time.time()

            is_full = len(batch) >= CFG.EMBED_BATCH_SIZE
            is_timeout = batch and (now - batch_start_time) >= CFG.EMBED_BATCH_MAX_WAIT

            if is_full or is_timeout:
                aids = []
                texts = []

                for aid, text in batch:
                    if not text:
                        continue
                    aids.append(aid)
                    texts.append(text)

                # Ensure adapter availability
                if not aids or not tcmm.embedder or not hasattr(tcmm.embedder, "embed_batch"):
                    # Cleanup and continue
                    for _ in batch:
                        try:
                            tcmm.semantic_embed_queue.task_done()
                        except Exception: pass
                    batch = []
                    batch_start_time = None
                    continue

                try:
                    flush_started = time.time()
                    reason = "size" if len(batch) >= CFG.EMBED_BATCH_SIZE else "timeout"
                    accepted = len(texts)
                    queue_size = tcmm.semantic_embed_queue.qsize()

                    # Separation: items needing embedding vs items already embedded
                    to_embed_aids = []
                    to_embed_texts = []
                    ready_embeddings = {} # aid -> emb
                    
                    valid_items = zip(aids, texts)
                    for aid, txt in valid_items:
                         entry = tcmm.archive.get(aid)
                         if entry and entry.get("embedding") is not None:
                             ready_embeddings[aid] = entry["embedding"]
                         else:
                             to_embed_aids.append(aid)
                             to_embed_texts.append(txt)
                    
                    if len(to_embed_aids) < len(texts):
                         # Skipped some - using imported log_perf
                         log_perf(f"Skipped {len(texts)-len(to_embed_aids)} pre-calculated embeddings")

                    # 1. Process new embeddings
                    if to_embed_aids:
                        if tcmm.embedder:
                             try:
                                 new_embs = tcmm.embedder.embed_batch(to_embed_texts)
                                 if len(new_embs) != len(to_embed_aids):
                                     _log_warn(f"[SemanticEmbedWorker] mismatch {len(new_embs)} vs {len(to_embed_aids)}")
                                 
                                 for sub_aid, sub_emb in zip(to_embed_aids, new_embs):
                                     ready_embeddings[sub_aid] = sub_emb
                             except Exception as e:
                                 log_exception("[SemanticEmbedWorker] Batch embed failed", e)
                    
                    # 2. Add to index
                    written = 0
                    for aid, emb in ready_embeddings.items():
                        if emb is None: continue
                        entry = tcmm.get_archive_entry(aid)
                        if entry:
                            entry["embedding"] = emb
                            entry["isDirty"] = True
                            tcmm.archive[aid] = entry  # Write back to provider
                        tcmm.archive_embeddings[aid] = emb
                        if tcmm.archive_vector_index:
                            try:
                                tcmm.archive_vector_index.add(aid, emb)
                                written += 1
                            except Exception as e:
                                log_exception(f"[SemanticEmbedWorker] Index add fail", e)

                    elapsed_emb = time.time() - flush_started
                    print(f">>> [METRIC] Embedding batch of {len(aids)} nodes took {elapsed_emb:.3f}s ({(elapsed_emb/len(aids)):.3f}s per node)", flush=True)

                    # Log
                    queue_wait_ms = (flush_started - batch_start_time) * 1000.0 if batch_start_time else 0.0
                    
                    _log_batch(
                         tcmm,
                         worker="SemanticEmbedWorker",
                         reason=reason,
                         batch_size=len(batch),
                         queue_size=queue_size,
                         accepted=len(texts),
                         queue_wait_ms=queue_wait_ms,
                         written=written
                    )
                    summary.increment_counter("embedding_worker", "batched", len(batch))
                    summary.increment_counter("semantic_worker", "processed", len(batch))

                    # Checkpoint: persist re-embedded nodes
                    if aids:
                        try:
                            tcmm.archive.persist(aids)
                        except Exception as pe:
                            log_exception("SemanticEmbedWorker persist failed", pe)

                except Exception as e:
                    log_exception("SemanticEmbedWorker batch processing failed", e)

                finally:
                    for _ in batch:
                        try:
                            tcmm.semantic_embed_queue.task_done()
                        except Exception: pass
                    batch.clear()
                    batch_start_time = None

        except Exception as e:
            log_exception("Semantic embedding worker main loop failure", e)

# ============================================================
# Archive queue worker (legacy cleanup path)
# ============================================================

def archive_worker_loop(tcmm):
    """
    Consumes tcmm._archive_queue
    Queue item: List[Block]
    """
    while True:
        blocks = tcmm._archive_queue.get()
        if blocks is None:
            tcmm._archive_queue.task_done()
            break
        try:
            # Phase 130: Filter Non-Persistent Blocks
            # We explicitly prevent de-duplicated blocks from being processed for archival
            persistent_blocks = [b for b in blocks if not getattr(b, "non_persistent", False)]

            if persistent_blocks:
                tcmm._compress_and_archive(persistent_blocks)
        except Exception as e:
            log_exception("TCMM-ARCHIVE worker error", e)
        finally:
            tcmm._archive_queue.task_done()


# ============================================================
# Live embedding worker (hot tier)
# ============================================================

def live_embed_worker_loop(tcmm):
    """
    Consumes tcmm._live_embed_queue
    Strict Batched Implementation (Phase 108)
    """
    batch = []
    last_flush = time.time()
    
    should_exit = False
    
    while True:
        # Check queue existence
        if not hasattr(tcmm, "_live_embed_queue"):
            time.sleep(0.05)
            continue
            
        try:
            now = time.time()
            if batch:
                elapsed = now - last_flush
                timeout = max(0, CFG.EMBED_BATCH_MAX_WAIT - elapsed)
            else:
                timeout = 1.0 # Long wait if empty

            item = tcmm._live_embed_queue.get(timeout=timeout)
            
            if item is None:
                 tcmm._live_embed_queue.task_done()
                 should_exit = True
            else:
                 if not batch:
                     last_flush = time.time() # Start timer
                 batch.append(item)
            
        except queue.Empty:
            pass
        except Exception:
            pass

        # Check flush
        now = time.time()
        is_full = len(batch) >= CFG.EMBED_BATCH_SIZE
        is_timeout = (now - last_flush) >= CFG.EMBED_BATCH_MAX_WAIT
        
        if batch and (is_full or is_timeout or should_exit):
            # Phase 110: Opportunistic Drain
            while len(batch) < CFG.EMBED_BATCH_SIZE:
                try:
                    extra = tcmm._live_embed_queue.get_nowait()
                    if extra is None:
                        should_exit = True
                        break
                    batch.append(extra)
                except queue.Empty:
                    break
                except Exception:
                    break

            block_ids = []
            texts = []
            
            for (bid, txt) in batch:
                if txt and txt.strip():
                    block_ids.append(bid)
                    texts.append(txt)
            
            if block_ids:
                flush_started = time.time()
                reason = "size" if len(batch) >= CFG.EMBED_BATCH_SIZE else "timeout"
                if should_exit: reason = "exit"
                accepted = len(texts)
                queue_size = tcmm._live_embed_queue.qsize()

                # ---- STRICT batch-only contract ----
                if not tcmm.embedder:
                    pass
                elif not hasattr(tcmm.embedder, "embed_batch"):
                    log_exception(
                        "[LiveEmbedWorker] adapter missing embed_batch()",
                        RuntimeError(type(tcmm.embedder).__name__)
                    )
                else:
                    try:
                        embed_start = time.time()
                        embeddings = tcmm.embedder.embed_batch(texts)
                        embed_ms = (time.time() - embed_start) * 1000.0

                        if len(embeddings) != len(block_ids):
                            _log_warn(
                                "[LiveEmbedWorker] embed_batch returned "
                                f"{len(embeddings)} vectors for {len(block_ids)} inputs"
                            )

                        written = 0
                        for bid, emb in zip(block_ids, embeddings):
                            if emb is None:
                                _log_warn(f"[LiveEmbedWorker] empty embedding for block={bid}")
                                continue

                            tcmm.live_embedding_cache[bid] = emb
                            written += 1
                        
                        waited_ms = (flush_started - last_flush) * 1000.0
                        _log_batch(
                            tcmm,
                            worker="LiveEmbedWorker",
                            reason=reason,
                            batch_size=len(batch),
                            queue_size=queue_size,
                            accepted=accepted,
                            waited_ms=waited_ms,
                            embed_ms=embed_ms,
                            written=written
                        )

                    except Exception as e:
                        log_exception(
                            "LiveEmbedWorker batch failure (embed_batch)",
                            e
                        )
            
            # Cleanup
            for _ in batch:
                 tcmm._live_embed_queue.task_done()
            batch = []
            last_flush = time.time()
        
        if should_exit and not batch:
             break

# ============================================================
# Persistence Worker (DEPRECATED)
# ============================================================
# Replaced by checkpoint-based persistence: each worker and archive
# operation calls archive.persist(aids) at its natural completion point.
# The functions below are kept for backward compatibility but are no
# longer started as background threads.

def _strip_embedding_for_persistence(doc):
    # PATCH 4 — Prevent embedding persistence in archive.json
    needs_copy = "embedding" in doc or "topic_embedding" in doc
    if needs_copy:
        doc = dict(doc)
        doc.pop("embedding", None)
        doc.pop("topic_embedding", None)
    return doc

def persistence_worker_loop(tcmm):
    """
    Scans for dirty archive nodes and persists them.
    DB-backed providers (SQLite, LanceDB) handle writes immediately --
    this worker just triggers periodic compaction/flush.
    Falls back to TinyDB for local (in-memory) storage.
    """
    # Check if provider handles its own persistence.
    # DB providers (SQLite, LanceDB, etc.) write immediately on __setitem__ --
    # they don't need TinyDB dirty-node scanning. Check the self_persisting flag.
    if getattr(tcmm.archive, "self_persisting", False):
        # DB providers handle writes immediately -- just flush/compact periodically
        while not getattr(tcmm, "_stop_persistence", False):
            try:
                time.sleep(5.0)
                tcmm.archive.flush_dirty()
            except Exception as e:
                log_exception("Persistence Worker Error (DB provider)", e)
                time.sleep(5.0)
        return

    if TinyDB is None:
        _log_warn("TinyDB not installed, persistence disabled.")
        return

    # Ensure directory (Fix: Use root data/archive.json)
    db_dir = tcmm._data_dir
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "archive.json")
    emb_db_path = os.path.join(db_dir, "archive_embeddings.json")
    topic_emb_db_path = os.path.join(db_dir, "topic_embeddings.json")

    try:
        db = TinyDB(db_path)
        emb_db = TinyDB(emb_db_path)
        topic_emb_db = TinyDB(topic_emb_db_path)
        tcmm.db = db # Expose for debugging
        tcmm.emb_db = emb_db
        tcmm.topic_emb_db = topic_emb_db
    except Exception as e:
        log_exception("Failed to initialize TinyDB", e)
        return

    ArchiveNode = Query()
    
    loop_count = 0
    while not getattr(tcmm, "_stop_persistence", False):
        try:
            time.sleep(1.0) # Periodically scan
            loop_count += 1
            
            # Phase 0: Tier Migration (Cooling System) - Run every 10s
            # This updates node['tier'] and sets isDirty=True if changed
            run_migration = (loop_count % 10 == 0)

            # Phase 1: Scan Archive Nodes
            # ShardedArchive supports .items()
            # We iterate a snapshot or direct? Direct is okay for dirty check.
            # Using list(items) to avoid runtime change error if iterating?
            # ShardedArchive might change size. Safer to iterate keys or handle concurrently.
            # But here we just want to save dirty ones.
            
            # Optimization: Maintain a dirty set? 
            # User requirement: "archiveworker, just scans the lists of nodes or blocks for dirty blocks"
            
            # We'll iterate keys first to be safe
            keys = list(tcmm.archive.keys())
            
            count = 0
            for aid in keys:
                # Apply Tier Migration
                if run_migration:
                    if hasattr(tcmm, "_migrate_tier"):
                        tcmm._migrate_tier(aid)

                node = tcmm.archive.get(aid)
                if not node: continue
                
                if node.get("isDirty"):
                    # Persist
                    # We remove 'isDirty' before saving? Or keep it?
                    # Remove it from the dict we save, but keep in memory as False.
                    # Actually, TinyDB saves the dict.
                    
                    # Create clean copy for DB
                    to_save = node.copy()
                    if "isDirty" in to_save:
                        del to_save["isDirty"]
                        
                    # Handle EntropyDiagnostics (dataclass)
                    import dataclasses
                    if "entropy_static" in to_save and dataclasses.is_dataclass(to_save["entropy_static"]):
                        to_save["entropy_static"] = dataclasses.asdict(to_save["entropy_static"])
                    
                    # Handle unserializable types if any (embedding might be numpy array?)
                    # Node["embedding"] might be list or None.
                    # We extract it to completely separate it from the main JSON metadata lookup.
                    embedding_data = None
                    if "embedding" in to_save:
                        embedding_data = to_save.pop("embedding")
                    elif tcmm.archive_embeddings.get(aid) is not None:
                        # Fallback to in-mem store if stripped
                        embedding_data = tcmm.archive_embeddings[aid]
                        
                    if embedding_data is not None:
                        # Convert ndarray to list if needed
                        if hasattr(embedding_data, "tolist"):
                            embedding_data = embedding_data.tolist()

                        emb_doc = {"embedding": embedding_data}
                        if emb_db.get(doc_id=aid) is not None:
                             emb_db.update(emb_doc, doc_ids=[aid])
                        else:
                             from tinydb.table import Document
                             emb_db.insert(Document(emb_doc, doc_id=aid))

                    # Topic embedding sidecar (same pattern)
                    topic_emb_data = None
                    if "topic_embedding" in to_save:
                        topic_emb_data = to_save.pop("topic_embedding")
                    elif tcmm.topic_embeddings.get(aid) is not None:
                        topic_emb_data = tcmm.topic_embeddings[aid]

                    if topic_emb_data is not None:
                        if hasattr(topic_emb_data, "tolist"):
                            topic_emb_data = topic_emb_data.tolist()
                        temb_doc = {"topic_embedding": topic_emb_data}
                        if topic_emb_db.get(doc_id=aid) is not None:
                            topic_emb_db.update(temb_doc, doc_ids=[aid])
                        else:
                            from tinydb.table import Document
                            topic_emb_db.insert(Document(temb_doc, doc_id=aid))

                    # Fix: Enforce storage key (doc_id) matches archive ID (aid)
                    # This prevents the off-by-one error where doc_id = aid + 1
                    # Use db.get(doc_id=...) explicitly for compatibility
                    if db.get(doc_id=aid) is not None:
                         save_doc = _strip_embedding_for_persistence(to_save)
                         db.update(save_doc, doc_ids=[aid])
                    else:
                         from tinydb.table import Document
                         # TCMM FIX: enforce invariant before persistence
                         if aid < 1:
                             raise ValueError(f"Refusing to persist invalid archive ID: {aid}")
                         
                         save_doc = _strip_embedding_for_persistence(to_save)
                         db.insert(Document(save_doc, doc_id=aid))
                    
                    # Mark clean
                    node["isDirty"] = False
                    count += 1
            
            if count > 0:
                summary.increment_counter("persistence", "saved", count)
                # log_perf(f"[Persistence] Saved {count} dirty nodes")
                
        except Exception as e:
            log_exception("Persistence Worker Error", e)
            time.sleep(5.0)
