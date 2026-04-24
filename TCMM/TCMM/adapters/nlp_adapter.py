import os
import logging
import spacy
import warnings
import torch

# ── Kill "Batches: 1/1" tqdm spam from SentenceTransformer.encode() ──────────
# SentenceTransformer shows progress bars when its logger level ≤ INFO.
# Setting to WARNING silences all encode() progress bars at the source.
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Optimum / ONNX-Runtime is a local-only optimization path. On cloud
# deployments we use VertexNLPAdapter and never touch the ONNX quantizer
# pipeline — optimum split the onnxruntime subpackage into a separate
# wheel (optimum-onnxruntime) in 2.x and a fresh cloud install may not
# have it. Make the import optional so this module stays importable
# even when the ONNX deps are missing; LocalNLPAdapter will raise
# clearly if anything actually tries to use the quantizer.
try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    _ONNX_AVAILABLE = True
except ImportError as _e:
    ORTModelForFeatureExtraction = None  # type: ignore[assignment]
    ORTQuantizer = None  # type: ignore[assignment]
    AutoQuantizationConfig = None  # type: ignore[assignment]
    _ONNX_AVAILABLE = False
    _ONNX_IMPORT_ERROR = _e
    logging.getLogger(__name__).info(
        "optimum.onnxruntime not installed — LocalNLPAdapter ONNX quantizer "
        "path unavailable. This is expected on cloud (vertex) deployments."
    )

# Safe parallel configuration for transformer pipelines on Windows
CPU_COUNT = os.cpu_count()
DEFAULT_BATCH_SIZE = 32 # Larger batches = better SpaCy throughput

os.environ["OMP_NUM_THREADS"] = str(CPU_COUNT)
os.environ["MKL_NUM_THREADS"] = str(CPU_COUNT)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_COUNT)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Patch 1: PyTorch Thread Control - set to half cores to avoid over-subscription with BLAS
torch.set_num_threads(max(1, CPU_COUNT // 2))
torch.set_num_interop_threads(max(1, CPU_COUNT // 4))

# Suppress known warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Episodic classification ontology (Gemma-local) ──────────────────────
# Duplicated intentionally in VertexNLPAdapter so each adapter can tune
# its own prompt for its target model without a shared module forcing
# lockstep edits. Ontology labels must stay in sync by convention — the
# gate in recall_scoring.py reads the derived recallable bool, not the
# class, so drift only hurts downstream score boosts and audit consistency.
EPISODIC_CLASSES = (
    "FACT", "DECISION", "INSIGHT", "PROCEDURE", "STATE", "INTENT",
    "DERIVED_FACT", "ARTIFACT",
    "CHATTER", "ACK", "QUERY", "TRANSIENT_DATA", "EXECUTION_LOG",
)
NON_RECALLABLE_CLASSES = frozenset({
    "CHATTER", "ACK", "QUERY", "TRANSIENT_DATA", "EXECUTION_LOG",
})
_EPISODIC_CLASS_SET = frozenset(EPISODIC_CLASSES)

class LocalNLPAdapter:
    """
    NLP Adapter for TCMM. Uses Gemma (cloud Ollama) for entity + topic
    extraction. Keeps SpaCy for sentence splitting and dependency parse
    only — no local NER, no GLiNER, no KeyBERT.

    History: GLiNER + KeyBERT were the original pipeline but were
    replaced by Gemma because (a) Gemma produces richer topics and
    semantic entities and (b) GLiNER missed lowercase common nouns
    like "train", "bus", "taxi" which broke numeric fact grouping.
    """

    def __init__(self, spacy_model="en_core_web_sm", keybert_model="sentence-transformers/all-mpnet-base-v2", shared_st_model=None):
        # Gemma is always the NLP path now. The flag is kept for
        # backwards compat but has no effect — the old non-Gemma
        # branches have been removed.
        self._gemma_mode = True

        # 1. Load optimized SpaCy (sentence splitting + dependency parse — needed in all modes)
        print(f"[LocalNLPAdapter] Loading SpaCy model: {spacy_model} (Parallel threads: {CPU_COUNT})")
        try:
            self.nlp = spacy.load(
                spacy_model,
                disable=["lemmatizer", "ner"]
            )
            self.nlp.max_length = 2_000_000
        except OSError:
            print(f"[LocalNLPAdapter] Downloading SpaCy model: {spacy_model}")
            from spacy.cli import download
            download(spacy_model)
            self.nlp = spacy.load(
                spacy_model,
                disable=["lemmatizer", "ner"]
            )
            self.nlp.max_length = 2_000_000

        # GLiNER removed — Gemma handles entity extraction via
        # process_batch_gemma / _gemma_extract / _parse_gemma_output.
        self.gliner_model = None

        # KeyBERT + claim-quality scorer removed along with GLiNER.
        # Gemma now handles topic extraction inside _gemma_extract and
        # claim filtering is done by SpaCy sentence-split plus downstream
        # dream-engine reasoning. These fields remain so legacy code that
        # peeks at them doesn't crash.
        self.kw_model = None
        self._claim_fact_center = None
        self._claim_fill_center = None
        self._claim_scorer_model = None
        print(f"[LocalNLPAdapter] Gemma-only NLP (GLiNER/KeyBERT removed)")

    def _init_claim_scorer(self, shared_st_model=None):
        """Pre-compute factual/filler prototype embeddings for claim quality scoring."""
        st_model = shared_st_model
        if st_model is None and self.kw_model is not None:
            # Extract ST model from KeyBERT
            st_model = getattr(self.kw_model, "model", None)
        if st_model is None:
            return

        try:
            from sentence_transformers import util as st_util
            self._st_util = st_util
            self._claim_scorer_model = st_model

            factual_protos = [
                "I own a red car.",
                "I bought a house in Boston.",
                "I subscribe to Netflix.",
                "My cat is named Luna.",
                "I work at Google.",
                "The package arrived yesterday.",
                "I canceled my gym membership.",
                "We moved to Seattle last year.",
            ]
            filler_protos = [
                "That sounds great!",
                "Feel free to let me know.",
                "I appreciate your help.",
                "Certainly!",
                "I think that is wonderful.",
                "Of course, happy to help!",
                "No problem at all.",
                "Sure thing!",
            ]
            fact_embs = st_model.encode(factual_protos, convert_to_tensor=True, show_progress_bar=False)
            fill_embs = st_model.encode(filler_protos, convert_to_tensor=True, show_progress_bar=False)
            self._claim_fact_center = fact_embs.mean(dim=0)
            self._claim_fill_center = fill_embs.mean(dim=0)
            print(f"[LocalNLPAdapter] Claim quality scorer ready (embedding-based)")
        except Exception as e:
            print(f"[LocalNLPAdapter] Claim scorer init failed ({e})")

    def score_claims(self, claims):
        """Score claims as factual (>0) vs filler (<0). Returns list of (claim, score) tuples."""
        if not claims or self._claim_fact_center is None:
            return [(c, 0.0) for c in claims]
        try:
            embs = self._claim_scorer_model.encode(claims, convert_to_tensor=True, show_progress_bar=False)
            fact_scores = self._st_util.cos_sim(embs, self._claim_fact_center.unsqueeze(0)).squeeze(-1)
            fill_scores = self._st_util.cos_sim(embs, self._claim_fill_center.unsqueeze(0)).squeeze(-1)
            diffs = (fact_scores - fill_scores).tolist()
            return list(zip(claims, diffs))
        except Exception:
            return [(c, 0.0) for c in claims]

    def filter_claims(self, claims, threshold=-0.05):
        """Return only claims scoring above threshold (factual > filler)."""
        scored = self.score_claims(claims)
        return [c for c, s in scored if s > threshold]

    def _export_and_quantize(self, model_id, save_path):
        """Helper to export to ONNX and apply INT8 dynamic quantization."""
        print(f"[LocalNLPAdapter] Exporting {model_id} to ONNX and quantizing...")
        
        # Temporary path for the FP32 ONNX model
        tmp_onnx_path = save_path + "_tmp"
        
        # 1. Export to FP32 ONNX
        model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
        model.save_pretrained(tmp_onnx_path)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(tmp_onnx_path)

        # 2. Apply Dynamic INT8 Quantization
        quantizer = ORTQuantizer.from_pretrained(tmp_onnx_path)
        q_config = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        
        quantizer.quantize(save_dir=save_path, quantization_config=q_config)
        tokenizer.save_pretrained(save_path)
        
        print(f"[LocalNLPAdapter] Quantized model saved to {save_path}")

    def extract_topics_batch(self, texts):
        """Patch 2: Batch KeyBERT extraction using internal document batching.
        Returns list of lists of (keyword, score) tuples, sorted by score descending.

        In Gemma mode, KeyBERT is not loaded (self.kw_model is None) — in that
        case we return empty topic lists per doc.  Callers merge results with
        blocks[i].get('topics', []) which were already extracted during ingest,
        so no topic information is lost; the dream cycle simply skips adding
        NEW topics it would have discovered via KeyBERT.
        """
        if not texts:
            return []
        if self.kw_model is None:
            return [[] for _ in texts]
        keywords_batch = self.kw_model.extract_keywords(
            docs=texts,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=8
        )
        topics = []
        for kw_list in keywords_batch:
            extracted = []
            for item in kw_list:
                if isinstance(item, tuple) and len(item) >= 2:
                    extracted.append((item[0], float(item[1])))
                elif isinstance(item, tuple):
                    extracted.append((item[0], 0.5))
                elif isinstance(item, str):
                    extracted.append((item, 0.5))
                elif isinstance(item, dict) and "keyword" in item:
                    extracted.append((item["keyword"], float(item.get("score", 0.5))))
            extracted.sort(key=lambda x: x[1], reverse=True)
            topics.append(extracted)
        return topics

    def refine_topics(self, keybert_topics, doc, entities):
        """Refine raw KeyBERT topics using 2-layer pipeline:

        Layer 1: Deduplicate KeyBERT by embedding similarity (>0.8 = same topic)
        Layer 2: Rank by KeyBERT confidence + entity overlap bonus

        Accepts (keyword, score) tuples from extract_topics_batch.
        Returns top 5 refined topics as strings, ordered by confidence.
        """
        # Normalize input: accept both (keyword, score) tuples and plain strings
        scored_input = []
        for item in (keybert_topics or []):
            if isinstance(item, tuple) and len(item) >= 2:
                scored_input.append((item[0], float(item[1])))
            elif isinstance(item, str):
                scored_input.append((item, 0.5))

        if not scored_input:
            return []

        # Layer 1: Deduplicate by embedding similarity
        if len(scored_input) <= 2:
            deduped = list(scored_input)
        else:
            deduped = self._deduplicate_topics(scored_input)

        # Layer 2: Rank by KeyBERT confidence + entity overlap bonus
        entity_lower = {e.lower() for e in entities}
        final_scored = []
        for topic, kb_score in deduped:
            entity_bonus = sum(0.1 for ew in entity_lower if ew in topic.lower())
            length_bonus = 0.05 if len(topic.split()) >= 2 else 0.0
            final_scored.append((topic, kb_score + entity_bonus + length_bonus))

        final_scored.sort(key=lambda x: -x[1])
        return [t for t, s in final_scored[:5]]

    def _deduplicate_topics(self, topics):
        """Cluster topics by embedding similarity, keep one per cluster.
        Accepts (keyword, score) tuples. Returns deduplicated tuples."""
        try:
            st_model = self._claim_scorer_model or getattr(self.kw_model, 'model', None)
            if st_model is None:
                return topics[:5]

            keywords = [t[0] if isinstance(t, tuple) else t for t in topics]
            from sentence_transformers import util as st_util
            embs = st_model.encode(keywords, convert_to_tensor=True, show_progress_bar=False)
            sims = st_util.cos_sim(embs, embs)

            used = set()
            deduped = []
            for i in range(len(topics)):
                if i in used:
                    continue
                deduped.append(topics[i])
                for j in range(i + 1, len(topics)):
                    if j not in used and float(sims[i][j]) > 0.80:
                        used.add(j)
            return deduped
        except Exception:
            return topics[:5]

    def canonicalize_entities(self, entities):
        """Merge near-duplicate entities by embedding similarity.

        "Drain Screen Or Filter" and "Drain Screens And Filters" → one canonical form.
        Keeps the shorter/cleaner form as canonical.
        """
        if len(entities) <= 1:
            return entities
        try:
            st_model = self._claim_scorer_model or getattr(self.kw_model, 'model', None)
            if st_model is None:
                return entities
            from sentence_transformers import util as st_util
            embs = st_model.encode(entities, convert_to_tensor=True, show_progress_bar=False)
            sims = st_util.cos_sim(embs, embs)

            used = set()
            canonical = []
            for i in range(len(entities)):
                if i in used:
                    continue
                # Find all duplicates of this entity
                group = [i]
                for j in range(i + 1, len(entities)):
                    if j not in used and float(sims[i][j]) > 0.85:
                        group.append(j)
                        used.add(j)
                # Pick canonical: prefer shorter name (less noise), then title case
                best = min(group, key=lambda idx: len(entities[idx]))
                canonical.append(entities[best])
            return canonical
        except Exception:
            return entities

    def canonicalize_topics(self, all_topics):
        """Build a global topic canonicalization map from a list of all topics.

        Two-pass merge:
        1. Substring containment — "morning yoga classes" absorbs into "yoga"
           (shorter, more general form wins when it's more frequent or equally common)
        2. Embedding similarity (>0.80) — catches semantic near-duplicates that
           don't share substrings (e.g. "weightlifting" / "strength training")

        Returns dict: raw_topic → canonical_topic.
        """
        if not all_topics:
            return {}
        from collections import Counter
        freq = Counter(t.lower() for t in all_topics)
        unique = list(freq.keys())
        if len(unique) <= 1:
            return {t: t for t in all_topics}

        # ── Pass 1: Substring containment merge ──
        # Sort shortest-first so general forms absorb specific variants
        by_len = sorted(unique, key=len)
        substr_map = {}  # long → short canonical
        absorbed = set()
        for i, short in enumerate(by_len):
            if short in absorbed or len(short) < 4:
                continue
            short_words = set(short.split())
            for j in range(i + 1, len(by_len)):
                long = by_len[j]
                if long in absorbed:
                    continue
                # Short must be a substring of long, or all words of short
                # appear in long (word containment)
                long_words = set(long.split())
                if short in long or (len(short_words) > 0 and short_words <= long_words):
                    # Absorb long into short if short is more frequent or equal
                    # If long is way more frequent (3x+), keep long as canonical
                    if freq[long] >= freq[short] * 3:
                        substr_map[short] = long
                        absorbed.add(short)
                        break  # short got absorbed, stop matching it
                    else:
                        substr_map[long] = short
                        absorbed.add(long)

        # Apply substring map transitively (A→B→C becomes A→C)
        def _resolve(t):
            visited = set()
            while t in substr_map and t not in visited:
                visited.add(t)
                t = substr_map[t]
            return t

        # Rebuild unique list after substring merge
        canon_after_substr = {}
        for t in unique:
            canon_after_substr[t] = _resolve(t)

        # Deduplicated set for embedding pass
        remaining = list(set(canon_after_substr.values()))
        if len(remaining) <= 1:
            result = {}
            for t in all_topics:
                result[t] = canon_after_substr.get(t.lower(), t)
            return result

        # Recalculate freq for merged forms
        merged_freq = Counter()
        for t in unique:
            merged_freq[canon_after_substr[t]] += freq[t]

        # ── Pass 2: Embedding similarity merge ──
        try:
            st_model = self._claim_scorer_model or getattr(self.kw_model, 'model', None)
            if st_model is None:
                result = {}
                for t in all_topics:
                    result[t] = canon_after_substr.get(t.lower(), t)
                return result
            from sentence_transformers import util as st_util
            embs = st_model.encode(remaining, convert_to_tensor=True,
                                   show_progress_bar=False, batch_size=256)
            sims = st_util.cos_sim(embs, embs)

            used = set()
            embed_map = {}  # remaining topic → canonical
            for i in range(len(remaining)):
                if i in used:
                    continue
                group = [i]
                for j in range(i + 1, len(remaining)):
                    if j not in used and float(sims[i][j]) > 0.80:
                        group.append(j)
                        used.add(j)
                # Pick canonical: most frequent merged form, then shortest
                best_idx = max(group, key=lambda idx: (merged_freq[remaining[idx]], -len(remaining[idx])))
                canonical = remaining[best_idx]
                for idx in group:
                    embed_map[remaining[idx]] = canonical

            # Compose both maps: original → substr_canonical → embed_canonical
            result = {}
            for t in all_topics:
                step1 = canon_after_substr.get(t.lower(), t.lower())
                step2 = embed_map.get(step1, step1)
                result[t] = step2
            return result
        except Exception:
            result = {}
            for t in all_topics:
                result[t] = canon_after_substr.get(t.lower(), t)
            return result

    # Pronouns and ungrounded references — must never become entities
    _PRONOUN_KILL = frozenset({
        "it", "them", "they", "he", "she", "him", "her", "its", "their",
        "his", "hers", "us", "we", "me", "i", "you", "your", "my",
        "this", "that", "these", "those", "here", "there",
        "something", "anything", "everything", "nothing",
        "thing", "things", "stuff", "way", "one", "ones",
    })

    # GLiNER max token limit — texts longer than this get chunked
    _GLINER_MAX_CHARS = 1400  # ~384 tokens ≈ ~1400 chars for English

    # Entity confirmation: tracks frequency across blocks. Only confirmed entities
    # (seen >= 2 times) get promoted. First occurrence is held as "candidate".
    _entity_candidates = None  # initialized per-batch session

    @staticmethod
    def _is_valid_entity(name):
        """Hard entity gate. Rejects garbage before it enters the pipeline.

        Structural checks — no word lists, no heuristics.
        """
        import re as _re
        # Too short
        if len(name) < 3:
            return False
        # Must contain at least one alpha character
        if not any(c.isalpha() for c in name):
            return False
        # Pure number or number-word
        if _re.match(r'^[\d,.\s]+$', name):
            return False
        # Hex-like strings (session IDs)
        key = name.lower().replace(' ', '')
        if len(key) >= 6 and all(c in '0123456789abcdef' for c in key):
            return False
        # Single lowercase word under 4 chars — too ambiguous
        if len(name.split()) == 1 and name.islower() and len(name) < 4:
            return False
        return True

    @staticmethod
    def _chunk_text(text, max_chars, overlap=200):
        """Split text into chunks at sentence boundaries with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chars
            if end >= len(text):
                chunks.append(text[start:])
                break
            # Try to break at a sentence boundary (. ! ?)
            boundary = text.rfind('. ', start + max_chars - overlap, end)
            if boundary == -1:
                boundary = text.rfind('? ', start + max_chars - overlap, end)
            if boundary == -1:
                boundary = text.rfind('! ', start + max_chars - overlap, end)
            if boundary > start:
                end = boundary + 1
            chunks.append(text[start:end])
            start = end - overlap  # Overlap to catch entities at boundaries
        return chunks


    def reset_entity_candidates(self):
        """Reset entity candidate tracker between sessions."""
        self._entity_candidates = {}
        self._garbage_sink = []

    def get_garbage_sink(self):
        """Return rejected entities for debugging."""
        return getattr(self, '_garbage_sink', [])

    def normalize_entity(self, text: str) -> str:
        """Standardize entity casing and whitespace."""
        return text.strip().title()

    def split_atomic_claims(self, text: str) -> list[str]:
        """Split complex sentences into atomic claims while maintaining meaning."""
        if not text:
            return []
        
        # Split on common conjunctions and purpose clauses
        # Using a regex that tries to find meaningful breaks
        delimiters = [", enabling ", ", but ", " and ", " while ", " to ", " facilitating "]
        
        parts = [text]
        for d in delimiters:
            new_parts = []
            for p in parts:
                if d in p:
                    split_ps = p.split(d)
                    # Only split if both sides have enough volume (approx 4 words)
                    if len(split_ps[0].split()) >= 4 and len(split_ps[1].split()) >= 4:
                        new_parts.append(split_ps[0])
                        # Contextual prefix for second part if it starts with a verb
                        if split_ps[1].strip().split()[0].endswith(('ing', 'es', 's')):
                            new_parts.append(split_ps[1].strip())
                        else:
                            new_parts.append(split_ps[1].strip())
                    else:
                        new_parts.append(p)
                else:
                    new_parts.append(p)
            parts = new_parts
            
        return [p.strip() for p in parts if len(p.split()) >= 3]

    def process_batch(self, texts, text_offsets=None, topic_hints=None, entity_hints=None, roles=None):
        """Phase 15/16/17: Process a batch of texts using parallel pipeline.

        GLiNER (zero-shot NER) replaces spaCy NER for entity extraction.
        SpaCy still handles sentence splitting. KeyBERT handles topics.

        text_offsets: optional list of int, one per text. Each is the character
        offset where the current block's text starts within the combined text.
        Entities before this offset came from prev_text context and are filtered out.

        topic_hints: optional list (one per text) of lists of topic name strings
        from the previous node. Fed to GLiNER as extra labels so it can detect
        topic continuations in short turns that would otherwise extract nothing.

        entity_hints: optional list (one per text) of lists of entity name strings
        from the previous node. Fed to GLiNER as extra labels for entity continuations.

        roles: optional list of str, one per text. Block origin role ("user",
        "assistant", etc.) from archive metadata. When provided, avoids parsing
        role from text headers — essential for clean (headerless) archive text.
        """
        if not texts:
            return []
        if text_offsets is None:
            text_offsets = [0] * len(texts)

        # Determine block roles and prepare clean texts for NLP processing
        # Split into current-block-only text (for KeyBERT/spaCy) and full combined text (for GLiNER)
        import re as _re
        _HEADER_RE = _re.compile(r'^\[([\w_]+)\|([^|]*)\|(\w+)\]\s*')

        # Use caller-supplied roles from metadata; fall back to header parsing (legacy data)
        if roles is None:
            roles = []
            for t in texts:
                m = _HEADER_RE.match(t)
                roles.append(m.group(3).lower() if m else "unknown")

        clean_texts = []      # combined text (for GLiNER — needs context for better extraction)
        current_texts = []    # current block only (for KeyBERT/spaCy — prevents topic contamination)
        for t, offset in zip(texts, text_offsets):
            # Clean markdown
            t_clean = _re.sub(r'\*\*|__', '', t)
            t_clean = _re.sub(r'^\s*\d+\.\s+', '', t_clean, flags=_re.MULTILINE)
            t_clean = _re.sub(r'^\s*[-*]\s+', '', t_clean, flags=_re.MULTILINE)
            clean_texts.append(t_clean)
            # Extract current block text only (skip prev_text context)
            current_only = t_clean[offset:] if offset > 0 else t_clean
            # Strip header from current block (legacy data with headers in text)
            m2 = _HEADER_RE.match(current_only)
            if m2:
                current_only = current_only[m2.end():]
            current_texts.append(current_only)

        # Run SpaCy (CPU) + KeyBERT (topics) + GLiNER (entities) in parallel
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as pool:
            fut_spacy = pool.submit(lambda: list(self.nlp.pipe(current_texts, batch_size=DEFAULT_BATCH_SIZE, n_process=1)))
            fut_gliner_ents = pool.submit(lambda: self.extract_entities_gliner(clean_texts, text_offsets, entity_hints_per_text=entity_hints, current_texts=current_texts))
            fut_keybert = pool.submit(lambda: self.extract_topics_batch(current_texts))
            docs = fut_spacy.result()
            entities_batch = fut_gliner_ents.result()
            keybert_topics_batch = fut_keybert.result()

        # Refine KeyBERT topics with spaCy + entity overlap, convert to dict format
        topics_batch = []
        for doc, kb_topics, gliner_ents in zip(docs, keybert_topics_batch, entities_batch):
            ent_names = [e[0] if isinstance(e, tuple) else (e.get("name", str(e)) if isinstance(e, dict) else str(e))
                         for e in (gliner_ents or [])]
            refined = self.refine_topics(kb_topics, doc, ent_names)
            # Convert to dict format expected downstream: {"name", "type", "score"}
            topic_dicts = [{"name": t, "type": "topic", "score": round(0.8 - i * 0.05, 3)}
                           for i, t in enumerate(refined)]
            topics_batch.append(topic_dicts)

        HARD_STOPWORDS = {
            "today", "this week", "first", "second", "sa", "recently",
            "now", "currently", "soon", "tomorrow", "yesterday", "next week", "last week",
            # Conversational roles — must never become entities
            "user", "assistant", "you", "ai",
        }

        results = []
        for doc, topics, gliner_ents, role in zip(docs, topics_batch, entities_batch, roles):
            # Entities from GLiNER (returns tuples of (name, score, label))
            if gliner_ents:
                entity_dicts = []
                for item in gliner_ents:
                    if isinstance(item, tuple) and len(item) == 3:
                        name, score, label = item
                    elif isinstance(item, tuple) and len(item) == 2:
                        name, score = item
                        label = "unknown"
                    else:
                        name, score, label = item, 0.5, "unknown"
                    if name.lower() in HARD_STOPWORDS:
                        continue
                    entity_dicts.append({"name": name, "type": label, "score": round(score, 3)})
            else:
                # Fallback to spaCy NER if GLiNER unavailable
                entity_dicts = [
                    {"name": self.normalize_entity(ent.text), "type": ent.label_.lower(), "score": 0.5}
                    for ent in doc.ents
                    if ent.text.strip() and ent.text.strip().lower() not in HARD_STOPWORDS
                ]

            # Fast entity dedup: normalize to canonical key (lowercase, strip trailing s/es)
            import re as _re2
            _seen_ent = {}
            _deduped_ent = []
            for ed in entity_dicts:
                key = _re2.sub(r'\s+', ' ', ed["name"].lower().strip())
                key = _re2.sub(r'(?:ies)$', 'y', key)  # batteries → battery
                key = _re2.sub(r'(?:es)$', '', key)     # dishes → dish
                key = _re2.sub(r's$', '', key)           # chairs → chair
                if key not in _seen_ent:
                    _seen_ent[key] = ed
                    _deduped_ent.append(ed)
            entity_dicts = _deduped_ent

            # Substring containment merge: "Jefferson" absorbed by "Thomas Jefferson"
            _sorted_by_len = sorted(entity_dicts, key=lambda e: len(e["name"]), reverse=True)
            _absorbed = set()
            _merged = []
            for i, ed in enumerate(_sorted_by_len):
                if i in _absorbed:
                    continue
                key_i = ed["name"].lower()
                for j in range(i + 1, len(_sorted_by_len)):
                    if j in _absorbed:
                        continue
                    key_j = _sorted_by_len[j]["name"].lower()
                    # Short name is substring of longer name → absorb
                    if key_j in key_i and len(key_j) >= 3:
                        _absorbed.add(j)
                        # Keep best score
                        ed["score"] = max(ed["score"], _sorted_by_len[j]["score"])
                _merged.append(ed)
            entity_dicts = _merged

            # Plain name list for downstream consumers that expect strings
            entities = [ed["name"] for ed in entity_dicts]

            # Filter stopword topics and dedup by plural normalization
            # topics is now list of dicts: {"name", "type", "score"}
            # Structural quality gate: reject DET-prefixed (entities), bare ADJ
            # (except genre labels like "classical"), and pronouns
            _topic_names = [td["name"] for td in topics]
            _topic_pos = {}
            if _topic_names:
                for _tname, _tdoc in zip(_topic_names, self.nlp.pipe(_topic_names, batch_size=len(_topic_names))):
                    _toks = [t for t in _tdoc if not t.is_space]
                    _topic_pos[_tname] = [t.pos_ for t in _toks] if _toks else []
            def _topic_structural_ok(td):
                _pos = _topic_pos.get(td["name"], [])
                if not _pos:
                    return False
                if _pos[0] == "DET":
                    return False
                if len(_pos) == 1 and _pos[0] == "ADJ" and td.get("type") != "genre":
                    return False
                if len(_pos) == 1 and _pos[0] == "PRON":
                    return False
                return True
            clean_topic_dicts = [td for td in topics if td["name"].lower() not in HARD_STOPWORDS and _topic_structural_ok(td)]
            _seen_topic = {}
            _deduped_topic_dicts = []
            for td in clean_topic_dicts:
                tkey = _re2.sub(r'\s+', ' ', td["name"].lower().strip())
                tkey = _re2.sub(r'(?:ies)$', 'y', tkey)
                tkey = _re2.sub(r'(?:es)$', '', tkey)
                tkey = _re2.sub(r's$', '', tkey)
                if tkey not in _seen_topic:
                    _seen_topic[tkey] = td
                    _deduped_topic_dicts.append(td)
                else:
                    # Keep best score for duplicate
                    existing = _seen_topic[tkey]
                    if td["score"] > existing["score"]:
                        existing["score"] = td["score"]
            topic_dicts = _deduped_topic_dicts[:5]
            clean_topics = [td["name"] for td in topic_dicts]

            # Entity-grounded sentences as claims
            # Assistant blocks require higher entity count + stricter quality threshold
            # to filter advice/recommendations while keeping researched facts.
            # User questions: extract the declarative prefix before '?' as a claim
            # (e.g. "I'm going to Denver for a concert. Any BBQ recs?" → keep first part)
            _header_residue = _re2.compile(r'^[\w|/\(\)\s:]+\]\s*')
            entity_lower = {e.lower() for e in entities}
            is_assistant = (role == "assistant")
            min_entities = 2 if is_assistant else 1
            seen_claims = set()
            filtered_claims = []
            if entity_lower:
                for sent in doc.sents:
                    s = sent.text.strip()
                    # Strip any residual header fragments (e.g. "4|2023/05/27...|user] ")
                    hm = _header_residue.match(s)
                    if hm and '|' in s[:hm.end()]:
                        s = s[hm.end():].strip()
                    if len(s) < 15 or len(s) > 300:
                        continue
                    # Questions: skip for assistant, extract declarative prefix for user
                    if s.endswith('?'):
                        if is_assistant:
                            continue
                        # User question — try to salvage declarative prefix
                        # "I'm going back to Denver for a concert. Do you know any BBQ?"
                        # Split at last sentence boundary before the question
                        parts = _re2.split(r'(?<=[.!])\s+', s)
                        declarative = [p for p in parts if not p.rstrip().endswith('?')]
                        if declarative:
                            s = ' '.join(declarative)
                        else:
                            continue
                    # Skip assistant advice patterns
                    if is_assistant:
                        s_check = s.lower()
                        if any(s_check.startswith(p) for p in (
                            "you should", "you could", "you might", "i'd recommend",
                            "i recommend", "i suggest", "consider ", "try ", "feel free",
                            "don't hesitate", "let me know", "happy to help",
                            "i hope", "glad to", "sure!", "of course",
                        )):
                            continue
                    s_lower = s.lower()
                    # Count entity mentions
                    ent_count = sum(1 for ent in entity_lower if ent in s_lower)
                    if ent_count < min_entities:
                        continue
                    if len(s) < 15:
                        continue
                    # Dedup
                    key = s_lower[:60]
                    if key in seen_claims:
                        continue
                    seen_claims.add(key)
                    filtered_claims.append(s)

            # Quality filter: score and remove filler claims using embedding scorer
            # User blocks need lenient threshold — entity-grounded sentences are
            # already high-signal; scorer prototypes can under-score narrative claims.
            # Assistant blocks need stricter filtering (advice vs researched facts).
            claim_scores = []
            if filtered_claims:
                threshold = 0.05 if is_assistant else -0.10
                scored = self.score_claims(filtered_claims)
                filtered_claims = [c for c, s in scored if s > threshold]
                claim_scores = [round(s, 3) for c, s in scored if s > threshold]

            results.append({
                "doc": doc,
                "entities": entities,
                "entity_dicts": entity_dicts,
                "claims": filtered_claims,
                "claim_scores": claim_scores,
                "topics": clean_topics,
                "topic_dicts": topic_dicts,
            })
        return results

    # ── Gemma E2B Extraction (replaces GLiNER + KeyBERT) ─────────────────
    # Supports two backends:
    #   - Local: llama.cpp (GGUF on local GPU)
    #   - Cloud: Ollama API (e.g. on a GCP VM with L4 GPU)
    # Set GEMMA_OLLAMA_URL=http://host:11434 to use cloud backend
    _gemma_model = None
    _gemma_backend = None  # "local" or "ollama"
    _ollama_url = None

    def _ensure_gemma(self):
        """Lazy-load Gemma E2B model on first use (local or cloud)."""
        if self._gemma_backend is not None:
            return True

        # Check for cloud Ollama endpoint first
        _ollama_url = os.environ.get("GEMMA_OLLAMA_URL", "").strip()
        if _ollama_url:
            try:
                import requests
                r = requests.get(f"{_ollama_url}/api/tags", timeout=5)
                if r.status_code == 200:
                    LocalNLPAdapter._ollama_url = _ollama_url
                    LocalNLPAdapter._gemma_backend = "ollama"
                    print(f"[NLP] Gemma E2B via Ollama cloud: {_ollama_url}")
                    return True
            except Exception as e:
                print(f"[NLP] Ollama at {_ollama_url} unreachable ({e}), trying local...")

        # Fall back to local llama.cpp
        try:
            from llama_cpp import Llama
        except ImportError:
            print("[NLP] llama_cpp not available, falling back to GLiNER+KeyBERT")
            return False
        _pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _model_path = os.path.join(_pkg_root, "models", "google_gemma-3n-E2B-it-Q4_K_M.gguf")
        if not os.path.exists(_model_path):
            print(f"[NLP] Gemma E2B GGUF not found at {_model_path}, falling back to GLiNER+KeyBERT")
            return False
        print(f"[NLP] Loading Gemma E2B for entity+topic extraction (local)...")
        _cpu = os.cpu_count() or 8
        LocalNLPAdapter._gemma_model = Llama(
            model_path=_model_path,
            n_ctx=1024,
            n_gpu_layers=-1,
            n_batch=512,
            n_threads=max(1, _cpu),
            n_threads_batch=max(1, _cpu),
            verbose=False,
        )
        LocalNLPAdapter._gemma_backend = "local"
        print(f"[NLP] Gemma E2B loaded (local, n_ctx=1024)")
        return True

    def _gemma_generate(self, prompt):
        """Generate from Gemma — routes to local llama.cpp or cloud Ollama."""
        if self._gemma_backend == "ollama":
            import requests
            import re as _re
            m = _re.search(r"<start_of_turn>user\n(.*?)<end_of_turn>", prompt, _re.DOTALL)
            user_msg = m.group(1) if m else prompt
            m2 = _re.search(r"<start_of_turn>model\n(.*?)$", prompt, _re.DOTALL)
            seed = m2.group(1) if m2 else ""

            r = requests.post(f"{self._ollama_url}/api/chat", json={
                "model": "gemma4:e2b",
                "messages": [{"role": "user", "content": user_msg}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0.1, "num_predict": 100, "repeat_penalty": 1.15},
            }, timeout=30)
            d = r.json()
            return seed + d.get("message", {}).get("content", "")
        else:
            # Local llama.cpp
            output = ""
            stream = self._gemma_model(
                prompt, max_tokens=100, stop=["<end_of_turn>"],
                stream=True, temperature=0.1, repeat_penalty=1.15,
            )
            for chunk in stream:
                output += chunk["choices"][0]["text"]
            return output

    def _gemma_batch_cloud(self, texts):
        """Send a batch of texts to the cloud batch server. One HTTP round trip.
        Returns list of raw output strings, one per text.
        """
        import requests, re as _re
        # Strip session prefixes
        clean_texts = []
        for text in texts:
            m = _re.match(r"^\[.*?\]\s*", text)
            clean = text[m.end():] if m else text
            clean_texts.append(clean[:2000])

        _batch_url = self._ollama_url.replace(":11434", ":5555")
        try:
            r = requests.post(
                f"{_batch_url}/extract_batch",
                json={"texts": clean_texts},
                timeout=max(30, len(texts) * 3),
            )
            d = r.json()
            return d.get("results", [""] * len(texts))
        except Exception as e:
            print(f"[NLP] Batch server error ({e}), falling back to sequential")
            return None

    def _gemma_extract(self, text):
        """Extract entities + topics from a single block using Gemma E2B.
        Returns (entities: list[str], entity_dicts: list[dict], topics: list[str], topic_dicts: list[dict]).
        """
        import re as _re
        # Strip [session|date|role] prefix
        _m = _re.match(r"^\[.*?\]\s*", text)
        clean = text[_m.end():] if _m else text
        clean = clean[:2000]

        # Prompt includes BOTH proper nouns AND key common nouns that
        # drive user reasoning (transportation modes, expense categories,
        # quantifiable things). This fixes the prior bug where
        # "train", "bus", "taxi" were excluded and their numeric facts
        # couldn't be grouped for arithmetic pairing.
        prompt = (
            "<start_of_turn>user\n"
            "Extract entities and topics from ONLY the text below.\n\n"
            "ENTITIES:\n"
            "- name | type (person, place, brand, product, organization, technology, transport, expense)\n"
            "Include: proper nouns (real names, brands, places, products)\n"
            "AND common nouns that carry quantifiable meaning: transport modes "
            "(train, bus, taxi, subway, bike), expense categories (rent, fare, "
            "commute, grocery, subscription), and domain concepts the user is "
            "tracking. Use the lowercase form for common nouns (train, not Train).\n"
            "Exclude: pronouns, fillers, sentence fragments.\n"
            "If nothing relevant exists, write: - none\n"
            "Max 6.\n\n"
            "TOPICS:\n"
            "- 2-4 word descriptive phrase. Max 4.\n\n"
            "CATEGORY:\n"
            "- one broad 1-3 word category (e.g. real estate, fitness, cooking)\n\n"
            f"Text: {clean}<end_of_turn>\n"
            "<start_of_turn>model\n"
            "ENTITIES:\n"
        )

        output = ""
        try:
            output = self._gemma_generate(prompt)
        except Exception as e:
            print(f"[NLP:Gemma] gen error: {e}")
            return [], [], [], []

        return self._parse_gemma_output(output)

    # Negation phrases that indicate Gemma returned "nothing found" as
    # a literal sentence instead of obeying "write: - none". Any entity
    # or topic line containing these gets rejected.
    _GEMMA_NULL_PATTERNS = (
        "none", "n/a", "no entities", "no topics", "no entity", "no topic",
        "no specific", "not found", "no such", "no proper", "no capitalized",
        "not specified", "not mentioned", "nothing relevant", "no relevant",
        "no category", "not applicable",
    )

    def _parse_gemma_output(self, output):
        """Parse Gemma raw output into (entities, entity_dicts, topics, topic_dicts)."""
        import re as _re
        entities, entity_dicts = [], []
        topics, topic_dicts = [], []
        section = "entities"
        for line in output.split("\n"):
            line = line.strip()
            if "TOPICS:" in line.upper():
                section = "topics"
                continue
            if "CATEGORY:" in line.upper():
                section = "category"
                continue
            if not line.startswith("- ") and not line.startswith("* "):
                continue
            # Normalize bullet format (Gemma uses * or -)
            if line.startswith("* "):
                line = "- " + line[2:]
            item = line[2:].strip()
            if not item or len(item) < 2:
                continue
            _item_low = item.lower()
            # Reject any negation / "nothing found" phrases Gemma sometimes
            # emits even when told to output "- none". Pattern match on the
            # full entry because Gemma likes to say things like "- No
            # Specific Capitalized Proper Nouns Found." which starts with
            # "no" but isn't exactly "none".
            if any(pat in _item_low for pat in self._GEMMA_NULL_PATTERNS):
                continue
            # Reject entries that look like full sentences (5+ words and
            # ends with punctuation — those are prose, not entities/topics)
            if section == "entities" and _item_low.count(" ") >= 4 and _item_low.endswith((".", "!", "?")):
                continue
            if section == "entities":
                parts = item.split("|")
                name = parts[0].strip()
                etype = parts[1].strip().lower() if len(parts) > 1 else "unknown"
                # Clean type annotations in parens: "PowerPoint (product)" -> "PowerPoint"
                paren = _re.match(r'^(.+?)\s*\(', name)
                if paren:
                    name = paren.group(1).strip()
                if len(name) >= 2:
                    # Title-case the name (Gemma output casing is inconsistent)
                    # But preserve all-caps acronyms (REI, AI, NFL, etc)
                    if not name.isupper():
                        name = name.title()
                    # Reject common generic nouns that aren't entities
                    _GENERIC_NOUNS = frozenset({
                        "Chicken", "Chicken Breast", "Kitchen", "Kitchen Knives",
                        "Cooking", "Goals", "City", "Product", "Products", "Salt",
                        "Office", "Company", "Bookshelves", "Phone", "Laptop",
                        "Vegetables", "Men", "Women", "Children", "People",
                        "Students", "Nurses", "Coupons", "Tips", "Budget", "Price",
                        "Recipes", "Food", "Music", "Movie", "Movies", "Travel",
                        "Shopping", "Exercise", "Sleep", "Water", "Protocol",
                        "Yoga", "Dinner", "Breakfast", "Lunch", "Weekend",
                        "Soy", "Gym", "Garden", "Beach", "Park", "Hotel",
                        "Restaurant", "Store", "Bar", "Club", "Church",
                    })
                    _DAYS = frozenset({
                        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                        "Saturday", "Sunday", "Mondays", "Tuesdays", "Wednesdays",
                        "Thursdays", "Fridays", "Saturdays", "Sundays",
                    })
                    if name in _GENERIC_NOUNS or name in _DAYS:
                        continue
                    # Single-word entities under 3 chars are junk (but keep REI, AI, etc)
                    if len(name.split()) == 1 and len(name) < 3:
                        continue
                    entities.append(name)
                    entity_dicts.append({"name": name, "type": etype, "score": 0.8})
            elif section == "topics":
                topics.append(item)
                topic_dicts.append({"name": item, "type": "topic", "score": round(0.8 - len(topic_dicts) * 0.05, 3)})
            elif section == "category":
                # Add category as an extra topic (broad label for cross-block linking)
                cat = item.strip().rstrip(".*")
                if cat and len(cat) >= 3 and cat.lower() not in ("none", "n/a", "general"):
                    topics.append(cat)
                    topic_dicts.append({"name": cat, "type": "category", "score": 0.6})

        return entities, entity_dicts, topics, topic_dicts

    # ── Episodic classification ──────────────────────────────────────────
    # In-memory cache keyed on normalized text. Repeat ACKs / CHATTER
    # dominate real traffic; a tiny dict avoids re-invoking Gemma for
    # obvious short repeats. Context intentionally NOT part of the key —
    # the cache hit-rate win is on short filler, where context rarely
    # flips the classification.
    _EPISODIC_CACHE = {}
    _EPISODIC_CACHE_MAX = 4096

    _EPISODIC_PROMPT_CATEGORIES = (
        "FACT: personal fact, stable preference, or biographical detail\n"
        "DECISION: a choice or commitment being made\n"
        "INSIGHT: a realization, conclusion, or reasoning result\n"
        "PROCEDURE: how-to steps or instructions\n"
        "STATE: current condition, status, or situation\n"
        "INTENT: future plan, goal, or commitment to do\n"
        "DERIVED_FACT: a concrete fact produced by a tool RESULT. Never the call itself.\n"
        "ARTIFACT: reusable output (code, document, plan)\n"
        "CHATTER: small talk, pleasantry, filler, assistant meta-refusals ('I cannot do that', 'I don't have access')\n"
        "ACK: acknowledgment (ok, yes, thanks, got it)\n"
        "QUERY: a question being asked\n"
        "TRANSIENT_DATA: temporary raw data (timestamps, logs, scratch numbers, 'no record found')\n"
        "EXECUTION_LOG: message BEGINS with literal 'TOOL CALL NAME(' or 'TOOL RESULT [toolu_'. "
        "A user/assistant describing tools in prose is NOT EXECUTION_LOG — classify by content."
    )

    def _normalize_episodic_text(self, text):
        """Strip session/role prefix and whitespace for cache + prompt use."""
        import re as _re
        _m = _re.match(r"^\[.*?\]\s*", text or "")
        clean = (text or "")[_m.end():] if _m else (text or "")
        return clean.strip()

    def _parse_episodic_output(self, output):
        """Extract a single class token from Gemma output. Returns a valid
        class or 'unclassified' on any parse failure."""
        if not output:
            return "unclassified"
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # Strip common lead-ins: "Category:", "- ", "* "
            if line.lower().startswith("category:"):
                line = line.split(":", 1)[1].strip()
            if line.startswith(("- ", "* ")):
                line = line[2:].strip()
            # First word, uppercased, stripped of trailing punctuation
            token = line.split()[0] if line.split() else ""
            token = token.strip(".,:;!?\"'()[]").upper()
            if token in _EPISODIC_CLASS_SET:
                return token
        return "unclassified"

    def classify_episodic(self, text, prev_text=None, next_text=None):
        """Classify a block into the flat episodic ontology using Gemma.
        Returns one of EPISODIC_CLASSES or 'unclassified' on failure.

        Context (prev/next) is optional; passing None is fine. Never raises —
        any failure falls back to 'unclassified' so callers can default to
        recallable=True without branching on exceptions.
        """
        clean = self._normalize_episodic_text(text)
        if not clean:
            return "unclassified"

        cache_key = clean[:512]
        cached = self._EPISODIC_CACHE.get(cache_key)
        if cached is not None:
            return cached

        if not self._ensure_gemma():
            return "unclassified"

        # Trim context + text to keep under n_ctx=1024
        clean = clean[:1500]
        prev_s = (self._normalize_episodic_text(prev_text) or "")[:200] if prev_text else ""
        next_s = (self._normalize_episodic_text(next_text) or "")[:200] if next_text else ""

        ctx_lines = ""
        if prev_s:
            ctx_lines += f"Previous: {prev_s}\n"
        if next_s:
            ctx_lines += f"Next: {next_s}\n"

        prompt = (
            "<start_of_turn>user\n"
            "Classify the message below into ONE category. "
            "Output only the category name, nothing else.\n\n"
            "Categories:\n"
            f"{self._EPISODIC_PROMPT_CATEGORIES}\n\n"
            f"{ctx_lines}"
            f"Message: {clean}\n\n"
            "Category:<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

        try:
            output = self._gemma_generate(prompt)
        except Exception as e:
            print(f"[NLP:Gemma] classify_episodic error: {e}")
            return "unclassified"

        cls = self._parse_episodic_output(output)

        if len(self._EPISODIC_CACHE) >= self._EPISODIC_CACHE_MAX:
            self._EPISODIC_CACHE.clear()
        self._EPISODIC_CACHE[cache_key] = cls
        return cls

    def classify_episodic_recallable(self, text, prev_text=None, next_text=None):
        """Classify and derive recallable in one call. Returns (cls, recallable).
        'unclassified' maps to recallable=True (spec §11: failure defaults open).
        """
        cls = self.classify_episodic(text, prev_text, next_text)
        return cls, cls not in NON_RECALLABLE_CLASSES

    def process_batch_gemma(self, texts, roles=None):
        """Process a batch of texts using Gemma E2B for entity+topic extraction.
        Claims are kept as SpaCy sentence splits (unchanged from current pipeline).
        Returns same format as process_batch() for drop-in compatibility.
        Uses concurrent requests for cloud (ollama) backend to hide latency.
        """
        if not self._ensure_gemma():
            # Fallback to original pipeline
            return self.process_batch(texts, roles=roles)

        import re as _re
        _header_re = _re.compile(r'^[\w|/\(\)\s:]+\]\s*')

        # For cloud backend, try batch server first (one HTTP call for N blocks)
        if self._gemma_backend == "ollama" and len(texts) > 1:
            batch_outputs = self._gemma_batch_cloud(texts)
            if batch_outputs:
                # Parse each output through the standard parser
                gemma_results = []
                for output in batch_outputs:
                    gemma_results.append(self._parse_gemma_output(output))
            else:
                # Batch server failed, fall back to concurrent individual calls
                from concurrent.futures import ThreadPoolExecutor, as_completed
                _PARALLEL = min(8, len(texts))
                gemma_results = [None] * len(texts)
                def _extract_one(idx):
                    return idx, self._gemma_extract(texts[idx])
                with ThreadPoolExecutor(max_workers=_PARALLEL) as pool:
                    futures = {pool.submit(_extract_one, i): i for i in range(len(texts))}
                    for fut in as_completed(futures):
                        idx, result = fut.result()
                        gemma_results[idx] = result
        else:
            # Local: sequential (GPU can only run one at a time)
            gemma_results = [self._gemma_extract(text) for text in texts]

        results = []
        for i, text in enumerate(texts):
            role = (roles[i] if roles and i < len(roles) else "user")

            # Gemma extraction: entities + topics (already computed above)
            entities, entity_dicts, topics, topic_dicts = gemma_results[i]

            # Claims: SpaCy sentence split (same as current pipeline)
            doc = self.nlp(text[:5000])
            filtered_claims = []
            claim_scores = []
            entity_lower = {e.lower() for e in entities}
            if entity_lower:
                is_assistant = (role == "assistant")
                min_entities = 2 if is_assistant else 1
                seen_claims = set()
                for sent in doc.sents:
                    s = sent.text.strip()
                    hm = _header_re.match(s)
                    if hm and '|' in s[:hm.end()]:
                        s = s[hm.end():].strip()
                    if len(s) < 15 or len(s) > 300:
                        continue
                    if s.endswith('?'):
                        if is_assistant:
                            continue
                        parts = _re.split(r'(?<=[.!])\s+', s)
                        declarative = [p for p in parts if not p.rstrip().endswith('?')]
                        if declarative:
                            s = ' '.join(declarative)
                        else:
                            continue
                    if is_assistant:
                        s_check = s.lower()
                        if any(s_check.startswith(p) for p in (
                            "you should", "you could", "you might", "i'd recommend",
                            "i recommend", "i suggest", "consider ", "try ", "feel free",
                            "don't hesitate", "let me know", "happy to help",
                            "i hope", "glad to", "sure!", "of course",
                        )):
                            continue
                    s_lower = s.lower()
                    ent_count = sum(1 for ent in entity_lower if ent in s_lower)
                    if ent_count < min_entities:
                        continue
                    key = s_lower[:60]
                    if key in seen_claims:
                        continue
                    seen_claims.add(key)
                    filtered_claims.append(s)

            # Score claims if scorer is available
            if filtered_claims and self._claim_fact_center is not None:
                is_assistant = (role == "assistant")
                threshold = 0.05 if is_assistant else -0.10
                scored = self.score_claims(filtered_claims)
                filtered_claims = [c for c, s in scored if s > threshold]
                claim_scores = [round(s, 3) for c, s in scored if s > threshold]

            results.append({
                "doc": doc,
                "entities": entities,
                "entity_dicts": entity_dicts,
                "claims": filtered_claims,
                "claim_scores": claim_scores,
                "topics": topics,
                "topic_dicts": topic_dicts,
            })
        return results

    def process_text(self, text: str):
        """Process text to extract entities and topics. Compatibility wrapper for process_batch."""
        if not text:
            return {"entities": [], "topics": [], "doc": None}

        # Use batch processor internally
        batch_res = self.process_batch([text])[0]

        return {
            "doc": batch_res["doc"],
            "topics": batch_res["topics"],
            "entities": batch_res["entities"]
        }


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()