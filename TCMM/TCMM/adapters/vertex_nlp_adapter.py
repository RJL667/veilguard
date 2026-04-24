"""
Vertex AI NLP Adapter for TCMM.

Uses Gemini 2.5 Flash for entity + topic extraction instead of local
Gemma E2B / SpaCy. Drop-in alongside LocalNLPAdapter — the semantic
worker calls process_batch_gemma() which this adapter implements.

No GPU required. Pay per API call (~$0.015 per million input tokens).

Usage:
    from adapters.vertex_nlp_adapter import VertexNLPAdapter
    nlp = VertexNLPAdapter(project_id="my-project")
    # Used by TCMM core — same interface as LocalNLPAdapter
"""

import json
import logging
import os
import re
import time
import threading
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger("tcmm.vertex-nlp")

# Same prompt as local Gemma adapter
_EXTRACT_PROMPT = (
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
    "Text: {text}"
)

# Null patterns from local adapter
_NULL_PATTERNS = (
    "none", "n/a", "no entities", "no topics", "no entity", "no topic",
    "no specific", "not found", "no such", "no proper", "no capitalized",
    "not specified", "not mentioned", "nothing relevant", "no relevant",
    "no category", "not applicable",
)

# ── Episodic classification ontology (Vertex-local copy) ─────────────────
# Label space must match LocalNLPAdapter by convention — the recall gate
# reads the derived recallable bool, not the class, so drift only affects
# downstream score boosts and audit consistency. Kept duplicated (not
# shared) so each adapter can tune its prompt for its target model.
EPISODIC_CLASSES = (
    "FACT", "DECISION", "INSIGHT", "PROCEDURE", "STATE", "INTENT",
    "DERIVED_FACT", "ARTIFACT",
    "CHATTER", "ACK", "QUERY", "TRANSIENT_DATA", "EXECUTION_LOG",
)
NON_RECALLABLE_CLASSES = frozenset({
    "CHATTER", "ACK", "QUERY", "TRANSIENT_DATA", "EXECUTION_LOG",
})
_VERTEX_EPISODIC_CLASS_SET = frozenset(EPISODIC_CLASSES)


# ── Deterministic pre/post classification rules ──────────────────────────
# The LLM classifier is generally right but historically drops ~10-15%
# of obvious cases (audit 23 Apr 2026: 103 ``TOOL CALL``/``TOOL RESULT``
# rows mis-tagged as PROCEDURE/INTENT/unclassified, 4 bare ``Noted.``
# rows as unclassified, 9 short questions as non-QUERY).  These patterns
# are unambiguous string-matches — no reason to ever pay an LLM call for
# them.  Running as a pre-filter (for EXECUTION_LOG) saves latency and
# cost; running as a post-override (for ACK) catches cases the LLM got
# semantically close but not exactly right.
_EXECUTION_LOG_PREFIX_RE = re.compile(
    r"^\s*(?:TOOL CALL \w|TOOL RESULT \[toolu_)"
)
_ACK_ONELINER_RE = re.compile(
    r"^(noted|ack|acknowledged|ok|okay|got it|sure|thanks|thank you|"
    r"yes|yep|yeah|no|nope|roger|confirmed|understood|will do|"
    r"sounds good)[.\!]?$",
    re.IGNORECASE,
)
# Very short + heavy exclamation is almost never FACT. Caught aid=638
# ``you have my name!!`` which the LLM wrongly tagged as FACT and
# poisoned recall for every subsequent "what is my name?" turn.
_EMOTIONAL_SHORT_RE = re.compile(r"!!+")

# Prompt tuned for Gemini Flash: category boundaries tightened and
# explicit rules added for the three observed failure modes in live
# traffic — ACK misclassified as FACT ("Noted"), STATE vs FACT confusion
# ("My name is X" → STATE), and empty-state DB answers recalled as facts.
_VERTEX_EPISODIC_PROMPT = (
    "Classify the message below into exactly ONE category. "
    "Respond with only the category label in ALL CAPS, nothing else.\n\n"
    "Categories:\n"
    "FACT: stable personal fact, preference, or biographical detail "
    "(name, job, birthplace, long-term preference)\n"
    "DECISION: a choice or commitment being made\n"
    "INSIGHT: a realization, conclusion, or reasoning result\n"
    "PROCEDURE: how-to steps or instructions\n"
    "STATE: TRANSIENT condition only (hungry, tired, busy, offline) — "
    "NEVER for identity, ownership, or stable traits\n"
    "INTENT: future plan, goal, or commitment to do\n"
    "DERIVED_FACT: a concrete fact (value, number, identifier, "
    "extracted field) produced by a tool RESULT. Never the call itself.\n"
    "ARTIFACT: reusable output (code, document, plan)\n"
    "CHATTER: small talk, pleasantry, filler, assistant meta-refusals "
    "(\"I cannot do that\", \"I don't have access\", \"contact your team\")\n"
    "ACK: acknowledgment of receipt or understanding "
    "(ok, yes, thanks, got it, noted, understood, will do, sounds good)\n"
    "QUERY: a question being asked\n"
    "TRANSIENT_DATA: raw data or empty-state answers that go stale "
    "(\"no record found\", \"I don't have that yet\", timestamps, scratch numbers)\n"
    "EXECUTION_LOG: a record of a tool/command invocation or its raw "
    "result — anything starting with \"TOOL CALL\", \"TOOL RESULT\", "
    "command output, HTTP logs, stack traces, or structured tool JSON.\n\n"
    "Rules:\n"
    "- EXECUTION_LOG applies ONLY when the message text BEGINS with the "
    "literal prefix 'TOOL CALL ' (followed by a function signature like "
    "NAME(...)) OR 'TOOL RESULT [toolu_' (the literal Anthropic tool_use_id "
    "bracket). If the message is a user or assistant describing tools in "
    "natural language (\"please use run_command\", \"I'll call spawn_agent\", "
    "\"the tool returned X\"), it is NOT EXECUTION_LOG — classify by content:\n"
    "  * step-by-step instructions -> PROCEDURE\n"
    "  * a plan or stated goal -> INTENT\n"
    "  * a stated fact -> FACT\n"
    "  * a question -> QUERY\n"
    "- Never DERIVED_FACT on a CALL — DERIVED_FACT is only for a fact "
    "extracted from a RESULT. And never on a user prompt, even if the "
    "prompt describes fetching data.\n"
    "- Raw tool output (agentic agent transcripts, curl body, file-read "
    "dumps, JSON API responses) that was produced BY a tool is EXECUTION_LOG "
    "— but ONLY if it is itself the output rather than a description of it.\n"
    "- Assistant refusals or meta-statements about inability / access / "
    "permissions (\"I cannot proceed\", \"I don't have access to X\", "
    "\"please contact your admin\", \"this request cannot be completed\") "
    "are CHATTER — they describe the assistant's state, not knowledge.\n"
    "- Short assistant replies that only confirm receipt (<= 5 words) are ACK, not FACT.\n"
    "- Name, address, job title, birth date are FACT (stable), never STATE.\n"
    "- \"No record\", \"nothing found\", \"I don't have X\" are TRANSIENT_DATA — "
    "they become stale the moment real data arrives.\n\n"
    "{context}"
    "Message: {text}\n\n"
    "Category:"
)


class VertexNLPAdapter:
    """Vertex AI NLP adapter using Gemini Flash. Same interface as LocalNLPAdapter."""

    DEFAULT_BATCH_SIZE = 32
    # Compatibility shim: the local NLP adapter exposes `gliner_model`
    # (a GLiNER checkpoint) for the benchmark's GPU cleanup path to
    # reload between samples. Vertex has no local model, so expose None
    # to prevent AttributeError in shared teardown code.
    gliner_model = None

    def __init__(
        self,
        project_id: str = "",
        region: str = "us-central1",
        model: str = "gemini-2.5-flash",
        api_key: str = "",
        spacy_model: str = "en_core_web_sm",
    ):
        self.project_id = (project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")).strip()
        self.region = (region or "").strip()
        self.model = (model or "").strip()

        if self.project_id:
            # Vertex AI endpoint — project set, use OAuth (ADC)
            self.api_key = ""
            self._base_url = (
                f"https://{self.region}-aiplatform.googleapis.com/v1/"
                f"projects/{self.project_id}/locations/{self.region}/"
                f"publishers/google/models/{self.model}"
            )
            _auth = f"oauth/ADC (Vertex AI, project={self.project_id})"
        else:
            # Fallback: AI Studio / Generative Language API with API key
            self.api_key = (api_key or os.environ.get("VERTEX_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")).strip()
            self._base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}"
            _auth = "api_key (AI Studio)"

        # SpaCy for sentence splitting (claims) — lightweight, CPU only
        import spacy
        self.nlp = spacy.load(spacy_model)

        self._token_lock = threading.Lock()

        logger.info(f"[VertexNLP] model={model} auth={_auth}")

    def _get_oauth_token(self) -> str:
        """Get OAuth token from ADC. Thread-safe — single refresh at a time."""
        with self._token_lock:
            if hasattr(self, '_oauth_token') and self._oauth_token and time.time() < self._oauth_expiry - 60:
                return self._oauth_token
            import google.auth
            import google.auth.transport.requests
            creds, _ = google.auth.default()
            creds.refresh(google.auth.transport.requests.Request())
            self._oauth_token = creds.token
            self._oauth_expiry = creds.expiry.timestamp() if creds.expiry else time.time() + 3600
            return self._oauth_token

    def canonicalize_topics(self, all_topics):
        """Build a global topic canonicalization map.
        Two-pass merge mirroring LocalNLPAdapter:
        1. Substring containment (short general form absorbs long specific form)
        2. Embedding similarity (>0.80) via Vertex text-embedding-005
        """
        if not all_topics:
            return {}
        from collections import Counter
        import numpy as np
        freq = Counter(t.lower() for t in all_topics)
        unique = list(freq.keys())
        if len(unique) <= 1:
            return {t: t for t in all_topics}

        # ── Pass 1: Substring containment merge ──
        by_len = sorted(unique, key=len)
        substr_map = {}
        absorbed = set()
        for i, short in enumerate(by_len):
            if short in absorbed or len(short) < 4:
                continue
            short_words = set(short.split())
            for j in range(i + 1, len(by_len)):
                long = by_len[j]
                if long in absorbed:
                    continue
                long_words = set(long.split())
                if short in long or (len(short_words) > 0 and short_words <= long_words):
                    if freq[long] >= freq[short] * 3:
                        substr_map[short] = long
                        absorbed.add(short)
                        break
                    else:
                        substr_map[long] = short
                        absorbed.add(long)

        def _resolve(t):
            visited = set()
            while t in substr_map and t not in visited:
                visited.add(t)
                t = substr_map[t]
            return t

        canon_after_substr = {}
        for t in unique:
            canon_after_substr[t] = _resolve(t)

        remaining = list(set(canon_after_substr.values()))
        if len(remaining) <= 1:
            result = {}
            for t in all_topics:
                result[t] = canon_after_substr.get(t.lower(), t)
            return result

        merged_freq = Counter()
        for t in unique:
            merged_freq[canon_after_substr[t]] += freq[t]

        # ── Pass 2: Embedding similarity merge via Vertex ──
        try:
            from adapters.vertex_embedding_adapter import VertexEmbeddingAdapter
            _embedder = VertexEmbeddingAdapter()
            embs = _embedder.embed_batch(remaining)
            embs_np = np.array(embs, dtype=np.float32)
            # Normalize for cosine similarity
            norms = np.linalg.norm(embs_np, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embs_np = embs_np / norms
            sims = embs_np @ embs_np.T

            used = set()
            embed_map = {}
            for i in range(len(remaining)):
                if i in used:
                    continue
                group = [i]
                for j in range(i + 1, len(remaining)):
                    if j not in used and sims[i][j] > 0.80:
                        group.append(j)
                        used.add(j)
                best_idx = max(group, key=lambda idx: (merged_freq[remaining[idx]], -len(remaining[idx])))
                canonical = remaining[best_idx]
                for idx in group:
                    embed_map[remaining[idx]] = canonical

            result = {}
            for t in all_topics:
                step1 = canon_after_substr.get(t.lower(), t.lower())
                step2 = embed_map.get(step1, step1)
                result[t] = step2
            return result
        except Exception as e:
            logger.warning(f"[VertexNLP] Embedding pass failed, using substring-only: {e}")
            result = {}
            for t in all_topics:
                result[t] = canon_after_substr.get(t.lower(), t)
            return result

    def _call_gemini(self, prompt: str, enum_values: Optional[List[str]] = None) -> str:
        """Call Gemini Flash API and return the text response.

        If ``enum_values`` is provided, uses Gemini's structured-output
        feature (``responseSchema`` with ``type=STRING`` + ``enum``) so
        the model is constrained to return EXACTLY one of those values —
        no prose wrappers, no markdown, no hallucinated class names.
        This is what makes the episodic classifier deterministic.
        See https://ai.google.dev/gemini-api/docs/structured-output

        Retries up to 3 times on 429 (rate limit) with exponential backoff.
        """
        import httpx
        url = f"{self._base_url}:generateContent"
        if self.api_key:
            url += f"?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._get_oauth_token()}",
            }

        generation_config: Dict[str, Any] = {"temperature": 0.1}
        if enum_values:
            # Enum mode: model MUST return one of these strings, full stop.
            generation_config["responseMimeType"] = "text/x.enum"
            generation_config["responseSchema"] = {
                "type": "STRING",
                "enum": list(enum_values),
            }

        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }

        for attempt in range(4):
            resp = httpx.post(url, headers=headers, json=body, timeout=60)
            if resp.status_code == 429:
                wait = 2 ** attempt + 1  # 2, 3, 5, 9
                logger.warning(f"[VertexNLP] 429 rate limit, retrying in {wait}s (attempt {attempt+1}/4)")
                time.sleep(wait)
                # Refresh auth header in case token expired during wait
                if not self.api_key:
                    headers["Authorization"] = f"Bearer {self._get_oauth_token()}"
                continue
            break

        if resp.status_code != 200:
            logger.error(f"[VertexNLP] API error {resp.status_code}: {resp.text[:200]}")
            return ""
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        return candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")

    def _parse_output(self, output: str) -> Tuple[List[str], List[Dict], List[str], List[Dict], str]:
        """Parse Gemini output into (entities, entity_dicts, topics, topic_dicts, category).
        Same parser as local Gemma adapter for consistency.
        """
        entities = []
        entity_dicts = []
        topics = []
        topic_dicts = []
        category = ""

        if not output:
            return entities, entity_dicts, topics, topic_dicts, category

        section = None
        for line in output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            upper = line.upper().rstrip(":")
            if upper.startswith("ENTITIES") or upper == "ENTITY":
                section = "entities"
                continue
            elif upper.startswith("TOPICS") or upper == "TOPIC":
                section = "topics"
                continue
            elif upper.startswith("CATEGOR"):
                section = "category"
                continue

            if not line.startswith("-") and not line.startswith("*"):
                continue
            val = line.lstrip("-*").strip()
            if not val:
                continue

            # Check null patterns
            if any(p in val.lower() for p in _NULL_PATTERNS):
                continue

            if section == "entities":
                # Parse "name | type" format
                parts = [p.strip() for p in val.split("|")]
                name = parts[0].strip('"\'').strip()
                etype = parts[1].strip() if len(parts) > 1 else "entity"
                if name and len(name) > 1:
                    entities.append(name)
                    # ``{name, type}`` schema matches the local adapter
                    # and what archive.py filters on.  Using ``text/label``
                    # here caused archive.py:1364 to silently drop every
                    # dict because ``td.get("name")`` was None.
                    entity_dicts.append({"name": name, "type": etype})

            elif section == "topics":
                topic = val.strip('"\'').strip()
                if topic and len(topic) > 2:
                    topics.append(topic)
                    # ``{name, type, score}`` matches the local Gemma
                    # adapter + archive.py filter.  Using ``{topic}`` here
                    # meant archive.py:1372 dropped every row so the
                    # ``topic_dicts`` column was always empty in Lance
                    # despite topics being extracted correctly.  Score
                    # declines gently across positions to mirror the
                    # local adapter's ranking heuristic.
                    topic_dicts.append({
                        "name": topic,
                        "type": "topic",
                        "score": round(max(0.2, 0.8 - 0.05 * len(topics)), 3),
                    })

            elif section == "category":
                category = val.strip('"\'').strip()

        if category and category not in topics:
            topics.append(category)
            topic_dicts.append({
                "name": category,
                "type": "category",
                "score": 0.5,
            })

        return entities, entity_dicts, topics, topic_dicts, category

    def _extract(self, text: str) -> Tuple[List[str], List[Dict], List[str], List[Dict], str]:
        """Extract entities + topics from a single text using Gemini Flash."""
        clean = re.sub(r'^\[.*?\]\s*', '', text)[:2000]
        prompt = _EXTRACT_PROMPT.format(text=clean)
        output = self._call_gemini(prompt)
        return self._parse_output(output)

    # ── Episodic classification ──────────────────────────────────────────
    # In-memory cache keyed on normalized text. Same rationale as the
    # Gemma adapter — ACK / CHATTER dominate traffic and re-invoking the
    # cloud API for identical short filler is wasteful and slow.
    _EPISODIC_CACHE = {}
    _EPISODIC_CACHE_MAX = 4096

    def _normalize_episodic_text(self, text: str) -> str:
        """Strip [session|date|role] prefix and whitespace."""
        if not text:
            return ""
        m = re.match(r"^\[.*?\]\s*", text)
        clean = text[m.end():] if m else text
        return clean.strip()

    def _parse_episodic_output(self, output: str) -> str:
        """Extract a single valid class token from Gemini output.
        Returns a member of EPISODIC_CLASSES or 'unclassified' on any parse failure.

        Logs the raw output on any fall-through so operators can tell
        WHY a row ended up ``unclassified`` — the previous silent-fail
        made it impossible to distinguish "Gemini returned nothing"
        from "Gemini returned something we don't recognise".
        """
        if not output:
            logger.warning("[VertexNLP] episodic classifier: empty Gemini output")
            return "unclassified"
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.lower().startswith("category:"):
                line = line.split(":", 1)[1].strip()
            if line.startswith(("- ", "* ")):
                line = line[2:].strip()
            parts = line.split()
            if not parts:
                continue
            token = parts[0].strip(".,:;!?\"'()[]").upper()
            if token in _VERTEX_EPISODIC_CLASS_SET:
                return token
        # Log the raw output (truncated) so we can diagnose why a row
        # slipped through.  Common culprits: Gemini wrapping the answer
        # in prose ("The category is: QUERY"), markdown fences, or
        # hallucinated class names.
        _sample = output[:200].replace("\n", " ⏎ ")
        logger.warning(
            f"[VertexNLP] episodic classifier: no recognised class in "
            f"output — raw={_sample!r}"
        )
        return "unclassified"

    def classify_episodic(self, text: str, prev_text: str = None, next_text: str = None) -> str:
        """Classify a block into the flat episodic ontology using Gemini Flash.

        Flow:
          1. Deterministic pre-filter: ``TOOL CALL ``/``TOOL RESULT [toolu_``
             prefixes are EXECUTION_LOG without needing the LLM (free + fast +
             removes the ~103 mis-classifications observed on 23 Apr 2026).
          2. LLM call with enum-constrained response schema — Gemini cannot
             return anything outside EPISODIC_CLASSES. No prose wrappers, no
             markdown fences, no hallucinated labels.
          3. Deterministic post-overrides for patterns the LLM semantically
             gets close to but not exactly right (bare ``Noted.`` → ACK,
             short ``!!`` exclamations → CHATTER).

        Returns one of EPISODIC_CLASSES or 'unclassified' on API failure.
        Never raises.
        """
        clean = self._normalize_episodic_text(text)
        if not clean:
            return "unclassified"

        # ── 1. Pre-filter ──────────────────────────────────────────────
        # Absolutely unambiguous patterns. Free, instant, deterministic.
        # No LLM call needed — these strings ARE execution logs by spec.
        if _EXECUTION_LOG_PREFIX_RE.match(text or ""):
            return "EXECUTION_LOG"

        stripped = (text or "").strip()
        if _ACK_ONELINER_RE.match(stripped):
            return "ACK"

        cache_key = clean[:512]
        cached = self._EPISODIC_CACHE.get(cache_key)
        if cached is not None:
            return cached

        clean = clean[:1500]
        prev_s = (self._normalize_episodic_text(prev_text) or "")[:200] if prev_text else ""
        next_s = (self._normalize_episodic_text(next_text) or "")[:200] if next_text else ""

        context = ""
        if prev_s:
            context += f"Previous: {prev_s}\n"
        if next_s:
            context += f"Next: {next_s}\n"

        prompt = _VERTEX_EPISODIC_PROMPT.format(context=context, text=clean)

        # ── 2. LLM call with enum-constrained output ──────────────────
        try:
            output = self._call_gemini(prompt, enum_values=list(EPISODIC_CLASSES))
        except Exception as e:
            logger.warning(f"[VertexNLP] classify_episodic error: {e}")
            return "unclassified"

        cls = self._parse_episodic_output(output)

        # ── 3. Post-overrides ──────────────────────────────────────────
        # Short text with heavy exclamation is almost never FACT — it's
        # emotional filler. Downgrade to CHATTER. This caught aid=638
        # ("you have my name!!") which had poisoned recall for months.
        if cls == "FACT" and len(stripped) < 60 and _EMOTIONAL_SHORT_RE.search(stripped):
            cls = "CHATTER"

        if len(self._EPISODIC_CACHE) >= self._EPISODIC_CACHE_MAX:
            self._EPISODIC_CACHE.clear()
        self._EPISODIC_CACHE[cache_key] = cls
        return cls

    def classify_episodic_recallable(self, text: str, prev_text: str = None, next_text: str = None):
        """Classify and derive recallable in one call. Returns (cls, recallable).
        'unclassified' maps to recallable=True (spec §11 fail-safe).
        """
        cls = self.classify_episodic(text, prev_text, next_text)
        return cls, cls not in NON_RECALLABLE_CLASSES

    # ── Interface methods (same as LocalNLPAdapter) ──────────────────────

    def _build_batch_prompt(self, texts: List[str]) -> str:
        """Build a single prompt that extracts entities/topics for multiple texts."""
        parts = []
        for i, text in enumerate(texts):
            clean = re.sub(r'^\[.*?\]\s*', '', text)[:1500]
            parts.append(f"[TEXT {i}]\n{clean}")
        texts_block = "\n\n".join(parts)

        return (
            "Extract entities and topics from each text below. "
            "Output results per text using the exact format shown.\n\n"
            "For each text output:\n"
            f"[TEXT N]\n"
            "ENTITIES:\n"
            "- name | type (person, place, brand, product, organization, technology, transport, expense)\n"
            "Include: proper nouns AND common nouns with quantifiable meaning.\n"
            "If nothing relevant, write: - none\n"
            "Max 6 per text.\n"
            "TOPICS:\n"
            "- 2-4 word descriptive phrase. Max 4 per text.\n"
            "CATEGORY:\n"
            "- one broad 1-3 word category\n\n"
            f"{texts_block}"
        )

    def _parse_batch_output(self, output: str, n_texts: int) -> List[Tuple]:
        """Parse a multi-text Gemini response into per-text results."""
        # Split by [TEXT N] markers
        chunks = re.split(r'\[TEXT\s+(\d+)\]', output)
        # chunks = ['', '0', 'content0', '1', 'content1', ...]
        per_text = {}
        for i in range(1, len(chunks) - 1, 2):
            try:
                idx = int(chunks[i])
                per_text[idx] = chunks[i + 1]
            except (ValueError, IndexError):
                continue

        results = []
        for i in range(n_texts):
            if i in per_text:
                results.append(self._parse_output(per_text[i]))
            else:
                results.append(([], [], [], [], ""))
        return results

    def process_batch_gemma(self, texts: List[str], roles: List[str] = None) -> List[Dict]:
        """Process a batch of texts using Gemini Flash for NER + topics.
        Returns same format as LocalNLPAdapter.process_batch_gemma().
        Sends up to 8 texts per API call to stay within rate limits.
        """
        # Pre-warm OAuth token
        if not self.api_key:
            self._get_oauth_token()

        # Sub-batch: 8 texts per API call, 2 sub-batches in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed
        SUB_BATCH = 8
        PARALLEL = 4
        gemma_results = [None] * len(texts)

        def _process_sub(start):
            sub_texts = texts[start:start + SUB_BATCH]
            prompt = self._build_batch_prompt(sub_texts)
            output = self._call_gemini(prompt)
            return start, self._parse_batch_output(output, len(sub_texts))

        offsets = list(range(0, len(texts), SUB_BATCH))
        with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            futures = {pool.submit(_process_sub, s): s for s in offsets}
            for fut in as_completed(futures):
                start = futures[fut]
                try:
                    _, parsed = fut.result()
                    for j, p in enumerate(parsed):
                        gemma_results[start + j] = p
                except Exception as e:
                    logger.error(f"[VertexNLP] Batch extract failed at offset {start}: {e}")
                    for j in range(min(SUB_BATCH, len(texts) - start)):
                        if gemma_results[start + j] is None:
                            gemma_results[start + j] = ([], [], [], [], "")

        # Build results with SpaCy claims (same as local adapter)
        results = []
        for i, text in enumerate(texts):
            _r = gemma_results[i] or ([], [], [], [], "")
            entities, entity_dicts, topics, topic_dicts = _r[0], _r[1], _r[2], _r[3]
            category = _r[4] if len(_r) > 4 else ""

            # Claims: SpaCy sentence split
            doc = self.nlp(text[:5000])
            filtered_claims = []
            claim_scores = []
            entity_lower = {e.lower() for e in entities}

            if entity_lower:
                for sent in doc.sents:
                    sent_text = sent.text.strip()
                    if len(sent_text) < 10:
                        continue
                    sent_lower = sent_text.lower()
                    if any(ent in sent_lower for ent in entity_lower):
                        filtered_claims.append(sent_text)
                        claim_scores.append(1.0)

            results.append({
                "entities": entities,
                "entity_dicts": entity_dicts,
                "topics": topics,
                "topic_dicts": topic_dicts,
                "claims": filtered_claims,
                "claim_scores": claim_scores,
                "category": category,
            })

        return results

    def extract_topics_batch(self, texts):
        """Stub for dream engine compatibility. Topics are already extracted
        during ingest via process_batch_gemma, so we return empty lists here
        (same behaviour as LocalNLPAdapter in Gemma mode).
        """
        return [[] for _ in texts]

    def _gemma_generate(self, prompt: str) -> str:
        """Generate raw text from a prompt via Gemini Flash.
        Compatible with LocalNLPAdapter._gemma_generate() so callers
        (e.g. contextual link builder) can use either adapter.
        """
        return self._call_gemini(prompt)

    def process_batch(self, texts, text_offsets=None, topic_hints=None, entity_hints=None, roles=None):
        """Fallback — routes to process_batch_gemma."""
        return self.process_batch_gemma(texts, roles=roles)
