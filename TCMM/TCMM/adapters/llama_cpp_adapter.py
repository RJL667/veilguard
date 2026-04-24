import os
import time
import sys

# 1. Backend Detection: LlamaCPP (Fast GGUF)
try:
    from llama_cpp import Llama
    HAS_LLAMA = True
except ImportError:
    HAS_LLAMA = False

# 2. Backend Detection: ONNX (Compatible Fallback)
try:
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoTokenizer, pipeline, TextStreamer
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

# Streamer for ONNX
if HAS_ONNX:
    class SpeedStreamer(TextStreamer):
        def __init__(self, tokenizer, skip_prompt=True, **decode_kwargs):
            super().__init__(tokenizer, skip_prompt=skip_prompt, **decode_kwargs)
            self.start_time = None
            self.token_count = 0
            
        def put(self, value):
            if self.start_time is None: self.start_time = time.time()
            if hasattr(value, "shape") and len(value.shape) > 1:
                self.token_count += value.shape[1]
            else:
                self.token_count += len(value)
            super().put(value)

        def end(self):
            super().end()
            duration = time.time() - self.start_time if self.start_time else 0
            if duration > 0:
                print(f" \033[90m({self.token_count / duration:.1f} tok/s)\033[0m")

class DreamSynthesizer:
    # Chat template definitions: (system_wrap, user_wrap, assistant_prefix, stop_tokens)
    CHAT_FORMATS = {
        "chatml": {
            "system": "<|im_start|>system\n{text}<|im_end|>\n",
            "user": "<|im_start|>user\n{text}<|im_end|>\n",
            "assistant": "<|im_start|>assistant\n",
            "stop": ["<|im_end|>"],
        },
        "gemma": {
            "system": "",  # Gemma folds system into user turn
            "user": "<start_of_turn>user\n{text}<end_of_turn>\n",
            "assistant": "<start_of_turn>model\n",
            "stop": ["<end_of_turn>"],
        },
    }

    def __init__(self, gguf_path, chat_format="chatml", ollama_url=None, ollama_model=None):
        self.backend = None
        self.model = None
        self.pipeline = None
        self._chat_format = chat_format
        self._fmt = self.CHAT_FORMATS.get(chat_format, self.CHAT_FORMATS["chatml"])
        self._ollama_url = None
        self._ollama_model = None
        # Shared HTTP session with a pooled connection adapter. Without
        # this, every requests.post() builds a fresh TLS/TCP connection
        # which (a) adds ~100-300ms latency per call and (b) can serialize
        # concurrent callers if the default urllib3 pool is sized 1.
        # maxsize=8 covers our 4-worker ThreadPoolExecutor with headroom.
        self._http_session = None

        # Priority 0: Ollama cloud endpoint (if explicitly provided)
        if ollama_url and ollama_model and self._try_init_ollama(ollama_url, ollama_model):
            self.backend = "ollama"
            print(f"\n[DreamSynthesizer] Backend: \033[35mOLLAMA CLOUD\033[0m  url={ollama_url} model={ollama_model}")
            return

        # Priority 1: LlamaCPP (GGUF) - Fast
        if HAS_LLAMA and self._try_init_llama(gguf_path):
            self.backend = "llama"
            print(f"\n[DreamSynthesizer] Backend: \033[32mLLAMA.CPP (GGUF)\033[0m  format={chat_format}")
            return

        # Priority 2: ONNX (Optimum) - Compatible
        if HAS_ONNX and self._try_init_onnx():
            self.backend = "onnx"
            print(f"\n[DreamSynthesizer] Backend: \033[33mONNX (Optimum)\033[0m")
            return

        print("\n[DreamSynthesizer] \033[31mNO BACKEND AVAILABLE\033[0m")

    # ─────────────────────────────────────────────────────────────────
    # Light model: small local GGUF or cloud E2B for simple tasks
    # (labels, descriptions, polish, arithmetic humanization).
    # Falls back to the main backend if not available.
    _light_model = None
    _light_backend = None  # "local" or "cloud"
    _light_ollama_url = None

    def init_light_model(self, local_gguf_path=None, cloud_url=None, cloud_model="gemma4:e2b"):
        """Initialize a lightweight model for simple generation tasks.

        Two modes:
          * local: load a small GGUF (e.g. Gemma E2B Q4_K_M) via llama.cpp
                   with full GPU offload. ~0.25s per label.
          * cloud: route to an Ollama endpoint running a small model.
                   Supports batching for higher throughput.

        Falls back to the main backend if neither is available.
        """
        # Try local GGUF first
        if local_gguf_path and HAS_LLAMA:
            import os
            if os.path.exists(local_gguf_path):
                try:
                    self._light_model = Llama(
                        model_path=local_gguf_path,
                        n_ctx=1024,
                        n_gpu_layers=-1,
                        n_batch=512,
                        verbose=False,
                    )
                    DreamSynthesizer._light_backend = "local"
                    print(f"[DreamSynthesizer] \033[32mLIGHT MODEL (local GPU)\033[0m: {os.path.basename(local_gguf_path)}")
                    return True
                except Exception as e:
                    print(f"[Light Model Init Failed] local: {e}")

        # Try cloud Ollama E2B
        if cloud_url:
            try:
                import requests
                r = requests.get(f"{cloud_url.rstrip('/')}/api/tags", timeout=5)
                names = {m.get("name") for m in r.json().get("models", [])}
                if cloud_model in names:
                    DreamSynthesizer._light_backend = "cloud"
                    DreamSynthesizer._light_ollama_url = cloud_url.rstrip("/")
                    DreamSynthesizer._light_model = cloud_model
                    print(f"[DreamSynthesizer] \033[32mLIGHT MODEL (cloud)\033[0m: {cloud_model} @ {cloud_url}")
                    return True
            except Exception as e:
                print(f"[Light Model Init Failed] cloud: {e}")

        print("[DreamSynthesizer] \033[33mNo light model — using main backend for all tasks\033[0m")
        return False

    def _light_generate(self, prompt, max_tokens=100, quiet=True):
        """Generate using the light model. Falls back to main _generate.

        Use for: labels, descriptions, polish, arithmetic humanization.
        Do NOT use for: arc state extraction, claims, fact distillation,
        profiles — those need the full 26B brain.
        """
        if self._light_backend == "local" and self._light_model is not None:
            formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            output = ""
            try:
                for chunk in self._light_model(
                    formatted,
                    max_tokens=max_tokens,
                    stop=["<end_of_turn>"],
                    stream=True,
                    temperature=0.3,
                    repeat_penalty=1.15,
                ):
                    output += chunk["choices"][0]["text"]
                return output.strip()
            except Exception as e:
                log_perf(f"[ERR light_local] {e}")
                # Fall through to main backend

        elif self._light_backend == "cloud" and self._light_model is not None:
            try:
                import requests
                r = requests.post(
                    f"{self._light_ollama_url}/api/chat",
                    json={
                        "model": self._light_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "think": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": max_tokens,
                            "repeat_penalty": 1.15,
                        },
                    },
                    timeout=30,
                )
                return r.json().get("message", {}).get("content", "").strip()
            except Exception as e:
                log_perf(f"[ERR light_cloud] {e}")
                # Fall through to main backend

        # Fallback: use the main (heavy) backend
        return self._generate(
            self._wrap("", prompt),
            max_tokens=max_tokens,
            quiet=quiet,
            min_predict=768,
        )

    def generate_light_or_heavy(self, prompt, max_tokens=200, quiet=True, use_light=True):
        """Route to light model for simple tasks, heavy model for complex ones.

        Passes call this instead of _generate directly. When use_light=True
        AND a light model is available, uses the fast local/cloud E2B.
        When use_light=False OR no light model, uses the main 26B backend.

        The prompt should be a RAW user prompt (no chat template wrapping) —
        _light_generate handles its own wrapping, and this method wraps for
        the heavy path automatically.
        """
        if use_light and self._light_backend:
            return self._light_generate(prompt, max_tokens=max_tokens, quiet=quiet)
        # Heavy path: wrap in chat template and use main backend
        return self._generate(
            self._wrap("", prompt),
            max_tokens=max_tokens,
            quiet=quiet,
            min_predict=768,
        )

    def _try_init_ollama(self, url, model):
        """Verify Ollama endpoint is reachable and the target model is available."""
        try:
            import requests
            from requests.adapters import HTTPAdapter
            # Build a shared session with a wide connection pool so
            # concurrent ThreadPoolExecutor workers don't serialize on a
            # 1-connection default pool.
            session = requests.Session()
            adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            r = session.get(f"{url.rstrip('/')}/api/tags", timeout=10)
            r.raise_for_status()
            tags = r.json().get("models", [])
            names = {m.get("name") for m in tags}
            if model not in names:
                print(f"[DreamSynthesizer] Ollama model '{model}' not found at {url}. Available: {sorted(names)}")
                return False
            self._ollama_url = url.rstrip("/")
            self._ollama_model = model
            self._http_session = session
            print(f"[OLLAMA BACKEND INITIALIZED] {url} (model={model}, pool_maxsize=8)")
            return True
        except Exception as e:
            print(f"[Ollama Init Failed] {e}")
            return False

    def _wrap(self, system_text, user_text, assistant_seed=""):
        """Build a prompt using the configured chat template."""
        parts = []
        if system_text and self._fmt["system"]:
            parts.append(self._fmt["system"].format(text=system_text))
        if self._chat_format == "gemma" and system_text:
            # Gemma: prepend system instructions to user message
            user_text = system_text + "\n\n" + user_text
        parts.append(self._fmt["user"].format(text=user_text))
        parts.append(self._fmt["assistant"] + assistant_seed)
        return "".join(parts)

    def _try_init_llama(self, model_path):
        if not os.path.exists(model_path):
            return False
        try:
            # GTX 1650 4GB: 3B Q4 ~2GB + SentenceTransformer ~0.5GB = fits
            # Gemma E4B: 4096 covers 10-15 block batches; truncation handles overflow
            _cpu_count = os.cpu_count() or 8
            _n_ctx = 4096 if self._chat_format == "gemma" else 8192
            self.model = Llama(
                model_path=model_path,
                n_ctx=_n_ctx,
                n_gpu_layers=-1,            # All layers on GPU
                n_batch=512,
                n_threads=max(1, _cpu_count),
                n_threads_batch=max(1, _cpu_count),
                verbose=False
            )
            print("[LLAMA BACKEND INITIALIZED]")
            return True
        except Exception as e:
            print(f"[LlamaCPP Init Failed] {e}")
            return False

    def _try_init_onnx(self):
        # Using 1.5B for export safety and speed
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, "models", "onnx_gen")
        save_path = os.path.join(model_dir, "qwen2.5-1.5b-onnx")
        
        if not os.path.exists(save_path):
            print(f"[DreamSynthesizer] Exporting {model_id} to ONNX...")
            try:
                model = ORTModelForCausalLM.from_pretrained(model_id, export=True)
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
            except Exception as e:
                print(f"[ONNX Export Failed] {e}")
                return False
                
        try:
            model = ORTModelForCausalLM.from_pretrained(save_path)
            tokenizer = AutoTokenizer.from_pretrained(save_path)
            self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
            return True
        except Exception:
            return False

    def synthesize(self, texts):
        prompt = self.build_prompt(texts)
        print(f"\n\033[36m[Dreaming]\033[0m Synthesizing cluster ({len(texts)} blocks)...")
        return self._generate(prompt)

    def synthesize_epistemic(self, texts):
        """Structured synthesis: returns dict with canonical_text, topics, entities, claims.

        Uses section-based output (FACTS/ENTITIES/TOPICS) instead of JSON
        for reliable parsing with small models.
        """
        import re as _re
        prompt = self._build_epistemic_prompt(texts)
        print(f"\n\033[36m[Epistemic Synthesis]\033[0m Extracting facts+entities+topics from {len(texts)} blocks...")
        raw = self._generate(prompt, max_tokens=256, quiet=True)
        if not raw:
            return None
        return self._parse_epistemic_sections(raw)

    def _parse_epistemic_sections(self, raw):
        """Parse section-based LLM output into structured dict."""
        import re as _re

        result = {"canonical_text": "", "claims": [], "entities": [], "topics": []}

        # Prepend FACTS: since we pre-seed it in the prompt
        text = "FACTS:\n" + raw.strip()

        # Split into sections
        section_pat = _re.compile(r'^(FACTS|CLAIMS|ENTITIES|TOPICS)\s*:', _re.MULTILINE)
        parts = section_pat.split(text)

        # parts = ['', 'FACTS', '\n- line1\n- line2\n', 'CLAIMS', '\n- ...\n', 'ENTITIES', ...]
        sections = {}
        i = 1
        while i < len(parts) - 1:
            header = parts[i].strip().upper()
            body = parts[i + 1].strip()
            lines = [ln.strip().lstrip("-\u2022* ").strip() for ln in body.splitlines()]
            lines = [ln for ln in lines if ln and len(ln) > 1]
            sections[header] = lines
            i += 2

        # Facts -> canonical_text
        facts = sections.get("FACTS", [])
        if facts:
            result["canonical_text"] = "\n".join(facts)

        # Claims -> structured dicts with subject/predicate/object
        raw_claims = sections.get("CLAIMS", [])
        if raw_claims:
            parsed_claims = []
            for line in raw_claims:
                # Skip example lines the model might echo back
                if line.lower().startswith("example"):
                    continue
                parts_c = [p.strip() for p in line.split("|")]
                if len(parts_c) >= 3:
                    subj, pred, obj = parts_c[0], parts_c[1], parts_c[2]
                    if len(subj) > 0 and len(pred) > 2 and len(obj) > 0:
                        # Extract numeric value if present
                        num_match = _re.search(r'(\d+(?:\.\d+)?)', obj)
                        numeric_val = float(num_match.group(1)) if num_match else None
                        parsed_claims.append({
                            "subject": subj.lower(),
                            "predicate": f"{subj} {pred} {obj}",
                            "object": obj,
                            "numeric_value": numeric_val,
                            "score": 1.0 if numeric_val is not None else 0.7,
                        })
            result["claims"] = parsed_claims

        # Entities -- apply structural filter for safety
        raw_entities = sections.get("ENTITIES", [])
        if raw_entities:
            try:
                from TCMM.core.dream.utils_structural_filters import structural_entity_filter
                result["entities"] = [e for e in raw_entities if structural_entity_filter(e)]
            except ImportError:
                result["entities"] = raw_entities

        # Topics
        raw_topics = sections.get("TOPICS", [])
        if raw_topics:
            # Reject single stopwords, keep meaningful descriptors
            result["topics"] = [t for t in raw_topics if len(t.split()) >= 1 and len(t) > 3][:8]

        # Fallback: if no sections parsed, treat entire output as canonical_text
        if not facts and not raw_entities and not raw_topics and not raw_claims:
            result["canonical_text"] = raw.strip()

        return result

    def _build_epistemic_prompt(self, texts):
        """Prompt for combined fact extraction + entity/topic/claim extraction."""
        joined = "\n\n".join(texts[:20])
        return self._wrap(
            "You consolidate memory blocks into structured knowledge.",
            f"""Consolidate these memory blocks into facts, claims, entities, and topics.

Output format (follow exactly):
FACTS:
- Subject verb object (one fact per line)

CLAIMS:
- Subject | predicate phrase | object or value
- Example: User | subscribes to | National Geographic
- Example: User | owns | 4 pieces of furniture from IKEA

ENTITIES:
- Only proper nouns: person names, places, brands, products, specific technologies

TOPICS:
- 2-4 word descriptors of what the facts are about

Rules:
- Preserve exact numbers and specific values
- Claims must have Subject | predicate | object format with pipe separators
- No generic words as entities (system, method, process, time)
- No timestamps, dates, or numbers as entities
- Maximum 8 facts, 8 claims, 6 entities, 5 topics

Memory blocks:
{joined}""",
            "FACTS:\n"
        )

    def synthesize_arc_state(self, arc_label, block_texts, nlp_entities=None, action_hints=None):
        """Extract structured state sheet — single LLM call.

        The LLM produces a clean list of ``you``-form distilled statements
        that preserve all numbers verbatim. These become the arc's
        canonical events. Entities come from NLP passthrough. Numeric
        facts are regex-extracted from both the LLM output and the raw
        statements. Search terms are taken from NLP topics (the caller
        merges KeyBERT topics downstream).

        Empirically validated against Gemma 4 26B a4b MoE: the previous
        multi-section (classifications + NUMBERS + TERMS) prompt triggered
        the thinking channel and returned empty output on 100% of real
        cc06de0d batches. The current single-purpose "Output user facts"
        prompt works on all real test cases (taxi, guitar amp, photography).

        Returns dict with keys: events, preferences, goals, entities,
        numeric_facts, emotional_causal, current_state, statements.
        """
        if not self.backend or not block_texts:
            return None

        import re

        # ── Step 1: Pre-split user messages into statements ──────
        _filler = {"thanks", "thank you", "sounds good", "great", "okay",
                   "ok", "got it", "i see", "cool", "nice", "awesome",
                   "that's really helpful", "those sound like great recommendations"}
        statements = []
        for text in block_texts[:20]:
            text = text[:400].strip()
            sents = re.split(r'(?<=[.!?])\s+', text)
            for s in sents:
                s = s.strip()
                if len(s) < 15:
                    continue
                s_check = s.rstrip("!.,? ").lower()
                if s_check in _filler:
                    continue
                statements.append(s)

        if not statements:
            return None

        result = {
            "events": [],
            "preferences": [],
            "goals": [],
            "entities": [],
            "numeric_facts": [],
            "search_terms": [],
            "emotional_causal": "",
            "current_state": "",
            "statements": statements,
        }

        # ── Single LLM call: distill statements into clean you-form facts ──
        # Uses /api/chat with think:false (Ollama 0.20+ thinking control)
        # to prevent Gemma 4 IT's thinking channel from silently burning
        # the token budget on large production prompts. See _fix_llm.py
        # diagnostic for the alternatives we rejected.
        joined = " ".join(s.replace("\n", " ")[:180] for s in statements[:15])
        user_prompt = (
            f"Output user facts as plain 'you' statements. "
            f"Keep all numbers verbatim. No preamble. Source: {joined}"
        )
        combined_prompt = self._wrap("", user_prompt)

        raw = self._ollama_generate(
            combined_prompt,
            max_tokens=180,
            temperature=0.3,
            quiet=True,
            min_predict=768,
        ) if self.backend == "ollama" else self._generate_extraction(
            self._truncate_prompt(combined_prompt, 512), max_tokens=512,
        )

        print(".", end="", flush=True)

        # ── Parse LLM output: one "you" statement per line ────────
        distilled_lines = []
        if raw:
            for line in raw.splitlines():
                line = line.strip().lstrip("-*• ").strip()
                line = line.rstrip(".,;:!?").strip()
                if not line or len(line) < 10 or len(line) > 400:
                    continue
                if line.lower() in {"none", "n/a", "not specified"}:
                    continue
                distilled_lines.append(line)

        n_distilled = len(distilled_lines)
        print(f"D({n_distilled}/{len(statements)})", end="", flush=True)

        # ── Promote distilled lines to events / preferences / goals ──
        # The LLM output is clean "you"-form statements. Classify each
        # line by verb pattern into the right bucket:
        #   * preferences: "you prefer/like/love/enjoy/favor/dislike/hate"
        #   * goals: "you want to / plan to / are trying to / your goal is"
        #   * events: everything else (default)
        #
        # This replaces the old multi-section LLM prompt that used to
        # ask for preferences + goals explicitly (dead since Gemma 4's
        # thinking channel broke it). Zero new LLM calls; just regex
        # on the already-distilled lines.
        _PREF_RE = re.compile(
            r"\byou\s+("
            r"prefer|like|love|enjoy|favor|favour|adore|"
            r"dislike|hate|detest|avoid|don't\s+(?:like|enjoy|prefer)|"
            r"are\s+(?:into|fond\s+of)|"
            r"can'?t\s+stand"
            r")\b",
            re.IGNORECASE,
        )
        _GOAL_RE = re.compile(
            r"\byou(?:'re|\s+are)?\s+("
            r"want(?:\s+to)?|plan(?:ning)?(?:\s+to)?|hope(?:\s+to)?|"
            r"intend(?:\s+to)?|aim(?:\s+to)?|trying(?:\s+to)?|"
            r"working\s+on|going\s+to|"
            r"looking\s+(?:to|for)|saving\s+(?:up\s+)?for"
            r")\b",
            re.IGNORECASE,
        )
        _GOAL_NOUN_RE = re.compile(
            r"\byour\s+goal\b|\byour\s+plan\b|\byour\s+target\b",
            re.IGNORECASE,
        )

        # Outcome detection: identify when the user reports the result
        # of a previous recommendation or action.
        _OUTCOME_RE = re.compile(
            r"\byou\s+("
            r"completed|finished|did|accomplished|achieved|succeeded|"
            r"started|began|initiated|tried|attempted|"
            r"failed|couldn't|could\s+not|gave\s+up|abandoned|stopped|quit|"
            r"decided\s+(?:not\s+to|against)"
            r")\b",
            re.IGNORECASE,
        )
        _OUTCOME_SCORES = {
            "completed": 1.0, "finished": 1.0, "did": 0.8,
            "accomplished": 1.0, "achieved": 1.0, "succeeded": 1.0,
            "started": 0.5, "began": 0.5, "initiated": 0.5,
            "tried": 0.4, "attempted": 0.4,
            "failed": -0.5, "couldn't": -0.5, "could not": -0.5,
            "gave up": -0.8, "abandoned": -0.8, "stopped": -0.3,
            "quit": -0.7, "decided not to": -0.3, "decided against": -0.3,
        }

        def _classify_line(text):
            """Return 'preference', 'goal', 'outcome', or 'event'."""
            if _PREF_RE.search(text):
                return "preference", None
            if _GOAL_RE.search(text) or _GOAL_NOUN_RE.search(text):
                return "goal", None
            m = _OUTCOME_RE.search(text)
            if m:
                verb = m.group(1).lower()
                score = _OUTCOME_SCORES.get(verb, 0.0)
                return "outcome", score
            return "event", None

        if distilled_lines:
            for dl in distilled_lines:
                kind, outcome_score = _classify_line(dl)
                if kind == "preference":
                    result["preferences"].append({
                        "preference": dl,
                        "strength": "stated",
                    })
                elif kind == "goal":
                    result["goals"].append({
                        "goal": dl,
                        "status": "stated",
                    })
                elif kind == "outcome":
                    result["events"].append({
                        "description": dl,
                        "status": "distilled",
                        "outcome_score": outcome_score,
                    })
                else:
                    result["events"].append({
                        "description": dl,
                        "status": "distilled",
                    })
            # Also keep the top raw statements as backup events
            for stmt in statements[:5]:
                result["events"].append({"description": stmt, "status": "stated"})
        else:
            for stmt in statements:
                result["events"].append({"description": stmt, "status": "detected"})

        # ── Action hint fallback: inject spaCy verb→obj hints ────
        if action_hints:
            _existing_events = {e["description"].lower() for e in result["events"]}
            _injected = 0
            for hint in action_hints[:15]:
                hint_lower = hint.lower()
                already_covered = any(
                    hint_lower.split(" -> ")[1] in ev if " -> " in hint_lower else False
                    for ev in _existing_events
                )
                if not already_covered:
                    verb_obj = hint.split(" -> ", 1)
                    if len(verb_obj) == 2:
                        verb, obj = verb_obj
                        for stmt in statements:
                            if obj.lower() in stmt.lower() and verb.lower() in stmt.lower():
                                if stmt.lower() not in _existing_events:
                                    result["events"].append({"description": stmt, "status": "detected"})
                                    _existing_events.add(stmt.lower())
                                    _injected += 1
                                break
                        else:
                            synthetic = f"User {hint.replace(' -> ', ' ')}"
                            if synthetic.lower() not in _existing_events:
                                result["events"].append({"description": synthetic, "status": "detected"})
                                _existing_events.add(synthetic.lower())
                                _injected += 1
            if _injected:
                print(f"H+{_injected}", end="", flush=True)

        # ── Regex-extract numeric facts from BOTH LLM output and raw ──
        # Scan distilled lines first (cleaner), then raw statements for
        # anything the LLM missed. De-dup by lowercase text so we don't
        # double-count the same fact in both forms.
        print("F", end="", flush=True)
        seen_facts = set()
        seen_fact_signatures = set()

        def _fact_signature(text):
            # Collapse to digit-bearing tokens so 'taxi fare is $8' matches
            # both 'Your taxi fare is $8 each way' and 'The taxi fare is $8'.
            nums = re.findall(r'\$?\d[\d,\.]*', text)
            words = re.findall(r'[a-z]{3,}', text.lower())
            sig_words = [w for w in words if w not in {"you","your","the","this","that","and","with","for","are","was","were","have","has","will","can","each","way","stated","about"}]
            return tuple(sorted(set(nums))) + tuple(sorted(sig_words[:3]))

        for s in distilled_lines + statements:
            if not re.search(r'\d', s):
                continue
            key = s.lower()
            if key in seen_facts:
                continue
            sig = _fact_signature(s)
            if sig in seen_fact_signatures:
                continue
            seen_facts.add(key)
            seen_fact_signatures.add(sig)
            result["numeric_facts"].append({"key": s, "value": "stated"})

        # ── Entity passthrough from NLP (no LLM call) ────────────
        print("E", end="", flush=True)
        if nlp_entities and len(nlp_entities) > 0:
            _seen_ents = set()
            for ed in nlp_entities:
                if isinstance(ed, dict):
                    name = ed.get("name", "")
                    etype = ed.get("type", "unknown")
                    score = ed.get("score", 0.5)
                else:
                    name = str(ed)
                    etype = "unknown"
                    score = 0.5
                if not name or len(name) < 2 or len(name) > 50:
                    continue
                if name.lower() in _seen_ents:
                    continue
                _seen_ents.add(name.lower())
                result["entities"].append({"name": name, "type": etype, "score": score})
            print(f"({len(result['entities'])})", end="", flush=True)

        # ── search_terms: derived from the distilled lines ──
        # We no longer ask the LLM for a separate TERMS section — the
        # caller merges KeyBERT topics downstream. Here we just extract
        # the content words from distilled lines as a starter set.
        print("T", end="", flush=True)
        _seen_terms = set()
        if nlp_entities:
            for ed in nlp_entities:
                name = ed.get("name", str(ed)) if isinstance(ed, dict) else str(ed)
                _seen_terms.add(name.lower().strip())
        # Cap at 10 terms
        result["search_terms"] = []
        print(f"({len(result['search_terms'])})", end="", flush=True)

        # ── Derive state + context (no extra LLM) ──
        event_stmts = [e["description"] for e in result["events"][:3]]
        result["current_state"] = ""
        result["emotional_causal"] = "; ".join(event_stmts)[:500] if event_stmts else ""

        print(f" ok", flush=True)
        return result

    def _generate_extraction(self, prompt, max_tokens=256):
        """Low-temperature generation for structured extraction (no creativity)."""
        if self.backend == "llama":
            prompt = self._truncate_prompt(prompt, max_tokens)
            full_text = ""
            try:
                stream = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    stop=self._fmt["stop"],
                    stream=True,
                    temperature=0.3,
                    repeat_penalty=1.15
                )
                for chunk in stream:
                    full_text += chunk["choices"][0]["text"]
                return full_text.strip()
            except Exception as e:
                print(f"[Extraction Gen Failed] {e}")
                return None
        if self.backend == "ollama":
            return self._ollama_generate(prompt, max_tokens=max_tokens, temperature=0.3, quiet=True)
        # Fallback to standard generate for non-llama backends
        return self._generate(prompt, max_tokens=max_tokens, quiet=True)

    def synthesize_identity_label(self, texts):
        """Extract a short identity label (1-5 words) from related concepts."""
        # Route to light model if available (E2B is faster and equally
        # good for 2-4 word labels). Falls back to main backend.
        if self._light_backend:
            ctx = texts[0] if texts else ""
            prompt = (
                f"Give a 2-4 word label for this grouping. "
                f"No explanation, just the label.\n{ctx}"
            )
            result = self._light_generate(prompt, max_tokens=20)
        else:
            prompt = self._build_identity_prompt(texts)
            print(f"\n\033[35m[Identity Extract]\033[0m Extracting label from {len(texts)} blocks...")
            result = self._generate(prompt, max_tokens=32, min_predict=768)
        if result:
            result = result.strip().rstrip(".!?:;,").strip()
            result = result.split("\n")[0].strip()
        return result

    def synthesize_domain_label(self, texts):
        """Extract a broad domain category (1-3 words) from related concepts."""
        if self._light_backend:
            ctx = texts[0] if texts else ""
            prompt = (
                f"Give a 2-3 word domain label for this grouping. "
                f"No explanation, just the label.\n{ctx}"
            )
            result = self._light_generate(prompt, max_tokens=16)
        else:
            prompt = self._build_domain_prompt(texts)
            print(f"\n\033[34m[Domain Extract]\033[0m Naming domain from {len(texts)} blocks...")
            result = self._generate(prompt, max_tokens=16, min_predict=768)
        if result:
            result = result.strip().rstrip(".!?:;,").strip()
            result = result.split("\n")[0].strip()
        return result

    def _truncate_prompt(self, prompt, max_tokens):
        """Truncate prompt to fit within n_ctx minus max_tokens headroom."""
        if self.backend != "llama":
            return prompt
        n_ctx = self.model.n_ctx()
        headroom = max_tokens + 64  # tokens for generation + safety margin
        max_prompt_tokens = n_ctx - headroom
        if max_prompt_tokens < 128:
            max_prompt_tokens = 128
        token_ids = self.model.tokenize(prompt.encode("utf-8"), add_bos=False)
        if len(token_ids) <= max_prompt_tokens:
            return prompt
        print(f"  [PROMPT TRUNCATE] {len(token_ids)} tokens -> {max_prompt_tokens} (n_ctx={n_ctx})")
        # Keep the system/instruction prefix and truncate the middle content
        # Find the assistant tag at the end to preserve it
        assistant_tag = self._fmt["assistant"]
        tag_pos = prompt.rfind(assistant_tag)
        if tag_pos > 0:
            suffix = prompt[tag_pos:]
            prefix = prompt[:tag_pos]
            suffix_tokens = self.model.tokenize(suffix.encode("utf-8"), add_bos=False)
            available = max_prompt_tokens - len(suffix_tokens)
            if available > 64:
                prefix_tokens = self.model.tokenize(prefix.encode("utf-8"), add_bos=False)
                truncated = self.model.detokenize(prefix_tokens[:available]).decode("utf-8", errors="replace")
                return truncated + suffix
        # Fallback: hard truncate from the end
        truncated = self.model.detokenize(token_ids[:max_prompt_tokens]).decode("utf-8", errors="replace")
        return truncated

    def _ollama_generate(self, prompt, max_tokens=256, temperature=0.7, quiet=False, min_predict=None):
        """Generate text via remote Ollama HTTP endpoint (non-streaming).

        CRITICAL flow quirks for Gemma 4 IT via Ollama:

          1. The caller (via _build_*_prompt + _wrap) pre-wraps the prompt in
             ``<start_of_turn>user\\n...\\n<end_of_turn>\\n<start_of_turn>model\\n``.
             For the local llama.cpp backend that's correct. For Ollama it is
             NOT — Ollama will either double-wrap (if raw=False) or skip the
             template entirely (if raw=True), and Gemma 4 IT's reasoning
             channels get confused either way, producing empty output.

             Fix: strip the manual chat wrapping before sending, then let
             Ollama apply the model's own template (raw=False).

          2. Gemma 4 IT uses an internal thinking channel for reasoning. Short
             tasks still consume 100-300 tokens of silent reasoning before
             emitting the visible answer. num_predict must be generous
             (minimum ~256, ideally 512+) even for "one word" outputs.

          3. Do NOT pass the Gemma stop tokens (<end_of_turn>) via the options
             list — Ollama's template applies stops itself. Passing them
             manually can truncate the thinking channel mid-stream.
        """
        try:
            start = time.time()

            # Strip any manual chat-template wrapping. Keep only the inner
            # user text + trailing task phrase; Ollama will re-wrap with
            # the model's template.
            clean_prompt = self._unwrap_chat_prompt(prompt)

            # Minimum budget for Gemma 4 IT thinking channel.
            # Empirically: "one word" outputs consume ~200-400 tokens of
            # silent reasoning, so identity/domain/fact calls need >= 768.
            # But multi-section extraction prompts (arc_state) produce
            # empty output at ANY size — don't over-inflate those, just
            # let them hit the caller's max_tokens cap cheaply and fall
            # back to heuristic parsing downstream.
            if min_predict is not None:
                effective_predict = max(max_tokens, min_predict)
            else:
                effective_predict = max_tokens

            # NOTE: do NOT set num_parallel here. It was previously set to
            # 1 which silently forced the server to serialize every
            # concurrent request onto a single slot, killing our 4-worker
            # ThreadPoolExecutor parallelism. The server's env-var
            # OLLAMA_NUM_PARALLEL controls slot count globally.
            #
            # keep_alive MUST be a top-level field (NOT in options). When
            # placed inside options it is silently ignored and the model
            # expires on Ollama's default timer, triggering a 100-200s
            # cold reload on every call. "30m" keeps the model resident
            # for the full benchmark run.
            #
            # think=False disables Gemma 4 IT's reasoning channel.
            # Without this flag, large prompts cause the model to burn
            # the entire num_predict budget on silent internal reasoning
            # tokens and emit nothing visible (verified: streaming shows
            # 0 chunks in 60s on production-size prompts). Ollama 0.20+
            # exposes `think` as a top-level request field.
            payload = {
                "model": self._ollama_model,
                "prompt": clean_prompt,
                "stream": False,
                "keep_alive": "30m",
                "think": False,
                "options": {
                    "num_ctx": 8192,
                    "temperature": temperature,
                    "num_predict": effective_predict,
                },
            }

            # Use the shared session if available (pooled connections,
            # no per-call TLS/TCP handshake, no single-connection pool
            # lock). Fall back to a fresh requests.post only if init
            # didn't set up the session.
            if self._http_session is not None:
                r = self._http_session.post(
                    f"{self._ollama_url}/api/generate",
                    json=payload,
                    timeout=600,
                )
            else:
                import requests
                r = requests.post(
                    f"{self._ollama_url}/api/generate",
                    json=payload,
                    timeout=600,
                )
            r.raise_for_status()
            data = r.json()
            text = (data.get("response") or "").strip()
            text = self._strip_gemma_thinking(text)
            if not quiet:
                eval_count = data.get("eval_count") or 0
                eval_duration = (data.get("eval_duration") or 0) / 1e9
                tps = (eval_count / eval_duration) if eval_duration > 0 else 0
                dur = time.time() - start
                try:
                    print(text)
                except UnicodeEncodeError:
                    print(text.encode("ascii", "replace").decode())
                print(f" \033[90m({tps:.1f} tok/s, {dur:.1f}s)\033[0m")
            return text
        except Exception as e:
            print(f"[Ollama Gen Failed] {e}")
            return None

    @staticmethod
    def _unwrap_chat_prompt(prompt: str) -> str:
        """Strip Gemma / ChatML chat-template markers from a pre-wrapped prompt.

        Used when the prompt was built for a local llama.cpp backend (which
        needs explicit template markers) but is being sent to Ollama (which
        applies the template itself). Removes the outermost user-turn
        markers and the trailing empty assistant-turn marker.
        """
        if not prompt:
            return prompt
        import re as _re
        cleaned = prompt
        # Gemma
        cleaned = _re.sub(r'<start_of_turn>user\s*\n?', '', cleaned)
        cleaned = _re.sub(r'<end_of_turn>\s*\n?', '', cleaned)
        cleaned = _re.sub(r'<start_of_turn>model\s*\n?', '', cleaned)
        # ChatML
        cleaned = _re.sub(r'<\|im_start\|>system\s*\n?', '', cleaned)
        cleaned = _re.sub(r'<\|im_start\|>user\s*\n?', '', cleaned)
        cleaned = _re.sub(r'<\|im_start\|>assistant\s*\n?', '', cleaned)
        cleaned = _re.sub(r'<\|im_end\|>\s*\n?', '', cleaned)
        return cleaned.strip()

    @staticmethod
    def _strip_gemma_thinking(text: str) -> str:
        """Strip Gemma 4's thinking-channel markers from a raw response.

        Gemma 4 (instruction-tuned) wraps reasoning steps in channel tags
        like '<|channel>thought\\n<channel|>actual_answer'. We want only
        the text AFTER the final channel marker. If no markers are found,
        return the text unchanged.

        Also handles the alternative long-form 'channel>final' markers.
        """
        if not text:
            return text
        import re as _re
        # Look for the LAST channel marker (e.g. <|channel|>final or <channel|>)
        # and keep everything after it.
        # Patterns observed: '<|channel>thought', '<channel|>', '<|final|>'
        markers = [
            r'<\|?channel\|?>',
            r'<\|?final\|?>',
            r'<\|?answer\|?>',
            r'<channel\|>',
        ]
        combined = "|".join(markers)
        # Split on any marker; keep the last non-empty segment
        parts = _re.split(combined, text)
        parts = [p.strip() for p in parts if p and p.strip()]
        if not parts:
            return text.strip()
        # Return the last segment that isn't just a label like 'thought'/'thinking'
        _LABEL_ONLY = {"thought", "thinking", "reasoning", "analysis",
                       "scratchpad", "final", "answer"}
        for seg in reversed(parts):
            cleaned = seg.strip()
            if cleaned.lower() in _LABEL_ONLY:
                continue
            return cleaned
        return parts[-1]

    def _generate(self, prompt, max_tokens=256, quiet=False, min_predict=None):
        """Shared backend execution for all prompt types.

        ``min_predict`` (Ollama-only) sets a floor for num_predict so the
        Gemma 4 IT thinking channel has headroom. Only short-output calls
        (identity/domain labels, fact distillation) should set this;
        extraction calls (arc_state) should leave it None so they fail
        cheaply on an empty LLM response and fall through to heuristics.
        """
        if self.backend == "llama":
            prompt = self._truncate_prompt(prompt, max_tokens)
            start = time.time()
            full_text = ""
            try:
                stream = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    stop=self._fmt["stop"],
                    stream=True,
                    temperature=0.7,
                    repeat_penalty=1.1
                )
                count = 0
                for chunk in stream:
                    text = chunk["choices"][0]["text"]
                    if not quiet:
                        try:
                            print(text, end="", flush=True)
                        except UnicodeEncodeError:
                            print(text.encode('ascii', 'replace').decode(), end="", flush=True)
                    full_text += text
                    count += 1

                dur = time.time() - start
                if dur > 0 and not quiet: print(f" \033[90m({count/dur:.1f} tok/s)\033[0m")
                return full_text.strip()
            except Exception as e:
                print(f"[Llama Gen Failed] {e}")
                return None

        elif self.backend == "ollama":
            return self._ollama_generate(
                prompt, max_tokens=max_tokens, temperature=0.7,
                quiet=quiet, min_predict=min_predict,
            )

        elif self.backend == "onnx":
            try:
                streamer = SpeedStreamer(self.pipeline.tokenizer, skip_prompt=True)
                out = self.pipeline(
                    prompt, 
                    max_new_tokens=max_tokens, 
                    do_sample=True,
                    temperature=0.7,
                    streamer=streamer
                )
                return out[0]["generated_text"][len(prompt):].strip()
            except Exception as e:
                print(f"[ONNX Gen Failed] {e}")
                return None
        
        return None

    def build_prompt(self, texts):
        joined = "\n\n".join(texts)
        return self._wrap(
            "You extract and consolidate concrete facts from memory blocks.",
            f"""Extract and consolidate the concrete facts from these memory blocks.

Rules:
- List specific facts: quantities, names, states, ownership, preferences
- Preserve exact numbers and specific values
- When the same fact has different values at different times, keep only the LATEST value
- Format: one fact per line, "Subject verb object"
- Be concise — facts only, no explanations or advice

Memory blocks:
{joined}

Consolidated facts:"""
        )

    def _build_identity_prompt(self, texts):
        """Prompt for extracting a short identity label.

        Kept deliberately minimal AND inline. Gemma 4 IT's thinking channel
        activates on:
          - poison tokens like 'snake_case', 'extract', 'determine'
          - multi-line prompts with rules/bullets
          - trailing cues like 'Label:' or 'Answer:'
        The working pattern is a single-line imperative with inline content
        and a 'Respond with ONLY ...' suffix.
        """
        joined = " ".join(t.replace("\n", " ") for t in texts)
        return self._wrap(
            "",  # no system prompt — it triggers longer reasoning
            f"Short topic label (2-4 words) for: {joined} Respond with ONLY the label."
        )

    def _build_domain_prompt(self, texts):
        """Prompt for extracting a broad domain category.

        Uses the same single-line inline pattern as _build_identity_prompt.
        """
        joined = " ".join(t.replace("\n", " ") for t in texts)
        return self._wrap(
            "",  # no system prompt — avoids burning the thinking channel
            f"Broad domain category (1-2 words) for these concepts: {joined} Respond with ONLY the category."
        )

