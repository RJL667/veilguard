"""
Claude Dream Adapter — drop-in replacement for DreamSynthesizer.

Uses 'claude --print' CLI for all dream synthesis methods.
Same interface as llama_cpp_adapter.DreamSynthesizer so DreamEngine
can swap backends with zero code changes.

Usage:
    from adapters.claude_dream_adapter import ClaudeDreamSynthesizer
    synth = ClaudeDreamSynthesizer()
    # synth.backend, synth.synthesize(), synth.synthesize_epistemic(), etc.
"""

import json
import os
import re
import subprocess
import time


class ClaudeDreamSynthesizer:

    def __init__(self, model_name=None):
        self.backend = "claude"
        self._model = model_name or os.environ.get(
            "CLAUDE_DREAM_MODEL", "claude-sonnet-4-20250514")
        print(f"\n[DreamSynthesizer] Backend: \033[35mCLAUDE ({self._model})\033[0m")

    # ── Shared generation ───────────────────────────────────────

    def _generate(self, prompt, label="", timeout=120):
        """Call claude --print and return raw text."""
        print(f"[CLAUDE:{label}] ", end="", flush=True)
        try:
            result = subprocess.run(
                ["claude", "--print", "--model", self._model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode != 0:
                err = result.stderr.strip()[:100]
                print(f"ERR({err}) ", end="", flush=True)
                return None
            raw = result.stdout.strip()
            print("ok ", end="", flush=True)
            return raw
        except subprocess.TimeoutExpired:
            print("TIMEOUT ", end="", flush=True)
            return None
        except Exception as e:
            print(f"ERR({e}) ", end="", flush=True)
            return None

    # ── 1. synthesize (fact consolidation) ──────────────────────

    def synthesize(self, texts):
        joined = "\n\n".join(texts[:20])
        prompt = (
            "Extract and consolidate the concrete facts from these memory blocks.\n\n"
            "Rules:\n"
            "- List specific facts: quantities, names, states, ownership, preferences\n"
            "- Preserve exact numbers and specific values\n"
            "- When the same fact has different values at different times, keep only the LATEST value\n"
            "- Format: one fact per line, 'Subject verb object'\n"
            "- Be concise - facts only, no explanations or advice\n\n"
            f"Memory blocks:\n{joined}\n\nConsolidated facts:"
        )
        return self._generate(prompt, label="synthesize")

    # ── 2. synthesize_epistemic (structured extraction) ─────────

    def synthesize_epistemic(self, texts):
        """Returns dict with canonical_text, topics, entities, claims."""
        joined = "\n\n".join(texts[:20])
        prompt = (
            "Consolidate these memory blocks into structured knowledge.\n\n"
            "Output format (follow EXACTLY — no markdown, no commentary):\n"
            "FACTS:\n- Subject verb object (one fact per line)\n\n"
            "CLAIMS:\n- Subject | predicate phrase | object or value\n\n"
            "ENTITIES:\n- Only proper nouns: person names, places, brands, products\n\n"
            "TOPICS:\n- 2-4 word descriptors of what the facts are about\n\n"
            "Rules:\n"
            "- Preserve exact numbers and specific values\n"
            "- Claims must have Subject | predicate | object format with pipe separators\n"
            "- No generic words as entities (system, method, process, time)\n"
            "- Maximum 8 facts, 8 claims, 6 entities, 5 topics\n\n"
            f"Memory blocks:\n{joined}"
        )
        raw = self._generate(prompt, label="epistemic")
        if not raw:
            return None
        return self._parse_epistemic_sections(raw)

    def _parse_epistemic_sections(self, raw):
        """Parse section-based output into structured dict.

        Identical logic to llama_cpp_adapter._parse_epistemic_sections
        so downstream code works the same way.
        """
        result = {"canonical_text": "", "claims": [], "entities": [], "topics": []}

        # Ensure FACTS: header exists for parsing
        if not raw.strip().upper().startswith("FACTS"):
            text = "FACTS:\n" + raw.strip()
        else:
            text = raw.strip()

        section_pat = re.compile(r'^(FACTS|CLAIMS|ENTITIES|TOPICS)\s*:', re.MULTILINE)
        parts = section_pat.split(text)

        sections = {}
        i = 1
        while i < len(parts) - 1:
            header = parts[i].strip().upper()
            body = parts[i + 1].strip()
            lines = [ln.strip().lstrip("-\u2022* ").strip() for ln in body.splitlines()]
            lines = [ln for ln in lines if ln and len(ln) > 1]
            sections[header] = lines
            i += 2

        facts = sections.get("FACTS", [])
        if facts:
            result["canonical_text"] = "\n".join(facts)

        raw_claims = sections.get("CLAIMS", [])
        if raw_claims:
            parsed_claims = []
            for line in raw_claims:
                if line.lower().startswith("example"):
                    continue
                parts_c = [p.strip() for p in line.split("|")]
                if len(parts_c) >= 3:
                    subj, pred, obj = parts_c[0], parts_c[1], parts_c[2]
                    if len(subj) > 0 and len(pred) > 2 and len(obj) > 0:
                        num_match = re.search(r'(\d+(?:\.\d+)?)', obj)
                        numeric_val = float(num_match.group(1)) if num_match else None
                        parsed_claims.append({
                            "subject": subj.lower(),
                            "predicate": f"{subj} {pred} {obj}",
                            "object": obj,
                            "numeric_value": numeric_val,
                            "score": 1.0 if numeric_val is not None else 0.7,
                        })
            result["claims"] = parsed_claims

        raw_entities = sections.get("ENTITIES", [])
        if raw_entities:
            try:
                from TCMM.core.dream.utils_structural_filters import structural_entity_filter
                result["entities"] = [e for e in raw_entities if structural_entity_filter(e)]
            except ImportError:
                result["entities"] = raw_entities

        raw_topics = sections.get("TOPICS", [])
        if raw_topics:
            result["topics"] = [t for t in raw_topics if len(t.split()) >= 1 and len(t) > 3][:8]

        if not facts and not raw_entities and not raw_topics and not raw_claims:
            result["canonical_text"] = raw.strip()

        return result

    # ── 3. synthesize_arc_state (structured state sheet) ────────

    def synthesize_arc_state(self, arc_label, block_texts, nlp_entities=None, action_hints=None):
        """Extract structured state sheet — single Claude call.

        Returns dict with keys: events, preferences, goals, entities,
        numeric_facts, emotional_causal, current_state, statements.
        """
        if not block_texts:
            return None

        numbered = "\n".join(f"{i+1}. {t[:400]}" for i, t in enumerate(block_texts[:20]))

        hint_block = ""
        if action_hints:
            hint_lines = "\n".join(f"- {h}" for h in action_hints[:15])
            hint_block = (
                f"\nDetected actions (NLP pre-extraction):\n{hint_lines}\n"
            )

        ent_block = ""
        if nlp_entities:
            ent_names = [
                e.get("name", "") if isinstance(e, dict) else str(e)
                for e in nlp_entities[:30]
            ]
            ent_block = f"\nKnown entities: {', '.join(n for n in ent_names if n)}\n"

        prompt = (
            f"You are extracting structured memory from a user's conversation history.\n"
            f"Arc/topic: {arc_label}\n\n"
            f"User messages:\n{numbered}\n"
            f"{hint_block}{ent_block}\n"
            f"Extract ALL of the following as JSON. Be thorough - capture every event, preference, "
            f"goal, and fact. For events, include implicit actions (things the user did, bought, "
            f"built, assembled, visited, etc.) even if stated casually.\n\n"
            f"Return ONLY valid JSON:\n"
            f'{{\n'
            f'  "events": [{{"description": "...", "status": "stated|completed|planned"}}],\n'
            f'  "preferences": [{{"preference": "...", "strength": "strong|stated|weak"}}],\n'
            f'  "goals": [{{"goal": "...", "status": "active|completed|abandoned", "timeline": "..."}}],\n'
            f'  "numeric_facts": [{{"key": "...", "value": "...", "context": "..."}}],\n'
            f'  "search_terms": ["activity or concept phrases for search indexing"],\n'
            f'  "current_state": "1-2 sentence summary of where the user is now in this topic",\n'
            f'  "emotional_causal": "emotional context or motivation behind the events"\n'
            f'}}'
        )

        try:
            raw = self._generate(prompt, label="arc_state")
            if not raw:
                return None

            # Strip markdown code fences if present
            raw = re.sub(r'^```json\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)

            data = json.loads(raw)
            print(f"E={len(data.get('events',[]))} P={len(data.get('preferences',[]))} "
                  f"G={len(data.get('goals',[]))} F={len(data.get('numeric_facts',[]))} "
                  f"T={len(data.get('search_terms',[]))} ", end="", flush=True)

        except json.JSONDecodeError as e:
            print(f"JSON_ERR({e}) ", end="", flush=True)
            return None
        except Exception as e:
            print(f"ERR({e}) ", end="", flush=True)
            return None

        # Build result in the same format as DreamSynthesizer
        result = {
            "events": [],
            "preferences": [],
            "goals": [],
            "entities": [],
            "numeric_facts": [],
            "search_terms": data.get("search_terms", [])[:10],
            "emotional_causal": data.get("emotional_causal", "")[:500],
            "current_state": data.get("current_state", ""),
            "statements": [t[:400] for t in block_texts[:20]],
        }

        for ev in data.get("events", []):
            if isinstance(ev, dict) and ev.get("description"):
                result["events"].append({
                    "description": ev["description"],
                    "status": ev.get("status", "stated"),
                })
            elif isinstance(ev, str):
                result["events"].append({"description": ev, "status": "stated"})

        for p in data.get("preferences", []):
            if isinstance(p, dict) and p.get("preference"):
                result["preferences"].append({
                    "preference": p["preference"],
                    "strength": p.get("strength", "stated"),
                })

        for g in data.get("goals", []):
            if isinstance(g, dict) and g.get("goal"):
                result["goals"].append({
                    "goal": g["goal"],
                    "status": g.get("status", "active"),
                    "timeline": g.get("timeline", ""),
                })

        for f_ in data.get("numeric_facts", []):
            if isinstance(f_, dict) and f_.get("key"):
                result["numeric_facts"].append({
                    "key": f_["key"],
                    "value": f_.get("value", "stated"),
                    "context": f_.get("context", ""),
                })

        # NLP entity passthrough (same as Phi-4 path)
        if nlp_entities:
            _seen = set()
            for ed in nlp_entities:
                if isinstance(ed, dict):
                    name = ed.get("name", "")
                    etype = ed.get("type", "unknown")
                    score = ed.get("score", 0.5)
                else:
                    name = str(ed)
                    etype = "unknown"
                    score = 0.5
                if not name or len(name) < 2 or name.lower() in _seen:
                    continue
                _seen.add(name.lower())
                result["entities"].append({"name": name, "type": etype, "score": score})

        return result

    # ── 4. synthesize_identity_label ────────────────────────────

    def synthesize_identity_label(self, texts):
        """Extract a short identity label (1-5 words) from related concepts."""
        joined = "\n\n".join(texts[:20])
        prompt = (
            "From the following related concepts, extract the single most stable identity label.\n\n"
            "Rules:\n"
            "- 1 to 5 words ONLY\n"
            "- No sentences or explanations\n"
            "- Return ONLY lowercase underscore format (e.g. cherry_rain)\n"
            "- Must be a referential entity or stable concept name\n\n"
            f"Related concepts:\n{joined}\n\nIdentity label:"
        )
        result = self._generate(prompt, label="identity")
        if result:
            result = result.strip().rstrip(".!?:;,").strip()
            result = result.split("\n")[0].strip()
        return result

    # ── 5. synthesize_domain_label ──────────────────────────────

    def synthesize_domain_label(self, texts):
        """Extract a broad domain category (1-3 words) from related concepts."""
        joined = "\n\n".join(texts[:20])
        prompt = (
            "Provide a broad domain category (1-3 words) that best groups these concepts.\n"
            "Return ONLY the category, nothing else.\n\n"
            f"Concepts:\n{joined}\n\nDomain category:"
        )
        result = self._generate(prompt, label="domain")
        if result:
            result = result.strip().rstrip(".!?:;,").strip()
            result = result.split("\n")[0].strip()
        return result
