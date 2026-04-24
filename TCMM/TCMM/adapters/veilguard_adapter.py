"""
TCMM Veilguard Adapter — Proxy-layer integration for LibreChat + PII Gateway.

This adapter sits inside the Veilguard PII proxy. It intercepts the request/response
cycle and injects TCMM memory into the LLM prompt.

Flow:
    REQUEST:
        LibreChat sends messages[] → proxy extracts latest user message (raw, with PII)
        → TCMM ingests user message → recall → stage → build_prompt()
        → adapter returns a SINGLE prompt string with memory context
        → proxy REDACTs this prompt → sends to Gemini

    RESPONSE:
        Gemini responds (redacted) → proxy REHYDRATEs response
        → adapter receives rehydrated response → splits answer/heatmap
        → TCMM processes heatmap + ingests answer → step++
        → proxy returns answer to LibreChat

TCMM stores REAL data (with PII). Redaction only happens on the wire to Gemini.
LibreChat's conversation history is ignored — TCMM IS the memory.

Usage in PII proxy:
    from tcmm_adapter import VeilguardTCMM

    # On startup
    tcmm = VeilguardTCMM(system_prompt="You are Veilguard...")

    # On each request
    prompt = tcmm.pre_request(user_message)  # Returns enriched prompt
    # ... redact prompt, send to Gemini ...

    # On each response
    answer = tcmm.post_response(raw_llm_output)  # Returns clean answer
"""

import os
import sys
import json
import time
import logging
import re
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger("veilguard-tcmm")

# ── Resolve TCMM imports ─────────────────────────────────────────────────────
# The adapter can be imported from different locations, so we handle path resolution.

_TCMM_ROOT = os.environ.get("TCMM_ROOT", os.path.join(os.path.dirname(__file__), ".."))
if _TCMM_ROOT not in sys.path:
    sys.path.insert(0, _TCMM_ROOT)

from core.tcmm_core import TCMM, split_answer_and_heatmap
from core.blocks import Block


# ── Helper functions (from tcmm_console.py) ───────────────────────────────────

def _classify_user_prompt(text: str) -> str:
    """Classify user prompt as 'query' (ephemeral) or 'input' (archivable).

    'input' = contains facts, preferences, or context worth remembering.
    'query' = questions or commands that are ephemeral.
    """
    if not text or not text.strip():
        return "query"
    t = text.strip()
    t_lower = t.lower()

    # Questions ending with ? are queries
    if t.rstrip().endswith("?"):
        return "query"

    # Interrogative starts → query
    interrogative_starts = (
        "what ", "who ", "where ", "when ", "why ", "how ",
        "which ", "is ", "are ", "was ", "were ", "do ", "does ",
        "did ", "can ", "could ", "would ", "should ", "will ",
        "has ", "have ", "had ",
        "tell me", "explain ", "describe ", "show me", "list ",
        "summarize ", "summarise ",
        "compare ", "confirm ", "identify ", "name ", "recall ",
        "retrieve ", "find ", "look up", "check ",
        "give me", "provide ",
    )
    if t_lower.startswith(interrogative_starts):
        return "query"

    # Statements with personal facts are archivable even if short
    # "My name is Rudolph" (4 words) should be remembered
    personal_markers = (
        "my name", "i am ", "i'm ", "i work", "i live", "i like",
        "i have ", "i prefer", "i use ", "i bought", "i started",
        "i moved", "my favorite", "my email", "my phone",
        "remember ", "note that", "keep in mind", "for the record",
        "i own ", "i drive", "my address", "my birthday",
    )
    if any(m in t_lower for m in personal_markers):
        return "input"

    # Very short messages without personal markers are likely commands
    if len(t.split()) < 5:
        return "query"

    return "input"


def _is_meta_query(text: str) -> bool:
    """Detect meta/conversational queries that should skip recall."""
    if not text:
        return False
    t = text.strip().lower()
    meta_patterns = [
        "repeat the previous", "repeat your previous", "repeat that",
        "say that again", "what did you just say", "what was your last",
        "repeat your answer", "say it again", "tell me again",
    ]
    return any(p in t for p in meta_patterns)


def _is_low_value_meta_answer(answer_text: str) -> bool:
    """Detect low-value assistant responses that should be filtered from the prompt.

    Used as a safety net for legacy live blocks ingested before the heatmap-based
    filter was added. New responses are filtered at ingestion time via the
    knowledge_class heatmap signal (derived + empty used = skip).

    This catches:
    - Very short filler responses ("Noted.", "Hello.", "Rudolph Lamprecht.")
    - Negative recall responses ("No information about X in my records")
    """
    if not answer_text:
        return True
    t = answer_text.strip().lower()
    # Very short = almost always meta/filler
    if len(t) < 40:
        return True
    # Negative recall patterns (LLM admitting it has no info)
    negative_markers = [
        "no information about",
        "in my records",
        "no record of",
        "don't have any information",
        "do not have any information",
        "not in my memory",
        "no memory of",
        "i have no information",
        "i have no record",
        "no stored information",
    ]
    return any(m in t for m in negative_markers)


def _is_system_prompt(text: str) -> bool:
    """Detect LibreChat system/internal prompts that should bypass TCMM entirely.

    LibreChat sends title generation, moderation, and other internal prompts
    through the same API path. These are not user messages and should not
    be ingested into memory or trigger recall.
    """
    if not text:
        return False
    t = text.strip().lower()
    system_markers = [
        "provide a concise",          # Title generation
        "5-word-or-less title",       # Title generation
        "title for the conversation", # Title generation
        "using title case",           # Title generation
        "you are a helpful assistant", # Default system prompt
        "moderate the following",     # Content moderation
        "classify this message",      # Auto-classification
    ]
    return any(m in t for m in system_markers)


# ── Veilguard TCMM Adapter ───────────────────────────────────────────────────

class VeilguardTCMM:
    """
    Proxy-layer TCMM adapter for Veilguard.

    Manages the full TCMM lifecycle within the PII proxy request/response cycle.
    Each conversation gets its own TCMM instance (keyed by conversation_id).
    """

    def __init__(
        self,
        system_prompt: str = "",
        embedder=None,
        nlp_adapter=None,
        data_dir: str = "",
        namespace=None,
        storage: str = "local",
        vector_store: str = "faiss",
        sparse_store: str = "bm25",
    ):
        """
        Initialize TCMM for Veilguard.

        Args:
            system_prompt: Base system prompt (Veilguard instructions).
                          TCMM will append memory blocks and answer contract to this.
            embedder: Optional EmbeddingProvider. Defaults to LocalEmbeddingAdapter.
            nlp_adapter: Optional NLP adapter. Defaults to LocalNLPAdapter.
            data_dir: Directory for TCMM persistence. Defaults to ./tcmm_data/
            namespace: Namespace config — str or dict with {user_id, namespace}.
            storage: Storage backend: "local", "lance", "sqlite".
            vector_store: Vector backend: "faiss", "lance".
            sparse_store: Sparse backend: "bm25", "lance".
        """
        self._data_dir = data_dir or os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(self._data_dir, exist_ok=True)

        # Initialize TCMM core with data_dir for per-session persistence
        self.tcmm = TCMM(
            system_prompt=system_prompt or "You are Veilguard, a Phishield AI assistant.",
            embedder=embedder,
            nlp_adapter=nlp_adapter,
            data_dir=self._data_dir,
            namespace=namespace,
            storage=storage,
            vector_store=vector_store,
            sparse_store=sparse_store,
        )

        self._last_user_message = ""
        self._last_recalled_ids = []

        # Structured tool-call metadata cache, keyed by live-block id.
        # Mirrors what we write to the archive entry's "tool_meta" field
        # — this cache covers the window between ingest and archive
        # flush, and is also used when querying live_blocks that aren't
        # yet persisted. Contents:
        #   { block_id: {tool_name, tool_use_id, param_hash,
        #                 params, result, role} }
        self._block_tool_meta: dict = {}

        # Live block persistence — MUST be per-(user_id, namespace) when the
        # archive is a shared DB (LanceDB), otherwise every new conversation
        # restores another session's blocks and the prompt prefix becomes
        # non-stable (kills upstream KV cache).
        _uid = getattr(self.tcmm, "_user_id", None) or "default"
        _ns = getattr(self.tcmm, "_ns_key", None) or "default"
        # Sanitise path segments
        def _safe(x):
            return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(x))[:64] or "default"
        self._live_dir = os.path.join(
            self.tcmm._data_dir, "live_sessions", _safe(_uid), _safe(_ns)
        )
        os.makedirs(self._live_dir, exist_ok=True)

        # Try restoring live blocks from this exact (user, conversation) session.
        # Other sessions' blocks are in sibling directories and won't leak in.
        self._restore_live_blocks()

        logger.info(f"[TCMM] Initialized. Archive: {len(self.tcmm.archive)} blocks, "
                     f"Live: {len(self.tcmm.live_blocks)} blocks")

    # ── Request Phase ─────────────────────────────────────────────────────

    # PII-proxy origin → TCMM field mapping.
    #
    # TCMM carries role info across three fields with *different*
    # vocabularies; pick the right one or archive classification breaks:
    #
    #   block.origin         → archive.py stores this verbatim on the
    #                          archive entry; the NLP-path episodic
    #                          classifier reads entry["text"] (not origin)
    #                          when assigning block_class, so any origin
    #                          string is accepted but only "user" /
    #                          "assistant" are meaningful downstream.
    #   block.source         → free-form short tag shown in logs / UI
    #                          (adapters/base.py uses user|assistant|
    #                          tool|system). Safe place for our finer-
    #                          grained provenance.
    #   block.priority_class → heat-scoring enum: USER | THOUGHT | TOOL
    #                          | SYSTEM | RECALL | FILE. Bumps base heat
    #                          (SYSTEM=+2.0, USER=+0.5, THOUGHT=+0.3).
    #
    # Tuple: (priority_class, is_assistant, archivable_default, tcmm_origin, tcmm_source_tag).
    # archivable_default=None → use the block's own classifier (user text)
    # or heatmap signal (assistant text).
    _ORIGIN_MAP = {
        # proxy tag         pc         asst   arch   tcmm_origin   tcmm_source
        "user":           ("USER",    False, None,  "user",       "user"),
        "user_image":     ("USER",    False, True,  "user",       "user"),
        "assistant_text": ("THOUGHT", True,  None,  "assistant",  "assistant"),
        "tool_use":       ("THOUGHT", True,  True,  "assistant",  "tool_use"),
        "tool_result":    ("TOOL",    False, True,  "user",       "tool_result"),
        "tool":           ("TOOL",    False, True,  "user",       "tool_result"),
        "system":         ("SYSTEM",  False, True,  "assistant",  "system"),
    }

    def pre_request(
        self,
        user_message: str,
        session_id: str = "",
        origin: str = "user",
    ) -> str:
        """
        Process a user message BEFORE sending to the LLM.

        This is the main integration point. Call this in the proxy BEFORE redaction.
        It returns a complete prompt string with memory context injected.

        The proxy should then:
          1. Redact PII from this prompt
          2. Send the redacted prompt to Gemini
          3. Call post_response() with the rehydrated LLM output

        Args:
            user_message: The raw user message (with real PII)
            session_id: LibreChat conversation/session ID for block tagging
            origin:     Provenance tag from the PII proxy classifier —
                        "user" | "user_image" | "tool_result" | "tool".
                        Stored on the ingested block so later reads can
                        reconstruct the role of the source message.

        Returns:
            str: Complete prompt with system prompt + memory blocks + user task.
                 Ready for PII redaction then LLM.
        """
        # Skip LibreChat system prompts (title generation, moderation, etc.)
        if _is_system_prompt(user_message):
            logger.debug(f"[TCMM] Skipping system prompt: {user_message[:50]}...")
            return ""

        self._last_user_message = user_message
        self._session_id = session_id

        # Instrumentation: break pre_request down so the [PRE] line
        # isn't a single "took=11s" black box. Each section is an
        # independent slowdown source — query embed + cleanup happen
        # before recall, flush + stage_shadow happen after — and
        # without this split we can't point at which one needs the fix.
        import time as _t
        _perf = {}
        _t0 = _t.perf_counter()

        # 1. Cleanup previous turn
        self.tcmm.run_cleanup_cycle()
        self.tcmm._last_recall_trace = {}
        self.tcmm._last_shadow_trace = []
        _t1 = _t.perf_counter(); _perf["cleanup"] = _t1 - _t0

        # 2. Ingest user message (tagged with session ID + origin)
        pclass, is_asst, arch_default, tcmm_origin, tcmm_source = self._ORIGIN_MAP.get(
            origin, self._ORIGIN_MAP["user"]
        )
        # Archive every user turn. The old _classify_user_prompt gate
        # (queries ending in ? marked non-archivable) was a pre-classifier
        # filter that silently dropped anything the user asked — including
        # the actual probes people expect to be remembered. The real
        # filtering belongs to a dedicated classifier downstream; for now
        # every turn goes to the archive so linker / recall / analytics
        # have the full conversation to work with.
        user_archivable = True if arch_default is None else arch_default

        # block.source carries the fine-grained proxy tag (tool_result
        # etc.) + session tail; block.origin carries the TCMM canonical
        # "user"|"assistant" value so archive.py's block classifier fires.
        src_tag = f"{tcmm_source}:{session_id[:12]}" if session_id else tcmm_source

        # User-message ingestion is DEFERRED to a background task —
        # it's the 6-7 second slow path on the [PRE-PROFILE] breakdown
        # (LanceDB's delete-then-add pattern on the archive table +
        # file-lock contention in next_id). Recall on this turn doesn't
        # need the user message in the archive: the user_message IS the
        # query, not a recall target. Live_blocks rendering for the
        # prompt doesn't need it either — LibreChat appends the current
        # user turn to messages[] separately from TCMM memory. By the
        # time the NEXT turn's pre_request runs, the per-session lock
        # guarantees this turn's background ingest has landed so that
        # recall sees it.
        #
        # Record what we'd ingest so server.py can fire the bg task.
        self._pending_user_ingest = {
            "text": user_message,
            "priority_class": pclass,
            "is_assistant": is_asst,
            "archivable": user_archivable,
            "source": src_tag,
            "tcmm_origin": tcmm_origin,
        }
        _t2 = _t.perf_counter(); _perf["ingest_user"] = _t2 - _t1

        # 3. Recall relevant memories (skip for meta-queries)
        self._last_recalled_ids = []
        if not _is_meta_query(user_message):
            self._last_recalled_ids = self.tcmm.recall(user_message) or []
        _t3 = _t.perf_counter(); _perf["recall"] = _t3 - _t2

        # 4. Stage recalled blocks into shadow
        if self._last_recalled_ids:
            self.tcmm.stage_shadow_blocks(self._last_recalled_ids)
            logger.info(f"[TCMM] Recalled {len(self._last_recalled_ids)} blocks for query")
        _t4 = _t.perf_counter(); _perf["stage_shadow"] = _t4 - _t3

        # 5. Build the memory context (blocks + instructions, no system prompt or user task)
        memory_context = self._build_memory_context(user_message)
        _t5 = _t.perf_counter(); _perf["build_ctx"] = _t5 - _t4

        _lb = len(self.tcmm.live_blocks)
        _sb = len(self.tcmm.shadow_blocks)
        logger.info(f"[TCMM] Memory context: {len(memory_context)} chars | live={_lb} | shadow={_sb}")
        logger.info(
            "[PRE-PROFILE] "
            + "  ".join(f"{k}={v*1000:.0f}ms" for k, v in _perf.items())
            + f"  total={(_t5 - _t0)*1000:.0f}ms"
        )

        # ── Session log: full trace of what TCMM did ──
        self._log_session_trace(session_id, user_message, memory_context)

        return memory_context

    def flush_pending_user_ingest(self, session_id: str = "") -> bool:
        """Actually run the user-message add_new_block + archive write.

        Called as a background task from server.py's /pre_request
        endpoint AFTER the HTTP response has been returned. This is
        where the 6-7 second LanceDB write happens — the user never
        sees it because the prompt already went out.

        The per-session asyncio lock in server.py guarantees this
        completes before the next pre_request/post_response runs for
        the same session, so the NEXT turn's recall will see this
        turn's user message in the archive.

        Side effect: records ``tcmm._current_turn_user_aid`` so any
        subsequent tool_use / tool_result blocks in the same turn
        (fed via ``ingest_turn``) can stamp ``lineage.parents`` back
        to this user message. Without that, tool blocks sit as
        orphan self-rooted entries — fine for flat recall but a gap
        for dream-engine canonical-state synthesis which wants to
        reassemble a turn as one unit.

        Returns True if ingest ran, False if nothing was pending.
        """
        pending = getattr(self, "_pending_user_ingest", None)
        if not pending:
            return False
        self._pending_user_ingest = None

        try:
            created = self.tcmm.add_new_block(
                text=pending["text"],
                priority_class=pending["priority_class"],
                is_assistant=pending["is_assistant"],
                archivable=pending["archivable"],
                source=pending["source"],
            )
            if created is not None:
                try:
                    created.origin = pending["tcmm_origin"]
                except Exception:
                    pass
                # Sync origin back to the archive entry (same as the old
                # synchronous path used to do).
                try:
                    aid = getattr(created, "origin_archive_id", None)
                    if aid is not None:
                        entry = self.tcmm.archive.get(aid)
                        if isinstance(entry, dict):
                            entry["source"] = pending["source"]
                            entry["priority_class"] = pending["priority_class"]
                            entry["origin"] = pending["tcmm_origin"]
                            entry["is_assistant"] = pending["is_assistant"]
                            try:
                                self.tcmm.archive[aid] = entry
                            except Exception:
                                pass
                        # Record this as the current turn's "anchor aid"
                        # so the tool_use / tool_result blocks that
                        # follow via ingest_turn can backlink to it.
                        self.tcmm._current_turn_user_aid = int(aid)
                except Exception:
                    pass
            self.tcmm.flush_current_block()
            return True
        except Exception as e:
            logger.error(
                f"[TCMM] deferred user ingest failed for "
                f"session={session_id[:16]}: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return False

    def _build_memory_context(self, task_query: str) -> str:
        """Build the memory context with the Phase-6 cache-boundary layout.

        Matches TCMM build_prompt's live/shadow split:
          L0  = MEMORY CONTEXT header + usage instructions (byte-identical)
          L1  = live blocks, insertion-ordered, append-only between checkpoints
          ----- END LIVE MEMORY -----  ← cache boundary (PII proxy caches here)
          L2  = shadow blocks (volatile recall — churns turn-to-turn)
          L3  = answer contract (volatile for structural consistency with the
                rest of the volatile tail; constant bytes but lives below the
                cache boundary so shadow churn is cheap)

        The `--- END LIVE MEMORY ---` marker is always emitted — even with
        no shadow blocks — so the boundary sits at a deterministic position
        every turn. That's what the PII proxy's _split_for_cache splits on.
        """
        from core.blocks import RecallCandidate

        self.tcmm.flush_pending()

        # ── L1: live candidates (stable, cacheable) ──
        live_candidates = []
        for b in self.tcmm.live_blocks:
            if getattr(b, "is_open", False):
                continue
            # Filter negative/failure assistant responses so they don't
            # poison the prompt. Content-based filter is deterministic.
            if getattr(b, "priority_class", "") == "THOUGHT":
                bt = getattr(b, "text", "").strip().lower()
                if _is_low_value_meta_answer(bt):
                    continue
            live_candidates.append(RecallCandidate(
                aid=-1, source="live", block=b,
                lineage_id=getattr(b, "lineage_root", None),
            ))

        # ── L2: shadow candidates (volatile, below cache boundary) ──
        shadow_candidates = []
        for b in self.tcmm.shadow_blocks:
            if b.is_open:
                continue
            shadow_candidates.append(RecallCandidate(
                aid=-1, source="shadow", block=b,
                archive_id=getattr(b, "origin_archive_id", None),
                lineage_id=getattr(b, "lineage_root", None),
            ))

        # Cap total size to avoid runaway prompts. Live first, then shadow
        # fills the remaining budget — preserves the live region intact.
        _MAX_BLOCKS = 64
        if len(live_candidates) >= _MAX_BLOCKS:
            live_candidates = live_candidates[:_MAX_BLOCKS]
            shadow_candidates = []
        else:
            _remaining = _MAX_BLOCKS - len(live_candidates)
            shadow_candidates = shadow_candidates[:_remaining]

        unified = live_candidates + shadow_candidates

        # Stable IDs (no ephemeral 0..N-1): live uses block.id, shadow uses
        # origin_archive_id (falls back to block.id if not archived).
        for c in unified:
            b = c.block
            if c.source == "shadow":
                c.aid = getattr(b, "origin_archive_id", None) or b.id
            else:
                c.aid = b.id
            b.times_in_prompt += 1

        self.tcmm._last_presented_candidates = unified

        if not unified:
            return ""

        # ── L0: header + constant instructions ──
        parts = []
        parts.append("--- MEMORY CONTEXT (from previous conversations) ---")
        parts.append("Use these memory blocks as context. Do not mention block IDs to the user.")
        parts.append("Names may appear as REF_PERSON_N tokens (privacy placeholders). Treat them as real names — if the user asks about REF_PERSON_2, search ALL memory blocks for REF_PERSON_2 and report what you find.\n")

        # ── L1: live blocks (cacheable region) ──
        # No topic/entity overlay — those populate asynchronously from the
        # NLP worker and would flip block labels between turns.
        for c in live_candidates:
            b = c.block
            role = getattr(b, "priority_class", "") or "?"
            label = f"[Memory index={c.aid} | role={role} | src=live]"
            parts.append(f"{label}\n{b.text}")

        # ── Cache boundary (always emitted, deterministic position) ──
        # PII proxy's _split_for_cache finds this marker and attaches the
        # cache_control breakpoint here. Must stay byte-identical across turns.
        parts.append("\n--- END LIVE MEMORY ---")

        # ── L2: shadow blocks (volatile tail) ──
        # Framing: we used to label this section "POSSIBLE MEMORY
        # CANDIDATES (volatile)" which caused the model to treat the
        # blocks as uncertain retrievals rather than trusted facts.
        # The blocks ARE facts — they are user/assistant statements
        # from this user's previous conversations, re-surfaced by
        # recall. Volatility is a cache-prefix concept (they change
        # turn-to-turn) and has nothing to do with their truth value.
        # Treat them as memory, not as "candidates".
        if shadow_candidates:
            parts.append("--- RECALLED MEMORY (from earlier in this user's history) ---")
            for c in shadow_candidates:
                b = c.block
                role = getattr(b, "priority_class", "") or "?"
                label = f"[Memory index={c.aid} | role={role}]"
                parts.append(f"{label}\n{b.text}")

        parts.append("\n--- END MEMORY CONTEXT ---")

        # Answer contract — volatile by convention; shadow churn already
        # broke the prefix above this point, so no extra harm in keeping it
        # here for layout clarity.
        parts.append("""
After your answer, append a JSON object (no markdown fence) with:
{"knowledge_class": "derived"|"novel"|"mixed", "used": {"<block_id>": <relevance 0-1>}}
- "derived" = answer uses only memory/general knowledge
- "novel" = answer contains new information worth remembering
- "mixed" = combination
- "used" = which Memory block IDs you referenced (empty {} if none)
""")

        return "\n".join(parts)

    # ── Session Trace Logging ────────────────────────────────────────────

    def _log_session_trace(self, session_id: str, user_message: str, memory_context: str):
        """Write a detailed session trace to tcmm_session.log for debugging."""
        import datetime
        log_path = os.path.join(self._data_dir, "tcmm_session.log")
        try:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            tcmm = self.tcmm
            lines = []
            lines.append(f"\n{'='*80}")
            lines.append(f"[{ts}] SESSION TRACE — session={session_id}")
            lines.append(f"{'='*80}")
            lines.append(f"QUERY: {user_message}")
            lines.append(f"")

            # Recall results
            lines.append(f"RECALL: {len(self._last_recalled_ids)} archive IDs returned: {self._last_recalled_ids[:20]}")
            if self._last_recalled_ids:
                for aid in self._last_recalled_ids[:10]:
                    entry = tcmm.get_archive_entry(aid)
                    if entry:
                        lines.append(f"  AID {aid}: {str(entry.get('text',''))[:80]}")
                        lines.append(f"    entities={entry.get('entities',[])} topics={entry.get('topics',[])[:3]}")
                    else:
                        lines.append(f"  AID {aid}: NOT FOUND (cross-ns resolve failed)")

            # Live blocks
            lines.append(f"")
            lines.append(f"LIVE BLOCKS: {len(tcmm.live_blocks)}")
            for i, b in enumerate(tcmm.live_blocks[-30:]):
                text = getattr(b, "text", "")[:60]
                src = getattr(b, "source", "")
                role = getattr(b, "priority_class", "")
                aid = getattr(b, "origin_archive_id", None)
                lines.append(f"  [{i}] {role}|{src} aid={aid}: {text}")

            # Shadow blocks
            lines.append(f"")
            lines.append(f"SHADOW BLOCKS: {len(tcmm.shadow_blocks)}")
            for i, b in enumerate(tcmm.shadow_blocks):
                text = getattr(b, "text", "")[:60]
                aid = getattr(b, "origin_archive_id", None)
                lines.append(f"  [{i}] aid={aid}: {text}")

            # Archive state
            lines.append(f"")
            lines.append(f"ARCHIVE: {len(tcmm.archive)} blocks (namespace-scoped)")

            # The full prompt being sent
            lines.append(f"")
            lines.append(f"PROMPT ({len(memory_context)} chars, {len(memory_context.split())} words):")
            lines.append(f"--- START PROMPT ---")
            lines.append(memory_context)
            lines.append(f"--- END PROMPT ---")

            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

        except Exception as e:
            logger.warning(f"[TCMM] Session trace logging failed: {e}")

    def _log_session_response(self, session_id: str, answer: str, flag_obj: dict):
        """Log the LLM response trace."""
        import datetime
        log_path = os.path.join(self._data_dir, "tcmm_session.log")
        try:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            lines = []
            lines.append(f"")
            lines.append(f"[{ts}] RESPONSE — session={session_id}")
            lines.append(f"ANSWER ({len(answer)} chars): {answer[:200]}")
            if flag_obj:
                lines.append(f"HEATMAP: class={flag_obj.get('knowledge_class','')} used={flag_obj.get('used',{})}")
            lines.append(f"")

            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as e:
            logger.warning(f"[TCMM] Response trace logging failed: {e}")

    # ── Response Phase ────────────────────────────────────────────────────

    def post_response(
        self,
        raw_llm_output: str,
        session_id: str = "",
        origin: str = "assistant_text",
    ) -> str:
        """
        Process the LLM response AFTER rehydration.

        Call this in the proxy AFTER rehydrating PII tokens back to real values.
        It parses the heatmap, updates TCMM memory, and returns the clean answer.

        Handles the auto-recall retry loop: if process_heatmap triggers lazy recall,
        it returns None to signal that the proxy should re-send with an updated prompt.

        Args:
            raw_llm_output: The full LLM output (rehydrated, with real PII).
                           May contain answer text + heatmap JSON.

        Returns:
            str: The clean answer text (heatmap stripped).
                 Returns empty string if output was empty/invalid.
        """
        if not raw_llm_output or not raw_llm_output.strip():
            logger.warning("[TCMM] Empty LLM output — skipping ingestion")
            return ""

        # 1. Try to split answer and heatmap (may not have heatmap in streaming mode)
        answer, flag_obj = split_answer_and_heatmap(raw_llm_output)

        if not answer:
            answer = raw_llm_output.strip()

        if not flag_obj:
            flag_obj = {}

        # Log response trace
        self._log_session_response(session_id, answer, flag_obj)

        # 2. Process heatmap if present (reinforcement learning signal)
        if flag_obj.get("used") or flag_obj.get("knowledge_class"):
            try:
                self.tcmm.process_heatmap(
                    flag_obj,
                    task_query=self._last_user_message,
                    generated_answer=answer,
                )
            except Exception as e:
                logger.warning(f"[TCMM] process_heatmap error (non-fatal): {e}")

        # 3. Ingest assistant answer (if valuable)
        #    Use the LLM's own heatmap signal to decide:
        #    - knowledge_class="derived" + empty used={} → LLM didn't use memory, generic response
        #    - knowledge_class="novel"/"mixed" or used={...} → LLM contributed real content
        knowledge_class = flag_obj.get("knowledge_class", "novel")
        if knowledge_class not in ("derived", "novel", "mixed"):
            knowledge_class = "novel"
        used_blocks = flag_obj.get("used", {})

        # Archive every assistant turn. The old gates (knowledge_class
        # != "derived" for archivability, and the is_empty_derived
        # ingest-skip for "derived + used={}") were pre-classifier
        # heuristics that quietly lost turns the LLM emitted, leaving
        # holes in the conversation record. With dedup already disabled
        # and no promotion step by design, the archive should be a
        # complete-fidelity log of every turn; a downstream classifier
        # will tag / filter later.
        is_archivable = True

        if not answer:
            pass  # skip only truly empty payloads
        else:
            pclass, is_asst, _arch, tcmm_origin, tcmm_source = self._ORIGIN_MAP.get(
                origin, self._ORIGIN_MAP["assistant_text"]
            )
            src_tag = f"{tcmm_source}:{session_id[:12]}" if session_id else tcmm_source
            created = self.tcmm.add_new_block(
                text=answer,
                priority_class=pclass,
                is_assistant=is_asst,
                archivable=is_archivable,
                source=src_tag,
            )
            if created is not None:
                try:
                    created.origin = tcmm_origin
                except Exception:
                    pass
                # Sync archive entry — same placeholder-drift fix as
                # pre_request / ingest_turn (see _ensure_archive_entry).
                try:
                    aid = getattr(created, "origin_archive_id", None)
                    if aid is not None:
                        entry = self.tcmm.archive.get(aid)
                        if isinstance(entry, dict):
                            entry["source"] = src_tag
                            entry["priority_class"] = pclass
                            entry["origin"] = tcmm_origin
                            entry["is_assistant"] = is_asst
                            # Lineage: link the assistant's response back
                            # to the user message that prompted it. Same
                            # pattern as tool blocks in ingest_turn — all
                            # blocks in the turn share a lineage.root,
                            # so dream can cluster by root.
                            parent_aid = getattr(
                                self.tcmm, "_current_turn_user_aid", None
                            )
                            if parent_aid is not None:
                                _lin = entry.get("lineage") or {}
                                if not isinstance(_lin, dict):
                                    try:
                                        _lin = dict(_lin)
                                    except Exception:
                                        _lin = {}
                                _lin["parents"] = [int(parent_aid)]
                                _lin["root"] = int(parent_aid)
                                entry["lineage"] = _lin
                                entry["lineage_root"] = int(parent_aid)
                            try:
                                self.tcmm.archive[aid] = entry
                            except Exception:
                                pass
                except Exception:
                    pass

            # Note: the old "derived → recallable=False, archivable=False"
            # override is removed. The archive layer stays complete; any
            # recall-time filtering will be delegated to the future
            # classifier, not decided at ingest from the LLM's own
            # knowledge_class self-report (which was frequently wrong
            # and quietly suppressed legitimate turns).

            self.tcmm.flush_current_block()
            logger.info(
                f"[TCMM] Ingested answer (class={knowledge_class}, "
                f"proxy_origin={origin}, tcmm_origin={tcmm_origin}, "
                f"src={tcmm_source}, len={len(answer)})"
            )

        # 4. Advance step
        self.tcmm.current_step += 1

        # 5. Persist live blocks after each turn
        self._persist_live_blocks(session_id=session_id)

        return answer

    # ── Auxiliary ingestion (tool_use / tool_result / mid-turn blocks) ───
    #
    # pre_request / post_response cover the primary user↔assistant arc.
    # Tool-round-trip turns also carry material that must be remembered:
    # the `tool_use` the assistant emitted and the `tool_result` the
    # runtime sent back. The PII proxy classifier already tags them; we
    # just need a channel to store them without kicking off recall / a
    # new step. This method is that channel.

    def ingest_turn(self, items: list, session_id: str = "") -> int:
        """Ingest auxiliary turn items (tool_use / tool_result / etc.) into
        the live archive with their origin tag preserved.

        Each item is a dict with:
          {"text": str, "origin": str}

        Supported origins: "user", "user_image", "assistant_text",
        "tool_use", "tool_result", "tool", "system". Unknown origins
        default to "user" semantics.

        No recall, no prompt build, no step advance — this is a pure
        ingestion call, intended for the mechanical hand-offs that
        surround a tool round-trip. Returns the number of blocks added.
        """
        if not items:
            return 0

        added = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            text = item.get("text") or ""
            origin = (item.get("origin") or "user").strip() or "user"
            if not text.strip():
                continue
            # Cap runaway tool output so a single tool_result can't
            # blow out the live archive. TCMM chunks internally but
            # very large blocks still churn the cache.
            if len(text) > 8000:
                text = text[:8000] + "\n…[truncated by ingest_turn]"

            pclass, is_asst, arch_default, tcmm_origin, tcmm_source = self._ORIGIN_MAP.get(
                origin, self._ORIGIN_MAP["user"]
            )
            archivable = True if arch_default is None else arch_default

            # Build block.source. For tool_use / tool_result, embed
            # (tool_name, param_hash, tool_use_id) so a scan can group
            # by tool command+params to spot:
            #   - duplicate invocations of the same (tool_name, params)
            #     that returned different results → regression / state drift
            #   - orphan tool_use with no matching tool_result
            #   - orphan tool_result with no matching tool_use
            #
            # Format:
            #   tool_use:<sess>:<tool_name>:<param_hash>:<tool_use_id>
            #   tool_result:<sess>:<tool_name>:<param_hash>:<tool_use_id>
            #   <tag>:<sess>        (non-tool blocks — unchanged)
            sess = session_id[:12] if session_id else ""
            if origin in ("tool_use", "tool_result", "tool"):
                tool_name = str(item.get("tool_name") or "unknown").replace(":", "_")
                param_hash = str(item.get("param_hash") or "").replace(":", "_")
                tool_use_id = str(item.get("tool_use_id") or "").replace(":", "_")
                src_tag = f"{tcmm_source}:{sess}:{tool_name}:{param_hash}:{tool_use_id}"
            else:
                src_tag = f"{tcmm_source}:{sess}" if sess else tcmm_source

            created = self.tcmm.add_new_block(
                text=text,
                priority_class=pclass,
                is_assistant=is_asst,
                archivable=archivable,
                source=src_tag,
            )
            if created is not None:
                try:
                    created.origin = tcmm_origin
                except Exception:
                    pass

                # For tool blocks, capture structured params / result
                # as first-class data, not just the hash + free text.
                # Two places:
                #   (a) self._block_tool_meta[block_id]  — fast access
                #       while the block is live / before archive flush.
                #   (b) archive entry["tool_meta"]       — persistent,
                #       survives consolidation, queryable later.
                if origin in ("tool_use", "tool_result", "tool"):
                    meta = {
                        "role": origin,
                        "tool_name": str(item.get("tool_name") or "unknown"),
                        "tool_use_id": str(item.get("tool_use_id") or ""),
                        "param_hash": str(item.get("param_hash") or ""),
                        "params": item.get("params"),
                        "result": item.get("result"),
                    }
                    try:
                        self._block_tool_meta[created.id] = meta
                    except Exception:
                        pass
                    # Stamp onto the archive entry. add_new_block with
                    # archivable=True creates the archive entry inline,
                    # so origin_archive_id is non-None here.
                    #
                    # _ensure_archive_entry in archive.py creates entries
                    # via a hardcoded placeholder Block whose source,
                    # priority_class, origin and is_assistant all carry
                    # default values ("atomic_write", "USER", None,
                    # False) — NOT the values we pass to add_new_block.
                    # We need to correct them explicitly so archive-only
                    # scans (after live-block LRU eviction) still find
                    # our tool-pairing source tag.
                    try:
                        aid = getattr(created, "origin_archive_id", None)
                        if aid is not None:
                            entry = self.tcmm.archive.get(aid)
                            if isinstance(entry, dict):
                                entry["tool_meta"] = meta
                                # Sync top-level fields to match the real
                                # block, so find_tool_invocations can
                                # pair blocks from the archive alone.
                                entry["source"] = src_tag
                                entry["priority_class"] = pclass
                                entry["origin"] = tcmm_origin
                                entry["is_assistant"] = is_asst
                                # Lineage-link this tool block back to
                                # the current turn's user message so
                                # dream-engine can reassemble the whole
                                # "user ask → assistant tool call →
                                # tool result → assistant answer" unit
                                # as one canonical state. Without this
                                # each tool block is a self-rooted
                                # orphan (assessed 23 Apr: 0/301 rows
                                # had lineage.parents set). The anchor
                                # aid is recorded by flush_pending_user
                                # _ingest on this TCMM instance.
                                parent_aid = getattr(
                                    self.tcmm, "_current_turn_user_aid", None
                                )
                                if parent_aid is not None:
                                    _lin = entry.get("lineage") or {}
                                    if not isinstance(_lin, dict):
                                        try:
                                            _lin = dict(_lin)
                                        except Exception:
                                            _lin = {}
                                    _lin["parents"] = [int(parent_aid)]
                                    # Root stays as the turn's anchor so
                                    # all blocks in the turn share a
                                    # lineage.root — dream can cluster
                                    # them with a single GROUP BY root.
                                    _lin["root"] = int(parent_aid)
                                    entry["lineage"] = _lin
                                    entry["lineage_root"] = int(parent_aid)
                                try:
                                    self.tcmm.archive[aid] = entry
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.debug(f"[TCMM] tool_meta attach failed: {e}")

                added += 1

        if added:
            self.tcmm.flush_current_block()
            # Persist so a crash mid-turn doesn't lose the tool round-trip record.
            self._persist_live_blocks(session_id=session_id)

        logger.info(f"[TCMM] ingest_turn session={session_id[:12] or 'none'} added={added}/{len(items)}")
        return added

    # ── Tool-call pairing / orphan detection ─────────────────────────────
    #
    # block.source for tool blocks is encoded as:
    #   tool_use:<session>:<tool_name>:<param_hash>:<tool_use_id>
    #   tool_result:<session>:<tool_name>:<param_hash>:<tool_use_id>
    #
    # Pairing key is (tool_name, param_hash) — meaningful identity of a
    # tool call. Two independent invocations of the same tool with the
    # same params share this key, so we can spot flaky tools, cache
    # misses, or state drift (same call, different answer). The
    # tool_use_id is retained as a secondary identifier for exact
    # per-invocation pairing within a single turn.

    @staticmethod
    def _parse_tool_source(src):
        """Parse block.source for tool blocks.

        Returns dict {tag, session, tool_name, param_hash, tool_use_id}
        or None if the source isn't a tool block's structured format.
        Tolerates older 4-field sources (no param_hash) from pre-
        upgrade blocks — returns param_hash="" for those.
        """
        if not isinstance(src, str):
            return None
        parts = src.split(":")
        if len(parts) < 4:
            return None
        tag = parts[0]
        if tag not in ("tool_use", "tool_result", "tool"):
            return None
        # Expected 5-field layout; tolerate older 4-field layout.
        if len(parts) >= 5:
            return {
                "tag": tag,
                "session": parts[1],
                "tool_name": parts[2],
                "param_hash": parts[3],
                "tool_use_id": ":".join(parts[4:]),
            }
        return {
            "tag": tag,
            "session": parts[1],
            "tool_name": parts[2],
            "param_hash": "",
            "tool_use_id": parts[3],
        }

    def find_tool_invocations(
        self,
        session_id: str = "",
        tool_name: str = "",
        param_hash: str = "",
    ) -> dict:
        """Group tool_use / tool_result blocks by (tool_name, param_hash).

        Every *distinct command* (same tool + same inputs) becomes one
        invocation group. Each group lists every call+result pair that
        hit that command, so you can compare outcomes across repeated
        calls — the "marrying up calls that don't work" signal.

        Filters (all optional):
            session_id — restrict to one conversation
            tool_name  — restrict to one tool, e.g. "read_file"
            param_hash — restrict to one command shape, e.g. the hash
                         for read_file({"path":"/etc/passwd"})

        Returns:
            {
              "invocations": [
                {
                  "tool_name": str,
                  "param_hash": str,
                  "params": dict | None,   # structured input (all calls
                                           # in the group share this)
                  "calls": [
                    {
                      "tool_use_id": str,
                      "use_block_id": int,
                      "result_block_id": int | None,
                      "result": str | list | None,   # structured output
                    }, …
                  ],
                  "has_orphan": bool,      # any call missing a result?
                }, …
              ],
              "orphan_uses":    [{tool_name, param_hash, tool_use_id, block_id, params, result}],
              "orphan_results": [{tool_name, param_hash, tool_use_id, block_id, params, result}],
            }
        """
        sess_filter = session_id[:12] if session_id else None
        name_filter = tool_name or None
        hash_filter = param_hash or None

        def _match(parsed):
            if sess_filter and parsed["session"] != sess_filter:
                return False
            if name_filter and parsed["tool_name"] != name_filter:
                return False
            if hash_filter and parsed["param_hash"] != hash_filter:
                return False
            return True

        # uses / results keyed by tool_use_id for exact pair matching,
        # but we also remember (tool_name, param_hash) for grouping.
        uses: dict = {}     # tool_use_id -> {name, phash, block_id}
        results: dict = {}

        def _meta_for(block_id, entry_meta):
            """Best-effort lookup of structured tool_meta.

            Archive entry (entry_meta) wins when present; otherwise fall
            back to the live-block cache populated by ingest_turn.
            Either path returns the dict we stored at ingest time —
            {tool_name, tool_use_id, param_hash, params, result, role}.
            """
            if isinstance(entry_meta, dict) and entry_meta:
                return entry_meta
            return self._block_tool_meta.get(block_id)

        def _consume(tag, parsed, block_id, meta):
            if not _match(parsed):
                return
            rec = {
                "tool_name": parsed["tool_name"],
                "param_hash": parsed["param_hash"],
                "tool_use_id": parsed["tool_use_id"],
                "block_id": block_id,
                "params": (meta or {}).get("params") if meta else None,
                "result": (meta or {}).get("result") if meta else None,
            }
            (uses if tag == "tool_use" else results)[parsed["tool_use_id"]] = rec

        # Live blocks first. Their tool_meta lives in the adapter cache
        # keyed by block id (the archive entry write-back may also have
        # happened, but we prefer the live cache here to avoid a lookup).
        for b in self.tcmm.live_blocks:
            parsed = self._parse_tool_source(getattr(b, "source", ""))
            if parsed:
                bid = getattr(b, "id", None)
                _consume(parsed["tag"], parsed, bid, self._block_tool_meta.get(bid))

        # Archive entries (fall back if live doesn't have the tuid).
        # entry["tool_meta"] is the authoritative structured record for
        # everything that has been flushed to the archive.
        try:
            for aid, entry in self.tcmm.archive.items():
                if not isinstance(entry, dict):
                    continue
                parsed = self._parse_tool_source(entry.get("source"))
                if not parsed:
                    continue
                if not _match(parsed):
                    continue
                tuid = parsed["tool_use_id"]
                bucket = uses if parsed["tag"] == "tool_use" else results
                if tuid in bucket:
                    # Live path already captured it with its meta.
                    continue
                meta = _meta_for(aid, entry.get("tool_meta"))
                bucket[tuid] = {
                    "tool_name": parsed["tool_name"],
                    "param_hash": parsed["param_hash"],
                    "tool_use_id": tuid,
                    "block_id": aid,
                    "params": (meta or {}).get("params") if meta else None,
                    "result": (meta or {}).get("result") if meta else None,
                }
        except Exception:
            pass

        # Group uses by (tool_name, param_hash). Each group is one
        # invocation-shape — all calls to that exact command.
        groups: dict = {}   # (name, phash) -> list of call records
        for tuid, u in uses.items():
            key = (u["tool_name"], u["param_hash"])
            groups.setdefault(key, []).append(u)

        invocations = []
        orphan_uses = []
        for (name, phash), call_list in groups.items():
            # All calls in this group share the same (name, phash), so
            # params are identical; lift the first non-null as the
            # group-level `params` for easy display / aggregation.
            group_params = next(
                (u.get("params") for u in call_list if u.get("params") is not None),
                None,
            )
            group_entry = {
                "tool_name": name,
                "param_hash": phash,
                "params": group_params,
                "calls": [],
                "has_orphan": False,
            }
            for u in call_list:
                tuid = u["tool_use_id"]
                r = results.get(tuid)
                if r is None:
                    group_entry["has_orphan"] = True
                    orphan_uses.append(u)
                group_entry["calls"].append({
                    "tool_use_id": tuid,
                    "use_block_id": u["block_id"],
                    "result_block_id": r["block_id"] if r else None,
                    # Per-call structured result — key signal for
                    # comparing repeated invocations. Same params,
                    # different results here = drift/flake.
                    "result": r.get("result") if r else None,
                })
            invocations.append(group_entry)

        orphan_results = [
            r for tuid, r in results.items() if tuid not in uses
        ]

        return {
            "invocations": invocations,
            "orphan_uses": orphan_uses,
            "orphan_results": orphan_results,
        }

    # Back-compat alias for callers that grew up on find_tool_pairs
    find_tool_pairs = find_tool_invocations

    # ── Live Block Persistence ───────────────────────────────────────────

    def _block_to_dict(self, b) -> dict:
        """Serialize a Block to a JSON-safe dict."""
        return {
            "id": b.id,
            "text": b.text,
            "token_count": b.token_count,
            "fingerprint": b.fingerprint,
            "created_step": b.created_step,
            "last_used_step": b.last_used_step,
            "heat": b.heat,
            "origin_archive_id": b.origin_archive_id,
            "source": b.source,
            "last_heat_source": b.last_heat_source,
            "priority_class": b.priority_class,
            "volatility": b.volatility,
            "protected_until_step": b.protected_until_step,
            "last_heat_update_step": b.last_heat_update_step,
            "entropy_score": b.entropy_score,
            "lineage_root": getattr(b, "lineage_root", None),
            "is_assistant": getattr(b, "is_assistant", False),
            "is_open": getattr(b, "is_open", False),
            "recallable": getattr(b, "recallable", True),
            "archivable": getattr(b, "archivable", True),
            "times_in_prompt": getattr(b, "times_in_prompt", 0),
            # Provenance tag from the PII proxy classifier
            # ("user", "tool_use", "tool_result", …). Persisted so a
            # restart can reconstruct the role of each saved block.
            "origin": getattr(b, "origin", None),
            # Structured tool call metadata (params, result, etc.).
            # Authoritative copy is on the archive entry, but we also
            # snapshot it here so a crash before archive flush doesn't
            # lose the round-trip data.
            "tool_meta": self._block_tool_meta.get(b.id),
        }

    def _dict_to_block(self, d: dict):
        """Deserialize a dict back to a Block."""
        b = Block(
            id=d["id"],
            text=d["text"],
            token_count=d["token_count"],
            fingerprint=d["fingerprint"],
            created_step=d["created_step"],
            last_used_step=d["last_used_step"],
            heat=d["heat"],
            origin_archive_id=d.get("origin_archive_id"),
            source=d.get("source", ""),
            last_heat_source=d.get("last_heat_source", ""),
            priority_class=d.get("priority_class", ""),
            volatility=d.get("volatility", 1.0),
            protected_until_step=d.get("protected_until_step", 0),
            last_heat_update_step=d.get("last_heat_update_step", 0),
            entropy_score=d.get("entropy_score", 0.5),
            lineage_root=d.get("lineage_root"),
            is_assistant=d.get("is_assistant", False),
        )
        b.is_open = d.get("is_open", False)
        b.recallable = d.get("recallable", True)
        b.archivable = d.get("archivable", True)
        b.times_in_prompt = d.get("times_in_prompt", 0)
        # Restore proxy-classifier origin if present in the snapshot.
        _origin = d.get("origin")
        if _origin is not None:
            try:
                b.origin = _origin
            except Exception:
                pass
        # Restore tool_meta into the adapter cache (not the Block — its
        # __slots__ don't include it). Keyed by block id so subsequent
        # find_tool_invocations calls can find it.
        _meta = d.get("tool_meta")
        if isinstance(_meta, dict):
            try:
                self._block_tool_meta[b.id] = _meta
            except Exception:
                pass
        return b

    def _persist_live_blocks(self, session_id: str = ""):
        """Save live blocks to disk, keyed by session ID."""
        try:
            # Save session-specific file
            sid = session_id[:12] if session_id else "default"
            session_file = os.path.join(self._live_dir, f"session_{sid}.json")
            blocks = [self._block_to_dict(b) for b in self.tcmm.live_blocks
                      if not getattr(b, "is_open", False)]

            data = {
                "session_id": session_id,
                "saved_at": time.time(),
                "current_step": self.tcmm.current_step,
                "next_block_id": self.tcmm.next_block_id,
                "blocks": blocks,
            }
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Also save a combined "latest" snapshot
            latest_file = os.path.join(self._live_dir, "_latest.json")
            with open(latest_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"[TCMM] Persisted {len(blocks)} live blocks (session={sid})")
        except Exception as e:
            logger.error(f"[TCMM] Failed to persist live blocks: {e}")

    def _restore_live_blocks(self):
        """Restore live blocks from the latest snapshot on startup."""
        latest_file = os.path.join(self._live_dir, "_latest.json")
        if not os.path.exists(latest_file):
            return

        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            blocks = data.get("blocks", [])
            if not blocks:
                return

            restored = 0
            for d in blocks:
                b = self._dict_to_block(d)
                # Check for duplicate IDs
                if any(lb.id == b.id for lb in self.tcmm.live_blocks):
                    continue
                self.tcmm.live_blocks.append(b)
                restored += 1

            # Restore counters
            saved_step = data.get("current_step", 0)
            if saved_step > self.tcmm.current_step:
                self.tcmm.current_step = saved_step
            saved_bid = data.get("next_block_id", 0)
            if saved_bid > self.tcmm.next_block_id:
                self.tcmm.next_block_id = saved_bid

            logger.info(f"[TCMM] Restored {restored} live blocks from disk")
        except Exception as e:
            logger.error(f"[TCMM] Failed to restore live blocks: {e}")

    # ── Tool Integration ──────────────────────────────────────────────────

    def save_tool_output(self, tool_name: str, output: str):
        """
        Save MCP tool output to TCMM memory.

        Call this when a tool returns a result that should be remembered
        (e.g., analysis results, file contents, search results).

        Args:
            tool_name: Name of the MCP tool (e.g., "read_file", "google_search")
            output: The tool's output text
        """
        if not output or len(output.strip()) < 10:
            return

        # Truncate very large tool outputs
        max_tool_tokens = 2000
        if len(output.split()) > max_tool_tokens:
            output = " ".join(output.split()[:max_tool_tokens]) + "\n[... truncated]"

        self.tcmm.add_new_block(
            text=f"[TOOL:{tool_name}] {output}",
            priority_class="TOOL",
            is_assistant=False,
            archivable=True,
        )
        self.tcmm.flush_current_block()

    # File ingestion removed — files are now tool artifacts, captured
    # on the archive via the proxy's tool_use/tool_result pairing with
    # structured params + result on block.tool_meta.

    # ── Dream Consolidation ───────────────────────────────────────────────

    def run_dream_cycle(self):
        """
        Run a dream consolidation cycle.

        Call this periodically (e.g., via scheduled task) to consolidate
        raw memory blocks into structured knowledge nodes.
        """
        if hasattr(self.tcmm, "dream_engine"):
            try:
                self.tcmm.dream_engine.run_cycle()
                logger.info("[TCMM] Dream consolidation cycle completed")
            except Exception as e:
                logger.error(f"[TCMM] Dream cycle failed: {e}")
        else:
            logger.warning("[TCMM] Dream engine not available")

    # ── Status & Diagnostics ──────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get TCMM memory status for diagnostics."""
        return {
            "live_blocks": len(self.tcmm.live_blocks),
            "shadow_blocks": len(self.tcmm.shadow_blocks),
            "archive_blocks": len(self.tcmm.archive),
            "dream_nodes": len(getattr(self.tcmm, "dream_archive", {})),
            "current_step": self.tcmm.current_step,
            "last_recalled": len(self._last_recalled_ids),
        }

    # ── Prompt Extraction Helper ──────────────────────────────────────────

    @staticmethod
    def extract_user_message(messages: list) -> str:
        """
        Extract the latest user message from LibreChat's messages array.

        LibreChat sends the full conversation history. We only need the latest
        user message — TCMM manages its own history.

        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}]

        Returns:
            str: The latest user message text, or empty string.
        """
        if not messages:
            return ""

        # Walk backwards to find the last user message
        for msg in reversed(messages):
            role = msg.get("role", "")
            if role == "user":
                content = msg.get("content", "")
                # Handle multimodal content (list of parts)
                if isinstance(content, list):
                    text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                    return " ".join(text_parts).strip()
                return str(content).strip()

        return ""

    @staticmethod
    def build_single_message(prompt: str) -> list:
        """
        Convert TCMM's built prompt into an OpenAI-compatible messages array.

        TCMM's build_prompt() returns a single string with system prompt + memory + task.
        The Gemini OpenAI-compatible API expects a messages array.

        Args:
            prompt: The full prompt from pre_request()

        Returns:
            list: Messages array [{"role": "user", "content": prompt}]
        """
        return [{"role": "user", "content": prompt}]
