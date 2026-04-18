"""
TCMM HTTP Service for Veilguard.

Thin FastAPI wrapper around TCMM core. Each LibreChat conversation gets its own
TCMM instance (namespace), with shared NLP/embedding models across sessions.

Runs on the Windows host (not Docker) because TCMM needs ONNX, FAISS, spaCy.

Start: python mcp-tools/tcmm-service/server.py
"""

import logging
import os
import sys
import time
import threading
from collections import OrderedDict
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# ── Auto-detect GPU availability ──────────────────────────────────────────────
def _gpu_available():
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        t = torch.zeros(1, device="cuda")
        del t
        torch.cuda.empty_cache()
        return True
    except Exception:
        return False

if not _gpu_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TCMM_SKIP_NLP_GPU"] = "1"
    print("[TCMM-SVC] GPU unavailable (locked/dreaming) — running on CPU")
else:
    print("[TCMM-SVC] GPU available — using GPU")

# ── Resolve TCMM path ────────────────────────────────────────────────────────
TCMM_ROOT = os.environ.get(
    "TCMM_ROOT",
    r"C:\Users\rudol\.gemini\antigravity\tcmm\TCMM"
)
if TCMM_ROOT not in sys.path:
    sys.path.insert(0, TCMM_ROOT)

# Data directory for TCMM persistence
DATA_DIR = os.environ.get(
    "TCMM_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "tcmm-data")
)
os.makedirs(DATA_DIR, exist_ok=True)
os.chdir(DATA_DIR)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [TCMM-SVC] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("tcmm-service")

# ── Load TCMM ────────────────────────────────────────────────────────────────

logger.info(f"Loading TCMM from {TCMM_ROOT}...")
logger.info(f"Data directory: {DATA_DIR}")

from adapters.veilguard_adapter import VeilguardTCMM

SYSTEM_PROMPT = os.environ.get("TCMM_SYSTEM_PROMPT", """You are Veilguard, a Phishield AI assistant with persistent memory.
You remember information from all previous conversations.
Use your memory blocks to provide contextual, personalized responses.
When you learn new facts about the user or their work, remember them for future reference.""")

MAX_SESSIONS = int(os.environ.get("TCMM_MAX_SESSIONS", "10"))

# ── Shared Resources ─────────────────────────────────────────────────────────
# NLP adapter and embedding model are expensive to load (~5s, ~2GB VRAM).
# We load them ONCE and share across all session instances.

_shared_nlp = None
_shared_embedder = None


def _init_shared_resources():
    """Load NLP adapter and embedder once, shared across all sessions.

    Supports two backends:
        NLP_BACKEND=local   → LocalNLPAdapter (Gemma E2B + SpaCy, needs GPU)
        NLP_BACKEND=vertex  → VertexNLPAdapter (Gemini Flash API, no GPU)

        EMBED_BACKEND=local  → LocalEmbeddingAdapter (SentenceTransformer, needs GPU)
        EMBED_BACKEND=vertex → VertexEmbeddingAdapter (text-embedding-005 API, no GPU)
    """
    global _shared_nlp, _shared_embedder

    nlp_backend = os.environ.get("NLP_BACKEND", "local").lower()
    embed_backend = os.environ.get("EMBED_BACKEND", "local").lower()
    vertex_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    vertex_region = os.environ.get("VERTEX_REGION", "us-central1")

    # NLP adapter
    if nlp_backend == "vertex":
        from adapters.vertex_nlp_adapter import VertexNLPAdapter
        logger.info(f"Loading Vertex AI NLP adapter (Gemini Flash, project={vertex_project})...")
        _shared_nlp = VertexNLPAdapter(project_id=vertex_project, region=vertex_region)
    else:
        from adapters.nlp_adapter import LocalNLPAdapter
        logger.info("Loading local NLP adapter (Gemma E2B + SpaCy)...")
        _shared_nlp = LocalNLPAdapter()

    # Embedding adapter
    if embed_backend == "vertex":
        from adapters.vertex_embedding_adapter import VertexEmbeddingAdapter
        logger.info(f"Loading Vertex AI embedding adapter (text-embedding-005, project={vertex_project})...")
        _shared_embedder = VertexEmbeddingAdapter(project_id=vertex_project, region=vertex_region)
    else:
        from adapters.local_adapter import LocalEmbeddingAdapter
        logger.info("Loading local embedding adapter (SentenceTransformer)...")
        _shared_embedder = LocalEmbeddingAdapter()

    logger.info(f"Shared resources ready (nlp={nlp_backend}, embed={embed_backend})")


# ── Per-Session Stats ────────────────────────────────────────────────────────
# Anthropic pricing (claude-sonnet-4-6)
_COST_INPUT = 0.003
_COST_CACHE_WRITE = 0.003 * 1.25   # 25% premium
_COST_CACHE_READ = 0.003 * 0.10    # 90% discount
_COST_OUTPUT = 0.015
_CONTEXT_WINDOW = 190000            # naive compaction trigger
_COMPACT_SUMMARY = 3000             # summary size after compaction


def _new_session_stats() -> dict:
    """Create a fresh session stats dict."""
    return {
        "turns": 0,
        "tcmm_input_tokens": 0,
        "naive_input_tokens": 0,
        "output_tokens": 0,
        "naive_cost_no_cache": 0.0,
        "naive_cost_cached": 0.0,
        "tcmm_cost_no_cache": 0.0,
        "tcmm_cost_cached": 0.0,
        "output_cost": 0.0,
        "compactions": 0,
        "derived_skipped": 0,
        "novel_added": 0,
        # Per-session tracking state
        "_prev_naive_input": 0,
        "_prev_tcmm_stable": 0,
        "_prev_live_count": 0,
        "_naive_history_chars": 0,
        "_all_messages": [],
        "_token_history": [],
    }


# ── Session Pool ─────────────────────────────────────────────────────────────

STORAGE_BACKEND = os.environ.get("TCMM_STORAGE", "lance").lower()
VECTOR_BACKEND = os.environ.get("TCMM_VECTOR", "lance").lower()
SPARSE_BACKEND = os.environ.get("TCMM_SPARSE", "lance").lower()
LANCE_DB_NAME = os.environ.get("TCMM_LANCE_DB", "veilguard")


class SessionPool:
    """LRU cache of VeilguardTCMM instances, one per conversation_id.

    With LanceDB backend, all sessions share a single database.
    Isolation is via namespace (conversation_id) and user_id columns.
    """

    def __init__(self, max_size: int = 10):
        self._max_size = max_size
        self._instances: OrderedDict[str, VeilguardTCMM] = OrderedDict()
        self._stats: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._active_session: Optional[str] = None  # for heatmap/stats endpoints

    def get(self, conversation_id: str, user_id: str = "") -> VeilguardTCMM:
        """Get or create a TCMM instance for the given conversation."""
        sid = self._normalize_id(conversation_id)

        with self._lock:
            if sid in self._instances:
                # Move to end (most recently used)
                self._instances.move_to_end(sid)
                self._active_session = sid
                return self._instances[sid]

            # Create new instance with namespace
            logger.info(f"[POOL] Creating new session: {sid} (user={user_id[:12]})")
            start = time.time()

            # Shared LanceDB: one DB for all sessions, namespace=conversation_id
            _is_db_storage = STORAGE_BACKEND in ("lance", "lancedb", "sqlite")
            if _is_db_storage:
                # Shared DB directory — all sessions in one database
                shared_data_dir = os.path.join(DATA_DIR, LANCE_DB_NAME)
                os.makedirs(shared_data_dir, exist_ok=True)
                ns = {"user_id": user_id or "default", "namespace": sid}
                data_dir = shared_data_dir
            else:
                # Legacy: per-session directory
                data_dir = os.path.join(DATA_DIR, "sessions", sid)
                os.makedirs(data_dir, exist_ok=True)
                ns = None

            instance = VeilguardTCMM(
                system_prompt=SYSTEM_PROMPT,
                embedder=_shared_embedder,
                nlp_adapter=_shared_nlp,
                data_dir=data_dir,
                namespace=ns,
                storage=STORAGE_BACKEND,
                vector_store=VECTOR_BACKEND,
                sparse_store=SPARSE_BACKEND,
            )

            elapsed = time.time() - start
            status = instance.get_status()
            logger.info(
                f"[POOL] Session {sid} ready in {elapsed:.1f}s — "
                f"archive={status['archive_blocks']}, "
                f"live={status['live_blocks']}, "
                f"storage={STORAGE_BACKEND}"
            )

            # Evict oldest if at capacity
            if len(self._instances) >= self._max_size:
                evicted_id, evicted = self._instances.popitem(last=False)
                self._stats.pop(evicted_id, None)
                logger.info(f"[POOL] Evicted oldest session: {evicted_id}")

            self._instances[sid] = instance
            self._stats[sid] = _new_session_stats()
            self._active_session = sid
            return instance

    def get_stats(self, conversation_id: str) -> dict:
        """Get stats dict for a session, creating if needed."""
        sid = self._normalize_id(conversation_id)
        with self._lock:
            if sid not in self._stats:
                self._stats[sid] = _new_session_stats()
            return self._stats[sid]

    def get_active(self) -> tuple:
        """Get the most recently used (session_id, instance, stats)."""
        with self._lock:
            sid = self._active_session
            if sid and sid in self._instances:
                return sid, self._instances[sid], self._stats.get(sid, _new_session_stats())
            # Fallback: last item
            if self._instances:
                sid = next(reversed(self._instances))
                return sid, self._instances[sid], self._stats.get(sid, _new_session_stats())
            return None, None, _new_session_stats()

    def list_sessions(self) -> list:
        """List all active sessions with basic info."""
        with self._lock:
            result = []
            for sid, instance in self._instances.items():
                stats = self._stats.get(sid, {})
                status = instance.get_status()
                result.append({
                    "session_id": sid,
                    "turns": stats.get("turns", 0),
                    "archive_blocks": status.get("archive_blocks", 0),
                    "live_blocks": status.get("live_blocks", 0),
                })
            return result

    @staticmethod
    def _normalize_id(conversation_id: str) -> str:
        """Normalize conversation ID to a safe directory name."""
        sid = (conversation_id or "default").strip()
        # Keep first 24 chars, replace unsafe chars
        sid = sid[:24].replace("/", "_").replace("\\", "_").replace(":", "_")
        return sid or "default"


# ── Global Pool ──────────────────────────────────────────────────────────────

pool = SessionPool(max_size=MAX_SESSIONS)


# ── FastAPI ──────────────────────────────────────────────────────────────────

app = FastAPI(title="TCMM Memory Service")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class PreRequestBody(BaseModel):
    user_message: str
    conversation_id: str = ""
    user_id: str = ""
    recall_only: bool = False  # When True: recall from existing session, don't ingest query
    # Origin tag from the PII proxy's classify_message_origin().
    # Passed through to TCMM so the stored block records the source role
    # ("user" | "user_image" | "tool_result" | "tool"). Default is "user"
    # to preserve backward compatibility with callers that don't set it.
    origin: str = "user"


class PostResponseBody(BaseModel):
    raw_output: str
    conversation_id: str = ""
    user_id: str = ""
    # Origin tag for the assistant-side block. Usually "assistant_text";
    # set to "tool_use" when the assistant reply is a tool invocation.
    origin: str = "assistant_text"


class IngestTurnBody(BaseModel):
    """Auxiliary ingestion (tool_use / tool_result / mid-turn blocks).

    Called by the PII proxy when it sees a tool round-trip — both sides
    of the hand-off need to be persisted in the archive, even though they
    are not primary user↔assistant turns. `items` is a small list of
    {text, origin} objects; see the PII proxy's classify_message_origin
    for the recognised origin values.
    """
    conversation_id: str = ""
    user_id: str = ""
    items: list = []


import asyncio
_tcmm_lock = asyncio.Lock()


@app.on_event("startup")
async def startup():
    logger.info("Initializing shared resources...")
    start = time.time()
    _init_shared_resources()
    elapsed = time.time() - start
    logger.info(f"Shared resources ready in {elapsed:.1f}s (max_sessions={MAX_SESSIONS})")


@app.get("/health")
async def health():
    if _shared_nlp is None:
        return {"status": "initializing"}

    sid, instance, stats = pool.get_active()
    if instance is None:
        return {
            "status": "ok",
            "service": "tcmm-memory",
            "active_sessions": 0,
            "max_sessions": MAX_SESSIONS,
        }

    status = instance.get_status()
    return {
        "status": "ok",
        "service": "tcmm-memory",
        "active_session": sid,
        "active_sessions": len(pool._instances),
        "max_sessions": MAX_SESSIONS,
        **status,
    }


@app.post("/pre_request")
async def pre_request(body: PreRequestBody):
    """Called BEFORE PII redaction. Routes to per-session TCMM instance.

    When recall_only=True (sub-agent memory recall), only recall from existing
    session — don't ingest the query or create a new session if it doesn't exist.
    """
    if _shared_nlp is None:
        return {"error": "TCMM not initialized", "prompt": None}

    async with _tcmm_lock:
        try:
            # Recall-only mode: read from existing session, don't create new one
            if body.recall_only:
                sid = pool._normalize_id(body.conversation_id)
                with pool._lock:
                    instance = pool._instances.get(sid)
                if instance is None:
                    logger.info(f"[RECALL-ONLY] No session found for {sid}")
                    return {"prompt": "", "stats": {"recalled": 0}}
                # Just do recall without ingesting the query
                tcmm = instance.tcmm
                recalled_ids = tcmm.recall(body.user_message) or []
                if recalled_ids:
                    tcmm.stage_shadow_blocks(recalled_ids)
                prompt = instance._build_memory_context(body.user_message)
                logger.info(f"[RECALL-ONLY] session={sid} recalled={len(recalled_ids)} prompt={len(prompt)} chars")
                return {"prompt": prompt, "stats": {"recalled": len(recalled_ids)}}

            instance = pool.get(body.conversation_id, user_id=body.user_id)
            sess = pool.get_stats(body.conversation_id)

            start = time.time()
            prompt = instance.pre_request(
                body.user_message,
                session_id=body.conversation_id,
                origin=body.origin or "user",
            )
            elapsed = time.time() - start

            status = instance.get_status()
            logger.info(
                f"[PRE] session={pool._normalize_id(body.conversation_id)} "
                f"query='{body.user_message[:60]}...' "
                f"prompt_words={len(prompt.split())} "
                f"recalled={status['last_recalled']} "
                f"took={elapsed:.2f}s"
            )

            # ── Token & cost tracking (per-session) ──
            sess["turns"] += 1
            tcmm = instance.tcmm

            tcmm_input_tokens = len(prompt) // 4
            sess["tcmm_input_tokens"] += tcmm_input_tokens

            # Detect derived vs novel
            current_live_count = len(tcmm.live_blocks)
            live_changed = current_live_count != sess["_prev_live_count"]
            if sess["turns"] > 1 and not live_changed:
                sess["derived_skipped"] += 1
            elif sess["turns"] > 1:
                sess["novel_added"] += 1
            sess["_prev_live_count"] = current_live_count

            live_tokens = sum(len(getattr(b, "text", str(b))) for b in tcmm.live_blocks) // 4
            stable_tokens = 500 + live_tokens

            recalled_ids = getattr(instance, '_last_recalled_ids', set())
            recall_tokens = sum(
                len(e.get("text", "")) // 4
                for aid, e in tcmm.archive.items()
                if aid in recalled_ids
            ) if recalled_ids else 0
            variable_tokens = recall_tokens

            # TCMM no cache
            sess["tcmm_cost_no_cache"] += (tcmm_input_tokens / 1000) * _COST_INPUT

            # TCMM with cache
            prev_stable = sess["_prev_tcmm_stable"]
            if sess["turns"] == 1:
                sess["tcmm_cost_cached"] += (stable_tokens / 1000) * _COST_CACHE_WRITE
                sess["tcmm_cost_cached"] += (variable_tokens / 1000) * _COST_INPUT
            elif not live_changed and prev_stable > 0:
                sess["tcmm_cost_cached"] += (stable_tokens / 1000) * _COST_CACHE_READ
                sess["tcmm_cost_cached"] += (variable_tokens / 1000) * _COST_INPUT
            else:
                if prev_stable > 0 and len(tcmm.live_blocks) >= 20:
                    avg_blk = live_tokens // max(len(tcmm.live_blocks), 1)
                    changed = avg_blk * 2
                    cached_s = max(0, min(prev_stable, stable_tokens) - changed)
                else:
                    cached_s = max(0, prev_stable)
                uncached_s = max(0, stable_tokens - cached_s)
                sess["tcmm_cost_cached"] += (cached_s / 1000) * _COST_CACHE_READ
                sess["tcmm_cost_cached"] += (uncached_s / 1000) * _COST_CACHE_WRITE
                sess["tcmm_cost_cached"] += (variable_tokens / 1000) * _COST_INPUT

            sess["_prev_tcmm_stable"] = stable_tokens

            # Naive side
            sess["_all_messages"].append(body.user_message)
            sess["_naive_history_chars"] = sum(len(m) for m in sess["_all_messages"])
            naive_input_tokens = sess["_naive_history_chars"] // 4 + 500
            sess["naive_input_tokens"] += naive_input_tokens

            prev_naive = sess["_prev_naive_input"]
            if naive_input_tokens > _CONTEXT_WINDOW:
                sess["compactions"] += 1
                cc = (naive_input_tokens / 1000) * _COST_INPUT + (_COMPACT_SUMMARY / 1000) * _COST_OUTPUT
                sess["naive_cost_no_cache"] += cc
                sess["naive_cost_cached"] += cc
                sess["_all_messages"].clear()
                sess["_all_messages"].append("x" * (_COMPACT_SUMMARY * 4))
                sess["_naive_history_chars"] = sum(len(m) for m in sess["_all_messages"])
                naive_input_tokens = sess["_naive_history_chars"] // 4 + 500
                prev_naive = 0

            sess["naive_cost_no_cache"] += (naive_input_tokens / 1000) * _COST_INPUT

            if sess["turns"] == 1 or prev_naive == 0:
                sess["naive_cost_cached"] += (naive_input_tokens / 1000) * _COST_CACHE_WRITE
            else:
                cached = prev_naive
                new = max(0, naive_input_tokens - cached)
                sess["naive_cost_cached"] += (cached / 1000) * _COST_CACHE_READ
                sess["naive_cost_cached"] += (new / 1000) * _COST_CACHE_WRITE
            sess["_prev_naive_input"] = naive_input_tokens

            return {
                "prompt": prompt,
                "stats": {
                    "live_blocks": status["live_blocks"],
                    "shadow_blocks": status["shadow_blocks"],
                    "recalled": status["last_recalled"],
                    "elapsed_ms": int(elapsed * 1000),
                    "input_tokens": tcmm_input_tokens,
                    "session_id": pool._normalize_id(body.conversation_id),
                },
            }
        except Exception as e:
            logger.error(f"[PRE] Error: {e}", exc_info=True)
            return {"error": str(e), "prompt": None}


@app.post("/post_response")
async def post_response(body: PostResponseBody):
    """Called AFTER PII rehydration. Routes to per-session TCMM instance."""
    if _shared_nlp is None:
        return {"error": "TCMM not initialized", "answer": body.raw_output}

    async with _tcmm_lock:
        try:
            instance = pool.get(body.conversation_id, user_id=body.user_id)
            sess = pool.get_stats(body.conversation_id)

            start = time.time()
            answer = instance.post_response(
                body.raw_output,
                session_id=body.conversation_id,
                origin=body.origin or "assistant_text",
            )
            elapsed = time.time() - start

            status = instance.get_status()
            sid = pool._normalize_id(body.conversation_id)
            logger.info(
                f"[POST] session={sid} "
                f"answer_len={len(answer)} "
                f"step={status['current_step']} "
                f"archive={status['archive_blocks']} "
                f"took={elapsed:.2f}s"
            )

            output_tokens = len(answer) // 4
            sess["output_tokens"] += output_tokens
            output_cost = (output_tokens / 1000) * _COST_OUTPUT
            sess["output_cost"] += output_cost

            # Naive: assistant response always grows history
            sess["_all_messages"].append(answer)

            # Token history
            naive_total = sess["naive_cost_cached"] + sess["output_cost"]
            tcmm_total = sess["tcmm_cost_cached"] + sess["output_cost"]
            savings_cost = max(0, naive_total - tcmm_total)
            savings_pct = round(savings_cost * 100 / max(naive_total, 0.0001), 1)

            sess["_token_history"].append({
                "turn": sess["turns"],
                "step": status["current_step"],
                "naive_input": sess["naive_input_tokens"],
                "tcmm_input": sess["tcmm_input_tokens"],
                "output": output_tokens,
                "naive_cost": round(naive_total, 6),
                "tcmm_cost": round(tcmm_total, 6),
                "savings_pct": savings_pct,
            })
            if len(sess["_token_history"]) > 50:
                sess["_token_history"].pop(0)

            return {
                "answer": answer,
                "stats": {
                    "current_step": status["current_step"],
                    "archive_blocks": status["archive_blocks"],
                    "elapsed_ms": int(elapsed * 1000),
                    "output_tokens": output_tokens,
                    "session_id": sid,
                },
            }
        except Exception as e:
            logger.error(f"[POST] Error: {e}", exc_info=True)
            return {"error": str(e), "answer": body.raw_output}


@app.get("/tool_invocations")
async def tool_invocations(conversation_id: str = "", user_id: str = ""):
    """Group this session's tool_use / tool_result blocks by
    (tool_name, param_hash) — the identity of the *command* itself.

    This lets you spot:
      - the same command invoked multiple times (does it return the
        same thing each time, or has state drifted?)
      - tool_use blocks with no matching tool_result (call dropped)
      - tool_result blocks with no matching tool_use (envelope damage)

    Response shape:
        {
          "invocations": [
            {
              "tool_name", "param_hash",
              "calls": [{"tool_use_id", "use_block_id", "result_block_id"}, …],
              "has_orphan": bool,
            }, …
          ],
          "orphan_uses":    [{tool_name, param_hash, tool_use_id, block_id}],
          "orphan_results": [{tool_name, param_hash, tool_use_id, block_id}],
        }

    The proxy tags each tool block's `source` field as
        tool_use:<session>:<tool_name>:<param_hash>:<tool_use_id>
        tool_result:<session>:<tool_name>:<param_hash>:<tool_use_id>
    so the scan works without any schema changes.
    """
    if _shared_nlp is None:
        return {"error": "TCMM not initialized"}

    if not conversation_id:
        return {"error": "conversation_id required"}

    with pool._lock:
        sid = pool._normalize_id(conversation_id)
        instance = pool._instances.get(sid)
    if instance is None:
        return {
            "error": f"no session {sid}",
            "invocations": [],
            "orphan_uses": [],
            "orphan_results": [],
        }

    try:
        return instance.find_tool_invocations(session_id=conversation_id)
    except Exception as e:
        logger.error(f"[TOOL-INVOCATIONS] Error: {e}", exc_info=True)
        return {"error": str(e)}


# Back-compat alias — old clients may still be hitting /tool_pairs.
@app.get("/tool_pairs")
async def tool_pairs(conversation_id: str = "", user_id: str = ""):
    return await tool_invocations(conversation_id=conversation_id, user_id=user_id)


@app.get("/tool_invocations/{tool_name}")
async def tool_invocations_by_name(
    tool_name: str,
    conversation_id: str = "",
    user_id: str = "",
    param_hash: str = "",
):
    """Same as /tool_invocations but filtered to a single tool (and
    optionally a single param_hash — the exact command shape).

    Examples:
        /tool_invocations/read_file?conversation_id=abc
            → every call to read_file in that conversation, grouped
              by their param_hash (one group per distinct argument
              shape).
        /tool_invocations/read_file?conversation_id=abc&param_hash=c3b72e6607ba
            → only the specific command read_file({"path":"/etc/passwd"})
              (or whatever hashed to c3b72e6607ba) — handy for
              comparing multiple results of the same call.
    """
    if _shared_nlp is None:
        return {"error": "TCMM not initialized"}
    if not conversation_id:
        return {"error": "conversation_id required"}

    with pool._lock:
        sid = pool._normalize_id(conversation_id)
        instance = pool._instances.get(sid)
    if instance is None:
        return {
            "error": f"no session {sid}",
            "tool_name": tool_name,
            "param_hash": param_hash,
            "invocations": [],
            "orphan_uses": [],
            "orphan_results": [],
        }

    try:
        result = instance.find_tool_invocations(
            session_id=conversation_id,
            tool_name=tool_name,
            param_hash=param_hash,
        )
        # Echo the filter back so the caller can confirm what was applied.
        result["tool_name"] = tool_name
        if param_hash:
            result["param_hash"] = param_hash
        return result
    except Exception as e:
        logger.error(f"[TOOL-INVOCATIONS/{tool_name}] Error: {e}", exc_info=True)
        return {"error": str(e), "tool_name": tool_name}


@app.post("/ingest_turn")
async def ingest_turn(body: IngestTurnBody):
    """Ingest auxiliary turn items (tool_use / tool_result / etc.) into
    the per-session TCMM archive.

    The PII proxy calls this on tool-round-trip turns so the assistant's
    tool_use and the user-role tool_result are both recorded — they are
    NOT primary user/assistant turns, so they don't belong in
    pre_request / post_response. No recall, no prompt build, no step
    advance; pure ingestion.

    Returns: {"added": N, "requested": M}
    """
    if _shared_nlp is None:
        return {"error": "TCMM not initialized", "added": 0}

    items = body.items or []
    if not items:
        return {"added": 0, "requested": 0}

    async with _tcmm_lock:
        try:
            instance = pool.get(body.conversation_id, user_id=body.user_id)
            added = instance.ingest_turn(items, session_id=body.conversation_id)
            sid = pool._normalize_id(body.conversation_id)
            logger.info(
                f"[INGEST-TURN] session={sid} added={added}/{len(items)} "
                f"origins={[ (i or {}).get('origin') for i in items ]}"
            )
            return {"added": added, "requested": len(items), "session_id": sid}
        except Exception as e:
            logger.error(f"[INGEST-TURN] Error: {e}", exc_info=True)
            return {"error": str(e), "added": 0}


# ── Session Management Endpoints ─────────────────────────────────────────────

@app.get("/api/sessions")
async def list_sessions():
    """List all active TCMM sessions."""
    return {"sessions": pool.list_sessions(), "max_sessions": MAX_SESSIONS}


# ── Debug Endpoints ──────────────────────────────────────────────────────────

@app.get("/debug_persist")
async def debug_persist():
    """Debug: manually trigger live block persistence on active session."""
    sid, instance, _ = pool.get_active()
    if instance is None:
        return {"error": "No active session"}
    try:
        instance._persist_live_blocks(session_id=sid or "debug")
        live_dir = instance._live_dir
        files = os.listdir(live_dir) if os.path.exists(live_dir) else []
        return {
            "status": "ok",
            "session_id": sid,
            "live_dir": live_dir,
            "files": files,
            "live_blocks": len(instance.tcmm.live_blocks),
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.get("/debug_workers")
async def debug_workers():
    """Debug: check worker threads and NLP status."""
    sid, instance, _ = pool.get_active()
    if instance is None:
        return {"error": "No active session"}
    tcmm = instance.tcmm

    threads = {}
    for t in threading.enumerate():
        if "TCMM" in t.name:
            threads[t.name] = {"alive": t.is_alive(), "daemon": t.daemon}

    queues = {}
    for qname in ["semantic_queue", "semantic_embed_queue", "embedding_queue"]:
        q = getattr(tcmm, qname, None)
        if q is not None:
            queues[qname] = q.qsize()

    metrics = {}
    last_error = None
    if hasattr(tcmm, "_metrics"):
        metrics = {k: v for k, v in tcmm._metrics.items() if v != 0 and k != "semantic.last_error"}
        last_error = tcmm._metrics.get("semantic.last_error")

    nlp_status = {"shared": True, "type": type(_shared_nlp).__name__ if _shared_nlp else "None"}

    return {
        "session_id": sid,
        "threads": threads,
        "queues": queues,
        "metrics": metrics,
        "nlp": nlp_status,
        "last_semantic_error": last_error,
    }


@app.post("/dream")
async def trigger_dream():
    """Trigger a dream cycle on the active session."""
    sid, instance, _ = pool.get_active()
    if instance is None:
        return {"error": "No active session"}

    async with _tcmm_lock:
        try:
            tcmm = instance.tcmm
            archive_count = len(tcmm.archive)
            dream_count = len(getattr(tcmm, "dream_archive", {}))

            instance.run_dream_cycle()

            new_dream_count = len(getattr(tcmm, "dream_archive", {}))
            return {
                "status": "completed",
                "session_id": sid,
                "archive_blocks": archive_count,
                "dream_nodes_before": dream_count,
                "dream_nodes_after": new_dream_count,
            }
        except Exception as e:
            logger.error(f"Dream cycle failed: {e}")
            return {"error": str(e)}


@app.get("/api/memory_heatmap")
async def memory_heatmap():
    """Return memory blocks with heat scores for the active session."""
    sid, instance, sess = pool.get_active()
    if instance is None:
        return {"error": "No active session"}

    tcmm = instance.tcmm
    current_step = tcmm.current_step
    blocks = []

    # Live blocks
    live_count = len(tcmm.live_blocks)
    for i, block in enumerate(tcmm.live_blocks[-30:]):
        text = getattr(block, "text", str(block))[:100]
        role = getattr(block, "role", "unknown")
        position_ratio = (i + 1) / min(live_count, 30)
        heat = 0.5 + 0.5 * position_ratio
        blocks.append({
            "id": f"live-{i}",
            "tier": "live",
            "heat": round(heat, 2),
            "text": text,
            "role": role,
            "step": current_step - (min(live_count, 30) - i),
        })

    # Shadow blocks
    for i, block in enumerate(tcmm.shadow_blocks[-20:]):
        text = getattr(block, "text", str(block))[:100]
        aid = getattr(block, "origin_archive_id", None)
        block_id = getattr(block, "id", None)
        blocks.append({
            "id": f"shadow-{aid if aid is not None else block_id or i}",
            "tier": "shadow",
            "heat": 0.6,
            "text": text,
            "role": getattr(block, "role", "shadow"),
            "step": aid if aid is not None else (block_id if block_id is not None else i),
        })

    # Archive blocks
    for aid, entry in list(tcmm.archive.items())[-30:]:
        topics = entry.get("topics", [])
        entities = entry.get("entities", [])
        density = entry.get("density_score", 0)
        semantic_links = len(entry.get("semantic_links", {}))
        entity_links = len(entry.get("entity_links", {}))
        topic_links = len(entry.get("topic_links", {}))

        link_score = min(1.0, (semantic_links + entity_links + topic_links) / 15)
        density_score = min(1.0, density / 10) if density else 0
        recency = max(0, 1.0 - (current_step - int(aid)) / max(current_step, 1))
        heat = 0.1 + 0.3 * density_score + 0.3 * link_score + 0.3 * recency

        text = entry.get("semantic_text", entry.get("text", ""))[:100]
        blocks.append({
            "id": f"archive-{aid}",
            "tier": "archive",
            "heat": round(min(1.0, heat), 2),
            "text": text,
            "role": entry.get("origin", "unknown"),
            "step": int(aid),
            "topics": topics[:3],
            "entities": entities[:3],
            "links": semantic_links + entity_links + topic_links,
        })

    # Token stats
    total_live_chars = sum(len(getattr(b, "text", str(b))) for b in tcmm.live_blocks)
    total_archive_chars = sum(len(e.get("text", "")) for e in tcmm.archive.values())

    last_recalled_ids = getattr(instance, '_last_recalled_ids', set())
    recalled_chars = sum(
        len(e.get("text", ""))
        for aid, e in tcmm.archive.items()
        if aid in last_recalled_ids
    ) if last_recalled_ids else 0

    output_cost = sess.get("output_cost", 0)
    naive_total_cost = sess["naive_cost_cached"] + output_cost
    tcmm_total_cost = sess["tcmm_cost_cached"] + output_cost
    cost_savings = max(0, naive_total_cost - tcmm_total_cost)
    token_savings = max(0, sess["naive_input_tokens"] - sess["tcmm_input_tokens"])

    return {
        "blocks": blocks,
        "stats": {
            "live_blocks": len(tcmm.live_blocks),
            "shadow_blocks": len(tcmm.shadow_blocks),
            "archive_blocks": len(tcmm.archive),
            "current_step": current_step,
            "total_live_tokens": total_live_chars // 4,
            "total_archive_tokens": total_archive_chars // 4,
            "recalled_tokens": recalled_chars // 4,
            "token_savings": token_savings,
            "cost_savings": round(cost_savings, 6),
            "session_id": sid,
            "session_input": sess["tcmm_input_tokens"],
            "session_output": sess["output_tokens"],
            "session_naive": sess["naive_input_tokens"],
            "session_turns": sess["turns"],
        },
    }


@app.get("/api/token_stats")
async def token_stats():
    """Return per-session token/cost stats."""
    sid, instance, sess = pool.get_active()
    if instance is None:
        return {"error": "No active session"}

    output_cost = sess.get("output_cost", 0)
    naive_total = sess["naive_cost_cached"] + output_cost
    tcmm_total = sess["tcmm_cost_cached"] + output_cost
    cost_savings = max(0, naive_total - tcmm_total)
    token_savings = max(0, sess["naive_input_tokens"] - sess["tcmm_input_tokens"])

    return {
        "session": {
            "session_id": sid,
            "turns": sess["turns"],
            "naive_tokens": sess["naive_input_tokens"],
            "tcmm_tokens": sess["tcmm_input_tokens"],
            "output_tokens": sess["output_tokens"],
            "input_tokens": sess["tcmm_input_tokens"],
            "token_savings": token_savings,
            "token_savings_pct": round(token_savings * 100 / max(sess["naive_input_tokens"], 1), 1),
            "naive_total_no_cache": round(sess["naive_cost_no_cache"] + output_cost, 6),
            "naive_total_cached": round(naive_total, 6),
            "tcmm_total_no_cache": round(sess["tcmm_cost_no_cache"] + output_cost, 6),
            "tcmm_total_cached": round(tcmm_total, 6),
            "cost_savings": round(cost_savings, 6),
            "cost_savings_pct": round(cost_savings * 100 / max(naive_total, 0.0001), 1),
            "derived_skipped": sess["derived_skipped"],
            "novel_added": sess["novel_added"],
            "compactions": sess["compactions"],
        },
        "history": sess["_token_history"][-20:],
    }


@app.get("/dream_status")
async def dream_status():
    """Check if a dream cycle is needed on the active session."""
    sid, instance, _ = pool.get_active()
    if instance is None:
        return {"error": "No active session"}

    tcmm = instance.tcmm
    archive_count = len(tcmm.archive)
    dream_count = len(getattr(tcmm, "dream_archive", {}))
    has_engine = hasattr(tcmm, "dream_engine")

    return {
        "session_id": sid,
        "dream_engine_available": has_engine,
        "archive_blocks": archive_count,
        "dream_nodes": dream_count,
        "recommendation": "run" if archive_count > 10 and has_engine else "skip",
    }


if __name__ == "__main__":
    port = int(os.environ.get("TCMM_PORT", "8811"))
    logger.info(f"Starting TCMM service on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
