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


# ── Connector framework (optional: gated on env) ─────────────────────────────
#
# The connector framework lives alongside this service in
# ``mcp-tools/connectors/``. We add it to sys.path with the same
# pattern the documents/filesystem MCP servers use for ``_shared/``,
# then import the recall + rendering helpers we need.
#
# If the framework directory is missing or imports fail, the recall
# step degrades to "TCMM only" — no fan-out, no shadow blocks. This
# keeps the service runnable on minimal deployments.
_CONNECTORS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "connectors")
)
if os.path.isdir(_CONNECTORS_DIR) and _CONNECTORS_DIR not in sys.path:
    sys.path.insert(0, _CONNECTORS_DIR)

try:
    from _base import (  # type: ignore[import-not-found]
        SHADOW_BLOCK_SYSTEM_PROMPT,
        UserContext as _ConnectorUserContext,
        default_registry as _connector_registry,
        gather_hints as _gather_connector_hints,
        render_shadow_blocks as _render_shadow_blocks,
    )
    _CONNECTORS_AVAILABLE = True
except ImportError as _e:
    logger.info(f"Connector framework unavailable ({_e}) — recall stays TCMM-only")
    _CONNECTORS_AVAILABLE = False
    SHADOW_BLOCK_SYSTEM_PROMPT = ""
    _ConnectorUserContext = None  # type: ignore[assignment, misc]
    _connector_registry = None  # type: ignore[assignment]
    _gather_connector_hints = None  # type: ignore[assignment]
    _render_shadow_blocks = None  # type: ignore[assignment]


def _connectors_enabled() -> bool:
    """Master gate for connector recall fan-out.

    Default off — explicit opt-in via env var lets us ship the
    framework without changing live request behavior. Once a real
    connector ships and is paired with a PII-proxy preamble that
    teaches the LLM about shadow blocks, this flips to on by default.
    """
    if not _CONNECTORS_AVAILABLE:
        return False
    flag = os.environ.get("VEILGUARD_CONNECTORS_ENABLED", "").lower()
    return flag in ("1", "true", "yes", "on")


def _connector_hint_deadline_ms() -> int:
    try:
        return int(os.environ.get("VEILGUARD_CONNECTOR_HINT_DEADLINE_MS", "300"))
    except ValueError:
        return 300


def _snippet_to_hit_dict(snippet) -> dict:
    """Convert a Snippet (from `gather_hints`) into the dict shape that
    `render_shadow_blocks` consumes — same shape produced by
    `VeilguardTCMM.recall_structured()`. Single conversion site so
    rendering has one input contract regardless of source."""
    return {
        "text": snippet.content,
        "score": snippet.score,
        "source": snippet.source,
        "title": snippet.title,
        "etag": snippet.etag,
        "last_modified": snippet.last_modified,
        "tool_ref": {
            "connector": snippet.ref.connector,
            "tool": snippet.ref.tool,
            "args": dict(snippet.ref.args),
        },
    }


async def _augment_with_connector_hints(
    prompt: str,
    user_message: str,
    user_id: str,
    tenant_id: str,
) -> tuple[str, int]:
    """If connectors are enabled and registered, fan out hint() calls,
    render shadow blocks, and append them to ``prompt``. Returns
    ``(augmented_prompt, n_blocks_emitted)``.

    The shadow-block system-prompt fragment is appended once per
    augmentation so the LLM knows how to interpret the blocks even
    without separate prompt-side wiring. When zero blocks survive
    the fan-out, the original prompt is returned unchanged.
    """
    if not _connectors_enabled():
        return prompt, 0

    connectors = _connector_registry.all_hint_capable()
    if not connectors:
        return prompt, 0

    user_ctx = _ConnectorUserContext(
        tenant_id=tenant_id or "",
        user_id=user_id or "",
        principals=[],
    )

    snippets = await _gather_connector_hints(
        connectors,
        user_message,
        user_ctx,
        deadline_ms=_connector_hint_deadline_ms(),
    )
    if not snippets:
        return prompt, 0

    hits = [_snippet_to_hit_dict(s) for s in snippets]
    shadow_text = _render_shadow_blocks(hits)
    if not shadow_text:
        return prompt, 0

    augmented = (
        f"{prompt}\n\n"
        f"{SHADOW_BLOCK_SYSTEM_PROMPT}\n"
        f"{shadow_text}"
    )
    return augmented, shadow_text.count("<shadow ")


def _resolve_lineage_edge(parent_conv: str, user_id: str) -> tuple[int, int] | None:
    """Look up ``(root, parent_aid)`` for a child-namespace lineage stamp.

    TCMM archive invariant: **every ingested block lives in the
    LanceDB archive table**. ``live_blocks`` is a VIEW — the current
    eviction/activation subset of the same archive rows. There is no
    "live-only" or "archive-only" state. So a parent namespace that
    has received any ingestion has rows here.

    Queries the shared LanceDB archive table for the latest block in
    the parent's namespace under the given user_id. Returns ``(root,
    parent_aid)`` where ``root`` is inherited from that block's own
    ``lineage.root`` (or the parent_aid itself if the parent block's
    root is 0 / self-referential — same invariant as a freshly
    rooted episode).

    Returns ``None`` if parent_conv is empty or the namespace-filtered
    query genuinely has no rows (meaning no ingestion has happened
    for that conversation yet — should be rare).
    """
    if not parent_conv:
        return None
    try:
        import lancedb
        # TCMM's internal layout: DATA_DIR / LANCE_DB_NAME / "tcmm.db"
        # is the actual LanceDB root (see tcmm_core.py's db_path join).
        # First join gives the session-shared dir; second is the DB.
        db_dir = os.path.join(DATA_DIR, LANCE_DB_NAME, "tcmm.db")
        if not os.path.isdir(db_dir):
            return None
        db = lancedb.connect(db_dir)
        try:
            tbl = db.open_table("archive")
        except Exception:
            return None
        # Normalize parent_conv the same way the session pool does,
        # so we query the actual stored namespace rather than the
        # raw header value (defensive — they're usually identical).
        parent_sid = pool._normalize_id(parent_conv)
        uid = user_id or "default"
        # Grab the most recent block in the parent namespace owned
        # by this tenant. Note the `.limit(1)` is only used to cap
        # Lance's scan, not to order — Lance returns rows in table
        # order, not aid-desc — so we pull the full parent namespace
        # then sort by aid. Parent namespaces are bounded (tens of
        # rows) so this is cheap.
        df = (
            tbl.search()
               .where(f"namespace = '{parent_sid}' AND user_id = '{uid}'")
               .to_arrow()
               .to_pandas()
               .sort_values("aid", ascending=False)
               .head(1)
        )
        if len(df) == 0:
            # Parent namespace has zero rows for this tenant. Two
            # common reasons:
            #   1. LibreChat's MCP transport forwards a UUID-format
            #      conversation_id (e.g. ``9aa425aa-2d81-43fa-...``)
            #      that is DIFFERENT from the ``new-<uid>-XXX`` id
            #      its main flow archives under. The parent the
            #      sub-agent captured never matches a real stored
            #      namespace. Seen on 22 Apr 2026 — sub-agents were
            #      minting valid child namespaces but lineage kept
            #      resolving to None because no row answered the
            #      namespace filter.
            #   2. A fresh conversation that genuinely hasn't
            #      ingested its first user/assistant block yet.
            # Fallback: use the user's most recent archive row
            # regardless of namespace. That's a weaker anchor than
            # a real parent-namespace match but it still gives the
            # child SOMETHING to stamp lineage against, which is
            # what dream-engine / canonical-state synthesis needs.
            logger.info(
                f"[LINEAGE] parent_ns={parent_sid} has no rows for "
                f"uid={uid[:8]} — falling back to user's latest archive row"
            )
            df = (
                tbl.search()
                   .where(f"user_id = '{uid}'")
                   .to_arrow()
                   .to_pandas()
                   .sort_values("aid", ascending=False)
                   .head(1)
            )
            if len(df) == 0:
                logger.info(
                    f"[LINEAGE] no archive rows at all for uid={uid[:8]} — "
                    f"child will root on itself"
                )
                return None
        row = df.iloc[0]
        parent_aid = int(row["aid"])
        lineage = row.get("lineage")
        root = 0
        if lineage is not None:
            try:
                root_val = lineage.get("root", 0) if hasattr(lineage, "get") else (
                    lineage["root"] if "root" in lineage else 0
                )
                root = int(root_val or 0)
            except Exception:
                root = 0
        # Freshly rooted episode (parent was the first block of its
        # own conversation): use parent_aid as the root so every
        # descendant inherits it.
        if root <= 0:
            root = parent_aid
        return (root, parent_aid)
    except Exception as e:
        logger.warning(f"[LINEAGE] resolve failed for parent={parent_conv[:16]}: {e}")
        return None


def _stamp_child_lineage(
    instance, lineage_root: int, lineage_parent_aid: int, aid_cutoff: int
) -> int:
    """Stamp ``lineage`` on archive rows created during the current call.

    Only touches rows whose aid is > ``aid_cutoff`` — that's how we
    identify blocks the current ingest just added (we snapshot the
    max aid before calling instance.pre/post_response). All such
    rows get ``lineage.root = lineage_root`` and
    ``lineage.parents = [lineage_parent_aid]``.

    Writes go to the in-memory archive dict; the storage provider's
    lazy persistence picks them up on the next flush. Returns the
    number of rows stamped.
    """
    stamped = 0
    try:
        for aid, entry in list(instance.tcmm.archive.items()):
            if aid <= aid_cutoff:
                continue
            if not isinstance(entry, dict):
                continue
            entry["lineage"] = {
                "root": int(lineage_root),
                "parents": [int(lineage_parent_aid)],
            }
            instance.tcmm.archive[aid] = entry
            stamped += 1
        # Force-persist so dreams + recall see the edge immediately.
        try:
            if hasattr(instance.tcmm, "persist_archive"):
                instance.tcmm.persist_archive()
            elif hasattr(instance.tcmm, "flush_archive"):
                instance.tcmm.flush_archive()
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"[LINEAGE] stamp failed: {e}")
    return stamped

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

    NLP_BACKEND=local      → LocalNLPAdapter (Gemma E2B + SpaCy, needs GPU)
    NLP_BACKEND=vertex     → VertexNLPAdapter via Vertex AI (project + ADC)
    NLP_BACKEND=ai_studio  → VertexNLPAdapter routed at AI Studio
                             (generativelanguage.googleapis.com + API key).
                             The adapter already supports both modes; the
                             selector below picks which one by passing
                             either ``project_id`` (Vertex) or ``api_key``
                             (AI Studio) — never both.

    EMBED_BACKEND=local    → LocalEmbeddingAdapter (SentenceTransformer, GPU)
    EMBED_BACKEND=vertex   → VertexEmbeddingAdapter (text-embedding-005)
    """
    global _shared_nlp, _shared_embedder

    nlp_backend = os.environ.get("NLP_BACKEND", "local").lower()
    embed_backend = os.environ.get("EMBED_BACKEND", "local").lower()
    vertex_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    vertex_region = os.environ.get("VERTEX_REGION", "us-central1")
    nlp_model = os.environ.get("NLP_MODEL", "gemini-2.5-flash")

    # NLP adapter
    if nlp_backend == "ai_studio":
        from adapters.vertex_nlp_adapter import VertexNLPAdapter
        # Adapter env-var fallback chain is VERTEX_API_KEY → GEMINI_API_KEY,
        # but our prod env stores the key as GOOGLE_API_KEY. Pull it
        # explicitly here so the adapter receives it via constructor and
        # the env-var fallback never has to fire.
        api_key = (
            os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("VERTEX_API_KEY")
            or ""
        )
        if not api_key:
            raise RuntimeError(
                "NLP_BACKEND=ai_studio but no API key found "
                "(checked GOOGLE_API_KEY, GEMINI_API_KEY, VERTEX_API_KEY)"
            )
        logger.info(f"Loading NLP adapter via AI Studio (model={nlp_model})...")
        # project_id="" forces the adapter into its AI Studio code path.
        _shared_nlp = VertexNLPAdapter(
            project_id="",
            api_key=api_key,
            model=nlp_model,
        )
    elif nlp_backend == "vertex":
        from adapters.vertex_nlp_adapter import VertexNLPAdapter
        logger.info(
            f"Loading Vertex AI NLP adapter (model={nlp_model}, "
            f"project={vertex_project})..."
        )
        _shared_nlp = VertexNLPAdapter(
            project_id=vertex_project,
            region=vertex_region,
            model=nlp_model,
        )
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

    # ── Embedder compatibility check ─────────────────────────────────────────
    # Vectors built with one embedder are not comparable with vectors built
    # with another (different model, different backend, even different
    # dimensions). If the configured embedder doesn't match the one that
    # built the existing index, recall silently breaks. Refuse to start on
    # mismatch unless TCMM_ALLOW_EMBEDDER_SWAP=1 (operator has accepted that
    # they must re-embed).
    storage_backend_now = os.environ.get("TCMM_STORAGE", "lance").lower()
    if storage_backend_now in ("lance", "lancedb"):
        from core.providers.lance import check_embedder_compatibility
        lance_db_name_now = os.environ.get("TCMM_LANCE_DB", "veilguard")
        shared_data_dir = os.path.join(DATA_DIR, lance_db_name_now)
        os.makedirs(shared_data_dir, exist_ok=True)
        allow_swap = os.environ.get("TCMM_ALLOW_EMBEDDER_SWAP", "").lower() in (
            "1", "true", "yes",
        )
        result = check_embedder_compatibility(
            db_path=shared_data_dir,
            backend=_shared_embedder.backend,
            model_name=_shared_embedder.model_name,
            dim=_shared_embedder.dimension,
            allow_swap=allow_swap,
        )
        status = result["status"]
        if status == "ok":
            logger.info(
                f"Embedder verified against persisted metadata "
                f"(backend={_shared_embedder.backend}, "
                f"model={_shared_embedder.model_name}, "
                f"dim={_shared_embedder.dimension})"
            )
        elif status == "first_run":
            logger.info(
                f"Persisted embedder metadata for first run "
                f"(backend={_shared_embedder.backend}, "
                f"model={_shared_embedder.model_name}, "
                f"dim={_shared_embedder.dimension})"
            )
        elif status == "swap_allowed":
            logger.warning(
                f"Embedder swap accepted via TCMM_ALLOW_EMBEDDER_SWAP. "
                f"Existing vectors are incomparable with new ones; "
                f"re-embed required for correct recall. "
                f"Detail: {result['message']}"
            )
        elif status == "mismatch":
            logger.error(f"FATAL: {result['message']}")
            raise RuntimeError(result["message"])


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

            # Pre-warm the user archive + FTS index at session creation
            # so the first recall in this session doesn't pay the
            # one-time setup cost (~7s for FTS + node cache + vector
            # index warmup). Subsequent recalls already share these
            # caches via the LanceStorageProvider instance held by
            # this session.
            try:
                _arch = getattr(instance.tcmm, "archive", None)
                if _arch is not None and hasattr(_arch, "bulk_warm_user_archive"):
                    _arch.bulk_warm_user_archive()
                    instance.tcmm._last_user_warm_ts = time.time()
                # Trigger FTS index creation now (one-shot per session).
                _sparse = getattr(instance.tcmm, "archive_sparse_index", None)
                if _sparse is not None and hasattr(_sparse, "_ensure_fts_index"):
                    _sparse._ensure_fts_index()
                # Trigger vector index ensure (Lance lazy index build).
                _vec = getattr(instance.tcmm, "archive_vector_index", None)
                if _vec is not None and hasattr(_vec, "_ensure_index"):
                    try:
                        _vec._ensure_index()
                    except Exception:
                        pass
            except Exception as _e:
                logger.debug(f"[POOL] pre-warm skipped: {_e}")

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
    # Sub-agent spawn lineage. Set by the PII proxy when the incoming
    # LLM call came from a ``_spawn_scope`` in sub-agents (metadata
    # field ``lineage_parent_conv`` carried the parent's conv_id).
    # TCMM uses it to stamp ``lineage.parents[0]`` (latest aid in the
    # parent namespace) and inherit ``lineage.root`` on the child's
    # first archive block. Empty for top-level LibreChat turns.
    lineage_parent_conv: str = ""


class PostResponseBody(BaseModel):
    raw_output: str
    conversation_id: str = ""
    user_id: str = ""
    # Origin tag for the assistant-side block. Usually "assistant_text";
    # set to "tool_use" when the assistant reply is a tool invocation.
    origin: str = "assistant_text"
    lineage_parent_conv: str = ""


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
    lineage_parent_conv: str = ""


import asyncio

# Per-session locks keyed by normalized conversation_id. Replaces the
# old single ``_tcmm_lock = asyncio.Lock()`` that serialized EVERY
# TCMM endpoint across the whole process — a ``post_response`` or
# ``ingest_turn`` from one user's conversation would block an
# unrelated user's ``pre_request`` for the full duration of the
# other call. Measured 23 Apr 2026: a pre_request for conv
# bb400c87 reported ``took=11.02s`` of which only ~1.2s was its own
# work; the remaining ~9.8s was spent waiting for ``_tcmm_lock``.
#
# Each TCMM instance is already session-scoped (``pool.get(cid)``
# returns the one that owns that conversation's state), so concurrent
# calls on DIFFERENT sessions have no shared state to protect — they
# can run in parallel. Only overlapping calls on the SAME session
# need to serialize, and that's what the per-session lock does.
#
# The dict grows as new conversations come in; over long uptimes we
# may want an LRU cap, but at the current scale (<100 active sessions
# / day) a plain dict is fine — each Lock object is ~200 bytes.
# Access from a single asyncio event loop, so the dict itself doesn't
# need an outer mutex; ``setdefault`` is atomic under the GIL.
_session_locks: dict[str, asyncio.Lock] = {}
_GLOBAL_TCMM_LOCK_KEY = "__global__"

# Note: an earlier version of this file had a ``_strip_ref_tokens`` helper
# that replaced any ``REF_PERSON_N`` / ``REF_EMAIL_N`` / etc. with the
# literal ``[redacted]`` at ingest time, intended as defense-in-depth
# against a pii-proxy rehydrate regression. It was removed 23 Apr 2026
# after tracing the real bug to a substring-collision inside pii-proxy's
# ``pii_store.rehydrate`` (see session.py). The correct architecture is:
#   * pii-proxy redacts user + memory content on the way to Claude
#   * pii-proxy rehydrates Claude's response on the way back
#   * TCMM stores REAL content (post-rehydrate) — never sees REF tokens
# The sanitizer was masking the real bug and destroying info. If REF
# tokens ever land in TCMM again, treat it as a pii-proxy regression and
# fix upstream — don't paper over it here.


def _get_session_lock(conversation_id: str) -> asyncio.Lock:
    """Return the asyncio.Lock that protects ``conversation_id``'s TCMM state.

    Endpoints that don't operate on a specific conversation (``/dream``,
    ``/api/memory_heatmap``, etc.) pass an empty string and get the
    ``__global__`` sentinel lock — still per-key, so they don't block
    unrelated session endpoints, but they DO serialize with each other
    since they may touch pool-wide state.
    """
    key = conversation_id or _GLOBAL_TCMM_LOCK_KEY
    existing = _session_locks.get(key)
    if existing is not None:
        return existing
    # setdefault so a concurrent coroutine creating the same key in
    # parallel ends up using one Lock object, not two.
    new = asyncio.Lock()
    return _session_locks.setdefault(key, new)


# Legacy alias. Kept for any internal callers we might not have
# migrated yet — they'll hit ``__global__`` and serialize among
# themselves but won't block per-session endpoints.
_tcmm_lock = _get_session_lock("")


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

    # Per-session lock — concurrent pre_request on DIFFERENT conversations
    # now runs in parallel. Only overlap on the same conversation serializes.
    _sid = pool._normalize_id(body.conversation_id)
    async with _get_session_lock(_sid):
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
                # Connector hints (env-gated) augment the recall-only path
                # too, so sub-agent recall can surface SharePoint / Slack
                # candidates alongside their own TCMM memory.
                prompt, n_shadow = await _augment_with_connector_hints(
                    prompt,
                    body.user_message,
                    user_id=body.user_id,
                    tenant_id="",
                )
                logger.info(
                    f"[RECALL-ONLY] session={sid} recalled={len(recalled_ids)} "
                    f"connector_shadows={n_shadow} prompt={len(prompt)} chars"
                )
                return {
                    "prompt": prompt,
                    "stats": {
                        "recalled": len(recalled_ids),
                        "connector_shadows": n_shadow,
                    },
                }

            instance = pool.get(body.conversation_id, user_id=body.user_id)
            sess = pool.get_stats(body.conversation_id)

            # Lineage stamp prep: before we ingest, resolve the edge
            # back to the parent namespace if a hint was supplied and
            # not already cached on the session. We snapshot the
            # child's current max aid so we can stamp only blocks
            # this call creates (not ones from previous turns).
            if body.lineage_parent_conv and getattr(instance, "_lineage_root", None) is None:
                resolved = _resolve_lineage_edge(body.lineage_parent_conv, body.user_id)
                if resolved is not None:
                    instance._lineage_root, instance._lineage_parent_aid = resolved
                    logger.info(
                        f"[LINEAGE] child={pool._normalize_id(body.conversation_id)[:16]} "
                        f"parent={body.lineage_parent_conv[:16]} "
                        f"root={instance._lineage_root} parent_aid={instance._lineage_parent_aid}"
                    )
            aid_before = max(instance.tcmm.archive.keys(), default=0)

            start = time.time()
            prompt = instance.pre_request(
                body.user_message,
                session_id=body.conversation_id,
                origin=body.origin or "user",
            )
            elapsed = time.time() - start

            # Connector hint fan-out (env-gated). Adds a shadow-block
            # section to the prompt with candidates from registered
            # connectors. No-op when connectors are disabled, when
            # nothing is registered, or when no connector returns
            # results within its deadline.
            prompt, _n_shadow_blocks = await _augment_with_connector_hints(
                prompt,
                body.user_message,
                user_id=body.user_id,
                tenant_id="",
            )

            # Post-ingest stamp: any rows added during pre_request
            # (usually the user message) get their lineage wired.
            if getattr(instance, "_lineage_root", None):
                n = _stamp_child_lineage(
                    instance,
                    instance._lineage_root,
                    instance._lineage_parent_aid,
                    aid_before,
                )
                if n:
                    logger.info(f"[LINEAGE] stamped {n} rows (pre_request)")

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

            # Fire the deferred user-message ingest as a background task
            # AFTER we've built the return payload. The task acquires the
            # same per-session lock we're about to release, so subsequent
            # pre_request/post_response calls on this session still see
            # the user message in the archive — they just wait briefly
            # for the background ingest to land.
            has_pending = bool(getattr(instance, "_pending_user_ingest", None))
            _response_payload = {
                "prompt": prompt,
                "stats": {
                    "live_blocks": status["live_blocks"],
                    "shadow_blocks": status["shadow_blocks"],
                    "recalled": status["last_recalled"],
                    "elapsed_ms": int(elapsed * 1000),
                    "input_tokens": tcmm_input_tokens,
                    "session_id": pool._normalize_id(body.conversation_id),
                    "user_ingest": "deferred" if has_pending else "skipped",
                    "connector_shadows": _n_shadow_blocks,
                },
            }
        except Exception as e:
            logger.error(f"[PRE] Error: {e}", exc_info=True)
            return {"error": str(e), "prompt": None}

    # Lock released. Fire background ingest (it reacquires the session
    # lock so it serializes correctly behind any subsequent request).
    if has_pending:
        asyncio.create_task(_pre_request_ingest_user(_sid, body))

    return _response_payload


async def _pre_request_ingest_user(_sid: str, body: PreRequestBody) -> None:
    """Background-task body that runs the deferred user-message archive.

    Fired by ``/pre_request`` AFTER the prompt has already been returned
    to pii-proxy. Acquires the per-session lock so it serializes behind
    any other request on this conversation — subsequent pre_request /
    post_response calls will queue behind this and observe the user
    message as archived by the time they run recall.

    We call ``flush_pending_user_ingest`` on the adapter, which reads
    the ``_pending_user_ingest`` dict pre_request set synchronously
    and executes the full add_new_block + flush + archive-entry write.
    On a slow LanceDB (the 6-7s case) this runs off the critical path
    so the user gets the prompt instantly and typing into the chat
    feels responsive.
    """
    try:
        async with _get_session_lock(_sid):
            instance = pool.get(body.conversation_id, user_id=body.user_id)
            t0 = time.time()
            ran = await asyncio.to_thread(
                instance.flush_pending_user_ingest,
                body.conversation_id,
            )
            if ran:
                dt = time.time() - t0
                logger.info(
                    f"[PRE-BG-INGEST] session={_sid} user-message archived "
                    f"in {dt*1000:.0f}ms"
                )
    except Exception as e:
        logger.error(
            f"[PRE-BG-INGEST] failed for session={_sid}: {type(e).__name__}: {e}",
            exc_info=True,
        )


async def _post_response_ingest(body: PostResponseBody, _sid: str) -> None:
    """Background-task body for post_response ingestion.

    Runs the full TCMM ingestion pipeline (process_heatmap, add_new_block,
    flush_current_block, _persist_live_blocks) inside the session lock.
    The HTTP response has already returned ``answer`` to pii-proxy by now,
    so any time this takes no longer stalls the chat UI.

    Serialization is preserved: if the user sends another turn before this
    ingest finishes, their ``pre_request`` waits for the session lock —
    which is exactly what we want so turn N+1's recall sees turn N's
    ingested state.
    """
    try:
        async with _get_session_lock(_sid):
            instance = pool.get(body.conversation_id, user_id=body.user_id)
            sess = pool.get_stats(body.conversation_id)

            if body.lineage_parent_conv and getattr(instance, "_lineage_root", None) is None:
                resolved = _resolve_lineage_edge(body.lineage_parent_conv, body.user_id)
                if resolved is not None:
                    instance._lineage_root, instance._lineage_parent_aid = resolved
                    logger.info(
                        f"[LINEAGE] child={pool._normalize_id(body.conversation_id)[:16]} "
                        f"parent={body.lineage_parent_conv[:16]} "
                        f"root={instance._lineage_root} parent_aid={instance._lineage_parent_aid}"
                    )
            aid_before = max(instance.tcmm.archive.keys(), default=0)

            start = time.time()
            # Run the blocking TCMM ingest in a thread so we don't hog the
            # event loop. ``instance.post_response`` calls Vertex HTTP from
            # synchronous code (process_heatmap → lazy recall → embed +
            # classify); without ``to_thread`` the whole asyncio loop —
            # which is also serving pre_request for other users — blocks
            # on the first socket read.
            answer = await asyncio.to_thread(
                instance.post_response,
                body.raw_output,
                body.conversation_id,
                body.origin or "assistant_text",
            )
            elapsed = time.time() - start

            if getattr(instance, "_lineage_root", None):
                n = _stamp_child_lineage(
                    instance,
                    instance._lineage_root,
                    instance._lineage_parent_aid,
                    aid_before,
                )
                if n:
                    logger.info(f"[LINEAGE] stamped {n} rows (post_response)")

            status = instance.get_status()
            sid = pool._normalize_id(body.conversation_id)
            logger.info(
                f"[POST-BG] session={sid} "
                f"answer_len={len(answer)} "
                f"step={status['current_step']} "
                f"archive={status['archive_blocks']} "
                f"took={elapsed:.2f}s"
            )

            output_tokens = len(answer) // 4
            sess["output_tokens"] += output_tokens
            output_cost = (output_tokens / 1000) * _COST_OUTPUT
            sess["output_cost"] += output_cost
            sess["_all_messages"].append(answer)

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
    except Exception as e:
        logger.error(
            f"[POST-BG] ingestion failed for session={_sid}: {type(e).__name__}: {e}",
            exc_info=True,
        )


@app.post("/post_response")
async def post_response(body: PostResponseBody):
    """Called AFTER PII rehydration. Returns the clean answer IMMEDIATELY
    and defers the TCMM ingestion pipeline to a background task.

    Why: the synchronous ingestion path (process_heatmap → lazy recall
    hydration + new-block NLP classification + block embedding) was
    firing 5-9 sequential Vertex API calls taking 30+ seconds for a
    trivial response like "Noted.". Because pii-proxy awaits this
    endpoint before closing its stream to LibreChat, the frontend
    kept its stop button visible and the typing indicator pulsing for
    the full 30s — the response was "done" but the UI said otherwise.
    Measured 23 Apr 2026: ``[POST] took=32.87s`` for a 50-char answer.

    Fix: split_answer_and_heatmap synchronously (fast regex) so we can
    return the clean ``answer`` string within milliseconds, then fire
    the rest of the ingestion as an ``asyncio.create_task``. The
    background task still acquires the session lock, so subsequent
    ``pre_request`` / ``post_response`` calls for the same session
    serialize correctly behind it.
    """
    if _shared_nlp is None:
        return {"error": "TCMM not initialized", "answer": body.raw_output}

    # Fast path: extract the clean answer from raw_output so we can
    # return immediately. split_answer_and_heatmap is a regex parse —
    # microseconds, no I/O.
    try:
        from core.tcmm_core import split_answer_and_heatmap as _split
        answer, _flag = _split(body.raw_output or "")
    except Exception:
        answer = None
    if not answer:
        answer = (body.raw_output or "").strip()

    _sid = pool._normalize_id(body.conversation_id)

    # Fire the heavy ingest in the background. Do NOT await.
    # The task runs in the current event loop and will acquire the
    # session lock in ``_post_response_ingest`` so concurrent calls on
    # the same session still serialize.
    asyncio.create_task(_post_response_ingest(body, _sid))

    return {
        "answer": answer,
        "stats": {
            "session_id": _sid,
            "ingest": "deferred",
        },
    }


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

    _sid = pool._normalize_id(body.conversation_id)
    async with _get_session_lock(_sid):
        try:
            instance = pool.get(body.conversation_id, user_id=body.user_id)
            if body.lineage_parent_conv and getattr(instance, "_lineage_root", None) is None:
                resolved = _resolve_lineage_edge(body.lineage_parent_conv, body.user_id)
                if resolved is not None:
                    instance._lineage_root, instance._lineage_parent_aid = resolved
                    logger.info(
                        f"[LINEAGE] child={pool._normalize_id(body.conversation_id)[:16]} "
                        f"parent={body.lineage_parent_conv[:16]} "
                        f"root={instance._lineage_root} parent_aid={instance._lineage_parent_aid}"
                    )
            aid_before = max(instance.tcmm.archive.keys(), default=0)
            added = instance.ingest_turn(items, session_id=body.conversation_id)
            sid = pool._normalize_id(body.conversation_id)
            logger.info(
                f"[INGEST-TURN] session={sid} added={added}/{len(items)} "
                f"origins={[ (i or {}).get('origin') for i in items ]}"
            )
            if getattr(instance, "_lineage_root", None):
                n = _stamp_child_lineage(
                    instance,
                    instance._lineage_root,
                    instance._lineage_parent_aid,
                    aid_before,
                )
                if n:
                    logger.info(f"[LINEAGE] stamped {n} rows (ingest_turn)")
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


# ── Debug: per-session state + manual eviction ──────────────────────────────
# Added for eviction/KV-cache bench testing (23 Apr 2026). TCMM evicts
# based on per-block heat < COLD_EVICT_HEAT once live token budget is
# hit, so you can't observe the behaviour just by sending more turns
# until the 10%-of-context budget is exceeded. These endpoints let you
# snapshot a session, force eviction without waiting, then compare
# before/after + watch the PII-proxy cache logs on the next turn.


@app.get("/debug_session/{conversation_id:path}")
async def debug_session_state(conversation_id: str, user_id: str = ""):
    """Return detailed live/shadow/archive stats for one session.

    Use this to snapshot a session before an eviction test. Pair with
    ``/debug_evict`` to compare state change, and with PII-proxy's
    ``[CACHE] ... create= read=`` lines to see how eviction hits
    Anthropic's prompt-cache boundary.

    Uses ``pool.get()`` so that a session which was persisted to disk
    but hasn't been referenced since last service restart gets lazy-
    loaded + its live blocks restored. Without this, snapshotting
    right after a restart always returns "No active instance".
    """
    sid = pool._normalize_id(conversation_id)
    instance = pool.get(conversation_id, user_id=user_id or "default")
    if instance is None:
        return {"error": f"No active instance for sid={sid}"}
    tcmm = instance.tcmm

    def _tok_count(b):
        return int(getattr(b, "token_count", 0) or 0)

    live = list(tcmm.live_blocks)
    shadow = list(getattr(tcmm, "shadow_blocks", []) or [])
    live_tokens = sum(_tok_count(b) for b in live)
    shadow_tokens = sum(_tok_count(b) for b in shadow)

    # Show top / bottom live blocks by heat so we can predict which
    # will be evicted next.
    def _b_brief(b):
        return {
            "id": getattr(b, "id", None),
            "heat": round(float(getattr(b, "heat", 0.0)), 4),
            "tokens": _tok_count(b),
            "created_step": int(getattr(b, "created_step", 0) or 0),
            "role": getattr(b, "priority_class", "") or "?",
            "text_preview": (getattr(b, "text", "") or "")[:80],
        }

    by_heat = sorted(live, key=lambda b: float(getattr(b, "heat", 0.0)))
    return {
        "session_id": sid,
        "user_id": user_id or getattr(instance, "user_id", ""),
        "live": {
            "count": len(live),
            "tokens": live_tokens,
            "max_tokens": int(getattr(tcmm, "max_live_tokens", 0) or 0),
            "absolute_max_tokens": int(
                getattr(tcmm, "absolute_max_live_tokens", 0) or 0
            ),
            "coldest": [_b_brief(b) for b in by_heat[:5]],
            "hottest": [_b_brief(b) for b in by_heat[-5:]],
        },
        "shadow": {
            "count": len(shadow),
            "tokens": shadow_tokens,
            "max_tokens": int(getattr(tcmm, "max_shadow_tokens", 0) or 0),
        },
        "archive": {"count": len(tcmm.archive)},
        "current_step": int(getattr(tcmm, "current_step", 0) or 0),
    }


class EvictBody(BaseModel):
    conversation_id: str
    user_id: str = ""
    # Mode: ``cold`` (default) runs the normal cold-eviction policy
    # at whatever ``COLD_EVICT_HEAT`` threshold is configured, which
    # may evict 0 blocks if everything is above the floor. ``n``
    # force-evicts the N coldest blocks regardless of heat floor —
    # use for deterministic bench runs. ``budget`` temporarily lowers
    # ``max_live_tokens`` to ``target_tokens`` and evicts until the
    # budget is met — simulates "a bigger conversation hitting the
    # 10%-of-context cap".
    mode: str = "cold"
    n: int = 0
    target_tokens: int = 0


@app.post("/debug_evict")
async def debug_evict(body: EvictBody):
    """Force live → archive eviction on a specific session.

    Returns the before/after stats so you can see exactly which
    blocks moved. The eviction path is the same ``_evict_live_block``
    TCMM uses in production — blocks get enqueued to the archive
    table and disappear from ``live_blocks``. Shadow promotion is
    NOT triggered here (that's a separate recall pathway).
    """
    sid = pool._normalize_id(body.conversation_id)
    # Same lazy-load as /debug_session — evict_debug can fire right
    # after a service restart as long as the session has a persisted
    # live-block snapshot on disk.
    instance = pool.get(body.conversation_id, user_id=body.user_id or "default")
    if instance is None:
        return {"error": f"No active instance for sid={sid}"}
    tcmm = instance.tcmm

    def _snapshot():
        live = list(tcmm.live_blocks)
        return {
            "live_count": len(live),
            "live_tokens": sum(int(getattr(b, "token_count", 0) or 0) for b in live),
            "archive_count": len(tcmm.archive),
        }

    before = _snapshot()
    evicted_ids: list[int] = []

    async with _get_session_lock(sid):
        try:
            if body.mode == "cold":
                # Run the native cold-eviction policy. Records ids it
                # touches so we can report which blocks moved.
                pre_ids = {getattr(b, "id", None) for b in tcmm.live_blocks}
                tcmm._evict_cold_live_blocks()
                post_ids = {getattr(b, "id", None) for b in tcmm.live_blocks}
                evicted_ids = sorted(i for i in pre_ids - post_ids if i is not None)

            elif body.mode == "n":
                n = max(1, int(body.n or 1))
                # Evict the N coldest. Mirrors the cold-eviction
                # selection but ignores the heat floor — useful when
                # everything's above floor and you just want to see
                # the mechanism fire.
                by_heat = sorted(
                    tcmm.live_blocks,
                    key=lambda b: float(getattr(b, "heat", 0.0)),
                )[:n]
                for b in by_heat:
                    try:
                        tcmm.live_blocks.remove(b)
                    except ValueError:
                        continue
                    evicted_ids.append(int(getattr(b, "id", 0) or 0))
                    tcmm._evict_live_block(b, reason="debug-n")

            elif body.mode == "budget":
                # Temporarily lower the budget and let the built-in
                # ``_live_token_count() > max_live_tokens`` loop drain
                # cold blocks to fit. Restores the old budget after.
                target = max(0, int(body.target_tokens or 0))
                old_budget = int(getattr(tcmm, "max_live_tokens", 0) or 0)
                try:
                    tcmm.max_live_tokens = target
                    # The same loop shape as the internal enforcement.
                    while tcmm._live_token_count() > tcmm.max_live_tokens and tcmm.live_blocks:
                        # Pick coldest victim.
                        victim = min(
                            tcmm.live_blocks,
                            key=lambda b: float(getattr(b, "heat", 0.0)),
                        )
                        tcmm.live_blocks.remove(victim)
                        evicted_ids.append(int(getattr(victim, "id", 0) or 0))
                        tcmm._evict_live_block(victim, reason="debug-budget")
                finally:
                    tcmm.max_live_tokens = old_budget

            else:
                return {"error": f"unknown mode: {body.mode}"}

            # Persist live blocks so the disk snapshot reflects the
            # new state (otherwise a restart would restore the
            # pre-eviction set from ``_latest.json``).
            try:
                instance._persist_live_blocks(session_id=body.conversation_id)
            except Exception as e:
                logger.warning(f"[DEBUG-EVICT] persist_live_blocks failed: {e}")

        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}

    after = _snapshot()
    logger.info(
        f"[DEBUG-EVICT] sid={sid} mode={body.mode} "
        f"evicted={len(evicted_ids)} "
        f"live {before['live_count']}→{after['live_count']} "
        f"({before['live_tokens']}→{after['live_tokens']} tokens), "
        f"archive {before['archive_count']}→{after['archive_count']}"
    )
    return {
        "session_id": sid,
        "mode": body.mode,
        "evicted_ids": evicted_ids,
        "evicted_count": len(evicted_ids),
        "before": before,
        "after": after,
    }


@app.post("/dream")
async def trigger_dream():
    """Trigger a dream cycle on the active session."""
    sid, instance, _ = pool.get_active()
    if instance is None:
        return {"error": "No active session"}

    # Per-session lock so a dream cycle on conv A doesn't block
    # pre_request on conv B.
    async with _get_session_lock(sid or ""):
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
