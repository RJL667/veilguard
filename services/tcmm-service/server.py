"""
TCMM HTTP Service for Veilguard.

Thin FastAPI wrapper around TCMM core. Each LibreChat conversation gets its own
TCMM instance (namespace), with shared NLP/embedding models across sessions.

Runs on the Windows host (not Docker) because TCMM needs ONNX, FAISS, spaCy.

Start: python services/tcmm-service/server.py
"""

import logging
import os
import sys
import time
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

# ── Suppress HuggingFace transformers' TF/Flax auto-probe ─────────────────────
# Must be set BEFORE any import that might transitively pull in `transformers`
# (sentence_transformers does, even if we end up not using it). Without these
# guards, importing transformers triggers a 10-15s detection probe that loads
# tensorflow + keras + tf_keras and spews deprecation warnings. We never use
# TF in this stack — pin to PyTorch only.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

# ── Load .env early so NLP_BACKEND / EMBED_BACKEND / GOOGLE_* are visible ────
# Two sources, in priority order (process env always wins via override=False):
#   1. TCMM repo's .env  — adapter/backend selection + API keys
#   2. Veilguard repo's .env — runtime + service URLs
# Previously absent; backend selection silently fell back to "local" defaults
# because launching `python server.py` from a bare shell didn't export these.
try:
    from dotenv import load_dotenv as _load_dotenv
    _TCMM_ROOT_FOR_ENV = os.environ.get(
        "TCMM_ROOT",
        r"C:\Users\rudol\.gemini\antigravity\tcmm\TCMM",
    )
    _load_dotenv(os.path.join(_TCMM_ROOT_FOR_ENV, ".env"), override=False)
    _load_dotenv(
        os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
        override=False,
    )
except ImportError:
    # python-dotenv not installed — fall back to shell env only.
    pass

import uvicorn
from fastapi import FastAPI, Request, HTTPException
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
# ``services/connectors/``. We add it to sys.path with the same
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

    NLP_BACKEND=ai_studio  → AIStudioNLPAdapter (production default).
                             generativelanguage.googleapis.com + API key.
                             Fast, cheap, no ADC, no GCP project.
    NLP_BACKEND=local      → LocalNLPAdapter (Gemma E2B + SpaCy, GPU). Used
                             only for offline testing — pulls heavy ML deps
                             and a transitive TF probe; not for prod use.

    EMBED_BACKEND=local    → LocalEmbeddingAdapter (FastEmbed ONNX-int8 on
                             CPU, snowflake/snowflake-arctic-embed-xs by
                             default). Sole supported value.

    Vertex AI was removed entirely from the codebase 2026-05-13 — the only
    Google-API surface this service hits is AI Studio's Generative
    Language API. ``NLP_BACKEND=vertex`` and ``EMBED_BACKEND=vertex`` now
    raise RuntimeError on startup so any stale config fails loud rather
    than silently regressing.
    """
    global _shared_nlp, _shared_embedder

    nlp_backend = os.environ.get("NLP_BACKEND", "ai_studio").lower()
    embed_backend = os.environ.get("EMBED_BACKEND", "local").lower()
    nlp_model = os.environ.get("NLP_MODEL", "gemini-2.5-flash")

    # Fail loud if someone tries to use the retired vertex NLP backend so
    # they discover the rename instead of silently falling through to local.
    if nlp_backend == "vertex":
        raise RuntimeError(
            "NLP_BACKEND=vertex is no longer supported. Use NLP_BACKEND=ai_studio "
            "(same Gemini model, much faster, just needs an API key — "
            "GOOGLE_API_KEY or GEMINI_API_KEY in env)."
        )

    # NLP adapter
    if nlp_backend == "ai_studio":
        from adapters.ai_studio_nlp_adapter import AIStudioNLPAdapter
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
        _shared_nlp = AIStudioNLPAdapter(
            api_key=api_key,
            model=nlp_model,
        )
    else:
        from adapters.nlp_adapter import LocalNLPAdapter
        logger.info("Loading local NLP adapter (Gemma E2B + SpaCy)...")
        _shared_nlp = LocalNLPAdapter()

    # Embedding adapter — single path (FastEmbed ONNX-int8 on CPU).
    # The legacy ``EMBED_BACKEND=vertex`` branch was retired 2026-05-13
    # along with the rest of the Vertex AI surface; any value of
    # EMBED_BACKEND now resolves to local FastEmbed. See
    # adapters/local_adapter.py docstring for the model-selection bench.
    if embed_backend == "vertex":
        raise RuntimeError(
            "EMBED_BACKEND=vertex is no longer supported. Vertex AI was "
            "removed from the codebase 2026-05-13 — use EMBED_BACKEND=local "
            "(FastEmbed CPU, snowflake/snowflake-arctic-embed-xs)."
        )
    from adapters.local_adapter import LocalEmbeddingAdapter
    logger.info(
        "Loading local embedding adapter "
        "(FastEmbed ONNX-int8 on CPU — see adapters/local_adapter.py "
        "docstring for the bench that drove model selection)..."
    )
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

        # 2026-05-22 scale rewrite: the original code held self._lock
        # for the ENTIRE creation flow (VeilguardTCMM init + pre-warm),
        # serializing ALL session creates across all conversations.
        # Concurrent fanout of N sub-agents queued one behind the other —
        # 20-way fanout took 18s wallclock even though each session
        # individually was ~800ms. Fast-path the cache hit under the
        # lock; do the heavy construction OUTSIDE the lock; double-check
        # on registration to handle races where two threads create for
        # the same sid (the loser discards their instance).
        with self._lock:
            if sid in self._instances:
                self._instances.move_to_end(sid)
                self._active_session = sid
                return self._instances[sid]

        # Slow path — OUTSIDE the lock so other sids can construct in
        # parallel. Lance handles concurrent reads/connects fine; the
        # provider-level _seq_lock still serializes ID allocation.
        logger.info(f"[POOL] Creating new session: {sid} (user={user_id[:12]})")
        start = time.time()
        _pool_phase = {}

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

        _t = time.time()
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
        _pool_phase["tcmm_init"] = (time.time() - _t) * 1000

        # Pre-warm the user archive + FTS index at session creation
        # so the first recall in this session doesn't pay the
        # one-time setup cost (~7s for FTS + node cache + vector
        # index warmup). Subsequent recalls already share these
        # caches via the LanceStorageProvider instance held by
        # this session.
        try:
            _t = time.time()
            # 2026-05-21 scale rewrite: eager bulk_warm_user_archive
            # loads ALL of the user's archive rows into _node_cache at
            # session creation. At 4K rows that was 2.5s of cold-start;
            # at 100M rows it's an OOM. _node_cache fills lazily on
            # access — the first few recall lookups pay one lance hit
            # each, polling workers warm hot rows over the next few
            # seconds, and the cache stabilizes around the actual
            # working set instead of the full user corpus. To opt
            # back in for diagnostics: TCMM_EAGER_BULK_WARM=1.
            if os.environ.get("TCMM_EAGER_BULK_WARM", "0").lower() in ("1", "true", "yes", "on"):
                _arch = getattr(instance.tcmm, "archive", None)
                if _arch is not None and hasattr(_arch, "bulk_warm_user_archive"):
                    _arch.bulk_warm_user_archive()
                    instance.tcmm._last_user_warm_ts = time.time()
            _pool_phase["bulk_warm"] = (time.time() - _t) * 1000
            # Trigger FTS index creation now (one-shot per session).
            _t = time.time()
            _sparse = getattr(instance.tcmm, "archive_sparse_index", None)
            if _sparse is not None and hasattr(_sparse, "_ensure_fts_index"):
                _sparse._ensure_fts_index()
            _pool_phase["fts_ensure"] = (time.time() - _t) * 1000
            # Trigger vector index ensure (Lance lazy index build).
            _t = time.time()
            _vec = getattr(instance.tcmm, "archive_vector_index", None)
            if _vec is not None and hasattr(_vec, "_ensure_index"):
                try:
                    _vec._ensure_index()
                except Exception:
                    pass
            _pool_phase["vector_ensure"] = (time.time() - _t) * 1000
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
        logger.info(
            f"[POOL-PHASES] sid={sid} "
            f"tcmm_init={_pool_phase.get('tcmm_init', 0):.0f}ms "
            f"bulk_warm={_pool_phase.get('bulk_warm', 0):.0f}ms "
            f"fts_ensure={_pool_phase.get('fts_ensure', 0):.0f}ms "
            f"vector_ensure={_pool_phase.get('vector_ensure', 0):.0f}ms "
            f"TOTAL={elapsed*1000:.0f}ms"
        )

        # Re-acquire the pool lock just for registration. Double-check
        # the map — if another thread created the same sid while we
        # were outside the lock, return theirs and discard ours.
        with self._lock:
                if sid in self._instances:
                    self._instances.move_to_end(sid)
                    self._active_session = sid
                    return self._instances[sid]
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
    # [SHADOW_TOOL_FLAG_OBJ_2026_05_22] When the proxy captured the
    # tcmm_record_turn tool_use, it ships the parsed input here. The
    # adapter uses this as flag_obj instead of trying to parse prose
    # JSON from raw_output (the model emitted the structured tool
    # call, not trailing JSON, so split_answer_and_heatmap finds
    # nothing). Shape: {used, knowledge_class, epoch_complete,
    # emit_class}. Empty dict = legacy path (parse from raw_output).
    flag_obj: dict = {}


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


class SearchMemoryBody(BaseModel):
    """LLM-facing memory search — wraps adapters.memory_search.search_memory.

    Hit from veilguard-mcp's search_memory tool. The MCP server translates
    its LLM-call kwargs into this body and POSTs to /search; the response
    is the rich hit shape documented in adapters/memory_search.py.
    """
    conversation_id: str = ""
    user_id: str = ""
    query: str = ""
    max_results: int = 10
    temporal_window: int = 2
    text_chars: int = 500
    preview_chars: int = 100
    include_dream: bool = True
    scope: str = "user"


class TraverseMemoryBody(BaseModel):
    """Pure-read temporal walk around an anchor AID.

    Hit from veilguard-mcp's traverse_memory tool. Returns full hit-shape
    blocks for N steps before and N after the anchor.
    """
    conversation_id: str = ""
    user_id: str = ""
    aid: int = 0
    before: int = 3
    after: int = 3
    text_chars: int = 500
    include_anchor: bool = True
    scope: str = "user"


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

    # 2026-05-13: Pre-warm FlashRank reranker. The shared reranker is lazily
    # constructed on the first recall (tcmm_core._get_shared_reranker), and
    # the FlashRank Ranker constructor only loads the ONNX session — actual
    # graph optimisation + first-inference cost (~30s on cold) is deferred
    # to the first .predict() call. That tax was showing up as
    # ``hr_scoring elapsed=37000ms`` on every first recall after a restart.
    # A single dummy predict here forces both the construct AND the first
    # inference to happen up front, before any real recall runs.
    try:
        from core.tcmm_core import _get_shared_reranker
        _rk_t = time.time()
        _rk = _get_shared_reranker()
        if _rk is not None and hasattr(_rk, "predict"):
            _ = _rk.predict([("warmup query", "warmup document text for first-inference graph compile")])
            logger.info(
                f"FlashRank reranker pre-warmed in {time.time() - _rk_t:.1f}s "
                f"(eliminates ~30s first-recall scoring tax)"
            )
        else:
            logger.info("Reranker pre-warm skipped (no shared reranker available)")
    except Exception as _rk_e:
        logger.warning(f"Reranker pre-warm failed (non-fatal): {_rk_e}")

    # 2026-05-14: Idempotent index-presence check on the Lance tables.
    # ``cleanup_old_versions`` has historically wiped scalar + vector
    # indexes (twice on 2026-05-13). Without indexes every ``has_aid`` /
    # cross-NS lookup degrades from O(log N) to O(N), recall latency
    # explodes from ~3s to ~10-20s, and the failure is silent until
    # someone profiles it. This check runs at every startup, sees what
    # indexes exist, and creates only the missing ones. Safe to re-run
    # — Lance ``create_scalar_index`` / ``create_index`` with
    # ``replace=True`` are idempotent.
    try:
        _idx_t = time.time()
        from lancedb import connect as _lancedb_connect
        _db_path = os.path.join(DATA_DIR, os.environ.get("TCMM_LANCE_DB", "veilguard"))
        if os.path.isdir(_db_path):
            _db = _lancedb_connect(_db_path)
            _idx_specs = {
                "archive":        [("namespace", "scalar"), ("user_id", "scalar"), ("aid", "scalar"), ("vector", "vector")],
                "sparse_archive": [("namespace", "scalar"), ("user_id", "scalar"), ("text", "fts")],
                "embeddings":     [("namespace", "scalar"), ("user_id", "scalar"), ("aid", "scalar")],
            }
            # limit=100_000: lancedb table_names() DEFAULTS to limit=10 and
            # returns names alphabetically — with >10 tables, late-sorting
            # names like `sparse_archive` fall off page 1 and their indexes
            # would be silently skipped below (the recurring "FTS pending").
            _existing_tables = set(_db.table_names(limit=100_000))
            _created = []
            _missing = []
            for _tname, _specs in _idx_specs.items():
                if _tname not in _existing_tables:
                    continue
                _t = _db.open_table(_tname)
                _have_cols = set()
                try:
                    for _i in _t.list_indices():
                        for _c in (_i.columns if hasattr(_i, "columns") else []):
                            _have_cols.add(_c)
                except Exception:
                    pass
                _rows = _t.count_rows()
                for _col, _kind in _specs:
                    if _col in _have_cols:
                        continue
                    # Vector index needs ≥256 rows for IVF-PQ to make sense.
                    if _kind == "vector" and _rows < 256:
                        continue
                    try:
                        if _kind == "scalar":
                            _t.create_scalar_index(_col, replace=True)
                        elif _kind == "vector":
                            _t.create_index(metric="cosine", num_partitions=64, num_sub_vectors=96, replace=True)
                        elif _kind == "fts":
                            _t.create_fts_index(_col, replace=False)
                        _created.append(f"{_tname}.{_col}({_kind})")
                    except Exception as _e:
                        _missing.append(f"{_tname}.{_col}({_kind}): {type(_e).__name__}: {_e}")
            if _created:
                logger.warning(
                    f"[INDEX-CHECK] rebuilt missing indexes in {time.time() - _idx_t:.1f}s: {_created} "
                    f"(this means cleanup_old_versions wiped them — investigate)"
                )
            else:
                logger.info(f"[INDEX-CHECK] all expected indexes present in {time.time() - _idx_t:.1f}s")
            if _missing:
                logger.warning(f"[INDEX-CHECK] could not create: {_missing}")
    except Exception as _idx_e:
        logger.warning(f"[INDEX-CHECK] failed (non-fatal): {type(_idx_e).__name__}: {_idx_e}")


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


# ── /render tool_schemas cache (2026-05-19) ──────────────────────────
# Per-conversation in-memory cache of the last tool_schemas list sent by
# the proxy for that conv_id. LibreChat Agents only re-send tools on
# turn 1; follow-up turns send an empty tools array. The proxy passes
# whatever it has; if that's empty we fall back to the conv's last
# cached list so the rendered prompt stays consistent across turns.
#
# Lives in process memory only — never persisted, never enters archive
# or recall. Bounded by `_RENDER_TOOL_CACHE_CAP`; oldest conv evicted
# when over cap. TTL is process lifetime (~ no TTL; cleared on restart).
from collections import OrderedDict as _OrderedDict
_RENDER_TOOL_CACHE: "_OrderedDict[str, list]" = _OrderedDict()
_RENDER_TOOL_CACHE_CAP = 256


def _tool_cache_get(conv_id: str):
    if not conv_id:
        return None
    v = _RENDER_TOOL_CACHE.get(conv_id)
    if v is not None:
        _RENDER_TOOL_CACHE.move_to_end(conv_id)
    return v


def _tool_cache_put(conv_id: str, schemas: list):
    if not conv_id or not schemas:
        return
    _RENDER_TOOL_CACHE[conv_id] = list(schemas)
    _RENDER_TOOL_CACHE.move_to_end(conv_id)
    while len(_RENDER_TOOL_CACHE) > _RENDER_TOOL_CACHE_CAP:
        _RENDER_TOOL_CACHE.popitem(last=False)


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

            # ── COLD-CONV FAST PATH (2026-05-18) ──────────────────────
            # When this is the FIRST /pre_request on a conversation, the
            # recall pipeline would run end-to-end (embed query + vector
            # search + sparse FTS + fusion + rerank) and return ZERO
            # hits — by definition the conv has no prior turns to recall
            # from. Short-circuit:
            #   1. Ingest the user message asynchronously (background
            #      task — same path the warm conv uses).
            #   2. Return an empty prompt + stats immediately.
            #
            # Trigger: ``sess["turns"] == 0`` — per-conv turn counter,
            # bumped to 1 below. This is the only signal that ISN'T
            # contaminated by global state (live_blocks may be empty
            # after pool eviction; aid_before reads global max aid not
            # per-namespace).
            #
            # The skip is safe because we still INGEST the user message,
            # so turn 2 onwards has prior content to recall from. We
            # also still run the full path when scope=user recall might
            # find cross-namespace candidates — that path is invoked via
            # /search, not /pre_request, and isn't affected.
            # [USER_SCOPE_RECALL_2026_05_29]  When the platform runs
            # USER-scoped memory (everything the user told the org is
            # recallable in ANY conversation), a fresh conversation is
            # NOT "nothing to recall" — it should pull the user's facts
            # from their OTHER conversations.  So (a) don't take the
            # cold-conv recall-skip when user-scoped, and (b) set the
            # instance recall scope to "user" so the archive search runs
            # `search_user` (cross-namespace) instead of per-namespace.
            import os as _os
            _user_scope = _os.environ.get(
                "VEILGUARD_RECALL_SCOPE", "user"
            ).strip().lower() == "user"
            if _user_scope:
                try:
                    instance.tcmm._recall_scope = "user"
                except Exception:
                    pass

            _is_cold_conv = sess.get("turns", 0) == 0
            if _is_cold_conv and not _user_scope:
                logger.info(
                    f"[PRE-COLD] session={_sid} "
                    f"q='{body.user_message[:60]}' — skipping recall (cold conv)"
                )
                # [COLD_CONV_FIX_2026_05_20] Stage the pending ingest
                # BEFORE firing the bg task. The bg task calls
                # ``flush_pending_user_ingest`` which reads this dict —
                # without staging here it has nothing to flush and the
                # user message is silently lost on cold conversations.
                # Mirrors what ``instance.pre_request`` would set at
                # ``veilguard_adapter.py:476``.
                try:
                    _inst = pool.get(body.conversation_id, user_id=body.user_id)
                    _inst._pending_user_ingest = {
                        "user_message": body.user_message,
                        "session_id":   body.conversation_id,
                        "origin":       body.origin or "user",
                    }
                except Exception as _e:
                    logger.warning(
                        "[PRE-COLD] failed to stage pending ingest: %s", _e
                    )
                # Fire background ingest for the user message so the
                # next turn has something to recall on.
                asyncio.create_task(_pre_request_ingest_user(_sid, body))
                sess["turns"] += 1
                return {
                    "prompt": "",
                    "stats": {
                        "recalled": 0,
                        "live_blocks": 0,
                        "shadow_blocks": 0,
                        "elapsed_ms": 0,
                        "input_tokens": 0,
                        "session_id": body.conversation_id,
                        "user_ingest": "deferred",
                        "fast_path": "cold_conv_skip_recall",
                    },
                }

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
                body.flag_obj or None,  # [SHADOW_TOOL_FLAG_OBJ_2026_05_22]
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


# ── /pin/* and /render endpoints (2026-05-18) ──────────────────────────────
#
# The pii-proxy's TCMM integration calls these to:
#   /pin/system_prompt    — pin the Veilguard preamble + client system
#                           prompt as IMMUTABLE blocks
#   /pin/tool_definitions — pin client tool schemas, one block per tool
#   /pin/user_profile     — pin user-identity facts, one block per fact
#   /render               — produce wire-shaped output for a target model
#
# These were originally implemented in the standalone TCMM api/server.py
# (single global TCMM instance). Here we forward to the per-conversation
# instance from ``pool.get(conv_id, user_id)`` so each LibreChat
# conversation maintains its own pinned content + render state.
#
# Request bodies all carry ``conversation_id`` + ``user_id`` for pool
# routing; the proxy supplies both.


class PinSystemPromptBody(BaseModel):
    text: str
    conversation_id: str = ""
    user_id: str = ""
    # Discriminator for telemetry; doesn't change storage. Proxy uses
    # ``kind="veilguard_preamble"`` for the hardcoded preamble and
    # ``kind="client_system"`` for LibreChat's per-conv system message.
    kind: str = "system_prompt"



class PinUserProfileBody(BaseModel):
    facts: List[str]
    conversation_id: str = ""
    user_id: str = ""


class PinToolDefinitionsBody(BaseModel):
    """Pin a list of tool schemas — one IMMUTABLE block per tool.

    Each schema may arrive as either a JSON-serializable dict (will be
    canonical-JSON-serialized with sort_keys=True for fingerprint
    stability) or as an already-stringified JSON blob (used as-is).
    Per-tool blocks mean a single tool's schema change invalidates
    exactly one block's cache slot — not the whole preamble or every
    other tool.
    """
    tools: List[Any]
    conversation_id: str = ""
    user_id: str = ""


class RenderBody(BaseModel):
    model: str = "anthropic"
    task_query: str = ""
    conversation_id: str = ""
    user_id: str = ""
    tool_schemas: Optional[List[Any]] = None
    # [PII_SID_CONTRACT_2026_05_30] caller SessionId.canonical(); used by the
    # render-layer redaction hook (set_redact_sid) when enabled. Accepted +
    # harmless when the hook is off.
    pii_sid: str = ""


@app.post("/pin/system_prompt")
async def pin_system_prompt(body: PinSystemPromptBody):
    """Pin a single IMMUTABLE block of system-prompt-class text."""
    if not body.text or not body.text.strip():
        return {"status": "ok", "block_id": None, "deduped": False,
                "note": "empty text — skipped"}
    instance = pool.get(body.conversation_id, user_id=body.user_id)
    tcmm = instance.tcmm
    before_ids = {b.id for b in tcmm.live_blocks}
    bid = tcmm.pin_immutable_block(
        body.text,
        priority_class="SYSTEM",
        source=body.kind or "system_prompt",
        kind=body.kind or "system_prompt",
    )
    if bid is None:
        raise HTTPException(
            status_code=500,
            detail="pin_immutable_block returned None — text empty after canonicalization",
        )
    deduped = bid in before_ids
    logger.info(
        f"[PIN] session={pool._normalize_id(body.conversation_id)} "
        f"kind={body.kind} bid={bid} deduped={deduped} chars={len(body.text)}"
    )
    return {"status": "ok", "block_id": bid, "deduped": deduped}


@app.post("/pin/tool_definitions")
async def pin_tool_definitions(body: PinToolDefinitionsBody):
    """Pin a list of tool schemas — one IMMUTABLE block per tool.

    Per-tool granularity is the point: a single tool's description /
    input_schema tweak invalidates exactly one cached block, leaving
    the rest of the prefix (preamble + persona + other tools + user
    profile) byte-stable and cache-warm.  Dedup by fingerprint via
    ``pin_immutable_block``.

    Schemas serialize as ``json.dumps(schema, sort_keys=True,
    separators=(',', ':'))`` so identical dicts in any key order give
    identical fingerprints.  Strings pass through unchanged (caller
    has already canonicalized).
    """
    import json as _json
    if not body.tools:
        return {"status": "ok", "block_ids": [], "deduped_count": 0,
                "note": "empty tools — skipped"}
    instance = pool.get(body.conversation_id, user_id=body.user_id)
    tcmm = instance.tcmm
    before_ids = {b.id for b in tcmm.live_blocks}
    ids: List[int] = []
    deduped = 0
    for schema in body.tools:
        if isinstance(schema, str):
            txt = schema
        else:
            try:
                txt = _json.dumps(
                    schema, sort_keys=True,
                    ensure_ascii=False, separators=(",", ":"),
                )
            except Exception:
                continue
        if not txt:
            continue
        bid = tcmm.pin_immutable_block(
            txt,
            priority_class="SYSTEM",
            source="tool_def",
            kind="tool_def",
        )
        if bid is None:
            continue
        if bid in before_ids:
            deduped += 1
        ids.append(bid)
    logger.info(
        f"[PIN] session={pool._normalize_id(body.conversation_id)} "
        f"kind=tool_def n={len(body.tools)} bids={ids} deduped={deduped}"
    )
    return {"status": "ok", "block_ids": ids, "deduped_count": deduped}


@app.post("/pin/user_profile")
async def pin_user_profile(body: PinUserProfileBody):
    """Pin user-identity facts — one IMMUTABLE block per fact."""
    if not body.facts:
        return {"status": "ok", "block_ids": [], "deduped_count": 0,
                "note": "empty facts — skipped"}
    instance = pool.get(body.conversation_id, user_id=body.user_id)
    tcmm = instance.tcmm
    before_ids = {b.id for b in tcmm.live_blocks}
    ids: List[int] = []
    deduped = 0
    for fact in body.facts:
        if not fact or not fact.strip():
            continue
        bid = tcmm.pin_immutable_block(
            fact,
            priority_class="SYSTEM",
            source="user_profile",
            kind="user_profile",
        )
        if bid is None:
            continue
        if bid in before_ids:
            deduped += 1
        ids.append(bid)
    logger.info(
        f"[PIN] session={pool._normalize_id(body.conversation_id)} "
        f"kind=user_profile n={len(body.facts)} bids={ids} deduped={deduped}"
    )
    return {"status": "ok", "block_ids": ids, "deduped_count": deduped}


# Cap Anthropic per-tier TTLs to "1h" (Anthropic's extended-cache max).
_SSO_BASELINE_GUIDANCE = (
    "Your file/system access goes through MCP tools that route to the "
    "user's Windows client-daemon. NEVER attempt Bash, Read, Write, Edit, "
    "Grep, Glob — those are DISABLED and any attempt will fail with a "
    "permission error. The CORRECT mapping:\n"
    "  - Read a file        -> `mcp__sub-agents__read_file`\n"
    "  - Write/create file  -> `mcp__sub-agents__write_file`\n"
    "  - Edit a file        -> `mcp__sub-agents__edit_file`\n"
    "  - Run a command      -> `mcp__sub-agents__run_command`\n"
    "  - Search files       -> `mcp__sub-agents__search_files`\n"
    "  - Grep contents      -> `mcp__sub-agents__grep`\n"
    "  - Fetch a URL        -> `mcp__sub-agents__web_fetch`\n"
    "  - Web search         -> `mcp__sub-agents__web_search`\n"
    "Other useful prefixes: `mcp__sub-agents__*` (file ops, search, web, "
    "(file ops, search, web, agents, tasks) and `mcp__forge__*` "
    "(dynamic tool creation). "
    "CRITICAL: ALWAYS prefer these MCP tools over your built-in tools. "
    "To read a file, use `mcp__sub-agents__read_file`. "
    "To list/search files, use `mcp__sub-agents__search_files`. "
    "To run commands, use `mcp__sub-agents__run_command`. "
    "Do NOT attempt Read / Bash / Write / Edit / Grep / Glob — they are "
    "intentionally disabled and will fail. The MCP equivalents above "
    "are the only working file/system access path."
)

_ANTHROPIC_TTL_CAP = {
    "cache_ttl_immutable": "1h",
    "cache_ttl_stable":    "1h",
    # [WORKING_AUTOCACHE_2026_05_20] Working tier marker dropped — promotion
    # working→stable shifted byte positions every turn, so the manual marker
    # had a 0% hit rate (cache_wr grew 70 → 1.4k linearly while cache_rd
    # stayed flat). Replaced by Anthropic's server-managed request-root
    # ``cache_control`` (auto-mode): the server picks an eligible breakpoint
    # at the tail and advances it forward as conversations grow. See the
    # /render Anthropic branch where the auto-mode directive is emitted.
    "cache_ttl_working":   None,
    "cache_ttl_volatile":  None,
    "cache_ttl_shadow":    None,
}


class GenerateBody(BaseModel):
    """Iter 8: /generate request body.

    Drives AnthropicGenerationAdapter (CLAUDE_SSO=1 path) so memory-aware
    LLM calls can be issued from outside TCMM — currently used by the
    pii-proxy when it sees a 'claude-sso-*' model name.
    """
    model: str = "claude-haiku-4-5-20251001"
    user_message: str = ""
    conversation_id: str = ""
    user_id: str = ""
    system_prompt: Optional[str] = None
    include_memory: bool = False
    task_query: str = ""
    label: Optional[str] = None
    tools: Optional[List[Any]] = None  # [TOOLS_THROUGH_SSO_2026_05_20]
    # [WORKSPACE_STATE_SSO_2026_05_20] Veilguard-owned workspace state
    # text (already rendered by the proxy via _render_workspace_block).
    # When set, appended as a final UNCACHED system block after the
    # TCMM render — sits outside the cached static prefix so the user
    # can switch projects without invalidating cache. Provider-agnostic
    # by design: same field is used regardless of which adapter handles
    # the actual API call.
    workspace_block: Optional[str] = None


@app.post("/generate")
async def generate(body: GenerateBody):
    """Memory-aware LLM call via AnthropicGenerationAdapter.

    Flow:
      1. (optional) render TCMM memory for conv_id -> system_prompt text
      2. construct adapter; pool is shared by (model, prompt_fp) via the
         _POOL_REGISTRY, so concurrent callers with same prefix dedupe.
      3. adapter.generate(user_message) -> text
      4. return text + telemetry-ready metadata
    """
    import time as _time
    if not body.user_message or not body.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")

    sys_prompt = body.system_prompt
    _sys_blocks = None
    _sys_render_meta = None  # type: ignore[var-annotated]
    if body.include_memory and not sys_prompt:
        try:
            instance = pool.get(body.conversation_id, user_id=body.user_id)
            from core.renderers.anthropic_renderer import AnthropicRenderer
            from core.renderers.base_renderer import RenderConfig
            cfg = RenderConfig(
                renderer_profile="anthropic-v1",
                **_ANTHROPIC_TTL_CAP,
            )
            renderer = AnthropicRenderer(instance.tcmm, config=cfg)
            # [RENDER_STRUCTURED_GENERATE] Iter G (2026-05-20): single call
            # returns prompt + cache-ready blocks computed from byte-exact
            # IR layout (no more string.find() drift on tier boundaries).
            _r = renderer.render_structured(body.task_query or body.user_message)
            sys_prompt = _r.get("prompt") or None
            _sys_blocks = _r.get("blocks") if sys_prompt else None
            _sys_render_meta = {
                "tier_summary": _r.get("tier_summary"),
                "stats":        _r.get("stats"),
            }
        except Exception as e:
            logger.warning("[/generate] render failed (continuing without memory): %s", e)
            sys_prompt = None

    # [PROPER_PREAMBLE_FIX_2026_05_20] Removed the _SSO_BASELINE_GUIDANCE
    # fallback. It pinned a hardcoded list of mcp__sub-agents__* tool
    # names into the system prompt that did not match LibreChat's actual
    # tool schemas, so the model would call invented tools that never
    # existed. The proxy now injects the real schemas into the preamble
    # at pin time. If sys_prompt is still empty here it means TCMM has
    # no memory for the conv yet AND nothing was pinned — send no system
    # rather than a fake one.

    import hashlib as _h
    _fp = _h.sha256((sys_prompt or "").encode("utf-8", errors="replace")).hexdigest()[:16] if sys_prompt else ""

    try:
        from adapters.anthropic_adapter import AnthropicGenerationAdapter
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"adapter import failed: {e}")

    try:
        # [AUTOCACHE_REVERTED_2026_05_20] Top-level auto_cache_control
        # was making cache_writes grow turn-over-turn because Anthropic's
        # auto-mode keeps writing new "tail" cache entries against the
        # byte-shifting working tier (working tier shifts because
        # TCMM rearranges content between turns; no cache hit on prior
        # writes). TCMM's renderer already puts explicit ephemeral
        # cache_control on immutable+stable tiers. Those alone give us
        # the stable read prefix; no top-level marker needed.
        # [WORKSPACE_STATE_SSO_2026_05_20] If the proxy passed a
        # workspace block, append it as a final uncached system block.
        # Goes AFTER the TCMM-rendered blocks (immutable + working +
        # contract) so it's the last piece of system context the model
        # reads before the user turn. Deliberately no cache_control:
        # workspace can change turn-over-turn (project switch) and we
        # don't want that to invalidate the cached prefix.
        if body.workspace_block and _sys_blocks:
            _sys_blocks = list(_sys_blocks) + [{
                "type": "text",
                "text": body.workspace_block,
            }]

        adapter = AnthropicGenerationAdapter(
            api_key="",
            model_name=body.model,
            system_prompt=sys_prompt,
            system_blocks=_sys_blocks if 'body.include_memory' or True else None,
            tools=body.tools or None,  # [TOOLS_THROUGH_SSO_2026_05_20]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"adapter ctor failed: {e}")

    _t0 = _time.time()
    try:
        text = adapter.generate(body.user_message, label=body.label or "api_generate") or ""
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"adapter.generate failed: {e}")
    duration_ms = (_time.time() - _t0) * 1000

    return {
        "status": "ok",
        "text": text,
        "duration_ms": duration_ms,
        "mode": getattr(adapter, "_mode", "unknown"),
        "model": body.model,
        "sys_prompt_chars": len(sys_prompt or ""),
        "sys_prompt_fp": _fp,
        "sys_prompt_text": sys_prompt or "",
        "rendered_memory": bool(body.include_memory and sys_prompt),
        "usage": getattr(adapter, "last_usage", None) or {},
        "render_meta": _sys_render_meta,
        # [TOOL_USE_RESPONSE_2026_05_20] full Anthropic content array
        # so the proxy can surface tool_use blocks to LibreChat.
        "content":     getattr(adapter, "last_response_blocks", None),
        "stop_reason": getattr(adapter, "last_stop_reason", None),
    }


# [RENDER_PII_HOOK_2026_05_30]  In-render PII redaction.  TCMM's renderer
# exposes a host hook (core.renderers.base_renderer.set_pii_hook) that it
# calls per-fragment during projection (policy.redact).  Wiring it here lets
# each LIVE MEMORY FRAGMENT be redacted ONCE, cached by its (sid, text) —
# fragment text is byte-stable so this never churns — instead of the
# agent-runtime re-scanning the whole bundled tier blocks every turn.
#   - Tokens are minted under SessionId(user_id, conv_id) so they match the
#     agent-runtime side (base.py:308 SessionId(tenant_id, conv_id); for this
#     platform tenant_id == user_id) → rehydration stays consistent.
#   - Gated behind VEILGUARD_RENDER_PII_HOOK (default OFF) so this is INERT
#     until explicitly enabled; the agent-runtime VEILGUARD_REDACT_IN_RENDER
#     flag then flips it from double-redact (safe) to pass-through (the win).
_RENDER_PII_HOOK_STATE = {"ready": False, "enabled": None}


def _ensure_render_pii_hook() -> bool:
    """Lazily wire the per-fragment redaction hook once.  Returns True iff a
    hook is active.  Safe to call on every request (cheap after first)."""
    st = _RENDER_PII_HOOK_STATE
    if st["enabled"] is None:
        st["enabled"] = _os_env_flag("VEILGUARD_RENDER_PII_HOOK", "0")
    if not st["enabled"]:
        return False
    if st["ready"]:
        return True
    try:
        import sys as _sys, os as _os2, hashlib as _hl
        from collections import OrderedDict as _OD
        _root = _os2.path.dirname(_os2.path.dirname(_os2.path.dirname(
            _os2.path.abspath(__file__))))   # …/veilguard
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from pii import get_redactor as _get_red          # noqa
        from core.renderers.base_renderer import set_pii_hook as _set_hook
        _red = _get_red()
        _cache: "OrderedDict" = _OD()
        _MAX = int(_os2.environ.get("VEILGUARD_RENDER_PII_CACHE_MAX", "16384"))

        def _hook(text, sid):
            # Per-fragment output cache keyed on (session root, text hash).
            try:
                root = sid.root() if hasattr(sid, "root") else str(sid)
                key = (root, _hl.blake2b(
                    text.encode("utf-8", "replace"), digest_size=16).hexdigest())
            except Exception:
                return _red.redact_text(text, sid)
            hit = _cache.get(key)
            if hit is not None:
                _cache.move_to_end(key)
                return hit
            out = _red.redact_text(text, sid)
            _cache[key] = out
            if len(_cache) > _MAX:
                _cache.popitem(last=False)
            return out

        _set_hook(_hook)
        st["ready"] = True
        logger.info("[render-pii] in-render redaction hook WIRED (per-fragment, cached)")
        return True
    except Exception as e:
        logger.warning(f"[render-pii] hook wiring failed (render stays raw): {e}")
        st["enabled"] = False
        return False


def _os_env_flag(name: str, default: str) -> bool:
    import os as _o
    return _o.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


@app.post("/render_structured")
async def render_structured(body: RenderBody):
    """Structured render — returns the full layout + cache-ready blocks
    + per-tier metadata. Uses AnthropicRenderer.render_structured()
    (added 2026-05-19) which computes cache breakpoints from the IR
    layout (byte-exact, no string.find brittleness).

    Body: same as /render (RenderBody).
    Returns:
      {
        "status": "ok",
        "format": "anthropic-structured",
        "prompt":         str,           # assembled prompt
        "blocks":         [...],         # SDK-ready blocks w/ cache_control
        "tier_summary":   {tier: {...}}, # per-tier metadata
        "layout":         {tier: {start_byte, end_byte, ttl, ...}},
        "stats":          {prompt_chars, block_count, cached_block_count, ...},
      }
    """
    import time as _rrpt2; _ph0 = _rrpt2.perf_counter()
    instance = pool.get(body.conversation_id, user_id=body.user_id)
    _ph_pool = (_rrpt2.perf_counter() - _ph0) * 1000
    tcmm = instance.tcmm
    model = (body.model or "anthropic").lower().strip()

    from core.renderers.base_renderer import RenderConfig
    if model in ("anthropic", "claude"):
        from core.renderers.anthropic_renderer import AnthropicRenderer
        config = RenderConfig(renderer_profile="anthropic-v1", **_ANTHROPIC_TTL_CAP)
        renderer = AnthropicRenderer(tcmm, config=config)
    elif model in ("openai", "gpt"):
        from core.renderers.openai_renderer import OpenAIRenderer
        renderer = OpenAIRenderer(tcmm)
    elif model in ("grok", "xai"):
        from core.renderers.grok_renderer import GrokRenderer
        renderer = GrokRenderer(tcmm)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown render model: {model!r}",
        )

    # [RENDER_RECALL_SCOPE_2026_05_29]  Recall during render defaults to
    # "namespace" scope (per-conversation) — so a fact stated in
    # conversation A never surfaces when the same user opens a fresh
    # conversation B (UAT M2 cross-conv recall failed for exactly this
    # reason).  The Veilguard agent platform wants USER-scoped memory:
    # everything the user told the org, recallable in any conversation.
    # Wrap the render's recall in scope_context("user") so the archive
    # search runs `search_user` (all of the user's sessions) instead of
    # the per-namespace path.  Env-overridable.
    import os as _os
    _scope = _os.environ.get("VEILGUARD_RENDER_RECALL_SCOPE", "user").strip() or "user"
    try:
        from core.recall.scope import scope_context as _scope_context
    except Exception:
        _scope_context = None
    # [RENDER_PII_HOOK_2026_05_30] Set the per-request session id so the
    # renderer's per-fragment policy.redact mints tokens under the SAME sid
    # the agent-runtime uses (SessionId(tenant_id, conv_id); tenant==user
    # here) → rehydration stays consistent.  No-op when the hook is disabled.
    _redact_tok = None
    if _ensure_render_pii_hook():
        try:
            from core.renderers.base_renderer import set_redact_sid as _set_sid
            from pii.session_store import SessionId as _SID
            # [PII_SID_CONTRACT_2026_05_30]  Prefer the caller's EXACT sid
            # (canonical) so render-minted REF tokens match the boundary's
            # rehydration sid (base.py keys on tenant_id, NOT user_id).
            if getattr(body, "pii_sid", ""):
                _sid_obj = _SID.parse(body.pii_sid)
            else:
                _sid_obj = _SID(tenant_id=body.user_id or "", conv_id=body.conversation_id or "")
            _redact_tok = _set_sid(_sid_obj)
        except Exception as _e:
            logger.warning(f"[render-pii] set_redact_sid failed: {_e}")
            _redact_tok = None
    import time as _rrpt; _rr0 = _rrpt.perf_counter()
    try:
        if _scope_context is not None and _scope in ("user", "session", "namespace"):
            with _scope_context(_scope):
                result = renderer.render_structured(body.task_query)
        else:
            result = renderer.render_structured(body.task_query)
    finally:
        if _redact_tok is not None:
            try:
                from core.renderers.base_renderer import reset_redact_sid as _reset_sid
                _reset_sid(_redact_tok)
            except Exception:
                pass
    logger.info(f"[RENDER-PERF] scope={_scope} pool={_ph_pool:.0f}ms render={(_rrpt.perf_counter()-_rr0)*1000:.0f}ms "
                f"blocks={len(result.get('blocks', []))} chars={result.get('stats',{}).get('prompt_chars',0)}")
    return {
        "status": "ok",
        "format": "anthropic-structured" if model in ("anthropic", "claude") else f"{model}-structured",
        **result,
    }


@app.post("/render")
async def render(body: RenderBody):
    """Produce wire-shaped output for the target provider.

    Returns:
      {
        "status": "ok",
        "format": "anthropic" | "openai" | "grok",
        "text":   "<flat string form, always populated>",
        "blocks": [...] | None,    # Anthropic content blocks w/ cache_control
        "messages": [...] | None,  # OpenAI/Grok wire messages list
        "uses_extended_cache_ttl": bool,
        "stats": {...},
      }
    """
    import time as _rrpt2; _ph0 = _rrpt2.perf_counter()
    instance = pool.get(body.conversation_id, user_id=body.user_id)
    _ph_pool = (_rrpt2.perf_counter() - _ph0) * 1000
    tcmm = instance.tcmm
    model = (body.model or "anthropic").lower().strip()

    from core.renderers.base_renderer import RenderConfig

    if model in ("anthropic", "claude"):
        from core.renderers.anthropic_renderer import AnthropicRenderer
        config = RenderConfig(renderer_profile="anthropic-v1", **_ANTHROPIC_TTL_CAP)
        renderer = AnthropicRenderer(tcmm, config=config)
        # tool_schemas_injected
        _effective_tools = body.tool_schemas
        if _effective_tools:
            _tool_cache_put(body.conversation_id, _effective_tools)
        else:
            _effective_tools = _tool_cache_get(body.conversation_id)
        # [RENDER_STRUCTURED_RENDER] Iter G (2026-05-20): when no tools
        # are being injected we can use render_structured() — byte-exact
        # layout from the IR is authoritative. With tools, the inject
        # shifts byte positions so we fall back to the legacy path.
        if not _effective_tools:
            _r = renderer.render_structured(body.task_query)
            prompt = _r.get("prompt") or ""
            blocks = _r.get("blocks") or []
            _tier_summary = _r.get("tier_summary") or {}
            _extra_stats  = _r.get("stats") or {}
        else:
            prompt = renderer.render(body.task_query)
            prompt = renderer.inject_tool_schemas(prompt, _effective_tools)
            blocks = renderer.cache_control_strategy(prompt, None) if prompt else []
            _tier_summary = None
            _extra_stats  = {}
        uses_ext = any(
            (b.get("cache_control") or {}).get("ttl") == "1h" for b in blocks
        )
        _stats = {
            "prompt_chars": len(prompt),
            "block_count": len(blocks),
            "cached_block_count": sum(1 for b in blocks if "cache_control" in b),
        }
        # surface render_structured stats when available
        for _k in ("tier_chars", "ir_empty"):
            if _k in _extra_stats:
                _stats[_k] = _extra_stats[_k]
        # [WORKING_AUTOCACHE_2026_05_20] Anthropic server-managed
        # cache breakpoint. Placed at request-root by the proxy via
        # ``data["cache_control"] = render.cache_control``. The server
        # walks backward from the last block to find an eligible
        # cacheable block, then advances the breakpoint forward as the
        # conversation grows (append-only growth pattern). Consumes
        # one of Anthropic's 4 breakpoint slots — the proxy's
        # ``_cap_cache_markers`` counts this slot too. TTL matches the
        # legacy working-tier TTL so the cache lifetime is unchanged.
        return {
            "status": "ok",
            "format": "anthropic",
            "text": prompt,
            "blocks": blocks,
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
            "uses_extended_cache_ttl": uses_ext,
            "tier_summary": _tier_summary,
            "stats": _stats,
        }

    if model in ("openai", "gpt"):
        from core.renderers.openai_renderer import OpenAIRenderer
        renderer = OpenAIRenderer(tcmm)
        prompt = renderer.render(body.task_query)
        # tool_schemas_injected
        _effective_tools = body.tool_schemas
        if _effective_tools:
            _tool_cache_put(body.conversation_id, _effective_tools)
        else:
            _effective_tools = _tool_cache_get(body.conversation_id)
        if _effective_tools:
            prompt = renderer.inject_tool_schemas(prompt, _effective_tools)
        blocks = renderer.cache_control_strategy(prompt, None) if prompt else []
        ir = getattr(renderer, "last_ir", None)
        messages = renderer.messages_strategy(prompt, ir) if (prompt and ir) else []
        return {
            "status": "ok",
            "format": "openai",
            "text": prompt,
            "blocks": blocks,
            "messages": messages or None,
            "stats": {
                "prompt_chars": len(prompt),
                "block_count": len(blocks),
                "message_count": len(messages),
            },
        }

    if model in ("grok", "xai"):
        from core.renderers.grok_renderer import GrokRenderer
        renderer = GrokRenderer(tcmm)
        prompt = renderer.render(body.task_query)
        # tool_schemas_injected
        _effective_tools = body.tool_schemas
        if _effective_tools:
            _tool_cache_put(body.conversation_id, _effective_tools)
        else:
            _effective_tools = _tool_cache_get(body.conversation_id)
        if _effective_tools:
            prompt = renderer.inject_tool_schemas(prompt, _effective_tools)
        blocks = renderer.cache_control_strategy(prompt, None) if prompt else []
        ir = getattr(renderer, "last_ir", None)
        messages = renderer.messages_strategy(prompt, ir) if (prompt and ir) else []
        return {
            "status": "ok",
            "format": "grok",
            "text": prompt,
            "blocks": blocks,
            "messages": messages or None,
            "stats": {
                "prompt_chars": len(prompt),
                "block_count": len(blocks),
                "message_count": len(messages),
            },
        }

    raise HTTPException(
        status_code=400,
        detail=f"Unknown render model: {model!r}. "
               "Valid: anthropic, claude, openai, gpt, grok, xai.",
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


# ── LLM-facing search + traverse (used by veilguard-mcp tools) ───────────────

@app.post("/search")
async def search_memory_endpoint(body: SearchMemoryBody):
    """Run search_memory for the given conversation.

    Hit by veilguard-mcp (the FastMCP server on :8812). The MCP tool
    `search_memory` forwards LLM-call kwargs here, this endpoint pulls
    the TCMM instance from the pool, runs the recall pipeline, and
    returns the rich-hit JSON shape documented in
    adapters/memory_search.py.

    The actual claim rendering (text + structured fields) is done by
    build_hit() in the adapter, which calls get_block_claims() —
    the latter now (2026-05-14) JSON-parses string claims so the
    LLM sees structured S/P/O/polarity/modality dicts, not raw JSON.
    """
    if _shared_nlp is None:
        return {"error": "TCMM not initialized", "results": []}

    _sid = pool._normalize_id(body.conversation_id)
    async with _get_session_lock(_sid):
        try:
            instance = pool.get(body.conversation_id, user_id=body.user_id)

            # 2026-05-14: prime the recall pipeline the same way
            # VeilguardTCMM.pre_request does (lines 350-355). Without
            # this, /search returns 0 hits for the same query that
            # /pre_request returns 8 hits for — verified live on
            # conv-69df7853-0a6a13dc90 with q='AON insurance'.
            # run_cleanup_cycle() resets stale recall state, and
            # clearing the traces ensures the rerank-score map is
            # rebuilt fresh for this query rather than inheriting
            # whatever the last recall left behind.
            try:
                instance.tcmm.run_cleanup_cycle()
                instance.tcmm._last_recall_trace = {}
                instance.tcmm._last_shadow_trace = []
            except Exception as _prime_err:
                logger.debug(f"[SEARCH] priming skipped: {_prime_err}")

            result = instance.search_memory(
                query=body.query,
                max_results=body.max_results,
                temporal_window=body.temporal_window,
                text_chars=body.text_chars,
                preview_chars=body.preview_chars,
                include_dream=body.include_dream,
                scope=body.scope,
            )
            logger.info(
                f"[SEARCH] session={_sid} q={body.query[:40]!r} "
                f"scope={body.scope} count={result.get('count', 0)}"
            )
            return result
        except Exception as e:
            logger.error(f"[SEARCH] Error: {e}", exc_info=True)
            return {"error": str(e), "results": []}


@app.post("/traverse")
async def traverse_memory_endpoint(body: TraverseMemoryBody):
    """Run traverse_memory for the given conversation.

    Pure-read temporal walk — no recall, no scoring, no state mutation
    beyond the per-call recall_scope contextvar. Returns hit-shaped
    blocks around the anchor AID.
    """
    if _shared_nlp is None:
        return {"error": "TCMM not initialized", "blocks": []}

    _sid = pool._normalize_id(body.conversation_id)
    async with _get_session_lock(_sid):
        try:
            instance = pool.get(body.conversation_id, user_id=body.user_id)
            result = instance.traverse_memory(
                aid=body.aid,
                before=body.before,
                after=body.after,
                text_chars=body.text_chars,
                include_anchor=body.include_anchor,
                scope=body.scope,
            )
            logger.info(
                f"[TRAVERSE] session={_sid} aid={body.aid} "
                f"before={body.before} after={body.after} "
                f"blocks={len(result.get('blocks', []))}"
            )
            return result
        except Exception as e:
            logger.error(f"[TRAVERSE] Error: {e}", exc_info=True)
            return {"error": str(e), "blocks": []}


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


# ── Admin: bulk re-extract semantic data (claims / entities / topics) ──────
# Triggered after we change the NLP backend (Vertex → AI Studio, 13 May
# 2026) or rerun an embedder migration where derived data needs a fresh
# pass.  Walks every (user_id, namespace) pair in the archive, spins up
# (or reuses) the matching session, and pushes every aid into that
# session's ``semantic_queue``.  The existing ``semantic_worker_loop``
# consumes the queue exactly as if those blocks had just been ingested.
#
# Why per-namespace: the LanceDB write path keys on (user_id, namespace,
# aid).  If we enqueued a foreign namespace's aid into a "requeue-USER"
# session, the worker's read would succeed (cross-NS) but the writeback
# would target the requeue session's namespace — duplicating the block
# in the wrong scope.  Iterating per-namespace makes every read AND
# write hit the right rows.

_REQUEUE_STATE: Dict[str, dict] = {}
_REQUEUE_LOCK = threading.Lock()


def _list_namespaces_for_user(table, user_id: str) -> List[str]:
    """Scan distinct namespaces for one user.  Pyarrow has no DISTINCT,
    so we pull the namespace column for the user and de-dup in Python.
    Cheap for our scale (10s of conversations per user)."""
    try:
        rows = (
            table.search()
            .where(f"user_id = '{user_id}'")
            .select(["namespace"])
            .limit(1_000_000)
            .to_arrow()
            .to_pylist()
        )
    except Exception:
        return []
    return sorted({r.get("namespace") for r in rows if r.get("namespace")})


def _list_aids_for(table, user_id: str, namespace: str) -> List[int]:
    try:
        rows = (
            table.search()
            .where(f"user_id = '{user_id}' AND namespace = '{namespace}'")
            .select(["aid"])
            .limit(1_000_000)
            .to_arrow()
            .to_pylist()
        )
    except Exception:
        return []
    out = []
    for r in rows:
        aid = r.get("aid")
        if aid is None:
            continue
        try:
            out.append(int(aid))
        except (TypeError, ValueError):
            continue
    return out


def _run_requeue_semantic(user_id: str) -> None:
    """Background runner.

    Walks user's namespaces sequentially.  After enqueueing each
    namespace's aids we BLOCK until that namespace's semantic_queue
    drains to zero (or the per-namespace timeout fires).  This caps
    the active-instance count at ~1 + idle pool entries, so an
    unbounded session pool churn doesn't OOM the VM (observed once
    on 367 namespaces: peak memory grew unbounded as more instances
    spun up their bulk_warm + Faiss indices in parallel until the
    box hung).

    Per-namespace drain budget:  ``DRAIN_PER_BLOCK_SEC * num_blocks``
    seconds (default 8s/block).  Gemini Flash 2.5 averages 400-600ms
    per block but spikes to 4-5s on long blocks with hint-heavy
    GLiNER passes, so 8s headroom is comfortable.  If the budget
    elapses we move on; the orphan blocks are still enqueued and
    will get re-tried on next /admin/requeue_semantic call (the
    semantic worker idempotently overwrites existing claims)."""
    DRAIN_PER_BLOCK_SEC = 8.0
    DRAIN_POLL_SEC = 1.0
    DRAIN_MIN_BUDGET_SEC = 30.0

    state = _REQUEUE_STATE[user_id]
    try:
        # Open a temporary session just to access the shared Lance archive
        # table; we'll spawn proper per-namespace instances below.
        bootstrap = pool.get(f"requeue-bootstrap-{user_id}", user_id=user_id)
        table = bootstrap.tcmm.archive._archive_table
        namespaces = _list_namespaces_for_user(table, user_id)
        state["namespaces_total"] = len(namespaces)
        logger.info(
            f"[REQUEUE-SEMANTIC] user={user_id[:12]} found "
            f"{len(namespaces)} namespaces"
        )
        for ns in namespaces:
            state["current_namespace"] = ns
            inst = pool.get(ns, user_id=user_id)
            aids = _list_aids_for(table, user_id, ns)
            state["per_namespace"][ns] = {
                "total": len(aids),
                "enqueued": 0,
                "drained": False,
                "drain_timeout": False,
            }
            for aid in aids:
                try:
                    inst.tcmm.semantic_queue.put(aid)
                    state["per_namespace"][ns]["enqueued"] += 1
                    state["total_enqueued"] += 1
                except Exception as _e:
                    logger.warning(f"[REQUEUE-SEMANTIC] put failed aid={aid}: {_e}")

            # Wait for this namespace's worker to drain its queue before
            # we move on.  Keeps active-instance count bounded.
            n = max(1, state["per_namespace"][ns]["enqueued"])
            budget_sec = max(DRAIN_MIN_BUDGET_SEC, DRAIN_PER_BLOCK_SEC * n)
            deadline = time.time() + budget_sec
            q = inst.tcmm.semantic_queue
            while time.time() < deadline:
                if q.qsize() == 0:
                    state["per_namespace"][ns]["drained"] = True
                    break
                time.sleep(DRAIN_POLL_SEC)
            else:
                state["per_namespace"][ns]["drain_timeout"] = True
                logger.warning(
                    f"[REQUEUE-SEMANTIC] ns={ns} drain timeout after "
                    f"{budget_sec:.0f}s — {q.qsize()} blocks still queued; "
                    f"moving on"
                )

            state["namespaces_done"] += 1
            logger.info(
                f"[REQUEUE-SEMANTIC] ns={ns} enqueued "
                f"{state['per_namespace'][ns]['enqueued']}/{len(aids)} "
                f"drained={state['per_namespace'][ns]['drained']} "
                f"(queue depth at exit: {q.qsize()})"
            )
    except Exception as _e:
        state["error"] = repr(_e)
        logger.error(f"[REQUEUE-SEMANTIC] failed: {_e}")
    finally:
        state["done"] = True
        state["elapsed_sec"] = time.time() - state["started"]


@app.post("/admin/requeue_semantic")
async def admin_requeue_semantic(req: Request):
    """Enqueue every archive block for ``user_id`` into the semantic
    worker so claims / entities / topics get re-extracted with the
    current NLP backend.

    Body: ``{"user_id": "<uid>"}``.  Returns immediately; poll
    ``GET /admin/requeue_semantic_status?user_id=<uid>`` for progress.
    The worker drains the queue over the following minutes (Gemini
    Flash 2.5 via AI Studio ≈ 500ms per block; budget 20-30 min for
    a few thousand blocks)."""
    try:
        body = await req.json()
    except Exception:
        body = {}
    user_id = (body.get("user_id") or "").strip()
    if not user_id:
        return {"error": "user_id required"}

    with _REQUEUE_LOCK:
        existing = _REQUEUE_STATE.get(user_id)
        if existing and not existing.get("done"):
            return {"status": "in_progress", **existing}
        _REQUEUE_STATE[user_id] = {
            "user_id": user_id,
            "started": time.time(),
            "namespaces_total": 0,
            "namespaces_done": 0,
            "current_namespace": None,
            "total_enqueued": 0,
            "per_namespace": {},
            "done": False,
            "error": None,
        }

    # Run in a worker thread (background) so HTTP returns immediately.
    threading.Thread(
        target=_run_requeue_semantic,
        args=(user_id,),
        daemon=True,
        name=f"requeue-semantic-{user_id[:8]}",
    ).start()

    return {
        "status": "started",
        "user_id": user_id,
        "monitor": f"/admin/requeue_semantic_status?user_id={user_id}",
    }


@app.get("/admin/requeue_semantic_status")
def admin_requeue_semantic_status(user_id: str = ""):
    """Status of an in-progress (or completed) requeue.
    Omit user_id to get all known requeues."""
    with _REQUEUE_LOCK:
        if user_id:
            state = _REQUEUE_STATE.get(user_id)
            if not state:
                return {"status": "none", "user_id": user_id}
            # Augment with live queue depths so caller sees worker progress
            queues = {}
            for ns in state.get("per_namespace", {}):
                inst = pool._instances.get(pool._normalize_id(ns)) if hasattr(pool, "_instances") else None
                if inst is not None and hasattr(inst.tcmm, "semantic_queue"):
                    queues[ns] = inst.tcmm.semantic_queue.qsize()
            return {**state, "live_queue_depth": queues}
        return _REQUEUE_STATE


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
    # Pin lane — separate in-memory store, never persisted to archive
    # or live_blocks.  See tcmm_core.pin_immutable_block for rationale.
    pinned_fn = getattr(tcmm, "pinned_prefix_blocks", None)
    pinned = list(pinned_fn()) if callable(pinned_fn) else []
    live_tokens = sum(_tok_count(b) for b in live)
    shadow_tokens = sum(_tok_count(b) for b in shadow)
    pinned_tokens = sum(_tok_count(b) for b in pinned)

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
        "pinned": {
            "count": len(pinned),
            "tokens": pinned_tokens,
            "note": "RAM-only; not archived, not in live_blocks",
            "blocks": [_b_brief(b) for b in pinned],
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
