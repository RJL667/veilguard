"""
LanceDB-backed unified provider: Storage + Vector + Sparse.

Single embedded database replaces TinyDB, FAISS, and BM25 with one backend.
Nodes, embeddings, dense ANN search, and full-text search all in one place.

LanceDB uses the Lance columnar format which gives:
  - Versioned storage (crash-safe, no corruption)
  - Columnar compression (~4x smaller than TinyDB JSON)
  - Built-in ANN vector search (IVF-PQ under the hood)
  - Full-text search via Tantivy (BM25-class relevance)
  - Incremental index updates (no full rebuilds)
  - Same API from OSS embedded to LanceDB Cloud/Enterprise

Tables created per instance (names parameterizable via archive_table / sparse_table):
  archive / dream_archive  -- nodes + main embeddings (StorageProvider + VectorProvider)
  embeddings               -- secondary embeddings: topic, etc. (shared; aid keyed)
  sparse / sparse_dream    -- full-text search corpus (SparseProvider)

Filtering strategy (matches SQLite/Chroma pattern):
  - namespace: string filter on every query
  - user_id: optional additional filter (AND with namespace)

Dependencies: pip install lancedb tantivy pylance

Usage:
    from core.providers import create_storage, create_vector, create_sparse

    # One instance does everything
    provider = create_storage("lance", db_path="data/lance.db", dim=3072,
                              namespace="my_app", user_id="user_123")
    # provider implements StorageProvider, VectorProvider, AND SparseProvider
"""

import os
import json
import logging
import numpy as np
import pyarrow as pa
from typing import Dict, List, Tuple, Optional, Any

from .base import StorageProvider, VectorProvider, SparseProvider

logger = logging.getLogger(__name__)


# Sentinel for the embedding cache — distinguishes "known to be missing"
# from "not yet fetched". Without this, a None cache entry would force a
# re-query on every read for nodes that legitimately have no vector.
_MISSING_EMB = object()


# ═════════════════════════════════════════════════════════════════════
# Flat archive schema — every node field is its own column.
# Before: {aid, namespace, user_id, data (JSON blob), vector}
# Now:    {aid, namespace, user_id, <26 scalar cols>, <4 list cols>,
#         <3 struct cols>, <6 map-like cols>, <4 residual-json cols>,
#         extras_json, vector}
# Links (semantic / contextual / topic / entity) live on the node as
# list<struct<key, ...>> columns, so recall can project them in one
# query and deleting the row deletes the links with it (option 3).
# ═════════════════════════════════════════════════════════════════════

# Every top-level node key the provider promotes. Everything outside
# this set gets stuffed into extras_json so nothing is silently dropped.
_PROMOTED_KEYS = frozenset([
    # scalars
    "id", "text", "origin", "semantic_text", "semantic_text_tail",
    "fingerprint", "created_step", "created_ts", "archived_step",
    "token_count", "heat", "last_decay_step", "source", "priority_class",
    "original_id", "recallable", "block_class", "tier", "break_frequency",
    "semantic_break_count", "semantic_flow_count", "lineage_root",
    "timestamp", "session_id", "session_date", "density_score",
    # lists
    "source_block_ids", "topics", "entities", "claims",
    # structs
    "temporal", "archive_stats", "lineage",
    # map-like
    "semantic_links", "entity_links", "topic_links", "contextual_links",
    "topic_dicts", "entity_dicts",
    # residual json
    "behavioural", "suppresses", "semantic", "entropy_static",
    # vectors live in dedicated columns / tables
    "embedding", "topic_embedding",
])


def _archive_schema(dim: int) -> pa.Schema:
    """Build the flat archive-table schema. `dim` = main vector dim."""
    return pa.schema([
        # partition + id
        pa.field("aid", pa.int64()),
        pa.field("namespace", pa.utf8()),
        pa.field("user_id", pa.utf8()),

        # scalars
        pa.field("id", pa.int64()),
        pa.field("text", pa.utf8()),
        pa.field("origin", pa.utf8()),
        pa.field("semantic_text", pa.utf8()),
        pa.field("semantic_text_tail", pa.utf8()),
        # fingerprint is an unsigned 64-bit hash; ~half of the values in
        # real data don't fit in signed int64. Use uint64.
        pa.field("fingerprint", pa.uint64()),
        pa.field("created_step", pa.int64()),
        pa.field("created_ts", pa.int64()),
        pa.field("archived_step", pa.int64()),
        pa.field("token_count", pa.int64()),
        pa.field("heat", pa.float64()),
        pa.field("last_decay_step", pa.int64()),
        pa.field("source", pa.utf8()),
        pa.field("priority_class", pa.utf8()),
        pa.field("original_id", pa.int64()),
        pa.field("recallable", pa.bool_()),
        pa.field("block_class", pa.utf8()),
        pa.field("tier", pa.utf8()),
        pa.field("break_frequency", pa.float64()),
        pa.field("semantic_break_count", pa.int64()),
        pa.field("semantic_flow_count", pa.int64()),
        pa.field("lineage_root", pa.int64()),
        pa.field("timestamp", pa.float64()),
        pa.field("session_id", pa.utf8()),
        pa.field("session_date", pa.utf8()),
        pa.field("density_score", pa.float64()),

        # lists
        pa.field("source_block_ids", pa.list_(pa.int64())),
        pa.field("topics", pa.list_(pa.utf8())),
        pa.field("entities", pa.list_(pa.utf8())),
        pa.field("claims", pa.list_(pa.utf8())),

        # structs
        pa.field("temporal", pa.struct([
            pa.field("prev_aid", pa.int64()),
            pa.field("next_aid", pa.int64()),
            pa.field("prev_weight", pa.float64()),
            pa.field("next_weight", pa.float64()),
        ])),
        pa.field("archive_stats", pa.struct([
            pa.field("attempts", pa.int64()),
            pa.field("used", pa.int64()),
        ])),
        pa.field("lineage", pa.struct([
            pa.field("root", pa.int64()),
            pa.field("parents", pa.list_(pa.int64())),
        ])),

        # map-like (list<struct<key, val>>) — same shape as a JSON object
        # when serialized, but queryable per-row and easy to unnest.
        pa.field("semantic_links", pa.list_(pa.struct([
            pa.field("key", pa.utf8()),
            pa.field("val", pa.float64()),
        ]))),
        pa.field("entity_links", pa.list_(pa.struct([
            pa.field("key", pa.utf8()),
            pa.field("val", pa.float64()),
        ]))),
        pa.field("topic_links", pa.list_(pa.struct([
            pa.field("key", pa.utf8()),
            pa.field("val", pa.float64()),
        ]))),
        pa.field("contextual_links", pa.list_(pa.struct([
            pa.field("key", pa.utf8()),
            pa.field("weight", pa.float64()),
            pa.field("last_step", pa.int64()),
        ]))),
        pa.field("topic_dicts", pa.list_(pa.struct([
            pa.field("name", pa.utf8()),
            pa.field("type", pa.utf8()),
            pa.field("score", pa.float64()),
        ]))),
        pa.field("entity_dicts", pa.list_(pa.struct([
            pa.field("name", pa.utf8()),
            pa.field("type", pa.utf8()),
            pa.field("score", pa.float64()),
        ]))),

        # residual json — rarely populated, shape not worth pinning
        pa.field("behavioural_json", pa.utf8()),
        pa.field("suppresses_json", pa.utf8()),
        pa.field("semantic_json", pa.utf8()),
        pa.field("entropy_static_json", pa.utf8()),

        # forward-compat: anything not in the schema above lands here
        pa.field("extras_json", pa.utf8()),

        # main embedding
        pa.field("vector", pa.list_(pa.float32(), list_size=dim)),
    ])


# Column name groups for (de)serialization loops.
# fingerprint is kept outside _SCALAR_INT_COLS because it's uint64
# (unsigned 64-bit hash — signed int64 overflows for ~half the values).
_SCALAR_INT_COLS = (
    "id", "created_step", "created_ts", "archived_step",
    "token_count", "last_decay_step", "original_id",
    "semantic_break_count", "semantic_flow_count", "lineage_root",
)
_SCALAR_UINT_COLS = ("fingerprint",)
_SCALAR_STR_COLS = (
    "text", "origin", "semantic_text", "semantic_text_tail", "source",
    "priority_class", "block_class", "tier", "session_id", "session_date",
)
_SCALAR_FLOAT_COLS = ("heat", "break_frequency", "timestamp", "density_score")
_SCALAR_BOOL_COLS = ("recallable",)
_LIST_INT_COLS = ("source_block_ids",)
_LIST_STR_COLS = ("topics", "entities", "claims")


def _map_to_rows(d):
    """dict{key -> val} -> list[{key, val}] for map-like columns.
    Preserves null vs empty: None -> null list, {} -> empty list."""
    if d is None:
        return None
    if not d:
        return []
    out = []
    for k, v in d.items():
        try:
            out.append({"key": str(k), "val": float(v)})
        except (TypeError, ValueError):
            continue
    return out


def _ctx_to_rows(d):
    """dict{key -> {weight, last_step}} -> list[{key, weight, last_step}].
    Preserves null vs empty."""
    if d is None:
        return None
    if not d:
        return []
    out = []
    for k, v in d.items():
        if not isinstance(v, dict):
            continue
        try:
            out.append({
                "key": str(k),
                "weight": float(v.get("weight", 0.0)),
                "last_step": int(v.get("last_step", 0)),
            })
        except (TypeError, ValueError):
            continue
    return out


def _dicts_to_rows(xs):
    """list[{name, type, score}] -> same, normalized. Preserves null vs []."""
    if xs is None:
        return None
    if not xs:
        return []
    out = []
    for x in xs:
        if not isinstance(x, dict):
            continue
        out.append({
            "name": str(x.get("name", "")),
            "type": str(x.get("type", "")),
            "score": float(x.get("score", 0.0)),
        })
    return out


def _is_null(v) -> bool:
    """True if pyarrow/pandas gave us a null (None or NaN float)."""
    if v is None:
        return True
    if isinstance(v, float):
        try:
            return np.isnan(v)
        except Exception:
            return False
    return False


def _is_empty_list(rows) -> bool:
    """True if `rows` is an empty list/array (but NOT null). Caller must
    check _is_null first to distinguish null from empty."""
    try:
        return len(rows) == 0
    except TypeError:
        return False


def _rows_to_map(rows):
    """Inverse of _map_to_rows. Preserves null vs empty: null -> None, [] -> {}."""
    if _is_null(rows):
        return None
    if _is_empty_list(rows):
        return {}
    out = {}
    for r in rows:
        if r is None:
            continue
        out[str(r["key"])] = float(r["val"])
    return out


def _rows_to_ctx(rows):
    """Inverse of _ctx_to_rows. Preserves null vs empty."""
    if _is_null(rows):
        return None
    if _is_empty_list(rows):
        return {}
    out = {}
    for r in rows:
        if r is None:
            continue
        out[str(r["key"])] = {
            "weight": float(r["weight"]),
            "last_step": int(r["last_step"]),
        }
    return out


def _rows_to_dicts(rows):
    """Inverse of _dicts_to_rows. Preserves null vs empty."""
    if _is_null(rows):
        return None
    if _is_empty_list(rows):
        return []
    return [
        {"name": str(r["name"]), "type": str(r["type"]),
         "score": float(r["score"])}
        for r in rows if r is not None
    ]


def _dump_json(obj):
    """Serialize a residual field. Pre-normalize dataclasses."""
    import dataclasses as _dc
    if _dc.is_dataclass(obj):
        obj = _dc.asdict(obj)
    try:
        return json.dumps(obj) if obj is not None else ""
    except TypeError:
        return json.dumps(str(obj))


def _load_json(s):
    """Inverse of _dump_json. Empty string -> None."""
    if not s:
        return None
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return None


class _NullLock:
    """No-op context manager when filelock is not available."""
    def __enter__(self): return self
    def __exit__(self, *args): pass


class LanceStorageProvider(StorageProvider, VectorProvider, SparseProvider):
    # ── Class-level auto-compaction state ─────────────────────────────
    # Shared across ALL instances because the underlying Lance tables
    # are shared. Each TCMM session creates its own provider instance
    # but they all write to the same ``archive`` table. If we tracked
    # compaction per-instance, N sessions would each need to do their
    # own 200-write warm-up before any compaction fires — bad scaling.
    # Single class-level counter + lock = one compaction cadence for
    # the whole process.
    _global_writes_since_compact = 0
    _global_compact_lock = None  # created lazily in __init__
    _global_last_compact_version = 0

    """LanceDB-backed unified archive storage + vector index + sparse search.

    Implements all three provider ABCs:
      - StorageProvider: dict-like node CRUD with colocated embeddings
      - VectorProvider: dense ANN search on the archive table's vector column
      - SparseProvider: full-text search via LanceDB FTS (Tantivy)

    Tables:
      - 'archive': nodes + main embedding vectors
      - 'embeddings': secondary embeddings (topic, etc.)
      - 'sparse': full-text search corpus (text + aid mapping)

    Multi-tenant isolation via namespace + user_id filters on every operation.
    """

    def __init__(self, dim: int = 3072, db_path: str = "data/lance.db",
                 namespace: str = "default", user_id: str = None,
                 archive_table: str = "archive",
                 sparse_table: str = "sparse", **kwargs):
        """
        Args:
            dim: Embedding dimensionality (default 3072 for text-embedding-3-large).
            db_path: Path to LanceDB database directory.
            namespace: Tenant namespace for multi-tenant isolation.
            user_id: Optional user-level isolation within namespace.
            archive_table: Name of the main node+vector table. Default "archive";
                           use "dream_archive" (or other) for separate dream storage.
                           Schema is identical; AID range disambiguates content.
            sparse_table: Name of the FTS table (allows multiple sparse indexes
                         in the same DB by using different table names, e.g.
                         "sparse_entity", "sparse_topic", "sparse_claims").
        """
        try:
            import lancedb as _lancedb
        except ImportError:
            raise ImportError(
                "LanceDB not installed. Run: pip install lancedb"
            )

        self._dim = dim
        self._db_path = db_path
        self._namespace = namespace
        self._user_id = user_id
        self._lancedb = _lancedb
        self._archive_table_name = archive_table
        self._sparse_table_name = sparse_table
        self._fts_ready = False  # FTS index created on first search

        # Ensure parent directory exists
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self._db = _lancedb.connect(db_path)
        self._archive_table = None
        self._embeddings_table = None
        self._sparse_table = None
        self._sequences_table = None
        self._seq_cache = {}  # In-memory cache for sequence values
        # Read-through caches — populated on demand and cleared on any write.
        # Without these, every get() / get_embedding() does a full
        # search().where().limit(1) + checkout_latest() round-trip; recall
        # triggers hundreds of lookups per query and CPU spikes hard.
        self._node_cache = {}   # aid -> deserialized node dict
        self._emb_cache = {}    # (aid, emb_type) -> np.ndarray
        # True once items() has done a full scan and the cache holds every
        # row in this (namespace, user_id). Flipped back to False on any
        # put/del that could add new aids. Lets items()/values() serve
        # subsequent calls from the cache instead of re-scanning Lance.
        # _compute_entry_priors iterates items() on every recall — without
        # this flag that was a 148ms wall-clock hit per query.
        self._cache_complete = False
        import threading
        self._seq_lock = threading.Lock()
        self._write_lock = threading.Lock()  # Serialize writes to LanceDB (not thread-safe)

        # ── Auto-compaction ──────────────────────────────────────────
        # Lance is MVCC: every ``delete + add`` creates a new fragment
        # AND a new table version. Each block-write is ``delete + add``
        # so every ingest spawns 2 versions. At 8-12 blocks per chat
        # turn that's ~20 versions per turn. Observed 23 Apr 2026: a
        # 1,980-row archive sitting at **version 31,300** with writes
        # taking 5-14s each. Cure: periodic ``optimize()`` +
        # ``cleanup_old_versions()`` running in a daemon thread every
        # N writes across ALL provider instances (class-level counter).
        # Threshold via ``VEILGUARD_LANCE_COMPACT_EVERY`` (default 50
        # writes — ~6 chat turns).
        self._compact_every_writes = int(
            os.environ.get("VEILGUARD_LANCE_COMPACT_EVERY", "50")
        )
        # Lazily initialise the class-level lock (can't create it at
        # class-definition time because ``threading`` may not be
        # imported yet in that path).
        if LanceStorageProvider._global_compact_lock is None:
            LanceStorageProvider._global_compact_lock = threading.Lock()
        # File lock for cross-process sequence coordination
        try:
            import filelock
            lock_dir = db_path if os.path.isdir(db_path) else os.path.dirname(db_path) or "."
            lock_path = os.path.join(lock_dir, ".lance_seq.lock")
            self._seq_file_lock = filelock.FileLock(lock_path, timeout=5)
        except ImportError:
            self._seq_file_lock = None
        self._open_tables()

    def _open_or_create(self, name, schema):
        """Open table if it exists, create if not. Handles race conditions
        where another instance created the table between our check and create.
        """
        try:
            return self._db.open_table(name)
        except Exception:
            pass
        try:
            return self._db.create_table(name, schema=schema)
        except (ValueError, Exception):
            # Table was created by another instance between our check
            return self._db.open_table(name)

    def _open_tables(self):
        """Open or create the archive, embeddings, and sparse tables."""
        existing = set(self._db.table_names())

        # ── Archive table: flat node columns + main embedding vector ──
        # Table name parameterized: "archive" for regular nodes,
        # "dream_archive" for dream nodes (same schema, separated storage).
        self._archive_table = self._open_or_create(
            self._archive_table_name, _archive_schema(self._dim)
        )

        # ── Secondary embeddings table (topic, etc.) ──
        self._embeddings_table = self._open_or_create("embeddings", pa.schema([
            pa.field("aid", pa.int64()),
            pa.field("namespace", pa.utf8()),
            pa.field("user_id", pa.utf8()),
            pa.field("emb_type", pa.utf8()),
            pa.field("vector", pa.list_(pa.float32(), list_size=self._dim)),
        ]))

        # ── Sequences table (atomic ID generation) ──
        self._sequences_table = self._open_or_create("_sequences", pa.schema([
            pa.field("name", pa.utf8()),
            pa.field("value", pa.int64()),
        ]))
        # Load existing sequences into cache
        try:
            df = self._sequences_table.to_pandas()
            for _, row in df.iterrows():
                self._seq_cache[row["name"]] = int(row["value"])
        except Exception:
            pass

        # ── Sparse / FTS table ──
        self._sparse_table = self._open_or_create(self._sparse_table_name, pa.schema([
            pa.field("aid", pa.int64()),
            pa.field("namespace", pa.utf8()),
            pa.field("user_id", pa.utf8()),
            pa.field("text", pa.utf8()),
        ]))

    # ══════════════════════════════════════════════════════════════
    # Filter helpers
    # ══════════════════════════════════════════════════════════════

    def _ns_filter(self) -> str:
        """Filter by user_id AND namespace (session-scoped).
        Always includes user_id when set. Never namespace-only without user_id.
        """
        if self._user_id:
            return f"user_id = '{self._user_id}' AND namespace = '{self._namespace}'"
        return f"namespace = '{self._namespace}'"

    def _user_filter(self) -> str:
        """Filter by user_id only (cross-namespace, same user).
        Used for linking and traversal across sessions.
        """
        if self._user_id:
            return f"user_id = '{self._user_id}'"
        return self._ns_filter()  # fallback if no user_id set

    def _uid(self) -> str:
        """Return user_id value for inserts."""
        return self._user_id or ""

    # ══════════════════════════════════════════════════════════════
    # JSON serialization
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _clean_for_json(obj):
        """Recursively prepare an object for JSON serialization.
        - Convert dict keys to strings (orjson requires str keys)
        - Convert numpy arrays to lists
        - Handle nested structures (semantic_links={2: 0.5}, embeddings, etc.)
        """
        if isinstance(obj, dict):
            return {str(k): LanceStorageProvider._clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [LanceStorageProvider._clean_for_json(i) for i in obj]
        if hasattr(obj, "tolist"):  # numpy array
            return obj.tolist()
        return obj

    @staticmethod
    def _serialize(node: Dict) -> str:
        try:
            import orjson
            clean = LanceStorageProvider._clean_for_json(node)
            return orjson.dumps(clean).decode("utf-8")
        except (ImportError, TypeError):
            return json.dumps(node, default=str)

    @staticmethod
    def _deserialize(data: str) -> Dict:
        try:
            import orjson
            return orjson.loads(data)
        except ImportError:
            return json.loads(data)

    # ═════════════════════════════════════════════════════════════
    # Flat-schema codec: node dict <-> Lance row dict
    # ═════════════════════════════════════════════════════════════

    def _node_to_row(self, aid: int, node: Dict, vec: Optional[list]) -> Dict:
        """Serialize a node dict to a row matching _archive_schema."""
        aid_int = int(aid)
        row = {
            "aid": aid_int,
            "namespace": self._namespace,
            "user_id": self._uid(),
            "vector": vec if vec is not None else self._zero_vector(),
        }

        # Scalars — None is allowed and stays null
        for k in _SCALAR_INT_COLS:
            v = node.get(k)
            row[k] = int(v) if isinstance(v, (int, float)) and v is not None else (v if v is None else None)
        for k in _SCALAR_UINT_COLS:
            v = node.get(k)
            # Clamp to uint64 range; fingerprints > 2^64-1 get masked.
            if v is None:
                row[k] = None
            else:
                try:
                    row[k] = int(v) & 0xFFFFFFFFFFFFFFFF
                except (TypeError, ValueError):
                    row[k] = None
        for k in _SCALAR_STR_COLS:
            v = node.get(k)
            row[k] = str(v) if v is not None else None
        for k in _SCALAR_FLOAT_COLS:
            v = node.get(k)
            row[k] = float(v) if isinstance(v, (int, float)) and v is not None else None
        for k in _SCALAR_BOOL_COLS:
            v = node.get(k)
            row[k] = bool(v) if v is not None else None

        # Lists
        for k in _LIST_INT_COLS:
            v = node.get(k) or []
            row[k] = [int(x) for x in v if isinstance(x, (int, float))]
        for k in _LIST_STR_COLS:
            v = node.get(k) or []
            row[k] = [str(x) for x in v if isinstance(x, str)]

        # Structs (nullable leaves)
        tmp = node.get("temporal") or {}
        row["temporal"] = {
            "prev_aid": tmp.get("prev_aid"),
            "next_aid": tmp.get("next_aid"),
            "prev_weight": tmp.get("prev_weight"),
            "next_weight": tmp.get("next_weight"),
        }
        stats = node.get("archive_stats") or {}
        row["archive_stats"] = {
            "attempts": int(stats.get("attempts", 0)),
            "used": int(stats.get("used", 0)),
        }
        lin = node.get("lineage") or {}
        row["lineage"] = {
            "root": lin.get("root"),
            "parents": [int(x) for x in (lin.get("parents") or []) if isinstance(x, (int, float))],
        }

        # Map-like
        row["semantic_links"] = _map_to_rows(node.get("semantic_links"))
        row["entity_links"] = _map_to_rows(node.get("entity_links"))
        row["topic_links"] = _map_to_rows(node.get("topic_links"))
        row["contextual_links"] = _ctx_to_rows(node.get("contextual_links"))
        row["topic_dicts"] = _dicts_to_rows(node.get("topic_dicts"))
        row["entity_dicts"] = _dicts_to_rows(node.get("entity_dicts"))

        # Residual json — shape too variable / mostly empty
        row["behavioural_json"] = _dump_json(node.get("behavioural"))
        row["suppresses_json"] = _dump_json(node.get("suppresses"))
        row["semantic_json"] = _dump_json(node.get("semantic"))
        row["entropy_static_json"] = _dump_json(node.get("entropy_static"))

        # Forward-compat: anything outside the promoted set
        extras = {k: v for k, v in node.items() if k not in _PROMOTED_KEYS}
        row["extras_json"] = _dump_json(extras) if extras else ""

        return row

    def _row_to_node(self, row) -> Dict:
        """Reverse of _node_to_row. `row` is a pandas Series from to_pandas()."""
        node: Dict[str, Any] = {}

        # Scalars — pandas yields numpy scalars; coerce to Python types
        for k in _SCALAR_INT_COLS + _SCALAR_UINT_COLS:
            v = row.get(k) if hasattr(row, "get") else row[k]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                try:
                    node[k] = int(v)
                except (TypeError, ValueError):
                    node[k] = None
            else:
                node[k] = None
        for k in _SCALAR_STR_COLS:
            v = row.get(k) if hasattr(row, "get") else row[k]
            node[k] = str(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None
        for k in _SCALAR_FLOAT_COLS:
            v = row.get(k) if hasattr(row, "get") else row[k]
            node[k] = float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None
        for k in _SCALAR_BOOL_COLS:
            v = row.get(k) if hasattr(row, "get") else row[k]
            node[k] = bool(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None

        # Lists
        for k in _LIST_INT_COLS:
            v = row.get(k) if hasattr(row, "get") else row[k]
            node[k] = [int(x) for x in (v if v is not None else [])]
        for k in _LIST_STR_COLS:
            v = row.get(k) if hasattr(row, "get") else row[k]
            node[k] = [str(x) for x in (v if v is not None else [])]

        # Structs — pyarrow gives back dicts (or None)
        tmp = row.get("temporal") if hasattr(row, "get") else row["temporal"]
        if tmp is None:
            node["temporal"] = {"prev_aid": None, "next_aid": None,
                                "prev_weight": None, "next_weight": None}
        else:
            node["temporal"] = {
                "prev_aid": (None if tmp["prev_aid"] is None else int(tmp["prev_aid"])),
                "next_aid": (None if tmp["next_aid"] is None else int(tmp["next_aid"])),
                "prev_weight": (None if tmp["prev_weight"] is None else float(tmp["prev_weight"])),
                "next_weight": (None if tmp["next_weight"] is None else float(tmp["next_weight"])),
            }
        stats = row.get("archive_stats") if hasattr(row, "get") else row["archive_stats"]
        node["archive_stats"] = {
            "attempts": int(stats["attempts"]) if stats else 0,
            "used": int(stats["used"]) if stats else 0,
        }
        lin = row.get("lineage") if hasattr(row, "get") else row["lineage"]
        node["lineage"] = {
            "root": (None if not lin or lin["root"] is None else int(lin["root"])),
            "parents": [int(x) for x in ((lin["parents"] if lin else None) or [])],
        }

        # Map-like
        node["semantic_links"] = _rows_to_map(row.get("semantic_links") if hasattr(row, "get") else row["semantic_links"])
        node["entity_links"] = _rows_to_map(row.get("entity_links") if hasattr(row, "get") else row["entity_links"])
        node["topic_links"] = _rows_to_map(row.get("topic_links") if hasattr(row, "get") else row["topic_links"])
        node["contextual_links"] = _rows_to_ctx(row.get("contextual_links") if hasattr(row, "get") else row["contextual_links"])
        node["topic_dicts"] = _rows_to_dicts(row.get("topic_dicts") if hasattr(row, "get") else row["topic_dicts"])
        node["entity_dicts"] = _rows_to_dicts(row.get("entity_dicts") if hasattr(row, "get") else row["entity_dicts"])

        # Residual json
        node["behavioural"] = _load_json(row.get("behavioural_json") if hasattr(row, "get") else row["behavioural_json"])
        node["suppresses"] = _load_json(row.get("suppresses_json") if hasattr(row, "get") else row["suppresses_json"])
        node["semantic"] = _load_json(row.get("semantic_json") if hasattr(row, "get") else row["semantic_json"])
        node["entropy_static"] = _load_json(row.get("entropy_static_json") if hasattr(row, "get") else row["entropy_static_json"])

        # Normalize empty residuals to {} instead of None to match old code
        for k in ("behavioural", "suppresses", "semantic"):
            if node[k] is None:
                node[k] = {}

        # Forward-compat merge
        extras = _load_json(row.get("extras_json") if hasattr(row, "get") else row["extras_json"])
        if isinstance(extras, dict):
            node.update(extras)

        return node

    # ══════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════

    # Short-lived all-rows snapshot used by _get_row to collapse N single-aid
    # lookups into ONE full-table scan per recall pass. Populated by _get_row
    # on first miss, served from dict on subsequent calls in the same recall.
    # Cleared on __setitem__ so post-write reads are fresh.
    _row_cache = None  # type: dict[int, dict] | None

    def _get_row(self, aid: int):
        """Fetch a single archive row by aid + namespace filter.

        Returns a plain Python dict. We full-scan via to_arrow() and
        filter in Python instead of using search().where() because Lance's
        push-down filter on list<struct> columns materializes nulls as
        empty lists — losing the null-vs-empty distinction for fields
        like topic_dicts / semantic_links.

        Caching: a single recall pass calls this once per hit (typically
        15-30+ cross-NS aids). Without caching, each call did its own
        full to_arrow() scan — dominant cost was 50-80 seconds of
        wall-clock CPU per recall on a 35-row archive. With caching,
        the first miss fills _row_cache with EVERY row in the table
        keyed by (aid, user_id, namespace) and all subsequent reads
        in the same recall hit the in-process dict. Writes (__setitem__)
        clear the cache so staleness is bounded to one recall.
        """
        import time as _t
        _gr_t0 = _t.perf_counter()
        aid_int = int(aid)
        uid = self._uid()
        ns = self._namespace

        # Fast path: previously-scanned snapshot still valid.
        if self._row_cache is not None:
            res = self._row_cache.get((aid_int, uid, ns))
            # Only log slow cache hits (should never happen, but catches
            # pathological dict lookups if something's weird).
            _gr_dt = (_t.perf_counter() - _gr_t0) * 1000.0
            if _gr_dt > 50.0:
                print(f"    [PERF/GET_ROW] cache-hit aid={aid_int} +{_gr_dt:.1f}ms (suspiciously slow)", flush=True)
            return res

        # Slow path: fresh scan, populate cache with EVERY row (not
        # just our target). A recall loops through 15-30 hits — the
        # scan cost amortizes across all of them.
        try:
            self._archive_table.checkout_latest()
        except Exception:
            pass
        _co_dt = (_t.perf_counter() - _gr_t0) * 1000.0
        try:
            arr = self._archive_table.to_arrow()
            rows = arr.to_pylist()
        except Exception:
            return None
        _scan_dt = (_t.perf_counter() - _gr_t0) * 1000.0 - _co_dt

        cache = {}
        for row in rows:
            try:
                _aid = int(row.get("aid"))
            except Exception:
                continue
            _uid = row.get("user_id") or ""
            _ns = row.get("namespace") or ""
            cache[(_aid, _uid, _ns)] = row
        self._row_cache = cache
        _gr_dt = (_t.perf_counter() - _gr_t0) * 1000.0
        print(f"    [PERF/GET_ROW] FRESH_SCAN rows={len(rows)} checkout_latest={_co_dt:.1f}ms scan={_scan_dt:.1f}ms build_cache={(_gr_dt - _co_dt - _scan_dt):.1f}ms total={_gr_dt:.1f}ms", flush=True)
        return cache.get((aid_int, uid, ns))

    def _zero_vector(self) -> list:
        """Return a zero vector for nodes without embeddings."""
        return [0.0] * self._dim

    def _ensure_fts_index(self):
        """Ensure the FTS index exists on the sparse table.
        Called once on first search. LanceDB's built-in FTS auto-indexes
        new records after the index is created — no rebuilds needed.
        """
        if self._fts_ready:
            return
        try:
            self._sparse_table.create_fts_index("text", replace=True)
            self._fts_ready = True
        except Exception as e:
            logger.debug("FTS index creation failed: %s", e)

    # ══════════════════════════════════════════════════════════════
    # ID Sequence Generation (DB-assigned integer AIDs)
    # ══════════════════════════════════════════════════════════════

    def next_id(self, sequence: str = "archive") -> int:
        """Atomically allocate the next ID from a named sequence.

        Thread-safe via threading lock. Cross-process safe via file lock
        (requires `pip install filelock`, graceful fallback without it).

        Sequences:
          "archive"    -- archive node AIDs (starts at 1)
          "dream"      -- dream node IDs (starts at 10_000_000)
          "entity_vid" -- entity vector IDs (starts at 20_000_000)
        """
        defaults = {"archive": 1, "dream": 10_000_000, "entity_vid": 20_000_000}

        # Acquire file lock (cross-process) then thread lock (in-process)
        lock_ctx = self._seq_file_lock if self._seq_file_lock else _NullLock()
        with lock_ctx:
            with self._seq_lock:
                # Refresh to latest version (see cross-instance writes)
                try:
                    self._sequences_table.checkout_latest()
                except Exception:
                    pass

                db_val = None
                try:
                    df = self._sequences_table.search().where(
                        f"name = '{sequence}'"
                    ).limit(1).to_pandas()
                    if len(df) > 0:
                        db_val = int(df.iloc[0]["value"])
                except Exception:
                    pass

                cache_val = self._seq_cache.get(sequence, 0)
                default_val = defaults.get(sequence, 1)

                # Use the highest of: DB value, cache, or default
                current = max(db_val or 0, cache_val, default_val)

                aid = current
                self._seq_cache[sequence] = aid + 1
                self._persist_sequence(sequence, aid + 1)

                return aid

    def current_id(self, sequence: str = "archive") -> int:
        """Read current sequence value without incrementing."""
        with self._seq_lock:
            if sequence in self._seq_cache:
                return self._seq_cache[sequence]
        # Not in cache — check DB
        try:
            df = self._sequences_table.search().where(
                f"name = '{sequence}'"
            ).limit(1).to_pandas()
            if len(df) > 0:
                return int(df.iloc[0]["value"])
        except Exception:
            pass
        defaults = {"archive": 1, "dream": 10_000_000, "entity_vid": 20_000_000}
        return defaults.get(sequence, 1)

    def set_id(self, sequence: str, value: int) -> None:
        """Set sequence to a specific value (for migration/restore).

        Only increases the counter — never decreases it to avoid
        reusing previously allocated IDs.
        """
        lock_ctx = self._seq_file_lock if self._seq_file_lock else _NullLock()
        with lock_ctx:
            with self._seq_lock:
                # Read DB value to ensure we never decrease
                db_val = 0
                try:
                    df = self._sequences_table.search().where(
                        f"name = '{sequence}'"
                    ).limit(1).to_pandas()
                    if len(df) > 0:
                        db_val = int(df.iloc[0]["value"])
                except Exception:
                    pass
                cache_val = self._seq_cache.get(sequence, 0)
                new_val = max(cache_val, db_val, int(value))
                self._seq_cache[sequence] = new_val
                self._persist_sequence(sequence, new_val)

    def _persist_sequence(self, name: str, value: int) -> None:
        """Write sequence value to the _sequences table.

        Uses delete+add (not merge_insert) for reliable cross-instance
        visibility — LanceDB's append-based storage makes delete+add
        create a new version that other instances see on table reopen.
        """
        try:
            self._sequences_table.delete(f"name = '{name}'")
        except Exception:
            pass
        self._sequences_table.add([{"name": name, "value": int(value)}])

    # ══════════════════════════════════════════════════════════════
    # StorageProvider -- dict-like interface
    # ══════════════════════════════════════════════════════════════

    def get(self, aid: int, default=None) -> Optional[Dict]:
        aid_int = int(aid)
        cached = self._node_cache.get(aid_int)
        if cached is not None:
            return cached
        row = self._get_row(aid_int)
        if row is None:
            return default
        node = self._row_to_node(row)
        self._node_cache[aid_int] = node
        return node

    def __getitem__(self, aid: int) -> Dict:
        result = self.get(aid)
        if result is None:
            raise KeyError(aid)
        return result

    def __setitem__(self, aid: int, node: Dict) -> None:
        aid_int = int(aid)
        emb = node.get("embedding")
        if emb is not None:
            vec = np.array(emb, dtype=np.float32).tolist()
            if len(vec) != self._dim:
                vec = (vec[:self._dim] if len(vec) > self._dim
                       else vec + [0.0] * (self._dim - len(vec)))
        else:
            # Preserve existing vector if no new embedding provided
            # (prevents race where semantic worker clobbers embedding
            # worker's vector with zeros).
            existing = self._get_row(aid_int)
            if existing is not None:
                existing_vec = existing["vector"]
                if existing_vec is not None and any(abs(v) > 1e-6 for v in existing_vec[:5]):
                    vec = list(existing_vec)
                else:
                    vec = self._zero_vector()
            else:
                vec = self._zero_vector()

        row = self._node_to_row(aid_int, node, vec)
        table = pa.Table.from_pylist([row], schema=self._archive_table.schema)

        # Atomic upsert via merge_insert on the compound logical key
        # (user_id, namespace, aid). Previously this was a delete+add pair
        # with a bare ``except Exception: pass`` on the delete — if the
        # delete failed (permission, timeout) OR two writers raced on the
        # same aid, both add() calls succeeded and the table ended up with
        # duplicate rows. As of 23 Apr 2026, an audit of the archive found
        # 569 identical-content duplicates across 572 logical records
        # (~26% redundancy). merge_insert performs the match + insert or
        # match + update as one transaction, so races can't leak dupes.
        #
        # Fallback: if merge_insert raises (API mismatch on older lancedb),
        # we fall back to the old delete+add pattern so TCMM stays up.
        # The fallback keeps the silent-except since it's a best-effort
        # path only taken on version mismatch.
        with self._write_lock:
            try:
                (self._archive_table
                    .merge_insert(["user_id", "namespace", "aid"])
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .execute(table))
            except Exception as _merge_err:
                logger.warning(
                    "archive merge_insert failed (aid=%d), falling back to delete+add: %s",
                    aid_int, _merge_err,
                )
                try:
                    self._archive_table.delete(f"aid = {aid_int} AND {self._ns_filter()}")
                except Exception:
                    pass
                self._archive_table.add(table)
        # Partial invalidation for BOTH caches: patch just this row
        # rather than wiping the dicts. Same rationale as the _row_cache
        # patch below, applied to _node_cache and _cache_complete too.
        #
        # Before this fix: every write set ``_cache_complete = False``,
        # which made the next ``items()`` call do a full-table scan.
        # With background ingestion firing constantly (deferred user
        # ingest + deferred post_response ingest), every recall paid
        # for a full scan via ``_compute_entry_priors`` —
        # ~780ms on a 2K-row archive, ~4s on a 10K-row archive.
        # Observed 23 Apr 2026: prep jumped from 29ms to 780ms after
        # a series of writes invalidated _cache_complete between
        # recalls. With this patch, the writes update _node_cache in
        # place and _cache_complete stays valid — items() serves from
        # the warmed cache instead of re-scanning.
        try:
            patched_node = self._row_to_node(row)
        except Exception:
            patched_node = None

        if patched_node is not None:
            self._node_cache[aid_int] = patched_node
            # vector: write whatever we just produced (non-zero wins over zero)
            if any(abs(v) > 1e-6 for v in vec[:5]):
                self._emb_cache[(aid_int, "archive")] = np.array(vec, dtype=np.float32)
            # _cache_complete stays True — we patched the one row that changed.
        else:
            # Failed to serialize node — fall back to full invalidation.
            self._node_cache.pop(aid_int, None)
            self._emb_cache.pop((aid_int, "archive"), None)
            self._cache_complete = False

        if self._row_cache is not None:
            try:
                _patched_row = self._node_to_row(aid_int, node, vec)
                self._row_cache[(aid_int, self._uid(), self._namespace)] = _patched_row
            except Exception:
                self._row_cache = None

        # Fire auto-compaction check — non-blocking, runs in background.
        LanceStorageProvider._global_writes_since_compact += 1
        self._maybe_compact_async()

    def _maybe_compact_async(self):
        """Run ``optimize()`` + ``cleanup_old_versions()`` periodically.

        Uses CLASS-level state so concurrent sessions share one
        compaction cadence — they all write to the same Lance
        tables, so one session's compact benefits every other
        session's reads/writes.

        Non-blocking: try-lock the class-level compact lock. If
        another compaction is in flight, skip and try again on the
        next write that crosses the threshold. The compaction itself
        runs in a daemon thread so it doesn't block the triggering
        write.
        """
        if LanceStorageProvider._global_writes_since_compact < self._compact_every_writes:
            return
        lock = LanceStorageProvider._global_compact_lock
        if lock is None or not lock.acquire(blocking=False):
            return
        # Reset the global counter BEFORE spawning so concurrent
        # writes don't all queue up compactions.
        LanceStorageProvider._global_writes_since_compact = 0

        import threading as _threading
        import time as _time

        # Capture a reference to the live archive table (other
        # instances share the same table via the same db_path, so
        # this is fine — ``optimize()`` is a server-side operation,
        # all handles see the result).
        archive = self._archive_table

        def _run_compact():
            try:
                t0 = _time.time()
                try:
                    version_before = archive.version
                except Exception:
                    version_before = -1
                try:
                    archive.optimize()
                except Exception as e:
                    logger.warning(f"[lance-compact] optimize() failed: {e}")
                try:
                    import datetime as _dt
                    archive.cleanup_old_versions(older_than=_dt.timedelta(hours=1))
                except Exception as e:
                    logger.warning(f"[lance-compact] cleanup_old_versions() failed: {e}")
                try:
                    version_after = archive.version
                except Exception:
                    version_after = -1
                elapsed = _time.time() - t0
                logger.info(
                    f"[lance-compact] archive optimized in {elapsed:.2f}s "
                    f"(version {version_before} → {version_after})"
                )
                LanceStorageProvider._global_last_compact_version = version_after
            finally:
                lock.release()

        _threading.Thread(
            target=_run_compact, daemon=True, name="TCMM-LanceCompact"
        ).start()

    def __delitem__(self, aid: int) -> None:
        aid_int = int(aid)
        filt = f"aid = {aid_int} AND {self._ns_filter()}"
        if not self.__contains__(aid):
            raise KeyError(aid)
        self._archive_table.delete(filt)
        self._node_cache.pop(aid_int, None)
        self._emb_cache.pop((aid_int, "archive"), None)
        self._cache_complete = False
        self._row_cache = None

    def __contains__(self, aid: int) -> bool:
        return self._get_row(aid) is not None

    def __bool__(self) -> bool:
        """Provider is always truthy (exists and is usable).
        Without this, bool(provider) delegates to __len__ which returns 0
        for empty databases, causing 'if provider:' checks to fail.
        """
        return True

    def __len__(self) -> int:
        try:
            df = self._archive_table.search().where(self._ns_filter()).limit(100_000).select(["aid"]).to_pandas()
            return len(df)
        except Exception:
            return 0

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        try:
            df = self._archive_table.search().where(self._ns_filter()).limit(100_000).select(["aid"]).to_pandas()
            return sorted(df["aid"].tolist())
        except Exception:
            return []

    def values(self):
        # Delegate to items() so the node cache warms in the same scan.
        return [node for _aid, node in self.items()]

    def items(self):
        # Serve from the warmed node cache when we've previously scanned
        # the whole (ns, uid) partition and no writes have invalidated it.
        # Recall calls items() every query via _compute_entry_priors —
        # without this fast path each recall paid ~150ms for the scan.
        if self._cache_complete and self._node_cache:
            return list(self._node_cache.items())

        # Full-table to_arrow() then Python-side filter by (user_id, namespace).
        # search().where().to_arrow() silently materializes null list<struct>
        # cells as empty lists — a LanceDB query-engine quirk that erases
        # the null-vs-empty distinction for fields like topic_dicts /
        # entity_dicts / semantic_links. Full-table to_arrow() preserves
        # nulls correctly.
        #
        # We warm the node cache so downstream single-row get(aid) calls
        # (the hot recall path) resolve from the in-process dict, not Lance.
        try:
            try:
                self._archive_table.checkout_latest()
            except Exception:
                pass
            arr = self._archive_table.to_arrow()
            rows = arr.to_pylist()
            uid = self._uid()
            ns = self._namespace
            result = []
            for row in rows:
                if row.get("user_id") != uid:
                    continue
                if row.get("namespace") != ns:
                    continue
                aid_int = int(row["aid"])
                node = self._row_to_node(row)
                self._node_cache[aid_int] = node
                # Warm the archive-embedding cache from the same scan;
                # recall will call get_embedding(aid) hundreds of times.
                vec = row.get("vector")
                if vec is not None:
                    arr = np.array(vec, dtype=np.float32)
                    if np.allclose(arr, 0.0):
                        self._emb_cache[(aid_int, "archive")] = _MISSING_EMB
                    else:
                        self._emb_cache[(aid_int, "archive")] = arr
                else:
                    self._emb_cache[(aid_int, "archive")] = _MISSING_EMB
                result.append((aid_int, node))
            self._cache_complete = True
            return result
        except Exception as e:
            logger.debug("items() failed: %s", e)
            return []

    def pop(self, aid: int, *args):
        node = self.get(aid)
        if node is not None:
            del self[aid]
            return node
        if args:
            return args[0]
        raise KeyError(aid)

    def setdefault(self, aid: int, default=None):
        node = self.get(aid)
        if node is not None:
            return node
        if default is not None:
            self[aid] = default
        return default

    def clear(self) -> None:
        self._archive_table.delete(self._ns_filter())
        self._node_cache.clear()
        self._emb_cache.clear()
        self._cache_complete = False

    # ── Batch operations ──

    def put_batch(self, items: List[Tuple[int, Dict]]) -> None:
        """Batch insert/update for bulk ingestion."""
        if not items:
            return
        rows = []
        aids_written = []
        for aid, node in items:
            emb = node.get("embedding")
            vec = np.array(emb, dtype=np.float32).tolist() if emb is not None else self._zero_vector()
            if emb is not None and len(vec) != self._dim:
                vec = (vec[:self._dim] if len(vec) > self._dim
                       else vec + [0.0] * (self._dim - len(vec)))
            rows.append(self._node_to_row(int(aid), node, vec))
            aids_written.append(int(aid))
        # Build pyarrow Table with the explicit schema so uint64/struct/map
        # columns don't get auto-inferred down to (overflowing) int64 or
        # wrong struct field orders.
        table = pa.Table.from_pylist(rows, schema=self._archive_table.schema)
        try:
            (self._archive_table
             .merge_insert("aid")
             .when_matched_update_all()
             .when_not_matched_insert_all()
             .execute(table))
        except Exception as e:
            logger.warning("Batch upsert failed, falling back to sequential: %s", e)
            for aid in aids_written:
                try:
                    self._archive_table.delete(f"aid = {aid} AND {self._ns_filter()}")
                except Exception:
                    pass
            self._archive_table.add(table)
        for aid in aids_written:
            self._node_cache.pop(aid, None)
            self._emb_cache.pop((aid, "archive"), None)
        self._cache_complete = False

    # ══════════════════════════════════════════════════════════════
    # Embeddings (colocated + secondary table)
    # ══════════════════════════════════════════════════════════════

    def store_embedding(self, aid: int, embedding, emb_type: str = "archive") -> None:
        """Store an embedding. 'archive' type updates the main vector column.
        Other types (e.g. 'topic') go to the secondary embeddings table.
        """
        vec = np.array(embedding, dtype=np.float32).tolist()

        if emb_type == "archive":
            filt = f"aid = {int(aid)} AND {self._ns_filter()}"
            try:
                self._archive_table.update(where=filt, values={"vector": vec})
            except Exception as e:
                logger.debug("store_embedding update failed for aid=%d: %s", aid, e)
        else:
            row = {
                "aid": int(aid),
                "namespace": self._namespace,
                "user_id": self._uid(),
                "emb_type": emb_type,
                "vector": vec,
            }
            try:
                (self._embeddings_table
                 .merge_insert("aid")
                 .when_matched_update_all()
                 .when_not_matched_insert_all()
                 .execute([row]))
            except Exception:
                try:
                    self._embeddings_table.delete(
                        f"aid = {int(aid)} AND {self._ns_filter()} AND emb_type = '{emb_type}'"
                    )
                except Exception:
                    pass
                self._embeddings_table.add([row])

    def get_embedding(self, aid: int, emb_type: str = "archive"):
        """Retrieve an embedding by AID. Cached per (aid, emb_type)."""
        aid_int = int(aid)
        cache_key = (aid_int, emb_type)
        cached = self._emb_cache.get(cache_key)
        if cached is not None:
            # Sentinel for known-missing vectors so repeat misses don't re-query
            return None if cached is _MISSING_EMB else cached

        if emb_type == "archive":
            row = self._get_row(aid_int)
            if row is None:
                self._emb_cache[cache_key] = _MISSING_EMB
                return None
            vec = row["vector"]
            if vec is None:
                self._emb_cache[cache_key] = _MISSING_EMB
                return None
            arr = np.array(vec, dtype=np.float32)
            if np.allclose(arr, 0.0):
                self._emb_cache[cache_key] = _MISSING_EMB
                return None
            self._emb_cache[cache_key] = arr
            return arr
        else:
            # Secondary (e.g. topic) embeddings live in _embeddings_table.
            # Warm all of them at once on first miss — the proxy pattern
            # does per-aid reads, so one scan saves N round-trips.
            try:
                all_embs = self.get_all_embeddings(emb_type)
                for k, v in all_embs.items():
                    self._emb_cache[(int(k), emb_type)] = v
            except Exception:
                pass
            cached = self._emb_cache.get(cache_key)
            if cached is None:
                self._emb_cache[cache_key] = _MISSING_EMB
                return None
            return None if cached is _MISSING_EMB else cached

    def count_embeddings(self, emb_type: str = "archive") -> int:
        """Return the number of stored embeddings of this type WITHOUT
        materializing the vectors. Projects only the aid column.

        EmbeddingProxy.__len__ routes here on Lance — the fallback path
        pulls every row's 768-d vector off disk just to take a len(),
        which shows up as a fat hot-path item on recall profiles.
        """
        try:
            if emb_type == "archive":
                df = (self._archive_table.search()
                      .where(self._ns_filter())
                      .limit(100_000)
                      .select(["aid"])
                      .to_pandas())
                return len(df)
            filt = f"{self._ns_filter()} AND emb_type = '{emb_type}'"
            df = (self._embeddings_table.search()
                  .where(filt)
                  .limit(100_000)
                  .select(["aid"])
                  .to_pandas())
            return len(df)
        except Exception:
            return 0

    def get_all_embeddings(self, emb_type: str = "archive") -> dict:
        """Retrieve all embeddings as {aid: np.array} dict."""
        if emb_type == "archive":
            try:
                df = self._archive_table.search().where(self._ns_filter()).limit(100_000).select(["aid", "vector"]).to_pandas()
                result = {}
                for _, row in df.iterrows():
                    vec = row["vector"]
                    if vec is not None:
                        arr = np.array(vec, dtype=np.float32)
                        if not np.allclose(arr, 0.0):
                            result[int(row["aid"])] = arr
                return result
            except Exception:
                return {}
        else:
            try:
                filt = f"{self._ns_filter()} AND emb_type = '{emb_type}'"
                df = self._embeddings_table.search().where(filt).limit(100_000).select(["aid", "vector"]).to_pandas()
                return {int(row["aid"]): np.array(row["vector"], dtype=np.float32) for _, row in df.iterrows()}
            except Exception:
                return {}

    def store_embedding_batch(self, items, emb_type: str = "archive") -> None:
        """Batch store embeddings. items: [(aid, embedding), ...]"""
        if not items:
            return
        if emb_type == "archive":
            for aid, emb in items:
                vec = np.array(emb, dtype=np.float32).tolist()
                filt = f"aid = {int(aid)} AND {self._ns_filter()}"
                try:
                    self._archive_table.update(where=filt, values={"vector": vec})
                except Exception:
                    pass
        else:
            rows = [{
                "aid": int(aid),
                "namespace": self._namespace,
                "user_id": self._uid(),
                "emb_type": emb_type,
                "vector": np.array(emb, dtype=np.float32).tolist(),
            } for aid, emb in items]
            try:
                (self._embeddings_table
                 .merge_insert("aid")
                 .when_matched_update_all()
                 .when_not_matched_insert_all()
                 .execute(rows))
            except Exception:
                self._embeddings_table.add(rows)

    # ══════════════════════════════════════════════════════════════
    # VectorProvider -- dense similarity search
    # ══════════════════════════════════════════════════════════════

    def add(self, aid: int, embedding) -> None:
        """Add or update a vector for the given AID.

        If the AID already exists as an archive node, updates its vector column.
        If not, creates a minimal archive row with the vector.
        """
        vec = np.array(embedding, dtype=np.float32).tolist()

        if self.__contains__(aid):
            filt = f"aid = {int(aid)} AND {self._ns_filter()}"
            try:
                self._archive_table.update(where=filt, values={"vector": vec})
            except Exception as e:
                logger.debug("Vector update failed for aid=%d: %s", aid, e)
        else:
            row = {
                "aid": int(aid),
                "namespace": self._namespace,
                "user_id": self._uid(),
                "data": "{}",
                "vector": vec,
            }
            self._archive_table.add([row])

    def remove(self, aid: int) -> None:
        """Remove vector for the given AID (set to zero vector, keep node)."""
        filt = f"aid = {int(aid)} AND {self._ns_filter()}"
        try:
            self._archive_table.update(where=filt, values={"vector": self._zero_vector()})
        except Exception:
            pass

    def search(self, query_embedding, k: int) -> Tuple[List[int], List[float]]:
        """Search for k nearest vectors, filtered by namespace + user_id."""
        emb_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else list(query_embedding)

        try:
            results = (self._archive_table
                       .search(emb_list)
                       .where(self._ns_filter())
                       .limit(k)
                       .to_pandas())
        except Exception as e:
            logger.debug("Vector search failed: %s", e)
            return [], []

        if results.empty:
            return [], []

        aids = results["aid"].astype(int).tolist()
        if "_distance" in results.columns:
            distances = results["_distance"].tolist()
            scores = [max(0.0, 1.0 / (1.0 + d)) for d in distances]
        else:
            scores = [1.0] * len(aids)

        return aids, scores

    @property
    def dim(self) -> int:
        return self._dim

    # ── Cross-Namespace User-Scoped Access ──

    def get_user(self, aid: int, default=None):
        """Get node by AID across ALL namespaces for the same user_id.
        Uses _user_filter() (user_id only, no namespace restriction).
        Refreshes table snapshot to see cross-instance writes.
        """
        aid_int = int(aid)

        # Fast path: check _row_cache for a (aid, uid, *) match. The
        # cache is populated eagerly at recall entry via _get_row(-1);
        # a hit here avoids a 5-6ms cross-namespace FTS query.
        #
        # IMPORTANT: a cache miss does NOT imply the AID is absent.
        # Background workers (NLP, embedding, semantic linker) write
        # rows during recall — those writes appear in LanceDB but not
        # in our snapshot. Falling through to the filtered search on
        # miss keeps correctness intact; we only skip LanceDB on
        # positive cache hits.
        if self._row_cache is not None:
            uid = self._uid()
            for (_aid, _uid, _ns), _row in self._row_cache.items():
                if _aid == aid_int and _uid == uid:
                    return self._row_to_node(_row)
            # Cache miss — fall through to LanceDB; do not short-circuit.

        # Slow path: filtered cross-namespace search.
        filt = f"aid = {aid_int} AND {self._user_filter()}"
        try:
            self._archive_table.checkout_latest()
            arr = self._archive_table.search().where(filt).limit(1).to_arrow()
            rows = arr.to_pylist()
            if not rows:
                return default
            return self._row_to_node(rows[0])
        except Exception:
            return default

    def search_user(self, query_embedding, k: int) -> Tuple[List[int], List[float]]:
        """Vector search across ALL namespaces for the same user_id.
        Used by linking workers to find cross-namespace candidates.
        Refreshes table snapshot to see cross-instance writes.
        """
        try:
            self._archive_table.checkout_latest()
        except Exception:
            pass
        emb_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else list(query_embedding)
        try:
            results = (self._archive_table
                       .search(emb_list)
                       .where(self._user_filter())
                       .limit(k)
                       .to_pandas())
        except Exception as e:
            logger.debug("User-scoped vector search failed: %s", e)
            return [], []

        if results.empty:
            return [], []

        aids = results["aid"].astype(int).tolist()
        if "_distance" in results.columns:
            distances = results["_distance"].tolist()
            scores = [max(0.0, 1.0 / (1.0 + d)) for d in distances]
        else:
            scores = [1.0] * len(aids)
        return aids, scores

    # ── VectorProvider backward compat ──

    @property
    def index(self):
        return _LanceIndexProxy(self._archive_table, self._ns_filter())

    @index.setter
    def index(self, value):
        pass  # Allow setting to None for cleanup

    @property
    def _pending(self):
        return {}

    @_pending.setter
    def _pending(self, value):
        pass

    @property
    def _dirty(self):
        return False

    @_dirty.setter
    def _dirty(self, value):
        pass

    def _ensure_index(self):
        """No-op -- LanceDB manages its index automatically."""
        pass

    def debug_state(self):
        try:
            count = len(self)
        except Exception:
            count = -1
        sparse_count = 0
        try:
            df = self._sparse_table.search().where(self._ns_filter()).limit(100_000).select(["aid"]).to_pandas()
            sparse_count = len(df)
        except Exception:
            pass
        return {
            "provider": "lance",
            "db_path": self._db_path,
            "namespace": self._namespace,
            "user_id": self._user_id,
            "archive_count": count,
            "sparse_count": sparse_count,
            "sparse_table": self._sparse_table_name,
            "dim": self._dim,
            "tables": self._db.table_names(),
        }

    # ══════════════════════════════════════════════════════════════
    # SparseProvider -- full-text search (replaces BM25)
    # ══════════════════════════════════════════════════════════════

    def add_sparse(self, texts: List[str], ids: List[int],
                   topic_overlays=None, **kwargs) -> None:
        """SparseProvider.add -- add documents to the FTS index.

        Named add_sparse to disambiguate from VectorProvider.add.
        The factory/caller uses this as the SparseProvider.add method.

        Args:
            texts: List of text strings to index.
            ids: List of AIDs corresponding to each text.
            topic_overlays: Optional topic overlay texts (appended to main text).
        """
        if not texts or not ids:
            return

        rows = []
        for i, (text, aid) in enumerate(zip(texts, ids)):
            # Append topic overlay if provided
            full_text = text
            if topic_overlays and i < len(topic_overlays) and topic_overlays[i]:
                full_text = f"{text} {topic_overlays[i]}"

            rows.append({
                "aid": int(aid),
                "namespace": self._namespace,
                "user_id": self._uid(),
                "text": full_text,
            })

        if rows:
            try:
                self._sparse_table.add(rows)
                logger.debug("add_sparse: added %d rows to %s", len(rows), self._sparse_table_name)
            except Exception as e:
                logger.error("add_sparse FAILED for %s: %s", self._sparse_table_name, e)

    def remove_sparse(self, aid: int) -> None:
        """SparseProvider.remove -- remove document by AID from FTS index."""
        filt = f"aid = {int(aid)} AND {self._ns_filter()}"
        try:
            self._sparse_table.delete(filt)
        except Exception:
            pass

    def search_sparse(self, query: str, k: int) -> Tuple[List[int], List[float]]:
        """SparseProvider.search -- full-text search for k most relevant docs.

        Named search_sparse to disambiguate from VectorProvider.search.
        Returns (aids, scores) matching the SparseProvider contract.
        """
        if not query or not query.strip():
            return [], []

        # Ensure FTS index is up to date
        self._ensure_fts_index()

        try:
            results = (self._sparse_table
                       .search(query, query_type="fts")
                       .where(self._ns_filter())
                       .limit(k)
                       .to_pandas())
        except Exception as e:
            logger.debug("FTS search failed: %s", e)
            return [], []

        if results.empty:
            return [], []

        aids = results["aid"].astype(int).tolist()
        scores = results["_score"].tolist() if "_score" in results.columns else [1.0] * len(aids)

        return aids, scores

    def search_user_sparse(self, query: str, k: int) -> Tuple[List[int], List[float]]:
        """FTS search across ALL namespaces for the same user_id.
        Mirrors search_user() (vector) for sparse/FTS queries.
        """
        if not query or not query.strip():
            return [], []

        self._ensure_fts_index()

        try:
            self._sparse_table.checkout_latest()
        except Exception:
            pass

        try:
            results = (self._sparse_table
                       .search(query, query_type="fts")
                       .where(self._user_filter())
                       .limit(k)
                       .to_pandas())
        except Exception as e:
            logger.debug("User-scoped FTS search failed: %s", e)
            return [], []

        if results.empty:
            return [], []

        aids = results["aid"].astype(int).tolist()
        scores = results["_score"].tolist() if "_score" in results.columns else [1.0] * len(aids)

        return aids, scores

    # ── SparseProvider backward compat ──

    @property
    def corpus(self):
        """Return corpus texts for backward compat diagnostics."""
        try:
            df = self._sparse_table.search().where(self._ns_filter()).limit(100_000).select(["text"]).to_pandas()
            return df["text"].tolist()
        except Exception:
            return []

    @property
    def ids(self):
        """Return corpus IDs for backward compat diagnostics."""
        try:
            df = self._sparse_table.search().where(self._ns_filter()).limit(100_000).select(["aid"]).to_pandas()
            return df["aid"].tolist()
        except Exception:
            return []

    @property
    def id_map(self):
        """Return {aid: index} map for backward compat."""
        return {aid: i for i, aid in enumerate(self.ids)}

    @property
    def _retriever(self):
        return None

    @_retriever.setter
    def _retriever(self, value):
        pass

    # ══════════════════════════════════════════════════════════════
    # Persistence
    # ══════════════════════════════════════════════════════════════

    def flush_dirty(self) -> int:
        """Compact all Lance tables to reclaim space from updates/deletes."""
        try:
            self._archive_table.compact_files()
            self._embeddings_table.compact_files()
            self._sparse_table.compact_files()
        except Exception as e:
            logger.debug("Compaction failed: %s", e)
        return 0

    def save(self, path: str) -> None:
        """LanceDB auto-persists. Save triggers compaction + cleanup."""
        for tbl in (self._archive_table, self._embeddings_table, self._sparse_table):
            if tbl is not None:
                try:
                    tbl.compact_files()
                    tbl.cleanup_old_versions()
                except Exception as e:
                    logger.debug("Lance save/compact failed: %s", e)

    def load(self, path: str, **kwargs) -> None:
        """Reopen database from path. Supports VectorProvider.load signature."""
        if path and path != self._db_path:
            self._db_path = path
            self._db = self._lancedb.connect(path)
            self._open_tables()
        return True  # VectorProvider compat

    def close(self) -> None:
        """Release LanceDB connection."""
        self._archive_table = None
        self._embeddings_table = None
        self._sparse_table = None
        self._sequences_table = None
        self._db = None

    # ══════════════════════════════════════════════════════════════
    # Migration from TinyDB
    # ══════════════════════════════════════════════════════════════

    def import_from_tinydb(self, archive_json_path: str,
                           embeddings_json_path: str = None,
                           topic_embeddings_json_path: str = None) -> int:
        """Import data from existing TinyDB JSON files into LanceDB.
        Returns number of nodes imported.
        """
        count = 0

        # Load archive embeddings first (so we can colocate them)
        emb_map = {}
        if embeddings_json_path and os.path.exists(embeddings_json_path):
            with open(embeddings_json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            emb_data = raw.get("_default", raw)
            for aid_str, val in emb_data.items():
                try:
                    aid = int(aid_str)
                    emb = val if isinstance(val, list) else val.get("embedding", [])
                    if emb and len(emb) == self._dim:
                        emb_map[aid] = emb
                except (ValueError, TypeError):
                    continue

        # Import archive nodes
        if os.path.exists(archive_json_path):
            with open(archive_json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            nodes = raw.get("_default", raw)
            rows = []
            for aid_str, node in nodes.items():
                try:
                    aid = int(aid_str)
                except (ValueError, TypeError):
                    continue

                vec = emb_map.get(aid)
                if vec is not None:
                    vec = np.array(vec, dtype=np.float32).tolist()
                else:
                    node_emb = node.get("embedding")
                    if node_emb and len(node_emb) == self._dim:
                        vec = np.array(node_emb, dtype=np.float32).tolist()
                    else:
                        vec = self._zero_vector()

                rows.append(self._node_to_row(aid, node, vec))

            if rows:
                BATCH_SIZE = 500
                sch = self._archive_table.schema
                for i in range(0, len(rows), BATCH_SIZE):
                    batch = pa.Table.from_pylist(rows[i:i + BATCH_SIZE], schema=sch)
                    self._archive_table.add(batch)
                count = len(rows)
                logger.info("Imported %d archive nodes into LanceDB", count)

        # Import topic embeddings into secondary table
        if topic_embeddings_json_path and os.path.exists(topic_embeddings_json_path):
            with open(topic_embeddings_json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            temb_data = raw.get("_default", raw)
            rows = []
            for aid_str, val in temb_data.items():
                try:
                    aid = int(aid_str)
                    emb = val.get("topic_embedding", []) if isinstance(val, dict) else val
                    if emb and len(emb) == self._dim:
                        rows.append({
                            "aid": aid,
                            "namespace": self._namespace,
                            "user_id": self._uid(),
                            "emb_type": "topic",
                            "vector": np.array(emb, dtype=np.float32).tolist(),
                        })
                except (ValueError, TypeError):
                    continue
            if rows:
                BATCH_SIZE = 500
                for i in range(0, len(rows), BATCH_SIZE):
                    self._embeddings_table.add(rows[i:i + BATCH_SIZE])
                logger.info("Imported %d topic embeddings into LanceDB", len(rows))

        return count


class LanceSparseProvider(LanceStorageProvider):
    """Thin wrapper that exposes SparseProvider methods under the ABC names.

    The base LanceStorageProvider uses add_sparse/remove_sparse/search_sparse
    to avoid name collisions with VectorProvider. This subclass maps the
    SparseProvider ABC methods (add/remove/search) to those sparse-specific
    implementations.

    Used when creating a standalone sparse index via create_sparse("lance").
    The archive and vector tables still exist but are unused in this role.

    Usage:
        sparse = create_sparse("lance", db_path="data/lance.db",
                                sparse_table="sparse_entity")
        sparse.add(["John Doe", "Jane Smith"], [1, 2])
        aids, scores = sparse.search("John", k=5)
    """

    def add(self, texts: List[str], ids: List[int],
            topic_overlays=None, **kwargs) -> None:
        """SparseProvider.add -- delegates to add_sparse."""
        return self.add_sparse(texts, ids, topic_overlays=topic_overlays, **kwargs)

    def remove(self, aid: int) -> None:
        """SparseProvider.remove -- delegates to remove_sparse."""
        return self.remove_sparse(aid)

    def search(self, query, k: int) -> Tuple[List[int], List[float]]:
        """SparseProvider.search -- delegates to search_sparse.

        Handles both str (FTS query) and array-like (vector query) inputs.
        If given a string, routes to FTS. If given an array, routes to vector search.
        """
        if isinstance(query, str):
            return self.search_sparse(query, k)
        else:
            # Array-like: delegate to vector search
            return super().search(query, k)


class _LanceIndexProxy:
    """Minimal proxy so code that checks .index.ntotal still works."""

    def __init__(self, table, ns_filter):
        self._table = table
        self._ns_filter = ns_filter

    @property
    def ntotal(self):
        try:
            df = self._table.search().where(self._ns_filter).limit(100_000).select(["aid"]).to_pandas()
            return len(df)
        except Exception:
            return 0
