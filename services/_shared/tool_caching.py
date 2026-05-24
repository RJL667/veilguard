"""TTL cache decorator for MCP tools.

Caches tool results in-memory per process, keyed by tool name + args
hash. The same query within the TTL window returns the cached result
plus a small "(cached, age Ns)" footer so the LLM knows it didn't go
out to the upstream service.

Use cases this is designed for:
- ``web.google_search(query)`` — same query → same results for 5-10min
- ``web.browse_url(url)`` — same URL → same content for ~5min
- ``documents.read_pdf(path)`` — same file → same text (until mtime
  changes; for that case, use ``cached_with_mtime`` instead)
- ``tcmm`` recall calls — same key → same memory

NOT designed for:
- File-mutating tools (writes shouldn't be cached)
- Tools that depend on external state with no good cache key
- Tools whose freshness matters more than latency (e.g. real-time data)

In-memory only — process restart wipes the cache. That's intentional:
a hot cache should not survive a crash that might have been caused by
stale data, and persistence would require coordination across MCP
server processes that aren't worth the complexity.

The decorator works on both ``async def`` and ``def`` tools — it
detects awaitability via ``inspect.iscoroutinefunction``.
"""

import functools
import hashlib
import inspect
import json
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger("veilguard.tool_caching")

# Cache structure: {tool_name: {arg_hash: (expires_at, result, hit_count)}}
# Per-tool sub-dict so a single tool's stale entries can be flushed
# independently and the cache size per tool can be bounded.
_CACHE: dict[str, dict[str, tuple[float, Any, int]]] = {}

# Hard cap on entries per tool — if a tool gets called with thousands
# of distinct arg combos in one process lifetime, we don't want the
# cache to grow unboundedly. Oldest-by-expiry entries get evicted first
# (rough LRU; not strict LRU because we don't track access time).
_MAX_ENTRIES_PER_TOOL = 200


def _hash_args(args: tuple, kwargs: dict) -> str:
    """Canonical string hash of the call arguments."""
    # JSON-serialise so dict ordering doesn't change the hash. Fall back
    # to repr for unhashable / un-serialisable args (rare for MCP tools
    # which take primitive types).
    try:
        payload = json.dumps([args, kwargs], sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = repr((args, kwargs))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _prune(tool_name: str) -> None:
    """Drop expired entries; if still over cap, drop oldest by expiry."""
    bucket = _CACHE.get(tool_name)
    if not bucket:
        return
    now = time.time()
    expired = [k for k, (exp, _, _) in bucket.items() if exp <= now]
    for k in expired:
        del bucket[k]
    if len(bucket) > _MAX_ENTRIES_PER_TOOL:
        # Drop oldest-expiry entries until under cap.
        sorted_keys = sorted(bucket.items(), key=lambda kv: kv[1][0])
        for k, _ in sorted_keys[: len(bucket) - _MAX_ENTRIES_PER_TOOL]:
            del bucket[k]


def cached(ttl_seconds: int = 300, name: Optional[str] = None,
            footer: bool = True) -> Callable:
    """Decorate a tool function so identical-arg calls return cached results.

    Args:
        ttl_seconds: How long a cached entry stays valid. Tune per tool —
            web search results 600s, browsed pages 300s, file reads 60s
            (file reads should probably use ``cached_with_mtime`` instead).
        name: Override the cache key namespace. Defaults to ``fn.__name__``;
            override only if you need two functions to share a cache.
        footer: When True (default), append a ``\\n\\n_[cache] hit (age Ns)_``
            line to cached results so the LLM sees that the result was
            served from cache. Set False if the tool's output format must
            be byte-identical to the upstream call (e.g. when piping
            through structured parsers).
    """
    def decorator(fn: Callable) -> Callable:
        tool_name = name or fn.__name__
        is_async = inspect.iscoroutinefunction(fn)

        if is_async:
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                key = _hash_args(args, kwargs)
                bucket = _CACHE.setdefault(tool_name, {})
                _prune(tool_name)
                entry = bucket.get(key)
                now = time.time()
                if entry and entry[0] > now:
                    expires_at, result, hits = entry
                    bucket[key] = (expires_at, result, hits + 1)
                    age = int(ttl_seconds - (expires_at - now))
                    if footer and isinstance(result, str):
                        return result + f"\n\n_[cache] hit (age {age}s, hits={hits + 1})_"
                    return result
                # Miss — compute, store, return raw (no footer on first
                # call so it looks identical to a non-cached invocation).
                result = await fn(*args, **kwargs)
                bucket[key] = (now + ttl_seconds, result, 0)
                return result
            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            key = _hash_args(args, kwargs)
            bucket = _CACHE.setdefault(tool_name, {})
            _prune(tool_name)
            entry = bucket.get(key)
            now = time.time()
            if entry and entry[0] > now:
                expires_at, result, hits = entry
                bucket[key] = (expires_at, result, hits + 1)
                age = int(ttl_seconds - (expires_at - now))
                if footer and isinstance(result, str):
                    return result + f"\n\n_[cache] hit (age {age}s, hits={hits + 1})_"
                return result
            result = fn(*args, **kwargs)
            bucket[key] = (now + ttl_seconds, result, 0)
            return result
        return sync_wrapper
    return decorator


def cached_with_mtime(path_arg: str = "path", ttl_seconds: int = 3600,
                       name: Optional[str] = None, footer: bool = True) -> Callable:
    """Like ``cached``, but the cache key includes the file's mtime.

    Use for file-reading tools: the same path returning the same content
    is only valid while the file hasn't been modified. Adding mtime to
    the key means an edit to the file invalidates the cache automatically
    on the next call without us having to track external mutations.

    Args:
        path_arg: Name of the kwarg / positional arg that holds the file
            path. Used to look up the file's mtime via ``os.stat``.
        ttl_seconds: Belt-and-suspenders ceiling on cache age, in case a
            symlink target rotates without touching the link's mtime.
            1 hour is a reasonable default for source-code-style files.
        name: Override the cache namespace.
        footer: Whether to append the cache-hit footer.
    """
    import os

    def decorator(fn: Callable) -> Callable:
        tool_name = name or fn.__name__
        sig = inspect.signature(fn)
        is_async = inspect.iscoroutinefunction(fn)

        def _resolve_path(args: tuple, kwargs: dict) -> Optional[str]:
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                return bound.arguments.get(path_arg)
            except TypeError:
                return None

        def _key(args: tuple, kwargs: dict) -> str:
            path = _resolve_path(args, kwargs)
            mtime: float = -1.0
            if path:
                try:
                    mtime = os.stat(path).st_mtime
                except OSError:
                    mtime = -1.0
            base = _hash_args(args, kwargs)
            return f"{base}:{int(mtime)}"

        if is_async:
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                key = _key(args, kwargs)
                bucket = _CACHE.setdefault(tool_name, {})
                _prune(tool_name)
                entry = bucket.get(key)
                now = time.time()
                if entry and entry[0] > now:
                    expires_at, result, hits = entry
                    bucket[key] = (expires_at, result, hits + 1)
                    age = int(ttl_seconds - (expires_at - now))
                    if footer and isinstance(result, str):
                        return result + f"\n\n_[cache] hit (age {age}s, mtime-stable)_"
                    return result
                result = await fn(*args, **kwargs)
                bucket[key] = (now + ttl_seconds, result, 0)
                return result
            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            key = _key(args, kwargs)
            bucket = _CACHE.setdefault(tool_name, {})
            _prune(tool_name)
            entry = bucket.get(key)
            now = time.time()
            if entry and entry[0] > now:
                expires_at, result, hits = entry
                bucket[key] = (expires_at, result, hits + 1)
                age = int(ttl_seconds - (expires_at - now))
                if footer and isinstance(result, str):
                    return result + f"\n\n_[cache] hit (age {age}s, mtime-stable)_"
                return result
            result = fn(*args, **kwargs)
            bucket[key] = (now + ttl_seconds, result, 0)
            return result
        return sync_wrapper
    return decorator


def cache_stats() -> dict:
    """Return a per-tool snapshot of cache size and hit counts.

    Useful for an /api/admin/cache-stats endpoint or a periodic log.
    Doesn't reset counters — call ``cache_clear()`` for that.
    """
    snap = {}
    for tool, bucket in _CACHE.items():
        total_hits = sum(hits for _, _, hits in bucket.values())
        snap[tool] = {
            "entries": len(bucket),
            "total_hits": total_hits,
        }
    return snap


def cache_clear(tool_name: Optional[str] = None) -> None:
    """Drop all cached entries, or just one tool's entries."""
    if tool_name is None:
        _CACHE.clear()
    else:
        _CACHE.pop(tool_name, None)
