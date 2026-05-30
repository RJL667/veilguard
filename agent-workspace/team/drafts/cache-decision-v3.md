# TCMM Render Caching Strategy

## TLDR

Cache TCMM renders at the prefix boundary using a short TTL (≤60 s), invalidating on any write to the live memory region. Do not cache the volatile tail.

## Recommendation

**Cache hit rates:** The cacheable prefix (preamble + stable live blocks) is read-heavy and changes infrequently; hit rates above 80% are realistic for active sessions. The volatile tail changes every turn and must never be cached.

**Staleness risk:** A write to any live block must immediately invalidate the prefix cache. Without write-through invalidation, a stale render can surface evicted or overwritten blocks, corrupting recall.

**Memory overhead:** One cached prefix per active session. At ~8 KB per render, 1 000 concurrent sessions ≈ 8 MB — negligible. Evict on session idle >5 min.

**Operational complexity:** Keep invalidation logic co-located with the TCMM write path. A separate cache service adds a failure domain; an in-process LRU with a write hook is sufficient and easier to reason about under incident conditions.
