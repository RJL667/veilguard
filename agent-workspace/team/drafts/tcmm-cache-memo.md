# TCMM Render Caching Strategy — Internal Architecture Memo

## TLDR

Cache the rendered system-message prefix (preamble + live memory blocks) at the KV layer. Invalidate on any live-block mutation. Accept up to ~30 s of staleness on read-heavy sessions; force synchronous invalidation on writes.

## Tradeoffs

| Dimension | Aggressive caching | Conservative caching |
|---|---|---|
| **Latency** | Low TTFT; prefix reuse avoids re-tokenisation | Full re-render on every turn; higher TTFT |
| **Memory** | One cached prefix per user session; bounded | No cache overhead |
| **Staleness** | Stale blocks served until TTL expires or write invalidates | Always fresh; no stale-read risk |
| **Consistency** | Write-through required; missed invalidation = ghost blocks | Trivially consistent |

Key tension: the live-block region is append-heavy during active incidents. A TTL-only strategy will serve stale heat scores and outdated block content during fast-moving IR sessions. Write-through invalidation eliminates this but adds a synchronous cache-bust on every `observe()` call.

Volatile (shadow) blocks are never cached — they are turn-scoped and assembled fresh each inference.

## Recommendation

Use **write-through invalidation with a 30 s fallback TTL**. On any live-block write (observe, dream-cycle promotion, eviction), immediately invalidate the affected user's cached prefix. The 30 s TTL is a safety net for missed invalidations only, not the primary freshness mechanism. Cache keys should be scoped to `(user_id, live_block_hash)` so a hash mismatch auto-busts without an explicit invalidation signal. Do not cache volatile/shadow blocks under any circumstances.
