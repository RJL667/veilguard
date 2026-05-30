# TCMM Render Caching Strategy

## TLDR

Caching the rendered live-tier prefix eliminates redundant KV recomputation, cutting per-request latency. Staleness risk is real: hit rates are high when block churn is low but degrade sharply during dream-cycle writes or rapid promotion/eviction. Memory overhead is manageable at typical live-tier sizes (~8–32 blocks) but must be bounded.

## Recommendation

Cache the live-tier prefix only. Invalidate on any block mutation (write, promote, evict). Set a hard TTL of 60 seconds as a staleness ceiling. Never cache the volatile/shadow tail — it changes every turn. If hit rate falls below 70%, throttle dream-cycle scheduling to restore cache efficiency.
