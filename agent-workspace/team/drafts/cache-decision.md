# TCMM Render Caching — Decision Memo

## TLDR

Caching TCMM renders is **not viable** in its current form because the heat-driven recall model produces renders that are per-query dynamic, making cache hit rates too low to justify the invalidation and memory overhead.

## Recommendation

**Do not implement render caching at this time.** Three concrete trade-offs drive this decision:

1. **Cache invalidation complexity vs. latency gain**: TCMM renders are invalidated whenever any block's heat score changes — which happens on every turn via the `used` map. A write-through or TTL cache would either stale-serve incorrect heat rankings or expire so aggressively that hit rates approach zero, negating latency gains.

2. **Memory overhead vs. recall accuracy**: Storing rendered context snapshots per query requires significant memory proportional to context window size. Because heat scores shift continuously, cached snapshots diverge from ground truth within one or two turns, degrading recall accuracy in ways that are silent and hard to detect.

3. **Latency gains are marginal**: The dominant latency cost in TCMM is the BM25 + dense vector + cross-encoder rerank pipeline, not the final render step. Caching renders skips only the cheapest stage, yielding negligible end-to-end improvement while adding cache management complexity.

**Revisit** if TCMM introduces stable, heat-frozen "dream" snapshots that are explicitly versioned — those would be safe cache targets.
