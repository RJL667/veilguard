# TCMM Render Caching — Internal Decision Memo

## TLDR

Caching TCMM renders offers measurable latency gains but introduces staleness risk that is structurally incompatible with TCMM's core contract: memory blocks must reflect the most recent heat scores, PII-gateway redactions, and dream-cycle outputs at inference time. A stale render served from cache can silently surface evicted blocks or miss newly promoted ones, degrading recall quality in ways that are invisible to the caller.

## Recommendation

**Do not cache TCMM renders at the render layer.** Cache at lower-cost boundaries instead: cache raw archive fetches (BM25/vector retrieval results) with a short TTL (≤30 s), and cache dream-summary blocks independently since they are explicitly versioned. Full render assembly is fast relative to retrieval I/O and must remain live to honour heat-reinforcement writes from each `tcmm_record_turn` call. Operational complexity of a render cache (invalidation on heat updates, PII-token rotation, dream-cycle completions) outweighs the marginal latency saving.
