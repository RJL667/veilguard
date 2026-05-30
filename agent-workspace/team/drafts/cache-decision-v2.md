# TCMM Render Caching — Internal Decision Memo

**To:** Phishield Engineering  
**Re:** Whether to cache TCMM context renders

---

## TLDR

Caching TCMM renders offers measurable latency gains but introduces staleness risk that is structurally incompatible with TCMM's heat-reinforcement model. The cost of serving a stale context (wrong block promotion, missed PII rehydration, incorrect recall ranking) outweighs the performance benefit for most request patterns.

---

## Recommendation

**Do not cache full renders. Cache selectively at the block level with short TTLs.**

- **Performance:** Full-render caching saves ~40–80 ms per request but TCMM renders are already incremental; the bottleneck is retrieval (BM25 + dense vector), not serialisation.
- **Staleness:** Each user turn mutates heat scores. A cached render from even one turn ago reflects stale heat, causing the wrong blocks to be promoted or evicted in the next cycle.
- **Consistency:** PII rehydration and REF token resolution happen at render time. Caching a rendered context risks serving resolved PII tokens to the wrong session if cache keys are misconfigured — a direct POPIA violation risk.
- **Mitigation:** Cache individual cold archive blocks (src=shadow, heat < 0.1, TTL ≤ 60 s) where mutation probability is near zero. Live-tier blocks must never be cached across turns.

