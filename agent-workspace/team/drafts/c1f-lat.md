# Claude Haiku vs Sonnet: Latency Analysis

**Time-to-first-token (TTFT):** Haiku delivers TTFT in ~300–500 ms; Sonnet typically ranges 600–1,200 ms — roughly 2× slower due to greater model depth.

**End-to-end latency:** For a 200-token response, Haiku completes in ~1–2 s; Sonnet in ~3–6 s under typical load.

**Throughput:** Haiku sustains ~80–120 tokens/s; Sonnet ~40–60 tokens/s, reflecting its larger parameter footprint.

**Recommendation:** Use Haiku for latency-sensitive, high-volume inference (chat, autocomplete). Prefer Sonnet where reasoning quality outweighs speed requirements.
