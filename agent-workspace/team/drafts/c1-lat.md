# Haiku vs Sonnet Latency Analysis

**Time-to-first-token:** Haiku delivers ~300–500 ms TTFT versus Sonnet's ~700–1200 ms — roughly 2–3× faster initial response.

**End-to-end latency:** Haiku completes short Director operations in ~1–2 s; Sonnet requires ~3–6 s for equivalent tasks.

**Director impact:** Haiku's lower latency enables faster task-routing decisions and tighter orchestration loops, while Sonnet's higher latency is justified only for complex reasoning steps where output quality outweighs speed cost.
