# Claude Haiku vs Sonnet: Latency Comparison for Director Agent Operations

> **Note:** Web research bridge was offline during authorship. Figures drawn from Anthropic public documentation and community benchmarks known as of early 2025. Claims that could not be independently verified are marked [unverified]. Verify at https://www.anthropic.com/pricing and https://docs.anthropic.com before acting on specific numbers.

---

## RESPONSE_TIME

Haiku delivers time-to-first-token (TTFT) in the range of 300–500 ms under typical load, compared to 700–1,200 ms for Sonnet [unverified]. For Director agent operations — where the orchestrator must dispatch subtasks, parse tool results, and issue follow-up instructions in rapid succession — this 2–3× TTFT advantage compounds across every hop in the task graph. A 10-hop shallow pipeline can accumulate 4–7 seconds of latency savings on Haiku vs Sonnet, which is material for interactive or near-real-time workloads.

---

## THROUGHPUT

Haiku sustains higher tokens-per-second generation rates than Sonnet [unverified], making it preferable for high-frequency Director loops that emit many short orchestration messages. Under concurrent load (multiple parallel subtasks reporting back simultaneously), Haiku's lower per-request compute footprint also means less queuing delay at the API tier. Sonnet's throughput advantage is in quality-per-token, not volume.

---

## CONTEXT_LIMITS

Both models support a 200K-token context window. However, Sonnet handles dense, deeply nested multi-agent state more reliably at the upper end of that window [unverified]. Haiku may exhibit accuracy degradation when the accumulated Director context (task graph state, tool outputs, agent histories) approaches 100K+ tokens. For shallow task graphs with modest context, this distinction is irrelevant.

---

## KEY_FINDING

**Default to Haiku as Director for latency-sensitive pipelines with shallow-to-moderate task graphs.** The 2–3× TTFT advantage and lower per-hop cost make it the correct default for most Phishield SME workloads (phishing triage, alert enrichment, classification). Switch to Sonnet only when the Director must reason over large accumulated context (>100K tokens) or coordinate many tightly interdependent sub-agents where decision quality outweighs speed.
