# Latency Profile: Claude Haiku vs Sonnet — Director Decision Analysis

*Research bridge offline during authoring; benchmark figures marked [unverified] where not independently confirmed.*

---

## Executive Summary

Claude Haiku and Claude Sonnet occupy distinct positions on the speed-quality spectrum. Haiku is optimised for low-latency, high-throughput workloads; Sonnet trades some speed for substantially deeper reasoning. For a Director agent whose primary job is task routing, proposal ranking, and orchestration-loop management, Haiku's latency advantage is decisive in most configurations. Sonnet becomes the right choice only when the Director must perform complex multi-step synthesis or reason over large accumulated context where decision quality directly affects downstream agent correctness.

---

## Response Time

**Time-to-first-token (TTFT)** is the most operationally relevant latency metric for streaming Director loops, because the Director can begin dispatching sub-agents as soon as the first tokens arrive.

- **Haiku TTFT:** typically 200–500 ms under normal API load [unverified]
- **Sonnet TTFT:** typically 600–1,500 ms under normal API load [unverified]

This 3–5× gap compounds in multi-hop orchestration. A Director that issues 20 routing decisions per workflow adds roughly 4–20 seconds of pure TTFT overhead with Sonnet vs 0.8–2 seconds with Haiku — a meaningful wall-clock difference for interactive or near-real-time pipelines.

**Total response latency** (full completion) depends heavily on output length. Director turns are typically short (50–200 output tokens), which keeps both models fast, but Haiku's higher tokens-per-second rate [unverified] means it finishes first even at equal output length.

---

## Throughput

Throughput matters when the Director must fan out many parallel sub-tasks or process high-volume event streams.

- **Haiku tokens/second:** approximately 100–150 tok/s output [unverified]
- **Sonnet tokens/second:** approximately 60–90 tok/s output [unverified]

At the API tier, Haiku also benefits from higher rate limits (requests per minute and tokens per minute) on Anthropic's standard tiers [unverified], meaning a Director running on Haiku is less likely to hit throttling during burst orchestration events. Sonnet's lower throughput ceiling can become a bottleneck when the Director is coordinating 10+ concurrent sub-agents and must process their results in rapid succession.

For batch-style Director workloads (e.g., nightly task-graph compilation, bulk proposal ranking), Haiku's throughput advantage translates directly into lower wall-clock job time and lower cost per unit of work.

---

## Load Behavior

Under sustained load, both models exhibit latency degradation as API concurrency increases, but the degradation curves differ:

**Haiku under load:** Latency increases modestly at moderate concurrency (5–20 parallel requests). The model's smaller parameter footprint means Anthropic can serve more instances per compute unit, providing better horizontal scaling headroom [unverified]. P95 latency remains relatively stable up to moderate load.

**Sonnet under load:** Latency is more sensitive to concurrency. At high request rates, P95 TTFT can spike significantly above median values [unverified]. For a Director that issues bursts of requests (e.g., spawning 15 sub-agents simultaneously and waiting for all to complete), Sonnet's tail latency under load is a material risk to overall pipeline SLA.

**Context-window effects:** Both models support 200K-token context windows. However, Sonnet processes long contexts more accurately under complex multi-agent state accumulation. Haiku's latency advantage narrows slightly as context length grows, but it retains a speed edge throughout the practical range of Director context sizes (typically 10K–80K tokens for orchestration state).

**Cold-start behaviour:** Neither model has a meaningful cold-start penalty at the API level (Anthropic manages instance warm-up transparently), so this is not a differentiating factor.

---

## Recommendation

**Default to Haiku as the Director model.** The latency and throughput advantages are decisive for orchestration workloads:

1. **TTFT 3–5× lower** — critical for interactive pipelines and tight orchestration loops.
2. **Higher throughput** — reduces bottleneck risk during sub-agent fan-out.
3. **Better load behaviour** — more stable P95 latency under burst concurrency.
4. **Lower cost** — Haiku is approximately 9× cheaper per token [unverified], so the latency win comes with a cost win, not a trade-off.

**Switch to Sonnet as Director only when:**
- The Director must perform deep multi-step reasoning over ambiguous or conflicting sub-agent outputs.
- The task graph exceeds ~15 interdependent nodes where routing errors cascade.
- Accumulated context exceeds ~100K tokens and decision quality is safety-critical.

In practice, a hybrid approach works well: run Haiku as the default Director, with a Sonnet escalation path triggered by a confidence-threshold check on routing decisions. This captures Haiku's latency benefits for the 80–90% of routine orchestration turns while preserving Sonnet's reasoning depth for the edge cases that warrant it.
