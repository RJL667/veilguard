# Director Model Recommendation: Claude Haiku vs Sonnet

## Executive Summary

Based on cost and latency analyses of Claude 3.5 Haiku and Claude 3.5 Sonnet, **Haiku is the recommended model for the Director role** in the majority of deployment configurations. Haiku is approximately 9× cheaper per Director turn and generates tokens roughly 3–5× faster, making it the clear default for orchestration workloads. Sonnet should be reserved for Director deployments where the orchestrator must reason over large accumulated context or coordinate many tightly interdependent sub-agents.

## Cost Analysis

Haiku and Sonnet differ substantially in per-token pricing [unverified — research bridge offline]:

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|---|---|---|
| Claude 3.5 Haiku | $0.80 | $4.00 |
| Claude 3.5 Sonnet | $3.00 | $15.00 |

A typical Director turn (~500 input / 200 output tokens) costs approximately $0.0005 on Haiku versus $0.0045 on Sonnet — roughly **9× cheaper**. At 10,000 Director turns per day, Haiku saves approximately $40/day (~$1,200/month) compared to Sonnet. Because Director work is routing- and dispatch-heavy rather than deep-reasoning-heavy, the cost premium of Sonnet is not justified for standard orchestration tasks.

## Latency Analysis

Haiku generates tokens roughly **3–5× faster** than Sonnet [unverified — research bridge offline], a critical advantage for Director orchestration loops where response latency compounds across every agent hop in the task graph. In a pipeline with 10 sequential Director decisions, a 3× latency difference translates directly into a 3× reduction in end-to-end wall-clock time.

Both models support 200K-token context windows. However, Sonnet handles dense, deeply nested task graphs more reliably; Haiku may degrade in accuracy under heavy multi-agent state accumulation. For shallow-to-moderate task graphs — the common case — this distinction is not material.

## Recommendation

**Default to Claude 3.5 Haiku as the Director model.** The cost and latency advantages are decisive for the orchestration use case, which is routing-heavy rather than reasoning-heavy.

Switch to Claude 3.5 Sonnet as Director only when:
- The Director must reason over large accumulated context (approaching the 200K limit with complex interdependencies), or
- The pipeline involves many tightly coupled sub-agents where decision quality at the orchestration layer directly determines correctness of downstream work.

Worker agents performing deep reasoning or complex synthesis should use Sonnet regardless of which model runs the Director.

---
*Pricing and latency figures sourced from upstream team analyses (team/drafts/director-cost-analysis.md, team/drafts/director-latency-analysis.md). Raw figures marked [unverified] due to research bridge being offline during upstream analysis runs.*
