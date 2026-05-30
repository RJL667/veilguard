# Cost Analysis: Claude Haiku vs Sonnet for Director Workloads

## Executive Summary

Claude 3.5 Haiku is the cost-efficient default for Director orchestration. At roughly 4–9× lower token cost and 3–5× faster inference than Claude 3.5 Sonnet [unverified — research bridge offline], Haiku handles the routing-heavy, low-reasoning-depth work that dominates Director loops. Sonnet's higher cost is justified only when the Director must perform deep multi-step reasoning, resolve ambiguous task decomposition, or synthesise large accumulated context. For most SME-scale deployments, running the Director on Haiku and reserving Sonnet for specialist worker agents is the optimal split.

---

## Pricing Model

Anthropic prices both models on a per-million-token basis, billed separately for input and output tokens. Output tokens are consistently more expensive than input tokens across both models.

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|---|---|---|
| Claude 3.5 Haiku | $0.80 | $4.00 |
| Claude 3.5 Sonnet | $3.00 | $15.00 |

*Source: Anthropic published rate card [unverified — research bridge offline]. Figures reflect pricing as of late 2024; verify against [https://www.anthropic.com/pricing](https://www.anthropic.com/pricing) before budgeting.*

Key structural differences:
- **Haiku** is optimised for high-throughput, low-latency tasks. Its pricing reflects a commodity-tier positioning.
- **Sonnet** is a mid-tier reasoning model. Its 3.75× input and 3.75× output premium over Haiku reflects meaningfully higher capability for complex synthesis and judgment tasks.
- Both models share a 200K-token context window, so context-length alone does not differentiate cost structure.

---

## Token Cost Breakdown

### Typical Director Turn Profile

A Director turn in an orchestration loop typically involves:
- **Input:** system prompt (~300 tokens) + task graph state (~150 tokens) + agent response (~50 tokens) ≈ **500 tokens input**
- **Output:** routing decision or task proposal ≈ **150–200 tokens output**

| Model | Input cost (500 tok) | Output cost (175 tok avg) | Total per turn |
|---|---|---|---|
| Claude 3.5 Haiku | $0.00040 | $0.00070 | **~$0.0011** |
| Claude 3.5 Sonnet | $0.00150 | $0.00263 | **~$0.0041** |

Sonnet costs approximately **3.7× more per Director turn** than Haiku under this profile [unverified].

### Scale Projections

| Daily Director turns | Haiku daily cost | Sonnet daily cost | Daily saving (Haiku) |
|---|---|---|---|
| 1,000 | $1.10 | $4.10 | $3.00 |
| 10,000 | $11.00 | $41.00 | $30.00 |
| 100,000 | $110.00 | $410.00 | $300.00 |

At 10,000 Director turns per day — a moderate production workload — Haiku saves approximately **$900/month** over Sonnet.

---

## Total Cost of Ownership (TCO) Estimate

TCO for a Director agent extends beyond raw token cost:

### Monthly cost model (10,000 Director turns/day, 30 days)

| Cost component | Haiku | Sonnet |
|---|---|---|
| Token cost (Director turns) | $330 | $1,230 |
| Worker agent tokens (Sonnet, fixed) | $1,500 | $1,500 |
| Retry overhead (~5% of turns) | $17 | $62 |
| **Total estimated monthly** | **~$1,847** | **~$2,792** |

*Worker agent cost is held constant in both scenarios — the comparison isolates the Director model choice.*

**Haiku saves approximately $945/month (~34%) on total platform cost** in this model [unverified — figures are illustrative estimates].

### Non-token TCO factors

- **Latency compounding:** Haiku's 3–5× faster inference [unverified] reduces wall-clock time per orchestration cycle. In pipelines with 5–10 sequential Director hops, this can cut end-to-end latency by minutes per complex task.
- **Error/retry cost:** Haiku's lower per-turn cost means retry storms are cheaper to absorb. Sonnet errors are proportionally more expensive.
- **Context accumulation:** Long-running tasks that accumulate large context windows will see cost grow faster on Sonnet. Haiku's lower input rate partially offsets context bloat.

---

## Recommendation

**Run the Director on Claude 3.5 Haiku.** Director orchestration is routing-heavy and decision-shallow — it dispatches tasks, ranks proposals, and manages state. These operations do not require Sonnet's reasoning depth.

**Switch to Sonnet for the Director only when:**
1. The Director must resolve genuinely ambiguous task decomposition requiring multi-step inference.
2. The accumulated task-graph context exceeds ~50K tokens and accuracy under long context is critical.
3. The pipeline has fewer than ~500 Director turns/day, making the cost differential negligible relative to quality risk.

**Hybrid approach:** A tiered Director — Haiku for standard routing, Sonnet escalation for flagged ambiguous cases — captures most of the cost saving while preserving quality on edge cases. Escalation rate in practice is typically under 10% of turns [unverified], keeping blended cost close to the Haiku baseline.

---

*All pricing figures marked [unverified] should be confirmed against [https://www.anthropic.com/pricing](https://www.anthropic.com/pricing) before use in financial planning.*
