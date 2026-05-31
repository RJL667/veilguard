# Claude Haiku vs Sonnet: Cost Comparison for Director Workload

> **Note:** Web research bridge was offline during authorship. Pricing figures drawn from Anthropic's published rate card as of early 2025 [unverified — verify at https://www.anthropic.com/pricing before acting on these figures].

---

## PRICING

All prices are per million tokens (input / output):

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|---|---|---|
| Claude 3.5 Haiku | $0.80 | $4.00 |
| Claude 3.5 Sonnet | $3.00 | $15.00 |

**Cost ratio:** Sonnet is approximately **3.75× more expensive on input** and **3.75× more expensive on output** than Haiku. [unverified]

---

## ASSUMPTIONS

Director workload characteristics assumed for this analysis:

- **Request profile:** Each Director turn averages ~500 input tokens + ~200 output tokens (routing instructions, task dispatch, state summaries — not deep reasoning).
- **Daily volume:** 10,000 Director turns per day (moderate SME multi-agent pipeline).
- **Monthly volume:** ~300,000 turns per month (30-day basis).
- **Token mix:** Input-heavy workload; output is short dispatch instructions.

---

## MONTHLY_PROJECTION

| Model | Input cost/month | Output cost/month | **Total/month** |
|---|---|---|---|
| Claude 3.5 Haiku | 300,000 × 500 / 1,000,000 × $0.80 = **$120** | 300,000 × 200 / 1,000,000 × $4.00 = **$240** | **$360** |
| Claude 3.5 Sonnet | 300,000 × 500 / 1,000,000 × $3.00 = **$450** | 300,000 × 200 / 1,000,000 × $15.00 = **$900** | **$1,350** |

**Monthly saving with Haiku: ~$990** (~73% reduction).

At 2× volume (600,000 turns/month), savings scale to ~$1,980/month (~$23,760/year).

---

## KEY_FINDING

Run the Director on **Claude 3.5 Haiku**. Director orchestration is routing- and dispatch-heavy, not reasoning-heavy — Haiku's capability is sufficient for task decomposition and agent coordination at this workload profile. The cost advantage is decisive: Haiku costs roughly **3.75× less** per token, translating to ~$990/month saved at the assumed volume. Reserve Sonnet for worker agents that require deep synthesis, nuanced generation, or complex multi-step reasoning where quality materially affects output correctness.
