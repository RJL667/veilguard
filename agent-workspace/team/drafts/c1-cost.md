# Claude Haiku vs Sonnet — Director Cost Analysis

| Model | Input ($/1M tok) | Output ($/1M tok) |
|-------|-----------------|------------------|
| Haiku 3.5 | $0.80 | $4.00 |
| Sonnet 3.5 | $3.00 | $15.00 |

**Cost ratio:** Sonnet is ~3.75× more expensive on input, ~3.75× on output.

**Director workloads** (task creation, proposal ranking, synthesis) are structured, low-ambiguity operations. Haiku delivers sufficient reasoning quality at roughly one-quarter the cost, making it the cost-efficient default for high-volume Director inference.
