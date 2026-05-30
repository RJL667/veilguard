# Director Cost Analysis: Haiku vs Sonnet

## Pricing
| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|---|---|---|
| Claude 3.5 Haiku | $0.80 | $4.00 |
| Claude 3.5 Sonnet | $3.00 | $15.00 |
*[unverified — research bridge offline]*

## Typical Turn Cost
A Director turn (~500 input / 200 output tokens): Haiku ~$0.0005 vs Sonnet ~$0.0045 — roughly **9× cheaper**.

## Projection
At 10,000 Director turns/day: Haiku ~$5/day ($150/mo) vs Sonnet ~$45/day ($1,350/mo). [unverified]

## Recommendation
Run the Director on Haiku. Orchestration is routing-heavy, not reasoning-heavy; Haiku's 9× cost advantage is decisive. Reserve Sonnet for worker agents requiring deep synthesis.
