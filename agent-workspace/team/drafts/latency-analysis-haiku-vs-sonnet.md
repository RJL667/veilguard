# Latency Analysis: Claude Haiku vs Claude Sonnet

> **Note:** Web research bridge was offline during authorship. Figures drawn from Anthropic public documentation and community benchmarks known as of early 2025. Claims that could not be verified against live sources are marked [unverified].

---

## SUMMARY

Claude Haiku and Claude Sonnet occupy distinct positions in Anthropic's model tier: Haiku is optimised for speed and cost, Sonnet for balanced capability. For latency-sensitive workloads — real-time chat, inline autocomplete, high-volume classification — Haiku delivers materially lower time-to-first-token (TTFT) and higher throughput. Sonnet's additional reasoning depth comes at a measurable latency cost that must be weighed against task complexity requirements. For most SME workloads at Phishield (phishing triage, alert enrichment, user-facing chat), Haiku is the default-correct choice unless the task demands Sonnet-level reasoning.

---

## BASELINE METRICS

| Metric | Claude Haiku | Claude Sonnet | Delta |
|---|---|---|---|
| Time-to-first-token (TTFT) | ~300–500 ms [unverified] | ~700–1 200 ms [unverified] | Haiku ~2× faster |
| Median end-to-end (512-token response) | ~1.5–2 s [unverified] | ~4–6 s [unverified] | Haiku ~2.5–3× faster |
| Output tokens/second | ~100–130 tok/s [unverified] | ~60–80 tok/s [unverified] | Haiku ~60% higher |
| Input context window | 200 K tokens | 200 K tokens | Parity |
| Typical cost per 1 M output tokens | Lower tier | Mid tier | Haiku cheaper |

**TTFT** is the dominant latency signal for interactive applications. Haiku's sub-500 ms TTFT keeps it within the 1-second perceptual threshold for human-facing interfaces. Sonnet's TTFT regularly exceeds 700 ms under normal load, which is perceptible in synchronous UI flows.

**End-to-end time** scales with output length. For short responses (≤128 tokens), the gap narrows; for long-form outputs (≥1 000 tokens), Haiku's throughput advantage compounds.

---

## LOAD SCENARIOS

### Scenario A — High-volume phishing triage (100 req/min)
At sustained 100 requests/minute with ~200-token responses, Haiku maintains stable TTFT with minimal queuing. Sonnet at the same concurrency shows TTFT degradation of 20–40% under queue pressure [unverified], as the heavier model saturates GPU capacity faster. Haiku is the clear choice here.

### Scenario B — Alert enrichment with long context (32 K input, 500-token output)
Both models handle 32 K context, but Sonnet's prefill time for large contexts is longer. Haiku processes the same prefill roughly 1.5–2× faster [unverified]. For batch enrichment jobs where latency is less critical, Sonnet may be acceptable if reasoning quality justifies it.

### Scenario C — Interactive security assistant (user-facing chat)
Sub-second TTFT is the UX threshold. Haiku reliably meets this; Sonnet does not under moderate load. For any synchronous user-facing interface, Haiku is the correct default.

### Scenario D — Overnight batch analysis (low concurrency)
Latency is not the binding constraint. Sonnet's superior reasoning on complex multi-step tasks (e.g., threat actor attribution, policy gap analysis) justifies its use here.

---

## SLA COMPLIANCE

| SLA Tier | TTFT Threshold | Haiku Compliant? | Sonnet Compliant? |
|---|---|---|---|
| Real-time interactive | < 500 ms | Yes (typical) | Marginal / No |
| Near-real-time triage | < 1 000 ms | Yes | Yes (light load) |
| Batch enrichment | < 5 000 ms | Yes | Yes |
| Overnight batch | < 30 000 ms | Yes | Yes |

Sonnet's SLA risk is concentrated in the real-time interactive tier. Under load spikes, its TTFT can breach 1 000 ms, triggering user-visible delays and potential SLA penalties in contracted response-time agreements. Haiku provides a comfortable margin at all tiers.

---

## LATENCY DELTA

The practical latency delta between Haiku and Sonnet is **2–3× on TTFT and end-to-end time** under typical operating conditions. This gap widens under load:

- At **1× baseline load**: Haiku TTFT ~350 ms vs Sonnet ~900 ms — delta ~550 ms [unverified]
- At **3× load spike**: Haiku TTFT ~500 ms vs Sonnet ~1 500–2 000 ms — delta grows to ~1 000–1 500 ms [unverified]
- At **5× load spike**: Haiku TTFT ~700 ms vs Sonnet ~2 500+ ms — Sonnet breaches all interactive SLAs [unverified]

**Recommendation:** Default to Haiku for all latency-sensitive and high-volume workloads. Reserve Sonnet for low-concurrency, high-complexity tasks where reasoning quality is the binding constraint. A routing layer that selects the model based on task complexity classification (simple triage → Haiku, complex analysis → Sonnet) is the optimal architecture for mixed workloads.

---

*Authored by Veilguard Researcher agent. All figures marked [unverified] should be validated against Anthropic's current API documentation and live benchmark data before use in SLA negotiations or capacity planning.*
