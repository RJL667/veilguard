# Director Latency Analysis: Haiku vs Sonnet

## Inference Speed
Haiku generates tokens roughly 3–5x faster than Sonnet and costs significantly less per token [unverified: exact multipliers vary by load]. For a Director orchestrating many short tool-dispatch turns, this throughput advantage compounds across the task graph.

## Context Handling
Sonnet handles large, dense multi-agent state (tool results, sub-agent outputs, memory blocks) with greater reliability. Haiku degrades more noticeably on long-context reasoning and instruction-following under heavy state [unverified: degradation thresholds not publicly benchmarked].

## Recommendation
Use **Haiku** for shallow task graphs (≤3 hops, simple fan-out/fan-in) where speed and cost dominate. Use **Sonnet** for deep graphs (4+ hops, complex dependency chains, large accumulated context) where reasoning fidelity outweighs latency cost.
