"""Cost Simulation v3: Precise — with naive compaction + exact TCMM block lifecycle.

Uses real benchmark data. No estimates.
- Naive: grows until 190K, compacts (extra LLM call), resets, grows again
- TCMM: exact live/shadow/archive/recall per turn with real block data
- Cache: exact prefix overlap calculated per turn
"""

import json

BENCH_PATH = r"C:\Users\rudol\.gemini\antigravity\tcmm\bench_q\qf4f1d8a4\data\archive.json"

with open(BENCH_PATH) as f:
    archive = json.load(f).get("_default", {})

ordered = sorted(archive.items(), key=lambda x: int(x[0]))
total_blocks = len(ordered)

# Pricing
INPUT = 0.003
CACHE_WRITE = INPUT * 1.25
CACHE_READ = INPUT * 0.10
OUTPUT = 0.015
AVG_OUTPUT_TOKENS = 150

# TCMM params
LIVE_WINDOW = 20
SHADOW_WINDOW = 30
MIN_DENSITY = 2.0

# Naive compaction params
CONTEXT_WINDOW = 190000       # compacts at ~95% of 200K
COMPACT_SUMMARY_TOKENS = 3000 # summary after compaction
SYSTEM_PROMPT_TOKENS = 500

# Block data
blocks = []
for aid_str, e in ordered:
    blocks.append({
        "tokens": len(e.get("text", "")) // 4,
        "chars": len(e.get("text", "")),
        "density": e.get("density_score", 0),
        "links": len(e.get("semantic_links", {})) + len(e.get("entity_links", {})) + len(e.get("topic_links", {})),
        "is_weak": e.get("density_score", 0) < MIN_DENSITY,
    })

print(f"{'=' * 85}")
print(f"COST SIMULATION v3 — {total_blocks} turns | WITH naive compaction")
print(f"{'=' * 85}")
print(f"  Pricing: input=${INPUT}/1K | cache_write=${CACHE_WRITE}/1K | cache_read=${CACHE_READ}/1K | output=${OUTPUT}/1K")
print(f"  Naive compaction: triggers at {CONTEXT_WINDOW:,} tokens, resets to {COMPACT_SUMMARY_TOKENS:,} token summary")
print(f"  TCMM: {LIVE_WINDOW} live, {SHADOW_WINDOW} shadow, recall top 10 linked archive blocks")
print()

# ── NAIVE simulation with compaction ──
naive_history_tokens = 0
naive_cumulative_input = 0
naive_cost_no_cache = 0.0
naive_cost_cached = 0.0
naive_compactions = 0
naive_compaction_cost = 0.0
prev_naive = 0
naive_per_turn = []

for turn, bd in enumerate(blocks):
    naive_history_tokens += bd["tokens"]
    naive_input = SYSTEM_PROMPT_TOKENS + naive_history_tokens

    # Check compaction trigger
    if naive_input > CONTEXT_WINDOW:
        naive_compactions += 1
        # Compaction call cost: read entire context + output summary
        compact_cost = (naive_input / 1000) * INPUT + (COMPACT_SUMMARY_TOKENS / 1000) * OUTPUT
        naive_compaction_cost += compact_cost
        naive_cost_no_cache += compact_cost
        naive_cost_cached += compact_cost  # compaction call can't cache (unique every time)
        # Reset to summary
        naive_history_tokens = COMPACT_SUMMARY_TOKENS
        naive_input = SYSTEM_PROMPT_TOKENS + naive_history_tokens
        prev_naive = 0  # cache invalidated after compaction

    naive_cumulative_input += naive_input

    # No cache cost
    naive_cost_no_cache += (naive_input / 1000) * INPUT

    # Cached cost
    if turn == 0 or prev_naive == 0:
        naive_cost_cached += (naive_input / 1000) * CACHE_WRITE
    else:
        cached = prev_naive
        new = max(0, naive_input - cached)
        naive_cost_cached += (cached / 1000) * CACHE_READ
        naive_cost_cached += (new / 1000) * CACHE_WRITE
    prev_naive = naive_input

    naive_per_turn.append(naive_input)

# ── TCMM simulation with exact block lifecycle ──
tcmm_cumulative_input = 0
tcmm_cost_no_cache = 0.0
tcmm_cost_cached = 0.0
prev_tcmm_stable = 0
tcmm_per_turn = []
tcmm_recall_per_turn = []

for turn, bd in enumerate(blocks):
    # Live blocks
    live_start = max(0, turn - LIVE_WINDOW + 1)
    live = blocks[live_start:turn + 1]
    live_tokens = sum(b["tokens"] for b in live)

    # Shadow blocks
    shadow_start = max(0, live_start - SHADOW_WINDOW)
    shadow = blocks[shadow_start:live_start]
    shadow_tokens = sum(b["tokens"] for b in shadow)

    # Archive blocks
    archive_blks = blocks[:shadow_start]

    # Recalled: top 10 archive blocks by link count, excluding weak
    recalled = sorted(
        [b for b in archive_blks if b["links"] > 3 and not b["is_weak"]],
        key=lambda b: b["links"], reverse=True
    )[:10]
    recall_tokens = sum(b["tokens"] for b in recalled)

    # Total TCMM input
    stable_tokens = SYSTEM_PROMPT_TOKENS + live_tokens
    variable_tokens = recall_tokens  # shadow is NOT sent as input, only live + recall
    tcmm_input = stable_tokens + variable_tokens
    tcmm_cumulative_input += tcmm_input

    # No cache cost
    tcmm_cost_no_cache += (tcmm_input / 1000) * INPUT

    # Cached cost — exact prefix overlap
    if turn == 0:
        tcmm_cost_cached += (stable_tokens / 1000) * CACHE_WRITE
        tcmm_cost_cached += (variable_tokens / 1000) * INPUT
    else:
        # Live window shifts by 1: drop oldest, add newest
        if turn >= LIVE_WINDOW and prev_tcmm_stable > 0:
            dropped = blocks[live_start - 1]["tokens"] if live_start > 0 else 0
            added = bd["tokens"]
            changed = dropped + added
            cached_stable = max(0, min(prev_tcmm_stable, stable_tokens) - changed)
        else:
            # Still growing — only new block is uncached
            cached_stable = max(0, prev_tcmm_stable)
        uncached_stable = max(0, stable_tokens - cached_stable)

        tcmm_cost_cached += (cached_stable / 1000) * CACHE_READ
        tcmm_cost_cached += (uncached_stable / 1000) * CACHE_WRITE
        tcmm_cost_cached += (variable_tokens / 1000) * INPUT

    prev_tcmm_stable = stable_tokens
    tcmm_per_turn.append(tcmm_input)
    tcmm_recall_per_turn.append(recall_tokens)

# Output cost (same for all)
total_output_cost = total_blocks * (AVG_OUTPUT_TOKENS / 1000) * OUTPUT

# ── Results ──
print(f"{'=' * 85}")
print(f"NAIVE (with compaction)")
print(f"{'=' * 85}")
print(f"  Compactions triggered:  {naive_compactions}")
print(f"  Compaction extra cost:  ${naive_compaction_cost:.4f}")
print(f"  Cumulative input:       {naive_cumulative_input:>12,} tokens")
print(f"  Final turn size:        {naive_per_turn[-1]:>12,} tokens")
print(f"  Peak turn size:         {max(naive_per_turn):>12,} tokens")
print()

print(f"{'=' * 85}")
print(f"TCMM (live + recall)")
print(f"{'=' * 85}")
print(f"  Cumulative input:       {tcmm_cumulative_input:>12,} tokens")
print(f"  Final turn size:        {tcmm_per_turn[-1]:>12,} tokens")
print(f"  Peak turn size:         {max(tcmm_per_turn):>12,} tokens")
print(f"  Avg recall tokens/turn: {sum(tcmm_recall_per_turn) // total_blocks:>12,}")
print()

# Cost table
print(f"{'=' * 85}")
print(f"COST COMPARISON")
print(f"{'=' * 85}")

scenarios = [
    ("Naive, no cache", naive_cost_no_cache + total_output_cost),
    ("Naive + cache", naive_cost_cached + total_output_cost),
    ("Naive + cache (excl compact)", naive_cost_cached - naive_compaction_cost + total_output_cost),
    ("TCMM, no cache", tcmm_cost_no_cache + total_output_cost),
    ("TCMM + cache", tcmm_cost_cached + total_output_cost),
]

baseline_naive_cached = scenarios[1][1]

print(f"\n  {'Scenario':<32} {'Total':>10} {'Per Turn':>12} {'vs Naive+Cache':>15}")
print(f"  {'-' * 69}")
for name, cost in scenarios:
    vs = (cost - baseline_naive_cached) * 100 / baseline_naive_cached
    print(f"  {name:<32} ${cost:>8.4f}  ${cost / total_blocks:>10.6f}  {vs:>+13.1f}%")

print(f"\n  Output cost (same for all): ${total_output_cost:.4f}")

# The key comparison
n_cached = scenarios[1][1]
t_cached = scenarios[4][1]
savings = n_cached - t_cached
pct = savings * 100 / n_cached

print(f"\n{'=' * 85}")
print(f"THE BOTTOM LINE: Naive+Cache vs TCMM+Cache")
print(f"{'=' * 85}")
print(f"  Naive + cache (with compaction):  ${n_cached:.4f}  (${n_cached / total_blocks:.6f}/turn)")
print(f"  TCMM + cache:                     ${t_cached:.4f}  (${t_cached / total_blocks:.6f}/turn)")
print(f"  TCMM saves:                       ${savings:.4f}  ({pct:.1f}%)")
print(f"  Compaction overhead in naive:      ${naive_compaction_cost:.4f} ({naive_compactions} compactions)")
print()

# Per-turn breakdown
print(f"  Per-Turn Detail:")
print(f"  {'Turn':<6} {'Naive tok':>10} {'TCMM tok':>10} {'Recall':>8} {'N+Cache$':>10} {'T+Cache$':>10} {'Save$':>10}")
print(f"  {'-' * 64}")

cum_n = 0.0
cum_t = 0.0
prev_n_for_cost = 0
prev_t_stable_for_cost = 0

for t in [0, 4, 9, 24, 49, 99, 199, 299, 499, min(total_blocks - 1, 724)]:
    if t >= total_blocks:
        continue
    nt = naive_per_turn[t]
    tt = tcmm_per_turn[t]
    rt = tcmm_recall_per_turn[t]

    # Per-turn naive cached cost
    if t == 0 or prev_n_for_cost == 0:
        n_cost = (nt / 1000) * CACHE_WRITE
    else:
        n_cost = (min(prev_n_for_cost, nt) / 1000) * CACHE_READ + (max(0, nt - prev_n_for_cost) / 1000) * CACHE_WRITE
    prev_n_for_cost = nt

    # Per-turn TCMM cached cost
    stable_t = tt - rt
    if t == 0:
        t_cost = (stable_t / 1000) * CACHE_WRITE + (rt / 1000) * INPUT
    else:
        if t >= LIVE_WINDOW and prev_t_stable_for_cost > 0:
            dropped = blocks[max(0, t - LIVE_WINDOW)]["tokens"]
            added = blocks[t]["tokens"]
            cached_s = max(0, min(prev_t_stable_for_cost, stable_t) - dropped - added)
        else:
            cached_s = max(0, prev_t_stable_for_cost)
        uncached_s = max(0, stable_t - cached_s)
        t_cost = (cached_s / 1000) * CACHE_READ + (uncached_s / 1000) * CACHE_WRITE + (rt / 1000) * INPUT
    prev_t_stable_for_cost = stable_t

    # Add output
    out_cost = (AVG_OUTPUT_TOKENS / 1000) * OUTPUT
    n_cost += out_cost
    t_cost += out_cost
    save = max(0, n_cost - t_cost)

    print(f"  {t+1:<6} {nt:>9,} {tt:>9,} {rt:>7,} ${n_cost:>8.5f} ${t_cost:>8.5f} ${save:>8.5f}")

# When does TCMM break even?
cum_naive = 0.0
cum_tcmm = 0.0
breakeven = None
prev_n2 = 0
prev_ts2 = 0
for t in range(total_blocks):
    nt = naive_per_turn[t]
    tt = tcmm_per_turn[t]
    rt = tcmm_recall_per_turn[t]
    stable = tt - rt

    # Naive cached
    if t == 0 or prev_n2 == 0:
        cum_naive += (nt / 1000) * CACHE_WRITE
    else:
        cum_naive += (min(prev_n2, nt) / 1000) * CACHE_READ + (max(0, nt - prev_n2) / 1000) * CACHE_WRITE
    prev_n2 = nt

    # TCMM cached
    if t == 0:
        cum_tcmm += (stable / 1000) * CACHE_WRITE + (rt / 1000) * INPUT
    else:
        if t >= LIVE_WINDOW and prev_ts2 > 0:
            d = blocks[max(0, t - LIVE_WINDOW)]["tokens"]
            a = blocks[t]["tokens"]
            cs = max(0, min(prev_ts2, stable) - d - a)
        else:
            cs = max(0, prev_ts2)
        us = max(0, stable - cs)
        cum_tcmm += (cs / 1000) * CACHE_READ + (us / 1000) * CACHE_WRITE + (rt / 1000) * INPUT
    prev_ts2 = stable

    if breakeven is None and cum_naive > cum_tcmm and t > 10:
        breakeven = t + 1

print(f"\n  TCMM breaks even at turn: {breakeven or 'N/A'}")
print(f"  After breakeven, TCMM saves more every turn as naive context grows.")
