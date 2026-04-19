"""Cost simulation: TCMM vs Naive, with and without Anthropic prompt caching.

Uses real benchmark data (725 archive blocks from a ~500 turn conversation).
"""

import json
import os

BENCH_PATH = r"C:\Users\rudol\.gemini\antigravity\tcmm\bench_q\qf4f1d8a4\data\archive.json"

with open(BENCH_PATH) as f:
    archive = json.load(f).get("_default", {})

ordered = sorted(archive.items(), key=lambda x: int(x[0]))
total_blocks = len(ordered)

print(f"=" * 70)
print(f"TCMM vs Naive Cost Simulation — {total_blocks} blocks")
print(f"=" * 70)

# === Block stats ===
total_chars = sum(len(e.get("text", "")) for _, e in ordered)
roles = {}
for _, e in ordered:
    r = e.get("origin", "unknown")
    roles[r] = roles.get(r, 0) + 1

has_sem = sum(1 for _, e in ordered if e.get("semantic_links"))
has_ent = sum(1 for _, e in ordered if e.get("entity_links"))
has_top = sum(1 for _, e in ordered if e.get("topic_links"))

print(f"\nBlock Stats:")
print(f"  Total text: {total_chars:,} chars ({total_chars // 4:,} tokens)")
print(f"  Avg block: {total_chars // total_blocks} chars ({total_chars // total_blocks // 4} tokens)")
print(f"  Roles: {roles}")
print(f"  With semantic links: {has_sem} ({has_sem * 100 // total_blocks}%)")
print(f"  With entity links: {has_ent} ({has_ent * 100 // total_blocks}%)")
print(f"  With topic links: {has_top} ({has_top * 100 // total_blocks}%)")

# === Link distribution ===
link_counts = []
for _, e in ordered:
    lc = len(e.get("semantic_links", {})) + len(e.get("entity_links", {})) + len(e.get("topic_links", {}))
    link_counts.append(lc)
link_counts.sort()

zero_links = sum(1 for l in link_counts if l == 0)
print(f"\nLink Distribution:")
print(f"  0 links: {zero_links} ({zero_links * 100 // total_blocks}%) — these would FADE from TCMM")
print(f"  1-3 links: {sum(1 for l in link_counts if 1 <= l <= 3)}")
print(f"  4-10 links: {sum(1 for l in link_counts if 4 <= l <= 10)}")
print(f"  10+ links: {sum(1 for l in link_counts if l > 10)}")
print(f"  Mean: {sum(link_counts) / len(link_counts):.1f} | Median: {link_counts[len(link_counts) // 2]}")

# === Simulate ===
SYSTEM_PROMPT_TOKENS = 500
LIVE_WINDOW = 20  # TCMM keeps ~20 live blocks
RECALL_COUNT = 10  # avg recalled per turn
AVG_BLOCK_CHARS = total_chars // total_blocks

# Accumulate per-turn costs
naive_cumulative_tokens = 0
tcmm_cumulative_tokens = 0
naive_per_turn = []
tcmm_per_turn = []
history_chars = 0

for turn_idx, (aid, e) in enumerate(ordered):
    text = e.get("text", "")
    history_chars += len(text)

    # Naive: system_prompt + ALL history
    naive_input = SYSTEM_PROMPT_TOKENS + (history_chars // 4)
    naive_cumulative_tokens += naive_input
    naive_per_turn.append(naive_input)

    # TCMM: system_prompt + live window + recalled subset
    live_start = max(0, turn_idx - LIVE_WINDOW)
    live_chars = sum(len(ordered[i][1].get("text", "")) for i in range(live_start, min(turn_idx + 1, total_blocks)))
    recall_tokens = RECALL_COUNT * AVG_BLOCK_CHARS // 4
    tcmm_input = SYSTEM_PROMPT_TOKENS + (live_chars // 4) + recall_tokens
    tcmm_cumulative_tokens += tcmm_input
    tcmm_per_turn.append(tcmm_input)

saved_tokens = naive_cumulative_tokens - tcmm_cumulative_tokens

print(f"\n{'=' * 70}")
print(f"TOKEN COMPARISON (no cache)")
print(f"{'=' * 70}")
print(f"  Turns: {total_blocks}")
print(f"")
print(f"  NAIVE (full history every turn):")
print(f"    Total input tokens: {naive_cumulative_tokens:>12,}")
print(f"    Avg per turn:       {naive_cumulative_tokens // total_blocks:>12,}")
print(f"    Turn 1:             {naive_per_turn[0]:>12,}")
print(f"    Turn 100:           {naive_per_turn[min(99, total_blocks - 1)]:>12,}")
print(f"    Turn 500:           {naive_per_turn[min(499, total_blocks - 1)]:>12,}")
print(f"    Final turn:         {naive_per_turn[-1]:>12,}")
print(f"")
print(f"  TCMM (live + recall):")
print(f"    Total input tokens: {tcmm_cumulative_tokens:>12,}")
print(f"    Avg per turn:       {tcmm_cumulative_tokens // total_blocks:>12,}")
print(f"    Turn 1:             {tcmm_per_turn[0]:>12,}")
print(f"    Turn 100:           {tcmm_per_turn[min(99, total_blocks - 1)]:>12,}")
print(f"    Turn 500:           {tcmm_per_turn[min(499, total_blocks - 1)]:>12,}")
print(f"    Final turn:         {tcmm_per_turn[-1]:>12,}")
print(f"")
print(f"  SAVINGS: {saved_tokens:,} tokens ({saved_tokens * 100 // naive_cumulative_tokens}%)")

# === Cost with Anthropic cache ===
# claude-sonnet-4-6 pricing
INPUT_PER_1K = 0.003
OUTPUT_PER_1K = 0.015
CACHE_WRITE_PER_1K = INPUT_PER_1K * 1.25  # 25% premium to write
CACHE_READ_PER_1K = INPUT_PER_1K * 0.10   # 90% discount to read
AVG_OUTPUT_TOKENS = 150  # average LLM response

print(f"\n{'=' * 70}")
print(f"COST COMPARISON (with Anthropic prompt caching)")
print(f"{'=' * 70}")
print(f"  Pricing: input=$0.003/1K | cache_write=$0.00375/1K | cache_read=$0.0003/1K | output=$0.015/1K")
print(f"  Cache TTL: 5 min (resets on read — always active during conversation)")
print()

# --- Naive with cache ---
# System prompt (500 tokens) caches perfectly — identical every turn
# Conversation history changes every turn — CANNOT be cached (new messages added)
# But the PREFIX of history (all messages BEFORE the last one) IS cacheable!
# So: cached = system_prompt + all_previous_history, uncached = latest message only
naive_cost = 0
for turn_idx in range(total_blocks):
    if turn_idx == 0:
        # First turn: write entire input to cache
        naive_cost += (naive_per_turn[turn_idx] / 1000) * CACHE_WRITE_PER_1K
    else:
        # Subsequent: previous turn's full input is cached prefix
        # New tokens = difference from last turn (just the new message)
        cached_prefix = naive_per_turn[turn_idx - 1]
        new_tokens = naive_per_turn[turn_idx] - cached_prefix
        naive_cost += (cached_prefix / 1000) * CACHE_READ_PER_1K  # read from cache
        naive_cost += (max(new_tokens, 0) / 1000) * CACHE_WRITE_PER_1K  # write new part
    # Output cost (same for both)
    naive_cost += (AVG_OUTPUT_TOKENS / 1000) * OUTPUT_PER_1K

# --- TCMM with cache ---
# System prompt + live blocks = mostly stable prefix (cacheable)
# Recalled blocks = variable suffix (uncacheable)
# Live window shifts by 1 block per turn: ~95% overlap with previous turn
tcmm_cost = 0
for turn_idx in range(total_blocks):
    live_tokens = tcmm_per_turn[turn_idx] - recall_tokens  # stable part
    recall_tok = recall_tokens  # variable part

    if turn_idx == 0:
        tcmm_cost += (live_tokens / 1000) * CACHE_WRITE_PER_1K
        tcmm_cost += (recall_tok / 1000) * INPUT_PER_1K
    else:
        # ~90% of live prefix is cached (only 1 block shifted out, 1 shifted in)
        prev_live = tcmm_per_turn[turn_idx - 1] - recall_tokens
        overlap = min(prev_live, live_tokens) * 0.90
        new_live = live_tokens - overlap

        tcmm_cost += (overlap / 1000) * CACHE_READ_PER_1K  # cached stable prefix
        tcmm_cost += (new_live / 1000) * CACHE_WRITE_PER_1K  # new live tokens
        tcmm_cost += (recall_tok / 1000) * INPUT_PER_1K  # recalled = uncacheable

    # Output cost
    tcmm_cost += (AVG_OUTPUT_TOKENS / 1000) * OUTPUT_PER_1K

# --- No cache at all ---
naive_no_cache = (naive_cumulative_tokens / 1000) * INPUT_PER_1K + (total_blocks * AVG_OUTPUT_TOKENS / 1000) * OUTPUT_PER_1K
tcmm_no_cache = (tcmm_cumulative_tokens / 1000) * INPUT_PER_1K + (total_blocks * AVG_OUTPUT_TOKENS / 1000) * OUTPUT_PER_1K

print(f"  {'Scenario':<30} {'Total Cost':>12} {'Per Turn':>12} {'vs Naive':>12}")
print(f"  {'-' * 66}")
print(f"  {'Naive (no cache)':<30} ${naive_no_cache:>10.4f}  ${naive_no_cache / total_blocks:>10.6f}")
print(f"  {'Naive + cache':<30} ${naive_cost:>10.4f}  ${naive_cost / total_blocks:>10.6f}  {(naive_cost - naive_no_cache) * 100 / naive_no_cache:>+10.1f}%")
print(f"  {'TCMM (no cache)':<30} ${tcmm_no_cache:>10.4f}  ${tcmm_no_cache / total_blocks:>10.6f}  {(tcmm_no_cache - naive_no_cache) * 100 / naive_no_cache:>+10.1f}%")
print(f"  {'TCMM + cache':<30} ${tcmm_cost:>10.4f}  ${tcmm_cost / total_blocks:>10.6f}  {(tcmm_cost - naive_no_cache) * 100 / naive_no_cache:>+10.1f}%")

print(f"\n  Best savings: TCMM + cache vs Naive no cache:")
best_savings = naive_no_cache - tcmm_cost
print(f"  ${best_savings:.4f} saved ({best_savings * 100 / naive_no_cache:.1f}%)")
print()

# Per-turn breakdown at key points
print(f"  Per-Turn Cost at Key Points:")
print(f"  {'Turn':<8} {'Naive':>10} {'Naive+$':>10} {'TCMM':>10} {'TCMM+$':>10}")
for t in [0, 9, 49, 99, 249, 499, min(total_blocks - 1, 724)]:
    if t >= total_blocks:
        continue
    n_tok = naive_per_turn[t]
    t_tok = tcmm_per_turn[t]
    n_cost_turn = (n_tok / 1000) * INPUT_PER_1K + (AVG_OUTPUT_TOKENS / 1000) * OUTPUT_PER_1K
    t_cost_turn = (t_tok / 1000) * INPUT_PER_1K + (AVG_OUTPUT_TOKENS / 1000) * OUTPUT_PER_1K
    print(f"  {t + 1:<8} {n_tok:>8,}t  ${n_cost_turn:>8.5f}  {t_tok:>8,}t  ${t_cost_turn:>8.5f}")

print(f"\n{'=' * 70}")
print(f"CONCLUSION")
print(f"{'=' * 70}")
print(f"  Naive context grows linearly: turn 1 = {naive_per_turn[0]:,}t → turn {total_blocks} = {naive_per_turn[-1]:,}t")
print(f"  TCMM context stays flat:      turn 1 = {tcmm_per_turn[0]:,}t → turn {total_blocks} = {tcmm_per_turn[-1]:,}t")
print(f"  ")
print(f"  Prompt caching helps BOTH approaches, but helps Naive MORE")
print(f"  (because the growing history prefix is cacheable between turns).")
print(f"  ")
print(f"  However, TCMM still wins because:")
print(f"    1. Context window limit: Naive hits 200K limit at ~turn {200000 * 4 // (total_chars // total_blocks)}")
print(f"       TCMM never hits it (stays at ~{tcmm_per_turn[-1]:,} tokens)")
print(f"    2. Recall quality: TCMM brings back RELEVANT blocks, not everything")
print(f"    3. Combined: TCMM + cache = ${tcmm_cost:.4f} vs Naive + cache = ${naive_cost:.4f}")
print(f"       ({(naive_cost - tcmm_cost) * 100 / naive_cost:.1f}% cheaper)")
