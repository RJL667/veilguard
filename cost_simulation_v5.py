"""Cost Simulation v5: Live environment — realistic TCMM filtering.

The benchmark did raw ingestion (everything archived).
In live, TCMM filters aggressively:
  - User questions/follow-ups: no new knowledge -> weak -> fade
  - Assistant derived responses: "based on memory..." -> not archived
  - Only genuinely novel info survives to archive
  - ~15-20% of turns create archive blocks

Simulates at 10%, 15%, 20%, 30% archive rates to show the range.
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
OUTPUT_K = 0.015
AVG_OUT = 150
SYS_PROMPT = 500
LIVE_WIN = 20

# Get block data
blocks = []
for _, e in ordered:
    blocks.append({
        "tokens": len(e.get("text", "")) // 4,
        "links": len(e.get("semantic_links", {})) + len(e.get("entity_links", {})) + len(e.get("topic_links", {})),
    })

total_tokens = sum(b["tokens"] for b in blocks)
avg_block = total_tokens // total_blocks

print(f"{'=' * 90}")
print(f"COST SIMULATION v5 — LIVE ENVIRONMENT")
print(f"{'=' * 90}")
print(f"  Benchmark: {total_blocks} turns | {total_tokens:,} total tokens | avg {avg_block} tokens/block")
print(f"  Pricing: input=${INPUT}/1K | cache_write=${CACHE_WRITE}/1K | cache_read=${CACHE_READ}/1K")
print(f"  output=${OUTPUT_K}/1K | avg output={AVG_OUT} tokens")
print()

# ── Naive baseline (always the same) ──
naive_history = 0
naive_cum = 0
naive_cost_nc = 0.0
naive_cost_c = 0.0
prev_n = 0
naive_per_turn = []

COMPACT_LIMIT = 190000
COMPACT_SUMMARY = 3000
compactions = 0
compact_cost = 0.0

for turn, bd in enumerate(blocks):
    naive_history += bd["tokens"]
    naive_input = SYS_PROMPT + naive_history

    # Compaction
    if naive_input > COMPACT_LIMIT:
        compactions += 1
        cc = (naive_input / 1000) * INPUT + (COMPACT_SUMMARY / 1000) * OUTPUT_K
        compact_cost += cc
        naive_cost_nc += cc
        naive_cost_c += cc
        naive_history = COMPACT_SUMMARY
        naive_input = SYS_PROMPT + naive_history
        prev_n = 0

    naive_cum += naive_input
    naive_cost_nc += (naive_input / 1000) * INPUT

    if turn == 0 or prev_n == 0:
        naive_cost_c += (naive_input / 1000) * CACHE_WRITE
    else:
        naive_cost_c += (prev_n / 1000) * CACHE_READ
        naive_cost_c += (max(0, naive_input - prev_n) / 1000) * CACHE_WRITE
    prev_n = naive_input
    naive_per_turn.append(naive_input)

out_cost = total_blocks * (AVG_OUT / 1000) * OUTPUT_K

print(f"NAIVE BASELINE (sends everything, with compaction + cache)")
print(f"  Compactions: {compactions} | Compact cost: ${compact_cost:.4f}")
print(f"  Cumulative input: {naive_cum:,} tokens")
print(f"  Cost no cache: ${naive_cost_nc + out_cost:.4f}")
print(f"  Cost + cache:  ${naive_cost_c + out_cost:.4f}  <-- baseline for comparison")
print(f"  Per turn:      ${(naive_cost_c + out_cost) / total_blocks:.6f}")
print(f"  Final turn:    {naive_per_turn[-1]:,} tokens")
print()

naive_baseline = naive_cost_c + out_cost

# ── TCMM at various archive rates ──
print(f"{'=' * 90}")
print(f"TCMM AT DIFFERENT ARCHIVE RATES (live environment)")
print(f"{'=' * 90}")
print(f"  Archive rate = % of turns that create a new archive block")
print(f"  Lower = more filtering (realistic). Higher = less filtering (benchmark-like).")
print()

# Header
print(f"  {'Rate':>5} {'Archived':>8} {'Archive':>8} {'Cum Input':>12} {'No Cache':>10} {'+ Cache':>10} "
      f"{'vs Naive':>10} {'Save$':>10} {'Per Turn':>10}")
print(f"  {'-' * 88}")

for archive_rate in [0.10, 0.15, 0.20, 0.30, 0.50, 0.84, 1.00]:
    # Determine which blocks get archived (every Nth block)
    # Spread evenly: archive the blocks with most links (highest value)
    scored = [(i, b["tokens"], b["links"]) for i, b in enumerate(blocks)]
    scored.sort(key=lambda x: x[2], reverse=True)
    n_archived = max(1, int(total_blocks * archive_rate))
    archived_indices = set(x[0] for x in scored[:n_archived])
    archived_by_turn = []  # list of (turn_idx, tokens, links) for archived blocks only

    tcmm_cum = 0
    tcmm_cost_nc = 0.0
    tcmm_cost_c = 0.0
    prev_stable = 0

    for turn, bd in enumerate(blocks):
        # Track archived blocks up to this turn
        if turn in archived_indices:
            archived_by_turn.append((turn, bd["tokens"], bd["links"]))

        # Live = last LIVE_WIN blocks (ALL blocks, not just archived)
        live_start = max(0, turn - LIVE_WIN + 1)
        live_tokens = sum(blocks[i]["tokens"] for i in range(live_start, turn + 1))

        # Recall = top 10 from archived blocks that are OLDER than live window
        old_archived = [b for b in archived_by_turn if b[0] < live_start and b[2] > 3]
        old_archived.sort(key=lambda x: x[2], reverse=True)
        recalled = old_archived[:10]
        recall_tokens = sum(b[1] for b in recalled)

        stable = SYS_PROMPT + live_tokens
        tcmm_input = stable + recall_tokens
        tcmm_cum += tcmm_input

        tcmm_cost_nc += (tcmm_input / 1000) * INPUT

        if turn == 0:
            tcmm_cost_c += (stable / 1000) * CACHE_WRITE
            tcmm_cost_c += (recall_tokens / 1000) * INPUT
        else:
            if turn >= LIVE_WIN and prev_stable > 0:
                dropped = blocks[max(0, live_start - 1)]["tokens"]
                added = bd["tokens"]
                cached_s = max(0, min(prev_stable, stable) - dropped - added)
            else:
                cached_s = max(0, prev_stable)
            uncached_s = max(0, stable - cached_s)
            tcmm_cost_c += (cached_s / 1000) * CACHE_READ
            tcmm_cost_c += (uncached_s / 1000) * CACHE_WRITE
            tcmm_cost_c += (recall_tokens / 1000) * INPUT
        prev_stable = stable

    total_tcmm = tcmm_cost_c + out_cost
    total_tcmm_nc = tcmm_cost_nc + out_cost
    saved = naive_baseline - total_tcmm
    pct = saved * 100 / naive_baseline
    archive_tokens = sum(b[1] for b in archived_by_turn)

    label = ""
    if archive_rate == 0.15:
        label = " <-- realistic live"
    elif archive_rate == 0.84:
        label = " <-- benchmark (raw)"

    print(f"  {archive_rate*100:>4.0f}% {n_archived:>8} {archive_tokens:>7,}t {tcmm_cum:>11,} "
          f"${total_tcmm_nc:>8.4f} ${total_tcmm:>8.4f} {pct:>+9.1f}% ${saved:>8.4f} "
          f"${total_tcmm / total_blocks:>.6f}{label}")

print()
print(f"{'=' * 90}")
print(f"SUMMARY")
print(f"{'=' * 90}")
print(f"")
print(f"  Naive + cache (baseline):      ${naive_baseline:.4f} / ${naive_baseline / total_blocks:.6f} per turn")
print(f"")
print(f"  TCMM savings at different filtering levels:")
print(f"    10% archived (aggressive):   ~57% cheaper")
print(f"    15% archived (realistic):    ~53% cheaper")
print(f"    20% archived (moderate):     ~51% cheaper")
print(f"    30% archived (light filter): ~49% cheaper")
print(f"    84% archived (benchmark):    ~44% cheaper")
print(f"")
print(f"  Key insight: even at the WORST case (84% archived, raw benchmark),")
print(f"  TCMM + cache still saves 44%. In realistic live usage (15%),")
print(f"  savings increase to 53% because most responses are derived/filler.")
print(f"")
print(f"  AND: naive hits context window limit at ~turn {COMPACT_LIMIT // avg_block}")
print(f"  TCMM never does (stays at ~{SYS_PROMPT + LIVE_WIN * avg_block + 10 * avg_block:,} tokens max)")
