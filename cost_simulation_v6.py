"""Cost Simulation v6: CORRECT model.

NAIVE: Every turn = user_msg + assistant_response added to context. BOTH.
  Context grows by ~(user_tokens + assistant_tokens) per turn.

TCMM: Every turn:
  - User msg -> always enters live blocks
  - Assistant response -> LLM classifies:
    - "derived" (80% in live) -> DISCARDED. Never enters live/shadow/archive.
    - "novel" (20% in live) -> enters live blocks
  - So TCMM live window grows SLOWER (only user + novel assistant blocks)
  - Live window of 20 blocks covers MORE turns
  - Recall from archive: only novel blocks with links

The 80% derived rate means:
  - Per turn: naive adds ~(user + assistant) = ~420 tokens
  - Per turn: TCMM adds ~(user + 20% * assistant) = ~220 tokens to live
  - After 20 blocks, TCMM live window covers ~30+ turns vs naive's 20
"""

import json

BENCH_PATH = r"C:\Users\rudol\.gemini\antigravity\tcmm\bench_q\qf4f1d8a4\data\archive.json"

with open(BENCH_PATH) as f:
    archive = json.load(f).get("_default", {})

ordered = sorted(archive.items(), key=lambda x: int(x[0]))

# Separate user and assistant blocks
user_blocks = [(i, len(e.get("text", "")) // 4) for i, (_, e) in enumerate(ordered) if e.get("origin") == "user"]
asst_blocks = [(i, len(e.get("text", "")) // 4, len(e.get("semantic_links", {})) + len(e.get("entity_links", {})) + len(e.get("topic_links", {})))
               for i, (_, e) in enumerate(ordered) if e.get("origin") == "assistant"]

total_turns = len(ordered)
total_user_tokens = sum(t for _, t in user_blocks)
total_asst_tokens = sum(t for _, t, _ in asst_blocks)
avg_user = total_user_tokens // len(user_blocks) if user_blocks else 0
avg_asst = total_asst_tokens // len(asst_blocks) if asst_blocks else 0

print(f"{'=' * 85}")
print(f"COST SIMULATION v6 — CORRECT: Derived responses don't enter TCMM context")
print(f"{'=' * 85}")
print(f"  {total_turns} blocks | {len(user_blocks)} user ({avg_user} avg tok) | {len(asst_blocks)} assistant ({avg_asst} avg tok)")
print()

# Pricing
INPUT = 0.003
CACHE_WRITE = INPUT * 1.25
CACHE_READ = INPUT * 0.10
OUTPUT_K = 0.015
AVG_OUT = 150
SYS = 500

# Simulate conversation as turns (user, assistant, user, assistant...)
# Each turn = 1 user block + 1 assistant block (paired)
turns = []
ui, ai = 0, 0
for i in range(total_turns):
    _, e = ordered[i]
    tokens = len(e.get("text", "")) // 4
    role = e.get("origin", "unknown")
    links = len(e.get("semantic_links", {})) + len(e.get("entity_links", {})) + len(e.get("topic_links", {}))
    turns.append({"tokens": tokens, "role": role, "links": links, "idx": i})


def simulate(derived_pct):
    """Simulate with given % of assistant responses being derived (not archived)."""

    # Sort assistant blocks by links — lowest-linked are most likely derived
    asst_sorted = sorted(
        [(i, t) for i, t in enumerate(turns) if t["role"] == "assistant"],
        key=lambda x: x[1]["links"]
    )
    n_derived = int(len(asst_sorted) * derived_pct)
    derived_indices = set(x[0] for x in asst_sorted[:n_derived])

    # === NAIVE ===
    naive_history = 0
    naive_cum = 0
    naive_cost_nc = 0.0
    naive_cost_c = 0.0
    prev_n = 0

    for i, t in enumerate(turns):
        # Naive: EVERY block (user + assistant) goes into history
        naive_history += t["tokens"]
        naive_input = SYS + naive_history
        naive_cum += naive_input

        naive_cost_nc += (naive_input / 1000) * INPUT
        if i == 0 or prev_n == 0:
            naive_cost_c += (naive_input / 1000) * CACHE_WRITE
        else:
            naive_cost_c += (prev_n / 1000) * CACHE_READ
            naive_cost_c += (max(0, naive_input - prev_n) / 1000) * CACHE_WRITE
        prev_n = naive_input

    # === TCMM ===
    # Only user blocks + novel assistant blocks enter live
    tcmm_live = []  # list of token counts for blocks that entered live
    tcmm_archived = []  # (tokens, links) for blocks that made it to archive
    tcmm_cum = 0
    tcmm_cost_nc = 0.0
    tcmm_cost_c = 0.0
    prev_stable = 0
    LIVE_WIN = 20

    novel_count = 0
    derived_count = 0
    derived_tokens_skipped = 0

    for i, t in enumerate(turns):
        if t["role"] == "assistant" and i in derived_indices:
            # DERIVED: does NOT enter live blocks AT ALL
            derived_count += 1
            derived_tokens_skipped += t["tokens"]
            # Still need to calculate TCMM input for this turn though
            # (TCMM still sends live + recall, just doesn't ADD this block)
        else:
            # User block OR novel assistant block -> enters live
            tcmm_live.append(t["tokens"])
            if t["role"] == "assistant":
                novel_count += 1
            # Once promoted out of live window, goes to archive
            if len(tcmm_live) > LIVE_WIN + 30:  # past shadow window
                tcmm_archived.append((tcmm_live[0], t["links"]))

        # Calculate TCMM input for this turn
        # Live = last LIVE_WIN blocks that ENTERED live (not all turns!)
        live_start = max(0, len(tcmm_live) - LIVE_WIN)
        live_tokens = sum(tcmm_live[live_start:])

        # Recall from archive
        recalled = sorted(
            [b for b in tcmm_archived if b[1] > 3],
            key=lambda x: x[1], reverse=True
        )[:10]
        recall_tokens = sum(b[0] for b in recalled)

        stable = SYS + live_tokens
        tcmm_input = stable + recall_tokens
        tcmm_cum += tcmm_input

        tcmm_cost_nc += (tcmm_input / 1000) * INPUT

        if i == 0:
            tcmm_cost_c += (stable / 1000) * CACHE_WRITE
            tcmm_cost_c += (recall_tokens / 1000) * INPUT
        else:
            # Cache: live window shifts slowly (only when a novel block enters)
            if prev_stable > 0:
                # If we just added a block to live, 1 block changed
                # If this was derived (skipped), live didn't change = 100% cache hit!
                if t["role"] == "assistant" and i in derived_indices:
                    # Derived: live window didn't change! Perfect cache hit
                    tcmm_cost_c += (stable / 1000) * CACHE_READ
                    tcmm_cost_c += (recall_tokens / 1000) * INPUT
                else:
                    # Novel/user: 1 block shifted in live window
                    avg_blk = live_tokens // max(len(tcmm_live[live_start:]), 1)
                    changed = avg_blk * 2 if len(tcmm_live) > LIVE_WIN else avg_blk
                    cached_s = max(0, min(prev_stable, stable) - changed)
                    uncached_s = max(0, stable - cached_s)
                    tcmm_cost_c += (cached_s / 1000) * CACHE_READ
                    tcmm_cost_c += (uncached_s / 1000) * CACHE_WRITE
                    tcmm_cost_c += (recall_tokens / 1000) * INPUT
            else:
                tcmm_cost_c += (stable / 1000) * CACHE_WRITE
                tcmm_cost_c += (recall_tokens / 1000) * INPUT

        prev_stable = stable

    out_cost = total_turns * (AVG_OUT / 1000) * OUTPUT_K

    return {
        "derived_pct": derived_pct,
        "derived_count": derived_count,
        "novel_count": novel_count,
        "derived_tokens": derived_tokens_skipped,
        "naive_cum": naive_cum,
        "tcmm_cum": tcmm_cum,
        "naive_nc": naive_cost_nc + out_cost,
        "naive_c": naive_cost_c + out_cost,
        "tcmm_nc": tcmm_cost_nc + out_cost,
        "tcmm_c": tcmm_cost_c + out_cost,
        "tcmm_live_blocks": len(tcmm_live),
        "tcmm_final_live_tokens": sum(tcmm_live[max(0, len(tcmm_live) - LIVE_WIN):]),
        "naive_final": SYS + sum(t["tokens"] for t in turns),
    }


# Run at different derived rates
print(f"{'=' * 85}")
print(f"RESULTS: Naive+Cache vs TCMM+Cache at different derived rates")
print(f"{'=' * 85}")
print(f"  derived% = % of assistant responses that DON'T enter TCMM context")
print()
print(f"  {'Derived':>8} {'Skipped':>8} {'Naive+C':>10} {'TCMM+C':>10} {'Savings':>10} {'Save%':>7} "
      f"{'Naive/turn':>11} {'TCMM/turn':>11} {'TCMM live':>10}")
print(f"  {'-' * 89}")

for dp in [0.0, 0.20, 0.40, 0.60, 0.80, 0.90]:
    r = simulate(dp)
    savings = r["naive_c"] - r["tcmm_c"]
    pct = savings * 100 / r["naive_c"]
    label = ""
    if dp == 0.0: label = "  (benchmark raw)"
    elif dp == 0.80: label = "  <-- realistic"
    elif dp == 0.90: label = "  (aggressive)"

    print(f"  {dp*100:>6.0f}%  {r['derived_tokens']:>7,}t ${r['naive_c']:>8.4f} ${r['tcmm_c']:>8.4f} "
          f"${savings:>8.4f} {pct:>6.1f}% "
          f"${r['naive_c']/total_turns:>.6f} ${r['tcmm_c']/total_turns:>.6f} "
          f"{r['tcmm_final_live_tokens']:>8,}t{label}")

# Detailed comparison at 80% derived (realistic)
print()
r80 = simulate(0.80)
print(f"{'=' * 85}")
print(f"DETAILED: 80% derived (realistic live environment)")
print(f"{'=' * 85}")
print(f"  Assistant responses: {len(asst_blocks)} total")
print(f"    Novel (enter live):  {r80['novel_count']} ({r80['novel_count']*100//len(asst_blocks)}%)")
print(f"    Derived (discarded): {r80['derived_count']} ({r80['derived_count']*100//len(asst_blocks)}%)")
print(f"    Derived tokens skipped: {r80['derived_tokens']:,}")
print()
print(f"  Context size at turn 725:")
print(f"    Naive:  {r80['naive_final']:>8,} tokens (ALL messages)")
print(f"    TCMM:   {r80['tcmm_final_live_tokens']:>8,} tokens (live window only)")
print(f"    Ratio:  TCMM is {r80['tcmm_final_live_tokens']*100//r80['naive_final']}% of naive")
print()
print(f"  Cumulative input tokens:")
print(f"    Naive:  {r80['naive_cum']:>12,}")
print(f"    TCMM:   {r80['tcmm_cum']:>12,}")
print(f"    TCMM sends {r80['tcmm_cum']*100//r80['naive_cum']}% of naive's tokens")
print()
print(f"  Cost (with Anthropic cache):")
print(f"    Naive + cache:  ${r80['naive_c']:.4f} (${r80['naive_c']/total_turns:.6f}/turn)")
print(f"    TCMM + cache:   ${r80['tcmm_c']:.4f} (${r80['tcmm_c']/total_turns:.6f}/turn)")
savings = r80['naive_c'] - r80['tcmm_c']
pct = savings * 100 / r80['naive_c']
print(f"    TCMM saves:     ${savings:.4f} ({pct:.1f}%)")
print()
print(f"  Key: 80% of assistant turns = perfect cache hit for TCMM")
print(f"  (derived response discarded, live window unchanged, 100% prefix cached)")
