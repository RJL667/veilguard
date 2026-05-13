"""
Domain-specific retrieval-quality benchmark for TCMM candidate embedders.

Speed alone is meaningless: a 2ms embedder that misranks your blocks is
worthless. This script builds a small golden set of (query, expected_block)
pairs in Phishield's domain (SA cybersecurity, banking, personnel/ops) plus
distractor blocks, then asks each model: "for each query, can you rank the
correct block in the top-K?".

Metrics per model:
  hits@1, hits@3, hits@5  — % of queries where the correct block landed in top-K
  MRR                      — mean reciprocal rank (1.0 = always #1, 0.5 = avg #2)
  worst_misses             — queries where the correct block didn't make top-5

This is a SMALL golden set (~20 pairs). Take results as directional, not
statistically rigorous — but a model that misses 30% of these is genuinely
worse for TCMM than a model that misses 5%.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import numpy as np
from fastembed import TextEmbedding


# ─────────────────────────────────────────────────────────────────────
# GOLDEN SET: query → correct_block_id
# Each block has an id, a body, and (optionally) the id of a query that
# should retrieve it. Distractor blocks have no query — they're noise
# the model must NOT rank above the correct block.
# ─────────────────────────────────────────────────────────────────────

BLOCKS = [
    # ── Banking / phishing (id 1-5)
    {"id": 1, "text": "User flagged a phishing email pretending to be FNB. Spoofed domain fnb-bank-verify.com, link pointed to an Argentina-hosted IP. Reported to SABRIC and FNB internal phishing team."},
    {"id": 2, "text": "Standard Bank SMS scam variant: 'your account has been locked, click here to verify'. URL ends in .ru. Blocked at the proxy and added to deny list."},
    {"id": 3, "text": "Capitec customer received a vishing call from someone claiming to be from fraud department asking for OTP. They hung up. We're advising them to call Capitec's real number to confirm no compromise."},
    {"id": 4, "text": "User reported invoice fraud — vendor email looks legitimate but the banking details in the PDF attachment had been changed. Real bank account vs. fraudulent account differ by one digit."},
    {"id": 5, "text": "Apple ID phishing wave hitting Mac users this week. Lures around 'your subscription will be renewed' with a fake support contact form."},

    # ── IT security operations (id 10-14)
    {"id": 10, "text": "Quarterly review of credential-stuffing against Avalear's M365 tenant. 340% increase in failed sign-ins from AS-9009 (M247) between March 15 and April 30, peaking on payroll dates."},
    {"id": 11, "text": "Patched a CVE in the Cisco ASA firewall yesterday; reboot scheduled for 2am Sunday during the maintenance window. Notified all change-board members."},
    {"id": 12, "text": "Kaspersky EDR alerted on Mimikatz signatures in a developer's laptop. Investigation: it's a security researcher running them in a sandbox for training material. Whitelisted the path."},
    {"id": 13, "text": "Failed login attempts hitting the VPN concentrator from a Russian IP range, ~50/min. Geo-blocked at the perimeter; user accounts unaffected."},
    {"id": 14, "text": "DKIM record for our outbound mail had a typo introduced last week, breaking deliverability to Gmail. Fixed and verified with a test send to a personal address."},

    # ── Personnel / access (id 20-24)
    {"id": 20, "text": "Sarah from finance needs read-only access to the SOC dashboard for her audit work. Adding her to the soc-readonly AD group, expiring on 2026-06-30."},
    {"id": 21, "text": "Petrus is leaving end of the month. Offboarding checklist: revoke M365, revoke VPN, transfer ownership of 3 shared mailboxes, disable badge, return laptop."},
    {"id": 22, "text": "New hire onboarding for the IR team: Themba starts Monday. Provision Splunk read, EDR console viewer, JIRA security project, Slack #incidents channel."},
    {"id": 23, "text": "Contractor access for the Pretoria audit firm — 2-week temporary credentials with read-only scope on the finance file share. MFA enforced."},
    {"id": 24, "text": "CIO requested admin elevation for the new Azure landing zone. Approved with break-glass procedure — temporary 24h window logged in PIM."},

    # ── Distractors (no matching query; must not outrank correct blocks)
    {"id": 90, "text": "Coffee machine in the Cape Town office is broken again. Facilities raised a ticket with the vendor for replacement."},
    {"id": 91, "text": "The Q1 team-building event is scheduled for the Constantia Wine Estate on May 30th. RSVPs needed by Wednesday."},
    {"id": 92, "text": "Reminder: bin day at the Johannesburg office is Tuesday. Please don't leave food in the kitchen over the long weekend."},
    {"id": 93, "text": "Pool car booking system was down for 3 hours this morning. Driver bookings reverted to email approval temporarily."},
    {"id": 94, "text": "Annual cybersecurity awareness week starts Monday. Lunch-and-learns daily at 12:30, all employees expected to attend at least one session."},
    {"id": 95, "text": "Server room AC unit failing intermittently. Backup unit kicked in. Maintenance team coming Wednesday to replace the compressor."},
]

# Queries: each maps to the ID of the block that should be #1 retrieval
GOLDEN_QUERIES = [
    ("what was that suspicious FNB email someone forwarded last week?", 1),
    ("the bank phishing message about account verification with a Russian link", 2),
    ("the OTP phone scam from a Capitec impersonator", 3),
    ("the case where invoice payment details were swapped", 4),
    ("Apple subscription renewal scam emails", 5),
    ("the M365 brute force from M247 we discussed at the quarterly review", 10),
    ("our Cisco firewall maintenance window plan", 11),
    ("false positive on Mimikatz from the security researcher", 12),
    ("VPN login brute force from Russia", 13),
    ("the outbound mail DKIM problem with Gmail", 14),
    ("Sarah's audit access to the SOC dashboard", 20),
    ("Petrus leaving — what do we need to revoke", 21),
    ("Themba's IR team onboarding tasks", 22),
    ("the Pretoria audit firm contractor account scope", 23),
    ("Azure landing zone admin elevation for the CIO", 24),
]


CANDIDATES = [
    "BAAI/bge-small-en-v1.5",
    "snowflake/snowflake-arctic-embed-xs",
    "snowflake/snowflake-arctic-embed-s",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def eval_model(model_name: str):
    print(f"\n{'='*72}")
    print(f"MODEL: {model_name}")
    print('='*72)

    try:
        model = TextEmbedding(model_name=model_name,
                              providers=["CPUExecutionProvider"],
                              threads=os.cpu_count())
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {e}")
        return None

    # Embed all blocks
    block_ids = [b["id"] for b in BLOCKS]
    block_texts = [b["text"] for b in BLOCKS]
    block_vecs = np.array(list(model.embed(block_texts)))
    # Embed all queries
    query_texts = [q[0] for q in GOLDEN_QUERIES]
    query_vecs = np.array(list(model.embed(query_texts)))

    hits1 = hits3 = hits5 = 0
    reciprocal_ranks = []
    misses = []

    for q_idx, (q_text, correct_id) in enumerate(GOLDEN_QUERIES):
        # Score every block, sort desc
        scores = block_vecs @ query_vecs[q_idx]  # cosine since vectors are normalized for these models
        # (some models don't auto-normalize, but for ranking the L2-not-cosine
        # gap is tiny; the order is what matters for hits@K)
        ranked = sorted(zip(block_ids, scores), key=lambda x: x[1], reverse=True)
        rank_of_correct = next(i + 1 for i, (bid, _) in enumerate(ranked) if bid == correct_id)
        reciprocal_ranks.append(1.0 / rank_of_correct)
        if rank_of_correct == 1: hits1 += 1
        if rank_of_correct <= 3: hits3 += 1
        if rank_of_correct <= 5: hits5 += 1
        if rank_of_correct > 5:
            misses.append((q_text[:60], correct_id, rank_of_correct,
                          [bid for bid, _ in ranked[:5]]))

    n = len(GOLDEN_QUERIES)
    mrr = sum(reciprocal_ranks) / n
    print(f"  hits@1:  {hits1}/{n}  ({100*hits1/n:.0f}%)")
    print(f"  hits@3:  {hits3}/{n}  ({100*hits3/n:.0f}%)")
    print(f"  hits@5:  {hits5}/{n}  ({100*hits5/n:.0f}%)")
    print(f"  MRR:     {mrr:.3f}   (1.0 = always #1, 0.5 = avg position 2)")
    if misses:
        print(f"  worst misses (correct didn't make top-5):")
        for q, cid, rank, top5 in misses:
            print(f"    Q: '{q}...'  expected #{cid}, actual rank #{rank}, top5: {top5}")
    return {"hits1": hits1, "hits3": hits3, "hits5": hits5, "mrr": mrr, "n": n}


def main():
    results = {}
    for name in CANDIDATES:
        r = eval_model(name)
        if r:
            results[name] = r

    print(f"\n{'='*72}")
    print("SUMMARY (sort by MRR — higher is better)")
    print('='*72)
    print(f"{'model':60s} {'hits@1':>7s} {'hits@3':>7s} {'hits@5':>7s} {'MRR':>6s}")
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["mrr"], reverse=True):
        n = r["n"]
        print(f"{name:60s} {r['hits1']:>3d}/{n:<3d} {r['hits3']:>3d}/{n:<3d} {r['hits5']:>3d}/{n:<3d} {r['mrr']:>6.3f}")


if __name__ == "__main__":
    main()
