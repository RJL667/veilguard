"""
CPU-only benchmark of small ONNX-quantized embedding models for TCMM.

Compares 4 candidates head-to-head on:
  - cold load time
  - single-encode latency (warmed + median of 10)
  - batch-encode throughput (32-item batch, median of 5)
  - vector dim (sanity)

All forced through ONNX CPUExecutionProvider — no GPU even if available.

Inputs mimic real TCMM traffic:
  - short_query   : a typical recall query
  - medium_block  : a typical archive block (~500 chars)
  - long_block    : a longer narrative block (~2000 chars)
"""
import os
import time
import statistics

# Force CPU at every layer.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

from fastembed import TextEmbedding


SHORT_QUERY = "what did the user say about phishing emails last week"

MEDIUM_BLOCK = (
    "User flagged a suspicious email purporting to be from FNB. Sender domain "
    "spelled 'fnb-bank-verify.com' (extra hyphenation), and the link pointed to "
    "an Argentinian-hosted IP. The email asked them to 'verify their account "
    "within 24 hours or face suspension'. They forwarded it to phishield-soc@ "
    "and asked whether they should also report to SABRIC. Yes — SABRIC takes "
    "phish reports at phishing@sabric.co.za, and the FNB internal team is at "
    "phishing@fnb.co.za. Both should be CC'd on the same forward. We've added "
    "the IP to the SOC blocklist already."
)

LONG_BLOCK = (
    "Phishield engagement with Avalear's finance team — quarterly review of "
    "credential-stuffing attempts against their Microsoft 365 tenant. "
    "Telemetry shows a 340% increase in failed sign-ins from the AS-9009 "
    "(M247) prefix between 2026-03-15 and 2026-04-30, peaking around payroll "
    "run dates. We confirmed conditional access policy is enforcing MFA on "
    "all finance-group accounts, but two service accounts (svc-payroll-cron "
    "and svc-reconcile) were exempt by policy carve-out written in 2024. "
    "Recommendation: rotate those service accounts to use managed identities "
    "with cert-based auth, OR enforce a network-restricted CA policy scoped "
    "to the on-prem payroll subnet only. Finance director (REF_PERSON_1) "
    "agreed to the second approach in principle but wanted CIO sign-off "
    "before changing anything. Action: Sarel to draft the policy diff, "
    "Petrus to validate the network range, then we present jointly at the "
    "next steering committee on 2026-05-21. Risk if we don't move: a "
    "successful credential-stuffing landing on svc-payroll-cron has direct "
    "blast-radius into the payroll batch generator, which would let an "
    "attacker modify pending payments in the 4-hour pre-run window. "
    "Compensating control today is daily reconciliation but that's "
    "after-the-fact, not preventive."
)

INPUTS = [
    ("short_query", SHORT_QUERY),
    ("medium_block", MEDIUM_BLOCK),
    ("long_block", LONG_BLOCK),
]

CANDIDATES = [
    "BAAI/bge-small-en-v1.5",
    "snowflake/snowflake-arctic-embed-xs",
    "snowflake/snowflake-arctic-embed-s",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]


def median_ms(samples):
    return statistics.median(samples) * 1000


def bench(model_name: str):
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print('='*70)

    # Cold load
    t0 = time.perf_counter()
    try:
        model = TextEmbedding(
            model_name=model_name,
            providers=["CPUExecutionProvider"],
            threads=os.cpu_count(),
        )
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {e}")
        return
    load_ms = (time.perf_counter() - t0) * 1000

    # Warmup encode (downloads stay out of timing budget)
    _ = list(model.embed(["warmup"]))
    _ = list(model.embed(["warmup again"]))

    # Sanity: get dim
    sample_vec = list(model.embed(["dim probe"]))[0]
    dim = len(sample_vec)

    print(f"  load:           {load_ms:8.0f} ms  (one-time)")
    print(f"  embedding dim:  {dim}")

    # Single-encode latency, median of 10
    for label, text in INPUTS:
        samples = []
        for _ in range(10):
            t = time.perf_counter()
            _ = list(model.embed([text]))
            samples.append(time.perf_counter() - t)
        print(f"  single [{label:13s}]  median={median_ms(samples):6.1f} ms  "
              f"min={min(samples)*1000:6.1f} ms  "
              f"max={max(samples)*1000:6.1f} ms")

    # Batch throughput: 32 medium blocks, median of 5 runs
    batch = [MEDIUM_BLOCK] * 32
    samples = []
    for _ in range(5):
        t = time.perf_counter()
        _ = list(model.embed(batch))
        samples.append(time.perf_counter() - t)
    batch_ms = median_ms(samples)
    print(f"  batch[32]:       {batch_ms:7.0f} ms total  "
          f"({batch_ms/32:5.1f} ms/item  ~{32000/batch_ms:5.0f} items/sec)")


def main():
    print(f"CPU threads available: {os.cpu_count()}")
    print(f"Running each model — first run downloads ONNX (~50-200MB), be patient...")
    for name in CANDIDATES:
        try:
            bench(name)
        except Exception as e:
            print(f"\n!!! {name} raised: {type(e).__name__}: {e}")

    print(f"\n{'='*70}")
    print("DONE — recommendation: lowest median single-encode for short_query is")
    print("the fastest for recall queries; lowest batch ms/item is best for ingest.")
    print('='*70)


if __name__ == "__main__":
    main()
