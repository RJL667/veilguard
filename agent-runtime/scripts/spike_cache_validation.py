"""Live-API cache validation spike — real-TCMM-output version.

THIS SCRIPT MUST BE RUN BY A HUMAN against a running TCMM service +
real ANTHROPIC_API_KEY.  It answers the ONE architectural unknown of
the agent-runtime bet:

  Does the Claude Agent SDK preserve TCMM's per-provider cache_control
  marker placement when it relays a system prefix to the Anthropic API?

Background: TCMM owns cache_control placement (per
`project_tcmm_renderer_architecture` memory).  The proxy slots TCMM's
output directly today and cache hits work.  agent-runtime intends to do
the same — pass TCMM bytes UNMODIFIED through the SDK.  If the SDK
rewrites the markers (the documented v2 bug), the cache breaks and we
need a workaround.

What this spike does:

  1. Calls TCMM's /render endpoint for a real conversation_id + user_id
     (you supply via flags) with --model=claude-sonnet-4-7 (or whatever
     you want to validate).
  2. Inspects the returned blocks: counts cache_control markers + their
     positions, prints a summary.
  3. Sends 5 sequential queries to Anthropic via the Claude Agent SDK,
     passing the TCMM blocks as system_prompt UNMODIFIED.
  4. Records per-call: input_new, cache_creation, cache_read, hit_rate,
     wall-clock.
  5. Verdict:
        - PASS if calls 2..N show ≥95% hit rate (SDK preserves markers
          and the Anthropic prompt cache works)
        - PARTIAL if cache works some of the time
        - FAIL if no hits (SDK is breaking marker placement)

If FAIL, the fix is to bypass the SDK's cache_control handling.  Options:
  - Patch the SDK locally (depends on version + maintainer responsiveness)
  - Use a raw Anthropic client for the LLM call inside agent-runtime
    (we already own the loop logic; the SDK's value is mostly subagents
    + MCP client, which we can wire to a different LLM caller)
  - Re-introduce normalize_cache_control() as a defensive normalizer
    BEFORE the SDK sees the prefix

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python scripts/spike_cache_validation.py \\
      --tcmm-url http://localhost:8811 \\
      --conversation-id <some-real-conv-id> \\
      --user-id <some-real-user-id> \\
      --calls 5

  # Or with a fixture (offline):
  python scripts/spike_cache_validation.py --fixture sample_render.json

Cost (rough):
  Real TCMM blob is ~260K tokens depending on conversation length.
  At Sonnet 4.7 rates:
    1 cache_write + 4 cache_reads × 260K tokens
    ≈ 260K × $6/M write + 4 × 260K × $0.30/M read
    ≈ $1.56 + $0.31 = ~$1.87 total.
  Shorter conversations are cheaper proportionally.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Make app.* importable when run from project root or scripts/ subdir.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _print_block_summary(blocks: list[dict]) -> None:
    """Print TCMM render details so you can see what TCMM gave us."""
    print(f"\n[spike] TCMM returned {len(blocks)} block(s):")
    total_chars = 0
    marker_positions = []
    for i, b in enumerate(blocks):
        text = b.get("text", "")
        n_chars = len(text)
        total_chars += n_chars
        has_marker = "cache_control" in b
        if has_marker:
            marker_positions.append(i)
        print(
            f"  block {i}: {n_chars:>9,} chars "
            f"cache_control={'YES' if has_marker else 'no '}"
            f"  preview: {text[:60].replace(chr(10), ' ')!r}..."
        )
    print(f"[spike] total: {total_chars:,} chars (~{total_chars//4:,} tokens estimated)")
    print(f"[spike] cache_control markers at positions: {marker_positions}")
    if not marker_positions:
        print("[spike] WARNING — TCMM returned 0 markers.  Without markers, the API")
        print("         will not cache.  Confirm TCMM's renderer is on the right")
        print("         provider path (model=claude-* should engage Anthropic renderer).")
    elif len(marker_positions) > 4:
        print("[spike] WARNING — TCMM returned >4 markers.  Anthropic hard-limits to 4")
        print("         per request; expect HTTP 400.")


async def fetch_tcmm_blocks(
    *,
    tcmm_url: str,
    conversation_id: str,
    user_id: str,
    model: str,
    secret: str,
) -> tuple[list[dict], str]:
    """Call TCMM /render and return (blocks, version)."""
    import httpx

    headers = {}
    if secret:
        headers["x-veilguard-internal-secret"] = secret

    payload = {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "model": model,
    }

    print(f"[spike] calling TCMM /render at {tcmm_url}...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{tcmm_url.rstrip('/')}/render",
            json=payload,
            headers=headers,
        )
    if r.status_code >= 400:
        print(f"[FAIL] TCMM /render returned {r.status_code}: {r.text[:300]}")
        sys.exit(2)

    data = r.json()
    blocks = data.get("blocks", [])
    version = data.get("version", "")
    return blocks, version


def load_fixture_blocks(path: Path) -> tuple[list[dict], str]:
    """Load a recorded TCMM /render response from disk."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("blocks", []), data.get("version", "fixture")


async def run_spike(
    *,
    system_blocks: list[dict],
    n_calls: int,
    model: str,
) -> None:
    try:
        from claude_agent_sdk import (
            query as sdk_query,
            ClaudeAgentOptions,
        )
    except ImportError:
        print("[FAIL] claude_agent_sdk not installed.")
        print("       Run: pip install claude-agent-sdk")
        sys.exit(2)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[FAIL] ANTHROPIC_API_KEY env var not set.")
        sys.exit(2)

    if not system_blocks:
        print("[FAIL] system_blocks is empty; nothing to send to the SDK.")
        sys.exit(2)

    _print_block_summary(system_blocks)

    results: list[dict] = []

    for n in range(1, n_calls + 1):
        # Vary only the user message between calls; system prefix is
        # byte-stable (TCMM's bytes, passed through unmodified).
        user_prompt = f"Call {n}: respond with exactly the word 'OK'."

        options = ClaudeAgentOptions(
            system_prompt=system_blocks,
            allowed_tools=[],
            model=model,
            max_turns=1,
        )

        print(f"\n[spike] call {n}/{n_calls} ...")
        t0 = time.time()

        usage_input = 0
        usage_output = 0
        cache_create = 0
        cache_read = 0

        try:
            stream = sdk_query(prompt=user_prompt, options=options)
            async for msg in stream:
                msg_type = getattr(msg, "type", None) or (
                    msg.get("type") if isinstance(msg, dict) else None
                )
                if msg_type == "assistant":
                    inner = getattr(msg, "message", None) or (
                        msg.get("message") if isinstance(msg, dict) else None
                    )
                    if inner is None:
                        continue
                    u = (
                        getattr(inner, "usage", None)
                        or (inner.get("usage") if isinstance(inner, dict) else None)
                    )
                    if u is not None:
                        usage_input += int(
                            getattr(u, "input_tokens", None)
                            or (u.get("input_tokens") if isinstance(u, dict) else 0)
                            or 0
                        )
                        usage_output += int(
                            getattr(u, "output_tokens", None)
                            or (u.get("output_tokens") if isinstance(u, dict) else 0)
                            or 0
                        )
                        cache_create += int(
                            getattr(u, "cache_creation_input_tokens", None)
                            or (u.get("cache_creation_input_tokens") if isinstance(u, dict) else 0)
                            or 0
                        )
                        cache_read += int(
                            getattr(u, "cache_read_input_tokens", None)
                            or (u.get("cache_read_input_tokens") if isinstance(u, dict) else 0)
                            or 0
                        )
        except Exception as e:
            print(f"[spike] call {n} ERROR: {e}")
            results.append({"call": n, "error": str(e)})
            continue

        dt = time.time() - t0
        total_input = usage_input + cache_create + cache_read
        hit_rate = (cache_read / total_input) if total_input > 0 else 0.0

        r = {
            "call": n,
            "wall_clock_s": round(dt, 2),
            "input_new": usage_input,
            "cache_create": cache_create,
            "cache_read": cache_read,
            "input_total": total_input,
            "output": usage_output,
            "hit_rate": round(hit_rate, 4),
        }
        results.append(r)
        print(
            f"        wall={dt:.2f}s  input_new={usage_input}  "
            f"cache_create={cache_create}  cache_read={cache_read}  "
            f"hit_rate={hit_rate:.1%}"
        )

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SPIKE SUMMARY — does the SDK preserve TCMM's cache_control markers?")
    print("=" * 70)

    print(f"{'Call':<6}{'Wall':<8}{'New':<10}{'Create':<10}{'Read':<10}{'Hit':<8}")
    print("-" * 60)
    for r in results:
        if "error" in r:
            print(f"{r['call']:<6}ERROR: {r['error'][:50]}")
            continue
        print(
            f"{r['call']:<6}{r['wall_clock_s']:<8}"
            f"{r['input_new']:<10}{r['cache_create']:<10}"
            f"{r['cache_read']:<10}{r['hit_rate']*100:<8.1f}"
        )

    print("-" * 60)

    successful = [r for r in results if "error" not in r]
    if len(successful) >= 2:
        post = successful[1:]
        avg_hit_rate = sum(r["hit_rate"] for r in post) / len(post)
        print(f"\nAverage cache hit rate calls 2..{len(successful)}: "
              f"{avg_hit_rate*100:.1f}%")

        if avg_hit_rate >= 0.95:
            print("[VERDICT] PASS — SDK preserves TCMM's cache_control markers.")
            print("          agent-runtime can pass TCMM bytes through unmodified.")
            print("          Phase 0.1 unblocked; deploy with confidence.")
        elif avg_hit_rate >= 0.50:
            print("[VERDICT] PARTIAL — some hits but lower than expected.")
            print("          Possible causes:")
            print("            - SDK is moving the marker between calls")
            print("            - TCMM's render bytes vary slightly per call")
            print("            - cache TTL shorter than expected (default 1h)")
            print("          Inspect SDK output bytes vs TCMM input bytes.")
        else:
            print("[VERDICT] FAIL — SDK is breaking cache_control placement.")
            print("          The SDK is rewriting markers between TCMM render")
            print("          and the Anthropic API call.  Workaround options:")
            print("            1. Bypass SDK's cache handling (raw Anthropic client)")
            print("            2. Patch claude-agent-sdk locally")
            print("            3. Re-enable normalize_cache_control() defensively")
    else:
        print("[VERDICT] INCONCLUSIVE — too few successful calls.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate that the Claude Agent SDK preserves TCMM's "
                    "cache_control markers end-to-end."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--tcmm-url",
        help="Live TCMM /render endpoint (e.g. http://localhost:8811).",
    )
    src.add_argument(
        "--fixture",
        type=Path,
        help="Path to a recorded /render response JSON.",
    )
    p.add_argument(
        "--conversation-id",
        help="Real conversation_id to render for (required if --tcmm-url).",
    )
    p.add_argument(
        "--user-id",
        help="Real user_id to render for (required if --tcmm-url).",
    )
    p.add_argument(
        "--secret",
        default=os.environ.get("VEILGUARD_INTERNAL_SECRET", ""),
        help="x-veilguard-internal-secret for TCMM (default: env var).",
    )
    p.add_argument(
        "--calls",
        type=int,
        default=5,
        help="Number of sequential SDK calls (default: 5).",
    )
    p.add_argument(
        "--model",
        default="claude-sonnet-4-7",
        help="Anthropic model id (default: claude-sonnet-4-7).",
    )
    return p.parse_args()


async def amain() -> None:
    args = _parse_args()

    if args.tcmm_url and not (args.conversation_id and args.user_id):
        print("[FAIL] --conversation-id and --user-id required when using --tcmm-url.")
        sys.exit(2)

    if args.fixture:
        blocks, version = load_fixture_blocks(args.fixture)
        print(f"[spike] loaded fixture: {args.fixture} (version={version})")
    else:
        blocks, version = await fetch_tcmm_blocks(
            tcmm_url=args.tcmm_url,
            conversation_id=args.conversation_id,
            user_id=args.user_id,
            model=args.model,
            secret=args.secret,
        )
        print(f"[spike] TCMM version: {version}")

    await run_spike(
        system_blocks=blocks,
        n_calls=args.calls,
        model=args.model,
    )


if __name__ == "__main__":
    asyncio.run(amain())
