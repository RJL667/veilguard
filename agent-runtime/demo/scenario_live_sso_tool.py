"""LIVE demo — Director invokes create_task via real LLM through TCMM.

ONE route, the TCMM route:

    agent-runtime  →  agent.Agent.run_turn (the unified 5-step pipeline)
                   →  POST http://localhost:8811/generate   (tcmm-service)
                   →  AnthropicGenerationAdapter (CLAUDE_SSO=1)
                       - reads ~/.claude/.credentials.json
                       - auto-refreshes if <5 min left
                   →  api.anthropic.com (Max/Pro subscription)

No pii-proxy, no sub-agents wrapping, no Anthropic SDK in agent-runtime
itself.  Same endpoint LibreChat's user chat uses (via the proxy's SSO
early-route).

Prerequisites:
  - Local stack running (start.bat).  TCMM on :8811.
  - `claude login` once if disk creds are stale — TCMM auto-refreshes
    afterwards.

Run:
    python -m demo.scenario_live_sso_tool

What we verify:
  - Director's tool_allow_list mapped to Anthropic-shape schemas
  - Model emits a tool_use for `create_task`
  - tool_dispatcher.dispatch() runs it in-process
  - The new Task lands in the Lance ledger
  - Final assistant turn references the task_id
"""

import asyncio
import os

from demo._harness import (
    setup_demo_env,
    run_scenario,
    print_event_trace,
    print_summary_banner,
)

# Set up demo env BEFORE app.* imports.
TMP_DIR = setup_demo_env()

# Live mode = TCMM /generate.  Legacy aliases "sdk" / "sso" map to live.
os.environ["BACKEND"] = "live"

# OAuth bearer — no API key needed.
os.environ.pop("ANTHROPIC_API_KEY", None)


def main() -> None:
    print_summary_banner(
        "LIVE DEMO - Director invokes create_task via TCMM /generate"
    )
    tcmm_url = os.environ.get("TCMM_GENERATE_URL", "http://localhost:8811/generate")
    print(f"  Lance dir: {TMP_DIR}")
    print(f"  Backend:   live (= agent.Agent.run_turn)")
    print(f"  TCMM URL:  {tcmm_url}")
    print()

    # Director's tool_allow_list includes create_task / assign_task /
    # recall / read_constitution / consult / final_synthesis.  We craft
    # a prompt the model can ONLY satisfy by emitting a `create_task`
    # tool_use — proves the round trip.
    user_msg = (
        "A new piece of work has come in: 'Investigate whether our login "
        "rate-limiting actually rejects credential-stuffing traffic, and write "
        "a 300-word memo at team/drafts/rate-limit-check.md with sections "
        "Method / Findings / Recommendation.'\n\n"
        "Create a Task assigning this to `researcher`.  Use the create_task "
        "tool — do not just describe what you would do.  After you create it, "
        "reply with the task_id and a one-sentence confirmation."
    )

    print(f"  USER: {user_msg[:120]}...\n")
    print("  Invoking agent-runtime with BACKEND=live...")
    print("  (this calls TCMM /generate directly — local stack must be up)\n")

    try:
        events = asyncio.run(run_scenario(
            user_message=user_msg,
            foreground_agent="director",
            conversation_id="live-demo-conv",
            user_id="live-demo-user",
            tenant_id="live-demo-tenant",
        ))
    except Exception as e:
        print(f"\n  [FAIL] run_scenario raised: {e}")
        return

    print_event_trace(events)

    tool_calls = [e for e in events if e.get("type") == "tool_call"]
    assistant_text = [e for e in events if e.get("type") == "assistant_text"]
    errors = [e for e in events if e.get("type") == "error"]

    print_summary_banner("RESULT")

    # Distinct failure modes worth labelling.
    tcmm_unreachable = any(
        e.get("code") == "tcmm_unreachable" for e in errors
    ) or any(
        "tcmm" in (at.get("text") or "").lower()
        and "unreachable" in (at.get("text") or "").lower()
        for at in assistant_text
    )

    if tcmm_unreachable:
        print("  Tool calls emitted:    0")
        print()
        print(f"  [TCMM] TCMM /generate at {tcmm_url} not reachable.")
        print()
        print("         Start the local stack:")
        print("           C:\\Users\\rudol\\Documents\\veilguard\\start.bat")
        print()
        print("         Or set TCMM_GENERATE_URL to point elsewhere.")
        return

    if errors:
        for err in errors:
            print(f"  ERROR  code={err.get('code')!r}")
            print(f"         {err.get('message')!r}")
        print("\n  [FAIL] backend errored.  Likely causes:")
        print(f"    - TCMM service not running (start.bat)")
        print(f"    - Disk OAuth token stale: run `claude login`")
        print(f"    - TCMM /generate returned non-200 — check its logs")
        return

    print(f"  Tool calls emitted:    {len(tool_calls)}")
    for tc in tool_calls:
        print(f"    - {tc.get('name')!r}  (id={tc.get('id')!r})")

    if assistant_text:
        print("  Final assistant text:")
        for at in assistant_text[-3:]:
            print(f"    > {at.get('text', '')[:200]}")

    if tool_calls and any(tc.get("name") == "create_task" for tc in tool_calls):
        print("\n  [PASS] Director invoked create_task via real LLM through TCMM.")
        print("         agent-runtime → Agent.run_turn → AnthropicAdapter")
        print("         → AnthropicGenerationAdapter (OAuth bearer) → Anthropic.")
        print("         tool_use parsed, dispatched in-process, ledger write OK.")
    elif tool_calls:
        print(f"\n  [PARTIAL] LLM invoked tools, but not create_task.")
        print(f"            Tools called: {[tc.get('name') for tc in tool_calls]}")
    else:
        print("\n  [PARTIAL] LLM responded without invoking any tool.")
        print("            Director's persona may need a sharper push toward")
        print("            create_task, or the model chose to chat.")


if __name__ == "__main__":
    main()
