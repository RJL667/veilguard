"""
Test: Heatmap stripping in STREAMING responses through the PII proxy.

This test hits the actual proxy → Gemini streaming path and checks that
the heatmap JSON is NOT visible in the streamed response.
"""

import json
import os
import sys
import requests

PROXY_URL = "http://localhost:4000"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyBBUtrDVFVaHt-i01rHU7nHOkUeVUOOCVc")


def test_streaming_heatmap_stripping():
    """Send a streaming chat completion through the proxy and check for heatmap leak."""
    print("=" * 60)
    print("TEST: Streaming Heatmap Stripping (via PII Proxy → Gemini)")
    print("=" * 60)

    # Use a prompt that will trigger TCMM memory recall + heatmap
    r = requests.post(
        f"{PROXY_URL}/gemini/v1beta/openai/chat/completions",
        json={
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "What do you know about me? Show me your memory."}],
            "stream": True,
        },
        headers={
            "Authorization": f"Bearer {GOOGLE_API_KEY}",
            "Content-Type": "application/json",
        },
        stream=True,
        timeout=60,
    )

    if r.status_code != 200:
        print(f"  FAILED: HTTP {r.status_code}")
        print(f"  Body: {r.text[:500]}")
        return False

    # Collect all streamed content
    full_content = []
    raw_lines = []
    for line in r.iter_lines(decode_unicode=True):
        raw_lines.append(line)
        if line and line.startswith("data: ") and line != "data: [DONE]":
            try:
                chunk = json.loads(line[6:])
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if "content" in delta:
                    full_content.append(delta["content"])
            except (json.JSONDecodeError, IndexError):
                pass

    assembled = "".join(full_content)

    print(f"\n  Response length: {len(assembled)} chars")
    print(f"  SSE lines received: {len(raw_lines)}")

    # Check for heatmap leak
    heatmap_leaked = False
    markers = ["knowledge_class", '"used":', '"derived"', '"novel"', '"recalled"']
    for marker in markers:
        if marker in assembled:
            heatmap_leaked = True
            print(f"  LEAKED: Found '{marker}' in response!")

    print(f"\n  --- Full response ---")
    print(f"  {assembled}")
    print(f"  --- End response ---\n")

    if heatmap_leaked:
        print("  FAIL: Heatmap JSON is leaking into the streamed response!")
        # Show the last 300 chars to see exactly what leaked
        print(f"\n  Last 300 chars: ...{assembled[-300:]}")
        return False
    else:
        print("  PASS: No heatmap found in streamed response!")
        return True


def test_non_streaming_heatmap_stripping():
    """Send a NON-streaming request and verify heatmap is stripped."""
    print("\n" + "=" * 60)
    print("TEST: Non-Streaming Heatmap Stripping (via PII Proxy → Gemini)")
    print("=" * 60)

    r = requests.post(
        f"{PROXY_URL}/gemini/v1beta/openai/chat/completions",
        json={
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "What is 2+2? Keep it brief."}],
            "stream": False,
        },
        headers={
            "Authorization": f"Bearer {GOOGLE_API_KEY}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )

    if r.status_code != 200:
        print(f"  FAILED: HTTP {r.status_code}: {r.text[:500]}")
        return False

    data = r.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    has_heatmap = "knowledge_class" in content
    print(f"  Response: {content[:200]}")
    print(f"  Heatmap in response: {has_heatmap}")

    if has_heatmap:
        print("  FAIL: Heatmap leaked!")
        return False
    print("  PASS: Clean response!")
    return True


if __name__ == "__main__":
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', closefd=False)

    results = []
    results.append(("streaming", test_streaming_heatmap_stripping()))
    results.append(("non-streaming", test_non_streaming_heatmap_stripping()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")

    all_pass = all(ok for _, ok in results)
    print(f"\n  {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
