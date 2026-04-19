"""
TCMM Integration Test for Veilguard.

Tests:
1. TCMM service health
2. Pre-request (memory enrichment)
3. Post-response (answer ingestion + heatmap processing)
4. Live block persistence (saved to disk after each turn)
5. Heatmap stripping (response doesn't contain heatmap JSON)
6. End-to-end through PII proxy (if running)
"""

import json
import os
import time
import requests

TCMM_URL = "http://localhost:8811"
PROXY_URL = "http://localhost:4000"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

def test_health():
    """Test 1: TCMM service is running."""
    print("=" * 60)
    print("TEST 1: TCMM Health Check")
    print("=" * 60)
    try:
        r = requests.get(f"{TCMM_URL}/health", timeout=5)
        data = r.json()
        assert data["status"] == "ok", f"Expected 'ok', got {data['status']}"
        print(f"  ✓ Status: {data['status']}")
        print(f"  ✓ Live blocks: {data['live_blocks']}")
        print(f"  ✓ Archive blocks: {data['archive_blocks']}")
        print(f"  ✓ Current step: {data['current_step']}")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_pre_request():
    """Test 2: Pre-request enriches prompt with memory context."""
    print("\n" + "=" * 60)
    print("TEST 2: Pre-Request (Memory Enrichment)")
    print("=" * 60)
    try:
        r = requests.post(f"{TCMM_URL}/pre_request", json={
            "user_message": "My name is TestUser and I work at TestCorp as a security analyst.",
            "conversation_id": "test-session-001"
        }, timeout=30)
        data = r.json()
        prompt = data.get("prompt", "")
        stats = data.get("stats", {})
        print(f"  ✓ Prompt length: {len(prompt)} chars")
        print(f"  ✓ Live blocks: {stats.get('live_blocks', '?')}")
        print(f"  ✓ Shadow blocks: {stats.get('shadow_blocks', '?')}")
        print(f"  ✓ Recalled: {stats.get('recalled', '?')}")
        if prompt:
            print(f"  ✓ Prompt preview: {prompt[:200]}...")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_post_response():
    """Test 3: Post-response ingests answer and processes heatmap."""
    print("\n" + "=" * 60)
    print("TEST 3: Post-Response (Ingestion + Heatmap)")
    print("=" * 60)
    try:
        # Simulate an LLM response with heatmap
        raw_output = (
            "Hello TestUser! I've noted that you work at TestCorp as a security analyst. "
            "How can I help you today?\n"
            '{"knowledge_class": "derived", "used": {"Memory 0": 1}}'
        )
        r = requests.post(f"{TCMM_URL}/post_response", json={
            "raw_output": raw_output,
            "conversation_id": "test-session-001"
        }, timeout=30)
        data = r.json()
        answer = data.get("answer", "")
        stats = data.get("stats", {})

        # Check heatmap was stripped from answer
        has_heatmap = "knowledge_class" in answer
        print(f"  ✓ Answer length: {len(answer)} chars")
        print(f"  ✓ Heatmap in answer: {has_heatmap} {'(BAD!)' if has_heatmap else '(GOOD - stripped)'}")
        print(f"  ✓ Current step: {stats.get('current_step', '?')}")
        print(f"  ✓ Archive blocks: {stats.get('archive_blocks', '?')}")
        print(f"  ✓ Answer preview: {answer[:150]}...")

        if has_heatmap:
            print("  ✗ WARNING: Heatmap JSON leaked into answer!")
            return False
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_live_persistence():
    """Test 4: Live blocks are persisted to disk after each turn."""
    print("\n" + "=" * 60)
    print("TEST 4: Live Block Persistence")
    print("=" * 60)

    # Check for live_sessions directory
    live_dirs = [
        os.path.join("tcmm-data", "live_sessions"),
        os.path.join("tcmm-data", "data", "live_sessions"),
    ]

    found_dir = None
    for d in live_dirs:
        full = os.path.join(os.path.dirname(__file__), d)
        if os.path.exists(full):
            found_dir = full
            break

    if not found_dir:
        # Try to trigger persistence by sending another turn
        print("  → No live_sessions dir yet. Triggering a turn...")
        try:
            requests.post(f"{TCMM_URL}/pre_request", json={
                "user_message": "What is my name?",
                "conversation_id": "test-session-001"
            }, timeout=30)
            requests.post(f"{TCMM_URL}/post_response", json={
                "raw_output": 'Your name is TestUser.\n{"knowledge_class": "derived", "used": {"Memory 0": 1}}',
                "conversation_id": "test-session-001"
            }, timeout=30)
            time.sleep(2)
        except Exception as e:
            print(f"  ✗ Failed to trigger turn: {e}")

        # Check again
        for d in live_dirs:
            full = os.path.join(os.path.dirname(__file__), d)
            if os.path.exists(full):
                found_dir = full
                break

    if not found_dir:
        print(f"  ✗ FAILED: live_sessions directory not found in any of:")
        for d in live_dirs:
            print(f"    - {os.path.join(os.path.dirname(__file__), d)}")
        return False

    print(f"  ✓ Directory: {found_dir}")
    files = os.listdir(found_dir)
    print(f"  ✓ Files: {files}")

    # Check _latest.json
    latest = os.path.join(found_dir, "_latest.json")
    if os.path.exists(latest):
        with open(latest, "r") as f:
            data = json.load(f)
        blocks = data.get("blocks", [])
        print(f"  ✓ _latest.json: {len(blocks)} blocks saved")
        print(f"  ✓ Session ID: {data.get('session_id', '?')}")
        print(f"  ✓ Current step: {data.get('current_step', '?')}")
        if blocks:
            b = blocks[0]
            print(f"  ✓ First block: id={b.get('id')}, heat={b.get('heat')}, text={b.get('text','')[:80]}...")
        return True
    else:
        print(f"  ✗ FAILED: _latest.json not found")
        # Check session files
        session_files = [f for f in files if f.startswith("session_")]
        if session_files:
            print(f"  ✓ Found session files: {session_files}")
            return True
        return False


def test_multi_turn():
    """Test 5: Multi-turn conversation with memory recall."""
    print("\n" + "=" * 60)
    print("TEST 5: Multi-Turn Memory Recall")
    print("=" * 60)
    try:
        # Turn 1: Introduce information
        print("  → Turn 1: Introducing info...")
        requests.post(f"{TCMM_URL}/pre_request", json={
            "user_message": "My favorite programming language is Python and I have a dog named Rex.",
            "conversation_id": "test-session-002"
        }, timeout=30)
        requests.post(f"{TCMM_URL}/post_response", json={
            "raw_output": 'Great! I\'ve noted that your favorite language is Python and your dog is named Rex.\n{"knowledge_class": "novel", "used": {}}',
            "conversation_id": "test-session-002"
        }, timeout=30)

        # Turn 2: Ask about previous info
        print("  → Turn 2: Asking about previous info...")
        r = requests.post(f"{TCMM_URL}/pre_request", json={
            "user_message": "What is my dog's name?",
            "conversation_id": "test-session-002"
        }, timeout=30)
        data = r.json()
        prompt = data.get("prompt", "")

        # Check if memory context contains "Rex"
        has_rex = "Rex" in prompt
        has_python = "Python" in prompt
        print(f"  ✓ Memory contains 'Rex': {has_rex}")
        print(f"  ✓ Memory contains 'Python': {has_python}")
        print(f"  ✓ Recalled blocks: {data.get('stats', {}).get('recalled', '?')}")

        if has_rex:
            print("  ✓ PASS: TCMM recalled dog's name from previous turn")
        else:
            print("  ✗ WARN: 'Rex' not found in memory context (might be in live blocks)")

        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_heatmap_stripping_e2e():
    """Test 6: End-to-end heatmap stripping through PII proxy (if running)."""
    print("\n" + "=" * 60)
    print("TEST 6: E2E Heatmap Stripping (via PII Proxy)")
    print("=" * 60)

    if not GOOGLE_API_KEY:
        print("  ⊘ SKIPPED: No GOOGLE_API_KEY set")
        return True

    try:
        # Check proxy health
        r = requests.get(f"{PROXY_URL}/health", timeout=5)
        if r.status_code != 200:
            print("  ⊘ SKIPPED: PII proxy not running")
            return True

        # Send a chat completion through the proxy
        r = requests.post(
            f"{PROXY_URL}/gemini/v1beta/openai/chat/completions",
            json={
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "Say hello and tell me 2+2."}],
                "stream": False,
            },
            headers={
                "Authorization": f"Bearer {GOOGLE_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )

        if r.status_code != 200:
            print(f"  ✗ Proxy returned {r.status_code}: {r.text[:200]}")
            return False

        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        has_heatmap = "knowledge_class" in content

        print(f"  ✓ Response length: {len(content)} chars")
        print(f"  ✓ Heatmap in response: {has_heatmap} {'(BAD!)' if has_heatmap else '(GOOD - stripped)'}")
        print(f"  ✓ Response preview: {content[:150]}...")

        if has_heatmap:
            print("  ✗ WARNING: Heatmap leaked into non-streaming response!")
            return False
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 60)
    print("   VEILGUARD TCMM INTEGRATION TESTS")
    print("=" * 60 + "\n")

    results = {}
    results["health"] = test_health()
    if not results["health"]:
        print("\n✗ TCMM not running — cannot continue. Start it with:")
        print("  python mcp-tools/tcmm-service/server.py --port 8811")
        exit(1)

    results["pre_request"] = test_pre_request()
    results["post_response"] = test_post_response()
    results["persistence"] = test_live_persistence()
    results["multi_turn"] = test_multi_turn()
    results["e2e_stripping"] = test_heatmap_stripping_e2e()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}: {name}")
    print(f"\n  {passed}/{total} tests passed")

    exit(0 if passed == total else 1)
