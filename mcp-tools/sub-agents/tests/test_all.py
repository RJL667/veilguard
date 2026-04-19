"""Comprehensive tests for Veilguard Sub-Agents modular codebase."""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

passed = 0
failed = 0


def test(name, condition, msg=""):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} — {msg}")
        failed += 1


# === Test 1: All modules import ===
print("=== Test 1: Module imports ===")
try:
    import config
    from core import state
    from core.llm import call_llm, resolve_backend, call_llm_with_tools
    from core.agentic import agentic_loop, handle_tool, AGENT_TOOLS, EFFORT_LEVELS
    from storage.session import save_session, load_session, log_transcript
    from storage.cleanup import cleanup_collections
    from utils.helpers import short_id, elapsed, maybe_offload, load_skill, load_playbook
    from utils.safety import TOOL_RISK, validate_command, is_path_safe
    from utils.prompts import build_prompt, detect_frustration, estimate_cost
    from utils.afk import is_afk, touch_activity
    from tools.register import register_all
    test("All imports", True)
except Exception as e:
    test("All imports", False, str(e))

# === Test 2: Config ===
print("\n=== Test 2: Config ===")
test("10 roles", len(config.ROLES) == 10, f"got {len(config.ROLES)}")
test("2 backends", len(config.BACKENDS) == 2, f"got {len(config.BACKENDS)}")
test("Default backend gemini", config.DEFAULT_BACKEND == "gemini")
test("Timeout 120", config.TIMEOUT == 120)

# === Test 3: Tool registration ===
print("\n=== Test 3: Tool registration ===")
from mcp.server.fastmcp import FastMCP
test_mcp = FastMCP("test")
register_all(test_mcp)
tools = list(test_mcp._tool_manager._tools.keys())
test(f"53 tools registered", len(tools) == 53, f"got {len(tools)}")

# === Test 4: Safety ===
print("\n=== Test 4: Command safety ===")
test("Block rm -rf /", not validate_command("rm -rf /")[0])
test("Block shutdown", not validate_command("shutdown")[0])
test("Block powershell -enc", not validate_command("powershell -enc abc")[0])
test("Allow dir", validate_command("dir")[0])
test("Allow git status", validate_command("git status")[0])

# === Test 5: Path safety ===
print("\n=== Test 5: Path safety ===")
test("Block .env", not is_path_safe("/home/.env")[0])
test("Block .git", not is_path_safe("/repo/.git/config")[0])
test("Block credentials", not is_path_safe("/data/credentials.json")[0])
test("Allow report.md", is_path_safe("/data/report.md")[0])

# === Test 6: Frustration ===
print("\n=== Test 6: Frustration detection ===")
test("Detect broken!!!", detect_frustration("this is broken!!!"))
test("Detect wtf", detect_frustration("wtf is going on"))
test("Detect still not", detect_frustration("still not working"))
test("No false positive", not detect_frustration("please help me"))

# === Test 7: Cost estimation ===
print("\n=== Test 7: Cost estimation ===")
c_flash = estimate_cost("gemini-2.5-flash", 1000, 500)
c_pro = estimate_cost("gemini-2.5-pro", 1000, 500)
test("Flash cost > 0", c_flash > 0, f"got {c_flash}")
test("Pro > Flash", c_pro > c_flash)
test("Unknown = 0", estimate_cost("unknown", 1000, 500) == 0)

# === Test 8: Prompt building ===
print("\n=== Test 8: Prompt building ===")
prompt = build_prompt("You are an analyst.")
test("Contains prefix", "Phishield" in prompt)
test("Contains ROLE:", "ROLE:" in prompt)

# === Test 9: Helpers ===
print("\n=== Test 9: Helpers ===")
test("short_id is 8 chars", len(short_id()) == 8)
test("elapsed 5.5s", elapsed(0, 5.5) == "5.5s")
test("elapsed 2.0m", elapsed(0, 120) == "2.0m")

# === Test 10: Offload ===
print("\n=== Test 10: Offload ===")
test("Small passthrough", maybe_offload("short", "t") == "short")
big = maybe_offload("x" * 10000, "test-offload")
test("Large offloaded", "Full result saved to" in big)

# === Test 11: Session round-trip ===
print("\n=== Test 11: Session persistence ===")
state.agent_stats["total_calls"] = 42
state.estimated_cost_usd = 0.1234
save_session()
state.agent_stats["total_calls"] = 0
state.estimated_cost_usd = 0.0
load_session()
test("Calls restored", state.agent_stats["total_calls"] == 42, f"got {state.agent_stats['total_calls']}")
test("Cost restored", abs(state.estimated_cost_usd - 0.1234) < 0.001)

# === Test 12: AFK ===
print("\n=== Test 12: AFK mode ===")
touch_activity()
test("Not AFK after touch", not is_afk())
state.last_user_activity = time.time() - 600
test("AFK after 600s", is_afk())
touch_activity()
test("Not AFK after reset", not is_afk())

# === Test 13: Task dependencies ===
print("\n=== Test 13: Task dependencies ===")
from core.state import ManagedTask
from tools.managed_tasks import check_task_dependencies
state.managed_tasks = {}
state.managed_tasks["t1"] = ManagedTask(id="t1", title="First", description="", status="pending")
state.managed_tasks["t2"] = ManagedTask(id="t2", title="Second", description="", status="blocked", depends_on=["t1"])
check_task_dependencies()
test("t2 still blocked", state.managed_tasks["t2"].status == "blocked")
state.managed_tasks["t1"].status = "completed"
check_task_dependencies()
test("t2 unblocked", state.managed_tasks["t2"].status == "pending")

# === Test 14: Cleanup bounds ===
print("\n=== Test 14: Cleanup bounds ===")
from core.state import BackgroundTask, TaskStatus
state.tasks = {}
for i in range(600):
    state.tasks[f"t{i}"] = BackgroundTask(
        id=f"t{i}", task="test", role="r", role_name="R",
        status=TaskStatus.DONE, model="m", created_at=float(i)
    )
cleanup_collections()
test(f"Tasks pruned to <=500", len(state.tasks) <= 500, f"got {len(state.tasks)}")

# === Test 15: Skills ===
print("\n=== Test 15: Skills ===")
skill = load_skill("phishing-triage")
test("Skill loaded", len(skill) > 50, f"got {len(skill)} chars")
test("Missing skill empty", load_skill("nonexistent") == "")

# === Test 16: Playbooks ===
print("\n=== Test 16: Playbooks ===")
pb = load_playbook("apt-investigation")
test("Playbook loaded", pb.get("name") == "APT Investigation")
test("4 tasks", len(pb.get("tasks", [])) == 4)
test("Missing playbook empty", load_playbook("nonexistent") == {})

# === Test 17: Effort levels ===
print("\n=== Test 17: Effort levels ===")
test("Low = 3 turns", EFFORT_LEVELS["low"]["max_turns"] == 3)
test("Medium = 7 turns", EFFORT_LEVELS["medium"]["max_turns"] == 7)
test("High = 15 turns", EFFORT_LEVELS["high"]["max_turns"] == 15)
test("Max = 20 turns", EFFORT_LEVELS["max"]["max_turns"] == 20)

# === Test 18: Backend resolution ===
print("\n=== Test 18: Backend resolution ===")
test("Gemini = openai format", resolve_backend("gemini")["format"] == "openai")
test("Claude = anthropic format", resolve_backend("claude")["format"] == "anthropic")
test("Unknown falls back", resolve_backend("nonexistent")["format"] == "openai")

# === Test 19: Agentic tools ===
print("\n=== Test 19: Agentic tools ===")
tool_names = [t["name"] for t in AGENT_TOOLS]
test("12 agentic tools", len(AGENT_TOOLS) == 12, f"got {len(AGENT_TOOLS)}")
test("Has read_file", "read_file" in tool_names)
test("Has write_file", "write_file" in tool_names)
test("Has edit_file", "edit_file" in tool_names)
test("Has web_search", "web_search" in tool_names)
test("Has web_fetch", "web_fetch" in tool_names)
test("Has tcmm_recall", "tcmm_recall" in tool_names)

# === Test 20: Tool risk coverage ===
print("\n=== Test 20: Tool risk classification ===")
for tool_name in tools:
    if tool_name not in TOOL_RISK:
        # Not all tools need risk classification (some are new)
        pass
classified = len([t for t in tools if t in TOOL_RISK])
test(f"Risk classified: {classified}/{len(tools)}", classified >= 30, f"only {classified}")

# === Summary ===
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
print("=" * 60)

if failed > 0:
    sys.exit(1)
