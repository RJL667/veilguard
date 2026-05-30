"""Veilguard Sub-Agents: Agentic tool loop and tool handlers."""

import json
import logging
import os
import re
import subprocess
import time
import uuid

import httpx
from pathlib import Path

from config import PROJECT_ROOT, SCRATCHPAD_DIR
from core.llm import call_llm_with_tools
from utils.safety import is_path_safe, validate_command
from utils.tcmm import tcmm_recall

logger = logging.getLogger("veilguard.agentic")

# ── Agentic Tool Schemas ─────────────────────────────────────────────────────

AGENT_TOOLS = [
    {"name": "read_file", "description": "Read a file. Returns content with line numbers.\n\nWHEN TO USE: To examine source code, configs, logs.\nIMPORTANT: Reads up to 500 lines. Use offset/limit for large files.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "search_files", "description": "Search files by glob pattern.\n\nWHEN TO USE: Find files by name. WHEN NOT TO USE: Search contents — use grep.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}},
    {"name": "grep", "description": "Search file contents for regex pattern. Returns matching lines.\n\nWHEN TO USE: Find code or strings across files.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}, "include": {"type": "string"}}, "required": ["pattern"]}},
    {"name": "write_file", "description": "Create or overwrite a file.\n\nIMPORTANT: OVERWRITES entire file. Read first if modifying. NEVER write to .env or .git.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Targeted string replacement in a file.\n\nIMPORTANT: old_string must match EXACTLY. Read the file first.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}, "required": ["path", "old_string", "new_string"]}},
    {"name": "run_command", "description": "Run a shell command. Max 30s timeout.\n\nWHEN NOT TO USE: Destructive operations (blocked).",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "web_search", "description": "Search the web. Returns top results.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "web_fetch", "description": "Fetch a URL and return text content. Max 3000 chars.",
     "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
    {"name": "write_scratchpad", "description": "Write intermediate data to shared scratchpad.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "content": {"type": "string"}}, "required": ["name", "content"]}},
    {"name": "read_scratchpad", "description": "Read from shared scratchpad.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "tcmm_recall", "description": "Recall past memory/context from TCMM.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "todo_write", "description": "Track progress with a task checklist.",
     "input_schema": {"type": "object", "properties": {"items": {"type": "string"}}, "required": ["items"]}},
]

EFFORT_LEVELS = {
    "low":    {"max_turns": 3,  "model_hint": "flash"},
    "medium": {"max_turns": 7,  "model_hint": None},
    "high":   {"max_turns": 15, "model_hint": "pro"},
    "max":    {"max_turns": 20, "model_hint": "pro"},
}


_REF_TOKEN_PATTERN = re.compile(r"\bREF_[A-Z]+_\d+\b")


async def _rehydrate_args(args: dict) -> dict:
    """Replace REF_TYPE_N tokens in tool args with real values via the
    pii-proxy ``/rehydrate`` endpoint.

    The proxy's outbound streaming rehydration is best-effort: SSE
    chunk boundaries can split a REF token mid-string, the regex
    misses the split occurrence, and the broken token leaks into
    ``tool_use.input``. By the time LibreChat extracts the tool call
    and forwards it here the token is still raw — the search engine /
    file system / shell would otherwise receive ``REF_PERSON_4``
    instead of the real name.

    This pass is the SECOND chance: walk every string value in the
    args dict; if any contains ``REF_``, hit /rehydrate. Idempotent —
    calling on already-rehydrated text is a no-op (no mapping match).
    """
    if not isinstance(args, dict) or not args:
        return args
    has_ref = any(
        isinstance(v, str) and _REF_TOKEN_PATTERN.search(v)
        for v in args.values()
    )
    if not has_ref:
        return args
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            new_args = {}
            changed = 0
            for k, v in args.items():
                if isinstance(v, str) and _REF_TOKEN_PATTERN.search(v):
                    resp = await client.post(
                        "http://localhost:4000/rehydrate",
                        json={"text": v},
                    )
                    if resp.status_code == 200:
                        rehy = resp.json().get("text", v)
                        new_args[k] = rehy
                        if rehy != v:
                            changed += 1
                    else:
                        new_args[k] = v
                else:
                    new_args[k] = v
            if changed:
                logger.info(
                    f"[REHYDRATE-ARGS] rehydrated {changed} field(s) of REF tokens "
                    f"(proxy stream-rehydrate missed them across SSE boundaries)"
                )
            return new_args
    except Exception as e:
        logger.warning(f"[REHYDRATE-ARGS] failed: {e}; passing args through as-is")
        return args


async def handle_tool(name: str, args: dict) -> str:
    """Execute an agentic tool. Routes client-side tools through the bridge when not in local mode."""
    # Backstop rehydration: turn any REF_PERSON_N / REF_EMAIL_N etc.
    # tokens that survived the proxy's streaming rehydration into the
    # original PII values. Without this, REF placeholders end up in
    # web_search queries / file paths / shell commands and tools fail.
    args = await _rehydrate_args(args)

    # Check if this tool should be routed to the client daemon
    from utils.tool_location import is_client_tool, SANDBOX_FALLBACK_TOOLS
    from core.client_bridge import SANDBOX_USER_ID
    import server as _srv
    if not getattr(_srv, "LOCAL_MODE", True) and is_client_tool(name):
        from core.client_bridge import get_bridge
        from core.request_ctx import get_user_id, get_conversation_id
        # Per-user routing: each LibreChat user's tool calls only reach
        # their own registered client daemon.  Without this, whoever's
        # daemon connected last would receive every user's tool calls.
        user_id = get_user_id()
        conv_id = get_conversation_id() or ""
        bridge = get_bridge(user_id)
        if not user_id:
            return (
                "Error: No user context — MCP request missing x-user-id "
                "header. Reconnect LibreChat so it forwards user identity."
            )

        # Sandbox fallback — RESERVED for server-side identities.
        #
        # 2026-05-25 incident: a real user's daemon flapped offline for a
        # few seconds between connect / heartbeat.  Their next tool call
        # silently routed to the sandbox container (Linux), so the LLM
        # saw `/sandbox-workspace` instead of their actual Windows
        # filesystem.  Pretty bad — felt like the chat was lying about
        # what it could see.
        #
        # New rule: fallback fires only when the caller is a server-side
        # identity (background daemons, scheduled workers, Director runs
        # — anything where there's no human in the loop to notice the
        # silent reroute).  Interactive chat users always see their own
        # machine OR a clean "no daemon connected" error.
        if (not bridge or not bridge.connected) and name in SANDBOX_FALLBACK_TOOLS:
            # Heuristic for "server-side": user_id is a colon-namespaced
            # synthetic ID (e.g. "system:sandbox", "daemon:inbox-watcher"),
            # NOT a MongoDB ObjectId from LibreChat (24 hex chars).
            is_server_caller = (
                ":" in user_id
                or user_id.startswith(("system:", "daemon:", "schedule:"))
            )
            if is_server_caller:
                sandbox = get_bridge(SANDBOX_USER_ID)
                if sandbox and sandbox.connected:
                    logger.info(
                        f"[agentic] server caller {user_id[:16]!r} daemon "
                        f"offline; routing {name!r} to server sandbox"
                    )
                    bridge = sandbox
            else:
                logger.warning(
                    f"[agentic] user {user_id[:8]} daemon transiently "
                    f"offline; NOT falling back to sandbox (interactive "
                    f"caller — failing loud)"
                )

        # ── Approval gate ────────────────────────────────────────────────
        # Run BEFORE dispatching to the bridge.  Combines static policy
        # (CLIENT_TOOLS risk class) with the user's permission_level
        # (auto / confirm / strict) and the conv-level override.  DENY /
        # timeout / user-refused → return an Error string the LLM sees
        # in the next message.  Approved → fall through to the bridge.
        try:
            from core.approval import gate as _approval_gate
            gate_result = await _approval_gate(
                tool=name, args=args,
                user_id=user_id, conv_id=conv_id,
                bridge=bridge, agent_id="user",
                is_background=False,
            )
        except Exception as e:
            logger.warning(f"[approval] gate raised; defaulting to deny: {e}")
            return f"Error: Approval gate failed: {e}"

        if not gate_result.proceed:
            return f"Error: Tool call blocked — {gate_result.reason}"

        # ── Submit through TaskDispatcher (Phase B-1) ───────────────────
        # Single ledger row per tool call means list_my_tasks / cancel /
        # subtree queries see every routing decision.  Executor picked
        # by which bridge we actually landed on (user's daemon vs the
        # server sandbox).  Falls back to direct bridge.execute_remote
        # if the dispatcher path errors — keeps tool calls working when
        # the unified store has a transient hiccup.
        if bridge and bridge.connected:
            try:
                from core.tasks import Task, TaskKind, TaskDispatcher
                # Pick executor name based on which bridge we resolved.
                # bridge.user_id was set at WS-auth time, so it's the
                # authoritative source of truth for "who am I routing to."
                exec_name = (
                    "sandbox" if bridge.user_id == SANDBOX_USER_ID
                    else "daemon"
                )
                # Pull parent task id from the contextvar if a caller
                # (e.g. a sub-agent runner) populated it.  Empty parent
                # is fine — task becomes its own root.
                from core.request_ctx import current_child_conversation_id
                parent_tid = ""   # populated by future B-2 sub-agent runner

                task = Task.new(
                    kind=TaskKind.TOOL_CALL,
                    executor=exec_name,
                    user_id=user_id,
                    conv_id=conv_id,
                    parent_task_id=parent_tid,
                    title=f"{name}({','.join(list(args.keys())[:3])})",
                    payload={"tool": name, "args": args},
                )
                dispatcher = TaskDispatcher.get()
                await dispatcher.submit(task)
                # The dispatcher spawns the executor and returns
                # immediately.  We need the result synchronously to
                # hand back to the LLM, so poll the store until the
                # task is terminal.  Short polling loop with backoff —
                # tool calls typically complete in <2s.  Cap at 90s to
                # match bridge.execute_remote's default timeout + slack.
                import asyncio as _asyncio
                deadline = time.time() + 90
                while time.time() < deadline:
                    fresh = dispatcher.get_task(task.task_id)
                    if fresh is not None and fresh.is_terminal:
                        return fresh.result or (
                            f"Error: {fresh.error}" if fresh.error else ""
                        )
                    await _asyncio.sleep(0.1)
                # Timeout — return what we have + a marker for the LLM.
                logger.warning(
                    f"[agentic] dispatcher task {task.task_id} did not "
                    "reach terminal state within 90s"
                )
                return "Error: Tool call timed out (dispatcher 90s)"
            except Exception as e:
                # Dispatcher unavailable / Lance hiccup → degrade to
                # direct bridge call so chat keeps working.  Log
                # loudly so the regression is visible.
                logger.exception(
                    f"[agentic] dispatcher failed for {name!r}; "
                    f"falling back to direct bridge: {e}"
                )
                return await bridge.execute_remote(name, args)
        return (
            "Error: No client daemon connected for this user. "
            "Install and start the Veilguard client daemon from the "
            "LibreChat Cowork panel."
        )

    # Local execution (--local mode or cloud-only tools)
    try:
        if name == "read_file":
            path = args.get("path", "")
            full = os.path.join(PROJECT_ROOT, path) if not os.path.isabs(path) else path
            if not os.path.exists(full):
                return f"Error: File not found: {full}"
            offset = int(args.get("offset", 0))
            limit = int(args.get("limit", 500))
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
            selected = all_lines[offset:offset + limit]
            return "".join(f"{offset + i + 1}\t{l}" for i, l in enumerate(selected))

        elif name == "search_files":
            pattern = args.get("pattern", "*")
            search_path = args.get("path", PROJECT_ROOT)
            if not os.path.isabs(search_path):
                search_path = os.path.join(PROJECT_ROOT, search_path)
            matches = sorted(Path(search_path).glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)[:50]
            return "\n".join(str(m.relative_to(PROJECT_ROOT)) if str(m).startswith(PROJECT_ROOT) else str(m) for m in matches) or "(no matches)"

        elif name == "grep":
            pattern = args.get("pattern", "")
            path = args.get("path", ".")
            include = args.get("include", "")
            # Use relative path from PROJECT_ROOT — avoids Windows path escaping issues
            if os.path.isabs(path):
                # Convert absolute to relative if under PROJECT_ROOT
                try:
                    search_path = os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")
                except ValueError:
                    search_path = path.replace("\\", "/")
            elif path and path != ".":
                search_path = path.replace("\\", "/")
            else:
                search_path = "."
            # Build grep args as a list (not shell string) — avoids quote mangling
            grep_args = ["grep", "-rn", "-m", "30"]
            if include:
                grep_args.append(f"--include={include}")
            else:
                for ext in ["*.py", "*.md", "*.json", "*.yaml", "*.yml", "*.txt", "*.js", "*.ts"]:
                    grep_args.append(f"--include={ext}")
            grep_args.extend(["--", pattern, search_path])
            try:
                result = subprocess.run(grep_args, capture_output=True, text=True, timeout=15, cwd=PROJECT_ROOT)
                output = result.stdout[:3000]
            except FileNotFoundError:
                # grep not available — fallback to findstr on Windows
                search_dir = os.path.join(PROJECT_ROOT, search_path)
                result = subprocess.run(
                    ["findstr", "/S", "/N", "/R", pattern, os.path.join(search_dir, "*.*")],
                    capture_output=True, text=True, timeout=15, cwd=PROJECT_ROOT
                )
                output = result.stdout[:3000]
            return output or "(no matches)"

        elif name == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")
            full = os.path.join(PROJECT_ROOT, path) if not os.path.isabs(path) else path
            safe, reason = is_path_safe(full)
            if not safe:
                return reason
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Written {len(content)} chars to {path}"

        elif name == "edit_file":
            path = args.get("path", "")
            old_string = args.get("old_string", "")
            new_string = args.get("new_string", "")
            full = os.path.join(PROJECT_ROOT, path) if not os.path.isabs(path) else path
            safe, reason = is_path_safe(full)
            if not safe:
                return reason
            if not os.path.exists(full):
                return f"Error: File not found: {full}"
            with open(full, "r", encoding="utf-8") as f:
                content = f.read()
            if old_string not in content:
                return f"Error: old_string not found in {path}. Read the file first."
            if content.count(old_string) > 1:
                return f"Error: old_string appears {content.count(old_string)} times. Provide more context."
            with open(full, "w", encoding="utf-8") as f:
                f.write(content.replace(old_string, new_string, 1))
            return f"Edited {path}: replaced {len(old_string)} chars with {len(new_string)} chars"

        elif name == "run_command":
            cmd = args.get("command", "")
            safe, reason = validate_command(cmd)
            if not safe:
                return reason
            # On Windows: use cmd /c for simple commands, bash for pipes/redirects
            if os.name == "nt":
                has_pipe = "|" in cmd or ">" in cmd or "&&" in cmd
                if has_pipe:
                    # Pipes need a proper shell — use cmd /c with shell=True
                    result = subprocess.run(
                        cmd, shell=True,
                        capture_output=True, text=True, timeout=30, cwd=PROJECT_ROOT
                    )
                else:
                    result = subprocess.run(
                        ["cmd", "/c", cmd],
                        capture_output=True, text=True, timeout=30, cwd=PROJECT_ROOT
                    )
            else:
                result = subprocess.run(
                    ["bash", "-c", cmd],
                    capture_output=True, text=True, timeout=30, cwd=PROJECT_ROOT
                )
            out = result.stdout[:2000]
            if result.stderr:
                out += f"\nstderr: {result.stderr[:500]}"
            return out or "(no output)"

        elif name == "web_search":
            # Vertex AI google_search grounding — same backend as the
            # `web` MCP's google_search tool.  Auth via ADC (VM's SA).
            # Cloud-side so the agent can search even when the user's
            # client daemon isn't connected (web search doesn't need
            # anything local).  Previous DuckDuckGo scrape returned
            # raw HTML noise; Vertex returns structured citations.
            from utils.vertex_search import vertex_grounded_search
            query = args.get("query", "")
            n = min(int(args.get("num_results", 8) or 8), 10)
            return await vertex_grounded_search(query, n)

        elif name == "web_fetch":
            url = args.get("url", "")
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "Veilguard-Agent/1.0"})
                if resp.status_code != 200:
                    return f"Error: HTTP {resp.status_code}"
                text = resp.text[:3000]
                text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text[:3000]

        elif name == "write_scratchpad":
            SCRATCHPAD_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r'[^\w\-.]', '_', args.get("name", "temp"))
            content = args.get("content", "")
            # Rehydrate PII tokens before writing — scratchpad stores real data
            if "REF_" in content:
                try:
                    async with httpx.AsyncClient(timeout=3) as client:
                        resp = await client.post("http://localhost:4000/rehydrate", json={"text": content})
                        if resp.status_code == 200:
                            content = resp.json().get("text", content)
                except Exception:
                    pass
            (SCRATCHPAD_DIR / f"{safe_name}.txt").write_text(content, encoding="utf-8")
            return f"Written to scratchpad: {safe_name}"

        elif name == "read_scratchpad":
            safe_name = re.sub(r'[^\w\-.]', '_', args.get("name", ""))
            p = SCRATCHPAD_DIR / f"{safe_name}.txt"
            return p.read_text(encoding="utf-8") if p.exists() else f"Scratchpad '{safe_name}' not found"

        elif name == "tcmm_recall":
            return await tcmm_recall(args.get("query", ""))

        elif name == "todo_write":
            items = json.loads(args.get("items", "[]"))
            return "\n".join(f"- [{'x' if i.get('done') else ' '}] {i.get('task', '?')}" for i in items)

        else:
            return f"Unknown tool: {name}"
    except Exception as e:
        # ``str(e)`` is empty for some exception types (httpx.ReadTimeout,
        # certain asyncio cancellations). Emit the class name + repr so
        # the log is actionable instead of ``Tool web_fetch error:``
        # with nothing after the colon. Same on the returned string so
        # the agentic loop's next turn can reason about what failed.
        err_class = type(e).__name__
        err_detail = str(e) or repr(e) or "no detail"
        logger.error(f"Tool {name} error ({err_class}): {err_detail}", exc_info=True)
        return f"Error executing {name}: [{err_class}] {err_detail}"


async def agentic_loop(
    system_prompt: str,
    task: str,
    max_turns: int = 10,
    backend: str = "",
    model: str = "",
    tools: list[dict] | None = None,
) -> str:
    """Multi-turn agentic loop. Agent uses tools iteratively until done."""
    if tools is None:
        tools = AGENT_TOOLS

    messages = [{"role": "user", "content": task}]
    all_text = []
    transcript = []

    for turn in range(max_turns):
        # Micro-compaction for long conversations
        total_chars = sum(len(json.dumps(m)) for m in messages)
        if total_chars > 30000:
            for i in range(len(messages) - 4):
                m = messages[i]
                if isinstance(m.get("content"), list):
                    for j, block in enumerate(m["content"]):
                        if isinstance(block, dict) and block.get("type") == "tool_result" and len(str(block)) > 500:
                            m["content"][j] = {"type": "tool_result", "tool_use_id": block.get("tool_use_id", ""), "content": "[compacted]"}

        content_blocks, stop_reason = await call_llm_with_tools(system_prompt, messages, tools, backend, model)

        tool_calls = []
        for block in content_blocks:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    all_text.append(block["text"])
                elif block.get("type") == "tool_use":
                    tool_calls.append(block)

        transcript.append({"turn": turn + 1, "tools": [tc["name"] for tc in tool_calls]})
        logger.debug(f"Agentic turn {turn+1}: {len(tool_calls)} tools, stop={stop_reason}")

        if stop_reason != "tool_use" or not tool_calls:
            break

        messages.append({"role": "assistant", "content": content_blocks})
        tool_results = []
        for tc in tool_calls:
            result = await handle_tool(tc["name"], tc.get("input", {}))
            tool_results.append({"type": "tool_result", "tool_use_id": tc["id"], "content": result[:3000]})
        messages.append({"role": "user", "content": tool_results})

    # If we exited via max_turns exhaustion while the last turn was
    # still calling tools, the assistant never got a chance to write
    # the final synthesis — the for-loop just walked off the end with
    # `tool_use` still pending. That's why the 22 Apr Mexican-gov run
    # had all four workers return "(Agent produced no text output)"
    # despite each having made 10+ grounded web_search/web_fetch
    # calls whose results were safely archived by TCMM. Fix: force ONE
    # synthesis turn with tools=[] so the model MUST return text.
    if stop_reason == "tool_use" and tool_calls and len(transcript) >= max_turns:
        logger.info(
            f"Agentic hit max_turns={max_turns} mid-tool-use — "
            f"forcing synthesis turn (tool_calls={[tc['name'] for tc in tool_calls]})"
        )
        # Append the pending assistant+tool_results so the model sees
        # everything it already learned before being asked to answer.
        messages.append({"role": "assistant", "content": content_blocks})
        tool_results = []
        for tc in tool_calls:
            result = await handle_tool(tc["name"], tc.get("input", {}))
            tool_results.append({"type": "tool_result", "tool_use_id": tc["id"], "content": result[:3000]})
        messages.append({"role": "user", "content": tool_results})
        messages.append({
            "role": "user",
            "content": (
                "You've reached the tool-turn limit. Using everything the tools "
                "returned above, write the final answer now — no more tool calls."
            ),
        })
        try:
            synth_blocks, _synth_stop = await call_llm_with_tools(
                system_prompt, messages, tools=[], backend=backend, model=model,
            )
            for block in synth_blocks:
                if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                    all_text.append(block["text"])
            transcript.append({"turn": "synth", "tools": ["final-answer"]})
        except Exception as e:
            logger.warning(f"synthesis turn failed: {type(e).__name__}: {e}")

    output = "\n\n".join(all_text) if all_text else "(Agent produced no text output)"
    turn_summary = ", ".join(f"T{t['turn']}:{'+'.join(t['tools']) or 'text'}" for t in transcript)
    return f"[Agentic: {len(transcript)} turns — {turn_summary}]\n\n{output}"
