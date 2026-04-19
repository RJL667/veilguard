"""Veilguard Sub-Agents: LLM calling (Gemini + Claude)."""

import json
import logging
import time

import httpx

from config import BACKENDS, DEFAULT_BACKEND, TIMEOUT
from core import state
from utils.prompts import build_prompt, detect_frustration, FRUSTRATION_PREFIX, estimate_cost
from utils.afk import touch_activity

logger = logging.getLogger("veilguard.llm")


def resolve_backend(backend: str = "") -> dict:
    """Resolve backend config by name, falling back to default."""
    name = (backend or DEFAULT_BACKEND).lower()
    if name not in BACKENDS:
        name = DEFAULT_BACKEND
    return BACKENDS[name]


def _track_error(backend_name: str, role: str):
    """Increment error counters."""
    state.agent_stats["total_errors"] += 1
    if backend_name in state.agent_stats["by_backend"]:
        state.agent_stats["by_backend"][backend_name]["errors"] += 1
    if role:
        state.agent_stats["by_role"].setdefault(role, {"calls": 0, "errors": 0})
        state.agent_stats["by_role"][role]["errors"] += 1


def _post_call_hook():
    """Called after every LLM call. Handles periodic save + cleanup."""
    touch_activity()
    state.calls_since_save += 1
    if state.calls_since_save >= 5:
        state.calls_since_save = 0
        from storage.session import save_session
        from storage.cleanup import cleanup_collections
        save_session()
        cleanup_collections()


async def call_llm(
    system_prompt: str,
    user_message: str,
    model: str = "",
    temperature: float = 0.7,
    backend: str = "",
    role: str = "",
) -> str:
    """Call LLM API (Gemini or Claude) with hooks, stats, and frustration detection."""
    cfg = resolve_backend(backend)
    api_key = cfg["key"]
    use_model = model if model else cfg["default_model"]
    fmt = cfg["format"]
    url = cfg["url"]
    backend_name = (backend or DEFAULT_BACKEND).lower()

    if not api_key:
        return f"Error: API key not configured for backend '{backend_name}'"

    # Pre-hook: frustration detection
    effective_prompt = system_prompt
    if detect_frustration(user_message):
        effective_prompt = FRUSTRATION_PREFIX + system_prompt

    # Pre-hook: cache-aware prompt composition
    effective_prompt = build_prompt(effective_prompt)

    # Stats: pre-call
    state.agent_stats["total_calls"] += 1
    if backend_name in state.agent_stats["by_backend"]:
        state.agent_stats["by_backend"][backend_name]["calls"] += 1
    if role:
        state.agent_stats["by_role"].setdefault(role, {"calls": 0, "errors": 0})
        state.agent_stats["by_role"][role]["calls"] += 1

    call_start = time.time()

    # Inherit parent conversation id so sub-agent LLM calls land in the
    # SAME TCMM namespace as the parent chat instead of each spawning
    # a fresh `new-<userid>-XXX` namespace. Without this, every
    # spawn_agent / spawn_agentic / coordinate call fragments memory:
    # tool calls + results get scattered across unrelated namespaces,
    # temporal chains are broken, and the semantic linker has nothing
    # to link across. The x-conversation-id middleware in server.py
    # captured this into state.active_conversation_id — we just have
    # to forward it.
    active_cid = getattr(state, "active_conversation_id", "") or ""

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            if fmt == "anthropic":
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": use_model,
                    "max_tokens": 4096,
                    "temperature": temperature,
                    "system": effective_prompt,
                    "messages": [{"role": "user", "content": user_message}],
                }
                if active_cid:
                    payload["metadata"] = {"conversation_id": active_cid}
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    _track_error(backend_name, role)
                    return f"Error: Claude API returned {response.status_code}: {response.text[:500]}"
                data = response.json()
                blocks = data.get("content", [])
                result = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            else:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": use_model,
                    "messages": [
                        {"role": "system", "content": effective_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "temperature": temperature,
                    "max_tokens": 4096,
                }
                if active_cid:
                    payload["metadata"] = {"conversation_id": active_cid}
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    _track_error(backend_name, role)
                    return f"Error: API returned {response.status_code}: {response.text[:500]}"
                data = response.json()
                result = data["choices"][0]["message"]["content"]

        # Post-hooks
        elapsed_ms = int((time.time() - call_start) * 1000)
        if backend_name in state.agent_stats["by_backend"]:
            state.agent_stats["by_backend"][backend_name]["total_ms"] += elapsed_ms

        call_cost = estimate_cost(use_model, len(user_message) + len(effective_prompt), len(result))
        state.estimated_cost_usd += call_cost

        _post_call_hook()

        state.call_log.append({
            "time": time.strftime("%H:%M:%S"),
            "backend": backend_name,
            "model": use_model,
            "role": role or "?",
            "elapsed_ms": elapsed_ms,
            "input_len": len(user_message),
            "output_len": len(result),
            "cost_usd": round(call_cost, 6),
        })
        if len(state.call_log) > 100:
            state.call_log.pop(0)

        logger.info(f"LLM call: {backend_name}/{use_model} role={role} {elapsed_ms}ms in={len(user_message)} out={len(result)}")
        return result

    except httpx.TimeoutException:
        _track_error(backend_name, role)
        logger.error(f"LLM timeout: {backend_name}/{use_model} after {TIMEOUT}s")
        return f"Error: Sub-agent timed out after {TIMEOUT}s"
    except Exception as e:
        _track_error(backend_name, role)
        logger.error(f"LLM error: {backend_name}/{use_model}: {e}")
        return f"Error calling LLM: {e}"


async def call_llm_with_tools(
    system_prompt: str,
    messages: list,
    tools: list[dict],
    backend: str = "",
    model: str = "",
) -> tuple[list, str]:
    """Call LLM with tool definitions. Returns (content_blocks, stop_reason)."""
    cfg = resolve_backend(backend)
    api_key = cfg["key"]
    use_model = model if model else cfg["default_model"]
    fmt = cfg["format"]
    url = cfg["url"]

    if not api_key:
        return [{"type": "text", "text": f"Error: No API key for {backend}"}], "error"

    # Inherit parent conversation id — see call_llm() above for rationale.
    active_cid = getattr(state, "active_conversation_id", "") or ""

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            if fmt == "anthropic":
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": use_model,
                    "max_tokens": 4096,
                    "system": build_prompt(system_prompt),
                    "messages": messages,
                    "tools": tools,
                }
                if active_cid:
                    payload["metadata"] = {"conversation_id": active_cid}
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code != 200:
                    return [{"type": "text", "text": f"API error {resp.status_code}: {resp.text[:300]}"}], "error"
                data = resp.json()
                return data.get("content", []), data.get("stop_reason", "end_turn")
            else:
                # OpenAI-compatible format
                oai_tools = [{
                    "type": "function",
                    "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]},
                } for t in tools]
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                oai_messages = [{"role": "system", "content": build_prompt(system_prompt)}]
                for m in messages:
                    if isinstance(m.get("content"), str):
                        oai_messages.append(m)
                    elif isinstance(m.get("content"), list):
                        text_parts = []
                        for block in m["content"]:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_parts.append(block["text"])
                            elif isinstance(block, dict) and block.get("type") == "tool_result":
                                text_parts.append(f"[Tool result]: {block.get('content', '')}")
                        oai_messages.append({"role": m["role"], "content": "\n".join(text_parts)})
                    else:
                        oai_messages.append(m)

                payload = {
                    "model": use_model,
                    "messages": oai_messages,
                    "tools": oai_tools if oai_tools else None,
                    "max_tokens": 4096,
                }
                if active_cid:
                    payload["metadata"] = {"conversation_id": active_cid}
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code != 200:
                    return [{"type": "text", "text": f"API error {resp.status_code}: {resp.text[:300]}"}], "error"
                data = resp.json()
                choice = data["choices"][0]
                msg = choice["message"]

                content = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for tc in (msg.get("tool_calls") or []):
                    content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"],
                    })
                stop = "tool_use" if msg.get("tool_calls") else "end_turn"
                return content, stop

    except Exception as e:
        logger.error(f"LLM with tools error: {e}")
        return [{"type": "text", "text": f"Error: {e}"}], "error"


async def compact_results(worker_outputs: list[str], backend: str = "", model: str = "") -> str:
    """If combined output exceeds threshold, summarize each worker's output."""
    from config import COMPACT_THRESHOLD
    combined = "\n\n---\n\n".join(worker_outputs)
    if len(combined) <= COMPACT_THRESHOLD:
        return combined

    return await call_llm(
        "Summarize each worker's findings into key points. Preserve all facts, "
        "IOCs, MITRE IDs, names, dates, and recommendations. Remove filler and "
        "repetition. Keep under 3000 chars total.",
        combined, backend=backend, model=model, role="compactor", temperature=0.2,
    )
