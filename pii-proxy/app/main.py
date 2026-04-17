"""
VEILGUARD PII Gateway
======================
Multi-LLM PII redaction gateway. Sits between LibreChat and any LLM API.

Routes:
  /anthropic/*  → https://api.anthropic.com/*
  /openai/*     → https://api.openai.com/*
  /gemini/*     → https://generativelanguage.googleapis.com/*

All user-authored content is scanned for PII before forwarding.
All responses are rehydrated (PII tokens → original values) before returning.

LibreChat config:
  ANTHROPIC_BASE_URL=http://pii-proxy:4000/anthropic
  OPENAI_BASE_URL=http://pii-proxy:4000/openai/v1
  (Google: custom endpoint with baseURL http://pii-proxy:4000/gemini)
"""

import json
import logging
import os
import sys
import uuid

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from .redactor import get_redactor
from .session import pii_store

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [VEILGUARD] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pii-proxy")

# ── PII Audit Log ────────────────────────────────────────────────────────────
# Logs full redacted prompts sent to LLM and responses received.
# Written to /app/logs/pii_audit.log (mounted volume) for inspection.

_AUDIT_ENABLED = os.environ.get("PII_AUDIT", "true").lower() in ("true", "1", "yes")
_audit_logger = None

if _AUDIT_ENABLED:
    _audit_dir = os.environ.get("PII_AUDIT_DIR", "/app/logs")
    os.makedirs(_audit_dir, exist_ok=True)
    _audit_logger = logging.getLogger("pii-audit")
    _audit_logger.setLevel(logging.DEBUG)
    _audit_logger.propagate = False
    _audit_handler = logging.FileHandler(os.path.join(_audit_dir, "pii_audit.log"), encoding="utf-8")
    _audit_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _audit_logger.addHandler(_audit_handler)


def audit_log(direction: str, conv_id: str, content: str, extra: str = ""):
    """Write to the PII audit log. direction: 'TO_LLM' or 'FROM_LLM'."""
    if _audit_logger:
        # Truncate very long content for the log
        truncated = content[:3000] + "..." if len(content) > 3000 else content
        _audit_logger.info(
            f"\n{'='*80}\n"
            f"[{direction}] conv={conv_id[:12] if conv_id else '?'} {extra}\n"
            f"{'─'*80}\n"
            f"{truncated}\n"
            f"{'='*80}"
        )

PORT = int(os.environ.get("PII_PROXY_PORT", "4000"))
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.7"))

# TCMM Integration
TCMM_ENABLED = os.environ.get("TCMM_ENABLED", "false").lower() in ("true", "1", "yes")
TCMM_URL = os.environ.get("TCMM_URL", "http://host.docker.internal:8811")

# Backend routing table
BACKENDS = {
    "anthropic": os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com"),
    "openai": os.environ.get("OPENAI_API_URL", "https://api.openai.com"),
    "gemini": os.environ.get("GEMINI_API_URL", "https://generativelanguage.googleapis.com"),
}

app = FastAPI(title="Veilguard PII Gateway")


@app.on_event("startup")
async def startup():
    logger.info("Loading Presidio NLP models...")
    get_redactor(min_score=MIN_SCORE)
    logger.info("=" * 50)
    logger.info(f"Veilguard PII Gateway ready on port {PORT}")
    for name, url in BACKENDS.items():
        logger.info(f"  /{name}/* → {url}")
    if TCMM_ENABLED:
        logger.info(f"  TCMM: {TCMM_URL} (ENABLED)")
    else:
        logger.info(f"  TCMM: disabled")
    logger.info("=" * 50)


# ── PII Rehydration Endpoint ─────────────────────────────────────────────────
# Used by sub-agents to rehydrate PII tokens in scratchpad/tool output for UI display

@app.post("/rehydrate")
async def rehydrate_endpoint(request: Request):
    """Rehydrate PII tokens in text. Called by sub-agents for scratchpad display."""
    body = await request.json()
    text = body.get("text", "")
    if not text:
        return {"text": ""}
    # Rehydrate across all sessions — try each conversation's PII map
    result = text
    for conv_id in list(pii_store._store.keys()):
        result = pii_store.rehydrate(conv_id, result)
    return {"text": result}


# ── TCMM Integration Helpers ─────────────────────────────────────────────────

def _extract_last_user_message(messages: list) -> str:
    """Extract the latest user message from OpenAI or Anthropic format messages array."""
    if not messages:
        return ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # Anthropic format: content can be a list of blocks
            if isinstance(content, list):
                return " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ).strip()
            return str(content).strip()
    return ""


async def _tcmm_pre_request(user_message: str, conversation_id: str, user_id: str = "") -> str | None:
    """Call TCMM service to get enriched prompt. Returns None on failure."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{TCMM_URL}/pre_request",
                json={"user_message": user_message, "conversation_id": conversation_id, "user_id": user_id},
            )
            if resp.status_code == 200:
                data = resp.json()
                prompt = data.get("prompt")
                if prompt:
                    stats = data.get("stats", {})
                    logger.info(
                        f"  [TCMM] pre_request OK — "
                        f"recalled={stats.get('recalled', 0)}, "
                        f"live={stats.get('live_blocks', 0)}, "
                        f"shadow={stats.get('shadow_blocks', 0)}, "
                        f"{stats.get('elapsed_ms', 0)}ms"
                    )
                    return prompt
                error = data.get("error", "no prompt returned")
                logger.warning(f"  [TCMM] pre_request failed: {error}")
            else:
                logger.warning(f"  [TCMM] pre_request HTTP {resp.status_code}")
    except httpx.ConnectError:
        logger.warning("  [TCMM] service unreachable — falling through without memory")
    except Exception as e:
        logger.warning(f"  [TCMM] pre_request error: {e}")
    return None


async def _tcmm_post_response(raw_output: str, conversation_id: str, user_id: str = "") -> str | None:
    """Call TCMM service to process response. Returns clean answer or None on failure."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{TCMM_URL}/post_response",
                json={"raw_output": raw_output, "conversation_id": conversation_id, "user_id": user_id},
            )
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer")
                stats = data.get("stats", {})
                logger.info(
                    f"  [TCMM] post_response OK — "
                    f"step={stats.get('current_step', 0)}, "
                    f"archive={stats.get('archive_blocks', 0)}, "
                    f"{stats.get('elapsed_ms', 0)}ms"
                )
                return answer
    except Exception as e:
        logger.warning(f"  [TCMM] post_response error: {e}")
    return None


def _is_chat_completion(remaining_path: str, method: str) -> bool:
    """Check if this is a chat completions request (the path TCMM should intercept)."""
    if method != "POST":
        return False
    # OpenAI/Gemini format
    if "chat/completions" in remaining_path:
        return True
    # Anthropic format
    if remaining_path.rstrip("/").endswith("v1/messages"):
        return True
    return False


def _is_anthropic_format(remaining_path: str) -> bool:
    """Check if this request uses Anthropic message format."""
    return "v1/messages" in remaining_path


# ── Anthropic prompt caching ─────────────────────────────────────────────────
# Anthropic supports `cache_control: {"type": "ephemeral"}` on any content
# block. The API caches all tokens up to the last block carrying the marker.
# On a subsequent call with a byte-identical prefix, `cache_read_input_tokens`
# in the response usage reports a hit (5–10× cheaper than reprocessing).
#
# Minimum cacheable segment is ~1024 tokens (~4000 chars). Up to 4 markers
# allowed. Any byte change in the cached prefix invalidates it.

_MIN_CACHE_CHARS = 4000


def _split_for_cache(prompt: str, marker: str | None = None) -> tuple[str | None, str]:
    """Split a prompt at a boundary marker for KV-cache reuse.

    Returns (prefix, tail) when the marker exists and the prefix is long
    enough to be worth caching; (None, prompt) otherwise — caller should
    send the prompt as a plain string in that case.

    The prefix ends just before the marker; the tail starts at the marker.
    """
    if not marker or not isinstance(prompt, str):
        return (None, prompt)
    idx = prompt.find(marker)
    if idx == -1 or idx < _MIN_CACHE_CHARS:
        return (None, prompt)
    return (prompt[:idx], prompt[idx:])


def _apply_anthropic_cache(data: dict) -> int:
    """Add cache_control markers to the Anthropic request body.

    Caches the system message (stable prefix containing TCMM memory +
    Veilguard prompt) and, when the conversation history is long, the
    penultimate message block. Leaves the latest user turn uncached.

    Returns the number of cache_control markers added.
    """
    markers = 0

    # 1. System message — the biggest stable chunk.
    system = data.get("system")
    if isinstance(system, str) and len(system) >= _MIN_CACHE_CHARS:
        data["system"] = [{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }]
        markers += 1
    elif isinstance(system, list) and system:
        # Already structured. Mark the last text block if the total is long enough.
        total_len = sum(
            len(b.get("text", "")) for b in system
            if isinstance(b, dict) and b.get("type") == "text"
        )
        if total_len >= _MIN_CACHE_CHARS:
            for blk in reversed(system):
                if isinstance(blk, dict) and blk.get("type") == "text":
                    blk.setdefault("cache_control", {"type": "ephemeral"})
                    markers += 1
                    break

    # 2. Conversation history — cache everything up to (but not including)
    #    the last user message. Only worth it if there are 3+ prior turns.
    messages = data.get("messages") or []
    if isinstance(messages, list) and len(messages) >= 4 and markers < 4:
        # Find the final user message; cache the message just before it.
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], dict) and messages[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx >= 2:
            target = messages[last_user_idx - 1]
            if isinstance(target, dict):
                content = target.get("content")
                # Normalise string → list so we can attach cache_control
                if isinstance(content, str):
                    if len(content) >= 200:  # tiny content isn't worth it
                        target["content"] = [{
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"},
                        }]
                        markers += 1
                elif isinstance(content, list) and content:
                    total_len = sum(
                        len(b.get("text", "")) for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                    if total_len >= 200:
                        for blk in reversed(content):
                            if isinstance(blk, dict) and blk.get("type") == "text":
                                blk.setdefault("cache_control", {"type": "ephemeral"})
                                markers += 1
                                break

    return markers


def _log_cache_metrics(usage: dict, context: str = ""):
    """Log Anthropic cache hit/miss from response usage field.

    `cache_creation_input_tokens`: tokens written to the cache (miss on first call).
    `cache_read_input_tokens`: tokens served from cache (hit on subsequent calls).
    """
    if not isinstance(usage, dict):
        return
    create = usage.get("cache_creation_input_tokens", 0) or 0
    read = usage.get("cache_read_input_tokens", 0) or 0
    inp = usage.get("input_tokens", 0) or 0
    out = usage.get("output_tokens", 0) or 0
    ctx = f" {context}" if context else ""
    logger.info(
        f"  [CACHE]{ctx} create={create} read={read} input={inp} output={out}"
    )


import re as _re

_HEATMAP_RE = _re.compile(
    r'\s*\{["\s]*knowledge_class["\s]*:.*?\}\s*$',
    _re.DOTALL
)


def _strip_heatmap_from_text(text: str) -> str:
    """Remove TCMM heatmap JSON from response text (keep for TCMM, strip for user)."""
    return _HEATMAP_RE.sub('', text).strip()


# Stub for Anthropic model listing — LibreChat calls this during auto-discovery
@app.get("/anthropic/v1/models")
async def anthropic_models():
    """Return available Anthropic models (stub for LibreChat model discovery)."""
    return {
        "data": [
            {"id": "claude-sonnet-4-6", "object": "model", "created": 1700000000, "owned_by": "anthropic"},
            {"id": "claude-3-haiku-20240307", "object": "model", "created": 1700000000, "owned_by": "anthropic"},
        ],
        "object": "list",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "veilguard-pii-gateway",
        "presidio": "active",
        "backends": list(BACKENDS.keys()),
    }


def extract_conversation_id(data: dict, headers: dict) -> str:
    """Extract conversation ID for TCMM session tracking.

    LibreChat sends conversationId="new" on the first message of a new chat.
    We skip that and fall through to generate a stable temporary ID.
    On the second message, LibreChat sends the real UUID.
    """
    _skip = {"", "new", "null", "undefined", "None"}

    # 1. Explicit headers
    conv_id = headers.get("x-conversation-id") or headers.get("x-request-id")
    if conv_id and conv_id not in _skip:
        return conv_id

    # 2. From Anthropic metadata (LibreChat patched)
    metadata = data.get("metadata", {})
    conv_id = metadata.get("conversation_id", "")
    if conv_id and conv_id not in _skip:
        return conv_id

    # 3. From request body
    conv_id = data.get("conversationId") or data.get("conversation_id") or ""
    if conv_id and conv_id not in _skip:
        return conv_id

    # 4. parent_message_id (stable across the conversation)
    parent_id = data.get("parentMessageId") or data.get("parent_message_id") or ""
    if parent_id and parent_id not in _skip:
        return f"parent-{parent_id[:24]}"

    # 5. Fallback: unique per request (isolates orphan first messages)
    user_id = metadata.get("user_id", "")
    if user_id:
        return f"new-{user_id[:16]}-{uuid.uuid4().hex[:8]}"

    return str(uuid.uuid4())


def extract_pii_session_id(data: dict) -> str:
    """Extract PII session ID — always per-user so token mappings are consistent
    across all conversations for the same user. This ensures REF_PERSON_2 always
    maps to the same person regardless of which conversation it appears in."""
    metadata = data.get("metadata", {})
    user_id = metadata.get("user_id", "")
    if user_id:
        return f"pii-{user_id[:24]}"
    return "pii-default"


def resolve_backend(path: str) -> tuple[str | None, str, str]:
    """Parse path to find backend and remaining path.

    Returns: (backend_url, remaining_path, backend_name)
    """
    parts = path.strip("/").split("/", 1)
    backend_name = parts[0].lower()
    remaining = parts[1] if len(parts) > 1 else ""

    if backend_name in BACKENDS:
        return BACKENDS[backend_name], remaining, backend_name

    return None, path, ""


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def gateway(request: Request, path: str):
    """Universal PII-redacting gateway.

    Routes /<backend>/... to the appropriate LLM API.
    Redacts PII from requests, rehydrates PII in responses.
    """
    backend_url, remaining_path, backend_name = resolve_backend(path)

    if not backend_url:
        return JSONResponse(
            {
                "error": f"Unknown backend: '{path.split('/')[0]}'. Available: {list(BACKENDS.keys())}",
                "usage": "Use /<backend>/... where backend is: anthropic, openai, gemini",
            },
            status_code=404,
        )

    target_url = f"{backend_url}/{remaining_path}" if remaining_path else backend_url

    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    redactor = get_redactor()
    is_stream = False
    conversation_id = None
    pii_session_id = "pii-default"  # Per-user PII session (consistent across conversations)

    # Redact PII from JSON request body
    if body:
        content_type = headers.get("content-type", "")
        if "json" in content_type or body.strip()[:1] == b"{":
            try:
                data = json.loads(body)
                is_stream = data.get("stream", False)
                conversation_id = extract_conversation_id(data, headers)
                pii_session_id = extract_pii_session_id(data)
                tcmm_user_id = data.get("metadata", {}).get("user_id", "")

                # Strip conversation_id from metadata before forwarding to LLM API
                # (Anthropic only allows user_id in metadata — extra fields cause 400)
                metadata = data.get("metadata")
                if isinstance(metadata, dict) and "conversation_id" in metadata:
                    del metadata["conversation_id"]

                logger.info(
                    f">>> {request.method} [{backend_name}] /{remaining_path} "
                    f"(stream={is_stream}, conv={conversation_id[:8]}...)"
                )

                # ── TCMM Integration ──
                tcmm_active = False
                if TCMM_ENABLED and _is_chat_completion(remaining_path, request.method):
                    messages = data.get("messages", [])
                    # Detect if THIS request is a tool result follow-up
                    # Only check the LAST message — not the full history
                    # (History may contain old tool_use from previous turns)
                    is_tool_followup = False
                    if messages:
                        last_msg = messages[-1]
                        # OpenAI: last message role="tool"
                        if last_msg.get("role") == "tool":
                            is_tool_followup = True
                        # Anthropic: last message contains tool_result blocks
                        last_content = last_msg.get("content", [])
                        if isinstance(last_content, list):
                            for block in last_content:
                                if isinstance(block, dict) and block.get("type") == "tool_result":
                                    is_tool_followup = True
                                    break
                        # Also: second-to-last is assistant with tool_use (awaiting result)
                        if len(messages) >= 2:
                            prev_msg = messages[-2]
                            if prev_msg.get("role") == "assistant":
                                prev_content = prev_msg.get("content", [])
                                if isinstance(prev_content, list):
                                    for block in prev_content:
                                        if isinstance(block, dict) and block.get("type") == "tool_use":
                                            is_tool_followup = True
                                            break

                    if is_tool_followup:
                        # Tool result follow-up — pass through untouched
                        logger.info(f"  [TCMM] Skipping — tool result follow-up")
                    else:
                        user_msg = _extract_last_user_message(messages)
                        if user_msg:
                            tcmm_context = await _tcmm_pre_request(user_msg, conversation_id, user_id=tcmm_user_id)
                            if tcmm_context:
                                # Inject TCMM context RAW (with real PII).
                                # redact_json below handles the ENTIRE payload in one pass,
                                # giving consistent PII tokens across system + messages.
                                if tcmm_context.strip():
                                    if _is_anthropic_format(remaining_path):
                                        # Anthropic format: inject into "system" field (string, not message)
                                        existing_system = data.get("system", "")
                                        data["system"] = f"{tcmm_context}\n\n{existing_system}".strip()
                                        # Keep messages as-is (Anthropic doesn't use system role in messages)
                                    else:
                                        # OpenAI/Gemini format: inject as system message
                                        new_messages = []
                                        new_messages.append({"role": "system", "content": tcmm_context})
                                        new_messages.append({"role": "user", "content": user_msg})
                                        data["messages"] = new_messages

                                tcmm_active = True
                                logger.info(f"  [TCMM] Memory={len(tcmm_context)} chars, format={'anthropic' if _is_anthropic_format(remaining_path) else 'openai'}")

                # Inject Veilguard system prompt for Anthropic requests
                if _is_anthropic_format(remaining_path):
                    veilguard_prompt = (
                        "You are Veilguard, a Phishield AI cybersecurity assistant.\n\n"
                        "STYLE RULES (mandatory):\n"
                        "- Be concise and direct. Lead with the answer, not reasoning.\n"
                        "- Do NOT use emojis under any circumstances.\n"
                        "- Do NOT use filler phrases (Sure!, Great question!, I'd be happy to help!, etc).\n"
                        "- Do NOT give time estimates or predictions.\n"
                        "- Do NOT add unrequested features or improvements.\n"
                        "- Keep responses short. One sentence beats three.\n"
                        "- Use markdown for structure. Reference files as path:line.\n"
                        "- When the user provides information, acknowledge briefly and move on.\n"
                        "- Do NOT call tools (scratchpad_write, etc) when user is just sharing info.\n"
                    )
                    existing_system = data.get("system", "")
                    data["system"] = f"{veilguard_prompt}\n{existing_system}".strip()

                    # Apply prompt caching markers (system + conversation history)
                    _cache_markers = _apply_anthropic_cache(data)
                    if _cache_markers:
                        logger.info(f"  [CACHE] applied {_cache_markers} cache_control marker(s)")

                # Redact PII
                redacted = redactor.redact_json(data, pii_session_id)
                body = json.dumps(redacted, ensure_ascii=False).encode("utf-8")
                headers["content-length"] = str(len(body))

                # Audit log: what we're sending to the LLM (redacted)
                _redacted_messages = redacted.get("messages", [])
                _redacted_system = redacted.get("system", "")
                _audit_text = ""
                if _redacted_system:
                    _audit_text += f"[SYSTEM] {_redacted_system[:500]}\n\n"
                for _m in _redacted_messages[-3:]:  # Last 3 messages
                    _role = _m.get("role", "?")
                    _content = _m.get("content", "")
                    if isinstance(_content, list):
                        _content = " ".join(str(b.get("text", b.get("content", "")))[:200] for b in _content if isinstance(b, dict))
                    _audit_text += f"[{_role.upper()}] {str(_content)[:500]}\n"
                audit_log("TO_LLM", conversation_id, _audit_text, f"model={redacted.get('model','?')}")

            except json.JSONDecodeError:
                logger.info(f">>> {request.method} [{backend_name}] /{remaining_path} (non-json)")
        else:
            logger.info(f">>> {request.method} [{backend_name}] /{remaining_path}")

    if is_stream:
        # For streaming: create client that lives as long as the generator
        client = httpx.AsyncClient(timeout=300)
        req = client.build_request(
            method=request.method, url=target_url,
            content=body, headers=headers,
        )
        response = await client.send(req, stream=True)

        # If upstream returned an error, read the body and log it
        if response.status_code >= 400:
            error_body = await response.aread()
            await response.aclose()
            await client.aclose()
            logger.error(f"<<< [{backend_name}] {response.status_code}: {error_body[:500]}")
            return JSONResponse(
                json.loads(error_body) if error_body else {"error": f"Upstream {response.status_code}"},
                status_code=response.status_code,
            )

        is_anthropic = _is_anthropic_format(remaining_path)

        async def stream_with_rehydration():
            """Stream through, rehydrating PII tokens in each chunk.
            If TCMM is active: collect all content, strip heatmap, re-emit clean SSE."""
            if not tcmm_active and not is_anthropic:
                # Normal non-TCMM, non-Anthropic path: stream through with rehydration
                try:
                    async for chunk in response.aiter_bytes():
                        if conversation_id:
                            try:
                                text = chunk.decode("utf-8")
                                text = redactor.rehydrate_text(text, pii_session_id)
                                yield text.encode("utf-8")
                            except UnicodeDecodeError:
                                yield chunk
                        else:
                            yield chunk
                except (httpx.ReadError, httpx.RemoteProtocolError) as e:
                    logger.warning(f"Stream ended: {e}")
                finally:
                    try:
                        await response.aclose()
                        await client.aclose()
                    except Exception:
                        pass
                return

            # Anthropic streaming: collect ALL events, strip heatmap from
            # COMBINED text (not per-delta), re-emit clean SSE.
            # The heatmap can be split across multiple deltas — only stripping
            # the combined text is reliable.
            if is_anthropic:
                all_events = []       # list of raw SSE event strings
                all_content_text = "" # combined text from all deltas (for TCMM)
                sse_buf = ""
                _rehydration_count = 0
                _cache_usage = {}     # accumulated cache/token usage from message_start + message_delta

                try:
                    async for chunk in response.aiter_bytes():
                        text = chunk.decode("utf-8", errors="replace")
                        if conversation_id:
                            before = text
                            text = redactor.rehydrate_text(text, pii_session_id)
                            if text != before:
                                _rehydration_count += 1
                                logger.info(f"  [REHYDRATE] chunk changed ({_rehydration_count}x)")
                        sse_buf += text

                        while "\n\n" in sse_buf:
                            event_str, sse_buf = sse_buf.split("\n\n", 1)
                            all_events.append(event_str + "\n\n")

                            # Track content text for TCMM + capture cache usage
                            for line in event_str.split("\n"):
                                line = line.strip()
                                if line.startswith("data: "):
                                    try:
                                        evt = json.loads(line[6:])
                                        etype = evt.get("type")
                                        if etype == "content_block_delta" and evt.get("delta", {}).get("type") == "text_delta":
                                            all_content_text += evt["delta"]["text"]
                                        elif etype == "message_start":
                                            # Contains input_tokens + cache_creation/read_input_tokens
                                            u = (evt.get("message") or {}).get("usage") or {}
                                            for k, v in u.items():
                                                _cache_usage[k] = v
                                        elif etype == "message_delta":
                                            # Contains final output_tokens
                                            u = evt.get("usage") or {}
                                            for k, v in u.items():
                                                _cache_usage[k] = v
                                    except (json.JSONDecodeError, ValueError):
                                        pass

                except (httpx.ReadError, httpx.RemoteProtocolError) as e:
                    logger.warning(f"Anthropic stream ended: {e}")

                # Handle remaining buffer
                if sse_buf.strip():
                    all_events.append(sse_buf)
                    for line in sse_buf.split("\n"):
                        line = line.strip()
                        if line.startswith("data: "):
                            try:
                                evt = json.loads(line[6:])
                                if evt.get("delta", {}).get("type") == "text_delta":
                                    all_content_text += evt["delta"]["text"]
                            except (json.JSONDecodeError, ValueError):
                                pass

                # Log cache hit/miss metrics (collected from message_start + message_delta)
                if _cache_usage:
                    _log_cache_metrics(_cache_usage,
                                       context=f"conv={conversation_id[:8] if conversation_id else '?'} stream=true")

                # Strip heatmap from the COMBINED content text
                clean_content = _strip_heatmap_from_text(all_content_text)
                heatmap_stripped = clean_content != all_content_text

                if heatmap_stripped:
                    # Rebuild ALL events with heatmap removed from content deltas
                    # Walk through events, reconstruct content from clean_content
                    clean_pos = 0
                    for event_str in all_events:
                        rebuilt_lines = []
                        skip_event = False
                        for line in event_str.split("\n"):
                            stripped_line = line.strip()
                            if stripped_line.startswith("data: "):
                                try:
                                    evt = json.loads(stripped_line[6:])
                                    if evt.get("type") == "content_block_delta" and evt.get("delta", {}).get("type") == "text_delta":
                                        original_text = evt["delta"]["text"]
                                        orig_len = len(original_text)
                                        # Map this delta's portion from clean_content
                                        remaining_clean = clean_content[clean_pos:]
                                        # Take up to orig_len chars from clean, or whatever is left
                                        chunk_text = remaining_clean[:orig_len]
                                        clean_pos += len(chunk_text)
                                        if chunk_text.strip():
                                            evt["delta"]["text"] = chunk_text
                                            rebuilt_lines.append(f"data: {json.dumps(evt)}")
                                        else:
                                            skip_event = True
                                    else:
                                        rebuilt_lines.append(stripped_line)
                                except (json.JSONDecodeError, ValueError):
                                    rebuilt_lines.append(stripped_line)
                            elif stripped_line.startswith("event:") or stripped_line == "":
                                rebuilt_lines.append(stripped_line)
                            elif stripped_line:
                                rebuilt_lines.append(stripped_line)

                        if not skip_event and rebuilt_lines:
                            rebuilt = "\n".join(rebuilt_lines) + "\n\n"
                            yield rebuilt.encode("utf-8")
                    logger.info(f"  Heatmap stripped from Anthropic stream")
                else:
                    # No heatmap — emit all events as-is
                    for event_str in all_events:
                        yield event_str.encode("utf-8")

                # Audit log: what the LLM returned
                audit_log("FROM_LLM", conversation_id, all_content_text or "(empty)", "stream=anthropic")

                # Feed full content (WITH heatmap) to TCMM for learning
                if tcmm_active and all_content_text:
                    await _tcmm_post_response(all_content_text, conversation_id, user_id=tcmm_user_id)
                    logger.info(f"  [TCMM] Anthropic stream done, ingested {len(all_content_text)} chars")

                try:
                    await response.aclose()
                    await client.aclose()
                except Exception:
                    pass
                return

            # TCMM path: PARSE-AND-RECONSTRUCT approach.
            #
            # Problem: aiter_bytes() gives raw TCP chunks that can contain
            # multiple SSE events or split events across chunks. We can't
            # treat raw chunks as atomic SSE events.
            #
            # Solution: Parse ALL incoming bytes into individual SSE events.
            # Each event has a content delta (or not). We maintain a FIFO
            # of parsed SSE events and only yield events that are far enough
            # ahead of the tail. When the stream ends, we inspect the tail
            # for heatmap JSON and strip it.
            #
            # The heatmap is ALWAYS the last content in the response:
            #   {"knowledge_class": "...", "used": {...}}

            HOLD_BACK = 30  # hold back last N SSE events

            sse_events = []    # list of (raw_sse_line, content_str_or_None)
            all_content = []   # all content strings for TCMM
            sse_buffer = ""    # partial SSE line accumulator

            try:
                async for chunk in response.aiter_bytes():
                    try:
                        text = chunk.decode("utf-8")
                    except UnicodeDecodeError:
                        continue

                    if conversation_id:
                        text = redactor.rehydrate_text(text, pii_session_id)

                    # Accumulate and split into SSE lines
                    sse_buffer += text
                    lines = sse_buffer.split("\n")
                    # Last element is incomplete (or empty) — keep for next chunk
                    sse_buffer = lines[-1]

                    for raw_line in lines[:-1]:
                        stripped = raw_line.strip()
                        if not stripped:
                            continue  # skip empty separator lines

                        content_str = None
                        if stripped.startswith("data: ") and stripped != "data: [DONE]":
                            try:
                                payload = json.loads(stripped[6:])
                                delta = payload.get("choices", [{}])[0].get("delta", {})
                                if "content" in delta:
                                    content_str = delta["content"]
                                    all_content.append(content_str)
                            except (json.JSONDecodeError, IndexError, KeyError):
                                pass

                        # Store with proper SSE framing: data line + blank line
                        sse_events.append((raw_line + "\n\n", content_str))

                        # Yield old events that are safely ahead of the tail
                        while len(sse_events) > HOLD_BACK:
                            old_line, _ = sse_events.pop(0)
                            yield old_line.encode("utf-8")

            except (httpx.ReadError, httpx.RemoteProtocolError) as e:
                logger.warning(f"Stream ended: {e}")
            finally:
                try:
                    await response.aclose()
                    await client.aclose()
                except Exception:
                    pass

                # Process any remaining partial line
                if sse_buffer.strip():
                    content_str = None
                    stripped = sse_buffer.strip()
                    if stripped.startswith("data: ") and stripped != "data: [DONE]":
                        try:
                            payload = json.loads(stripped[6:])
                            delta = payload.get("choices", [{}])[0].get("delta", {})
                            if "content" in delta:
                                content_str = delta["content"]
                                all_content.append(content_str)
                        except (json.JSONDecodeError, IndexError, KeyError):
                            pass
                    sse_events.append((sse_buffer + "\n\n", content_str))

                # Now inspect for heatmap.
                # Strategy: find the trailing heatmap JSON in the FULL content,
                # figure out how many chars to strip from the end, then
                # reconstruct SSE events from the held-back buffer, suppressing
                # events whose content falls within the heatmap region.
                full_content = "".join(all_content)

                # Find trailing JSON: scan backwards for last "{" that opens
                # a valid JSON dict with knowledge_class or used.
                heatmap_start = -1
                search_from = len(full_content)
                while search_from > 0:
                    pos = full_content.rfind("{", 0, search_from)
                    if pos < 0:
                        break
                    candidate_str = full_content[pos:].strip()
                    try:
                        candidate = json.loads(candidate_str)
                        if isinstance(candidate, dict) and (
                            "knowledge_class" in candidate or "used" in candidate
                        ):
                            heatmap_start = pos
                            break
                    except (json.JSONDecodeError, ValueError):
                        pass
                    search_from = pos  # try earlier {

                if heatmap_start >= 0:
                    # Found heatmap at position `heatmap_start` in full_content.
                    # Content from already-yielded events + held events = full_content.
                    # Already-yielded content length:
                    held_content_len = sum(
                        len(cs) for _, cs in sse_events if cs is not None
                    )
                    already_yielded_len = len(full_content) - held_content_len

                    # Heatmap starts at `heatmap_start` in full_content.
                    # In the held-back content, it starts at:
                    heatmap_offset_in_held = heatmap_start - already_yielded_len

                    if heatmap_offset_in_held < 0:
                        # Heatmap partially in already-yielded content — can't fix
                        logger.warning(
                            f"  [TCMM] Heatmap partially in yielded content "
                            f"(offset={heatmap_offset_in_held}), cannot strip fully"
                        )
                        for line, _ in sse_events:
                            yield line.encode("utf-8")
                    else:
                        # Walk through held events, tracking content position.
                        # Yield events before the heatmap; suppress the rest.
                        char_pos = 0
                        cut_idx = len(sse_events)
                        for idx, (_, cs) in enumerate(sse_events):
                            if cs is not None:
                                if char_pos + len(cs) > heatmap_offset_in_held:
                                    cut_idx = idx
                                    break
                                char_pos += len(cs)

                        # The event at cut_idx may contain BOTH answer and heatmap.
                        # Yield events before cut_idx as-is.
                        for i in range(cut_idx):
                            line, _ = sse_events[i]
                            yield line.encode("utf-8")

                        # For the cut event, if it has mixed content (answer + heatmap),
                        # emit a modified SSE event with only the answer portion.
                        if cut_idx < len(sse_events):
                            _, cut_cs = sse_events[cut_idx]
                            if cut_cs is not None:
                                # How many chars of this event are answer (not heatmap)
                                answer_chars = heatmap_offset_in_held - char_pos
                                if answer_chars > 0:
                                    clean_part = cut_cs[:answer_chars].rstrip()
                                    if clean_part:
                                        # Build a synthetic SSE event
                                        raw_line, _ = sse_events[cut_idx]
                                        try:
                                            # Parse original SSE, replace content
                                            for seg in raw_line.split("\n"):
                                                seg_s = seg.strip()
                                                if seg_s.startswith("data: ") and seg_s != "data: [DONE]":
                                                    obj = json.loads(seg_s[6:])
                                                    obj["choices"][0]["delta"]["content"] = clean_part
                                                    yield f"data: {json.dumps(obj)}\n\n".encode("utf-8")
                                                    break
                                        except Exception:
                                            pass  # skip if rewrite fails

                        suppressed = len(sse_events) - cut_idx
                        heatmap_text = full_content[heatmap_start:]
                        logger.info(
                            f"  [TCMM] Stripped heatmap from stream "
                            f"({len(heatmap_text)} chars, "
                            f"suppressed {suppressed} SSE events)"
                        )

                    # Send [DONE] so client knows stream ended
                    yield "data: [DONE]\n\n".encode("utf-8")
                else:
                    # No heatmap — flush all held-back events
                    for line, _ in sse_events:
                        yield line.encode("utf-8")

                # Audit log: what the LLM returned
                audit_log("FROM_LLM", conversation_id, full_content or "(empty)", "stream=openai")

                # Feed content to TCMM for learning
                if all_content:
                    await _tcmm_post_response(full_content, conversation_id, user_id=tcmm_user_id)
                    logger.info(f"  [TCMM] Stream done, ingested {len(full_content)} chars")

        resp_headers = {
            k: v for k, v in response.headers.items()
            if k.lower() not in ("content-encoding", "transfer-encoding", "content-length")
        }
        return StreamingResponse(
            stream_with_rehydration(),
            status_code=response.status_code,
            headers=resp_headers,
            media_type=response.headers.get("content-type", "text/event-stream"),
        )
    else:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.request(
                method=request.method, url=target_url,
                content=body if body else None, headers=headers,
            )

            logger.info(f"<<< [{backend_name}] {response.status_code} /{remaining_path}")

            # Rehydrate PII in response (must use same session ID as redaction)
            resp_body = response.content
            if pii_session_id:
                try:
                    resp_text = resp_body.decode("utf-8")
                    resp_text = redactor.rehydrate_text(resp_text, pii_session_id)
                    resp_body = resp_text.encode("utf-8")
                except UnicodeDecodeError:
                    pass

            # ── TCMM post-response + heatmap stripping ──
            if resp_body:
                try:
                    resp_json = json.loads(resp_body.decode("utf-8"))
                    raw_content = ""

                    # Extract content from OpenAI or Anthropic format
                    is_anthropic_resp = "content" in resp_json and isinstance(resp_json.get("content"), list) and resp_json.get("type") == "message"
                    if is_anthropic_resp:
                        # Anthropic format: {"content": [{"type": "text", "text": "..."}]}
                        blocks = resp_json.get("content", [])
                        raw_content = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                        # Log cache hit/miss metrics
                        _log_cache_metrics(resp_json.get("usage", {}),
                                           context=f"conv={conversation_id[:8] if conversation_id else '?'} stream=false")
                    else:
                        # OpenAI format: {"choices": [{"message": {"content": "..."}}]}
                        choices = resp_json.get("choices", [])
                        if choices:
                            raw_content = choices[0].get("message", {}).get("content", "")

                    if raw_content:
                        # Audit log
                        audit_log("FROM_LLM", conversation_id, raw_content or "(empty)", "stream=false")

                        # Feed REAL content to TCMM (no redaction — private local storage)
                        if tcmm_active:
                            clean_answer = await _tcmm_post_response(raw_content, conversation_id, user_id=tcmm_user_id)
                            logger.info(f"  [TCMM] Non-stream response processed")

                        # Strip heatmap from user-visible response
                        stripped = _strip_heatmap_from_text(raw_content)
                        if stripped != raw_content:
                            if is_anthropic_resp:
                                resp_json["content"] = [{"type": "text", "text": stripped}]
                            else:
                                if resp_json.get("choices"):
                                    resp_json["choices"][0]["message"]["content"] = stripped
                            logger.info(f"  Heatmap stripped from non-stream response")

                    resp_body = json.dumps(resp_json, ensure_ascii=False).encode("utf-8")
                except Exception as e:
                    logger.warning(f"  [TCMM] post-response parse error: {e}")

            resp_headers = {
                k: v for k, v in response.headers.items()
                if k.lower() not in ("content-encoding", "transfer-encoding", "content-length")
            }
            return StreamingResponse(
                iter([resp_body]),
                status_code=response.status_code,
                headers=resp_headers,
            )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
