"""
VEILGUARD PII Gateway
======================
Multi-LLM PII redaction gateway. Sits between LibreChat and any LLM API.

Routes:
  /anthropic/*  → https://api.anthropic.com/*
  /openai/*     → https://api.openai.com/*
  /gemini/*     → https://generativelanguage.googleapis.com/*
  /xai/*        → https://api.x.ai/*

All user-authored content is scanned for PII before forwarding.
All responses are rehydrated (PII tokens → original values) before returning.

LibreChat config:
  ANTHROPIC_BASE_URL=http://pii-proxy:4000/anthropic
  OPENAI_BASE_URL=http://pii-proxy:4000/openai/v1
  (Google: custom endpoint with baseURL http://pii-proxy:4000/gemini)
  (xAI:    custom endpoint with baseURL http://pii-proxy:4000/xai/v1)
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

# [PII_UNIFY_2026_05_29]  ONE live redactor.  The proxy and the
# agent-runtime now share `veilguard.pii` (Presidio + Lance-backed,
# fail-closed, block-cached) instead of the proxy keeping its own
# uncached `app/redactor.py` + in-memory `app/session.py`.  Same module,
# same shared Lance session store → byte-identical redaction across the
# LibreChat path and the multi-agent path.  See PII_FAST_REDACTION_SPEC.md.
# Make `from pii import ...` resolve from BOTH local dev (repo root) and
# docker (/pii mounted as a sibling of /app) — mirrors chat_agent_handler.
_HERE = Path(__file__).resolve()
for _cand in [*_HERE.parents, Path("/")]:
    if (_cand / "pii" / "__init__.py").is_file():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break
from pii import get_redactor, RedactionUnavailable, get_store as _get_pii_store

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
    """Write to the PII audit log. direction: 'TO_LLM' or 'FROM_LLM'.

    No truncation — the full payload needs to land both here (for
    tail -f debugging) and in the LanceDB audit table for replay /
    query.  Previous 3000-char cap hid TCMM memory + tool histories.
    """
    if _audit_logger:
        _audit_logger.info(
            f"\n{'='*80}\n"
            f"[{direction}] conv={conv_id[:12] if conv_id else '?'} {extra}\n"
            f"{'─'*80}\n"
            f"{content}\n"
            f"{'='*80}"
        )

PORT = int(os.environ.get("PII_PROXY_PORT", "4000"))
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.7"))

# TCMM Integration
TCMM_ENABLED = os.environ.get("TCMM_ENABLED", "false").lower() in ("true", "1", "yes")

# ── agent-runtime forwarding (Phase 0.1 — 2026-05-22) ───────────────────
# When AGENT_RUNTIME_ENABLED=true, Anthropic-bound chat-completion
# requests are forwarded to agent-runtime instead of going direct to
# api.anthropic.com.  agent-runtime owns the SDK loop + multi-agent
# orchestration; this proxy keeps doing redaction + audit + multi-
# provider routing for everything else (OpenAI / xAI / Gemini still
# go direct).
#
# Per-user allowlist for safe rollout: comma-separated AGENT_RUNTIME_USER_ALLOWLIST
# limits which user_ids get routed.  Empty = all users (when ENABLED=true).
# SSO models (claude-*-sso) bypass this; they have their own early-route.
AGENT_RUNTIME_ENABLED = os.environ.get("AGENT_RUNTIME_ENABLED", "false").lower() in ("true", "1", "yes")
AGENT_RUNTIME_URL = os.environ.get("AGENT_RUNTIME_URL", "http://agent-runtime:5000")
AGENT_RUNTIME_USER_ALLOWLIST = {
    u.strip()
    for u in os.environ.get("AGENT_RUNTIME_USER_ALLOWLIST", "").split(",")
    if u.strip()
}
# Default Director agent_id to dispatch with when no explicit override
# header is present.  Each conversation goes through the Director's
# reactive loop; ICs are spawned as SDK subagents internally.
AGENT_RUNTIME_DEFAULT_AGENT = os.environ.get("AGENT_RUNTIME_DEFAULT_AGENT", "director")
TCMM_URL = os.environ.get("TCMM_URL", "http://host.docker.internal:8811")
SUB_AGENTS_URL = os.environ.get("SUB_AGENTS_URL", "http://172.17.0.1:8809")
_VEILGUARD_INTERNAL_SECRET = os.environ.get("VEILGUARD_INTERNAL_SECRET", "")

# Backend routing table
BACKENDS = {
    "anthropic": os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com"),
    "openai": os.environ.get("OPENAI_API_URL", "https://api.openai.com"),
    "gemini": os.environ.get("GEMINI_API_URL", "https://generativelanguage.googleapis.com"),
    # xAI is OpenAI-compatible (Bearer auth, /v1/chat/completions shape).
    # Prompt caching is automatic server-side prefix caching — no client-
    # side cache_control markers like Anthropic. The Anthropic-specific
    # cache plumbing (extended-TTL beta header, multi-block cache_control
    # placement) is owned by TCMM's AnthropicRenderer and gated by
    # _is_anthropic_format so it never runs on xAI requests. See
    # https://docs.x.ai/developers/models/grok-4.3
    "xai": os.environ.get("XAI_API_URL", "https://api.x.ai"),
}

app = FastAPI(title="Veilguard PII Gateway")


@app.on_event("startup")
async def startup():
    logger.info("Loading Presidio NLP models...")
    redactor_inst = get_redactor(min_score=MIN_SCORE)
    # Pre-warm Presidio at startup. The first redact_text call would
    # otherwise pay ~600-1300ms of one-time cost (spacy model load,
    # NLP pipeline JIT-warm, allow_list compile) inside the user's
    # first request. By burning a dummy scan here the very first
    # real call already sees the warm path.
    try:
        import time as _pt
        _t = _pt.time()
        redactor_inst.redact_text(
            "Pre-warm probe: Alice met Bob at +27 21 123 4567 to discuss "
            "ID 8001015009087 and account 1234567890.",
            "_warmup_session_",
        )
        # [PII_LARGE_WARM_2026_05_30]  Also warm the LARGE-doc NER path so the
        # first real render (warm_batch over accumulated memory) doesn't pay
        # the one-time first-large-doc tax.  Zero recall cost.
        redactor_inst.warm()
        logger.info(
            f"Presidio pre-warm complete in {(_pt.time() - _t) * 1000:.0f}ms"
        )
    except Exception as e:
        logger.warning(f"Presidio pre-warm failed (non-fatal): {e}")
    logger.info("=" * 50)
    logger.info(f"Veilguard PII Gateway ready on port {PORT}")
    for name, url in BACKENDS.items():
        logger.info(f"  /{name}/* → {url}")
    if TCMM_ENABLED:
        logger.info(f"  TCMM: {TCMM_URL} (ENABLED)")
    else:
        logger.info(f"  TCMM: disabled")
    logger.info("=" * 50)


@app.on_event("shutdown")
async def shutdown():
    """Close the persistent TCMM HTTP client cleanly so connections drain.

    Without this, uvicorn shutdown leaves the connection pool dangling
    which surfaces as a deprecation warning on every reload. The
    operational impact is small but the warning clutters logs.
    """
    await _close_tcmm_client()
    logger.info("Veilguard shutdown: TCMM HTTP client closed")


# ── PII Rehydration Endpoint ─────────────────────────────────────────────────
# Used by sub-agents to rehydrate PII tokens in scratchpad/tool output for UI display

@app.post("/rehydrate")
async def rehydrate_endpoint(request: Request):
    """Rehydrate PII tokens in text. Called by sub-agents for scratchpad display."""
    body = await request.json()
    text = body.get("text", "")
    if not text:
        return {"text": ""}
    # [PII_UNIFY_2026_05_29]  Session-less best-effort rehydration via the
    # shared Lance store: looks up only the REF tokens PRESENT in `text`
    # (bounded by token count, not table size).  Replaces the old
    # in-memory `pii_store._store` scan.  If the caller supplies a
    # conversation/user, prefer the exact-session map (unambiguous).
    _conv = body.get("conversation_id") or request.headers.get("x-conversation-id")
    _user = body.get("user_id") or request.headers.get("x-user-id")
    _tenant = body.get("tenant_id") or request.headers.get("x-tenant-id") or _user
    store = _get_pii_store()
    if _conv or _user:
        from pii import SessionId as _SID
        result = store.rehydrate(_SID(tenant_id=_tenant or "_proxy", conv_id=_conv or _user), text)
        # Fall back to global if the scoped map didn't cover every token.
        if "REF_" in result:
            result = store.rehydrate_any(result)
    else:
        result = store.rehydrate_any(text)
    return {"text": result}


# ── TCMM Integration Helpers ─────────────────────────────────────────────────

def _extract_last_user_message(messages: list) -> str:
    """Extract the latest user-authored text from an OpenAI/Anthropic messages[] array.

    Skips tool_result wrappers. On Anthropic the last role=user message in a
    tool-followup turn contains ONLY tool_result blocks — that's a model
    echo, not a user turn, so we walk further back.
    """
    if not messages:
        return ""
    for msg in reversed(messages):
        role = msg.get("role")
        if role != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            # A user-role message is only a real user turn if it contains
            # at least one text/image block and no tool_result.
            has_tool_result = any(
                isinstance(p, dict) and p.get("type") == "tool_result"
                for p in content
            )
            if has_tool_result:
                continue  # keep looking further back for a real user turn
            text = " ".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ).strip()
            if text:
                return text
    return ""


# LibreChat fires a handful of synthetic, single-shot calls on the
# user's behalf — title generation, conversation summarisation, etc.
# They reach the proxy with a freshly-minted conversationId LibreChat
# never reuses, but our extract_conversation_id() can't tell the
# difference from a real first user turn, so each one (a) lands in
# pii_audit under its OWN ``conv-<userid>-<hash>`` row (fragmenting
# the dashboard view of what the user sees as one chat) AND (b) gets
# the full ~20-70 KB Veilguard preamble + TCMM render injected
# despite needing none of it (the model is being asked to summarise
# a 5-word title, not act on memory).
#
# Detection is purely string-prefix on the synthetic prompt text.
# All LibreChat's side-channel prompts are hardcoded English literals
# in its server source — we just match the unambiguous opening
# phrase. Returns the channel name for logging, or ``""`` for a real
# user turn.
_LIBRECHAT_SIDE_CHANNEL_PREFIXES = (
    # Default title-gen prompt (Anthropic + OpenAI flows). The exact
    # literal seen in audit row aid=3074:
    "Provide a concise, 5-word-or-less title for the conversation",
    # Alternate title prompts shipped by LibreChat for other locales /
    # endpoint configs:
    "Write a concise title for this conversation",
    "Please generate a title",
    # Summarisation (the auto-summary feature LibreChat runs when the
    # context window fills up — also a one-shot we don't want to
    # poison with TCMM rendering of itself):
    "Please summarize the conversation",
    "Write a concise summary of the conversation",
)


def _detect_librechat_side_channel(messages: list) -> str:
    """Return a non-empty channel label if this looks like a LibreChat
    synthetic call (title-gen, summary, etc.), else ``""``.

    Side-channel calls should bypass the TCMM pre_request / pin /
    render / ingest pipeline entirely — they are not part of the
    user's actual conversation and dragging them through TCMM both
    wastes tokens AND fragments the audit-dashboard's per-conv view.
    """
    last = _extract_last_user_message(messages)
    if not last:
        return ""
    head = last.lstrip()[:120]
    for prefix in _LIBRECHAT_SIDE_CHANNEL_PREFIXES:
        if head.startswith(prefix):
            # Return the first 30 chars of the matched prefix so the
            # log line is unique per channel without being verbose.
            return prefix[:30]
    return ""


def classify_message_origin(msg: dict) -> str:
    """Classify a single messages[] entry by its *real* origin.

    The Anthropic/OpenAI schema already tags tool traffic structurally;
    this helper just reads the envelope and returns one of:

      "user"           — human-authored text (role=user, no tool_result)
      "user_image"     — role=user message containing an image block
      "tool_result"    — role=user message whose content is tool output
                         (model-facing echo, NOT a real user turn)
      "assistant_text" — role=assistant text-only reply
      "tool_use"       — role=assistant containing a tool invocation
      "tool"           — OpenAI's role=tool (function-call response)
      "system"         — role=system (rare in messages[])
      "unknown"

    Use this to decide whether a message should be ingested into TCMM
    memory (ingest user/assistant_text, skip tool_result/tool_use/tool).
    """
    if not isinstance(msg, dict):
        return "unknown"
    role = msg.get("role")
    content = msg.get("content")

    # OpenAI function-calling: explicit role="tool"
    if role == "tool":
        return "tool"
    # OpenAI assistant with tool_calls[]
    if role == "assistant" and isinstance(msg.get("tool_calls"), list) and msg["tool_calls"]:
        return "tool_use"

    if role == "system":
        return "system"

    # Anthropic-style content blocks
    if isinstance(content, list):
        types = {
            blk.get("type") for blk in content
            if isinstance(blk, dict)
        }
        if role == "user":
            if "tool_result" in types:
                return "tool_result"
            if "image" in types:
                return "user_image"
            return "user"
        if role == "assistant":
            if "tool_use" in types:
                return "tool_use"
            return "assistant_text"

    # Plain-string content
    if isinstance(content, str):
        if role == "user":
            return "user"
        if role == "assistant":
            return "assistant_text"

    return "unknown"


def _is_tool_followup(messages: list) -> bool:
    """True when this request's latest turn is a tool-result being returned
    to the model for continuation — i.e. not a human turn. Used to skip
    TCMM ingestion on these purely mechanical hand-offs.

    Two signals:
      1. The last message itself is classified as tool_result or tool.
      2. The last message is role=user containing tool_result AND the
         prior assistant message held a tool_use (tool_use → tool_result
         round trip).
    """
    if not messages:
        return False
    last_origin = classify_message_origin(messages[-1])
    if last_origin in ("tool_result", "tool"):
        return True
    if len(messages) >= 2:
        prev_origin = classify_message_origin(messages[-2])
        if prev_origin == "tool_use" and last_origin == "tool_result":
            return True
    return False


class TCMMUnavailable(Exception):
    """Raised when TCMM /pre_request fails for ANY reason.

    2026-05-14: Veilguard now fails CLOSED on TCMM errors. The previous
    fail-open behaviour caused production prompts to ship without memory
    when TCMM hiccupped (PJ session, 05:44:24 UTC was one observed
    instance) — degrading answer quality silently. The proxy now returns
    HTTP 503 to the client so the failure surfaces immediately and the
    operator knows to investigate instead of debugging "why are the
    answers vague today" hours later.
    """
    pass


# ── Persistent httpx.AsyncClient for ALL TCMM calls ──────────────────────
#
# 2026-05-18: replaces 7 separate ``async with httpx.AsyncClient(...)``
# context managers across the TCMM helpers. Each new AsyncClient pays
# connection-pool init + DNS + (re)resolution cost — measured 1500ms+
# overhead vs tcmm-service's own 13ms processing time. With a persistent
# client we keep TCP connections alive across requests; cold call drops
# from ~1577ms wall-clock to <50ms.
#
# Lazily initialized so we don't construct it at import-time (FastAPI
# startup ordering can break that). Shared across all coroutines —
# httpx.AsyncClient is documented as concurrent-safe per-instance.
from typing import Optional as _Optional
_TCMM_HTTP_CLIENT: _Optional["httpx.AsyncClient"] = None


def _get_tcmm_client() -> "httpx.AsyncClient":
    """Return the process-wide TCMM HTTP client, creating on first use.

    Default timeout is 180s — Vertex-backed recall can take 60-90s on a
    cold user. Individual call sites can override via ``timeout=`` on
    the request method.
    """
    global _TCMM_HTTP_CLIENT
    if _TCMM_HTTP_CLIENT is None:
        _TCMM_HTTP_CLIENT = httpx.AsyncClient(
            timeout=180,
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=300.0,  # 5 min — covers idle gaps between turns
            ),
        )
    return _TCMM_HTTP_CLIENT


async def _close_tcmm_client() -> None:
    """Best-effort close on shutdown. Safe to call even if never initialized."""
    global _TCMM_HTTP_CLIENT
    if _TCMM_HTTP_CLIENT is not None:
        try:
            await _TCMM_HTTP_CLIENT.aclose()
        except Exception:
            pass
        _TCMM_HTTP_CLIENT = None




async def _tcmm_pre_request(
    user_message: str,
    conversation_id: str,
    user_id: str = "",
    origin: str = "user",
    lineage_parent_conv: str = "",
) -> str | None:
    """Call TCMM service to get enriched prompt. Returns None on failure.

    `origin` is the classified origin of the last user-role message —
    "user" for text, "user_image" for image attachments. TCMM stamps the
    stored block with this tag so role is recoverable from the archive.
    """
    # Timeout was 30s — Vertex-backed recall can legitimately take 60-90s
    # on a cold user (multiple embedding calls + Gemini Flash classifier +
    # fusion on 10+ candidates). When it timed out the proxy fell through
    # silently, the LLM got no memory, and the user saw "no memory blocks"
    # even though TCMM was computing the context perfectly in the
    # background. 180s gives headroom for the worst-case recall path;
    # typical is still <5s once the embedding cache warms.
    try:
        client = _get_tcmm_client()
        resp = await client.post(
            f"{TCMM_URL}/pre_request",
            json={
                "user_message": user_message,
                "conversation_id": conversation_id,
                "user_id": user_id,
                "origin": origin,
                # Sub-agent spawn lineage: when present, tells TCMM
                # "this conversation is a fork of <parent_conv>;
                # stamp lineage.parents[0] + lineage.root on my
                # first archive block". Empty for top-level
                # LibreChat turns. TCMM falls back to default
                # root-is-self if this is missing or the parent
                # namespace has no rows yet.
                "lineage_parent_conv": lineage_parent_conv,
            },
        )
        if resp.status_code == 200:
            data = resp.json()
            # 2026-05-18: explicit error wins. An empty prompt is a
            # LEGITIMATE response from tcmm-service for a fresh
            # conversation (no memory yet, no recall hits) — the
            # render path later will produce just the system /
            # contract content. Previously we treated ``prompt == ""``
            # as failure and 503-d every brand-new conversation;
            # that's worse than "no memory" because clients can't
            # even start a new chat. Only fail hard on an explicit
            # error field or a missing prompt key.
            if data.get("error"):
                error = data["error"]
                logger.error(f"  [TCMM] pre_request failed: {error}")
                raise TCMMUnavailable(f"TCMM pre_request failed: {error}")
            prompt = data.get("prompt")
            if prompt is None:
                logger.error(
                    "  [TCMM] pre_request returned no 'prompt' key — "
                    "contract violation"
                )
                raise TCMMUnavailable("TCMM pre_request missing 'prompt' key")
            stats = data.get("stats", {})
            logger.info(
                f"  [TCMM] pre_request OK — "
                f"recalled={stats.get('recalled', 0)}, "
                f"live={stats.get('live_blocks', 0)}, "
                f"shadow={stats.get('shadow_blocks', 0)}, "
                f"prompt_chars={len(prompt)} "
                f"{stats.get('elapsed_ms', 0)}ms"
            )
            return prompt
        else:
            logger.error(f"  [TCMM] pre_request HTTP {resp.status_code}")
            raise TCMMUnavailable(f"TCMM HTTP {resp.status_code}")
    except TCMMUnavailable:
        # Already a structured failure — let it propagate to the route handler.
        raise
    except httpx.ConnectError as _e:
        logger.error("  [TCMM] service unreachable — failing hard (no silent fallback)")
        raise TCMMUnavailable("TCMM service unreachable") from _e
    except httpx.ReadTimeout as _e:
        logger.error("  [TCMM] pre_request timed out — failing hard (no silent fallback)")
        raise TCMMUnavailable("TCMM pre_request timed out") from _e
    except Exception as _e:
        logger.error(f"  [TCMM] pre_request error: {type(_e).__name__}: {_e}")
        raise TCMMUnavailable(f"TCMM pre_request error: {type(_e).__name__}: {_e}") from _e
    # Defensive: every branch above either returns a prompt or raises.
    # If execution reaches here, TCMM's contract was violated — fail hard.
    raise TCMMUnavailable("TCMM pre_request completed without prompt or exception")


# ─── 2026-05-15 TCMM-renderer helpers ────────────────────────────────
#
# The proxy delegates ALL prompt assembly to TCMM:
#   1. /pin/system_prompt once per conversation — the Veilguard
#      preamble lands in the IMMUTABLE tier. Fingerprint-deduped on the
#      TCMM side so re-posting identical text is a no-op.
#   2. /render?model=anthropic|grok|openai per turn — returns
#      wire-format-ready blocks (via cache_control_strategy on the
#      provider-specific renderer). Anthropic gets multi-block list
#      with cache_control at tier boundaries; OpenAI/Grok get whatever
#      shape the renderer's cache_control_strategy produces (currently
#      a single block, but the proxy passes it through unchanged so any
#      future renderer split is honoured automatically).
#
# Proxy responsibilities: redact, relay TCMM's output into the
# provider-shaped JSON slot, forward to upstream. NO tier reasoning,
# NO cache_control assembly, NO preamble owning. If TCMM is down,
# /render and /pin raise TCMMUnavailable → 503 to client. No silent
# fallback path exists by design — degraded answers without memory
# are worse than a clear failure the operator can fix.

# In-process cache of pin keys we've already shipped. Key shape:
# ``"{conv_id}:{kind}:{sha256(content)[:16]}"``. The /pin endpoints
# fingerprint-dedup on the TCMM side regardless, so this set is purely
# a latency optimization — saves a 10-50ms round trip on every
# subsequent turn of the same conversation. Reset on process restart
# (which forces a re-pin call, server-side dedup makes that a no-op).
#
# 2026-05-18: was previously ``set[str]`` keyed on conv_id alone, which
# meant the FIRST pin in a conversation marked the conv "done" and
# subsequent calls (client system, tool defs) on the same conv silently
# skipped. Now keyed per-fingerprint so multi-pin per conv works.
import hashlib as _hashlib

_PINNED_KEYS: set[str] = set()


def _pin_cache_key(conv_id: str, kind: str, content: str) -> str:
    """Build the cache key used by ``_PINNED_KEYS``. Content fingerprint
    is sha256 prefix-truncated to 16 hex chars — collision-resistant
    enough for an in-process dedup set; cheaper than a full sha256."""
    h = _hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()[:16]
    return f"{conv_id}:{kind}:{h}"


async def _tcmm_pin_system_prompt(
    conv_id: str, text: str, kind: str = "veilguard_preamble",
    user_id: str = "",
) -> None:
    """Pin a system-prompt-class block for this conversation.

    ``kind`` distinguishes Veilguard's hardcoded preamble from
    LibreChat's per-conversation system prompt so both can coexist in
    the in-process dedup set without one starving the other. Both go to
    the same TCMM endpoint (``/pin/system_prompt``); ``kind`` is a
    proxy-side hint, not a wire field.

    Idempotent: returns immediately if the same (conv_id, kind, content)
    has been pinned in this process. Empty text is a no-op. Raises
    ``TCMMUnavailable`` on network / HTTP error — the caller's existing
    try/except returns 503 to the client.
    """
    if not text or not text.strip():
        return
    key = _pin_cache_key(conv_id, kind, text)
    if key in _PINNED_KEYS:
        return
    try:
        client = _get_tcmm_client()
        resp = await client.post(
            f"{TCMM_URL}/pin/system_prompt",
            json={
                "text": text,
                "conversation_id": conv_id,
                "user_id": user_id,
                "kind": kind,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            logger.error(
                f"  [TCMM] pin_system_prompt({kind}) HTTP {resp.status_code}: "
                f"{resp.text[:200]}"
            )
            raise TCMMUnavailable(
                f"TCMM pin_system_prompt({kind}) HTTP {resp.status_code}"
            )
        data = resp.json()
        logger.info(
            f"  [TCMM] pin_system_prompt({kind}): "
            f"block_id={data.get('block_id')} deduped={data.get('deduped')}"
        )
        _PINNED_KEYS.add(key)
    except TCMMUnavailable:
        raise
    except (httpx.ConnectError, httpx.ReadTimeout) as _e:
        logger.error(
            f"  [TCMM] pin_system_prompt({kind}) network error: "
            f"{type(_e).__name__}: {_e}"
        )
        raise TCMMUnavailable(
            f"TCMM pin_system_prompt({kind}): {type(_e).__name__}"
        ) from _e
    except Exception as _e:
        logger.error(
            f"  [TCMM] pin_system_prompt({kind}) error: {type(_e).__name__}: {_e}"
        )
        raise TCMMUnavailable(
            f"TCMM pin_system_prompt({kind}): {type(_e).__name__}: {_e}"
        ) from _e


async def _tcmm_pin_tool_definitions(
    conv_id: str, schemas: list, user_id: str = "",
) -> None:
    """Pin LibreChat's per-conversation tool schemas to TCMM's IMMUTABLE
    tier so they cache at 24h TTL instead of being re-billed as fresh
    input on every turn (xAI/OpenAI tool schemas are NOT prompt-cached
    when sent via ``data["tools"]`` — only the prompt prefix is).

    Each schema is one pinned block (TCMM-side); the schema-list
    fingerprint is used as the in-process dedup key, so a stable tool
    list re-pins zero times after turn 1. Empty list is a no-op. Raises
    ``TCMMUnavailable`` on network / HTTP error.

    Note: this PINs tools as cached text context. The proxy STILL sends
    ``data["tools"]`` to the upstream API — that field drives the
    actual function-call mechanism. The pinned text is parallel context
    that survives between turns at cache-read rates.
    """
    if not schemas:
        return
    # Build a stable fingerprint over the sorted-by-name schema list.
    try:
        import json as _json
        _norm = _json.dumps(schemas, sort_keys=True)
    except Exception:
        _norm = repr(schemas)
    key = _pin_cache_key(conv_id, "tool_defs", _norm)
    if key in _PINNED_KEYS:
        return
    try:
        client = _get_tcmm_client()
        resp = await client.post(
            f"{TCMM_URL}/pin/tool_definitions",
            json={
                "schemas": schemas,
                "conversation_id": conv_id,
                "user_id": user_id,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            logger.error(
                f"  [TCMM] pin_tool_definitions HTTP {resp.status_code}: "
                f"{resp.text[:200]}"
            )
            raise TCMMUnavailable(
                f"TCMM pin_tool_definitions HTTP {resp.status_code}"
            )
        data = resp.json()
        logger.info(
            f"  [TCMM] pin_tool_definitions: "
            f"block_ids={data.get('block_ids')} "
            f"deduped={data.get('deduped_count')} "
            f"count={len(schemas)}"
        )
        _PINNED_KEYS.add(key)
    except TCMMUnavailable:
        raise
    except (httpx.ConnectError, httpx.ReadTimeout) as _e:
        logger.error(
            f"  [TCMM] pin_tool_definitions network error: "
            f"{type(_e).__name__}: {_e}"
        )
        raise TCMMUnavailable(
            f"TCMM pin_tool_definitions: {type(_e).__name__}"
        ) from _e
    except Exception as _e:
        logger.error(
            f"  [TCMM] pin_tool_definitions error: {type(_e).__name__}: {_e}"
        )
        raise TCMMUnavailable(
            f"TCMM pin_tool_definitions: {type(_e).__name__}: {_e}"
        ) from _e


# ── Workspace state: surface the client-daemon's project_root to the LLM ──
#
# Without this, Grok (and any other model) has no idea where the user's
# files actually live — it has tools like ``run_command`` and ``read_file``
# but no path to point them at, so it resorts to blindly emitting ``pwd``
# / ``ls`` calls or just refusing to act.
#
# The sub-agents server already knows: it brokers the WebSocket connection
# to the user's client-daemon and exposes ``/api/client/folders`` +
# ``/api/client/status``. We poll those once per turn (cheap — sub-agents
# is in-process to the daemon bridge), then pin the result to TCMM as an
# IMMUTABLE block via ``/pin/user_profile``. The block lands in the
# stable tier of the static system prefix → caches at 99% → costs nothing
# on turn N+1.


async def _fetch_workspace_state(user_id: str) -> dict | None:
    """Ask sub-agents for the connected daemon's workspace state.

    Returns ``{folders: [...], client_id: "...", os_hint: "..."}`` if the
    user has a daemon connected, ``None`` if not connected or sub-agents
    is unreachable. NEVER raises — workspace context is best-effort.
    """
    if not user_id or not _VEILGUARD_INTERNAL_SECRET:
        return None
    try:
        client = _get_tcmm_client()  # reuse the persistent httpx client
        headers = {
            "x-internal-secret": _VEILGUARD_INTERNAL_SECRET,
            "x-user-id":         user_id,
        }
        # Status tells us if a daemon is actually connected — folders
        # alone will return a stale cache even if the daemon dropped.
        st = await client.get(
            f"{SUB_AGENTS_URL}/api/client/status",
            headers=headers, timeout=2,
        )
        if st.status_code != 200:
            return None
        st_data = st.json()
        if not st_data.get("connected"):
            return None
        fold = await client.get(
            f"{SUB_AGENTS_URL}/api/client/folders",
            headers=headers, timeout=2,
        )
        if fold.status_code != 200:
            return None
        folders = (fold.json() or {}).get("folders") or []
        if not folders:
            return None
        cid = str(st_data.get("client_id") or "")
        # 2026-05-18: prefer real platform fields from the daemon
        # auth handshake (``platform``, ``os_name``, ``os_release``,
        # ``shell`` — bridged via client_bridge.status()). Daemons
        # 0.2.4 and older don't send these so we fall back to a
        # path-prefix heuristic. ``sys.platform`` short codes:
        # ``win32`` / ``linux`` / ``darwin``.
        real_platform = str(st_data.get("platform") or "")
        real_os       = str(st_data.get("os_name") or "")
        real_release  = str(st_data.get("os_release") or "")
        real_shell    = str(st_data.get("shell") or "")
        if real_platform:
            if real_platform.startswith("win"):
                os_hint = (
                    f"{real_os or 'Windows'} {real_release} — use "
                    f"PowerShell / CMD syntax for run_command "
                    f"(shell: {real_shell or 'cmd.exe'})."
                )
            elif real_platform == "darwin":
                os_hint = (
                    f"macOS {real_release} (Darwin) — use bash/zsh "
                    f"syntax (shell: {real_shell or '/bin/zsh'})."
                )
            else:
                os_hint = (
                    f"{real_os or 'Linux'} {real_release} — use bash "
                    f"syntax (shell: {real_shell or '/bin/bash'})."
                )
        else:
            # Heuristic fallback for daemons predating 0.2.5 platform reporting.
            os_hint = ""
            if folders and (folders[0].startswith(("C:\\", "D:\\", "E:\\"))
                            or "\\" in folders[0]):
                os_hint = "Windows (use PowerShell / cmd syntax for run_command)"
            elif folders and folders[0].startswith("/"):
                os_hint = "Unix-like (use bash syntax for run_command)"
        return {
            "folders":      folders,
            "client_id":    cid,
            "os_hint":      os_hint,
            "platform":     real_platform,
            "os_name":      real_os,
            "os_release":   real_release,
            "shell":        real_shell,
        }
    except Exception as _e:
        logger.debug(f"  [workspace] fetch failed: {type(_e).__name__}: {_e}")
        return None


_MCP_TOOL_SCHEMAS_CACHE: list | None = None
_MCP_TOOL_SCHEMAS_TS: float = 0.0
_MCP_TOOL_SCHEMAS_TTL = 300  # 5 min — schemas only change on sub-agents redeploy


async def _fetch_mcp_tool_schemas() -> list:
    """Fetch the OpenAI-format MCP tool schemas from sub-agents server.

    Cached in-process for 5 minutes — schemas change only when
    sub-agents redeploys. Failure returns an empty list (degrades to
    "no tools available", same UX as if LibreChat sent none).

    Why this exists: LibreChat's custom xAI endpoint does NOT forward
    MCP tools to the upstream API — only its Agents endpoint does. Users
    selecting 'Grok' from the dropdown therefore lose function-calling
    entirely (Grok knows tool NAMES from the preamble text but has no
    schemas to invoke). This bridges the gap by injecting schemas on
    the proxy side, regardless of which LibreChat endpoint shipped the
    request.
    """
    global _MCP_TOOL_SCHEMAS_CACHE, _MCP_TOOL_SCHEMAS_TS
    import time as _tt
    now = _tt.time()
    if _MCP_TOOL_SCHEMAS_CACHE is not None and (now - _MCP_TOOL_SCHEMAS_TS) < _MCP_TOOL_SCHEMAS_TTL:
        return _MCP_TOOL_SCHEMAS_CACHE
    if not _VEILGUARD_INTERNAL_SECRET:
        return []
    try:
        client = _get_tcmm_client()
        resp = await client.get(
            f"{SUB_AGENTS_URL}/api/tools/openai_schemas",
            headers={"x-internal-secret": _VEILGUARD_INTERNAL_SECRET},
            timeout=4,
        )
        if resp.status_code != 200:
            return _MCP_TOOL_SCHEMAS_CACHE or []
        data_j = resp.json()
        schemas = data_j.get("tools") or []
        if not isinstance(schemas, list) or not schemas:
            return _MCP_TOOL_SCHEMAS_CACHE or []
        _MCP_TOOL_SCHEMAS_CACHE = schemas
        _MCP_TOOL_SCHEMAS_TS = now
        logger.info(
            f"  [services] schema cache refreshed: {len(schemas)} tools"
        )
        return schemas
    except Exception as _e:
        logger.debug(f"  [services] fetch failed: {type(_e).__name__}: {_e}")
        return _MCP_TOOL_SCHEMAS_CACHE or []


def _inject_mcp_tools_if_missing(data: dict, fmt: str, schemas: list) -> None:
    """Stamp MCP tool schemas onto ``data["tools"]`` if the client didn't
    send any. Only applies to OpenAI / xAI format (Anthropic uses its
    own ``tools`` shape and is handled separately by LibreChat's Agents
    runtime today).

    No-op if:
      - fmt is not openai/xai/grok (Anthropic format would need
        translation; skip until we hit that case)
      - client already sent tools (don't clobber)
      - schemas list is empty (sub-agents unreachable or registered
        nothing)
    """
    if fmt not in ("openai", "grok"):
        return
    if not schemas:
        return
    if data.get("tools"):
        return
    data["tools"] = list(schemas)
    # Default tool_choice to auto so the model is free to call OR not.
    if "tool_choice" not in data:
        data["tool_choice"] = "auto"


def _render_workspace_block(state: dict) -> str:
    """Format the daemon's workspace state as a short system block.

    Returns ``""`` for an empty / missing state. Output is deterministic
    (sorted folders) but NOT cached by xAI — it lives outside the static
    prefix on purpose so a workspace switch (user opens a different
    project, daemon reconnects with new folders) takes effect on the
    next turn with no cache-purge dance.
    """
    if not state:
        return ""
    folders = state.get("folders") or []
    if not folders:
        return ""
    lines = [
        "## CURRENT WORKSPACE STATE (live, may change between turns)",
        f"Active folders: {', '.join(repr(f) for f in sorted(folders))}",
    ]
    if state.get("os_hint"):
        lines.append(f"Environment: {state['os_hint']}")
    if state.get("client_id"):
        lines.append(f"Client daemon: {state['client_id']}")
    lines.append(
        "Use these paths as defaults for file_read / run_command / "
        "search_files — do NOT probe with `pwd` or `ls` to discover them."
    )
    return "\n".join(lines)


def _inject_workspace_state(data: dict, fmt: str, state: dict) -> None:
    """Append the workspace block as the LAST system context, right before
    the user turn. Sits outside the cached static prefix so it re-renders
    every turn — the user can switch projects without TCMM-side eviction.

    No-op if state is empty or no folders were reported.
    """
    block = _render_workspace_block(state)
    if not block:
        return

    if fmt in ("openai", "grok"):
        msgs = data.get("messages") or []
        # Find first non-system message — workspace block slots right
        # before it so the user / assistant / tool sequence is unbroken.
        insert_at = len(msgs)
        for i, m in enumerate(msgs):
            if isinstance(m, dict) and m.get("role") != "system":
                insert_at = i
                break
        msgs.insert(insert_at, {"role": "system", "content": block})
        data["messages"] = msgs
        return

    if fmt == "anthropic":
        sys_field = data.get("system")
        new_block = {"type": "text", "text": block}
        if isinstance(sys_field, str):
            data["system"] = [{"type": "text", "text": sys_field}, new_block]
        elif isinstance(sys_field, list):
            sys_field.append(new_block)
            data["system"] = sys_field
        else:
            data["system"] = [new_block]
        return


# ── Per-provider request shape helpers (extract / strip / apply) ──────────
#
# Goal: the proxy's request-handler is symmetric across Anthropic, OpenAI
# and Grok. The format-specific knowledge (where the client's system
# message lives, where the renderer's output lands) is concentrated in
# these helpers — the handler just calls them in sequence.


def _extract_client_system(data: dict, fmt: str) -> str:
    """Read the client's per-conversation system prompt from the request.

    Anthropic: ``data["system"]`` — either a string or a list of
        content blocks (``{"type":"text", "text":...}``). We
        concatenate text-typed blocks.
    OpenAI / Grok: first message in ``data["messages"]`` if its role is
        ``system``. ``content`` may be a string or (multi-modal) list
        of parts; we extract text parts only.

    Returns ``""`` when absent. Does not mutate ``data``.
    """
    if fmt == "anthropic":
        s = data.get("system")
        if isinstance(s, str):
            return s
        if isinstance(s, list):
            parts = []
            for blk in s:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    t = blk.get("text") or ""
                    if t:
                        parts.append(t)
            return "\n".join(parts)
        return ""
    if fmt in ("openai", "grok"):
        msgs = data.get("messages") or []
        if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
            c = msgs[0].get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                parts = []
                for blk in c:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        parts.append(blk.get("text") or "")
                return "\n".join(parts)
        return ""
    return ""


def _strip_client_system(data: dict, fmt: str) -> None:
    """Remove the client system prompt from its original location.

    Called AFTER ``_extract_client_system`` has pinned it to TCMM and
    BEFORE ``_apply_render_to_request`` slots the renderer's output.
    Without this, the system content would appear twice in the request
    (once from TCMM's render, once from the original slot).
    """
    if fmt == "anthropic":
        # data["system"] is about to be overwritten by the renderer's
        # blocks; clearing first is belt-and-braces in case any later
        # step reads it before the apply.
        data.pop("system", None)
        return
    if fmt in ("openai", "grok"):
        msgs = data.get("messages") or []
        if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
            data["messages"] = msgs[1:]


def _extract_client_tools(data: dict) -> list:
    """Read the tool schemas the client attached to this request.

    Both Anthropic and OpenAI use ``data["tools"]``. Shapes differ
    (Anthropic: ``{name, description, input_schema}``; OpenAI:
    ``{type:"function", function:{name, description, parameters}}``) but
    TCMM's ``/pin/tool_definitions`` stores either form opaquely via
    ``json.dumps(sort_keys=True)``, so we don't normalize here.

    Returns ``[]`` when absent / empty.
    """
    t = data.get("tools")
    return list(t) if isinstance(t, list) else []


def _tools_audit_section(tools) -> str:
    """Render native tool definitions as a LEAN, audit-only ``[TOOLS]`` block.

    The MODEL receives tools via the native ``tools`` API field, untouched —
    this text is ONLY for the admin dashboard's TOOLS section + token
    accounting. The dashboard parses the audit ``content``, which stopped
    carrying tool schemas when they moved to the native field (so the TOOLS
    section read empty and the per-section token math lost them). We show each
    tool's name + 1-line description, headed by the REAL token cost of the full
    schemas so the section reconciles with the API's total ``input_tokens``.
    Returns "" when there are no tools.
    """
    if not tools:
        return ""
    try:
        _full = json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
        _tok = max(1, len(_full) // 4)  # ~4 chars/token (no tokenizer in-proc)
    except Exception:
        _tok = 0
    _lines = [
        f"[TOOLS]  ({len(tools)} tools · ~{_tok} tokens · sent via the native "
        f"function-calling field, NOT prompt text)"
    ]
    for _t in tools:
        if not isinstance(_t, dict):
            continue
        _fn = _t.get("function") if isinstance(_t.get("function"), dict) else _t
        _name = _fn.get("name") or _t.get("name") or "?"
        _desc = _fn.get("description") or _t.get("description") or ""
        _desc = " ".join(str(_desc).split())[:100]
        _lines.append(f"  • {_name}: {_desc}")
    return "\n".join(_lines)


def _trim_to_current_turn(messages: list) -> list:
    """Drop every message before the latest user turn — they're already in
    TCMM memory blocks and re-sending them doubles the prompt.

    Returns the slice starting at the LAST ``role=user`` message (so any
    assistant tool_call / role=tool follow-ups that came after it are
    preserved — those represent the in-flight current turn, not history).

    If no user message exists (shouldn't happen for chat completions),
    returns the input unchanged as a safety belt.

    Examples
    --------
    Pure chat history (every turn was completed before this request):
        in : [u1, a1, u2, a2, u3]      →  out: [u3]
    Mid-tool-call sequence (assistant called a tool, now we have its
    result, model needs to continue):
        in : [u1, a1, u2, a2(tool_calls), tool(result)]
                                       →  out: [u2, a2(tool_calls), tool(result)]
    """
    # Find the last *real* user message — one whose content is NOT
    # exclusively ``tool_result`` blocks. On the Anthropic schema, a
    # tool-result follow-up is shaped as
    # ``{"role":"user", "content":[{"type":"tool_result", ...}]}``
    # and its matching tool_use lives in the PRIOR assistant message.
    # Trimming to "last role=user" without this check orphans the
    # tool_result and Anthropic returns 400:
    #
    #   messages.0.content.0: unexpected `tool_use_id` found in
    #   `tool_result` blocks. Each `tool_result` block must have a
    #   corresponding `tool_use` block in the previous message.
    #
    # 2026-05-19 fix: walk back to the last user message whose
    # content is a plain string OR a list with at least one NON-tool_result
    # block — that's the user's actual question. Keep from there;
    # tool_use/tool_result pairs that follow are part of the in-flight
    # current turn and stay intact.
    last_real_user = -1
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            # OpenAI/xAI text-only user message — always counts as real.
            last_real_user = i
            break
        if isinstance(content, list):
            has_non_tool_result = any(
                isinstance(b, dict) and b.get("type") != "tool_result"
                for b in content
            )
            if has_non_tool_result or not content:
                last_real_user = i
                break
        else:
            # Anything unexpected — treat as a real user msg.
            last_real_user = i
            break
    if last_real_user < 0:
        # No real user message found (e.g. every user msg is a
        # tool_result follow-up). Returning everything is safer than
        # returning an empty list — the LLM gets the full history and
        # the request is at worst expensive, never malformed.
        return list(messages)
    return list(messages[last_real_user:])


def _apply_render_to_request(
    data: dict, headers: dict, render_result: dict,
) -> None:
    """Slot the renderer's output into the provider-shaped request body.

    This is the SINGLE place per-provider wire-shape logic lives.
    Anthropic, OpenAI and Grok hit one of the branches below; nothing
    else in the request handler should touch ``data["system"]`` or
    prepend to ``data["messages"]``.

    Pre-condition: the client's system prompt + tool defs have already
    been pinned to TCMM via ``_tcmm_pin_*``, and ``_strip_client_system``
    has removed the client's system from its original location. Thus
    everything the model needs is in ``render_result``.

    2026-05-18: now also calls ``_trim_to_current_turn(data["messages"])``
    so the raw LibreChat conversation history isn't re-sent on top of
    the same turns already rendered as ``[Memory index=N | role=USER |
    src=live]`` blocks in TCMM's system message. Memory is the canonical
    copy; the messages array carries only the in-flight turn.

    Raises ``ValueError`` for unknown ``format`` — caller turns that
    into a 502 (TCMM bug, fail loud).
    """
    fmt = (render_result.get("format") or "").lower()
    if fmt == "anthropic":
        blocks = render_result.get("blocks") or []
        data["system"] = list(blocks)
        # [WORKING_AUTOCACHE_2026_05_20] Anthropic server-managed cache
        # breakpoint. Renderer dropped the manual working-tier marker
        # (which had a 0% hit rate due to promotion-driven byte shifts)
        # and instead asks the server to manage one breakpoint at the
        # tail of the cacheable prefix. The server advances it forward
        # as the conversation grows. Consumes 1 of the 4 breakpoint
        # slots — _cap_cache_markers below accounts for it.
        if _auto_cc := render_result.get("cache_control"):
            data["cache_control"] = _auto_cc
        if render_result.get("uses_extended_cache_ttl"):
            _ensure_extended_cache_ttl_beta(data, headers)
        # Anthropic: messages list carries the conversation. Trim
        # earlier turns now in memory; keep only current turn + any
        # tool_use / tool_result follow-ups after the last user message.
        if isinstance(data.get("messages"), list):
            data["messages"] = _trim_to_current_turn(data["messages"])
        return
    if fmt in ("openai", "grok"):
        # Prefer the renderer's wire-shaped messages list (one or more
        # role=system messages with string content). Falls back to a
        # single system message wrapping ``text`` for renderer versions
        # that don't yet populate ``messages``.
        msgs = render_result.get("messages") or [{
            "role": "system",
            "content": render_result.get("text", ""),
        }]
        existing = data.setdefault("messages", [])
        # Drop the conversation history before the current user turn —
        # those turns are already rendered as memory blocks above.
        trimmed = _trim_to_current_turn(existing)
        data["messages"] = list(msgs) + trimmed
        return
    raise ValueError(f"unknown render format: {fmt!r}")


async def _tcmm_render(
    model: str, task_query: str,
    conv_id: str = "", user_id: str = "",
) -> dict:
    """Ask TCMM to render the current memory state for ``model``.

    ``model`` is one of: 'anthropic', 'claude', 'openai', 'gpt',
    'grok', 'xai', 'vllm'. Returns the render dict (keys: format, text,
    blocks, regions, uses_extended_cache_ttl, stats). Raises
    ``TCMMUnavailable`` on any failure — the caller's existing
    try/except returns 503 to client. No silent fallback by design.
    """
    try:
        client = _get_tcmm_client()
        resp = await client.post(
            f"{TCMM_URL}/render",
            json={
                "model": model,
                "task_query": task_query,
                "conversation_id": conv_id,
                "user_id": user_id,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            logger.error(
                f"  [TCMM] render HTTP {resp.status_code}: {resp.text[:200]}"
            )
            raise TCMMUnavailable(
                f"TCMM render HTTP {resp.status_code}"
            )
        return resp.json()
    except TCMMUnavailable:
        raise
    except (httpx.ConnectError, httpx.ReadTimeout) as _e:
        logger.error(
            f"  [TCMM] render network error: {type(_e).__name__}: {_e}"
        )
        raise TCMMUnavailable(f"TCMM render: {type(_e).__name__}") from _e
    except Exception as _e:
        logger.error(f"  [TCMM] render error: {type(_e).__name__}: {_e}")
        raise TCMMUnavailable(
            f"TCMM render: {type(_e).__name__}: {_e}"
        ) from _e


async def _tcmm_post_response(
    raw_output: str,
    conversation_id: str,
    user_id: str = "",
    origin: str = "assistant_text",
    lineage_parent_conv: str = "",
    flag_obj: dict | None = None,
) -> str | None:
    """Call TCMM service to process response. Returns clean answer or None on failure.

    `origin` defaults to "assistant_text". Set to "tool_use" when the
    assistant's reply is itself a tool invocation (unusual — we usually
    catch that on the next turn via _extract_tool_pair).

    `flag_obj`: when the proxy captured the tcmm_record_turn shadow
    tool's input (universal across all backends since 2026-05-22),
    pass it through so the adapter doesn't have to parse prose JSON
    that won't be there. Shape: {used, knowledge_class, epoch_complete,
    emit_class}.
    """
    _body = {
        "raw_output": raw_output,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "origin": origin,
        "lineage_parent_conv": lineage_parent_conv,
    }
    if flag_obj and isinstance(flag_obj, dict):
        _body["flag_obj"] = flag_obj
    try:
        client = _get_tcmm_client()
        resp = await client.post(
            f"{TCMM_URL}/post_response",
            json=_body,
            timeout=120,
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


async def _tcmm_ingest_turn(
    items: list,
    conversation_id: str,
    user_id: str = "",
    lineage_parent_conv: str = "",
) -> int:
    """Persist auxiliary turn items (tool_use / tool_result) into TCMM.

    The tool round-trip isn't a primary user/assistant turn, so pre_request
    doesn't ingest it and post_response only sees the final text answer.
    Without this call the archive would have no record that a tool was
    invoked or what it returned. Returns the number of blocks TCMM added.
    """
    if not items:
        return 0
    try:
        client = _get_tcmm_client()
        resp = await client.post(
            f"{TCMM_URL}/ingest_turn",
            json={
                "conversation_id": conversation_id,
                "user_id": user_id,
                "items": items,
                "lineage_parent_conv": lineage_parent_conv,
            },
            timeout=120,
        )
        if resp.status_code == 200:
            data = resp.json()
            added = data.get("added", 0)
            logger.info(
                f"  [TCMM] ingest_turn OK — "
                f"added={added}/{data.get('requested', len(items))} "
                f"origins={[(i or {}).get('origin') for i in items]}"
            )
            return added
        logger.warning(f"  [TCMM] ingest_turn HTTP {resp.status_code}")
    except httpx.ConnectError:
        logger.warning("  [TCMM] service unreachable — tool round-trip not persisted")
    except Exception as e:
        logger.warning(f"  [TCMM] ingest_turn error: {e}")
    return 0


def _canonical_param_hash(tool_input) -> str:
    """Hash the canonical-JSON form of a tool call's input parameters.

    Same params → same hash, regardless of key order or whitespace.
    Used as a stable identifier so multiple invocations of
    `read_file({"path":"/etc/passwd"})` share a key and can be compared
    — two identical calls returning different results is a signal
    (state drift, flaky tool, broken result).

    12 hex chars = ~48 bits of entropy — collision risk is negligible
    at the scale of a conversation's tool calls.
    """
    try:
        canon = json.dumps(
            tool_input if tool_input is not None else {},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
    except Exception:
        canon = str(tool_input)
    import hashlib
    return hashlib.sha1(canon.encode("utf-8", errors="replace")).hexdigest()[:12]


def _extract_tool_pair(messages: list) -> list:
    """Extract the tool_use + tool_result pair from a tool-followup turn.

    A tool round-trip looks like:
        messages[-2] = {role: assistant, content: [..., {type: tool_use, id, name, input}]}
        messages[-1] = {role: user,      content: [{type: tool_result, tool_use_id, content}]}

    Both need to be persisted in TCMM so the archive has a faithful
    record of what the model invoked and what it got back.

    Pairing key is `(tool_name, param_hash)` — two different invocations
    of the *same* tool with the *same* input share this key, so the
    adapter's find_tool_invocations() scan can compare their results.
    tool_use_id is kept as a secondary tiebreaker for exact per-call
    matching within the same pair of messages.

    Each item returned:
        {text, origin, tool_name, tool_use_id, param_hash}.
    Safe on malformed messages — returns [] if the expected shape
    isn't there.
    """
    items: list = []
    if not messages or len(messages) < 2:
        return items

    # tool_use_id → (tool_name, param_hash) lookup from the assistant
    # turn. tool_result doesn't carry name/input, only the id, so we
    # reuse what we saw on the matching tool_use block.
    tu_id_to_meta: dict[str, tuple] = {}

    # --- messages[-2]: assistant with tool_use (maybe also text) ---
    asst = messages[-2]
    if isinstance(asst, dict) and asst.get("role") == "assistant":
        content = asst.get("content")
        if isinstance(content, list):
            for blk in content:
                if not isinstance(blk, dict):
                    continue
                btype = blk.get("type")
                if btype == "tool_use":
                    name = blk.get("name", "unknown")
                    tuid = blk.get("id", "")
                    inp = blk.get("input", {})
                    phash = _canonical_param_hash(inp)
                    if tuid:
                        # Carry raw input too so tool_result can inherit
                        # it without re-parsing (result envelope doesn't
                        # carry params).
                        tu_id_to_meta[tuid] = (name, phash, inp)
                    try:
                        inp_str = json.dumps(inp, ensure_ascii=False)[:2000]
                    except Exception:
                        inp_str = str(inp)[:2000]
                    text = f"TOOL CALL {name}({inp_str})"
                    items.append({
                        "text": text,
                        "origin": "tool_use",
                        "tool_name": name,
                        "tool_use_id": tuid,
                        "param_hash": phash,
                        # Raw structured input — stored on the archive
                        # entry so later analytics can aggregate by
                        # actual param values (not just the hash).
                        "params": inp,
                    })
                # Text blocks alongside tool_use are already ingested
                # via post_response on the prior turn — skip them here.

    # --- messages[-1]: user with tool_result ---
    usr = messages[-1]
    if isinstance(usr, dict) and usr.get("role") == "user":
        content = usr.get("content")
        if isinstance(content, list):
            for blk in content:
                if not isinstance(blk, dict) or blk.get("type") != "tool_result":
                    continue
                tuid = blk.get("tool_use_id", "")
                # Reuse (name, param_hash, params) from the matching
                # tool_use in this turn. Falls back to "unknown" /
                # empty hash if we can't see the matching call —
                # orphan scan will still catch this as a "result with
                # no call".
                tname, phash, tparams = tu_id_to_meta.get(
                    tuid, ("unknown", "", None)
                )
                raw = blk.get("content", "")
                # tool_result content can be a string OR a list of blocks
                # (Anthropic allows nested text / image results). Flatten
                # to a single string for ingestion.
                if isinstance(raw, list):
                    parts: list[str] = []
                    for sub in raw:
                        if isinstance(sub, dict):
                            if sub.get("type") == "text":
                                parts.append(sub.get("text", ""))
                            elif sub.get("type") == "image":
                                parts.append("[image omitted]")
                        elif isinstance(sub, str):
                            parts.append(sub)
                    raw_str = "\n".join(p for p in parts if p)
                else:
                    raw_str = str(raw) if raw is not None else ""
                text = f"TOOL RESULT [{tuid}]: {raw_str}" if tuid else f"TOOL RESULT: {raw_str}"
                items.append({
                    "text": text,
                    "origin": "tool_result",
                    "tool_name": tname,
                    "tool_use_id": tuid,
                    "param_hash": phash,
                    # Raw structured result content (string or list of
                    # Anthropic sub-blocks) + the params that produced
                    # it. Lets analytics compare "same command, same
                    # params, different result" without text parsing.
                    "params": tparams,
                    "result": raw if raw is not None else "",
                })

    return items


# ── Veilguard provenance envelope stripping ─────────────────────────────────
#
# Connectors (SharePoint, Slack, ...) wrap their tool output in this shape
# so the TCMM ingest path can extract acl/tool_ref/etag/title metadata::
#
#     {
#       "content": "<LLM-visible text>",
#       "_veilguard": {connector, tool_ref, acl, etag, title}
#     }
#
# By the time the gateway is about to forward the request to the LLM, TCMM
# ingest has already pulled the metadata via its own parser (in
# `veilguard_adapter.ingest_turn`). We strip the envelope here so the LLM
# only ever sees the inner `content` — the `_veilguard` block is internal
# infrastructure.
#
# Strip is unconditional on every chat-completion request: tool_result
# blocks from prior turns also carry envelopes (LibreChat re-sends history
# every turn), and the LLM never benefits from seeing them.
#
# The parser is a permissive mirror of the ones in
# services/connectors/_base/envelope.py and TCMM's veilguard_adapter.py.
# Plain text, malformed JSON, or JSON without `_veilguard` all pass through
# untouched.

_VEILGUARD_ENV_KEY = "_veilguard"


def _strip_envelope_from_str(text: str) -> str | None:
    """Return inner ``content`` if ``text`` is a `_veilguard` envelope.

    Returns ``None`` when the input is not an envelope — caller keeps
    the original text unchanged.
    """
    if not isinstance(text, str) or not text or not text.lstrip().startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    if not isinstance(parsed, dict) or _VEILGUARD_ENV_KEY not in parsed:
        return None
    inner = parsed.get("content", "")
    return inner if isinstance(inner, str) else str(inner)


def _strip_veilguard_envelopes_from_messages(messages: list) -> int:
    """Strip `_veilguard` envelopes from every tool_result content block
    in ``messages``. Mutates the messages list in place.

    Tool_result content can be a string OR a list of sub-blocks
    (Anthropic supports text/image sub-blocks). Both cases are handled.

    Returns the number of envelopes stripped, for logging.
    """
    if not isinstance(messages, list):
        return 0
    stripped_count = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        # Anthropic puts tool_results in role=user messages with content blocks
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for blk in content:
            if not isinstance(blk, dict) or blk.get("type") != "tool_result":
                continue
            blk_content = blk.get("content", "")
            if isinstance(blk_content, str):
                inner = _strip_envelope_from_str(blk_content)
                if inner is not None:
                    blk["content"] = inner
                    stripped_count += 1
            elif isinstance(blk_content, list):
                for sub in blk_content:
                    if not isinstance(sub, dict):
                        continue
                    if sub.get("type") == "text":
                        inner = _strip_envelope_from_str(sub.get("text", ""))
                        if inner is not None:
                            sub["text"] = inner
                            stripped_count += 1
    return stripped_count


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

_MIN_CACHE_CHARS = 9200  # ~2300 tokens. Empirically verified 23 Apr 2026
# via binary-search against api.anthropic.com directly:
#
#   4508B (1127 tok) create=0 read=0    ← silently refused
#   6758B (1689 tok) create=0 read=0    ← silently refused
#   9008B (2252 tok) create=2205 read=0 ← first cache write succeeds
#
# So Sonnet-4.6's real minimum cacheable segment is ~2048 tokens, NOT the
# ~1024 tokens quoted in public docs (which may apply to older Sonnet 3.x
# models, or may have been raised silently). When ANY cache_control marker
# sits below this threshold, Anthropic silently refuses to cache ANY marker
# in the entire request — so you get `create=0 read=0` and every turn pays
# full input cost instead of the 90%-discount cache_read rate. Was costing
# us real money on every chat turn because the static Veilguard preamble
# (5716B / 1430 tok) fell in the dead zone between published-doc-minimum
# and actual-minimum. Preamble is now expanded to >9200B to clear the floor.


# Phase 7 marker constants — TCMM adapter emits these to delimit the
# two-tier cache layout. Module-level so tests can import them and the
# adapter side can stay in sync.
TCMM_STABLE_BOUNDARY = "--- END STABLE MEMORY ---"
TCMM_LIVE_BOUNDARY = "--- END LIVE MEMORY ---"

# ────────────────────────────────────────────────────────────────────────
# 2026-05-14: Hoisted out of `if _is_anthropic_format` so the xAI/OpenAI
# branch can prepend the same byte-stable preamble for prefix caching.
# Static literal — no PII, no per-request interpolation. Used in both
# Anthropic's tiered system-blocks layout and xAI's tiered messages
# layout below.
# ────────────────────────────────────────────────────────────────────────
# [PROPER_PREAMBLE_FIX_2026_05_20] The preamble itself is the
# Claude-API-compliant system prefix. Anthropic's OAuth-bearer
# gate requires non-Haiku models see one of the official
# Claude Code identity strings as the literal start of the
# system prompt — without it Opus/Sonnet 429 with a generic
# rate_limit_error (it's a policy gate, not a quota check).
# See claude-code src/constants/system.ts:AGENT_SDK_PREFIX.
# Preamble unified into agent/preamble.py (the single source of truth
# shared between pii-proxy + agent-runtime + ChatAgent paths).  The
# template is tool-agnostic — tool schemas are pinned separately via
# TCMM /pin/tool_definitions for per-tool cache stability.  Keeping
# this section as a thin shim so existing call sites
# (_render_preamble_with_tools, _VEILGUARD_PREAMBLE_TEXT) stay
# importable without bringing 380 lines of duplicated template back.
try:
    from agent.preamble import render_preamble as _render_preamble_shared
except Exception as _e:  # pragma: no cover — bind mount missing
    import logging as _logging
    _logging.getLogger("veilguard.proxy").error(
        f"[preamble] failed to import agent.preamble: {_e}"
    )
    _render_preamble_shared = lambda *_a, **_kw: ""


def _render_preamble_with_tools(tools_list=None) -> str:
    """Render the Veilguard preamble.  The tools_list argument is
    retained for ABI compatibility with legacy call sites; it is
    IGNORED because tools are now pinned separately via
    /pin/tool_definitions (one immutable block per schema).  See
    agent/preamble.py for the unified template.
    """
    _ = tools_list
    return _render_preamble_shared()


# Cached tools-less render — same value as _render_preamble_with_tools()
# since the function ignores its argument.  Kept as a module-level
# constant for code that imported it directly.
_VEILGUARD_PREAMBLE_TEXT = _render_preamble_with_tools(None)


# 2026-05-15: legacy tier-splitting + system-block assembly helpers
# removed. TCMM's renderers own all of this now — the proxy calls
# /render and slots the result into the provider-shaped JSON field.
# Deleted: _split_tcmm_memory_into_tiers, _assemble_system_blocks_for_tiers,
# _split_for_cache. _count_cache_markers + _cap_cache_markers remain as
# the final 4-marker safety net (LibreChat itself can attach cache_control
# on tool_use/tool_result blocks, which can push the total past Anthropic's
# limit of 4 — the cap below strips the oldest to keep us under).


# Anthropic enforces a hard cap of 4 cache_control markers per request.
# LibreChat itself already emits some (tool_use/tool_result blocks in
# ongoing multi-turn conversations), and the TCMM split-cache path adds
# one more on the system head.  We must count what exists before
# adding any of our own, otherwise we blow past 4 and the API 400s
# with "A maximum of 4 blocks with cache_control may be provided".
_ANTHROPIC_CACHE_LIMIT = 4


def _count_cache_markers(data: dict) -> int:
    """Count cache_control markers already attached to system + messages.

    [WORKING_AUTOCACHE_2026_05_20] Anthropic's request-root ``cache_control``
    (auto-mode breakpoint) ALSO consumes one of the 4 slots — count it.
    Without this, the cap helper undercounts and Anthropic 400s with
    "A maximum of 4 blocks with cache_control may be provided" when TCMM
    emits 3 per-block markers + 1 auto-mode + LibreChat puts a marker on a
    tool_use/tool_result.
    """
    total = 0
    if isinstance(data.get("cache_control"), dict):
        total += 1
    system = data.get("system")
    if isinstance(system, list):
        for blk in system:
            if isinstance(blk, dict) and "cache_control" in blk:
                total += 1
    messages = data.get("messages") or []
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and "cache_control" in blk:
                        total += 1
    return total


def _scrub_malformed_thinking(data: dict) -> int:
    """Remove malformed extended-thinking content blocks before forwarding.

    Anthropic's extended-thinking API rejects a request with
    ``messages.N.content.M.thinking.thinking: Field required`` when an
    assistant content block has ``{type: "thinking"}`` without its inner
    ``thinking`` string (and ``signature`` is also required for echo-back
    verification on multi-turn). Observed in two independent sessions:

      22 Apr — Sarel, cursed conv bb400c87, 3 persisted blocks in MongoDB
      23 Apr — Petrus, fresh new conv, mid-tool-round-trip

    Upstream path handled the Mongo-stored case via the formatAgentMessages
    patch in the LibreChat fork, but that function only runs at the initial
    payload-to-LangGraph hydration. Mid-run AIMessages that LangGraph holds
    in memory go through @langchain/anthropic's own serializer, which can
    produce the same malformed shape without going near our patch. Scrubbing
    here — at the last hop before Anthropic — catches every path.

    Scrub logic: for each content block with ``type == "thinking"``, keep
    it only when ``thinking`` is a non-empty string AND ``signature`` is
    non-empty (both Anthropic requirements for cross-turn reuse). Drop
    anything else silently. If removing the thinking block leaves the
    parent ``content`` array empty (unlikely — usually there's also text
    or tool_use siblings), the whole content is replaced with a single
    placeholder text block so Anthropic doesn't reject an empty array.

    Returns the number of blocks scrubbed.
    """
    scrubbed = 0

    def _is_well_formed(blk):
        if not isinstance(blk, dict):
            return False
        if blk.get("type") != "thinking":
            return True  # we only police thinking blocks
        thinking_text = blk.get("thinking")
        signature = blk.get("signature")
        return (
            isinstance(thinking_text, str) and thinking_text.strip() != ""
            and isinstance(signature, str) and signature != ""
        )

    messages = data.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            cleaned = []
            for blk in content:
                if _is_well_formed(blk):
                    cleaned.append(blk)
                else:
                    scrubbed += 1
            if len(cleaned) != len(content):
                # Don't emit an empty content array — Anthropic 400s on
                # that too. Put a single text placeholder so the turn
                # still exists as a structural element.
                if not cleaned:
                    cleaned = [{
                        "type": "text",
                        "text": "[previous reasoning omitted]",
                    }]
                msg["content"] = cleaned

    # Same defense on the system field — unusual for system to carry a
    # thinking block, but belt-and-braces since we own this filter now.
    system = data.get("system")
    if isinstance(system, list):
        cleaned_sys = []
        for blk in system:
            if _is_well_formed(blk):
                cleaned_sys.append(blk)
            else:
                scrubbed += 1
        if len(cleaned_sys) != len(system):
            data["system"] = cleaned_sys

    return scrubbed


def _cap_cache_markers(data: dict, limit: int = _ANTHROPIC_CACHE_LIMIT) -> int:
    """Enforce Anthropic's 4-marker limit — strip oldest message markers first.

    Preserves system-field markers (largest cached prefix, highest
    value) and the most recent message markers.  Walks ``messages`` in
    chronological order, deleting ``cache_control`` keys until the total
    marker count drops to ``limit``.

    Returns the number of markers stripped.
    """
    total = _count_cache_markers(data)
    if total <= limit:
        return 0
    stripped = 0
    messages = data.get("messages") or []
    if not isinstance(messages, list):
        return 0
    for msg in messages:
        if total <= limit:
            break
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for blk in content:
            if total <= limit:
                break
            if isinstance(blk, dict) and "cache_control" in blk:
                del blk["cache_control"]
                stripped += 1
                total -= 1
    return stripped


# 2026-05-15: _apply_anthropic_cache removed. TCMM's AnthropicRenderer
# owns cache_control placement on the system field. Conversation-history
# caching (on the messages array) is no longer applied by the proxy —
# if we need it back as a separate optimization, it gets reintroduced
# as a focused helper rather than mixed into the renderer concern.


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


# Canonical Claude-only model lineup surfaced in LibreChat's dropdown.
# ``claude-opus-4-7-1m`` is a synthetic alias: same weights as
# claude-opus-4-7 but forwarded to Anthropic with the 1M-context beta
# header (see _rewrite_claude_1m_alias below).
VEILGUARD_CLAUDE_MODELS = [
    "claude-opus-4-7",
    "claude-opus-4-7-1m",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
]

# Canonical xAI lineup surfaced in LibreChat's dropdown.  Grok 4.3 is
# the current frontier xAI model (released 2026-04-30): 1M-token
# context, $1.25 in / $2.50 out per M tokens, $0.20/M cached input
# (automatic server-side prefix caching, no client markers).  Reasoning
# is built-in and controllable via the ``reasoning_effort`` request
# param (none/low/medium/high; default low) — passes through this proxy
# unchanged because we don't strip unknown fields.
VEILGUARD_XAI_MODELS = [
    "grok-4.3",
]

# Anthropic beta header that unlocks the 1M-token context window on
# Opus 4.7.  Appended to the request's ``anthropic-beta`` header (if
# any) when the synthetic ``-1m`` alias is used.
CLAUDE_1M_BETA = "context-1m-2025-08-07"

# Anthropic beta header that enables the 1-hour TTL on cache_control
# blocks. Without it, the server treats {"ttl": "1h"} as a no-op and the
# block silently falls back to the default 5-minute TTL. We attach this
# whenever the Phase-7 two-tier cache split places a 1h-TTL marker on
# the stable region.
EXTENDED_CACHE_TTL_BETA = "extended-cache-ttl-2025-04-11"


def _ensure_extended_cache_ttl_beta(data: dict, headers: dict) -> None:
    """Idempotently add the 1h-TTL beta header to the outgoing request.

    Mutates ``headers`` in place (and normalises Anthropic-Beta casing).
    Safe to call multiple times — the header is a comma-separated list
    and we de-dup on add.
    """
    existing = headers.get("anthropic-beta") or headers.get("Anthropic-Beta") or ""
    parts = [p.strip() for p in existing.split(",") if p.strip()]
    if EXTENDED_CACHE_TTL_BETA not in parts:
        parts.append(EXTENDED_CACHE_TTL_BETA)
    headers["anthropic-beta"] = ",".join(parts)
    # Strip alternate-casing variant so we don't double-send.
    headers.pop("Anthropic-Beta", None)


def _rewrite_claude_1m_alias(data: dict, headers: dict) -> None:
    """Translate the synthetic claude-opus-4-7-1m alias.

    LibreChat's model dropdown lists ``claude-opus-4-7-1m`` as a
    separate entry so users can explicitly pick the 1M-context variant
    (matches the Claude Code model selector).  Anthropic's API doesn't
    know that ID — it's the same underlying model as ``claude-opus-4-7``
    with a beta flag.  Before forwarding we rewrite the model ID and
    merge ``CLAUDE_1M_BETA`` into the outgoing ``anthropic-beta``
    header so downstream calls succeed.

    Mutates ``data`` and ``headers`` in place.
    """
    model = data.get("model")
    if not isinstance(model, str) or not model.endswith("-1m"):
        return
    if model != "claude-opus-4-7-1m":
        return  # guard — only Opus 4.7 has a 1M variant for now
    data["model"] = "claude-opus-4-7"
    existing = headers.get("anthropic-beta") or headers.get("Anthropic-Beta") or ""
    parts = [p.strip() for p in existing.split(",") if p.strip()]
    if CLAUDE_1M_BETA not in parts:
        parts.append(CLAUDE_1M_BETA)
    headers["anthropic-beta"] = ",".join(parts)
    # Strip the alternate-casing variant so we don't double-send.
    headers.pop("Anthropic-Beta", None)


# 2026-05-19: workspace-scoped model aliases.
#
# When we rotated to a new Anthropic workspace (key …R7h…), the new
# workspace only exposes some models under their DATED IDs, not the
# bare alias. e.g. it has ``claude-haiku-4-5-20251001`` but NOT
# ``claude-haiku-4-5``. The old workspace had both. LibreChat sends
# the bare alias (per librechat.yaml's ``models`` list), Anthropic
# 404s, user sees "model not available."
#
# We can't just edit librechat.yaml to the dated name without making
# the UI dropdown ugly. Instead: rewrite the model ID on the way
# through the proxy, same trick as the 1M alias above. If/when
# Anthropic exposes the bare alias on this workspace, the dated
# rewrite is still valid (just redundant), so this is forward-safe.
#
# Map: bare alias → dated ID. Extend when more aliases drop off the
# workspace's allowlist.
_ANTHROPIC_DATED_ALIASES = {
    "claude-haiku-4-5":  "claude-haiku-4-5-20251001",
    # Add others as needed. Don't add models that already work as
    # bare aliases on the current workspace — redundant rewrites are
    # harmless but noise in the diff.
}


def _rewrite_claude_dated_alias(data: dict) -> None:
    """Map ``claude-haiku-4-5`` → ``claude-haiku-4-5-20251001`` etc.

    Mutates ``data`` in place. No header changes (unlike the 1M
    alias). Logs the rewrite at info so it shows up in the audit
    trail if we need to debug "why is the upstream model name
    different from what the UI sent."
    """
    model = data.get("model")
    if not isinstance(model, str):
        return
    target = _ANTHROPIC_DATED_ALIASES.get(model)
    if target and target != model:
        data["model"] = target
        logger.info(
            f"  [MODEL-ALIAS] rewrote {model} → {target} "
            f"(new workspace doesn't expose the bare alias)"
        )


# Stub for Anthropic model listing — LibreChat calls this during auto-discovery
@app.get("/anthropic/v1/models")
async def anthropic_models():
    """Return available Anthropic models (stub for LibreChat model discovery)."""
    return {
        "data": [
            {
                "id": mid,
                "object": "model",
                "created": 1700000000,
                "owned_by": "anthropic",
            }
            for mid in VEILGUARD_CLAUDE_MODELS
        ],
        "object": "list",
    }


# Stub for xAI model listing.  xAI's real /v1/models returns the full
# catalogue (including grok-imagine, grok-4.20 variants, etc.) — we
# whitelist just the IDs Veilguard supports so the LibreChat dropdown
# stays curated.  LibreChat's custom-endpoint config uses ``fetch:
# false`` to skip discovery entirely, but the stub is here for any
# OpenAI-SDK client that probes /models.
@app.get("/xai/v1/models")
async def xai_models():
    """Return available xAI models (stub for LibreChat model discovery)."""
    return {
        "data": [
            {
                "id": mid,
                "object": "model",
                "created": 1700000000,
                "owned_by": "xai",
            }
            for mid in VEILGUARD_XAI_MODELS
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


@app.get("/cache/stats")
async def cache_stats(
    tenant_id: str | None = None,
    window_seconds: float | None = None,
):
    """Per-tenant cache telemetry (Phase 7 step 2).

    Query params:
      tenant_id:        if set, returns stats for that tenant only.
                        Omit to return every tracked tenant.
      window_seconds:   if set, restricts aggregation to samples newer
                        than now - window_seconds. Omit for the full
                        rolling window (default 1000 most recent).

    Returned `hit_rate`, `write_rate`, `effective_token_multiplier`
    are normalised over total input tokens (cache_read + cache_create +
    uncached input). `write_amplification` is cache_create / cache_read
    — useful for spotting tenants where caching costs more than it
    saves (>= 1.0 means strictly worse than no caching).

    Useful operational queries:
      • GET /cache/stats                              — overview
      • GET /cache/stats?tenant_id=...&window_seconds=86400 — last day
    """
    from app import cache_metrics
    if tenant_id is not None:
        return cache_metrics.get_tenant_stats(tenant_id, window_seconds=window_seconds)
    return {
        "tenants": cache_metrics.get_all_stats(window_seconds=window_seconds),
    }


def _is_unsubstituted_placeholder(value: str) -> bool:
    """Detect ``{{...}}`` template placeholders that upstream forgot to substitute.

    LibreChat's MCP SSE transport and (in some code paths) its main
    Anthropic client forward header/metadata values containing
    ``{{LIBRECHAT_BODY_CONVERSATIONID}}`` / ``{{LIBRECHAT_USER_ID}}``
    literally when ``processMCPEnv`` runs with an empty requestBody.
    We treat those as absent so downstream TCMM namespaces and
    ``pii_audit.user_id`` don't get a string full of braces as an
    identifier — that's how you end up with a ``default`` tenant
    accidentally holding another user's blocks.
    """
    return isinstance(value, str) and value.startswith("{{") and value.endswith("}}")


def _clean_conv_id(value: str, skip: set) -> str:
    """Return the value if it looks like a real conv id; ``""`` otherwise."""
    if not value or value in skip or _is_unsubstituted_placeholder(value):
        return ""
    return value


def extract_conversation_id(data: dict, headers: dict) -> str:
    """Extract conversation ID for TCMM session tracking.

    LibreChat sends conversationId="new" on the first message of a new chat.
    We skip that and fall through to generate a stable temporary ID.
    On the second message, LibreChat sends the real UUID.

    Also rejects unsubstituted ``{{LIBRECHAT_BODY_...}}`` placeholders
    (see ``_is_unsubstituted_placeholder``) at every lookup layer —
    otherwise they leak into namespace/user_id and corrupt tenancy.
    """
    # 2026-05-19: added the all-zeros UUID to the skip set. LibreChat
    # sends ``parentMessageId="00000000-0000-0000-0000-000000000000"``
    # as the root-parent sentinel on every fresh chat — without this,
    # layer 4 below would return ``parent-00000000-0000-0000-0000`` for
    # the first turn of EVERY new chat from any user, collapsing them
    # all into a single TCMM session.
    _skip = {
        "", "new", "null", "undefined", "None",
        "00000000-0000-0000-0000-000000000000",
    }

    # 1. Explicit headers
    conv_id = _clean_conv_id(
        headers.get("x-conversation-id") or headers.get("x-request-id") or "",
        _skip,
    )
    if conv_id:
        return conv_id

    # 2. From Anthropic metadata (LibreChat patched)
    metadata = data.get("metadata", {}) or {}
    conv_id = _clean_conv_id(metadata.get("conversation_id", ""), _skip)
    if conv_id:
        return conv_id

    # 3. From request body
    conv_id = _clean_conv_id(
        data.get("conversationId") or data.get("conversation_id") or "",
        _skip,
    )
    if conv_id:
        return conv_id

    # 4. parent_message_id (stable across the conversation)
    parent_id = _clean_conv_id(
        data.get("parentMessageId") or data.get("parent_message_id") or "",
        _skip,
    )
    if parent_id:
        return f"parent-{parent_id[:24]}"

    # 5. Stable derivation from (user_id, first-user-message).
    #
    # LibreChat doesn't reliably forward conv_id for every endpoint —
    # claude-sonnet-4-6 (and Grok / xAI / OpenAI custom) hit the proxy
    # with NO x-conversation-id header, no metadata.conversation_id,
    # and no conversationId in the body. Anchoring on
    # hash(user_id + first_user_msg) gives a stable ID across every
    # turn of the same UI chat, because the first user message never
    # changes within a thread → same hash → same TCMM session →
    # memory accumulates → cache_rd grows turn-to-turn.
    #
    # 2026-05-19 brief detour: tried per-request UUID synthesis for
    # "Anthropic-style" requests to avoid the cross-chat collision
    # bug (two chats opened with "helo" hashed to the same session).
    # That regression broke claude-sonnet-4-6's TCMM continuity
    # entirely — every turn got a fresh UUID, fresh session, no
    # memory accumulation, cache_rd stuck at 13,864 (preamble only).
    # Reverted: collisions are a less severe failure mode than cache
    # death. The cross-chat collision will be addressed via a richer
    # hash input (e.g. include parentMessageId when LibreChat sends
    # something other than the all-zeros sentinel) in a follow-up,
    # not by sacrificing within-chat continuity.
    #
    # 2026-05-13 original commit: previously this layer was a fresh
    # uuid4() per request, which spawned a new TCMM session pool
    # entry every turn and made live conversation memory effectively
    # useless for Grok. See pii_audit for the receipts: 26 rows in
    # 6h, every one a unique conv_id under the same user_id +
    # model=grok-4.3.
    user_id = extract_user_id(data, headers)
    if user_id and _is_unsubstituted_placeholder(user_id):
        user_id = ""

    first_user_msg = ""
    messages = data.get("messages", [])
    if isinstance(messages, list):
        for m in messages:
            if not isinstance(m, dict) or m.get("role") != "user":
                continue
            content = m.get("content", "")
            if isinstance(content, str):
                first_user_msg = content
            elif isinstance(content, list):
                # OpenAI multi-part content: [{"type":"text","text":...}, ...]
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        first_user_msg = blk.get("text", "") or ""
                        if first_user_msg:
                            break
            if first_user_msg:
                break

    if user_id and first_user_msg:
        h = hashlib.sha1(
            f"{user_id}|{first_user_msg}".encode("utf-8", "replace")
        ).hexdigest()
        synth = f"conv-{user_id[:8]}-{h[:10]}"
        logger.info(
            f"  [CONV] L5 hash-synth: {synth} "
            f"(user_id={user_id[:12]}, first_msg={first_user_msg[:30]!r})"
        )
        # Fits TCMM SessionPool's 24-char truncation: 5+8+1+10 = 24.
        return synth

    # 6. Fallback: user_id without a parseable first message — better
    # than a pure uuid4 but still per-request (rare path).
    if user_id:
        # 2026-05-19 debug: log WHY we fell through layer 5. The user
        # is observing every tool-followup get a fresh layer-6 UUID
        # because first_user_msg extraction is returning empty even
        # when the audit content clearly shows the original user turn.
        # Dump the message shape so we can spot whether content is
        # bytes / dict / tool_result-only / etc.
        _msg_shapes = []
        for _i, _m in enumerate(messages[:6] if isinstance(messages, list) else []):
            if not isinstance(_m, dict):
                _msg_shapes.append(f"[{_i}]non-dict")
                continue
            _r = _m.get("role", "?")
            _c = _m.get("content")
            if isinstance(_c, str):
                _msg_shapes.append(f"[{_i}]{_r}:str({len(_c)})")
            elif isinstance(_c, list):
                _types = []
                for _b in _c[:4]:
                    if isinstance(_b, dict):
                        _types.append(_b.get("type", "?"))
                    else:
                        _types.append(type(_b).__name__)
                _msg_shapes.append(f"[{_i}]{_r}:list[{','.join(_types)}]")
            else:
                _msg_shapes.append(f"[{_i}]{_r}:{type(_c).__name__}")
        synth = f"new-{user_id[:16]}-{uuid.uuid4().hex[:8]}"
        logger.warning(
            f"  [CONV] L6 fallback (first_user_msg empty): {synth} "
            f"user_id={user_id[:12]} n_msgs={len(messages) if isinstance(messages, list) else 'N/A'} "
            f"shapes={_msg_shapes}"
        )
        return synth

    # 7. No anchor at all — legacy behaviour, kept so we never crash.
    return str(uuid.uuid4())


def extract_user_id(data: dict, headers: dict) -> str:
    """Extract the LibreChat user_id from a request, trying multiple sources.

    LibreChat populates ``metadata.user_id`` only on some endpoints. First-
    message requests and /chat/completions from certain flows may omit it,
    leaving audit rows (and downstream TCMM namespaces) stamped with an
    empty tenant. We fall back through:

        1. headers['x-user-id']    — MCP/header convention
        2. data.metadata.user_id   — LibreChat's patched Anthropic SDK
        3. data.user_id / userId   — top-level body key (some endpoints)
        4. data.metadata.user      — alternate LibreChat metadata shape
        5. "" (empty)              — last resort

    Unsubstituted ``{{LIBRECHAT_USER_ID}}`` template literals are stripped
    at every layer so they never poison the audit log or TCMM namespace.
    """
    # 1. Explicit header (MCP convention)
    h_uid = headers.get("x-user-id", "") or ""
    if h_uid and not _is_unsubstituted_placeholder(h_uid):
        return h_uid

    # 2. Anthropic-metadata (LibreChat patched SDK)
    metadata = data.get("metadata", {}) or {}
    m_uid = metadata.get("user_id", "") or ""
    if m_uid and not _is_unsubstituted_placeholder(m_uid):
        return m_uid

    # 3. Top-level body key (some LibreChat routes put it here)
    for key in ("user_id", "userId", "user"):
        b_uid = data.get(key, "")
        # Skip if it's a dict (Anthropic "user" metadata block)
        if isinstance(b_uid, str) and b_uid and not _is_unsubstituted_placeholder(b_uid):
            return b_uid

    # 4. Alternate metadata shape
    alt = metadata.get("user", "")
    if isinstance(alt, str) and alt and not _is_unsubstituted_placeholder(alt):
        return alt

    return ""


def extract_pii_session_id(data: dict) -> str:
    """Extract PII session ID — always per-user so token mappings are consistent
    across all conversations for the same user. This ensures REF_PERSON_2 always
    maps to the same person regardless of which conversation it appears in.

    Rejects unsubstituted ``{{LIBRECHAT_USER_ID}}`` placeholders so
    multiple users don't silently share the same ``pii-{{LIBRECHAT_``
    session and cross-contaminate each other's redacted tokens."""
    metadata = data.get("metadata", {}) or {}
    user_id = metadata.get("user_id", "")
    if user_id and not _is_unsubstituted_placeholder(user_id):
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


# ── Iter 11: SSO routing (2026-05-19) ───────────────────────────────
# When data["model"] ends in "-sso", bypass api.anthropic.com and call
# TCMM /generate (which routes through AnthropicGenerationAdapter -> Claude
# CLI pool with the user's Max subscription credentials). Reuses TCMM's
# renderer for memory prefix + CLI's automatic prompt caching.
#
# The "-sso" suffix is stripped before calling /generate so the underlying
# model name (e.g. claude-haiku-4-5-20251001) is what claude CLI sees.

def _is_sso_model(model) -> bool:
    # [SSO_DEFAULT_ENV_2026_05_20] honor CLAUDE_SSO_DEFAULT=1 so all
    # claude-* models route through SSO without needing -sso suffix.
    # LibreChat sends bare model names so this is required.
    if not isinstance(model, str):
        return False
    if model.endswith("-sso"):
        return True
    if os.environ.get("CLAUDE_SSO_DEFAULT", "").strip().lower() in ("1", "true", "yes"):
        return model.lower().startswith("claude-")
    return False


def _extract_user_message(data: dict) -> str:
    """Pull the latest user-role message text out of an Anthropic-shaped
    request body. Handles both string content and list-of-blocks content.
    """
    messages = data.get("messages") or []
    if not isinstance(messages, list):
        return ""
    for m in reversed(messages):
        if not isinstance(m, dict):
            continue
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    txt = blk.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
            return "\n".join(parts)
    return ""


def _sso_anthropic_nonstream(text: str, model: str, usage: dict | None = None,
                              content: list | None = None,
                              stop_reason: str | None = None) -> dict:
    """Build an Anthropic-shaped non-stream response from a plain string.

    [SSO_USAGE_PASSTHROUGH 2026-05-20] If ``usage`` is provided (from
    TCMM /generate's adapter.last_usage), forward the real token counts
    (input/output/cache_creation/cache_read) so LibreChat's cost
    tracking sees what the model actually billed.
    """
    import time as _t
    _u = usage or {}
    return {
        "id":           f"msg_sso_{int(_t.time()*1000)}",
        "type":         "message",
        "role":         "assistant",
        "model":        model,
        # [TOOL_USE_RESPONSE_2026_05_20] use full block array from
        # the adapter when present (carries tool_use blocks); fall
        # back to single-text-block for pure-text responses.
        "content":      (content if content else [{"type": "text", "text": text}]),
        "stop_reason":  (stop_reason or "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens":                 _u.get("input_tokens"),
            "output_tokens":                _u.get("output_tokens"),
            "cache_creation_input_tokens":  _u.get("cache_creation_input_tokens"),
            "cache_read_input_tokens":      _u.get("cache_read_input_tokens"),
        },
    }


def _sso_anthropic_stream_chunks(text: str, model: str, usage: dict | None = None,
                                  content: list | None = None,
                                  stop_reason: str | None = None):
    """Yield Anthropic-SSE-compatible chunks for a fake stream.

    [STREAM_TOOL_USE_2026_05_20] Now supports tool_use blocks in
    addition to text. When ``content`` is provided (list of blocks
    from the adapter), iterates per-block and emits the appropriate
    SSE events for each. Falls back to single text-block emission
    when only ``text`` is provided (older callers).

    Anthropic SSE shape per block:
      text:
        content_block_start{type:text, text:\"\"}
        content_block_delta{delta:{type:text_delta, text:<chunk>}}
        content_block_stop
      tool_use:
        content_block_start{type:tool_use, id, name, input:{}}
        content_block_delta{delta:{type:input_json_delta, partial_json:<json>}}
        content_block_stop
    """
    import json as _json, time as _t
    msg_id = f"msg_sso_{int(_t.time()*1000)}"
    # Determine blocks to emit
    if content and isinstance(content, list):
        blocks = content
    else:
        blocks = [{"type": "text", "text": text or ""}]

    base_msg = {
        "id":           msg_id,
        "type":         "message",
        "role":         "assistant",
        "model":        model,
        "content":      [],
        "stop_reason":  None,
        "stop_sequence": None,
        "usage":        {
            "input_tokens":                 (usage or {}).get("input_tokens") or 0,
            "output_tokens":                (usage or {}).get("output_tokens") or 0,
            "cache_creation_input_tokens":  (usage or {}).get("cache_creation_input_tokens"),
            "cache_read_input_tokens":      (usage or {}).get("cache_read_input_tokens"),
        },
    }
    yield f"event: message_start\ndata: {_json.dumps({'type':'message_start','message':base_msg})}\n\n"

    for idx, blk in enumerate(blocks):
        btype = blk.get("type") if isinstance(blk, dict) else None
        if btype == "tool_use":
            # Emit tool_use start with empty input ({}), then a
            # single input_json_delta with the full input as JSON.
            tu_id   = blk.get("id") or f"toolu_sso_{idx}_{int(_t.time()*1000)}"
            tu_name = blk.get("name") or ""
            tu_input = blk.get("input") or {}
            yield (
                "event: content_block_start\n"
                f"data: {_json.dumps({'type':'content_block_start','index':idx,'content_block':{'type':'tool_use','id':tu_id,'name':tu_name,'input':{}}})}\n\n"
            )
            yield (
                "event: content_block_delta\n"
                f"data: {_json.dumps({'type':'content_block_delta','index':idx,'delta':{'type':'input_json_delta','partial_json':_json.dumps(tu_input)}})}\n\n"
            )
            yield (
                "event: content_block_stop\n"
                f"data: {_json.dumps({'type':'content_block_stop','index':idx})}\n\n"
            )
        else:
            # text block (or unknown; treat as text)
            btext = blk.get("text", "") if isinstance(blk, dict) else str(blk or "")
            yield (
                "event: content_block_start\n"
                f"data: {_json.dumps({'type':'content_block_start','index':idx,'content_block':{'type':'text','text':''}})}\n\n"
            )
            yield (
                "event: content_block_delta\n"
                f"data: {_json.dumps({'type':'content_block_delta','index':idx,'delta':{'type':'text_delta','text':btext}})}\n\n"
            )
            yield (
                "event: content_block_stop\n"
                f"data: {_json.dumps({'type':'content_block_stop','index':idx})}\n\n"
            )

    _final_stop = stop_reason or "end_turn"
    _out_tok    = (usage or {}).get("output_tokens") or len((text or "").split())
    yield (
        "event: message_delta\n"
        f"data: {_json.dumps({'type':'message_delta','delta':{'stop_reason':_final_stop,'stop_sequence':None},'usage':{'output_tokens':_out_tok}})}\n\n"
    )
    yield "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"


def _sso_audit_record(direction, conv_id, user_id, model, stream, content,
                      tokens_input=None, tokens_output=None,
                      cache_create=None, cache_read=None) -> None:
    """Iter 20: write a pii_audit row for an SSO turn. Best-effort; never
    raises out of the request handler. Same DB the dashboard reads from."""
    try:
        from app import audit_db as _audit_db
        _audit_db.record(
            direction=direction,
            conversation_id=conv_id or "",
            user_id=user_id or "",
            model=model,
            stream=bool(stream),
            content=content or "",
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cache_create=cache_create,
            cache_read=cache_read,
            extra={"path": "sso"},
        )
    except Exception as e:
        logger.warning("[SSO] audit_db record failed: %s", e)


async def _sso_pre_request(conv_id: str, user_id: str, user_msg: str) -> None:
    """Best-effort: ingest the user message so /generate's renderer sees
    it on this turn. Mirror what pii-proxy does for normal Anthropic flow
    just before forwarding to api.anthropic.com.
    Fire-and-forget; logs at debug on failure."""
    if not conv_id or not user_msg:
        return
    try:
        client = _get_tcmm_client()
        body = {
            "user_message":    user_msg,
            "conversation_id": conv_id,
            "user_id":         user_id or "",
            "recall_only":     False,  # ingest the user msg, not just recall
            "origin":          "user",
        }
        await client.post(f"{TCMM_URL}/pre_request", json=body, timeout=30)
    except Exception as e:
        logger.debug("[SSO] pre_request failed: %s", e)


async def _sso_post_response(conv_id: str, user_id: str,
                              assistant_text: str, model: str,
                              flag_obj: dict | None = None) -> None:
    """Best-effort: ingest the assistant's reply so the next turn's
    render sees it as recent memory. Fire-and-forget.

    `flag_obj`: shadow-tool capture from the just-completed turn
    (used/knowledge_class/epoch_complete/emit_class). Passing it
    through to /post_response is what makes block_class actually
    land in the archive — without it the adapter falls back to
    prose-JSON parsing which the shadow-tool path never produces.
    """
    if not conv_id or not assistant_text:
        return
    try:
        client = _get_tcmm_client()
        body = {
            "raw_output":      assistant_text,
            "conversation_id": conv_id,
            "user_id":         user_id or "",
            "origin":          "assistant_text",
        }
        if flag_obj and isinstance(flag_obj, dict):
            body["flag_obj"] = flag_obj
        await client.post(f"{TCMM_URL}/post_response", json=body, timeout=30)
    except Exception as e:
        logger.debug("[SSO] post_response failed: %s", e)


# ════════════════════════════════════════════════════════════════════
# [SHADOW_TOOL_TCMM_RECORD_2026_05_20] TCMM turn-record shadow tool
# ════════════════════════════════════════════════════════════════════
# Instead of having the model append JSON prose at the end of every
# answer (which requires regex stripping for the user-facing response
# and is fragile to malformed output), we inject this synthetic tool
# into the request. The model emits the metadata as a structured
# `tool_use` block — schema-validated by the API — and the proxy
# intercepts it BEFORE forwarding to LibreChat. LibreChat never sees
# the tool, so it doesn't try to dispatch it as a real MCP tool.
#
# Provider-uniform: Anthropic, OpenAI, Grok all support tool_use
# with this exact schema shape.

# [DEDUP_SHADOW_TOOL_2026-06-03] Single source of truth. The canonical
# tcmm_record_turn schema lives in agent/shadow_tool.py — the LIVE chat-agent
# path uses it. This (mostly-dormant) SSO path used to keep a SECOND inline
# copy, and the two DRIFTED: the emit_class enums diverged, so post_response
# rejected every class except "decision" and nothing got classified. Importing
# the one schema kills that whole class of bug. agent/ is mounted as a sibling
# of pii/ (PYTHONPATH=/ in the container; the sys.path probe above also puts
# the repo root on the path in local dev), so this resolves at module load
# exactly like the `from pii import ...` on line 52.
from agent.shadow_tool import TCMM_RECORD_TURN_TOOL as _TCMM_RECORD_TURN_TOOL


def _inject_tcmm_record_tool(tools_list):
    """Prepend the TCMM record-turn shadow tool to the user-provided
    tools list (or create a new list). Returns the augmented list.
    Idempotent — won't double-inject if the tool is already present."""
    out = list(tools_list) if tools_list else []
    for t in out:
        if isinstance(t, dict) and t.get("name") == "tcmm_record_turn":
            return out
    return [_TCMM_RECORD_TURN_TOOL] + out


# [UNIVERSAL_SHADOW_TOOL_2026_05_22] OpenAI/xAI function-shape variant.
# Same schema (the shadow tool's contract is provider-agnostic) wrapped
# in the OpenAI function-calling envelope. The "parameters" field maps
# to Anthropic's "input_schema" 1:1 — both follow JSON Schema.
_TCMM_RECORD_TURN_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name":        _TCMM_RECORD_TURN_TOOL["name"],
        "description": _TCMM_RECORD_TURN_TOOL["description"],
        "parameters":  _TCMM_RECORD_TURN_TOOL["input_schema"],
    },
}


def _inject_tcmm_record_tool_openai(tools_list):
    """OpenAI/xAI variant of _inject_tcmm_record_tool. Idempotent."""
    out = list(tools_list) if tools_list else []
    for t in out:
        if isinstance(t, dict):
            _fn = t.get("function") or {}
            if (_fn.get("name") or t.get("name")) == "tcmm_record_turn":
                return out
    return [_TCMM_RECORD_TURN_TOOL_OPENAI] + out


def _inject_shadow_tool_for_backend(data: dict, render_model: str) -> bool:
    """Universal shadow-tool injection. Mutates `data["tools"]` in place
    with the right shape for the backend (`render_model` is the value
    used by the gateway's renderer dispatch: "anthropic" / "openai" /
    "grok"). Returns True if injection happened, False otherwise.

    Idempotent — re-running on a request that already has the shadow
    tool is a no-op (helpers check by name).

    Also nudges `tool_choice` so the model actually consumes the tool:
      * If no real tools present (only our shadow tool) → set a
        provider-specific value that forces a tool call. This guarantees
        the model invokes tcmm_record_turn for chat-only turns where it
        otherwise just text-responds and skips classification.
      * If the user already set tool_choice OR has real tools alongside,
        we LEAVE tool_choice alone — real MCP tool dispatch must not be
        forced. The model's behavior for those turns degrades to "calls
        tcmm_record_turn alongside the real tool when it feels like it"
        — same ~60% rate we saw on SSO claude. A future iteration can
        enable parallel_tool_calls + a stronger prompt nudge.
    """
    if not isinstance(data, dict):
        return False
    current = data.get("tools")
    current = list(current) if isinstance(current, list) else []
    _had_real_tools = len(current) > 0
    if render_model == "anthropic":
        new_tools = _inject_tcmm_record_tool(current)
    else:
        # Both "openai" and "grok" use OpenAI function-calling shape.
        new_tools = _inject_tcmm_record_tool_openai(current)
    data["tools"] = new_tools

    # tool_choice forcing — only when client didn't set one AND we're
    # the only tool available. Anything else risks breaking real tool
    # dispatch or overriding explicit client intent.
    if "tool_choice" not in data and not _had_real_tools:
        if render_model == "anthropic":
            # "any" = call at least one tool (with only ours present,
            # this forces tcmm_record_turn). disable_parallel_tool_use=False
            # is the default — explicit for forward-compat clarity.
            data["tool_choice"] = {"type": "any", "disable_parallel_tool_use": False}
        else:
            # OpenAI / xAI: "required" = must call at least one function.
            # With only our tool present, the model must call it.
            data["tool_choice"] = "required"
    return True


def _strip_tcmm_tool_narration(text: str) -> str:
    """Remove Grok-style prose narrations of the tcmm_record_turn call
    from a response's text content. The model is supposed to call the
    tool silently; some providers (Grok especially) narrate it in
    prose anyway — sometimes describing the call ("call tool
    tcmm_record_turn with knowledge_class is derived..."), sometimes
    refusing it ("tcmm_record_turn is not a real tool..."), sometimes
    just dropping the bare name.

    Strategy: locate the FIRST mention of the tool name and truncate
    the text at the most-recent sentence boundary before it. The
    actual answer always comes before any tool-related narration, so
    this preserves the answer while removing the noise. If no sentence
    boundary exists before the mention (the whole content is just
    narration), return empty so the answer is just blank rather than
    leaking the tool name.

    Idempotent — re-running on already-scrubbed text is a no-op
    because the tool name won't be present."""
    if not text or "tcmm_record_turn" not in text:
        return text
    # First mention of the tool name
    idx = text.find("tcmm_record_turn")
    if idx <= 0:
        # Mention is at the very start — no answer to preserve
        return ""
    # Walk backwards from idx to the nearest sentence-end boundary
    # (period+space, newline, or run of whitespace). Keep everything
    # up to and including the punctuation if found, otherwise cut at
    # the boundary itself.
    prefix = text[:idx]
    # Common boundaries: "\n", ". ", "! ", "? ", or trailing whitespace
    for boundary in ("\n", ". ", "! ", "? "):
        last = prefix.rfind(boundary)
        if last >= 0:
            cut = last + len(boundary)
            return text[:cut].rstrip() or ""
    # No sentence boundary — fall back to just trimming trailing space
    # before the mention.
    return prefix.rstrip() or ""


def _extract_shadow_tool_from_openai_response(payload: dict) -> tuple[dict, bool]:
    """Parse a NON-STREAMING OpenAI/xAI response. Look for a
    `choices[*].message.tool_calls[*]` with function.name=tcmm_record_turn,
    extract the JSON arguments, strip the entry from the response so the
    downstream client never sees it. Also scrubs any Grok-style prose
    narration of the tool call from message.content.
    Returns (flag_obj, was_modified).
    """
    if not isinstance(payload, dict):
        return {}, False
    flag_obj: dict = {}
    modified = False
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return {}, False
    for ch in choices:
        if not isinstance(ch, dict):
            continue
        msg = ch.get("message")
        if not isinstance(msg, dict):
            continue
        # [GROK_TOOL_NARRATION_STRIP_2026_05_22] scrub prose-form
        # narration of the shadow tool from message.content regardless
        # of whether tool_calls is present (Grok sometimes emits it
        # even when also doing the function-call envelope correctly).
        _content = msg.get("content")
        if isinstance(_content, str) and "tcmm_record_turn" in _content:
            _scrubbed = _strip_tcmm_tool_narration(_content)
            if _scrubbed != _content:
                msg["content"] = _scrubbed
                modified = True
        tcs = msg.get("tool_calls")
        if not isinstance(tcs, list):
            continue
        kept: list = []
        for tc in tcs:
            if not isinstance(tc, dict):
                kept.append(tc)
                continue
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name")
            if name == "tcmm_record_turn":
                # Capture args (JSON string per OpenAI shape; dict per
                # some legacy / direct paths).
                args = fn.get("arguments")
                _parsed = {}
                if isinstance(args, str):
                    try:
                        _parsed = json.loads(args) or {}
                    except Exception:
                        _parsed = {}
                elif isinstance(args, dict):
                    _parsed = args
                if isinstance(_parsed, dict):
                    flag_obj = _parsed
                modified = True
                continue  # drop this tool_call from forwarded output
            kept.append(tc)
        if modified:
            msg["tool_calls"] = kept
            # If kept is empty AND finish_reason was "tool_calls", we
            # need to downgrade it so the client doesn't wait forever
            # for a tool_result that won't come.
            if not kept and ch.get("finish_reason") == "tool_calls":
                ch["finish_reason"] = "stop"
    return flag_obj, modified


def _intercept_tcmm_record_tool_use(content_blocks, stop_reason):
    """Walk through content blocks, extract any `tcmm_record_turn`
    tool_use, and return:
      - cleaned blocks (with the shadow tool_use removed)
      - flag_obj_dict (the input that was captured, or {})
      - new_stop_reason: if removing the shadow tool_use leaves no
        real tool_use blocks, downgrade "tool_use" → "end_turn" so
        LibreChat doesn't wait for a tool_result on a tool we just
        consumed internally.
    """
    if not isinstance(content_blocks, list):
        return content_blocks, {}, stop_reason
    cleaned = []
    flag_obj = {}
    for b in content_blocks:
        if (isinstance(b, dict)
            and b.get("type") == "tool_use"
            and b.get("name") == "tcmm_record_turn"):
            _inp = b.get("input")
            if isinstance(_inp, dict):
                flag_obj = _inp
            continue  # drop it from forwarded content
        cleaned.append(b)
    new_stop = stop_reason
    if stop_reason == "tool_use":
        # If no other tool_use remains, the turn is logically complete.
        has_real_tool_use = any(
            isinstance(b, dict) and b.get("type") == "tool_use"
            for b in cleaned
        )
        if not has_real_tool_use:
            new_stop = "end_turn"
    return cleaned, flag_obj, new_stop


async def _handle_sso_request(
    data: dict, conversation_id: str, user_id: str, is_stream: bool,
):
    """Route an Anthropic-shaped request to TCMM /generate via SSO.

    Returns a Response (StreamingResponse if is_stream else JSONResponse).
    """
    from fastapi.responses import JSONResponse, StreamingResponse
    user_msg = _extract_user_message(data)
    if not user_msg.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "no user message found in request"},
        )
    # Only strip "-sso" suffix when it's actually present; in CLAUDE_SSO_DEFAULT mode
    # the model name comes through as bare (e.g. claude-sonnet-4-6) and must stay bare.
    _m = data["model"]
    real_model = _m[:-4] if _m.endswith("-sso") else _m

    # [SSO_PINNING_2026_05_20] Pin Veilguard preamble + client_system
    # + tool definitions to TCMM's immutable tier BEFORE rendering so
    # /generate's renderer produces a cacheable prefix that starts
    # with the Claude-API-compliant magic line and carries the actual
    # tool schemas LibreChat sent.
    try:
        _client_system_text = _extract_client_system(data, "anthropic")
    except Exception:
        _client_system_text = ""
    try:
        _client_tools_list = _extract_client_tools(data)
    except Exception:
        _client_tools_list = []

    # [PIN_ORDER_FIX_2026_05_20] SEQUENTIAL pin — preamble FIRST so it
    # gets the lowest TCMM block_id (bid=0). That determines the order
    # in the rendered system array. Anthropic requires the first text
    # block to start with the Claude-API-compliant identity string,
    # which lives at the head of the preamble. asyncio.gather() raced
    # and let client_system land at bid=0, breaking the gate.
    try:
        _preamble_with_tools = _render_preamble_with_tools(_client_tools_list)
        await _tcmm_pin_system_prompt(
            conversation_id, _preamble_with_tools,
            kind="veilguard_preamble",
            user_id=user_id or "",
        )
        if _client_system_text:
            await _tcmm_pin_system_prompt(
                conversation_id, _client_system_text,
                kind="client_system",
                user_id=user_id or "",
            )
        if _client_tools_list:
            try:
                await _tcmm_pin_tool_definitions(
                    conversation_id, _client_tools_list,
                    user_id=user_id or "",
                )
            except Exception as _e_t:
                # /pin/tool_definitions may 404 on older TCMM builds — fine,
                # schemas are already in the preamble + the API tools field.
                logger.debug("[SSO] tool_definitions pin skipped: %s", _e_t)
    except Exception as _e:
        logger.warning("[SSO] pin step failed (continuing): %s", _e)

    # Iter 18 (2026-05-19): ingest user msg first so /generate's render
    # sees it as memory on this very turn.
    try:
        await _sso_pre_request(conversation_id, user_id, user_msg)
    except Exception as _e:
        logger.debug("[SSO] pre_request inline failed: %s", _e)

    # Iter 19 (2026-05-19): redact PII before /generate so claude CLI
    # (Max subscription) never sees raw PII. Same boundary contract as
    # the normal Anthropic forward path. # _handle_sso_request_redacted
    _pii_sid = _resolve_pii_session_id(
        conversation_id, user_id,
    ) if '_resolve_pii_session_id' in globals() else (
        conversation_id or user_id or "pii-default"
    )
    try:
        _redactor = get_redactor()
        _redacted_msg = _redactor.redact_text(user_msg, _pii_sid)
    except RedactionUnavailable as _re:
        logger.error("[SSO] redaction unavailable, refusing forward: %s", _re)
        from fastapi.responses import JSONResponse as _JR
        return _JR(status_code=503, content={
            "error": {"type":"redaction_unavailable","message":str(_re)},
        })
    except Exception as _re:
        logger.warning("[SSO] redact_text raised, falling back to raw msg: %s", _re)
        _redacted_msg = user_msg

    # [TO_LLM_FULL_PROMPT_AUDIT_2026_05_20] TO_LLM audit moved to
    # AFTER /generate returns — see below — so the row contains the
    # FULL prompt (system + memory + user) instead of just [USER].

    # [TOOLS_THROUGH_SSO_2026_05_20] Extract Anthropic-shape tool schemas
    # from the inbound LibreChat request and forward to /generate. LibreChat's
    # Agents framework attaches these for Anthropic; the previous SSO path
    # dropped them, so the model could only describe tools in prose.
    _sso_tools = _extract_client_tools(data) if "_extract_client_tools" in globals() else (
        data.get("tools") if isinstance(data.get("tools"), list) else None
    )
    # [SHADOW_TOOL_TCMM_RECORD_2026_05_20] inject the turn-record shadow
    # tool so the model emits metadata via tool_use instead of appended
    # prose JSON. The intercept later strips the matching block before
    # forwarding to LibreChat.
    _sso_tools = _inject_tcmm_record_tool(_sso_tools)
    # [WORKSPACE_STATE_SSO_2026_05_20] Fetch the user's live workspace
    # state (folders/OS/client_id from the connected daemon) and render
    # it via the existing provider-agnostic helper. Forward to /generate
    # which will append it as a final uncached system block.
    _sso_workspace_text = ""
    try:
        _sso_ws_state = await _fetch_workspace_state(user_id) if user_id else None
        if _sso_ws_state:
            _sso_workspace_text = _render_workspace_block(_sso_ws_state) or ""
    except Exception as _wse:
        logger.debug("[SSO] workspace state fetch failed: %s", _wse)
    body = {
        "user_message":    _redacted_msg,
        "model":           real_model,
        "conversation_id": conversation_id or "",
        "user_id":         user_id or "",
        "include_memory":  True,
        "task_query":      _redacted_msg,
        "label":           f"sso:{real_model}",
        "tools":           _sso_tools or None,
        "workspace_block": _sso_workspace_text or None,
    }
    try:
        client = _get_tcmm_client()
        resp = await client.post(f"{TCMM_URL}/generate", json=body, timeout=180)
    except Exception as e:
        logger.error("[SSO] /generate request failed: %s", e)
        return JSONResponse(status_code=502, content={"error": f"tcmm /generate unreachable: {e}"})

    if resp.status_code != 200:
        logger.error("[SSO] /generate returned %d: %s", resp.status_code, resp.text[:200])
        # [SSO_STREAM_ERROR_2026_05_20] When stream=True, LibreChat is
        # waiting for SSE events. Returning plain JSON makes the UI
        # spin forever. Emit a proper Anthropic SSE error event so the
        # client surfaces the failure to the user.
        if is_stream:
            import re as _re
            _detail = resp.text or ""
            # Try to extract a clean human message from the nested
            # `adapter.generate failed: Error code: 429 - {...}` string
            _err_type = "api_error"
            _err_msg  = _detail[:300]
            if "rate_limit_error" in _detail or " 429" in _detail:
                _err_type = "rate_limit_error"
                _err_msg  = (
                    "Anthropic rate limit hit for this model. "
                    "Your Max plan's per-window Opus quota is exhausted — "
                    "try claude-haiku-4-5 or claude-sonnet-4-6, or wait "
                    "a few hours for the Opus quota window to reset."
                )
            elif "authentication_error" in _detail or " 401" in _detail:
                _err_type = "authentication_error"
                _err_msg  = "OAuth credentials expired or invalid."
            elif "invalid_request_error" in _detail or " 400" in _detail:
                _err_type = "invalid_request_error"
                # Surface the inner message if present
                _m = _re.search(r"'message':\s*'([^']+)'", _detail)
                if _m: _err_msg = _m.group(1)
            def _err_stream():
                import json as _json
                # Anthropic SSE error event shape
                yield (
                    "event: error\n"
                    f"data: {_json.dumps({'type':'error','error':{'type':_err_type,'message':_err_msg}})}\n\n"
                )
            return StreamingResponse(
                _err_stream(), media_type="text/event-stream",
                status_code=200,  # HTTP layer ok, error is inside the stream
            )
        return JSONResponse(status_code=resp.status_code, content={"error": resp.text})

    j = resp.json()
    # [SHADOW_TOOL_TCMM_RECORD_2026_05_20] Intercept the model\'s
    # tcmm_record_turn tool_use BEFORE forwarding the response. Capture
    # its input as flag_obj_from_tool and remove the block from content
    # so LibreChat never sees it.
    _gen_content      = j.get("content") if isinstance(j.get("content"), list) else None
    _gen_stop_reason  = j.get("stop_reason")
    _gen_cleaned_content, _flag_obj_from_tool, _gen_stop_reason = (
        _intercept_tcmm_record_tool_use(_gen_content, _gen_stop_reason)
        if _gen_content is not None else (None, {}, _gen_stop_reason)
    )
    if _flag_obj_from_tool:
        logger.debug(
            "[SSO] captured tcmm_record_turn: knowledge_class=%s epoch_complete=%s emit_class=%s",
            _flag_obj_from_tool.get("knowledge_class"),
            _flag_obj_from_tool.get("epoch_complete"),
            _flag_obj_from_tool.get("emit_class"),
        )

    # [STRIP_HEATMAP_KEEP_RAW_2026_05_20] kept as a fallback. If the
    # model also appended prose JSON (legacy contract), strip it from
    # the user-facing text and pass the raw form to TCMM. If the model
    # ONLY used the shadow tool (new path), text_raw is already clean.
    text_raw_for_tcmm = j.get("text") or ""
    # [SHADOW_TOOL_AUTHORITATIVE_2026_05_20] If the shadow tool fired,
    # its input is authoritative — schema-validated, includes the
    # required emit_class. The model often ALSO appends a prose JSON
    # at the end of the text (training residue from the old contract);
    # that prose JSON lacks emit_class and would steal the parse.
    # So: strip any prose JSON first, then append the shadow tool\'s
    # JSON. When the shadow tool did NOT fire, leave prose JSON in
    # place as the parsing fallback.
    if _flag_obj_from_tool:
        import json as _jdump
        try:
            # Drop any prose heatmap JSON from the raw text — shadow
            # tool input replaces it as TCMM\'s sole flag_obj source.
            text_raw_for_tcmm = _strip_heatmap_from_text(text_raw_for_tcmm) or ""
            _shadow_json = _jdump.dumps(_flag_obj_from_tool, separators=(",", ":"))
            text_raw_for_tcmm = (text_raw_for_tcmm.rstrip() + "\n\n" + _shadow_json).lstrip()
        except Exception:
            pass
    text = _strip_heatmap_from_text(text_raw_for_tcmm) if text_raw_for_tcmm else ""
    # [SSO_USAGE_INIT_2026_05_20] usage dict from TCMM /generate
    # (adapter.last_usage). Used by both audit_record and the
    # stream/non-stream return paths.
    _sso_usage = j.get("usage") or {}
    # Iter 19: rehydrate PII tokens (REF_PERSON_N -> real names) before
    # sending back to LibreChat. Uses the same pii_session that was
    # used for redaction above so the mapping is consistent.
    try:
        text = _redactor.rehydrate_text(text, _pii_sid)
    except Exception as _re:
        logger.warning("[SSO] rehydrate failed (sending tokens through): %s", _re)
    logger.info(
        "  [SSO] model=%s sys_chars=%s fp=%s ms=%.0f -> %d chars text",
        real_model, j.get("sys_prompt_chars"), (j.get("sys_prompt_fp") or "")[:8],
        j.get("duration_ms") or 0, len(text),
    )

    # [TO_LLM_FULL_PROMPT_AUDIT_2026_05_20] Write the TO_LLM row NOW
    # with the FULL prompt the LLM saw. j["sys_prompt_text"] is the
    # rendered system prompt (preamble + TCMM memory blocks +
    # immutable/stable/working tiers, all redacted) — without this the
    # audit dashboard showed only the user message and looked like
    # memory wasn't being injected, which was misleading.
    try:
        _sys_text = j.get("sys_prompt_text") or ""
        _tools_sec = _tools_audit_section(_sso_tools)
        _to_llm_full = (
            (f"[SYSTEM]\n{_sys_text}\n\n" if _sys_text else "")
            + (f"{_tools_sec}\n\n" if _tools_sec else "")
            + f"[USER]\n{_redacted_msg}"
        )
        _sso_audit_record(
            direction="TO_LLM",
            conv_id=conversation_id,
            user_id=user_id,
            model=real_model,
            stream=is_stream,
            content=_to_llm_full,
            tokens_input=(
                (_sso_usage.get("input_tokens") or 0)
                + (_sso_usage.get("cache_creation_input_tokens") or 0)
                + (_sso_usage.get("cache_read_input_tokens") or 0)
            ) or None,
            cache_create=_sso_usage.get("cache_creation_input_tokens"),
            cache_read=_sso_usage.get("cache_read_input_tokens"),
        )
    except Exception as _e:
        logger.debug("[SSO] TO_LLM (deferred) audit failed: %s", _e)

    # Iter 20: write the FROM_LLM audit row with the REHYDRATED text
    # (what the user actually sees — matches what audit dashboard shows).
    # [SSO_FROM_LLM_USAGE_2026_05_20] Forward the same _sso_usage already
    # captured at line ~2943 from TCMM /generate's adapter.last_usage —
    # the same values Anthropic returned to TCMM's internal call. Without
    # this every SSO FROM_LLM row landed with cache_create/cache_read NULL
    # despite the API returning real numbers. Mirrors the TO_LLM record
    # site above so the dashboard's input/output/cache columns are
    # populated for SSO traffic just like for gateway-path traffic.
    try:
        _sso_audit_record(
            direction="FROM_LLM",
            conv_id=conversation_id,
            user_id=user_id,
            model=real_model,
            stream=is_stream,
            content=text,
            tokens_input=(
                (_sso_usage.get("input_tokens") or 0)
                + (_sso_usage.get("cache_creation_input_tokens") or 0)
                + (_sso_usage.get("cache_read_input_tokens") or 0)
            ) or None,
            tokens_output=_sso_usage.get("output_tokens"),
            cache_create=_sso_usage.get("cache_creation_input_tokens"),
            cache_read=_sso_usage.get("cache_read_input_tokens"),
        )
    except Exception as _e:
        logger.debug("[SSO] FROM_LLM audit failed: %s", _e)

    # Iter 18 (2026-05-19): ingest assistant turn so subsequent SSO calls
    # for this conv have memory. Best-effort; non-blocking.
    try:
        import asyncio as _ad_asyncio
        # [SHADOW_TOOL_FLAG_OBJ_2026_05_22] Pass the captured shadow-tool
        # input through to post_response so block_class (from emit_class)
        # actually lands on the new archive block. Without this, every
        # SSO turn shipped raw text only and block_class stayed NULL
        # waiting on the AIStudio NLP fallback (which has been 429-
        # storming and is effectively dead).
        _post_flag_obj = _flag_obj_from_tool if isinstance(_flag_obj_from_tool, dict) else None
        _ad_asyncio.create_task(_sso_post_response(
            conversation_id, user_id, text_raw_for_tcmm, real_model,
            flag_obj=_post_flag_obj,
        ))
    except Exception as _e:
        logger.debug("[SSO] post_response schedule failed: %s", _e)

    if is_stream:
        # [STREAM_TOOL_USE_2026_05_20] forward full content blocks
        # (text + tool_use) and stop_reason so the SSE stream carries
        # tool calls properly to LibreChat for dispatch.
        # [SHADOW_TOOL_TCMM_RECORD_2026_05_20] cleaned blocks already
        # have the tcmm_record_turn tool_use removed; stop_reason may
        # have been downgraded from "tool_use" → "end_turn" if no
        # other tool_use blocks remain.
        _sso_content_blocks = _gen_cleaned_content
        # [STRIP_HEATMAP_SSO_2026_05_20] strip heatmap JSON from each
        # text-type block before streaming to the user. tool_use blocks
        # are left untouched.
        if _sso_content_blocks:
            _sso_content_blocks = [
                ({**b, "text": _strip_heatmap_from_text(b.get("text", ""))} if isinstance(b, dict) and b.get("type") == "text" else b)
                for b in _sso_content_blocks
            ]
        _sso_stop_reason    = _gen_stop_reason  # [SHADOW_TOOL_TCMM_RECORD_2026_05_20]
        return StreamingResponse(
            _sso_anthropic_stream_chunks(
                text, real_model, usage=_sso_usage,
                content=_sso_content_blocks,
                stop_reason=_sso_stop_reason,
            ),
            media_type="text/event-stream",
        )
    # [SHADOW_TOOL_TCMM_RECORD_2026_05_20] forward CLEANED content blocks
    # (shadow tcmm_record_turn already stripped above).
    _sso_content_blocks = _gen_cleaned_content
    # [STRIP_HEATMAP_SSO_2026_05_20] strip heatmap from non-stream text blocks
    if _sso_content_blocks:
        _sso_content_blocks = [
            ({**b, "text": _strip_heatmap_from_text(b.get("text", ""))} if isinstance(b, dict) and b.get("type") == "text" else b)
            for b in _sso_content_blocks
        ]
    _sso_stop_reason    = _gen_stop_reason  # [SHADOW_TOOL_TCMM_RECORD_2026_05_20]
    return JSONResponse(content=_sso_anthropic_nonstream(
        text, real_model, usage=_sso_usage,
        content=_sso_content_blocks, stop_reason=_sso_stop_reason,
    ))


def _fold_agent_runtime_events(body: dict, *, default_model: str = "") -> dict:
    """Fold agent-runtime's `{"events": [...]}` envelope into the Anthropic
    Messages API non-stream response shape LibreChat expects.

    Inputs from agent-runtime (subset of typed events from `agent.events`):
      run_start        — agent_id, model, backend, started_at
      audit            — direction (TO_LLM | FROM_LLM | APPROVAL), content
      assistant        — full message dict with content + usage + stop_reason
      assistant_text   — text fragment (one per text block in the response)
      tool_call        — name, id, input (mirrors content_block tool_use)
      final_result     — terminal text (fallback when assistant not present)
      usage            — token totals + cache hits
      run_end          — stop_reason
      error            — code, message

    Output shape:
      {id, type:"message", role:"assistant", model, content:[...],
       stop_reason, stop_sequence:null, usage:{...}}

    We prefer the structured `assistant` event when present (it carries
    the full content_blocks ready for the API).  Falls back to building
    content from assistant_text + tool_call events when not present.
    """
    import uuid as _uuid

    events = body.get("events") or []
    if not isinstance(events, list):
        # Unexpected shape — pass through so LibreChat shows the raw error.
        return body

    # Look for the structured `assistant` event first.
    asst_msg = None
    usage_dict: dict = {}
    stop_reason = "end_turn"
    model = default_model
    text_parts: list[str] = []
    tool_use_blocks: list[dict] = []
    error_payload: _Optional[dict] = None

    for ev in events:
        if not isinstance(ev, dict):
            continue
        et = ev.get("type")
        if et == "run_start":
            model = ev.get("model") or model
        elif et == "assistant":
            inner = ev.get("message") or {}
            if isinstance(inner.get("content"), list):
                asst_msg = inner
        elif et == "assistant_text":
            t = ev.get("text") or ""
            if t:
                text_parts.append(t)
        elif et == "tool_call":
            tool_use_blocks.append({
                "type":  "tool_use",
                "id":    ev.get("id", ""),
                "name":  ev.get("name", ""),
                "input": ev.get("input", {}) or {},
            })
        elif et == "final_result":
            # final_result text — use as a fallback if no assistant blocks.
            if not text_parts and not asst_msg:
                fr = ev.get("result") or ""
                if fr:
                    text_parts.append(fr)
        elif et == "usage":
            # agent.events.usage carries normalised fields; map to Anthropic.
            usage_dict = {
                "input_tokens":               int(ev.get("tokens_input_new") or 0),
                "output_tokens":              int(ev.get("tokens_output") or 0),
                "cache_creation_input_tokens": int(ev.get("cache_create") or 0),
                "cache_read_input_tokens":     int(ev.get("cache_read") or 0),
            }
        elif et == "run_end":
            sr = ev.get("stop_reason")
            if sr:
                stop_reason = sr
        elif et == "error":
            error_payload = {
                "type":    ev.get("code", "agent_runtime_error"),
                "message": ev.get("message", "") or "",
            }

    if error_payload:
        return {"type": "error", "error": error_payload}

    # Build content array — prefer the assistant event's blocks; otherwise
    # synthesise from text_parts + tool_use_blocks.
    if asst_msg and isinstance(asst_msg.get("content"), list):
        content = asst_msg["content"]
        if asst_msg.get("usage") and not usage_dict:
            usage_dict = asst_msg["usage"]
        if asst_msg.get("stop_reason"):
            stop_reason = asst_msg["stop_reason"]
    else:
        content = []
        if text_parts:
            content.append({"type": "text", "text": "".join(text_parts)})
        content.extend(tool_use_blocks)

    # Ensure usage has all four expected keys (LibreChat / dashboard rely on
    # them being present even when zero).
    usage_dict = {
        "input_tokens":                int(usage_dict.get("input_tokens") or 0),
        "output_tokens":               int(usage_dict.get("output_tokens") or 0),
        "cache_creation_input_tokens": int(usage_dict.get("cache_creation_input_tokens") or 0),
        "cache_read_input_tokens":     int(usage_dict.get("cache_read_input_tokens") or 0),
    }

    return {
        "id":            f"msg_{_uuid.uuid4().hex}",
        "type":          "message",
        "role":          "assistant",
        "model":         model,
        "content":       content,
        "stop_reason":   stop_reason or "end_turn",
        "stop_sequence": None,
        "usage":         usage_dict,
    }


async def _handle_agent_runtime_request(
    *,
    data: dict,
    conversation_id: str,
    user_id: str,
    tenant_id: str,
    agent_id: str,
    stream: bool,
):
    """Forward an Anthropic-bound request to agent-runtime.

    agent-runtime owns the SDK loop + multi-agent state; we just route.
    Audit is still written by THIS proxy on the request side (so the
    pii_audit ``direction=TO_LLM`` row exists with redacted content);
    agent-runtime writes the ``direction=FROM_LLM`` row capturing the
    actual SDK call's usage.

    Streaming: agent-runtime emits SSE events.  We pass them through
    verbatim — LibreChat understands the same shape because we model
    our SSE event names after the Anthropic message stream API.
    """
    payload = {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "agent_id": agent_id,
        "messages": data.get("messages", []),
        "stream": stream,
        # [CLIENT_TOOLS_MERGE_2026_05_31] Forward LibreChat's client/MCP tool
        # schemas so the agent-runtime persona can MERGE them with its own
        # tools (web/shell/fs/client dispatch via tool_dispatcher Path 2).
        "tools": data.get("tools") or [],
    }

    # [WORKSPACE_BLOCK_2026_06_01] Fetch the connected client-daemon's
    # workspace state (folders + OS) and forward the rendered block on a
    # dedicated field. The unified harness (agent-runtime base.py) rebuilds
    # the prompt and DISCARDS body["system"], so the legacy
    # _inject_workspace_state(body, ...) was silently dropped. agent-runtime
    # reads `workspace_block` and appends it as an uncached system tail
    # block so the model sees the user's folders again.
    try:
        _ws_state_fwd = await _fetch_workspace_state(user_id) if user_id else None
        _ws_block_fwd = _render_workspace_block(_ws_state_fwd) if _ws_state_fwd else ""
        if _ws_block_fwd:
            payload["workspace_block"] = _ws_block_fwd
            logger.info(
                "  [AR-WORKSPACE] forwarding workspace block "
                "(%d folders)", len((_ws_state_fwd or {}).get("folders") or []),
            )
    except Exception as _wse:
        logger.debug(f"[forward] workspace fetch failed: {_wse}")

    # [TOOLS_TRACE_2026-06-01] Visibility into the client/MCP tool set
    # LibreChat actually forwards for this turn — so we can see WHAT the
    # Director ends up able to call (the set is dynamic: depends on which
    # MCP servers the user enabled in the UI).
    try:
        _fwd_tools = payload.get("tools") or []
        _fwd_names = [
            (t.get("name") if isinstance(t, dict) else None) for t in _fwd_tools
        ]
        logger.info(
            "  [AR-TOOLS] agent=%s fwd_count=%d names=%s",
            agent_id, len(_fwd_tools), [n for n in _fwd_names if n][:40],
        )
    except Exception:
        pass

    timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
    client = httpx.AsyncClient(timeout=timeout)

    try:
        if stream:
            import json as _json, time as _t

            # [AGENT_RUNTIME_TRUE_STREAM_2026_05_28] Real-time forwarder.
            # Calls agent-runtime WITH stream=True, parses each agent-
            # runtime SSE event as it arrives, and immediately emits
            # the equivalent Anthropic SSE event(s) to LibreChat.
            # First user-visible token now lands in ~1-2s instead of
            # waiting for the full LLM call to complete.
            #
            # Event mapping (agent-runtime → Anthropic):
            #   run_start     → emit message_start envelope
            #   text_delta    → emit content_block_start (idx 0) on first;
            #                   then content_block_delta with text_delta
            #   assistant_text→ if no text_delta seen yet, treat the
            #                   whole text as one delta + start/stop the
            #                   text content_block
            #   tool_call     → emit content_block_start (tool_use) +
            #                   input_json_delta + content_block_stop
            #   usage         → buffer for message_delta payload
            #   final_result  → close any open block, emit message_delta
            #   run_end       → emit message_stop, terminate
            #   error         → emit Anthropic-shape error event
            payload_stream = {**payload, "stream": True}
            model_for_msg = data.get("model", "claude-sonnet-4-6")

            async def _sse_passthrough():
                msg_id = f"msg_rt_{int(_t.time()*1000)}"
                state = {
                    "message_started": False,
                    "text_block_open": False,
                    "text_block_idx": 0,
                    "next_block_idx": 0,
                    "usage": {},
                    "stop_reason": None,
                    "saw_text_delta": False,
                }

                def _emit_message_start():
                    if state["message_started"]:
                        return None
                    state["message_started"] = True
                    base = {
                        "id":           msg_id,
                        "type":         "message",
                        "role":         "assistant",
                        "model":        model_for_msg,
                        "content":      [],
                        "stop_reason":  None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": 0, "output_tokens": 0,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens":     0,
                        },
                    }
                    return (
                        "event: message_start\n"
                        f"data: {_json.dumps({'type':'message_start','message':base})}\n\n"
                    ).encode("utf-8")

                def _open_text_block():
                    if state["text_block_open"]:
                        return None
                    idx = state["next_block_idx"]
                    state["text_block_idx"] = idx
                    state["next_block_idx"] += 1
                    state["text_block_open"] = True
                    return (
                        "event: content_block_start\n"
                        f"data: {_json.dumps({'type':'content_block_start','index':idx,'content_block':{'type':'text','text':''}})}\n\n"
                    ).encode("utf-8")

                def _close_text_block():
                    if not state["text_block_open"]:
                        return None
                    idx = state["text_block_idx"]
                    state["text_block_open"] = False
                    return (
                        "event: content_block_stop\n"
                        f"data: {_json.dumps({'type':'content_block_stop','index':idx})}\n\n"
                    ).encode("utf-8")

                req2 = client.build_request(
                    "POST", f"{AGENT_RUNTIME_URL}/agent/query",
                    json=payload_stream,
                )
                response2 = None
                try:
                    response2 = await client.send(req2, stream=True)
                    if response2.status_code >= 400:
                        body_err = (await response2.aread())[:500].decode("utf-8", "ignore")
                        s = _emit_message_start()
                        if s:
                            yield s
                        yield (
                            f"event: error\n"
                            f"data: {_json.dumps({'type':'error','error':{'type':'agent_runtime_error','message':body_err}})}\n\n"
                        ).encode("utf-8")
                        return

                    buffer = b""
                    async for chunk in response2.aiter_bytes():
                        buffer += chunk
                        while b"\n\n" in buffer:
                            block, buffer = buffer.split(b"\n\n", 1)
                            # parse `data: <json>` line from SSE block
                            for line in block.split(b"\n"):
                                if not line.startswith(b"data: "):
                                    continue
                                try:
                                    ev = _json.loads(line[6:].decode("utf-8"))
                                except Exception:
                                    continue
                                if not isinstance(ev, dict):
                                    continue
                                et = ev.get("type") or ""
                                # Lazy emit message_start on first
                                # meaningful event.
                                if not state["message_started"] and et in (
                                    "run_start", "text_delta", "assistant_text",
                                    "tool_call", "final_result"
                                ):
                                    s = _emit_message_start()
                                    if s:
                                        yield s

                                if et == "text_delta":
                                    state["saw_text_delta"] = True
                                    s = _open_text_block()
                                    if s:
                                        yield s
                                    idx = state["text_block_idx"]
                                    yield (
                                        "event: content_block_delta\n"
                                        f"data: {_json.dumps({'type':'content_block_delta','index':idx,'delta':{'type':'text_delta','text':ev.get('text','') or ''}})}\n\n"
                                    ).encode("utf-8")

                                elif et == "assistant_text" and not state["saw_text_delta"]:
                                    # No streaming deltas — emit the
                                    # whole text as a single delta in
                                    # a text block.
                                    s = _open_text_block()
                                    if s:
                                        yield s
                                    idx = state["text_block_idx"]
                                    yield (
                                        "event: content_block_delta\n"
                                        f"data: {_json.dumps({'type':'content_block_delta','index':idx,'delta':{'type':'text_delta','text':ev.get('text','') or ''}})}\n\n"
                                    ).encode("utf-8")
                                    s = _close_text_block()
                                    if s:
                                        yield s

                                elif et == "tool_call":
                                    # Close any open text block first.
                                    s = _close_text_block()
                                    if s:
                                        yield s
                                    idx = state["next_block_idx"]
                                    state["next_block_idx"] += 1
                                    tool_name = ev.get("name") or ev.get("tool") or "unknown"
                                    tool_id   = ev.get("id") or f"toolu_rt_{idx}"
                                    tool_inp  = ev.get("input") or ev.get("args") or {}
                                    yield (
                                        "event: content_block_start\n"
                                        f"data: {_json.dumps({'type':'content_block_start','index':idx,'content_block':{'type':'tool_use','id':tool_id,'name':tool_name,'input':{}}})}\n\n"
                                    ).encode("utf-8")
                                    yield (
                                        "event: content_block_delta\n"
                                        f"data: {_json.dumps({'type':'content_block_delta','index':idx,'delta':{'type':'input_json_delta','partial_json':_json.dumps(tool_inp)}})}\n\n"
                                    ).encode("utf-8")
                                    yield (
                                        "event: content_block_stop\n"
                                        f"data: {_json.dumps({'type':'content_block_stop','index':idx})}\n\n"
                                    ).encode("utf-8")

                                elif et == "usage":
                                    state["usage"] = {
                                        "input_tokens":  int(ev.get("tokens_input_new") or 0),
                                        "output_tokens": int(ev.get("tokens_output") or 0),
                                        "cache_creation_input_tokens": int(ev.get("cache_create") or 0),
                                        "cache_read_input_tokens":     int(ev.get("cache_read") or 0),
                                    }

                                elif et == "final_result":
                                    state["stop_reason"] = ev.get("stop_reason") or "end_turn"

                                elif et == "run_end":
                                    s = _close_text_block()
                                    if s:
                                        yield s
                                    yield (
                                        "event: message_delta\n"
                                        f"data: {_json.dumps({'type':'message_delta','delta':{'stop_reason':state['stop_reason'] or 'end_turn','stop_sequence':None},'usage':{'output_tokens': state['usage'].get('output_tokens',0)}})}\n\n"
                                    ).encode("utf-8")
                                    yield (
                                        "event: message_stop\n"
                                        f"data: {_json.dumps({'type':'message_stop'})}\n\n"
                                    ).encode("utf-8")
                                    return

                                elif et == "error":
                                    s = _close_text_block()
                                    if s:
                                        yield s
                                    yield (
                                        f"event: error\n"
                                        f"data: {_json.dumps({'type':'error','error':{'type':ev.get('code','agent_runtime_error'),'message':ev.get('message') or str(ev)}})}\n\n"
                                    ).encode("utf-8")
                                    return

                    # Stream ended without run_end — emit a clean stop.
                    s = _close_text_block()
                    if s:
                        yield s
                    yield (
                        "event: message_delta\n"
                        f"data: {_json.dumps({'type':'message_delta','delta':{'stop_reason':'end_turn','stop_sequence':None},'usage':{'output_tokens':state['usage'].get('output_tokens',0)}})}\n\n"
                    ).encode("utf-8")
                    yield (
                        "event: message_stop\n"
                        f"data: {_json.dumps({'type':'message_stop'})}\n\n"
                    ).encode("utf-8")

                except Exception as _e:
                    s = _emit_message_start()
                    if s:
                        yield s
                    yield (
                        f"event: error\n"
                        f"data: {_json.dumps({'type':'error','error':{'type':'stream_forward_error','message':f'{type(_e).__name__}: {_e}'}})}\n\n"
                    ).encode("utf-8")
                finally:
                    try:
                        if response2 is not None:
                            await response2.aclose()
                    except Exception:
                        pass
                    try:
                        await client.aclose()
                    except Exception:
                        pass

            return StreamingResponse(
                _sse_passthrough(),
                media_type="text/event-stream",
                status_code=200,
            )

        # Non-streaming: collect once, fold event stream → Anthropic shape.
        #
        # agent-runtime returns an `{"events": [...]}` envelope containing
        # SDK-style events (run_start, audit, assistant_text, assistant,
        # final_result, usage, tool_call, run_end, error).  LibreChat
        # expects the Anthropic Messages API non-stream shape:
        #
        #   {id, type:"message", role:"assistant", model, content:[...],
        #    stop_reason, stop_sequence, usage:{...}}
        #
        # Translate here so the agent-runtime path is a drop-in for
        # _handle_sso_request / chat_agent_handler.  Without this fold,
        # the response would be a literal events array and LibreChat
        # would render an empty bubble + warn about malformed JSON.
        try:
            r = await client.post(
                f"{AGENT_RUNTIME_URL}/agent/query",
                json=payload,
            )
            if r.status_code >= 400:
                return JSONResponse(r.json(), status_code=r.status_code)
            body = r.json()
            return JSONResponse(
                _fold_agent_runtime_events(body, default_model=data.get("model", "")),
                status_code=200,
            )
        finally:
            await client.aclose()

    except httpx.ConnectError as e:
        logger.error(
            f"  [AGENT-RUNTIME] connect failed: {e}; "
            "falling back to direct Anthropic forward"
        )
        await client.aclose()
        # Fall back to direct Anthropic (don't break the user-facing
        # chat just because agent-runtime is down).  Caller will retry
        # the gateway path WITHOUT our early-route on next request if
        # they want; for THIS request we degrade gracefully.
        return JSONResponse(
            {
                "error": {
                    "type": "agent_runtime_unavailable",
                    "message": f"agent-runtime at {AGENT_RUNTIME_URL} unreachable",
                },
            },
            status_code=503,
        )
    except Exception as e:
        logger.exception(f"  [AGENT-RUNTIME] forward error: {e}")
        try:
            await client.aclose()
        except Exception:
            pass
        return JSONResponse(
            {
                "error": {
                    "type": "agent_runtime_error",
                    "message": str(e),
                },
            },
            status_code=502,
        )


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
                # Translate our synthetic ``claude-opus-4-7-1m`` model
                # alias into the real Anthropic model + 1M beta header
                # before any downstream logic reads ``data["model"]``.
                # Safe no-op for every other model ID.
                _rewrite_claude_1m_alias(data, headers)
                # 2026-05-19: map bare aliases (e.g. claude-haiku-4-5)
                # to their dated form for workspaces that only expose
                # the dated id. Safe no-op when the alias works as-is.
                _rewrite_claude_dated_alias(data)
                # [MAX_TOKENS_FLOOR_2026-06-05] LibreChat defaults Opus to
                # max_tokens=8192, which silently truncates long answers
                # mid-word (the model stops at EXACTLY 8192 — observed reciting
                # a policy doc, cut at "...Communicable Dis"). Opus 4.7 supports
                # far more output, so raise the FLOOR for opus models. This is a
                # cap, not a target: short answers still stop at end_turn, so it
                # only affects responses that would otherwise be cut. Env-tunable
                # via VEILGUARD_OPUS_MAX_TOKENS (default 16384).
                try:
                    if "opus" in str(data.get("model") or "").lower():
                        _mt_floor = int(os.environ.get("VEILGUARD_OPUS_MAX_TOKENS", "16384"))
                        if int(data.get("max_tokens") or 0) < _mt_floor:
                            data["max_tokens"] = _mt_floor
                except Exception:
                    pass
                is_stream = data.get("stream", False)
                # Iter 11: SSO early-route ─ if model ends in '-sso',
                # bypass the api.anthropic.com forward and call TCMM
                # /generate (Claude CLI via the user's Max subscription).
                #
                # 2026-05-23 [PR #5a]: feature-flag VEILGUARD_USE_CHAT_AGENT=1
                # routes through the new ChatAgent-based handler instead.
                # Off by default — flip env to enable; flip back to roll back.
                if _is_sso_model(data.get("model")):
                    # Try to extract conv_id + user_id from headers
                    # (LibreChat fork injects x-conversation-id + x-user-id).
                    _sso_conv = headers.get("x-conversation-id") or ""
                    _sso_user = headers.get("x-user-id") or ""
                    _sso_tenant = headers.get("x-tenant-id") or _sso_user

                    try:
                        from . import chat_agent_handler as _chat_agent
                        _use_chat_agent = _chat_agent.is_enabled()
                    except Exception as _e:
                        logger.debug(
                            f"[SSO] chat_agent handler import failed "
                            f"(falling back to legacy): {_e}"
                        )
                        _use_chat_agent = False

                    if _use_chat_agent:
                        logger.info(
                            "  >>> [CHAT-AGENT] %s stream=%s conv=%s",
                            data.get("model"), is_stream, _sso_conv[:14],
                        )
                        return await _chat_agent.handle_chat_request(
                            data,
                            conversation_id=_sso_conv,
                            user_id=_sso_user,
                            tenant_id=_sso_tenant,
                            is_stream=is_stream,
                        )

                    # 2026-05-25: agent-runtime route inside the SSO
                    # branch.  Earlier, this exit-out fell through to the
                    # legacy SSO handler — meaning AGENT_RUNTIME_ENABLED
                    # was a no-op for any claude-* model under
                    # CLAUDE_SSO_DEFAULT=1 (i.e. every chat).  When the
                    # admin flips AGENT_RUNTIME_ENABLED=true they want the
                    # Director / IC routing on, regardless of OAuth/SSO
                    # vs API-key auth (agent-runtime now uses the same
                    # TCMM /generate OAuth-bearer path under the hood).
                    if AGENT_RUNTIME_ENABLED:
                        _ar_user = _sso_user
                        if (
                            not AGENT_RUNTIME_USER_ALLOWLIST
                            or _ar_user in AGENT_RUNTIME_USER_ALLOWLIST
                        ):
                            _ar_agent = (
                                headers.get("x-veilguard-agent-id")
                                or AGENT_RUNTIME_DEFAULT_AGENT
                            )
                            logger.info(
                                "  >>> [AGENT-RUNTIME] %s agent=%s stream=%s "
                                "conv=%s",
                                data.get("model"), _ar_agent, is_stream,
                                _sso_conv[:14],
                            )
                            return await _handle_agent_runtime_request(
                                data=data,
                                conversation_id=_sso_conv,
                                user_id=_sso_user,
                                tenant_id=_sso_tenant,
                                agent_id=_ar_agent,
                                stream=is_stream,
                            )

                    logger.info(
                        "  >>> [SSO] %s stream=%s conv=%s",
                        data.get("model"), is_stream, _sso_conv[:14],
                    )
                    return await _handle_sso_request(
                        data, _sso_conv, _sso_user, is_stream,
                    )

                # 2026-05-22: agent-runtime early-route.  When enabled
                # AND the user is in the allowlist (or allowlist empty),
                # Anthropic-bound requests go to the new agent-runtime
                # service instead of api.anthropic.com.  Non-Anthropic
                # backends (OpenAI/xAI/Gemini) bypass; they keep their
                # existing direct-forward path.  See spec §0a.
                #
                # 2026-05-25: the original `not _is_sso_model(...)` exclusion
                # was a holdover from when agent-runtime used the
                # ANTHROPIC_API_KEY path (and SSO went through TCMM /generate
                # OAuth bearer instead).  Now that agent-runtime's `live`
                # backend ALSO drives TCMM /generate via the unified
                # Agent.run_turn pipeline (using the same long-lived
                # setup-token OAuth bearer ChatAgent uses), the exclusion
                # would skip every Claude model when CLAUDE_SSO_DEFAULT=1.
                # Dropped — agent-runtime forwards regardless of SSO state.
                if (
                    AGENT_RUNTIME_ENABLED
                    and backend_name == "anthropic"
                ):
                    _ar_conv = headers.get("x-conversation-id") or ""
                    _ar_user = headers.get("x-user-id") or ""
                    if (
                        not AGENT_RUNTIME_USER_ALLOWLIST
                        or _ar_user in AGENT_RUNTIME_USER_ALLOWLIST
                    ):
                        _ar_agent = (
                            headers.get("x-veilguard-agent-id")
                            or AGENT_RUNTIME_DEFAULT_AGENT
                        )
                        _ar_tenant = (
                            headers.get("x-tenant-id")
                            or _ar_user  # single-tenant per-user fallback
                        )
                        logger.info(
                            "  >>> [AGENT-RUNTIME] %s agent=%s stream=%s "
                            "conv=%s user=%s",
                            data.get("model"), _ar_agent, is_stream,
                            _ar_conv[:14], _ar_user[:14],
                        )
                        return await _handle_agent_runtime_request(
                            data=data,
                            conversation_id=_ar_conv,
                            user_id=_ar_user,
                            tenant_id=_ar_tenant,
                            agent_id=_ar_agent,
                            stream=is_stream,
                        )
                # xAI/OpenAI only emit the final usage chunk when the
                # request opts in via ``stream_options.include_usage``.
                # Without it the proxy can't record tokens_input /
                # tokens_output / cache_read on FROM_LLM rows — they all
                # land as NULL and the dashboard renders dashes.
                # Anthropic streaming always emits usage, so this is a
                # no-op there (skipped on backend name).
                if is_stream and backend_name in ("xai", "openai"):
                    _so = data.get("stream_options")
                    if not isinstance(_so, dict):
                        _so = {}
                    if not _so.get("include_usage"):
                        _so["include_usage"] = True
                        data["stream_options"] = _so
                        # body is rebuilt from ``redacted`` (which copies
                        # all top-level keys, including this one) at
                        # ~line 2630, so no body re-dump needed here.
                conversation_id = extract_conversation_id(data, headers)
                pii_session_id = extract_pii_session_id(data)
                # Multi-source user_id extraction — see extract_user_id
                # docstring. Previously we only checked metadata.user_id
                # which left 712+ pii_audit rows stamped with empty user
                # because LibreChat omits the field on some endpoints.
                tcmm_user_id = extract_user_id(data, headers)
                # Sub-agent spawn detection. The sub-agents MCP wraps
                # every LLM call from a spawned agent in a ``_spawn_scope``
                # that plants the parent's conv id in
                # ``metadata.lineage_parent_conv``. We use it as a SIGNAL
                # (this is a sub-agent), then DISCARD it before calling
                # TCMM.
                #
                # Rationale (2026-05-18, per user spec):
                # Sub-agents should get TCMM memory under their OWN
                # namespace — they share user_id with the parent but have
                # a fresh session_id (= their own conv_id), so TCMM's
                # default recall scope ``"session"`` already isolates
                # their reads to their own conversation. The ONE link
                # that breaks this is ``lineage_parent_conv``: when
                # forwarded, tcmm-service stamps the sub-agent's first
                # archive block with ``lineage.parents[0]`` pointing into
                # the PARENT's namespace, which makes graph-expansion
                # recall traverse back into the parent. Dropping it here
                # keeps the sub-agent's namespace genuinely fresh.
                #
                # The parent's last-message memory is still ingested into
                # the sub-agent's namespace via the user-message itself
                # (which contains the spawn prompt) — TCMM observes that
                # as the sub-agent's first archive block, no cross-
                # namespace pointer needed.
                _raw_lineage = data.get("metadata", {}).get("lineage_parent_conv", "") or ""
                _parent_conv_hint = (
                    ""
                    if _is_unsubstituted_placeholder(_raw_lineage)
                    else _raw_lineage
                )
                _is_subagent_spawn = bool(_parent_conv_hint)
                if _is_subagent_spawn:
                    logger.info(
                        f"  [SUB-AGENT] spawn detected (parent={_parent_conv_hint[:12]}...). "
                        f"Sub-agent conv={conversation_id[:8]}... gets a FRESH TCMM "
                        f"namespace + session; parent memory + lineage NOT inherited."
                    )
                # Drop the cross-namespace pointer before any TCMM call.
                tcmm_lineage_parent = ""

                # Strip TCMM-only fields from metadata before forwarding to LLM API
                # (Anthropic only allows user_id in metadata — extra fields cause 400)
                metadata = data.get("metadata")
                if isinstance(metadata, dict):
                    metadata.pop("conversation_id", None)
                    metadata.pop("lineage_parent_conv", None)

                logger.info(
                    f">>> {request.method} [{backend_name}] /{remaining_path} "
                    f"(stream={is_stream}, conv={conversation_id[:8]}...)"
                )

                # 2026-05-14: temporary diagnostic — when Grok is
                # asked to "create a file" but doesn't emit write_file,
                # we want to know whether write_file was even in the
                # tools array LibreChat sent. Logs the tool count and
                # names so we can confirm/rule out a missing-tool bug
                # vs a pure model-hallucination bug. Cheap enough to
                # keep in prod for a few days.
                _tools = data.get("tools") if isinstance(data, dict) else None
                if isinstance(_tools, list):
                    _names = []
                    for _t in _tools:
                        if isinstance(_t, dict):
                            _fn = _t.get("function", _t)
                            if isinstance(_fn, dict):
                                _n = _fn.get("name") or _t.get("name")
                                if _n:
                                    _names.append(_n)
                    # Show raw names so we can see LibreChat's naming
                    # convention without assuming a split delimiter.
                    _has_write_file = any("write_file" in n for n in _names)
                    _has_run_cmd    = any("run_command" in n for n in _names)
                    _has_web_search = any("web_search" in n for n in _names)
                    _file_ish = sorted([n for n in _names
                                        if "file" in n.lower() or "write" in n.lower()])
                    logger.info(
                        f"  [TOOLS] backend={backend_name} count={len(_names)} "
                        f"has_write_file={_has_write_file} has_run_command={_has_run_cmd} "
                        f"has_web_search={_has_web_search}"
                    )
                    logger.info(
                        f"  [TOOLS] file/write tools in list ({len(_file_ish)}): {_file_ish}"
                    )
                else:
                    logger.info(f"  [TOOLS] backend={backend_name} (no tools array)")

                # ── TCMM Integration ──
                tcmm_active = False
                if TCMM_ENABLED and _is_chat_completion(remaining_path, request.method):
                    messages = data.get("messages", [])

                    # 2026-05-19: LibreChat side-channel bypass.
                    # Title-gen / summary calls have a synthetic prompt
                    # text but no real conversation_id reuse — each one
                    # would otherwise hash to its OWN conv row in
                    # pii_audit (e.g. aid=3074 conv-69df7853-9ee09e7441
                    # for a 5-word title call) AND eat the full 20-70KB
                    # Veilguard+TCMM injection. Skip the entire pipeline
                    # for these — they reach the upstream LLM as bare
                    # prompts (LibreChat's intent), the audit row still
                    # records them, but no TCMM session is touched.
                    _side_channel = _detect_librechat_side_channel(messages)
                    if _side_channel:
                        logger.info(
                            f"  [TCMM] side-channel call detected "
                            f"({_side_channel!r}) — bypass: no pre_request, "
                            f"no pin, no render, no ingest"
                        )
                        # Fall through to upstream forward with TCMM
                        # untouched. tcmm_active stays False so the
                        # FROM_LLM handler also skips post_response.

                    # Tool-followup detection reads the envelope via
                    # classify_message_origin — no LibreChat-side declaration
                    # needed. See classify_message_origin for the schema map.
                    is_tool_followup = (not _side_channel) and _is_tool_followup(messages)

                    if is_tool_followup:
                        # Tool round-trip turn. The user hasn't authored
                        # anything new, so we skip the recall *ingest*
                        # leg (no fresh user message to observe). BUT
                        # we MUST still render the static preamble and
                        # apply it to the request — otherwise xAI sees
                        # a totally different prefix on the
                        # tool-followup call vs the original user turn,
                        # cache misses every continuation, AND the LLM
                        # loses memory context mid-flow.
                        #
                        # 2026-05-19 fix: the prior "skip prompt rebuild"
                        # heuristic was inverted — it tried to preserve
                        # the cache by NOT touching the prompt, but the
                        # cache key IS the prompt. Stripping the
                        # TCMM-rendered msg[0] guarantees a miss. Keep
                        # the rendered prefix byte-stable across the
                        # whole turn (initial call + every tool
                        # followup) so xAI's cache_read hits ~99% on
                        # all but the very first cold call.
                        tool_items = _extract_tool_pair(messages)
                        if tool_items:
                            await _tcmm_ingest_turn(
                                tool_items,
                                conversation_id,
                                user_id=tcmm_user_id,
                                lineage_parent_conv=tcmm_lineage_parent,
                            )
                        else:
                            logger.info("  [TCMM] tool-followup with no extractable tool blocks — passthrough")

                        # Render + apply (read-only path — no ingest).
                        # Use the LAST user message as the task query
                        # for recall scoring; it's what the tool was
                        # invoked in service of, so recall should rank
                        # blocks the same way as the original turn.
                        # Best-effort: TCMM failure here doesn't fail
                        # the request — degrades to no-preamble (same
                        # behaviour as before the fix).
                        try:
                            _query_for_render = _extract_last_user_message(messages) or ""
                            if _is_anthropic_format(remaining_path):
                                _render_model_tf = "anthropic"
                            elif "xai" in (remaining_path or "").lower():
                                _render_model_tf = "grok"
                            else:
                                _render_model_tf = "openai"
                            _render_result_tf = await _tcmm_render(
                                _render_model_tf, _query_for_render,
                                conv_id=conversation_id,
                                user_id=tcmm_user_id,
                            )
                            _apply_render_to_request(
                                data, headers, _render_result_tf,
                            )
                            # Inject live workspace state — same call
                            # as the main-turn path so platform / cwd
                            # are visible on continuations too.
                            try:
                                _ws_state_tf = await _fetch_workspace_state(
                                    tcmm_user_id,
                                )
                            except Exception:
                                _ws_state_tf = None
                            if _ws_state_tf:
                                _inject_workspace_state(
                                    data, _render_model_tf, _ws_state_tf,
                                )
                            # [UNIVERSAL_SHADOW_TOOL_2026_05_22] Tool-followup
                            # turns are still assistant turns the user sees,
                            # so they must be classifiable. We inject the
                            # shadow tool here too. tool_choice forcing is
                            # SAFE-SKIPPED because real tools are always
                            # present on followup (the ones being used),
                            # so _inject_shadow_tool_for_backend won't
                            # force tool_choice — model can call shadow
                            # alongside its synthesis or skip it. ~60%
                            # rate expected on followups (same as SSO
                            # baseline before we forced choice).
                            try:
                                _inject_shadow_tool_for_backend(
                                    data, _render_model_tf,
                                )
                                logger.info(
                                    f"  [SHADOW-TOOL] injected for "
                                    f"tool-followup backend={_render_model_tf}"
                                )
                            except Exception as _st_e:
                                logger.warning(
                                    f"  [SHADOW-TOOL] tool-followup "
                                    f"inject failed: {_st_e}"
                                )
                        except Exception as _tf_render_err:
                            logger.warning(
                                f"  [TCMM] tool-followup render+apply failed: "
                                f"{type(_tf_render_err).__name__}: {_tf_render_err} "
                                f"— continuing without preamble (cache miss expected)"
                            )

                        # Activate TCMM for the downstream stream-end
                        # handler so the assistant's final prose response
                        # (the synthesis / report after tool execution)
                        # gets ingested via _tcmm_post_response.
                        tcmm_active = True
                    elif _side_channel:
                        # Side-channel call: do NOTHING TCMM-related.
                        # We already logged the detection above; the
                        # audit row is still written downstream, but
                        # tcmm_active stays False so post_response is
                        # skipped too.
                        pass
                    else:
                        user_msg = _extract_last_user_message(messages)
                        if user_msg:
                            # Classify the LAST user-role message so the
                            # ingested block records whether it was plain
                            # text or carried an image attachment.
                            user_origin = "user"
                            for _m in reversed(messages):
                                if isinstance(_m, dict) and _m.get("role") == "user":
                                    _o = classify_message_origin(_m)
                                    if _o in ("user", "user_image"):
                                        user_origin = _o
                                        break
                            # [PERF-INSTR 2026-05-07] Wrap the four owned
                            # legs (TCMM HTTP pre, redact, Anthropic, rehydrate)
                            # so we can attribute the e2e budget to whose
                            # latency it actually is. Stored on `request.state`
                            # because pii-proxy is async and we need to
                            # accumulate across multiple awaits.
                            import time as _pt
                            request.state.phase_t = {}
                            _t = _pt.perf_counter()
                            try:
                                tcmm_context = await _tcmm_pre_request(
                                    user_msg,
                                    conversation_id,
                                    user_id=tcmm_user_id,
                                    origin=user_origin,
                                    lineage_parent_conv=tcmm_lineage_parent,
                                )
                            except TCMMUnavailable as _tcmm_err:
                                # 2026-05-14: fail CLOSED. Previous behaviour
                                # silently dropped memory injection and let the
                                # request proceed degraded; that masked TCMM
                                # outages from the operator. Returning 503
                                # surfaces the failure to the LibreChat client
                                # which will display an error and prompt the
                                # user to retry once the operator restores TCMM.
                                logger.error(
                                    f"  [TCMM] hard-fail — returning 503 to client "
                                    f"(no silent fallback): {_tcmm_err}"
                                )
                                from fastapi.responses import JSONResponse as _JR
                                return _JR(
                                    status_code=503,
                                    content={
                                        "error": {
                                            "type": "tcmm_unavailable",
                                            "message": "TCMM memory service is unavailable. Request rejected to prevent silent memory loss.",
                                            "detail": str(_tcmm_err),
                                        }
                                    },
                                )
                            request.state.phase_t["tcmm_pre_http"] = (_pt.perf_counter() - _t) * 1000
                            # 2026-05-15: pin preamble (idempotent) + render
                            # via TCMM. TCMM owns ALL prompt assembly — the
                            # proxy's only job is to relay the resulting
                            # blocks into the provider-shaped JSON slot.
                            # Both calls fail hard via TCMMUnavailable —
                            # the wrapping try/except above returns 503 to
                            # the client. No silent fallback by design.
                            #
                            # 2026-05-18 BUG FIX: the gate was previously
                            # ``if tcmm_context:`` which short-circuited
                            # on empty-string returns. With the empty-
                            # prompt fix to _tcmm_pre_request (which now
                            # legitimately returns "" for fresh
                            # conversations with no recall), the bypass
                            # gate caused FRESH conversations to skip
                            # pin+render entirely — no Veilguard preamble,
                            # no client system, no tool defs, no memory
                            # render injected. The proxy forwarded a bare
                            # request to the upstream LLM. Real user
                            # session (RJ Lamprecht) hit this and saw
                            # the dashboard show 188 bytes (no TCMM
                            # context at all).
                            #
                            # ``tcmm_context is not None`` is the right
                            # gate: pre_request set it to the empty
                            # string on success-but-empty, or raised
                            # TCMMUnavailable on actual failure. If we
                            # got here without an exception, run the
                            # pin + render path unconditionally.
                            if tcmm_context is not None:
                                # 1. Resolve renderer format from upstream path
                                if _is_anthropic_format(remaining_path):
                                    _render_model = "anthropic"
                                elif "xai" in (remaining_path or "").lower():
                                    _render_model = "grok"
                                else:
                                    _render_model = "openai"
                                try:
                                    # 2. Pin EVERY piece of provider context to
                                    # TCMM BEFORE rendering, so the renderer's
                                    # output is the single source of truth on
                                    # the wire:
                                    #
                                    #   a) Veilguard hardcoded preamble (idempotent)
                                    #   b) LibreChat's per-conversation system
                                    #      prompt (used to be appended at proxy
                                    #      level — now goes through TCMM so it
                                    #      gets the same caching + memory
                                    #      treatment as Veilguard's preamble.
                                    #      Previously DROPPED entirely on the
                                    #      OpenAI/Grok path — a real bug.)
                                    #   c) Tool definitions — DISABLED 2026-05-19.
                                    #      Previously we pinned the client's
                                    #      tool schemas into TCMM as an
                                    #      immutable block, which landed them
                                    #      in ``live_blocks`` with
                                    #      cache_tier="stable" and made the
                                    #      renderer emit them as ~97 SYSTEM/
                                    #      src=live blocks in the rendered
                                    #      memory body. The model ALSO
                                    #      received them via ``data["tools"]``
                                    #      natively, so every turn-1 ate
                                    #      ~50-70KB of duplicate schema text.
                                    #      Audit row aid=3064 (conv-fbcf81ba00)
                                    #      showed read_file 13× / web_search
                                    #      21× / etc. inside one TO_LLM blob.
                                    #      Fix: stop pinning. ``data["tools"]``
                                    #      is the canonical tool channel for
                                    #      Anthropic/OpenAI/Grok — TCMM does
                                    #      not need a parallel copy.
                                    #
                                    # All pin helpers fingerprint-dedup
                                    # in-process AND server-side, so repeat
                                    # turns of the same conversation skip the
                                    # round-trip entirely.
                                    # 2026-05-18 perf: fan the 3 pin calls
                                    # out in parallel via asyncio.gather().
                                    # Each pin is a tcmm-service HTTP
                                    # round-trip (10-50ms warm) — running
                                    # them sequentially serialized 30-150ms
                                    # we don't have to pay. The in-process
                                    # dedup (``_PINNED_KEYS``) makes turn-2+
                                    # pins near-free, but turn-1 of any new
                                    # conv pays all 3.
                                    _client_system = _extract_client_system(
                                        data, _render_model,
                                    )
                                    _client_tools = _extract_client_tools(data)

                                    # [PROPER_PREAMBLE_FIX_2026_05_20] Render preamble
                                    # with the actual tool schemas LibreChat sent
                                    # (data["tools"]). Avoids the old behavior of
                                    # pinning a preamble with a hardcoded prose
                                    # tool list that didn't match reality.
                                    _preamble_with_tools = _render_preamble_with_tools(_client_tools)
                                    _pin_coros = [
                                        _tcmm_pin_system_prompt(
                                            conversation_id, _preamble_with_tools,
                                            kind="veilguard_preamble",
                                            user_id=tcmm_user_id,
                                        ),
                                    ]
                                    if _client_system:
                                        _pin_coros.append(_tcmm_pin_system_prompt(
                                            conversation_id, _client_system,
                                            kind="client_system",
                                            user_id=tcmm_user_id,
                                        ))
                                    # 2026-05-19: tool-definition pin DISABLED
                                    # (empirically validated). User pasted a
                                    # full TO_LLM prompt with ZERO tool
                                    # schemas anywhere in the rendered TCMM
                                    # memory, yet Claude still executed
                                    # search_files_mcp_sub-agents +
                                    # read_file_mcp_sub-agents correctly
                                    # AND answered "98 tools" when asked.
                                    # The model gets the tools entirely via
                                    # the native ``data["tools"]`` field
                                    # the proxy passes through untouched.
                                    # Pinning them into TCMM was pure
                                    # duplication on the wire (in audit row
                                    # aid=3072: 21 schemas × ~7KB each =
                                    # ~150KB of redundant immutable blocks).
                                    # Keep ``_client_tools`` extraction so
                                    # we can re-enable a SUMMARY pin later
                                    # (names + 1-liners, not full schemas)
                                    # if recall wants tool-awareness signals.
                                    if False and _client_tools:
                                        _pin_coros.append(_tcmm_pin_tool_definitions(
                                            conversation_id, _client_tools,
                                            user_id=tcmm_user_id,
                                        ))
                                    # 2026-05-18: workspace state is
                                    # fetched in parallel with pins but
                                    # NOT pinned (would land in the
                                    # cached static prefix and be wrong
                                    # the moment the user switches
                                    # project). Instead it's injected
                                    # as a separate system block right
                                    # before the user turn after render,
                                    # so it re-renders every turn and
                                    # tracks live daemon state. The
                                    # fetch is best-effort — sub-agents
                                    # unreachable or no daemon connected
                                    # → ``_workspace_state`` is None and
                                    # injection is a no-op.
                                    _ws_fetch_task = asyncio.create_task(
                                        _fetch_workspace_state(tcmm_user_id),
                                    )
                                    # 2026-05-18: MCP tool-schema fetch
                                    # runs in parallel with pins. The
                                    # proxy stamps these onto
                                    # ``data["tools"]`` after render so
                                    # Grok/OpenAI traffic gets function-
                                    # calling even when LibreChat's
                                    # custom endpoint doesn't forward
                                    # tools (only Agents does — but
                                    # users picking 'Grok' from the
                                    # dropdown still expect tool use to
                                    # work). Skipped when the client
                                    # already sent its own tools[].
                                    _tools_fetch_task = asyncio.create_task(
                                        _fetch_mcp_tool_schemas(),
                                    )
                                    # ``return_exceptions=False`` (default)
                                    # so ANY pin failure raises out of
                                    # ``gather()`` and the surrounding
                                    # try/except catches TCMMUnavailable
                                    # to return 503. Same hard-fail
                                    # contract as the sequential version.
                                    await asyncio.gather(*_pin_coros)

                                    # 3. Strip the client system from its
                                    # original location — it's now in TCMM's
                                    # render output via the pin, so leaving
                                    # the original in place would duplicate
                                    # it on the wire.
                                    _strip_client_system(data, _render_model)

                                    # 4. Render — TCMM returns format-aware
                                    # blocks + messages list. From here the
                                    # request handler is FORMAT-AGNOSTIC.
                                    render_result = await _tcmm_render(
                                        _render_model, user_msg,
                                        conv_id=conversation_id,
                                        user_id=tcmm_user_id,
                                    )
                                except TCMMUnavailable as _tcmm_err:
                                    logger.error(
                                        f"  [TCMM] hard-fail (pin or render) — "
                                        f"returning 503 to client: {_tcmm_err}"
                                    )
                                    from fastapi.responses import JSONResponse as _JR
                                    return _JR(
                                        status_code=503,
                                        content={
                                            "error": {
                                                "type": "tcmm_unavailable",
                                                "message": "TCMM memory service is unavailable. Request rejected to prevent silent memory loss.",
                                                "detail": str(_tcmm_err),
                                            }
                                        },
                                    )

                                # 5. Symmetric slot — one helper handles all
                                # three provider shapes. The proxy itself
                                # makes NO format decisions past this line.
                                try:
                                    _apply_render_to_request(
                                        data, headers, render_result,
                                    )
                                    # 5b. Inject live workspace state as
                                    # the last system block, right before
                                    # the user turn. Awaits the parallel
                                    # fetch we kicked off alongside pins.
                                    # Lives OUTSIDE the cached static
                                    # prefix on purpose — folders can
                                    # change between turns and we want
                                    # the model to see the new state
                                    # immediately, not after a cache TTL
                                    # rolls over.
                                    try:
                                        _ws_state = await _ws_fetch_task
                                    except Exception:
                                        _ws_state = None
                                    if _ws_state:
                                        _inject_workspace_state(
                                            data, _render_model, _ws_state,
                                        )
                                    # Inject MCP tool schemas if client
                                    # didn't send any — restores
                                    # function-calling on the custom
                                    # xAI endpoint.
                                    try:
                                        _tool_schemas = await _tools_fetch_task
                                    except Exception:
                                        _tool_schemas = []
                                    _inject_mcp_tools_if_missing(
                                        data, _render_model, _tool_schemas,
                                    )
                                    # [UNIVERSAL_SHADOW_TOOL_2026_05_22]
                                    # Inject the tcmm_record_turn shadow
                                    # tool for ALL backends (claude went
                                    # via SSO which already does this;
                                    # this branch covers Grok/OpenAI).
                                    # The model emits emit_class via a
                                    # structured tool_use, the proxy
                                    # intercepts and ships it back to
                                    # TCMM so block_class actually lands
                                    # in the archive — without this the
                                    # gateway-forward backends produced
                                    # 100% NULL block_class assistant
                                    # rows because the NLP fallback is
                                    # 429-storming.
                                    try:
                                        _inject_shadow_tool_for_backend(
                                            data, _render_model,
                                        )
                                        logger.info(
                                            f"  [SHADOW-TOOL] injected "
                                            f"tcmm_record_turn for "
                                            f"backend={_render_model}"
                                        )
                                    except Exception as _st_e:
                                        logger.warning(
                                            f"  [SHADOW-TOOL] inject "
                                            f"failed: {_st_e}"
                                        )
                                except ValueError as _ve:
                                    logger.error(
                                        f"  [TCMM] /render returned unknown "
                                        f"format — refusing to proceed: {_ve}"
                                    )
                                    from fastapi.responses import JSONResponse as _JR
                                    return _JR(
                                        status_code=502,
                                        content={
                                            "error": {
                                                "type": "tcmm_bad_response",
                                                "message": str(_ve),
                                            }
                                        },
                                    )

                                _fmt = render_result.get("format", "")
                                _stats = render_result.get("stats") or {}
                                _blocks = render_result.get("blocks") or []
                                logger.info(
                                    f"  [TCMM-RENDER] {_fmt}: "
                                    f"blocks={_stats.get('block_count', len(_blocks))} "
                                    f"cached={_stats.get('cached_block_count', 0)} "
                                    f"chars={_stats.get('prompt_chars', 0)} "
                                    f"ext_ttl={render_result.get('uses_extended_cache_ttl', False)} "
                                    f"client_sys_pinned={bool(_client_system)} "
                                    f"client_tools_pinned={len(_client_tools)}"
                                )
                                tcmm_active = True

                # 2026-05-15: legacy Anthropic cache_control assembly removed.
                # TCMM's renderer owns tier splitting + cache_control placement;
                # the proxy already slotted data["system"] = render.blocks above.
                # _apply_anthropic_cache + _split_tcmm_memory_into_tiers +
                # _cache_circuit_strip / _used_extended_ttl machinery deleted in
                # the same commit. _cap_cache_markers (below) still runs as the
                # 4-marker hard cap safety net.

                # Scrub malformed extended-thinking blocks from messages
                # before sending. LibreChat + LangGraph occasionally produce
                # a content block shaped ``{type:"thinking"}`` without the
                # required ``thinking`` text / ``signature`` strings — when
                # Anthropic sees that it 400s with "messages.N.content.M.
                # thinking.thinking: Field required". Seen twice (22 Apr
                # Sarel bb400c87 persisted, 23 Apr Petrus fresh turn) — we
                # police it here so no upstream path can bypass the filter.
                _thinking_scrubbed = _scrub_malformed_thinking(data)
                if _thinking_scrubbed:
                    logger.info(
                        f"  [SCRUB] dropped {_thinking_scrubbed} malformed "
                        f"thinking block(s) before sending to Anthropic"
                    )

                # Final safety net: hard-cap to Anthropic's 4-marker limit by
                # stripping the oldest cache_control from messages if needed.
                # LibreChat emits cache_control on tool_use/tool_result blocks
                # in long conversations, and that plus our TCMM/message markers
                # has already sent one request to 5 markers in prod ("A maximum
                # of 4 blocks with cache_control may be provided").
                _stripped = _cap_cache_markers(data)
                if _stripped:
                    logger.info(
                        f"  [CACHE] capped cache_control markers — stripped "
                        f"{_stripped} oldest to stay within Anthropic's limit of "
                        f"{_ANTHROPIC_CACHE_LIMIT}"
                    )

                # Strip Veilguard provenance envelopes from tool_result
                # blocks before the LLM sees them. TCMM ingest (run above)
                # has already pulled the metadata — the LLM only needs the
                # inner content.
                _vg_messages = data.get("messages")
                if isinstance(_vg_messages, list):
                    _stripped_envelopes = _strip_veilguard_envelopes_from_messages(_vg_messages)
                    if _stripped_envelopes:
                        logger.info(
                            f"  [VG-ENV] stripped {_stripped_envelopes} "
                            f"_veilguard envelope(s) from tool_result content"
                        )

                # Redact PII — fail-closed contract.
                #
                # If Presidio crashes mid-analyse (NLP model unload, OOM,
                # regex engine, etc.) the redactor raises
                # ``RedactionUnavailable``. We MUST NOT forward the
                # request — the user's raw input would land at the
                # upstream LLM with unredacted PII. Return 503 instead.
                # The TCMM hard-fail above uses the same pattern.
                import time as _pt
                _t = _pt.perf_counter()
                try:
                    redacted = redactor.redact_json(data, pii_session_id)
                except RedactionUnavailable as _redact_err:
                    logger.error(
                        f"  [PII] hard-fail — refusing to forward request "
                        f"(no silent fallback): {_redact_err}"
                    )
                    from fastapi.responses import JSONResponse as _JR
                    return _JR(
                        status_code=503,
                        content={
                            "error": {
                                "type": "redaction_unavailable",
                                "message": (
                                    "PII redaction service is unavailable. "
                                    "Request rejected to prevent leaking "
                                    "unredacted personal data to upstream LLM."
                                ),
                                "detail": str(_redact_err),
                            }
                        },
                    )
                _redact_ms = (_pt.perf_counter() - _t) * 1000
                if hasattr(request.state, "phase_t"):
                    request.state.phase_t["redact"] = _redact_ms
                _t = _pt.perf_counter()
                body = json.dumps(redacted, ensure_ascii=False).encode("utf-8")
                if hasattr(request.state, "phase_t"):
                    request.state.phase_t["json_dump"] = (_pt.perf_counter() - _t) * 1000
                headers["content-length"] = str(len(body))
                # [CACHE-WIRE-GROK-2026-05-20] log full body sha for diagnosis
                try:
                    import hashlib as _hashlib_cw
                    _full_sha = _hashlib_cw.sha1(body).hexdigest()[:12]
                    _r_tools = redacted.get("tools") or []
                    _t_canon = json.dumps(_r_tools, sort_keys=False, ensure_ascii=False)
                    _t_canon_sorted = json.dumps(_r_tools, sort_keys=True, ensure_ascii=False)
                    _t_sha = _hashlib_cw.sha1(_t_canon.encode("utf-8", "replace")).hexdigest()[:12] if _r_tools else "-"
                    _t_sha_sorted = _hashlib_cw.sha1(_t_canon_sorted.encode("utf-8", "replace")).hexdigest()[:12] if _r_tools else "-"
                    _t_names = []
                    for _tt in _r_tools[:5]:
                        if isinstance(_tt, dict):
                            _n = (_tt.get("function") or {}).get("name") or _tt.get("name") or "?"
                            _t_names.append(_n)
                    _msgs = redacted.get("messages") or []
                    _msg_roles = [m.get("role") for m in _msgs if isinstance(m, dict)]
                    _msg_shas = []
                    for _m in _msgs[:3]:
                        if isinstance(_m, dict):
                            _c = _m.get("content")
                            if isinstance(_c, str):
                                _msg_shas.append(_hashlib_cw.sha1(_c.encode("utf-8", "replace")).hexdigest()[:8])
                            elif isinstance(_c, list):
                                _bb = json.dumps(_c, sort_keys=False, ensure_ascii=False).encode("utf-8", "replace")
                                _msg_shas.append(_hashlib_cw.sha1(_bb).hexdigest()[:8])
                            else:
                                _msg_shas.append("-")
                    logger.info(
                        f"  [CACHE-WIRE-GROK] body_sha={_full_sha} bytes={len(body)} "
                        f"tools_sha={_t_sha} tools_sorted_sha={_t_sha_sorted} tools_count={len(_r_tools)} "
                        f"tool_names_first5={_t_names} "
                        f"tool_choice={redacted.get('tool_choice')!r} "
                        f"temp={redacted.get('temperature')!r} "
                        f"max_tokens={redacted.get('max_tokens')!r} "
                        f"parallel={redacted.get('parallel_tool_calls')!r} "
                        f"top_keys={sorted(redacted.keys())} "
                        f"msg_roles={_msg_roles} "
                        f"first_msg_shas={_msg_shas}"
                    )
                except Exception as _cw_e:
                    logger.warning(f"  [CACHE-WIRE-GROK] log failed: {_cw_e}")


                # Origin-aware diagnostic: count each message's classified origin
                # so we can see the tool/user/assistant mix per request in logs.
                # Opt-in full-body dump via VEILGUARD_TOOL_DUMP=1 for tool-bearing
                # requests (used to forensically inspect the Anthropic envelope).
                try:
                    _origin_counts = {}
                    _has_tool = False
                    for _m in redacted.get("messages", []):
                        _o = classify_message_origin(_m)
                        _origin_counts[_o] = _origin_counts.get(_o, 0) + 1
                        if _o in ("tool_use", "tool_result", "tool"):
                            _has_tool = True
                    if _origin_counts:
                        _mix = " ".join(f"{k}={v}" for k, v in sorted(_origin_counts.items()))
                        logger.info(f"  [ORIGIN] {_mix}")
                    if _has_tool and os.environ.get("VEILGUARD_TOOL_DUMP") == "1":
                        _dump_dir = "/app/logs/tool_dumps"
                        os.makedirs(_dump_dir, exist_ok=True)
                        _ts = __import__("time").time()
                        _fname = f"{_dump_dir}/{_ts:.3f}_full_body.json"
                        with open(_fname, "wb") as _fp:
                            _fp.write(body)
                        logger.info(f"  [TOOL-DUMP] {_fname} ({len(body)} bytes)")
                except Exception:
                    pass

                # Wire-level hash of the system field (post-redaction, pre-send).
                # Two consecutive turns with the same sys_sha share a cache key.
                # Set VEILGUARD_CACHE_DUMP=1 to also dump full bytes to disk.
                try:
                    import hashlib as _hashlib
                    _sys_field = redacted.get("system")
                    if isinstance(_sys_field, list):
                        _sys_bytes = "".join(
                            str(b.get("text", "")) for b in _sys_field if isinstance(b, dict)
                        ).encode("utf-8")
                    elif isinstance(_sys_field, str):
                        _sys_bytes = _sys_field.encode("utf-8")
                    else:
                        _sys_bytes = b""
                    if _sys_bytes:
                        _sys_sha = _hashlib.sha1(_sys_bytes).hexdigest()[:12]
                        logger.info(
                            f"  [CACHE-WIRE] sys_sha={_sys_sha} sys_bytes={len(_sys_bytes)}"
                        )
                        if os.environ.get("VEILGUARD_CACHE_DUMP") == "1":
                            try:
                                _dump_dir = "/app/logs/cache_dumps"
                                os.makedirs(_dump_dir, exist_ok=True)
                                _ts = __import__("time").time()
                                _fname = f"{_dump_dir}/{_ts:.3f}_{_sys_sha}.txt"
                                with open(_fname, "wb") as _fp:
                                    _fp.write(_sys_bytes)
                            except Exception:
                                pass
                except Exception:
                    pass

                # Audit log: what we're sending to the LLM (redacted).
                # Full payload, no truncation — the DB-backed audit
                # (app.audit_db) needs the complete envelope for
                # replay / debugging of long-context prompts, and the
                # text log file is kept in sync so tail -f still works.
                _redacted_messages = redacted.get("messages", [])
                _redacted_system = redacted.get("system", "")
                _audit_text_parts: list[str] = []
                if _redacted_system:
                    # System field may be a string or a list of content
                    # blocks (cached TCMM split).  Render both.
                    if isinstance(_redacted_system, list):
                        _sys_rendered = "\n".join(
                            str(b.get("text", "")) for b in _redacted_system
                            if isinstance(b, dict)
                        )
                    else:
                        _sys_rendered = str(_redacted_system)
                    _audit_text_parts.append(f"[SYSTEM]\n{_sys_rendered}")
                # [TOOLS_AUDIT_SECTION_2026-06-09] Re-attach the native tools to
                # the audit copy (the model still gets them via the native
                # ``tools`` field, untouched). Lets the admin dashboard show the
                # TOOLS section + account their tokens, which broke when tool
                # schemas left the prompt text for the native field.
                _tools_sec = _tools_audit_section(redacted.get("tools"))
                if _tools_sec:
                    _audit_text_parts.append(_tools_sec)
                for _m in _redacted_messages:  # ALL messages, not last 3
                    _role = _m.get("role", "?")
                    _content = _m.get("content", "")
                    if isinstance(_content, list):
                        # Render each block's text / content verbatim.
                        _content = "\n".join(
                            str(b.get("text", b.get("content", "")))
                            for b in _content if isinstance(b, dict)
                        )
                    # 2026-05-19: also render ``tool_calls`` (OpenAI /
                    # xAI function-calling envelope). When the assistant
                    # emits ``{"role":"assistant","content":null,"tool_calls":[...]}``
                    # the ``content`` field is empty, so the audit used to
                    # show ``[ASSISTANT]\n`` with a blank body — making it
                    # look like the model said nothing when in fact it
                    # asked for a tool to run. We surface a compact
                    # ``→ tool_call name(arg_summary)`` line per call so
                    # the audit faithfully reflects what the LLM did.
                    _tool_calls = _m.get("tool_calls")
                    if isinstance(_tool_calls, list) and _tool_calls:
                        _tc_lines: list[str] = []
                        for _tc in _tool_calls:
                            if not isinstance(_tc, dict):
                                continue
                            _fn = _tc.get("function") or {}
                            _name = _fn.get("name") or _tc.get("name") or "?"
                            _args = _fn.get("arguments")
                            # Arguments come as a JSON STRING in OpenAI's
                            # wire shape. Render the first ~200 chars so
                            # the audit shows what file/command/path the
                            # tool was invoked with.
                            if isinstance(_args, str):
                                _args_render = _args[:200] + ("…" if len(_args) > 200 else "")
                            elif isinstance(_args, dict):
                                try:
                                    _args_render = json.dumps(_args)[:200]
                                except Exception:
                                    _args_render = str(_args)[:200]
                            else:
                                _args_render = ""
                            _tc_lines.append(f"→ tool_call {_name}({_args_render})")
                        if _tc_lines:
                            # If content was empty, replace it; if it
                            # had text too, append. Either way, the
                            # audit now reflects the tool_call.
                            _content = ((_content or "") + ("\n" if _content else "")
                                        + "\n".join(_tc_lines))
                    _audit_text_parts.append(f"[{_role.upper()}]\n{_content}")
                _audit_text = "\n\n".join(_audit_text_parts)

                _model_id = redacted.get("model", "?")
                audit_log(
                    "TO_LLM", conversation_id, _audit_text,
                    f"model={_model_id}",
                )
                # Write the full payload to LanceDB as well (separate
                # from the text log — DB rows are queryable, multi-
                # tenant, and never truncated).  Lives in a sibling
                # table of TCMM's archive; TCMM itself is never aware.
                try:
                    from app import audit_db as _audit_db
                    _audit_db.record(
                        direction="TO_LLM",
                        conversation_id=conversation_id or "",
                        user_id=tcmm_user_id or "",
                        model=_model_id if _model_id != "?" else None,
                        stream=bool(is_stream),
                        content=_audit_text,
                    )
                except Exception as _e:
                    logger.warning(f"[audit_db] TO_LLM record failed: {_e}")

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

                # Combined-text rehydration pass.  The per-chunk
                # rehydrate at line 1408 catches tokens that arrive
                # whole within a single chunk, but misses tokens that
                # straddle a chunk boundary (e.g. "REF_PERSO" +
                # "N_2") — the regex needs the complete token to
                # match.  Opus 4.7 chunks differ from Sonnet's and
                # frequently split placeholders, so users see raw
                # ``REF_PERSON_1`` in the UI for Opus conversations
                # while Sonnet works fine.  After events reassemble
                # into all_content_text the token is whole again, so
                # a final rehydrate pass over clean_content recovers
                # the split tokens.  Idempotent for already-rehydrated
                # text.
                if conversation_id:
                    rehydrated = redactor.rehydrate_text(clean_content, pii_session_id)
                    if rehydrated != clean_content:
                        logger.info(
                            f"  [REHYDRATE] combined-text pass recovered "
                            f"{len(rehydrated) - len(clean_content):+d} chars "
                            f"(split tokens across chunks)"
                        )
                        clean_content = rehydrated
                        heatmap_stripped = True  # trigger rebuild path
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
                try:
                    from app import audit_db as _audit_db
                    # Stamp model from the request we just sent (_model_id
                    # captured at line ~1686 as redacted.get("model", "?")).
                    # Without this the FROM_LLM row shows model=null and
                    # cost analysis over the audit log has to join TO_LLM
                    # rows on conversation_id to reconstruct the model.
                    _from_model = _model_id if _model_id and _model_id != "?" else None
                    _audit_db.record(
                        direction="FROM_LLM",
                        conversation_id=conversation_id or "",
                        user_id=tcmm_user_id or "",
                        model=_from_model,
                        stream=True,
                        content=all_content_text or "",
                        # 2026-05-19: tokens_input = TOTAL input tokens
                        # (new + cache_creation + cache_read) so the
                        # dashboard column matches OpenAI's prompt_tokens
                        # semantics — total of which cache_read is a
                        # subset. Previously this stored only the
                        # uncached "new" portion, making cache-heavy
                        # turns look tiny (e.g. 757 tokens_in beside
                        # cache_rd=39.7k).
                        tokens_input=(
                            (_cache_usage.get("input_tokens") or 0)
                            + (_cache_usage.get("cache_creation_input_tokens") or 0)
                            + (_cache_usage.get("cache_read_input_tokens") or 0)
                        ) if _cache_usage else None,
                        tokens_output=_cache_usage.get("output_tokens") if _cache_usage else None,
                        cache_create=_cache_usage.get("cache_creation_input_tokens") if _cache_usage else None,
                        cache_read=_cache_usage.get("cache_read_input_tokens") if _cache_usage else None,
                    )
                except Exception as _e:
                    logger.warning(f"[audit_db] FROM_LLM record failed: {_e}")

                # Phase 7 step 2: per-tenant rolling cache-metrics rollup.
                # Feeds the /cache/stats endpoint and the circuit breaker
                # decision in step 6. Fire-and-forget — failures here
                # must not break the response stream.
                try:
                    from app import cache_metrics as _cache_metrics
                    _cache_metrics.record_from_usage(
                        _cache_usage,
                        tenant_id=tcmm_user_id or "",
                        conv_id=conversation_id or "",
                        model=_from_model,
                    )
                except Exception as _e:
                    logger.warning(f"[cache_metrics] record failed: {_e}")

                # Feed full content (WITH heatmap) to TCMM for learning.
                # We MUST rehydrate before ingest — per-chunk rehydrate at
                # line ~1408 misses tokens that straddle chunk boundaries
                # (Opus 4.7 chunks differently to Sonnet and regularly
                # splits ``REF_PERSON_1`` across deltas). ``clean_content``
                # got the combined-text rehydrate pass at ~1848 but had
                # heatmap stripped. TCMM wants WITH-heatmap content, so
                # we rehydrate ``all_content_text`` here. rehydrate_text
                # is idempotent — safe if it was already fully restored.
                # Without this, REF_* tokens leak into archive.text and
                # later conversations pull them back in as ghost
                # placeholders that Claude fills with fabricated names
                # like "Jun Hirata" (observed 23 Apr 2026, aid=641).
                if tcmm_active and all_content_text:
                    tcmm_content = (
                        redactor.rehydrate_text(all_content_text, pii_session_id)
                        if pii_session_id
                        else all_content_text
                    )
                    await _tcmm_post_response(tcmm_content, conversation_id, user_id=tcmm_user_id, lineage_parent_conv=tcmm_lineage_parent)
                    logger.info(f"  [TCMM] Anthropic stream done, ingested {len(tcmm_content)} chars")

                try:
                    await response.aclose()
                    await client.aclose()
                except Exception:
                    pass
                return

            # TCMM path: SAFE-BOUNDARY emit with whole-text rehydration.
            #
            # Two production bugs the previous PARSE-AND-RECONSTRUCT design
            # hit (observed 13 May 2026 against xAI/Grok):
            #   (A) The per-chunk rehydrate ran on each TCP chunk's decoded
            #       text — but Presidio token regexes need the COMPLETE
            #       token to match.  When ``REF_PERSON_1`` straddled a
            #       chunk boundary ("REF_PER" + "SON_1"), neither chunk
            #       matched and the raw token leaked into the UI.
            #   (B) HOLD_BACK=30 SSE events was too small.  Logs showed
            #       ``Heatmap partially in yielded content (offset=-28)``:
            #       the trailing {"knowledge_class": ...} JSON began
            #       BEFORE the 30-event tail, so we'd already shipped the
            #       opening brace and couldn't retract.
            #
            # New design:
            #   * Only the raw (still-redacted) content is buffered.
            #     finish_reason / usage / role events are held in
            #     ``end_events`` and forwarded after content.
            #   * After each network chunk we rehydrate the FULL
            #     accumulated content (rehydration is local + idempotent
            #     so prefix stays stable across calls) and emit a fresh
            #     synthetic content-delta SSE event for the prefix that
            #     sits ``SAFE_TAIL_CHARS`` behind the tail.  Anything
            #     within the tail is held — that window comfortably
            #     covers a split REF token AND a trailing heatmap.
            #   * At end-of-stream we run the heatmap detector on the
            #     fully-rehydrated text, trim, and emit whatever remains
            #     past ``yielded_content_len``.  Because the heatmap is
            #     always at the very end and ≤ SAFE_TAIL_CHARS in size,
            #     it never reaches the safe-emit prefix — no more
            #     "partially in yielded content" failures.
            #
            # SAFE_TAIL_CHARS = 512 covers:
            #   - PII token + margin (REF_PERSON_NN ≈ 14 chars + buffer)
            #   - longest observed heatmap (≈ 200 chars; 512 is 2.5× headroom)
            # Streaming impact: the user sees the last ~512 chars in one
            # final emit instead of token-by-token.  For Grok answers in
            # the 100-600 char range this is imperceptible.

            SAFE_TAIL_CHARS = 512

            raw_content = []          # raw (still-redacted) deltas from upstream
            yielded_content_len = 0   # chars of REHYDRATED text already emitted
            sse_buffer = ""           # partial SSE line accumulator
            end_events = []           # finish_reason/usage events held for end

            # [UNIVERSAL_SHADOW_TOOL_2026_05_22] OpenAI/xAI shadow-tool
            # capture. tool_calls deltas arrive interleaved by index;
            # the FIRST delta for an index carries the function.name,
            # subsequent ones only stream `function.arguments` chunks.
            # We tag each index that belongs to tcmm_record_turn and
            # accumulate its args string for end-of-stream parsing.
            # Indices we tag are also STRIPPED from the events queued
            # to end_events, so LibreChat never sees the shadow tool
            # and can't try to dispatch it as a real MCP tool.
            _shadow_tcmm_indices: set = set()       # tool_call indices that belong to tcmm_record_turn
            _shadow_args_by_idx: dict = {}          # idx -> accumulated args JSON string
            _shadow_flag_obj_captured: dict = {}    # final parsed dict after stream ends

            def _build_content_event(text: str) -> bytes:
                """Synthesize an OpenAI-style content-delta SSE event."""
                obj = {
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None,
                    }],
                }
                return f"data: {json.dumps(obj)}\n\n".encode("utf-8")

            # Incremental UTF-8 decoder: ``response.aiter_bytes()`` hands
            # us raw TCP chunks, and a multibyte char (em-dash, smart
            # quote, emoji) routinely straddles a chunk boundary.  A
            # one-shot ``bytes.decode("utf-8")`` per chunk would either
            # raise (errors="strict") or replace the partial sequence
            # with U+FFFD (errors="replace") — both corrupt the byte
            # stream.  The incremental decoder buffers any partial
            # multibyte sequence and emits it when the next chunk
            # completes it.  ``final=True`` is invoked once at end of
            # stream below to flush any dangling bytes.
            import codecs as _codecs
            _utf8_decoder = _codecs.getincrementaldecoder("utf-8")(errors="replace")

            try:
                async for chunk in response.aiter_bytes():
                    text = _utf8_decoder.decode(chunk)

                    # Accumulate raw bytes; split into complete SSE lines.
                    # NOTE: do NOT rehydrate ``text`` here — split tokens
                    # need the WHOLE concatenated content to rehydrate
                    # correctly.  We rehydrate once per outer chunk on the
                    # full buffer below.
                    sse_buffer += text
                    lines = sse_buffer.split("\n")
                    sse_buffer = lines[-1]  # incomplete tail

                    for raw_line in lines[:-1]:
                        stripped = raw_line.strip()
                        if not stripped:
                            continue
                        if stripped == "data: [DONE]":
                            # Forward at the very end after content drained
                            end_events.append("data: [DONE]\n\n")
                            continue
                        if not stripped.startswith("data: "):
                            # Comment / event: line — pass through immediately
                            yield (raw_line + "\n\n").encode("utf-8")
                            continue

                        # Parse the data: payload
                        try:
                            payload = json.loads(stripped[6:])
                        except (json.JSONDecodeError, ValueError):
                            # Malformed — forward as-is, don't try to be clever
                            yield (raw_line + "\n\n").encode("utf-8")
                            continue

                        try:
                            choice = payload.get("choices", [{}])[0]
                            delta = choice.get("delta", {}) or {}
                        except (IndexError, KeyError, AttributeError):
                            choice, delta = {}, {}

                        cs = delta.get("content")
                        content_consumed = isinstance(cs, str) and bool(cs)
                        if content_consumed:
                            raw_content.append(cs)

                        # Anything besides pure content?  ``tool_calls``,
                        # ``function_call``, ``reasoning_content``,
                        # ``role``, ``finish_reason``, ``usage`` — all
                        # must be preserved or the downstream client
                        # (LibreChat MCP dispatcher) will never see the
                        # function the model wants to call.
                        #
                        # The previous version of this branch only queued
                        # events with finish_reason / usage / role-only,
                        # so xAI's ``delta.tool_calls`` deltas fell off
                        # the bottom of the loop and were silently
                        # dropped — Grok's run_command invocation arrived
                        # at the daemon with mangled args.  Fix: queue
                        # any event that has data beyond ``content``.
                        has_finish = choice.get("finish_reason") is not None
                        has_usage = bool(payload.get("usage"))
                        extra_delta_keys = set(delta.keys()) - {"content"}
                        if has_finish or has_usage or extra_delta_keys:
                            if has_usage:
                                try:
                                    _openai_final_usage = payload.get("usage") or {}
                                except Exception:
                                    pass
                            # [UNIVERSAL_SHADOW_TOOL_2026_05_22] Inspect
                            # tool_calls deltas for tcmm_record_turn.
                            # Accumulate args, strip those entries from
                            # the forwarded event. Mutates a parsed copy
                            # of the payload so we can re-serialize.
                            _tc_deltas = delta.get("tool_calls")
                            _shadow_modified = False
                            if isinstance(_tc_deltas, list) and _tc_deltas:
                                _kept_tcs = []
                                for _tcd in _tc_deltas:
                                    if not isinstance(_tcd, dict):
                                        _kept_tcs.append(_tcd)
                                        continue
                                    _idx = _tcd.get("index")
                                    _fn = _tcd.get("function") or {}
                                    _name = _fn.get("name")
                                    # First delta for this index sets the
                                    # name. If it's our shadow tool, tag.
                                    if _name == "tcmm_record_turn":
                                        if _idx is not None:
                                            _shadow_tcmm_indices.add(_idx)
                                            _shadow_args_by_idx.setdefault(_idx, "")
                                        _shadow_modified = True
                                        # Also capture args fragment if present
                                        _args_frag = _fn.get("arguments")
                                        if isinstance(_args_frag, str) and _idx is not None:
                                            _shadow_args_by_idx[_idx] += _args_frag
                                        continue
                                    # No name in this delta. If the index
                                    # was previously tagged, this is a
                                    # continuation of the tcmm_record_turn
                                    # args stream.
                                    if _idx is not None and _idx in _shadow_tcmm_indices:
                                        _args_frag = _fn.get("arguments")
                                        if isinstance(_args_frag, str):
                                            _shadow_args_by_idx[_idx] += _args_frag
                                        _shadow_modified = True
                                        continue
                                    # Real tool_call delta — preserve.
                                    _kept_tcs.append(_tcd)
                                if _shadow_modified:
                                    # If nothing real remained, drop the
                                    # tool_calls key entirely from the
                                    # delta to avoid an empty array
                                    # confusing the client.
                                    if _kept_tcs:
                                        try:
                                            cleaned = json.loads(stripped[6:])
                                            cleaned_delta = cleaned.get("choices", [{}])[0].get("delta", {}) or {}
                                            cleaned_delta["tool_calls"] = _kept_tcs
                                            if content_consumed:
                                                cleaned_delta.pop("content", None)
                                            end_events.append(
                                                f"data: {json.dumps(cleaned)}\n\n"
                                            )
                                        except Exception:
                                            end_events.append(stripped + "\n\n")
                                    else:
                                        # Only shadow tool_calls in this
                                        # delta — drop the whole event if
                                        # there's also no finish/usage/
                                        # other-content to ship.
                                        if has_finish or has_usage or (extra_delta_keys - {"tool_calls"}) or content_consumed:
                                            try:
                                                cleaned = json.loads(stripped[6:])
                                                cleaned_delta = cleaned.get("choices", [{}])[0].get("delta", {}) or {}
                                                cleaned_delta.pop("tool_calls", None)
                                                if content_consumed:
                                                    cleaned_delta.pop("content", None)
                                                end_events.append(
                                                    f"data: {json.dumps(cleaned)}\n\n"
                                                )
                                            except Exception:
                                                pass
                                        # else: pure-shadow event, drop entirely
                                    continue
                            # Strip ``content`` from a mixed event to
                                                            # avoid double-emit: the synthetic content
                            # event we yield below already covers it.
                            if content_consumed and extra_delta_keys:
                                try:
                                    cleaned = json.loads(stripped[6:])
                                    cleaned_delta = cleaned.get("choices", [{}])[0].get("delta", {}) or {}
                                    cleaned_delta.pop("content", None)
                                    end_events.append(
                                        f"data: {json.dumps(cleaned)}\n\n"
                                    )
                                except Exception:
                                    end_events.append(stripped + "\n\n")
                            else:
                                end_events.append(stripped + "\n\n")
                            continue

                    # After draining this chunk's lines, try to emit a
                    # safe content prefix.
                    if raw_content:
                        full_raw = "".join(raw_content)
                        if pii_session_id:
                            full_rehydrated = redactor.rehydrate_text(full_raw, pii_session_id)
                        else:
                            full_rehydrated = full_raw
                        safe_len = max(
                            yielded_content_len,
                            len(full_rehydrated) - SAFE_TAIL_CHARS,
                        )
                        if safe_len > yielded_content_len:
                            delta_text = full_rehydrated[yielded_content_len:safe_len]
                            if delta_text:
                                yield _build_content_event(delta_text)
                                yielded_content_len = safe_len

            except (httpx.ReadError, httpx.RemoteProtocolError) as e:
                logger.warning(f"Stream ended: {e}")
            finally:
                try:
                    await response.aclose()
                    await client.aclose()
                except Exception:
                    pass

                # Flush any partial multibyte sequence still buffered in
                # the incremental decoder (rare, but possible if the
                # stream ended mid-character).
                try:
                    sse_buffer += _utf8_decoder.decode(b"", final=True)
                except Exception:
                    pass

                # Drain any final partial SSE line that didn't get a
                # trailing newline before the stream closed.  Apply
                # the same routing rules as the main loop so tool_call
                # / function_call / reasoning_content / usage in a
                # truncated last event aren't dropped.
                if sse_buffer.strip():
                    stripped = sse_buffer.strip()
                    if stripped == "data: [DONE]":
                        end_events.append("data: [DONE]\n\n")
                    elif stripped.startswith("data: "):
                        try:
                            payload = json.loads(stripped[6:])
                            choice = payload.get("choices", [{}])[0]
                            delta = choice.get("delta", {}) or {}
                            cs = delta.get("content")
                            content_consumed = isinstance(cs, str) and bool(cs)
                            if content_consumed:
                                raw_content.append(cs)
                            has_finish = choice.get("finish_reason") is not None
                            has_usage = bool(payload.get("usage"))
                            extra_delta_keys = set(delta.keys()) - {"content"}
                            if has_finish or has_usage or extra_delta_keys:
                                if has_usage:
                                    try:
                                        _openai_final_usage = payload.get("usage") or {}
                                    except Exception:
                                        pass
                                if content_consumed and extra_delta_keys:
                                    try:
                                        cleaned = json.loads(stripped[6:])
                                        cleaned_delta = cleaned.get("choices", [{}])[0].get("delta", {}) or {}
                                        cleaned_delta.pop("content", None)
                                        end_events.append(
                                            f"data: {json.dumps(cleaned)}\n\n"
                                        )
                                    except Exception:
                                        end_events.append(stripped + "\n\n")
                                else:
                                    end_events.append(stripped + "\n\n")
                        except (json.JSONDecodeError, ValueError, IndexError, KeyError):
                            pass

                # Final pass on the fully-accumulated content.
                full_raw = "".join(raw_content)
                if pii_session_id:
                    full_rehydrated = redactor.rehydrate_text(full_raw, pii_session_id)
                else:
                    full_rehydrated = full_raw
                full_content = full_rehydrated  # for audit/TCMM below

                # Detect trailing heatmap JSON: scan backwards for the
                # last "{" that opens a valid dict with knowledge_class
                # or used.
                heatmap_start = -1
                search_from = len(full_rehydrated)
                while search_from > 0:
                    pos = full_rehydrated.rfind("{", 0, search_from)
                    if pos < 0:
                        break
                    candidate_str = full_rehydrated[pos:].strip()
                    try:
                        candidate = json.loads(candidate_str)
                        if isinstance(candidate, dict) and (
                            "knowledge_class" in candidate or "used" in candidate
                        ):
                            heatmap_start = pos
                            break
                    except (json.JSONDecodeError, ValueError):
                        pass
                    search_from = pos  # try an earlier {

                # Decide what visible-answer text the client should see.
                if heatmap_start >= 0:
                    clean_end = heatmap_start
                    heatmap_text = full_rehydrated[heatmap_start:]
                else:
                    clean_end = len(full_rehydrated)
                    heatmap_text = ""
                visible_text = full_rehydrated[:clean_end].rstrip()

                # Emit whatever portion of visible_text we haven't shipped
                # yet.  Guarded against over-emit: if SAFE_TAIL_CHARS
                # was too small relative to the heatmap, we'd have
                # already shipped some heatmap chars — log it loudly.
                if len(visible_text) > yielded_content_len:
                    tail = visible_text[yielded_content_len:]
                    if tail:
                        yield _build_content_event(tail)
                        yielded_content_len = len(visible_text)
                elif len(visible_text) < yielded_content_len:
                    over = yielded_content_len - len(visible_text)
                    logger.warning(
                        f"  [TCMM] over-emitted by {over} chars before "
                        f"heatmap detection — bump SAFE_TAIL_CHARS"
                    )

                if heatmap_start >= 0:
                    logger.info(
                        f"  [TCMM] Stripped heatmap from stream "
                        f"({len(heatmap_text)} chars)"
                    )

                # Drain end_events: finish/usage/role/[DONE].  Move [DONE]
                # to the very end so clients that close on [DONE] still
                # see the finish_reason and usage events first.
                done_events = [e for e in end_events if e.strip() == "data: [DONE]"]
                other_end = [e for e in end_events if e.strip() != "data: [DONE]"]
                for ev in other_end:
                    yield ev.encode("utf-8")
                if done_events:
                    yield b"data: [DONE]\n\n"
                elif heatmap_start >= 0:
                    # Upstream didn't send [DONE] but we want clients to
                    # know the stream is over after our heatmap rewrite.
                    yield b"data: [DONE]\n\n"

                # Audit log: what the LLM returned
                audit_log("FROM_LLM", conversation_id, full_content or "(empty)", "stream=openai")
                try:
                    from app import audit_db as _audit_db
                    # OpenAI streaming responses emit usage in the final
                    # chunk (opt-in via ``stream_options: {"include_usage": true}``).
                    # If LibreChat hasn't enabled that, usage will be None.
                    _oai_usage = locals().get("_openai_final_usage") or {}
                    _from_model = _model_id if _model_id and _model_id != "?" else None
                    # 2026-05-18 BUG FIX: previously omitted cache_read for
                    # streaming OpenAI/xAI calls, leaving the column NULL
                    # in pii_audit. Dashboard showed "—" instead of the
                    # actual cache stats. xAI returns cached_tokens under
                    # ``usage.prompt_tokens_details.cached_tokens`` in the
                    # final SSE chunk (same shape as the non-streaming
                    # path at line ~3456). cache_create is None for xAI/
                    # OpenAI — they don't expose a creation counter.
                    _oai_details = _oai_usage.get("prompt_tokens_details") or {}
                    _oai_cache_read = _oai_details.get("cached_tokens")
                    _audit_db.record(
                        direction="FROM_LLM",
                        conversation_id=conversation_id or "",
                        user_id=tcmm_user_id or "",
                        model=_from_model,
                        stream=True,
                        content=full_content or "",
                        tokens_input=_oai_usage.get("prompt_tokens"),
                        tokens_output=_oai_usage.get("completion_tokens"),
                        cache_create=None,  # not exposed by OpenAI/xAI API
                        cache_read=_oai_cache_read,
                    )
                except Exception as _e:
                    logger.warning(f"[audit_db] FROM_LLM record failed: {_e}")

                # [UNIVERSAL_SHADOW_TOOL_2026_05_22] Parse accumulated
                # shadow-tool args (collected from tool_calls deltas
                # tagged tcmm_record_turn). One JSON object per index,
                # but in practice the model only invokes our tool once
                # per turn so there's almost always a single index.
                if _shadow_args_by_idx:
                    for _idx, _args_str in _shadow_args_by_idx.items():
                        if not _args_str:
                            continue
                        try:
                            _parsed = json.loads(_args_str)
                        except Exception as _pe:
                            logger.warning(
                                f"  [SHADOW-TOOL] failed to parse args "
                                f"idx={_idx} len={len(_args_str)}: {_pe}"
                            )
                            continue
                        if isinstance(_parsed, dict):
                            _shadow_flag_obj_captured = _parsed
                            logger.info(
                                f"  [SHADOW-TOOL] captured emit_class="
                                f"{_parsed.get('emit_class')!r} "
                                f"knowledge_class={_parsed.get('knowledge_class')!r} "
                                f"epoch_complete={_parsed.get('epoch_complete')!r}"
                            )
                            break

                # Feed content to TCMM for learning.  ``full_content`` is
                # already the rehydrated full response (assigned during
                # the safe-boundary emit pass), so no second rehydrate is
                # needed — that also makes the post a no-op for the
                # idempotent-but-not-free regex pass.
                # [GROK_TOOL_NARRATION_STRIP_2026_05_22] scrub any prose
                # narration of the tcmm_record_turn call (Grok quirk)
                # before persisting — keeps the archive text clean even
                # if the user-facing stream already shipped the noise.
                if raw_content:
                    tcmm_content = _strip_tcmm_tool_narration(full_content)
                    await _tcmm_post_response(
                        tcmm_content, conversation_id,
                        user_id=tcmm_user_id,
                        lineage_parent_conv=tcmm_lineage_parent,
                        flag_obj=_shadow_flag_obj_captured or None,
                    )
                    logger.info(f"  [TCMM] Stream done, ingested {len(tcmm_content)} chars")

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
        # [PERF-INSTR] Time the upstream Anthropic call separately from
        # rehydrate so we can attribute "Anthropic round-trip" vs "our
        # rehydrate cost" cleanly in logs.
        import time as _pt
        async with httpx.AsyncClient(timeout=300) as client:
            _t = _pt.perf_counter()
            response = await client.request(
                method=request.method, url=target_url,
                content=body if body else None, headers=headers,
            )
            if hasattr(request.state, "phase_t"):
                request.state.phase_t["anthropic"] = (_pt.perf_counter() - _t) * 1000

            logger.info(f"<<< [{backend_name}] {response.status_code} /{remaining_path}")

            # Rehydrate PII in response (must use same session ID as redaction)
            resp_body = response.content
            _t = _pt.perf_counter()
            if pii_session_id:
                try:
                    resp_text = resp_body.decode("utf-8")
                    resp_text = redactor.rehydrate_text(resp_text, pii_session_id)
                    resp_body = resp_text.encode("utf-8")
                except UnicodeDecodeError:
                    pass
            if hasattr(request.state, "phase_t"):
                request.state.phase_t["rehydrate"] = (_pt.perf_counter() - _t) * 1000
                _phases = request.state.phase_t
                _conv = (conversation_id or "?")[:8]
                logger.info(
                    "  [PHASE-PROXY] conv=%s  tcmm_pre_http=%.0f  redact=%.0f  json_dump=%.0f  anthropic=%.0f  rehydrate=%.0f  owned=%.0f"
                    % (
                        _conv,
                        _phases.get("tcmm_pre_http", 0),
                        _phases.get("redact", 0),
                        _phases.get("json_dump", 0),
                        _phases.get("anthropic", 0),
                        _phases.get("rehydrate", 0),
                        # owned = everything except the Anthropic call itself
                        _phases.get("tcmm_pre_http", 0)
                        + _phases.get("redact", 0)
                        + _phases.get("json_dump", 0)
                        + _phases.get("rehydrate", 0),
                    )
                )

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

                        # Non-streaming responses carry usage in the
                        # response body — extract it so FROM_LLM
                        # audit rows actually have token metrics.
                        # Previously this call site had no token
                        # fields, which is why audit-log token
                        # coverage was only ~18% (only anthropic
                        # streaming was populating them).
                        #
                        # Hoisted ABOVE the audit_db try-block so the
                        # downstream cache_metrics call (~30 lines
                        # below) sees these bindings even when
                        # audit_db's import itself raises (e.g. when
                        # the container is missing lancedb). Before
                        # the hoist, an audit_db import failure left
                        # _usage / _model_from_resp unbound, and
                        # cache_metrics.record() then died with an
                        # UnboundLocalError — surfacing as a noisy
                        # warning on every non-anthropic 200 response.
                        _usage = resp_json.get("usage", {}) or {}
                        _model_from_resp = resp_json.get("model") or (
                            _model_id if _model_id and _model_id != "?" else None
                        )
                        if is_anthropic_resp:
                            # 2026-05-19: total input = new + cache writes + cache reads.
                            # See parallel patch in the streaming record site.
                            _tok_in  = (
                                (_usage.get("input_tokens") or 0)
                                + (_usage.get("cache_creation_input_tokens") or 0)
                                + (_usage.get("cache_read_input_tokens") or 0)
                            )
                            _tok_out = _usage.get("output_tokens")
                            _cc      = _usage.get("cache_creation_input_tokens")
                            _cr      = _usage.get("cache_read_input_tokens")
                        else:
                            # OpenAI / xAI usage keys
                            _tok_in  = _usage.get("prompt_tokens")
                            _tok_out = _usage.get("completion_tokens")
                            # 2026-05-14: xAI / OpenAI report cache reads under
                            # ``usage.prompt_tokens_details.cached_tokens`` (the
                            # OpenAI prompt-cache shape). There's no separate
                            # "creation" counter on these providers — caching is
                            # automatic and we never explicitly flag cache blocks.
                            # Audit / dashboard previously showed "—" for every
                            # Grok call because we hard-coded None here.
                            _details = _usage.get("prompt_tokens_details") or {}
                            _cr = _details.get("cached_tokens")
                            _cc = None  # provider-implicit; never populated for xAI/OpenAI

                        try:
                            from app import audit_db as _audit_db
                            _audit_db.record(
                                direction="FROM_LLM",
                                conversation_id=conversation_id or "",
                                user_id=tcmm_user_id or "",
                                model=_model_from_resp,
                                stream=False,
                                content=raw_content or "",
                                tokens_input=_tok_in,
                                tokens_output=_tok_out,
                                cache_create=_cc,
                                cache_read=_cr,
                            )
                        except Exception as _e:
                            logger.warning(f"[audit_db] FROM_LLM record failed: {_e}")

                        # Phase 7 step 2: non-streaming path rollup.
                        try:
                            from app import cache_metrics as _cache_metrics
                            _cache_metrics.record_from_usage(
                                _usage if is_anthropic_resp else None,
                                tenant_id=tcmm_user_id or "",
                                conv_id=conversation_id or "",
                                model=_model_from_resp,
                            )
                        except Exception as _e:
                            logger.warning(f"[cache_metrics] record failed: {_e}")

                        # [UNIVERSAL_SHADOW_TOOL_2026_05_22] Capture shadow
                        # tool from the non-streaming response BEFORE
                        # ingesting. Anthropic: tool_use in content blocks.
                        # OpenAI/xAI: tool_calls in choices[0].message.
                        # Also strips the shadow entry from resp_json so
                        # the downstream client never sees it.
                        _ns_flag_obj: dict = {}
                        try:
                            if is_anthropic_resp:
                                _cb = resp_json.get("content")
                                _sr = resp_json.get("stop_reason")
                                _cleaned_cb, _ns_flag_obj, _new_sr = _intercept_tcmm_record_tool_use(_cb, _sr)
                                if _ns_flag_obj:
                                    resp_json["content"] = _cleaned_cb
                                    resp_json["stop_reason"] = _new_sr
                            else:
                                _ns_flag_obj, _was_modified = _extract_shadow_tool_from_openai_response(resp_json)
                            if _ns_flag_obj:
                                logger.info(
                                    f"  [SHADOW-TOOL] non-stream captured "
                                    f"emit_class={_ns_flag_obj.get('emit_class')!r}"
                                )
                        except Exception as _st_ns_e:
                            logger.warning(f"  [SHADOW-TOOL] non-stream extract failed: {_st_ns_e}")

                        # [GROK_TOOL_NARRATION_STRIP_2026_05_22] After
                        # _extract_shadow_tool_from_openai_response has
                        # scrubbed message.content in-place, re-read the
                        # content for TCMM ingest so the archive sees
                        # the clean text (not the Grok narration of the
                        # tcmm_record_turn call).
                        if not is_anthropic_resp:
                            _post_choices = resp_json.get("choices") or []
                            if _post_choices:
                                _scrubbed_content = (_post_choices[0].get("message") or {}).get("content")
                                if isinstance(_scrubbed_content, str):
                                    raw_content = _scrubbed_content

                        # Feed REAL content to TCMM (no redaction — private local storage)
                        if tcmm_active:
                            clean_answer = await _tcmm_post_response(
                                raw_content, conversation_id,
                                user_id=tcmm_user_id,
                                lineage_parent_conv=tcmm_lineage_parent,
                                flag_obj=_ns_flag_obj or None,
                            )
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
