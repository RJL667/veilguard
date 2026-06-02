"""MCP tools that bridge agents to TCMM memory + the constitution.

Three tools:
  recall(query, scope?)       — read TCMM via HTTP
  observe(text, kind?, ...)    — write TCMM; auto-stamps extracted_by=agent:<aid>
  read_constitution()         — return parsed constitution (for Director)

These wrap the existing middleware/tcmm.py + constitution/loader.py so
agents get a clean tool surface without each persona writing its own
HTTP code.  Tenant context (cid, user_id, agent_id) read from contextvars.

Provenance contract (spec §3.8.5):
  - The agent NEVER sets `source_kind`.  observe() hard-codes it based
    on context:
      * If ctx.is_background: source_kind=AGENT (extracted_by=agent:<aid>)
      * If foreground Director: source_kind=USER_PARAPHRASE (Director is
        summarizing user statements, not the user themselves)
  - Attempts by the LLM to override these via the `kind` parameter are
    ignored at the wrapper layer.  This closes the prompt-injection
    provenance-laundering hole.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from ..middleware import tenant
from ..middleware.tcmm import observe_agent_output

logger = logging.getLogger("agent-runtime.tools.memory_mcp")


_SDK_AVAILABLE = False

try:
    from claude_agent_sdk import (  # type: ignore
        create_sdk_mcp_server,
        tool,
    )
    _SDK_AVAILABLE = True
except Exception as e:  # pragma: no cover
    logger.warning(f"[memory_mcp] SDK import failed: {e!r}")

    def tool(name: str, description: str, input_schema: dict) -> Any:
        def _decorator(fn):
            fn._mcp_tool_meta = {
                "name": name,
                "description": description,
                "input_schema": input_schema,
            }
            return fn
        return _decorator

    def create_sdk_mcp_server(name: str, version: str, tools: list) -> None:
        return None


def _ctx_or_error() -> tuple[Optional[tenant.TenantContext], Optional[dict]]:
    ctx = tenant.current()
    if ctx is None:
        return None, {
            "content": [{
                "type": "text",
                "text": "ERROR: memory tool called without TenantContext",
            }],
            "isError": True,
        }
    return ctx, None


def _ok(data: dict) -> dict:
    return {
        "content": [{"type": "text", "text": json.dumps(data, default=str, indent=2)}],
    }


def _err(msg: str) -> dict:
    return {
        "content": [{"type": "text", "text": f"ERROR: {msg}"}],
        "isError": True,
    }


# ── recall ──────────────────────────────────────────────────────────────


@tool(
    name="recall",
    description=(
        "Search TCMM memory for content relevant to a query. Returns top-K "
        "matching memory blocks with their source_kind, extracted_by, and "
        "timestamps so you can judge provenance. scope filters where to "
        "search: 'conv' (current conversation), 'agent' (your private "
        "namespace), 'team_knowledge' (Critic-promoted team facts), "
        "'blackboard' (org-wide). Default is 'auto' (the platform picks "
        "based on your role)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "scope": {
                "type": "string",
                "enum": ["auto", "conv", "agent", "team_knowledge", "blackboard"],
                "default": "auto",
            },
            "limit": {"type": "integer", "default": 10},
        },
        "required": ["query"],
    },
)
async def recall_tool(args: dict[str, Any]) -> dict[str, Any]:
    ctx, err = _ctx_or_error()
    if err:
        return err

    import httpx
    from ..config import TCMM_URL, TCMM_RENDER_TIMEOUT_S, VEILGUARD_INTERNAL_SECRET

    headers = {}
    if VEILGUARD_INTERNAL_SECRET:
        headers["x-veilguard-internal-secret"] = VEILGUARD_INTERNAL_SECRET

    payload = {
        "query": args["query"],
        "conversation_id": ctx.conversation_id,
        "user_id": ctx.user_id,
        "agent_id": ctx.agent_id,
        "scope": args.get("scope", "auto"),
        "limit": args.get("limit", 10),
    }

    try:
        async with httpx.AsyncClient(timeout=TCMM_RENDER_TIMEOUT_S) as client:
            r = await client.post(
                f"{TCMM_URL.rstrip('/')}/recall",
                json=payload,
                headers=headers,
            )
        if r.status_code >= 400:
            return _err(f"recall returned {r.status_code}: {r.text[:200]}")
        return _ok(r.json())
    except Exception as e:
        logger.warning(f"[memory_mcp] recall failed: {e}")
        return _err(f"recall failed (TCMM unreachable?): {e}")


# ── observe ─────────────────────────────────────────────────────────────


@tool(
    name="observe",
    description=(
        "Write content to your private agent memory. Use for: (1) "
        "significant findings from a tool call (one observation per "
        "claim, with source link in body); (2) reflections at the end "
        "of a task. Do NOT use for: raw shell stdout, raw fetched HTML, "
        "or speculation without sources. The platform sets source_kind "
        "automatically based on your role; do not attempt to override."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The observation content. ≤ 4000 characters preferred.",
            },
            "source_url": {
                "type": ["string", "null"],
                "description": "URL where you found this; included as evidence",
            },
            "source_task_id": {
                "type": ["string", "null"],
                "description": "Task that produced this observation; for lineage",
            },
        },
        "required": ["text"],
    },
)
async def observe_tool(args: dict[str, Any]) -> dict[str, Any]:
    ctx, err = _ctx_or_error()
    if err:
        return err

    text = args.get("text", "").strip()
    if not text:
        return _err("observe text is empty")
    if len(text) > 8000:
        return _err(f"observe text too long ({len(text)} chars; max 8000)")

    # Append source markers if provided.  These show up in TCMM as part
    # of the observation body and improve recall context.
    suffix_parts = []
    if args.get("source_url"):
        suffix_parts.append(f"[source: {args['source_url']}]")
    if args.get("source_task_id"):
        suffix_parts.append(f"[task: {args['source_task_id']}]")
    if suffix_parts:
        text = f"{text}\n\n{' '.join(suffix_parts)}"

    try:
        persisted = await observe_agent_output(
            conversation_id=ctx.conversation_id,
            user_id=ctx.user_id,
            agent_id=ctx.agent_id,
            text=text,
            source="agent",
        )
        if not persisted:
            # [LOOP_FIX_2026_05_27]  observe used to silently return None
            # even when TCMM was down → LLM saw a "success" tool_result
            # and the next turn saw outputs[] empty → looped forever
            # re-checking task state.  Now we surface the failure so the
            # LLM stops calling observe() and either continues with
            # in-context reasoning or raises a blocker via add_comment.
            return _err(
                "TOOL UNAVAILABLE: TCMM did not persist this observation "
                "(memory service offline OR returned 4xx/5xx). "
                "Do NOT retry observe() — the next call will fail the "
                "same way and burn a turn. Instead: continue your "
                "reasoning IN-CONTEXT for this turn, then either "
                "(a) submit_for_review with the partial deliverable + a "
                "blocker_raised comment explaining memory was offline, OR "
                "(b) if your task does not require persistent claims, "
                "complete it using attach_output / submit_for_review "
                "directly. Multiple observe() retries when TCMM is down "
                "is the #1 cause of non-converging agent loops."
            )
        return _ok({
            "observed": True,
            "agent_id": ctx.agent_id,
            "text_length": len(text),
            "note": (
                "Platform stamped extracted_by=agent:" + ctx.agent_id +
                " and source_kind per role."
            ),
        })
    except Exception as e:
        logger.warning(f"[memory_mcp] observe failed: {e}")
        return _err(f"observe failed: {e}")


# ── read_constitution ──────────────────────────────────────────────────


@tool(
    name="read_constitution",
    description=(
        "Return the ORGANIZATION'S GOVERNANCE CONSTITUTION — the org's "
        "objectives (with weights), constraints (with rules), and metrics. "
        "This is internal policy/governance data, takes NO arguments, and "
        "does NOT read files of any kind. Do NOT use it to read a "
        "spreadsheet, PDF, document, or any file the user mentions — for "
        "those use read_xlsx (Excel), read_pdf (PDF), or read_file (text). "
        "Use read_constitution ONLY to check whether a proposed Task "
        "violates a constraint, or to compute objective_alignment when "
        "ranking proposals."
    ),
    input_schema={
        "type": "object",
        "properties": {},
    },
)
async def read_constitution_tool(args: dict[str, Any]) -> dict[str, Any]:
    ctx, err = _ctx_or_error()
    if err:
        return err

    from ..config import CONSTITUTION_PATH
    from ..constitution.loader import load_constitution, ConstitutionError

    try:
        if not CONSTITUTION_PATH.exists():
            return _err(f"CONSTITUTION.md not found at {CONSTITUTION_PATH}")
        result = load_constitution(CONSTITUTION_PATH)
        return _ok(result)
    except ConstitutionError as e:
        return _err(f"constitution parse error: {e}")
    except Exception as e:
        logger.exception(f"[memory_mcp] read_constitution failed: {e}")
        return _err(f"read_constitution failed: {e}")


# ── Server factory ──────────────────────────────────────────────────────


# `recall_tool` was retired 2026-05-25 (two-paths-to-memory confusion:
# the auto-render already carried the persona's memory, so an extra
# `recall` tool just invited redundant calls when the answer was already
# in the rendered context).
#
# [CHANNEL_2026_05_29 / P4] RE-ENABLED. Channels make the tool's scope
# DISTINCT from the auto-render: the always-on render carries the
# persona's default channels (+ the digest), while `recall` is for
# EXPLICIT, deeper, cross-channel pulls (e.g. "search team_knowledge for
# X") beyond what's already in context. The tool passes agent_id, so
# results are channel-scoped to the persona by default (spec §5.2/§5.3).
# Models should call it only when they need memory NOT already rendered.
_ALL_TOOLS = [recall_tool, observe_tool, read_constitution_tool]


def build_memory_mcp_server():
    if not _SDK_AVAILABLE:
        return None
    return create_sdk_mcp_server(
        name="veilguard_memory",
        version="0.1.0",
        tools=_ALL_TOOLS,
    )


__all__ = [
    "build_memory_mcp_server",
    "_ALL_TOOLS",
]
