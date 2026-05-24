"""Agent base class — the ONE LLM pipeline for all of Veilguard.

Every Veilguard agent inherits this.  LibreChat's user-facing chat is
just one subclass (ChatAgent); Director, ICs, Critics, Consultants
are others.  The pipeline shape never changes.

The 5-step pipeline (`Agent.run_turn`):

  STEP 0  Ingest the user's raw message into TCMM memory.
          TCMM stores unredacted by design (trust-zone), so the LLM
          can recall full context in the future without leaking PII
          via memory.

  STEP 1  TCMM render — pull memory + preamble, get system blocks
          with cache_control markers placed by TCMM's renderer.

  STEP 2  PII redaction — block-level redaction with a Lance-backed
          session-keyed mapping.  Byte-stable across turns
          (`pii.PIIRedactor` invariants A-E proven in pii/tests/).
          This is the boundary at which raw PII stops leaving the
          trust zone.

  STEP 3  AnthropicAdapter.generate — sends redacted bytes + redacted
          messages to api.anthropic.com (OAuth bearer or x-api-key
          depending on CLAUDE_SSO env).

  STEP 4  Rehydrate REF tokens in the response text and in any
          tool_use input dicts before yielding to the caller.

  STEP 5  Ingest the rehydrated assistant text back into TCMM memory.

Subclasses customize THREE knobs (not the pipeline):

  .tools()           → Anthropic tool schemas this agent advertises
  .model()           → which Claude model variant to use
  .include_memory()  → False for side-channel calls (title-gen, etc.)
                       that don't need TCMM render

Everything else — auth, redaction, caching, audit, error envelopes —
is identical across agents.
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

from pii import SessionId, get_redactor
from llm import (
    AnthropicAdapter,
    render_structured,
    ingest_user,
    ingest_assistant,
)
from . import events
from .persona import PersonaSpec

logger = logging.getLogger("veilguard.agent.base")


# ── Per-turn context ────────────────────────────────────────────────────


@dataclass(frozen=True)
class TurnContext:
    """All identifiers that scope this turn.

    Threaded through every step so the redactor, TCMM memory, and the
    LLM adapter all key off the same conv_id + tenant_id.
    """
    conversation_id: str
    user_id: str
    tenant_id: str
    parent_cid: Optional[str] = None
    is_background: bool = False


# ── Helpers ─────────────────────────────────────────────────────────────


def _last_user_text(messages: list[dict]) -> str:
    """Pull the most recent role=user message body as a plain string."""
    for m in reversed(messages):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    t = blk.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            return "\n".join(parts)
    return ""


# ── Agent base ──────────────────────────────────────────────────────────


class Agent(ABC):
    """Base class for every LLM-driven actor in Veilguard.

    Lifetime: instantiate per turn (cheap).  Per-conversation state
    lives in TCMM memory and the PII session store, not on the Agent
    instance.

    Dependency injection: `adapter_cls` lets demos / tests swap in a
    ScriptedAdapter that returns canned content while keeping the
    rest of the pipeline (TCMM render + PII redact + rehydrate)
    real.  Defaults to AnthropicAdapter for production.
    """

    def __init__(
        self,
        persona: PersonaSpec,
        *,
        adapter_cls: Optional[type] = None,
    ):
        self.persona = persona
        self.agent_id = persona.agent_id
        # Lazy import to avoid forcing TCMM_ROOT existence at class load.
        if adapter_cls is None:
            adapter_cls = AnthropicAdapter
        self._adapter_cls = adapter_cls

    # ── Customization knobs (override in subclasses) ──────────────────

    def tools(self) -> list[dict]:
        """Anthropic-shape tool schemas this agent advertises.

        Default reads from persona.tools and looks up schemas in the
        in-process MCP registries.  Subclasses can override (e.g.
        ChatAgent gets its tools from LibreChat's request body).
        """
        return self._default_tool_schemas()

    def model(self) -> str:
        return self.persona.model_for("reactive")

    def include_memory(self) -> bool:
        """Skip TCMM render for side-channel calls (title-gen, etc.)."""
        return True

    def preamble(self) -> str:
        """Return the Veilguard system preamble for this agent, with
        this agent's tool schemas inlined into the AVAILABLE TOOLS
        section.

        Default: uses `self.tools()` (the Anthropic-shape schemas for
        the persona's tool allow-list).  Pinned to TCMM by the default
        `prepare_session()` so it lands in the immutable tier of the
        cached prefix.  All agents share the same template.

        Subclasses can override for special cases (ChatAgent's preamble
        uses LibreChat-supplied tools, etc.) — but the default works
        for every persona that defines tools in its markdown.
        """
        from .preamble import render_preamble
        return render_preamble(self.tools())

    async def prepare_session(self, ctx: "TurnContext") -> None:
        """Hook fired after STEP 0 (user ingest) and before STEP 1
        (TCMM render).

        Default: pin this agent's `preamble()` (Veilguard frame +
        tool schemas) to TCMM's immutable tier as `veilguard_preamble`,
        then pin the persona system_prompt (if any) as `persona_prompt`.
        TCMM's renderer pulls these into the cached prefix on every
        subsequent /render_structured call.

        Subclasses can override for special cases (ChatAgent also pins
        LibreChat's client_system text + the original tool definitions
        as a separate pin).  Most agents should let the default run.
        """
        from llm import pin_system_prompt, pin_tool_definitions

        preamble = self.preamble()
        if preamble:
            await pin_system_prompt(
                ctx.conversation_id, ctx.user_id,
                preamble, kind="veilguard_preamble",
            )
        if self.persona.system_prompt:
            await pin_system_prompt(
                ctx.conversation_id, ctx.user_id,
                self.persona.system_prompt, kind="persona_prompt",
            )
        tools = self.tools()
        if tools:
            await pin_tool_definitions(
                ctx.conversation_id, ctx.user_id, tools,
            )

    def prepare_tools(self, tools: list[dict]) -> list[dict]:
        """Hook fired BEFORE the adapter sees the tools list.  Default
        returns unchanged.

        ChatAgent overrides to inject the `tcmm_record_turn` shadow
        tool so the model emits turn metadata as a structured tool_use
        block.  Multi-agent personas (Director, IC, etc.) don't use the
        shadow tool today — their tool_use blocks all go through the
        in-process tool_dispatcher.
        """
        return tools

    def intercept_response(
        self, content_blocks: list[dict], stop_reason: str,
    ) -> tuple[list[dict], dict, str]:
        """Hook fired AFTER the adapter returns, BEFORE rehydrate +
        yield.  Default returns unchanged + empty flag_obj.

        Returns: (cleaned_blocks, flag_obj, new_stop_reason)

        ChatAgent overrides to strip the `tcmm_record_turn` tool_use
        block (capturing its input as flag_obj for the next
        ingest_assistant call) and to downgrade stop_reason to
        "end_turn" when the shadow tool was the only tool_use.
        """
        return content_blocks, {}, stop_reason or "end_turn"

    # ── The pipeline (subclasses do NOT override) ─────────────────────

    async def run_turn(
        self,
        messages: list[dict],
        ctx: TurnContext,
    ) -> AsyncIterator[dict]:
        """Run one turn end-to-end.  Yields typed events from `events.py`.

        Caller drives the tool-dispatch loop externally (this method
        runs ONE LLM call per invocation).  If the assistant emits
        tool_use blocks, the caller dispatches them, appends
        tool_result blocks to `messages`, then calls run_turn again.
        """
        run_id = uuid.uuid4().hex[:12]
        sid = SessionId(ctx.tenant_id, ctx.conversation_id)

        yield events.run_start(
            run_id=run_id,
            agent_id=self.agent_id,
            model=self.model(),
            backend="anthropic-adapter",
            started_at=time.time(),
        )

        raw_user_msg = _last_user_text(messages)

        # STEP 0 — Ingest raw user message into TCMM memory.
        # Fire and forget (best-effort; logs and swallows errors).
        if raw_user_msg and self.include_memory():
            await ingest_user(ctx.conversation_id, ctx.user_id, raw_user_msg)

        # STEP 0.5 — Subclass hook (default no-op).  ChatAgent uses
        # this to pin the Veilguard preamble + client_system + tools to
        # TCMM so the rendered prefix is cacheable.
        if self.include_memory():
            try:
                await self.prepare_session(ctx)
            except Exception as e:
                logger.warning(
                    f"[agent] prepare_session hook failed (continuing): {e}"
                )

        # STEP 1 — TCMM render.  Skips when include_memory() is False
        # (e.g. side-channel title-gen calls).
        if self.include_memory():
            try:
                rendered = await render_structured(
                    conv_id=ctx.conversation_id,
                    user_id=ctx.user_id,
                    task_query=raw_user_msg,
                )
                raw_blocks = list(rendered.blocks)
            except RuntimeError as e:
                logger.warning(f"[agent] TCMM render failed (degrading): {e}")
                raw_blocks = []
        else:
            raw_blocks = []

        # Append persona system_prompt as a SEPARATE final block (no
        # cache_control marker — persona changes shouldn't invalidate
        # the cached memory prefix).
        if self.persona.system_prompt:
            raw_blocks.append(
                {"type": "text", "text": self.persona.system_prompt}
            )

        # STEP 2 — PII redaction.  Block-level so cache_control markers
        # are preserved.  Same SessionId means same tokens whether this
        # turn is being run by ChatAgent in pii-proxy or DirectorAgent
        # in agent-runtime — they see byte-identical prefixes.
        redactor = get_redactor()
        redacted_blocks = redactor.redact_blocks(raw_blocks, sid)
        redacted_messages = redactor.redact_messages(messages, sid)

        # STEP 3 — Adapter.  Sync .generate() runs in a thread inside
        # the adapter wrapper so our event loop stays free.
        # prepare_tools() hook lets subclasses inject the shadow tool.
        try:
            final_tools = self.prepare_tools(self.tools()) or None
            adapter = self._adapter_cls(
                model=self.model(),
                system_blocks=redacted_blocks if redacted_blocks else None,
                system_prompt=(
                    None if redacted_blocks else self.persona.system_prompt
                ),
                tools=final_tools,
                agent_id=self.agent_id,
            )
            redacted_user_msg = redactor.redact_text(raw_user_msg, sid)
            result = await adapter.generate(
                redacted_user_msg, label=f"agent:{self.agent_id}",
            )
        except Exception as e:
            logger.exception(f"[agent] adapter call failed: {e}")
            yield events.error(
                code="adapter_error",
                message=f"{type(e).__name__}: {e}",
            )
            yield events.run_end(
                run_id=run_id, ended_at=time.time(), stop_reason="error",
            )
            return

        # STEP 3.5 — Subclass response intercept hook (default no-op).
        # ChatAgent uses this to strip the tcmm_record_turn shadow
        # tool_use block and capture its input as `flag_obj` for the
        # ingest_assistant call.  Runs BEFORE rehydrate so the shadow
        # tool's input dict (which is metadata, not PII) doesn't go
        # through the redactor.
        intercepted_blocks, flag_obj, intercepted_stop = self.intercept_response(
            result.content_blocks, result.stop_reason,
        )

        # STEP 4 — Rehydrate before yielding to caller.  Both response
        # text and any tool_use input args.
        raw_content = redactor.rehydrate_blocks(intercepted_blocks, sid)
        raw_text = redactor.rehydrate_text(result.text, sid)

        yield events.assistant(
            content=raw_content,
            usage=result.usage,
            stop_reason=intercepted_stop,
        )

        # Emit per-block helper events for the runtime's tool dispatch
        # loop and any UI streaming.
        for blk in raw_content:
            btype = blk.get("type")
            if btype == "text":
                t = blk.get("text", "")
                if t:
                    yield events.assistant_text(t)
            elif btype == "tool_use":
                yield events.tool_call(
                    name=blk.get("name", ""),
                    id=blk.get("id", ""),
                    input=blk.get("input", {}) or {},
                )

        yield events.final_result(result=raw_text, stop_reason=intercepted_stop)

        u = result.usage
        yield events.usage(
            tokens_input_new=u.get("input_tokens") or 0,
            tokens_output=u.get("output_tokens") or 0,
            cache_create=u.get("cache_creation_input_tokens") or 0,
            cache_read=u.get("cache_read_input_tokens") or 0,
            model=result.model,
            iterations=1,
        )

        # STEP 5 — Ingest raw assistant text back into TCMM memory.
        # Best-effort.  If the shadow-tool intercept captured a flag_obj,
        # pass it along — that's what makes block_class actually land in
        # the archive (instead of falling back to prose-JSON parsing).
        if raw_text and self.include_memory():
            await ingest_assistant(
                ctx.conversation_id, ctx.user_id, raw_text,
                model=result.model,
                flag_obj=flag_obj or None,
            )

        yield events.run_end(
            run_id=run_id, ended_at=time.time(), stop_reason=intercepted_stop,
        )

    # ── Default tool-schema lookup ────────────────────────────────────

    def _default_tool_schemas(self) -> list[dict]:
        """Look up persona.tools in the in-process MCP registries.

        Imports are deferred so the agent package doesn't pull in the
        agent-runtime tools dir at module load (avoids a circular
        dep during the PR-by-PR rollout).
        """
        try:
            # Both registries live under agent-runtime today.  When
            # PR #4 lands they'll move into this package.
            import sys
            from pathlib import Path
            ar = Path(__file__).resolve().parent.parent / "agent-runtime"
            if str(ar) not in sys.path:
                sys.path.insert(0, str(ar))
            from app.tools.ledger_mcp import _ALL_TOOLS as L
            from app.tools.memory_mcp import _ALL_TOOLS as M
        except Exception as e:
            logger.debug(f"[agent] tool registry import skipped: {e}")
            return []

        by_name: dict[str, dict] = {}
        for tool_obj in list(L) + list(M):
            if hasattr(tool_obj, "name") and hasattr(tool_obj, "input_schema"):
                by_name[tool_obj.name] = {
                    "name": tool_obj.name,
                    "description": tool_obj.description,
                    "input_schema": tool_obj.input_schema,
                }
            elif hasattr(tool_obj, "_mcp_tool_meta"):
                meta = tool_obj._mcp_tool_meta
                by_name[meta["name"]] = {
                    "name": meta["name"],
                    "description": meta["description"],
                    "input_schema": meta["input_schema"],
                }

        out: list[dict] = []
        seen: set[str] = set()
        for t in self.persona.tools:
            if t in by_name and t not in seen:
                out.append(by_name[t])
                seen.add(t)
        return out


__all__ = ["Agent", "TurnContext", "_last_user_text"]
