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

import asyncio as _asyncio
import logging
import time
import uuid
from abc import ABC
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

from pii import SessionId, get_redactor, RedactionUnavailable
from llm import (
    AnthropicAdapter,
    render_structured,
    ingest_user,
    ingest_assistant,
)
from . import events
from .persona import PersonaSpec

logger = logging.getLogger("veilguard.agent.base")

# [INGEST_FIRE_AND_FORGET_2026_05_29]  Strong refs to in-flight
# fire-and-forget TCMM writes so the event loop doesn't GC them
# mid-await (asyncio only holds weak refs to tasks).
_BACKGROUND_TASKS: set = set()

# [PREPARE_SESSION_ONCE_2026_05_29]  (conversation_id, agent_id) pairs
# whose idempotent TCMM pins have already been fired this process
# lifetime.  Bounded growth is fine — one tuple per active conversation.
_PREPARED_SESSIONS: set = set()

# [RENDER_DISPATCH_CACHE_2026_05_29]  THE per-turn latency fix.
#
# render_structured(task_query=raw_user_msg) is called on EVERY inner
# agent-loop turn, and raw_user_msg (the per-turn rebuilt LOOP_CONTEXT
# message) changes each turn → recall returns a different ~30k-token
# memory blob → its bytes change → the PII-redaction cache misses → a
# full ~10-30s spaCy NER pass runs EVERY turn.  A single IC dispatch does
# 6+ inner turns → minutes of pure redaction.  (The pii-proxy never hits
# this: it redacts a small per-message delta, not a re-rendered blob.)
#
# Fix: cache the rendered blocks per (conversation, user, agent) for a
# short window so the inner turns of ONE dispatch reuse a single rendered
# (and therefore once-redacted, via the downstream analyze LRU) memory
# prefix.  Result: turn 1 of a dispatch pays render+redact once; turns
# 2..N are byte-identical → render skipped + redaction LRU-hits → free.
# The conversation tail is still redacted every turn, but each message is
# LRU-cached so only the NEW message is analyzed (the proxy's delta model).
# TTL bounds staleness for long-lived (Director) conversations.
_RENDER_CACHE: dict = {}
_RENDER_CACHE_TTL_S: float = 90.0


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
        """Return the Veilguard system preamble for this agent.

        Tool-agnostic: the preamble bytes are identical across every
        persona / agent in the system, so they share a single TCMM
        cache slot.  Tool schemas are pinned separately by
        `prepare_session()` via /pin/tool_definitions (one immutable
        block per tool, dedup by fingerprint) and arrive at Anthropic
        natively via the request `tools` field.

        Subclasses rarely need to override — the Veilguard frame is
        meant to be shared.
        """
        from .preamble import render_preamble
        return render_preamble()

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
            # Preamble is now tool-agnostic — one stable cache slot for
            # the whole Veilguard frame.  Tools are pinned separately
            # below via /pin/tool_definitions so a tool-schema tweak
            # invalidates exactly one block, not the whole preamble.
            await pin_system_prompt(
                ctx.conversation_id, ctx.user_id,
                preamble, kind="veilguard_preamble",
            )
        if self.persona.system_prompt:
            await pin_system_prompt(
                ctx.conversation_id, ctx.user_id,
                self.persona.system_prompt, kind="persona_prompt",
            )

        # Pin tool definitions as IMMUTABLE blocks — one per schema,
        # priority_class="SYSTEM", dedup by fingerprint.  Includes the
        # shadow tool (tcmm_record_turn) because prepare_tools runs the
        # injection.  Pinning them here means TCMM's /render returns
        # them in the cached prefix instead of us shoving the JSON into
        # the preamble (which would re-fingerprint the whole preamble
        # on every tool change).  Anthropic ALSO gets these natively in
        # the request `tools` field via prepare_tools later — the pin
        # is for cache stability + memory hygiene, not LLM visibility.
        try:
            tools_for_pin = self.prepare_tools(self.tools()) or []
            if tools_for_pin:
                await pin_tool_definitions(
                    ctx.conversation_id, ctx.user_id, tools_for_pin,
                )
        except Exception as e:
            logger.warning(
                f"[agent] pin_tool_definitions failed (continuing): {e}"
            )

    def prepare_tools(self, tools: list[dict]) -> list[dict]:
        """Hook fired BEFORE the adapter sees the tools list.

        2026-05-25: shadow-tool injection moved to base.  Every Agent
        subclass (ChatAgent, DirectorAgent, ICAgent, CriticClaimAgent,
        etc.) now gets ``tcmm_record_turn`` baked into its tool list
        so the model can emit turn-classification metadata as a
        structured tool_use block.  The preamble template references
        this tool by name in multiple places (preamble.py:103, 166,
        181, 328, 355) so any persona running through the unified
        pipeline NEEDS it in the surface — otherwise the LLM emits
        the tool_use anyway (per the preamble's instruction) and the
        runtime's tool_dispatcher rejects it as unknown.

        Subclasses can override to add MORE tools (e.g. a future
        BuilderAgent might inject sandboxing wrappers).  But removing
        the shadow tool here breaks the preamble's contract.
        """
        from .shadow_tool import inject_into_tools
        return inject_into_tools(tools)

    def intercept_response(
        self, content_blocks: list[dict], stop_reason: str,
    ) -> tuple[list[dict], dict, str]:
        """Hook fired AFTER the adapter returns, BEFORE rehydrate.

        2026-05-25: shadow-tool stripping moved to base.  Strips any
        ``tcmm_record_turn`` tool_use block from the response so:
          - The runtime's external tool dispatch loop doesn't see it
            (it's a meta-signal, not a real tool to invoke)
          - The input dict gets captured as ``flag_obj`` and is
            handed to the next ``ingest_assistant`` call so TCMM can
            classify the assistant turn.
          - If the shadow tool was the ONLY tool_use, downgrade
            stop_reason to ``end_turn`` so the caller doesn't think
            there's external dispatch work pending.

        Returns: (cleaned_blocks, flag_obj, new_stop_reason)
        """
        from .shadow_tool import intercept_response as _intercept
        return _intercept(content_blocks, stop_reason)

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

        # [LATENCY_TRACE_2026_05_29] per-step wall clock so we can find
        # the slow stage.  No-op when LOG_LEVEL > INFO.
        import time as _t
        _t_run_start = _t.time()
        _t_marks = {"start": _t_run_start}

        # STEP 0 — Ingest raw user message into TCMM memory.
        # [INGEST_DEFERRED_TO_END_2026_05_29]  This is a memory WRITE that
        # only feeds FUTURE turns' recall — it must not be on the
        # critical path.  An earlier attempt fired it as a background
        # task at turn START, but that made EVERY turn's ingest contend
        # with the NEXT turn's TCMM render (TCMM serializes per session),
        # turning a consistent ~4s warm turn into a 2-10s spiky one.
        # Instead we DEFER the ingest to AFTER the response is produced
        # (see end of run_turn) so it runs in the gap between turns, not
        # against the render.  Nothing to do here now.
        _t_marks["S0_ingest_user"] = _t.time()

        # STEP 0.5 — Subclass hook (default no-op).  ChatAgent uses
        # this to pin the Veilguard preamble + client_system + tools to
        # TCMM so the rendered prefix is cacheable.
        #
        # [PREPARE_SESSION_ONCE_2026_05_29]  The pins are IDEMPOTENT and
        # the pinned content (preamble, persona, tool schemas) is static
        # per (conversation, agent) — re-pinning every turn was pure
        # waste, and once ingest went fire-and-forget the per-turn pin
        # calls became the new bottleneck (7-8.8s — they serialize
        # behind other TCMM requests).  Pin ONCE per (conv, agent) and
        # fire it on a background task so it never blocks the turn.  The
        # essentials (magic prefix + persona) are also injected inline
        # below, so a render that races ahead of the pin still works.
        if self.include_memory():
            _pin_key = (ctx.conversation_id, self.agent_id)
            if _pin_key not in _PREPARED_SESSIONS:
                _PREPARED_SESSIONS.add(_pin_key)
                async def _do_prepare():
                    try:
                        await self.prepare_session(ctx)
                    except Exception as e:
                        _PREPARED_SESSIONS.discard(_pin_key)
                        logger.warning(
                            f"[agent] prepare_session hook failed: {e}"
                        )
                _pt = _asyncio.create_task(_do_prepare())
                _BACKGROUND_TASKS.add(_pt)
                _pt.add_done_callback(_BACKGROUND_TASKS.discard)
        _t_marks["S0_5_prepare_session"] = _t.time()

        # STEP 1 — TCMM render.  Skips when include_memory() is False
        # (e.g. side-channel title-gen calls).
        #
        # [F15_TCMM_DOWN_LOUD_FAIL_2026_05_27]  When TCMM is down the
        # render fails open with `raw_blocks=[]`.  That used to land
        # us with NO Claude-Code attribution magic prefix in the
        # system blocks → Anthropic Sonnet/Haiku-4.5+ OAuth-bearer
        # gate returns a generic 429 ("rate_limit_error: Error") that
        # LibreChat renders as "terminated" — confusing and obscures
        # the real root cause.  Now we synthesise the magic prefix
        # inline so the call survives; we also yield a `warning` event
        # so the client knows memory is degraded.  The prefix is the
        # SAME byte-exact string TCMM's renderer would produce.
        _MAGIC_PREFIX = (
            "You are a Claude agent, "
            "built on Anthropic's Claude Agent SDK."
        )
        _memory_degraded = False
        if self.include_memory():
            try:
                # [RENDER_DISPATCH_CACHE_2026_05_29] Reuse one rendered
                # memory prefix across the inner turns of a dispatch so we
                # render + redact the ~30k-token blob ONCE, not every turn.
                _rk = (ctx.conversation_id, ctx.user_id, self.persona.agent_id)
                _rhit = _RENDER_CACHE.get(_rk)
                if _rhit is not None and _t.time() < _rhit[1]:
                    raw_blocks = list(_rhit[0])
                    logger.info(
                        "[agent] render cache HIT for %s (skipped TCMM render "
                        "+ re-redact of memory prefix)", _rk[0]
                    )
                else:
                    rendered = await render_structured(
                        conv_id=ctx.conversation_id,
                        user_id=ctx.user_id,
                        task_query=raw_user_msg,
                        # [PII_SID_CONTRACT_2026_05_30]  Tell the render layer
                        # the EXACT sid we'll rehydrate with, so render-time
                        # redaction (when enabled) mints matching REF tokens.
                        pii_sid=sid.canonical(),
                    )
                    raw_blocks = list(rendered.blocks)
                    _RENDER_CACHE[_rk] = (
                        list(raw_blocks), _t.time() + _RENDER_CACHE_TTL_S
                    )
            except RuntimeError as e:
                logger.warning(
                    f"[agent] TCMM render failed (degrading + injecting "
                    f"magic prefix inline): {e}"
                )
                raw_blocks = []
                _memory_degraded = True
        else:
            raw_blocks = []

        # [CACHE_CONTROL_OWNERSHIP_2026_05_29]  Does TCMM's renderer
        # already own the cache_control layout?  When TCMM is UP it emits
        # blocks with cache_control markers (ttl='1h') in a deliberate,
        # Anthropic-valid order.  My persona-tail + tools-tail + inline-
        # preamble markers below were ONLY ever a workaround for the
        # DEGRADED case where TCMM emits zero markers.  Adding my 5m
        # (`{"type":"ephemeral"}`) markers on top of TCMM's 1h markers
        # produces a mixed-ttl ordering that Anthropic rejects with
        # 400 "a ttl='1h' cache_control block must not come after a
        # ttl='5m' cache_control block" → the Director turn errors out
        # with an empty reply and creates no tasks.  Caught by the UAT
        # the moment TCMM came up (it's a prod bug — TCMM is always up
        # in prod).  Rule: if TCMM provided ANY cache_control, trust it
        # entirely and add NONE of my own.
        _tcmm_owns_cache = any(
            isinstance(b, dict) and b.get("cache_control")
            for b in raw_blocks
        )

        # If the first block isn't the magic prefix, prepend it.  This
        # makes the call survive Anthropic's OAuth-bearer attribution
        # gate even when TCMM is offline.  Without this, a TCMM crash
        # cascades into LibreChat's "terminated" UX.
        if not raw_blocks or (raw_blocks[0].get("text", "").strip() != _MAGIC_PREFIX):
            raw_blocks = [{"type": "text", "text": _MAGIC_PREFIX}] + raw_blocks
        _t_marks["S1_render"] = _t.time()

        # [HAIKU_CACHE_FLOOR_INLINE_PREAMBLE_2026_05_28]  When the
        # TCMM-rendered prefix is too short to clear Haiku 4.5's
        # ~6000-token cache floor, inline the Veilguard preamble here
        # so the cached prefix gets fat enough to actually cache.
        #
        # Why "too short" is the right trigger, not "always":
        #   * On warm conversations TCMM-rendered memory blocks already
        #     push us well over 6k; inlining the preamble would just
        #     duplicate bytes already pinned-and-rendered by TCMM.
        #   * On cold conversations (no memory yet) OR when TCMM is
        #     degraded, the preamble is the only static bulk we have
        #     to clear the floor with.
        # Threshold tuned to half the floor — gives headroom for the
        # persona + tools to push us over.
        try:
            _existing_chars = sum(
                len(b.get("text", "")) for b in raw_blocks
                if isinstance(b, dict) and b.get("type") == "text"
            )
            # ~4 chars/token rule of thumb; 3000 tokens ≈ 12000 chars.
            # Skip entirely when TCMM owns the cache layout (warm prod
            # path) — TCMM's blocks already clear the floor and carry
            # their own markers.
            if (not _tcmm_owns_cache) and _existing_chars < 12000:
                from .preamble import _VEILGUARD_PREAMBLE_TEMPLATE
                # Insert RIGHT AFTER the magic prefix (block 0).  Order
                # matters for cache stability: prefix → preamble (static)
                # → memory (volatile) → persona (semi-static).  Personas
                # already cache_control-tail the prefix, and the preamble
                # being static means the full {prefix+preamble} segment
                # is stable byte-for-byte across all callers.
                _pre_blk = {
                    "type": "text",
                    "text": _VEILGUARD_PREAMBLE_TEMPLATE,
                }
                # Idempotency: don't double-insert if it's already there
                # (shouldn't happen since TCMM renders pins to non-magic
                # blocks, but cheap to be safe).
                _already_in = any(
                    isinstance(b, dict)
                    and b.get("text", "").startswith(_VEILGUARD_PREAMBLE_TEMPLATE[:80])
                    for b in raw_blocks
                )
                if not _already_in:
                    raw_blocks = [raw_blocks[0], _pre_blk] + raw_blocks[1:]
                    logger.info(
                        f"[agent] inlined Veilguard preamble "
                        f"({len(_VEILGUARD_PREAMBLE_TEMPLATE)} chars) — "
                        f"existing prefix {_existing_chars}c < 12000c "
                        f"floor; ensures Haiku 4.5 ~6k-token cache floor "
                        f"is cleared"
                    )
        except Exception as e:
            # Never fail the turn because of cache optimisation.
            logger.warning(f"[agent] preamble inline failed (continuing): {e}")

        # Append persona system_prompt as a SEPARATE final block.
        # [PROMPT_CACHE_2026_05_28] We DO mark the persona block with
        # cache_control: ephemeral.  Original concern was "persona
        # changes shouldn't invalidate the cached memory prefix" — but
        # TCMM currently emits ZERO cache_control markers on its own
        # blocks (cache_create=0 cache_read=0 verified live), so the
        # whole prefix was being processed from scratch on every turn.
        # Marking the persona's tail makes the FULL preceding prefix
        # (magic prefix + TCMM memory blocks + persona) a single cache
        # breakpoint.  Personas change daily at most, so a single
        # invalidation per persona edit is cheap; the win is 4000+
        # tokens of cache_read per warm turn → ~3-5× speedup on Haiku.
        #
        # [HAIKU_4_5_CACHE_FLOOR_2026_05_28]  Empirically verified the
        # Anthropic OAuth-bearer cache floor for `claude-haiku-4-5` is
        # ~6000 tokens, not the documented 2048.  pii_audit shows the
        # smallest successful cache_read on Haiku is 6087 tokens
        # (sub-task-e4-ibx-65321b7c); all calls below ~7300 tokens
        # input return cache_create=0 cache_read=0 even with valid
        # cache_control markers.  So COLD Director conversations
        # (magic prefix ~16 tok + persona ~1880 tok + tools ~2300 tok ≈
        # 4200 tok total) fall under the floor and get no cache.  WARM
        # conversations with TCMM-rendered memory blocks ride well over
        # 6000 and DO cache.  Future fix: pin a richer preamble (the
        # Constitution + recent_decisions) so even cold conversations
        # clear the floor; for now we accept the cold-start hit.
        if self.persona.system_prompt:
            _persona_blk = {
                "type": "text",
                "text": self.persona.system_prompt,
            }
            # Only add my own cache_control when TCMM ISN'T managing the
            # cache layout (degraded path).  When TCMM owns it, appending
            # a 5m marker after TCMM's 1h markers triggers the mixed-ttl
            # 400 error — see [CACHE_CONTROL_OWNERSHIP_2026_05_29].
            if not _tcmm_owns_cache:
                _persona_blk["cache_control"] = {"type": "ephemeral"}
            raw_blocks.append(_persona_blk)

        # STEP 2 — PII redaction.  Block-level so cache_control markers
        # are preserved.  Same SessionId means same tokens whether this
        # turn is being run by ChatAgent in pii-proxy or DirectorAgent
        # in agent-runtime — they see byte-identical prefixes.
        #
        # [REDACT_OFF_LOOP_2026_05_29]  redact_blocks is SYNCHRONOUS
        # (Presidio + Lance session-store writes).  Running it directly
        # on the event loop blocked /health and every other concurrent
        # turn for the redaction's duration (cold = seconds), which
        # showed up as flapping health checks + "empty reply from
        # server" under load.  Offload to a worker thread so the loop
        # stays responsive.  The redactor + session store are
        # thread-safe (write lock + process memo).
        redactor = get_redactor()
        def _do_redact():
            return (
                # [PII_AID_CACHE_2026_05_30] system = TCMM live-memory blocks
                # → redact keyed by live-memory-block aid (_vg_aid), not by
                # rendered/cache-marked bytes.  messages = latest prompt etc.
                # → redacted every turn (no aid).
                redactor.redact_memory_blocks(raw_blocks, sid),
                redactor.redact_messages(messages, sid),
            )
        # [FAIL_CLOSED_2026_05_29]  The shared redactor refuses (raises
        # RedactionUnavailable) on a real Presidio failure rather than
        # shipping raw PII to Anthropic.  Catch it HERE — this call is
        # outside the STEP 3 try/except below — and end the turn with an
        # error event instead of crashing run_turn.
        try:
            redacted_blocks, redacted_messages = await _asyncio.to_thread(_do_redact)
        except RedactionUnavailable as _re:
            logger.error(f"[agent] PII redaction unavailable — refusing turn: {_re}")
            yield events.error(
                code="redaction_unavailable",
                message=f"PII redaction unavailable: {_re}",
            )
            yield events.run_end(
                run_id=run_id, ended_at=time.time(), stop_reason="error",
            )
            return
        _t_marks["S2_pii_redact"] = _t.time()
        # [CACHE_DIAG_2026_05_28] one-line diagnostic so we can confirm
        # cache_control is actually leaving Agent.run_turn for Anthropic.
        try:
            _n_blocks = len(redacted_blocks or [])
            _n_with_cc = sum(
                1 for b in (redacted_blocks or [])
                if isinstance(b, dict) and b.get("cache_control")
            )
            logger.info(
                f"[agent] system_blocks → adapter: count={_n_blocks} "
                f"with_cache_control={_n_with_cc} agent={self.agent_id}"
            )
        except Exception:
            pass

        # STEP 3 — Adapter.  Sync .generate() runs in a thread inside
        # the adapter wrapper so our event loop stays free.
        # prepare_tools() hook lets subclasses inject the shadow tool.
        try:
            final_tools = self.prepare_tools(self.tools()) or None
            # [PROMPT_CACHE_TOOLS_2026_05_28] Add cache_control to the
            # LAST tool so Anthropic caches the whole tools array (and
            # the system prefix that comes before it).  Director has
            # ~17 tools = ~3000 tokens — adding this to the ~1000-token
            # system prefix puts us safely over Haiku's 2048-token
            # cache minimum.  Without this marker the cache silently
            # no-ops (verified live: cache_create=0 cache_read=0).
            # Only when TCMM isn't managing cache layout — a 5m tools
            # marker combined with TCMM's 1h system markers also trips
            # the mixed-ttl 400 (tools are processed before system).
            # [CACHE_CONTROL_OWNERSHIP_2026_05_29]
            if (not _tcmm_owns_cache) and final_tools and isinstance(final_tools, list) and len(final_tools) > 0:
                last_tool = dict(final_tools[-1])  # copy so we don't mutate caller's list
                last_tool["cache_control"] = {"type": "ephemeral"}
                final_tools = final_tools[:-1] + [last_tool]
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
            _t_marks["S3_adapter_pre"] = _t.time()
            result = await adapter.generate(
                redacted_user_msg, label=f"agent:{self.agent_id}",
            )
            _t_marks["S3_adapter_done"] = _t.time()
            try:
                _spans = []
                _prev = _t_marks["start"]
                for _k, _v in _t_marks.items():
                    if _k == "start":
                        continue
                    _spans.append(f"{_k}={(_v-_prev)*1000:.0f}ms")
                    _prev = _v
                _total = (_t.time() - _t_marks["start"]) * 1000
                logger.info(
                    f"[agent.timing] agent={self.agent_id} total={_total:.0f}ms "
                    + " ".join(_spans)
                )
            except Exception:
                pass
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

        # ── Audit rows (TO_LLM + FROM_LLM) ────────────────────────────
        # Writers (chat_agent_handler.py for LibreChat path; others for
        # multi-agent paths) drain these and persist to pii_audit /
        # ledger / log file as appropriate.  We emit BOTH after the
        # adapter returns so each row carries the final usage snapshot
        # — matches the legacy `_sso_audit_record` semantics in
        # main.py (lines 3498/3526).
        try:
            _u = result.usage or {}
            _in_total = (
                (_u.get("input_tokens") or 0)
                + (_u.get("cache_creation_input_tokens") or 0)
                + (_u.get("cache_read_input_tokens") or 0)
            )
            _cache_create = _u.get("cache_creation_input_tokens") or 0
            _cache_read = _u.get("cache_read_input_tokens") or 0
            _out_tokens = _u.get("output_tokens") or 0

            # TO_LLM content: the FULL envelope sent to api.anthropic.com,
            # so the audit row honors audit_db.py's "full, untruncated"
            # contract.  Three sections, JSON-encoded where useful so the
            # row round-trips back to a request body for diffing/replay:
            #   [SYSTEM]   — concatenated text from system_blocks
            #   [TOOLS]    — full tool schemas (after prepare_tools hook;
            #                includes shadow tools like tcmm_record_turn)
            #   [MESSAGES] — full redacted_messages array (not just the
            #                last user msg — assistant + tool_result
            #                turns matter for audit / replay)
            import json as _json
            _sys_text_parts: list[str] = []
            for _blk in (redacted_blocks or []):
                if isinstance(_blk, dict) and _blk.get("type") == "text":
                    _t = _blk.get("text")
                    if isinstance(_t, str) and _t:
                        _sys_text_parts.append(_t)
            _sys_text = "\n\n".join(_sys_text_parts)
            _tools_json = _json.dumps(final_tools or [], ensure_ascii=False)
            _msgs_json = _json.dumps(
                redacted_messages or [], ensure_ascii=False,
            )
            _sections: list[str] = []
            if _sys_text:
                _sections.append(f"[SYSTEM]\n{_sys_text}")
            _sections.append(f"[TOOLS]\n{_tools_json}")
            _sections.append(f"[MESSAGES]\n{_msgs_json}")
            _to_llm_full = "\n\n".join(_sections)

            yield events.audit(
                direction="TO_LLM",
                content=_to_llm_full,
                model=result.model,
                tokens_input_total=_in_total,
                cache_create=_cache_create,
                cache_read=_cache_read,
            )
            yield events.audit(
                direction="FROM_LLM",
                content=raw_text or "",
                model=result.model,
                tokens_input_total=_in_total,
                tokens_output=_out_tokens,
                cache_create=_cache_create,
                cache_read=_cache_read,
            )
        except Exception as _e:
            logger.warning(f"[agent] audit emit failed (continuing): {_e}")

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

        # STEP 5 — Ingest user message + assistant reply into TCMM memory.
        # [INGEST_DEFERRED_TO_END_2026_05_29]  Both are memory WRITES that
        # only feed FUTURE turns — fire them as ONE ordered background
        # task (user then assistant) AFTER the response is done, so they
        # run in the gap before the next turn instead of blocking this
        # turn or contending with the next turn's render.  Removes ~3s
        # (1.5s ingest_user + 1.5s ingest_assistant) from the critical
        # path.  Best-effort; errors are swallowed inside the ingest fns.
        if self.include_memory() and (raw_user_msg or raw_text):
            _cid, _uid, _mdl = ctx.conversation_id, ctx.user_id, result.model
            _fobj = flag_obj or None
            _um, _at = raw_user_msg, raw_text
            async def _deferred_ingest():
                if _um:
                    await ingest_user(_cid, _uid, _um)
                if _at:
                    await ingest_assistant(_cid, _uid, _at, model=_mdl, flag_obj=_fobj)
            _bg = _asyncio.create_task(_deferred_ingest())
            _BACKGROUND_TASKS.add(_bg)
            _bg.add_done_callback(_BACKGROUND_TASKS.discard)

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

        # Layer client (daemon-routed) tool schemas on top — see
        # agent/client_tools.py for the shared source of truth.  These
        # tools are dispatched via tool_dispatcher Path 2 (sub-agents
        # HTTP endpoint + WS bridge).  In-process schemas win on name
        # collision.
        try:
            from .client_tools import CLIENT_TOOL_SCHEMAS
            for name, schema in CLIENT_TOOL_SCHEMAS.items():
                by_name.setdefault(name, schema)
        except Exception as e:
            logger.debug(f"[agent] client tool schemas not loaded: {e}")

        out: list[dict] = []
        seen: set[str] = set()
        for t in self.persona.tools:
            if t in by_name and t not in seen:
                out.append(by_name[t])
                seen.add(t)
        return out


__all__ = ["Agent", "TurnContext", "_last_user_text"]
