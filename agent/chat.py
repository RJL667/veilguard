"""ChatAgent — the LibreChat-facing agent.

LibreChat is the source of MOST of these knobs (not a persona file):
  - model     comes from the request body
  - tools     comes from the request body (LibreChat MCP wiring)
  - system    comes from `system` field of the request

We synthesize a minimal PersonaSpec at construction so the Agent base
machinery still works.

Side-channel detection: LibreChat fires synthetic single-shot calls
behind the scenes (title-gen, summarization).  These don't need TCMM
memory rendering — TCMM render injects 20-70KB of preamble + memory
which is wasteful for a 5-word title.  We detect them by string-prefix
match on the user message and set include_memory()=False for that turn.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import Agent, TurnContext, _last_user_text
from .persona import PersonaSpec

logger = logging.getLogger("veilguard.agent.chat")


# String-prefix matches LibreChat uses for its synthetic side-channel
# calls.  Copied from agent-proxy/app/main.py _LIBRECHAT_SIDE_CHANNEL_PREFIXES.
_SIDE_CHANNEL_PREFIXES = (
    "Provide a concise, 5-word-or-less title for the conversation",
    "Write a concise title for this conversation",
    "Please generate a title",
    "Please summarize the conversation",
    "Write a concise summary of the conversation",
)


def is_side_channel(messages: list[dict]) -> tuple[bool, str]:
    """Detect synthetic LibreChat title-gen / summary calls.

    Returns (is_side_channel, prefix_matched).  Empty string when no
    match.  The caller can log the matched prefix for telemetry.
    """
    last = _last_user_text(messages)
    if not last:
        return (False, "")
    head = last.lstrip()[:120]
    for prefix in _SIDE_CHANNEL_PREFIXES:
        if head.startswith(prefix):
            return (True, prefix[:30])
    return (False, "")


class ChatAgent(Agent):
    """User-facing chat agent — LibreChat hits this.

    LibreChat's request body supplies:
      - model        → self._model
      - tools        → self._client_tools (MCP schemas from LC config)
      - system text  → persona.system_prompt (we synthesize a Persona)

    The PII redactor + TCMM memory pipeline run identically to every
    other agent.  This subclass only customizes:
      - tools()           returns LibreChat-supplied schemas, not
                          our in-process ledger/memory ones
      - model()           reads from LibreChat's request
      - include_memory()  flips to False when side-channel detected
    """

    def __init__(
        self,
        *,
        model: str,
        client_tools: Optional[list[dict]] = None,
        system_prompt: str = "",
        side_channel: bool = False,
        persona_id: str = "chat",
        veilguard_preamble: str = "",
        client_system_text: str = "",
    ):
        persona = PersonaSpec(
            agent_id=persona_id,
            role="chat",
            manager_id=None,
            team_id=None,
            model=model,
            model_map={"reactive": model},
            tools=[],            # we override .tools() to use client_tools
            system_prompt=system_prompt,
            display_name="LibreChat user",
            source_path=None,
            schema_version=1,
        )
        super().__init__(persona)
        self._model = model
        self._client_tools = list(client_tools or [])
        self._side_channel = side_channel
        # Pre-rendered text the proxy supplies for pinning to TCMM.
        # We pin them in prepare_session() so they end up in TCMM's
        # immutable tier — keeps the cached prefix stable across turns.
        self._veilguard_preamble = veilguard_preamble
        self._client_system_text = client_system_text
        if side_channel:
            logger.info(
                "[chat] side-channel turn — skipping TCMM render + ingest"
            )

    def prepare_tools(self, tools: list[dict]) -> list[dict]:
        """Inject the tcmm_record_turn shadow tool so the model emits
        turn metadata (knowledge_class, used citations, emit_class) as
        a structured tool_use that the proxy intercepts before the user
        sees it.
        """
        from .shadow_tool import inject_into_tools
        return inject_into_tools(tools)

    def intercept_response(
        self, content_blocks: list[dict], stop_reason: str, sid: str = None,
    ) -> tuple[list[dict], dict, str]:
        """Strip the shadow tcmm_record_turn block before yielding to
        the caller.  Capture its input as flag_obj — the next
        ingest_assistant call writes it to TCMM as turn-classification
        metadata.

        If the only tool_use was the shadow tool, downgrade
        stop_reason to "end_turn" so LibreChat doesn't wait for a
        tool_result on a tool we consumed internally.
        """
        from .shadow_tool import intercept_response as _intercept
        return _intercept(content_blocks, stop_reason, sid)

    async def prepare_session(self, ctx) -> None:
        """ChatAgent-specific pinning on top of the base preamble.

        The base `Agent.prepare_session` already pins:
          - veilguard_preamble (via self.preamble() → render_preamble(tools))
          - persona_prompt (empty for ChatAgent — no persona file)
          - tool_definitions

        ChatAgent ALSO pins LibreChat's `system` field as a distinct
        `client_system` kind, so TCMM's renderer can lay them out in
        the right tier order (preamble → client_system → memory).
        """
        # Run the base preamble + tool pins
        await super().prepare_session(ctx)

        from llm import pin_system_prompt
        if self._client_system_text:
            await pin_system_prompt(
                ctx.conversation_id, ctx.user_id,
                self._client_system_text,
                kind="client_system",
            )

    @classmethod
    def from_libre_chat_request(
        cls,
        body: dict,
        *,
        persona_system_prompt: str = "",
        veilguard_preamble: str = "",
    ) -> "ChatAgent":
        """Construct a ChatAgent from a LibreChat-shaped request body.

        Convenience entry point that the agent-proxy HTTP handler will
        call.  Extracts model + tools + side-channel flag in one place.
        """
        model = body.get("model") or "claude-sonnet-4-5"
        # Strip "-sso" suffix LibreChat may append when using SSO routing.
        if isinstance(model, str) and model.endswith("-sso"):
            model = model[:-4]

        # LibreChat may pass `system` as a string or as a list of
        # text blocks; flatten either form.
        sys_text = persona_system_prompt
        sys_field = body.get("system")
        if isinstance(sys_field, str) and sys_field:
            sys_text = (sys_text + "\n\n" + sys_field) if sys_text else sys_field
        elif isinstance(sys_field, list):
            for blk in sys_field:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    t = blk.get("text", "")
                    sys_text = (sys_text + "\n\n" + t) if sys_text else t

        client_tools = body.get("tools") or []
        if not isinstance(client_tools, list):
            client_tools = []

        messages = body.get("messages") or []
        side_channel, prefix = is_side_channel(messages)
        if side_channel:
            logger.info(f"[chat] side-channel match: {prefix!r}")

        return cls(
            model=model,
            client_tools=client_tools,
            system_prompt=sys_text,
            side_channel=side_channel,
            veilguard_preamble=veilguard_preamble,
            # `sys_field` from the request already captured by sys_text;
            # client_system_text is the same content as a pinnable kind.
            client_system_text=(
                sys_field if isinstance(sys_field, str) else ""
            ),
        )

    # ── Overrides ─────────────────────────────────────────────────────

    def tools(self) -> list[dict]:
        # LibreChat-supplied tool schemas (its MCP integration provides
        # these).  We do NOT add our in-process ledger tools here — those
        # are for the multi-agent runtime, not user-facing chat.
        return self._client_tools

    def model(self) -> str:
        return self._model

    def include_memory(self) -> bool:
        return not self._side_channel


__all__ = ["ChatAgent", "is_side_channel"]
