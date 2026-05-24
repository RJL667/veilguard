"""Anthropic LLM adapter — thin wrapper around TCMM's AnthropicGenerationAdapter.

Per the design call (your decision #1): use sys.path import to reach
TCMM's existing adapter instead of vendoring it.  TCMM's adapter
already handles:
  - OAuth bearer auth via ~/.claude/.credentials.json (CLAUDE_SSO=1)
  - Auto-refresh against platform.claude.com/v1/oauth/token
  - Beta headers per model (claude-code-20250219, oauth-2025-04-20,
    interleaved-thinking-2025-05-14 for opus/sonnet-4.x)
  - Required CLI system prefix policy gate
  - x-api-key fallback when CLAUDE_SSO is unset
  - Worker pool / connection reuse

We just put a clean class around it that hides:
  - The sys.path dance + import collision risk
  - The "mutate instance attrs after .generate()" pattern (we return
    a frozen dataclass instead)
  - The sync/async impedance (TCMM's .generate is sync; we await it
    in a thread to keep our event loop free)

Env vars:
  TCMM_ROOT  — path to the TCMM repo.  Default
               ~/.gemini/antigravity/tcmm/TCMM/  (the user's canonical
               local layout per `architecture_tcmm_canonical_path`).
  CLAUDE_SSO — propagated to the adapter.  Set to "1" or "bearer" to
               use OAuth Max subscription; unset/anything-else for
               x-api-key path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("veilguard.llm.adapter")


# ── Lazy import of TCMM's adapter ───────────────────────────────────────


_TCMM_ROOT_DEFAULT = Path.home() / ".gemini" / "antigravity" / "tcmm" / "TCMM"


def _tcmm_root() -> Path:
    return Path(os.environ.get("TCMM_ROOT", str(_TCMM_ROOT_DEFAULT)))


_ADAPTER_CLS = None  # populated on first call


def _import_tcmm_adapter():
    """Import TCMM's AnthropicGenerationAdapter, caching the class.

    We can't use `from adapters.anthropic_adapter import ...` because
    that triggers `adapters/__init__.py` which eagerly imports the
    gemini adapter and pulls in `google.generativeai` — a dependency
    we don't need and that's not installed in the pii-proxy container.

    Instead: put TCMM_ROOT on sys.path (so the adapter's own
    `from core.perf import ...` works), then importlib-load the
    anthropic_adapter.py file DIRECTLY, bypassing the package
    __init__.  This avoids touching gemini_adapter / langchain_adapter
    entirely.
    """
    global _ADAPTER_CLS
    if _ADAPTER_CLS is not None:
        return _ADAPTER_CLS

    root = _tcmm_root()
    if not root.is_dir():
        raise RuntimeError(
            f"TCMM_ROOT={root} does not exist.  Set the TCMM_ROOT env "
            f"var or symlink the TCMM tree to the default location."
        )

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        logger.info(f"[adapter] inserted {root_str} at sys.path[0]")

    # Load anthropic_adapter.py DIRECTLY (skip adapters/__init__.py).
    import importlib.util
    adapter_file = root / "adapters" / "anthropic_adapter.py"
    if not adapter_file.is_file():
        raise RuntimeError(
            f"AnthropicGenerationAdapter source not found at {adapter_file}. "
            f"Is TCMM_ROOT={root_str} pointing at a real TCMM tree?"
        )
    spec = importlib.util.spec_from_file_location(
        "tcmm_anthropic_adapter_loaded", str(adapter_file),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not build module spec for {adapter_file}")
    module = importlib.util.module_from_spec(spec)
    # Register under a unique name so re-imports don't collide.
    sys.modules["tcmm_anthropic_adapter_loaded"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "AnthropicGenerationAdapter"):
        raise RuntimeError(
            f"AnthropicGenerationAdapter class not found in {adapter_file}"
        )
    _ADAPTER_CLS = module.AnthropicGenerationAdapter
    logger.info(
        f"[adapter] TCMM AnthropicGenerationAdapter loaded from "
        f"{adapter_file} (direct file import — bypasses adapters/__init__.py)"
    )
    return _ADAPTER_CLS


# ── Public surface ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class AdapterResult:
    """Immutable result of one LLM call.

    Replaces TCMM's "mutate self.last_* attrs" pattern, which is
    inherently race-y under concurrent reuse and easy to misread.
    """
    text: str                       # joined text from all text blocks
    content_blocks: list[dict]      # Anthropic-shape content array
    usage: dict                     # {input_tokens, output_tokens,
                                    #  cache_creation_input_tokens,
                                    #  cache_read_input_tokens}
    stop_reason: str                # "end_turn" | "tool_use" |
                                    #  "max_tokens" | ...
    mode: str = ""                  # "Anthropic SDK (OAuth bearer)" | etc.
    model: str = ""


class AnthropicAdapter:
    """Cleaner interface over TCMM's AnthropicGenerationAdapter.

    Lifetime: instantiate per turn (cheap — the underlying client is
    pooled inside TCMM's adapter via `_POOL_REGISTRY` keyed on
    (model, prompt_fingerprint) so concurrent callers with the same
    cache prefix dedupe).
    """

    def __init__(
        self,
        *,
        model: str,
        system_blocks: Optional[list[dict]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        agent_id: str = "",   # ignored; matches ScriptedAdapter signature
    ):
        """
        Args:
          model: e.g. "claude-sonnet-4-5".  Aliases like "sonnet" do
            NOT work at the API layer — must be the full ID.
          system_blocks: pre-rendered system blocks (with cache_control
            markers placed by TCMM's renderer).  If provided, takes
            precedence over `system_prompt`.
          system_prompt: plain-text system prompt.  Used only when
            system_blocks is empty.
          tools: Anthropic-shape tool schemas (name, description,
            input_schema).
        """
        AdapterCls = _import_tcmm_adapter()
        self._inner = AdapterCls(
            api_key="",   # TCMM adapter resolves via CLAUDE_SSO or env
            model_name=model,
            system_prompt=system_prompt,
            system_blocks=list(system_blocks) if system_blocks else None,
            tools=list(tools) if tools else None,
        )
        self._model = model

    async def generate(
        self, user_message: str, *, label: str = "veilguard"
    ) -> AdapterResult:
        """Run one LLM call.  Returns a frozen result struct.

        TCMM's `.generate` is sync (does HTTPS via the anthropic SDK);
        we await it in a thread so our event loop stays free.
        """
        text = await asyncio.to_thread(
            self._inner.generate, user_message, label
        )
        text = text or ""

        # TCMM's adapter mutates self.last_* on the inner — read them
        # immediately while we own the only reference.
        blocks = getattr(self._inner, "last_response_blocks", None)
        if not isinstance(blocks, list) or not blocks:
            blocks = [{"type": "text", "text": text}]

        usage = getattr(self._inner, "last_usage", None) or {}
        stop_reason = (
            getattr(self._inner, "last_stop_reason", None) or "end_turn"
        )
        mode = getattr(self._inner, "_mode", "") or ""

        return AdapterResult(
            text=text,
            content_blocks=blocks,
            usage=dict(usage) if isinstance(usage, dict) else {},
            stop_reason=stop_reason,
            mode=mode,
            model=self._model,
        )


__all__ = ["AnthropicAdapter", "AdapterResult"]
