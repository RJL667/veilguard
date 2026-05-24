"""Cache control validator + normalizer.

The Claude Agent SDK auto-places `cache_control: ephemeral` markers on
system prompt blocks.  There's a documented TypeScript SDK bug where
it places the marker on *every* block instead of the last one; with
>4 blocks the Anthropic API returns HTTP 400.

The Python SDK is undocumented on this point but we don't trust it —
we own normalization here.

Rules:
  1. At most 4 cache_control markers per request (Anthropic hard limit).
  2. For the system prefix, exactly ONE marker on the LAST system block
     gives us prefix caching for everything before it (tools + system).
  3. Markers placed on per-call content (user messages, tool results)
     are pointless and waste budget; we strip them.

This module is a defensive wrapper.  Whenever we construct
ClaudeAgentOptions or pass system blocks to the SDK, we run them
through `normalize_cache_control()` first.  Unit tests verify the
output structure.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("agent-runtime.cache_control")

# Anthropic's hard limit; exceeding this returns HTTP 400.
MAX_CACHE_CONTROL_MARKERS = 4


def normalize_cache_control(
    system_blocks: list[dict[str, Any]],
    ttl: str = "1h",
) -> list[dict[str, Any]]:
    """Ensure exactly ONE cache_control marker, on the LAST system block.

    Input: list of system content blocks, each shaped like:
        {"type": "text", "text": "..."}
        {"type": "text", "text": "...", "cache_control": {...}}  ← may
                                                                    be
                                                                    present
                                                                    from
                                                                    SDK
                                                                    or
                                                                    caller

    Output: same list, with:
      - all cache_control fields stripped except on the last block
      - last block has cache_control = {"type": "ephemeral", "ttl": ttl}

    Edge cases:
      - Empty input: returns empty list (no markers, nothing to cache).
      - Single block: gets the marker.
      - >4 blocks total markers requested elsewhere: caller's
        responsibility to track total budget; this function is only
        about system-block placement.
    """
    if not system_blocks:
        return []

    # Copy each block so we don't mutate caller state.
    out: list[dict[str, Any]] = []
    stripped = 0
    for block in system_blocks:
        b = dict(block)
        if "cache_control" in b:
            b.pop("cache_control")
            stripped += 1
        out.append(b)

    if stripped > 0:
        logger.debug(
            f"[cache_control] stripped {stripped} auto-placed marker(s); "
            "re-placing on last system block only"
        )

    # Place the marker on the last block.
    out[-1]["cache_control"] = {"type": "ephemeral", "ttl": ttl}
    return out


def count_cache_control_markers(
    system_blocks: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> int:
    """Count total cache_control markers across all request sections.

    Anthropic counts markers from: tools + system + messages.  If the
    total exceeds 4, the API returns 400.  Use this to validate before
    submitting.
    """
    count = sum(1 for b in (system_blocks or []) if "cache_control" in b)
    for tool in tools or []:
        if "cache_control" in tool:
            count += 1
    for msg in messages or []:
        # Messages can have cache_control on individual content blocks.
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    count += 1
    return count


def validate_total_markers(
    system_blocks: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> None:
    """Raise ValueError if total cache_control markers exceed Anthropic's limit."""
    total = count_cache_control_markers(system_blocks, tools, messages)
    if total > MAX_CACHE_CONTROL_MARKERS:
        raise ValueError(
            f"cache_control marker budget exceeded: {total} > "
            f"{MAX_CACHE_CONTROL_MARKERS} (Anthropic API will reject with 400)"
        )


__all__ = [
    "MAX_CACHE_CONTROL_MARKERS",
    "normalize_cache_control",
    "count_cache_control_markers",
    "validate_total_markers",
]
