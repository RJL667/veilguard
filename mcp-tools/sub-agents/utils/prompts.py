"""Veilguard Sub-Agents: Prompt composition and frustration detection."""

import re

# ── Cache-Aware Prompt Composition ───────────────────────────────────────────

_STABLE_PREFIX = "You are a Phishield AI agent. Follow instructions precisely.\n\n"


def build_prompt(role_prompt: str, extra: str = "") -> str:
    """Compose system prompt with stable prefix for cache optimization."""
    parts = [_STABLE_PREFIX, "ROLE:\n", role_prompt, "\n"]
    if extra:
        parts.append(f"\n{extra}\n")
    return "".join(parts)


# ── Frustration Detection ────────────────────────────────────────────────────

_FRUSTRATION_PATTERNS = [
    r"\b(wtf|broken|useless|stupid|terrible|awful|garbage|trash|crap)\b",
    r"(!!!|\.\.\.\.+)",
    r"\b(still not|doesn't work|not working|wrong again|try again)\b",
    r"\b(i (already|just) (told|said|asked))\b",
]
_FRUSTRATION_RE = re.compile("|".join(_FRUSTRATION_PATTERNS), re.IGNORECASE)

FRUSTRATION_PREFIX = (
    "The user appears frustrated. Be extra concise, acknowledge the issue directly, "
    "and focus on solving it immediately. No filler or pleasantries.\n\n"
)


def detect_frustration(text: str) -> bool:
    return bool(_FRUSTRATION_RE.search(text))


# ── Cost Estimation ──────────────────────────────────────────────────────────

TOKEN_COSTS = {
    "gemini-2.5-flash": {"input": 0.00015, "output": 0.0006},
    "gemini-2.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-3-pro-preview": {"input": 0.00125, "output": 0.005},
    "gemini-3-flash-preview": {"input": 0.00015, "output": 0.0006},
    "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
    "claude-opus-4": {"input": 0.015, "output": 0.075},
}


def estimate_cost(model: str, input_chars: int, output_chars: int) -> float:
    """Rough cost estimate: ~4 chars per token."""
    costs = TOKEN_COSTS.get(model)
    if not costs:
        return 0.0
    input_tokens = input_chars / 4
    output_tokens = output_chars / 4
    return (input_tokens / 1000 * costs["input"]) + (output_tokens / 1000 * costs["output"])
