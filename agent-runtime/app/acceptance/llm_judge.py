"""Phase 7.5 — default LLM judge for `llm_verify` execution.

The `llm_verify` executor in `executors.py` accepts an arbitrary
`ctx['llm_judge']` callable so tests can inject a stub.  Production
callers can also bring their own — but for the common case (Critic
dispatch wants to re-verify an `llm_verify` AC server-side), this
module provides a default judge that wraps the existing LLM backend.

Contract:

  judge(prompt: str, model: str) -> dict
    {
      "verdict":    "pass" | "fail",
      "confidence": float in [0, 1],
      "reason":     str,
    }

The executor's `_build_llm_verify_prompt` already produces a JSON-
constrained prompt; the judge's job is to call the LLM and parse the
response back into the contract dict.

Pricing: Haiku-4-5 by default (per `architecture_token_accounting.md`
Haiku is the cost-correct judge for short rubrics).  Caller can
override via `args.model` in the AC's check_args.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

logger = logging.getLogger("agent-runtime.acceptance.llm_judge")


# ── JSON envelope parser ────────────────────────────────────────────────


# The executor's prompt asks for "Output JSON only" — but LLMs sometimes
# wrap it in ```json ... ``` fences or add a leading "Here's my
# verdict:".  This regex pulls out the first JSON object in the
# response without exploding on the noise.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_verdict_json(response_text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a (possibly fenced) response.

    Returns None on parse failure — the executor will then mark the
    check as `error` with reason "judge returned invalid JSON".
    """
    if not response_text:
        return None
    m = _JSON_OBJECT_RE.search(response_text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None


# ── Default judge factory ───────────────────────────────────────────────


def make_default_llm_judge() -> Callable[[str, str], dict]:
    """Build a default judge bound to the runtime's LLM backend.

    Lazily imports `app.runtime` so this module stays light at import
    time — no circular dependency with backends.

    The returned callable matches the `llm_verify` ctx['llm_judge']
    contract: `judge(prompt, model) -> {verdict, confidence, reason}`.

    On any backend error (network, auth, malformed response) it returns
    a dict with verdict='fail' and a reason explaining the failure —
    the executor then marks the CheckResult as `fail`, which blocks
    the AC gate.  This is the conservative default: a broken judge
    should NOT clear tasks.
    """
    def judge(prompt: str, model: str) -> dict:
        # Backend availability check.  When the LLM backend isn't even
        # importable, that's a system-level error (not a content
        # judgement) — RAISE so the executor catches it and returns
        # CheckResult(status='error') with a clear reason.  This keeps
        # the existing "missing judge → error" contract intact for
        # both explicit-None and unavailable-backend cases.
        try:
            from ..llm.adapter import call_anthropic_messages  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "default_llm_judge: backend (app.llm.adapter) not "
                "available — wire the Anthropic adapter or inject an "
                f"explicit ctx['llm_judge'].  Cause: {e}"
            ) from e
        try:
            response = call_anthropic_messages(
                model=model,
                system_prompt=(
                    "You are an acceptance criterion judge.  Reply with "
                    "strict JSON only."
                ),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning(f"[llm_judge] backend call failed: {type(e).__name__}: {e}")
            return {
                "verdict": "fail",
                "confidence": 0.0,
                "reason": f"judge backend error: {type(e).__name__}",
            }
        # Backend response shape varies by adapter; normalize.
        text = (
            response.get("text")
            if isinstance(response, dict)
            else getattr(response, "text", None)
        )
        if not text:
            return {
                "verdict": "fail",
                "confidence": 0.0,
                "reason": "judge returned empty response",
            }
        parsed = _parse_verdict_json(text)
        if parsed is None:
            return {
                "verdict": "fail",
                "confidence": 0.0,
                "reason": f"judge returned invalid JSON (first 80 chars: {text[:80]!r})",
            }
        # Normalize shape so the executor doesn't have to defensively
        # parse the keys.
        verdict = str(parsed.get("verdict") or "").strip().lower()
        if verdict not in ("pass", "fail"):
            return {
                "verdict": "fail",
                "confidence": 0.0,
                "reason": f"judge returned non-pass/fail verdict {verdict!r}",
            }
        try:
            confidence = float(parsed.get("confidence") or 0.0)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.0
        reason_text = str(parsed.get("reason") or "")[:1500]
        return {
            "verdict":    verdict,
            "confidence": confidence,
            "reason":     reason_text,
        }

    return judge


# Module-level default — callers do `from app.acceptance.llm_judge
# import default_llm_judge` and pass it as ctx['llm_judge'].
default_llm_judge = make_default_llm_judge()


__all__ = [
    "default_llm_judge",
    "make_default_llm_judge",
    "_parse_verdict_json",
]
