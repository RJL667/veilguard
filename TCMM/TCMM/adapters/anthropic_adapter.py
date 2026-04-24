"""
Anthropic Generation Adapter - Drop-in replacement for GeminiGenerationAdapter.

Supports three modes (auto-detected):
  1. CLAUDE_SSO=1          → Routes through 'claude --print' CLI (Max/Pro subscription)
  2. ANTHROPIC_API_KEY     → Direct Anthropic API key
  3. Fallback              → Direct Anthropic API with provided key
"""

import json
import os
import subprocess
import time

try:
    from src.memory.core.perf import perf_scope
    from src.memory.core.tcmm_logging import log_adapter, log_exception
except ImportError:
    from core.perf import perf_scope
    from core.tcmm_logging import log_adapter, log_exception


class AnthropicGenerationAdapter:
    """
    Anthropic-compatible generation provider for LLM calls.

    Drop-in replacement for GeminiGenerationAdapter / OpenAIGenerationAdapter.
    Same interface: __init__(api_key, model_name), generate(prompt, label) -> str.

    Priority:
      - If CLAUDE_SSO=1 → routes through 'claude --print' CLI (uses Max/Pro subscription)
      - Elif ANTHROPIC_API_KEY → direct Anthropic API
    """

    def __init__(self, api_key: str = "", model_name: str = "claude-sonnet-4-20250514"):
        use_sso = os.environ.get("CLAUDE_SSO", "").strip() == "1"

        if use_sso:
            self.client = None
            self.model_name = model_name
            self._mode = "Claude CLI (claude --print)"
            self._use_cli = True
        else:
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            )
            self.model_name = model_name
            self._mode = "Anthropic API"
            self._use_cli = False

    def _generate_via_cli(self, prompt: str) -> str:
        """Route generation through 'claude --print' CLI."""
        max_retries = 3
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                result = subprocess.run(
                    ["claude", "--print", "--model", self.model_name],
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    encoding="utf-8",
                    errors="replace",
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
                elif result.returncode != 0:
                    err = result.stderr.strip() or result.stdout.strip()
                    retryable = any(k in err.lower() for k in
                                    ["rate", "429", "overloaded", "timeout", "503"])
                    if retryable and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        log_adapter(f"Claude CLI error (retryable). Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    log_exception("Claude CLI failed", Exception(err[:200]))
                    return ""
            except subprocess.TimeoutExpired:
                if attempt < max_retries - 1:
                    log_adapter(f"Claude CLI timeout. Retrying... (Attempt {attempt+1}/{max_retries})")
                    continue
                log_exception("Claude CLI timed out", Exception("120s timeout"))
                return ""
            except Exception as e:
                log_exception("Claude CLI error", e)
                return ""
        return ""

    def generate(self, prompt: str, label: str = None) -> str:
        """Generate text completion — same signature as GeminiGenerationAdapter."""
        with perf_scope("llm.generate_text"):
            if self._use_cli:
                text = self._generate_via_cli(prompt)
                try:
                    log_adapter(f"Raw LLM output (first 800 chars): {text[:800]}")
                    if len(text) > 800:
                        log_adapter(f"Raw LLM output (last 800 chars): {text[-800:]}")
                except Exception:
                    pass
                return text

            max_retries = 3
            base_delay = 1.0

            for attempt in range(max_retries):
                try:
                    resp = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=8192,
                        temperature=0.0,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = resp.content[0].text if resp.content else ""

                    # ---- DEBUG TAP ----
                    try:
                        log_adapter(f"Raw LLM output (first 800 chars): {text[:800]}")
                        if len(text) > 800:
                            log_adapter(f"Raw LLM output (last 800 chars): {text[-800:]}")
                    except Exception:
                        pass
                    # -------------------

                    return text
                except Exception as e:
                    err_str = str(e).lower()
                    retryable = (
                        "429" in err_str or "rate" in err_str
                        or "529" in err_str or "overloaded" in err_str
                        or "502" in err_str or "503" in err_str
                        or "timeout" in err_str
                    )
                    if retryable and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        log_adapter(f"Anthropic API error (retryable). Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        log_exception("Anthropic generate failed", e)
                        return ""
            return ""
