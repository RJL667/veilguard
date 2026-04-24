"""
OpenAI Generation Adapter - Drop-in replacement for GeminiGenerationAdapter.

Supports three modes (auto-detected):
  1. CHATGPT_OAUTH=1         → codex-auth: Responses API via ChatGPT subscription
  2. OPENAI_API_KEY          → Direct OpenAI API (Chat Completions)
  3. OPENCLAW_BASE_URL       → OpenClaw proxy (Chat Completions)
"""

import os
import time

try:
    from src.memory.core.perf import perf_scope
    from src.memory.core.tcmm_logging import log_adapter, log_exception
except ImportError:
    from core.perf import perf_scope
    from core.tcmm_logging import log_adapter, log_exception


class OpenAIGenerationAdapter:
    """
    OpenAI-compatible generation provider for LLM calls.

    Drop-in replacement for GeminiGenerationAdapter.
    Same interface: __init__(api_key, model_name), generate(prompt, label) -> str.

    Priority:
      - If CHATGPT_OAUTH=1 → uses codex-auth + Responses API
        Opens browser on first run, caches token in ~/.codex-auth/auth.json
      - Elif OPENAI_API_KEY is set → direct OpenAI Chat Completions API
      - Elif OPENCLAW_BASE_URL is set → OpenClaw proxy (Chat Completions)
    """

    def __init__(self, api_key: str = "", model_name: str = "gpt-5.4-mini",
                 base_url: str = None):

        use_oauth = os.environ.get("CHATGPT_OAUTH", "").strip() == "1"

        if use_oauth:
            # codex-auth: uses ChatGPT subscription via OAuth, no API key needed
            import codex_auth
            from openai import OpenAI
            self.client = OpenAI()       # codex-auth patches the transport automatically
            self.model_name = model_name
            self._mode = "ChatGPT OAuth (codex-auth)"
        else:
            from openai import OpenAI
            # Determine base URL: explicit > env OPENCLAW_BASE_URL > None (default OpenAI)
            if base_url is None:
                base_url = os.environ.get("OPENCLAW_BASE_URL", "").strip() or None

            kwargs = {"api_key": api_key or "dummy"}
            if base_url:
                kwargs["base_url"] = base_url

            self.client = OpenAI(**kwargs)
            self.model_name = model_name
            self._mode = f"OpenClaw ({base_url})" if base_url else "OpenAI API"

    def generate(self, prompt: str, label: str = None) -> str:
        """Generate text completion — same signature as GeminiGenerationAdapter."""
        with perf_scope("llm.generate_text"):
            max_retries = 3
            base_delay = 1.0

            for attempt in range(max_retries):
                try:
                    if self._mode == "ChatGPT OAuth (codex-auth)":
                        # Codex endpoint uses Responses API, not Chat Completions
                        resp = self.client.responses.create(
                            model=self.model_name,
                            input=[{"role": "user", "content": prompt}],
                        )
                        text = resp.output_text or ""
                    else:
                        # Standard OpenAI Chat Completions API
                        resp = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.0,
                            timeout=60,
                        )
                        text = resp.choices[0].message.content or ""

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
                        or "502" in err_str or "503" in err_str
                        or "504" in err_str or "timeout" in err_str
                    )
                    if retryable and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        log_adapter(f"OpenAI API error (retryable). Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        log_exception("OpenAI generate failed", e)
                        return ""
            return ""
