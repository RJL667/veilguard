"""
Gemini CLI Adapter - Uses Gemini CLI directly for LLM calls.

Requires: pip install gemini-cli
Authentication: gemini-cli login (browser-based, uses your Google account)

No API keys needed, no GCP project required, no billing concerns.
"""

import subprocess
import json
import sys
from pathlib import Path

try:
    from core.perf import perf_scope
    from core.tcmm_logging import log_adapter, log_exception
except ImportError:
    from src.memory.core.perf import perf_scope
    from src.memory.core.tcmm_logging import log_adapter, log_exception


class GeminiCLIAdapter:
    """
    Gemini LLM via CLI.

    Uses gemini-cli which handles OAuth directly with Google Sign-In.
    No API keys, no GCP setup, no billing.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        self._mode = "Gemini CLI (Google Sign-In)"

        # Verify gemini-cli is installed
        result = subprocess.run(
            ["gemini", "--version"],
            capture_output=True,
            text=True,
            shell=(sys.platform == "win32")
        )
        if result.returncode != 0:
            raise RuntimeError(
                "gemini-cli not found. Install with: pip install gemini-cli\n"
                "Then authenticate with: gemini-cli login"
            )

    def generate(self, prompt: str, label: str = None) -> str:
        """Generate text using Gemini CLI."""
        import time

        with perf_scope("llm.generate_text"):
            max_retries = 3
            base_delay = 2.0

            for attempt in range(max_retries):
                try:
                    # Call gemini-cli with the prompt
                    result = subprocess.run(
                        ["gemini", "-m", self.model_name, prompt],
                        capture_output=True,
                        text=True,
                        timeout=300,
                        shell=(sys.platform == "win32")
                    )

                    if result.returncode != 0:
                        err = result.stderr.strip()
                        if "401" in err or "Unauthorized" in err or "token" in err.lower():
                            raise RuntimeError(
                                "Not authenticated. Run: gemini-cli login"
                            )
                        retryable = (
                            "429" in err or "rate" in err.lower()
                            or "unavailable" in err.lower()
                        )
                        if retryable and attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            log_adapter(f"Gemini CLI error (retryable). Retrying in {delay}s...")
                            time.sleep(delay)
                            continue
                        else:
                            log_exception("Gemini CLI generate failed", Exception(err))
                            return ""

                    text = result.stdout.strip()
                    if not text:
                        return ""

                    try:
                        log_adapter(f"Raw LLM output (first 800 chars): {text[:800]}")
                        if len(text) > 800:
                            log_adapter(f"Raw LLM output (last 800 chars): {text[-800:]}")
                    except Exception:
                        pass

                    return text

                except subprocess.TimeoutExpired:
                    log_exception("Gemini CLI timeout", TimeoutError("Request timed out"))
                    return ""
                except Exception as e:
                    log_exception("Gemini CLI generate failed", e)
                    return ""

            return ""
