"""Verification and security review tools."""

from config import ROLES, DEFAULT_BACKEND, PLAN_MODELS
from core.llm import call_llm, resolve_backend
from utils.afk import touch_activity


def register(mcp):
    @mcp.tool()
    async def verify_output(output: str, original_task: str = "", backend: str = "", model: str = "") -> str:
        """Run adversarial verification on an agent's output. Fact-checks claims and rates confidence."""
        touch_activity()
        cfg = resolve_backend(backend)
        use_model = model if model else PLAN_MODELS.get((backend or DEFAULT_BACKEND).lower(), cfg["default_model"])
        verify_task = (f"Original task: {original_task}\n\n" if original_task else "") + f"Output to verify:\n{output}\n\nVerify every factual claim."
        result = await call_llm(ROLES["verify"]["system"], verify_task, model=use_model, backend=(backend or DEFAULT_BACKEND).lower(), role="verify")
        return f"# Verification Report\n\n{result}"

    @mcp.tool()
    async def security_review(code_or_output: str, context: str = "", backend: str = "", model: str = "") -> str:
        """Run security review on code, commands, or configurations."""
        touch_activity()
        cfg = resolve_backend(backend)
        use_model = model if model else PLAN_MODELS.get((backend or DEFAULT_BACKEND).lower(), cfg["default_model"])
        review_task = (f"Context: {context}\n\n" if context else "") + f"Review for security vulnerabilities:\n\n{code_or_output}"
        result = await call_llm(ROLES["security"]["system"], review_task, model=use_model, backend=(backend or DEFAULT_BACKEND).lower(), role="security")
        return f"# Security Review\n\n{result}"
