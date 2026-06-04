"""Gemini google_search grounding for sub-agents' ``web_search`` tool.

[WEB_SEARCH_AISTUDIO_2026-06-04] Rewired off the dead Vertex AI path. The old
implementation authenticated via ADC against ``{region}-aiplatform.googleapis.com``
— but Vertex ADC is expired (and the prod SA lost aiplatform access), so every
sub-agent web_search returned "Vertex AI credentials unavailable" and researchers
silently degraded to memory-only. This now mirrors ``services/web/server.py::
google_search`` EXACTLY: AI Studio (Generative Language API) with ``google_search``
grounding, authed by the ``GOOGLE_API_KEY`` header — no ADC, no GOOGLE_CLOUD_PROJECT,
no region. Function name kept (``vertex_grounded_search``) so callers are unchanged.
"""

import asyncio
import os
import random
from typing import Optional  # noqa: F401 (kept for signature parity)

import httpx

# Same model + endpoint as services/web/server.py (verified 2026-06-03):
# gemini-3.1-flash-lite is the cheapest grounding tier (~1.4s median; 503 spikes
# covered by the retry below). Avoid 3-flash-preview/3.5/pro variants (slow/404).
_SEARCH_MODEL = os.environ.get(
    "GEMINI_SEARCH_MODEL", os.environ.get("VERTEX_SEARCH_MODEL", "gemini-3.1-flash-lite")
)
_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
_API_KEY = (
    os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
).strip()


async def vertex_grounded_search(query: str, num_results: int = 8) -> str:
    """Search the web via Gemini ``google_search`` grounding (AI Studio).

    Returns a markdown-ish string — the model's grounded summary + a
    ``## Sources:`` block with the real grounding-chunk URLs. On any failure
    returns a readable error string (callers never need to handle exceptions).
    """
    query = (query or "").strip()
    if not query:
        return "Error: empty query"
    num_results = max(1, min(int(num_results), 10))

    if not _API_KEY:
        return (
            "Error: web search is not configured — set GOOGLE_API_KEY (the "
            "AI Studio / Gemini API key) in the environment. "
            "(web_fetch works without a key.)"
        )

    url = f"{_ENDPOINT}/{_SEARCH_MODEL}:generateContent"
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text":
                f"Search Google and answer: {query}\n\n"
                f"Give a concise, factual summary grounded in the search "
                f"results, then list the {num_results} most relevant sources "
                f"(Title + URL + one-sentence summary each)."
            }],
        }],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
    }

    # Retry on 429 (quota) + transient 5xx with exponential backoff + jitter.
    # Parallel sub-agent bursts trip the per-minute cap; without this each
    # fan-out drops workers instead of producing results a second later.
    _RETRYABLE = {429, 500, 502, 503, 504}
    backoff = 0.5
    resp = None
    try:
        # Key via header (x-goog-api-key), not ?key= query param, so it never
        # lands in URL logs.
        async with httpx.AsyncClient(timeout=45) as client:
            for attempt in range(4):
                resp = await client.post(
                    url, headers={"x-goog-api-key": _API_KEY}, json=payload
                )
                if resp.status_code not in _RETRYABLE:
                    break
                if attempt == 3:
                    break
                retry_after = resp.headers.get("retry-after")
                try:
                    wait_s = float(retry_after) if retry_after else backoff
                except ValueError:
                    wait_s = backoff
                await asyncio.sleep(min(wait_s, 4.0) + random.uniform(0, 0.25))
                backoff *= 2
    except Exception as e:
        return f"Error: Gemini search request failed: {type(e).__name__}: {e}"

    if resp is None or resp.status_code != 200:
        code = resp.status_code if resp is not None else "no-response"
        body = resp.text[:300] if resp is not None else ""
        return f"Gemini search error: {code} {body}"

    data = resp.json()
    out: list[str] = []
    for cand in data.get("candidates", []) or []:
        for part in cand.get("content", {}).get("parts", []) or []:
            if "text" in part:
                out.append(part["text"])
        grounding = cand.get("groundingMetadata", {}) or {}
        chunks = grounding.get("groundingChunks", []) or []
        if chunks:
            out.append("\n## Sources:")
            for i, chunk in enumerate(chunks[:num_results], 1):
                web = chunk.get("web", {}) or {}
                out.append(f"{i}. **{web.get('title', 'Untitled')}**\n   {web.get('uri', '')}")
        break  # only the first candidate carries grounding
    return "\n\n".join(out) if out else f"No results for: {query}"
