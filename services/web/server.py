"""Web MCP Tool Server for Veilguard.

Tools: google_search, browse_url, fetch_page

Search backend: Grounding with Google Search via the Gemini API
(AI Studio — https://generativelanguage.googleapis.com) on a Gemini 3
model. The model runs Google searches and returns a grounded summary
plus the source URLs (groundingMetadata) for the agent to read/cite.

Why this path (chosen 2026-06-03):
  * Google Custom Search JSON API is being shut down (closed to new
    signups; "search entire web" sunsets 2027-01-01) — new projects get
    403 "no access".
  * google_search grounding is Google's current "search via API".
    Gemini 3 grounding is billed per query at ~$14/1k.
  * AI Studio uses a plain API key (GOOGLE_API_KEY) — no ADC/IAM/billing
    project setup like Vertex — and works the same locally and on the VM.

Auth: set ``GOOGLE_API_KEY`` (the AI Studio / Gemini API key). The key's
project must have billing credit — grounding needs the paid tier.
``browse_url`` / ``fetch_page`` need no key.
"""

import asyncio
import json
import os
import pathlib
import re
import sys
import time
from typing import Optional

import httpx
import html2text
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP

# Pull in shared MCP helpers (caching, offload, error hints, cost). These
# live under services/_shared/ so every server can import them with a
# 2-line shim. Stdlib only — no extra requirements.txt entries needed.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from tool_caching import cached  # noqa: E402
from result_offload import auto_offload  # noqa: E402
from tool_cost import record_cost, format_cost_hint  # noqa: E402

mcp = FastMCP("web", instructions="Web tools for searching Google, browsing URLs, and fetching pages.")

# Gemini API (AI Studio) — google_search grounding on a Gemini 3 model.
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
# gemini-3.1-flash-lite: CHEAPEST grounding tier ($14/1k, Gemini-3 per-query)
# + cheapest model tokens. Grounds in ~1.4s median (Gemini-3 "thinking" can
# spike to ~4s; the 503-retry below covers transient overload). Chosen for cost.
# Faster-but-pricier alt: gemini-2.5-flash-lite (consistent ~1.5s, $35/1k).
# Avoid: gemini-3-flash-preview (~22s+503s), gemini-3.5-flash (hangs),
# gemini-3-pro-preview/gemini-2.0-flash-lite (404). Verified 2026-06-03.
GEMINI_SEARCH_MODEL = os.environ.get("GEMINI_SEARCH_MODEL", "gemini-3.1-flash-lite")
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

h2t = html2text.HTML2Text()
h2t.ignore_links = False
h2t.ignore_images = True
h2t.body_width = 0

# Cached OAuth access token.  google.auth's default() call is synchronous
# and mildly expensive (metadata server round-trip), so we reuse the
# token until ~2 min before expiry.  refresh() on a loaded Credentials
# object is also synchronous; wrap the whole thing in a module-level
# cache with an expiry cushion.
_TOKEN_CACHE = {"token": None, "expiry": 0.0}
_TOKEN_LOCK = asyncio.Lock()


async def _get_access_token() -> Optional[str]:
    """Return a Vertex-capable OAuth access token via ADC, cached."""
    async with _TOKEN_LOCK:
        now = time.time()
        if _TOKEN_CACHE["token"] and _TOKEN_CACHE["expiry"] - now > 120:
            return _TOKEN_CACHE["token"]

        def _refresh():
            # Imported lazily — google.auth isn't a tiny dep and we don't
            # want to block MCP registration if ADC isn't configured.
            import google.auth
            from google.auth.transport.requests import Request
            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            credentials.refresh(Request())
            return credentials

        try:
            creds = await asyncio.to_thread(_refresh)
        except Exception as e:
            # Surface a readable error in the tool response rather than
            # an exception that the MCP framework would swallow.
            return None

        _TOKEN_CACHE["token"] = creds.token
        _TOKEN_CACHE["expiry"] = (
            creds.expiry.timestamp() if getattr(creds, "expiry", None) else now + 3600
        )
        return creds.token


@mcp.tool()
@cached(ttl_seconds=600)
async def google_search(query: str, num_results: int = 10) -> str:
    """Search the web via Google Search grounding on a Gemini 3 model.

    The model runs Google searches and returns a grounded, factual
    summary plus the real source URLs it used — feed both to the
    reasoning agent. Cached 10 minutes so identical queries don't
    re-bill; vary the query slightly for fresh breaking-news results.

    Args:
        query: Search query
        num_results: How many source citations to include (1-20)
    """
    num_results = min(max(num_results, 1), 20)

    if not GEMINI_API_KEY:
        return (
            "Error: web search is not configured — set GOOGLE_API_KEY (the "
            "AI Studio / Gemini API key) in the environment. "
            "(browse_url / fetch_page work without a key.)"
        )

    url = f"{GEMINI_ENDPOINT}/{GEMINI_SEARCH_MODEL}:generateContent"
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text":
                f"Search Google and answer: {query}\n\n"
                f"Give a concise, factual summary grounded in the search "
                f"results, then list the {num_results} most relevant sources."
            }],
        }],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
    }

    resp = None
    try:
        # Key via header, not ?key= query param, so it isn't logged in URLs.
        async with httpx.AsyncClient(timeout=45) as client:
            for attempt in range(3):
                resp = await client.post(
                    url, headers={"x-goog-api-key": GEMINI_API_KEY}, json=payload
                )
                if resp.status_code != 503:
                    break
                # 503 = transient model-overload ("high demand"); back off and retry.
                await asyncio.sleep(2 * (attempt + 1))
    except Exception as e:
        return f"Error: Gemini search request failed: {type(e).__name__}: {e}"

    if resp is None or resp.status_code == 503:
        return (
            "Error: web search model is overloaded (503) and didn't recover "
            "after retries — try again in a moment."
        )
    if resp.status_code == 429:
        return (
            "Error: web search hit 429 — the AI Studio project is out of "
            "prepayment credits (grounding needs the paid tier). Top up at "
            "https://ai.studio/projects, or point GOOGLE_API_KEY at a funded key."
        )
    if resp.status_code == 404:
        return (
            f"Error: model {GEMINI_SEARCH_MODEL!r} not available for this key "
            f"(404). Set GEMINI_SEARCH_MODEL to an available model "
            f"(e.g. gemini-3-pro-preview or gemini-3.5-flash)."
        )
    if resp.status_code != 200:
        return f"Error: Gemini search HTTP {resp.status_code}: {resp.text[:200]}"

    data = resp.json()
    cands = data.get("candidates") or []
    if not cands:
        return f"No results for: {query}"
    cand = cands[0]

    out: list[str] = []
    for part in cand.get("content", {}).get("parts", []) or []:
        if "text" in part:
            out.append(part["text"])

    # groundingMetadata carries the actual Google Search sources.
    chunks = (cand.get("groundingMetadata", {}) or {}).get("groundingChunks", []) or []
    if chunks:
        out.append("\n## Sources")
        for i, ch in enumerate(chunks[:num_results], 1):
            web = ch.get("web", {}) or {}
            out.append(f"{i}. {web.get('title', 'Untitled')}\n   {web.get('uri', '')}")

    if not out:
        return f"No results for: {query}"
    return "\n\n".join(out)


@mcp.tool()
@cached(ttl_seconds=300)
async def browse_url(url: str, max_length: int = 8000) -> str:
    """Fetch a URL and return its content as readable markdown text.

    Results are cached for 5 minutes — re-fetching the same URL with
    the same max_length returns the cached markdown.

    Args:
        url: The URL to fetch
        max_length: Maximum characters to return (default 8000)
    """
    from urllib.parse import urlparse
    cost_key = urlparse(url).netloc or url
    fetch_started = time.time()
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": USER_AGENT})

            if resp.status_code != 200:
                return f"HTTP {resp.status_code} fetching {url}"

            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return f"Non-text content type: {content_type} ({len(resp.content)} bytes)"

            html = resp.text
            # Convert HTML to markdown
            text = h2t.handle(html)
            # Clean up excessive whitespace
            text = re.sub(r"\n{3,}", "\n\n", text).strip()

            if len(text) > max_length:
                text = text[:max_length] + f"\n\n... (truncated, {len(text)} total chars)"

            # Cost telemetry — record actual upstream payload, not the
            # post-truncation size. Only fires on cache miss because
            # @cached short-circuits before reaching here on hits.
            wall = time.time() - fetch_started
            record_cost("browse_url", cost_key,
                         bytes_out=len(html), wall_seconds=wall)
            return (f"# Content from {url}\n\n{text}"
                    + format_cost_hint("browse_url", cost_key,
                                        bytes_out=len(html)))

    except httpx.TimeoutException:
        return f"Timeout fetching {url}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


@mcp.tool()
@cached(ttl_seconds=300)
@auto_offload(threshold=50_000)
async def fetch_page(url: str) -> str:
    """Fetch raw HTML from a URL. Useful when you need the full HTML structure.

    Cached for 5 minutes — same URL → same HTML. If the HTML is larger
    than ~50KB it gets offloaded to disk and the response includes an
    inline preview plus a path you can ``read_file`` for the rest.

    Args:
        url: The URL to fetch
    """
    from urllib.parse import urlparse
    cost_key = urlparse(url).netloc or url
    fetch_started = time.time()
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": USER_AGENT})

            if resp.status_code != 200:
                return f"HTTP {resp.status_code} fetching {url}"

            html = resp.text
            true_size = len(html)
            if len(html) > 50000:
                html = html[:50000] + "\n<!-- truncated -->"

            wall = time.time() - fetch_started
            record_cost("fetch_page", cost_key,
                         bytes_out=true_size, wall_seconds=wall)
            return html + format_cost_hint("fetch_page", cost_key,
                                              bytes_out=true_size)

    except httpx.TimeoutException:
        return f"Timeout fetching {url}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
