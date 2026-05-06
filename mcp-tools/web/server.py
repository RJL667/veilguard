"""Web MCP Tool Server for Veilguard.

Tools: google_search, browse_url, fetch_page

Search backend: Vertex AI Gemini with ``google_search`` grounding.
Reasons we moved off the Custom Search JSON API (Jan 2026):
  * Google deprecated "Search the entire web" on Programmable Search
    Engine — new engines are site-restricted (≤50 domains) and the
    feature sunsets entirely on 2027-01-01.
  * Google officially points users at Vertex AI for enterprise-grade
    web search, which is what TCMM already uses for embeddings and
    classification — so we're unifying onto one auth path.

Auth: Application Default Credentials.  On the prod VM this resolves
to the GCE instance's service account via the metadata server at
169.254.169.254.  Locally it picks up ``gcloud auth
application-default login``.  No API keys in ``.env`` any more — the
old GOOGLE_API_KEY / GOOGLE_CSE_KEY / GOOGLE_CSE_ID are unused.
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
# live under mcp-tools/_shared/ so every server can import them with a
# 2-line shim. Stdlib only — no extra requirements.txt entries needed.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from tool_caching import cached  # noqa: E402
from result_offload import auto_offload  # noqa: E402
from tool_cost import record_cost, format_cost_hint  # noqa: E402

mcp = FastMCP("web", instructions="Web tools for searching Google, browsing URLs, and fetching pages.")

# Vertex AI configuration — same project TCMM uses.
VERTEX_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
VERTEX_LOCATION = os.environ.get("VERTEX_REGION", "us-central1")
VERTEX_SEARCH_MODEL = os.environ.get("VERTEX_SEARCH_MODEL", "gemini-2.5-flash")

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
    """Search the web via Vertex AI Gemini with google_search grounding.

    Results are cached for 10 minutes — the same query in quick
    succession returns the same result set without re-billing Vertex.
    For news/breaking-event queries where 10min staleness matters,
    vary the query slightly to bypass the cache.

    Args:
        query: Search query
        num_results: Number of result-citations to return (max 10)
    """
    num_results = min(num_results, 10)

    if not VERTEX_PROJECT:
        return (
            "Error: GOOGLE_CLOUD_PROJECT env not set on the web MCP server. "
            "Vertex AI search requires the project the service account is "
            "scoped to."
        )

    token = await _get_access_token()
    if not token:
        return (
            "Error: Vertex AI credentials unavailable. On GCE the VM's "
            "service account must have roles/aiplatform.user; locally "
            "run `gcloud auth application-default login`."
        )

    url = (
        f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{VERTEX_PROJECT}/locations/{VERTEX_LOCATION}/"
        f"publishers/google/models/{VERTEX_SEARCH_MODEL}:generateContent"
    )
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text":
                f"Search the web and return the {num_results} most relevant "
                f"results for: {query}\n\n"
                f"For each result provide:\n"
                f"- Title\n- URL\n- One-sentence summary\n"
                f"Cite the sources in the grounding metadata."
            }],
        }],
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024,
        },
    }

    async with httpx.AsyncClient(timeout=45) as client:
        resp = await client.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )

    if resp.status_code != 200:
        return f"Vertex search error: {resp.status_code} {resp.text[:300]}"

    data = resp.json()
    results: list[str] = []
    candidates = data.get("candidates", [])
    if candidates:
        cand = candidates[0]
        for part in cand.get("content", {}).get("parts", []) or []:
            if "text" in part:
                results.append(part["text"])

        # Grounding metadata carries the actual Google Search sources —
        # emit them as a citations block so downstream agents can cite
        # URLs directly.
        grounding = cand.get("groundingMetadata", {})
        chunks = grounding.get("groundingChunks", []) or []
        if chunks:
            results.append("\n## Sources:")
            for i, chunk in enumerate(chunks[:num_results], 1):
                web = chunk.get("web", {}) or {}
                title = web.get("title", "Untitled")
                uri = web.get("uri", "")
                results.append(f"{i}. **{title}**\n   {uri}")

    if not results:
        return f"No results for: {query}"
    return "\n\n".join(results)


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
