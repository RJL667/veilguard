"""Web MCP Tool Server for Veilguard.

Tools: google_search, browse_url, fetch_page
"""

import json
import os
import re

import httpx
import html2text
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("web", instructions="Web tools for searching Google, browsing URLs, and fetching pages.")

# Optional: Google Custom Search API credentials
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID", "")

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

h2t = html2text.HTML2Text()
h2t.ignore_links = False
h2t.ignore_images = True
h2t.body_width = 0


@mcp.tool()
async def google_search(query: str, num_results: int = 10) -> str:
    """Search Google and return results.

    If Google API credentials are configured, uses the Custom Search JSON API.
    Otherwise, falls back to scraping DuckDuckGo HTML.

    Args:
        query: Search query
        num_results: Number of results to return (max 10)
    """
    num_results = min(num_results, 10)

    if GOOGLE_API_KEY and GOOGLE_CSE_ID:
        return await _google_api_search(query, num_results)
    else:
        return await _duckduckgo_search(query, num_results)


async def _google_api_search(query: str, num: int) -> str:
    """Search using Google Custom Search JSON API."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": GOOGLE_API_KEY,
                "cx": GOOGLE_CSE_ID,
                "q": query,
                "num": num,
            },
        )
        if resp.status_code != 200:
            return f"Google API error: {resp.status_code} {resp.text[:200]}"

        data = resp.json()
        items = data.get("items", [])
        if not items:
            return f"No results for: {query}"

        results = []
        for i, item in enumerate(items, 1):
            results.append(
                f"{i}. **{item.get('title', 'Untitled')}**\n"
                f"   {item.get('link', '')}\n"
                f"   {item.get('snippet', '')}"
            )
        return f"# Search results for: {query}\n\n" + "\n\n".join(results)


async def _duckduckgo_search(query: str, num: int) -> str:
    """Fallback: scrape DuckDuckGo HTML for search results."""
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        resp = await client.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": USER_AGENT},
        )
        if resp.status_code != 200:
            return f"Search error: {resp.status_code}"

        soup = BeautifulSoup(resp.text, "html.parser")
        result_links = soup.select(".result__a")

        results = []
        for i, link in enumerate(result_links[:num], 1):
            title = link.get_text(strip=True)
            href = link.get("href", "")
            # DuckDuckGo wraps URLs
            snippet_el = link.find_parent(".result").select_one(".result__snippet") if link.find_parent(".result") else None
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            results.append(f"{i}. **{title}**\n   {href}\n   {snippet}")

        if not results:
            return f"No results for: {query}"

        return f"# Search results for: {query}\n\n" + "\n\n".join(results)


@mcp.tool()
async def browse_url(url: str, max_length: int = 8000) -> str:
    """Fetch a URL and return its content as readable markdown text.

    Args:
        url: The URL to fetch
        max_length: Maximum characters to return (default 8000)
    """
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

            return f"# Content from {url}\n\n{text}"

    except httpx.TimeoutException:
        return f"Timeout fetching {url}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


@mcp.tool()
async def fetch_page(url: str) -> str:
    """Fetch raw HTML from a URL. Useful when you need the full HTML structure.

    Args:
        url: The URL to fetch
    """
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": USER_AGENT})

            if resp.status_code != 200:
                return f"HTTP {resp.status_code} fetching {url}"

            html = resp.text
            if len(html) > 50000:
                html = html[:50000] + "\n<!-- truncated -->"

            return html

    except httpx.TimeoutException:
        return f"Timeout fetching {url}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
