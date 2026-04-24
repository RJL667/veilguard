"""Vertex AI Gemini with google_search grounding.

Shared helper used by sub-agents' ``web_search`` tool (and anywhere
else that wants real, grounded web results).  Authenticates via ADC —
on the prod GCE VM this resolves to the instance's default service
account through the metadata server; locally it picks up whatever
``gcloud auth application-default login`` left behind.

Mirrors ``mcp-tools/web/server.py::google_search`` intentionally (same
endpoint, same token cache, same citations block) but lives in
sub-agents so both callers can use it without cross-package imports.
Happy to collapse the two implementations into a single shared module
once we add a ``packages/veilguard-common`` home for this kind of
plumbing.
"""

import asyncio
import os
import time
from typing import Optional

import httpx


_VERTEX_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
_VERTEX_LOCATION = os.environ.get("VERTEX_REGION", "us-central1")
_VERTEX_MODEL = os.environ.get("VERTEX_SEARCH_MODEL", "gemini-2.5-flash")

_TOKEN_CACHE = {"token": None, "expiry": 0.0}
_TOKEN_LOCK = asyncio.Lock()


async def _get_access_token() -> Optional[str]:
    """Fetch an OAuth access token via ADC; cache until 2 min before expiry."""
    async with _TOKEN_LOCK:
        now = time.time()
        if _TOKEN_CACHE["token"] and _TOKEN_CACHE["expiry"] - now > 120:
            return _TOKEN_CACHE["token"]

        def _refresh():
            import google.auth
            from google.auth.transport.requests import Request
            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            credentials.refresh(Request())
            return credentials

        try:
            creds = await asyncio.to_thread(_refresh)
        except Exception:
            return None

        _TOKEN_CACHE["token"] = creds.token
        _TOKEN_CACHE["expiry"] = (
            creds.expiry.timestamp()
            if getattr(creds, "expiry", None)
            else now + 3600
        )
        return creds.token


async def vertex_grounded_search(query: str, num_results: int = 8) -> str:
    """Search the web via Vertex AI Gemini ``google_search`` grounding.

    Returns a markdown-ish string — model-generated summaries + a
    ``## Sources:`` block with the grounding chunk URLs.  On any
    failure returns a readable error string (callers never need to
    handle exceptions).
    """
    query = (query or "").strip()
    if not query:
        return "Error: empty query"
    num_results = max(1, min(int(num_results), 10))

    if not _VERTEX_PROJECT:
        return (
            "Error: GOOGLE_CLOUD_PROJECT env not set — Vertex search "
            "needs the project the service account is scoped to."
        )

    token = await _get_access_token()
    if not token:
        return (
            "Error: Vertex AI credentials unavailable.  On GCE the "
            "VM's service account must have roles/aiplatform.user; "
            "locally run `gcloud auth application-default login`."
        )

    url = (
        f"https://{_VERTEX_LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{_VERTEX_PROJECT}/locations/{_VERTEX_LOCATION}/"
        f"publishers/google/models/{_VERTEX_MODEL}:generateContent"
    )
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text":
                f"Search the web and return the {num_results} most "
                f"relevant results for: {query}\n\n"
                f"For each result provide:\n"
                f"- Title\n- URL\n- One-sentence summary\n"
                f"Cite the sources in the grounding metadata."
            }],
        }],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
    }

    # Retry on 429 (per-project quota) and transient 5xx with an
    # exponential backoff + small jitter. Parallel sub-agent bursts
    # regularly trip the Gemini-Flash per-minute cap — without this,
    # every fan-out drops 1-3 workers on the floor instead of
    # producing results a second later. Max ~8s total before giving
    # up; anything longer and the caller looks wedged.
    import random as _random
    _RETRYABLE = {429, 500, 502, 503, 504}
    _MAX_ATTEMPTS = 4
    backoff = 0.5

    resp = None
    async with httpx.AsyncClient(timeout=45) as client:
        for attempt in range(_MAX_ATTEMPTS):
            resp = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
            if resp.status_code not in _RETRYABLE:
                break
            if attempt == _MAX_ATTEMPTS - 1:
                break
            # Honor Retry-After when present, else exponential.
            retry_after = resp.headers.get("retry-after")
            try:
                wait_s = float(retry_after) if retry_after else backoff
            except ValueError:
                wait_s = backoff
            wait_s = min(wait_s, 4.0) + _random.uniform(0, 0.25)
            await asyncio.sleep(wait_s)
            backoff *= 2

    if resp is None or resp.status_code != 200:
        code = resp.status_code if resp is not None else "no-response"
        body = resp.text[:300] if resp is not None else ""
        return f"Vertex search error: {code} {body}"

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
                title = web.get("title", "Untitled")
                uri = web.get("uri", "")
                out.append(f"{i}. **{title}**\n   {uri}")
        break  # only the first candidate matters for grounding

    return "\n\n".join(out) if out else f"No results for: {query}"
