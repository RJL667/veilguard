"""
[OCR_GEMINI_2026-06-11] Mistral-OCR-shaped shim backed by Google AI Studio
(Gemini vision).

LibreChat's `azure_mistral_ocr` strategy makes exactly ONE call:
    POST {OCR_BASEURL}/ocr
    Authorization: Bearer {OCR_API_KEY}
    {model, image_limit, include_image_base64,
     document: {type: "image_url"|"document_url",
                image_url|document_url: "data:<mime>;base64,<payload>"}}
and expects the Mistral OCR response shape back:
    {pages: [{index, markdown, images, dimensions}], model,
     usage_info: {pages_processed, doc_size_bytes}, document_annotation}

We strip the data-URI prefix and hand the base64 payload straight to Gemini
as inline_data (no decode/re-encode round trip), ask for a verbatim
markdown transcription, and wrap it back in the Mistral shape.

Boundary note: the image/PDF BYTES go to AI Studio — the same trust zone
that already receives raw block text for TCMM NLP enrichment and web-search
grounding.  The extracted TEXT then rides LibreChat's normal context-file
path -> agent prompt -> this proxy's redaction -> TCMM, so downstream PII
handling is unchanged.

Unsupported mimetypes (docx/xlsx/...) raise OCRUnsupportedType so the
caller can 415 — LibreChat catches the failure and falls back to its
built-in document_parser, which already handles office formats.
"""

import asyncio
import logging
import os
import re

import httpx

logger = logging.getLogger("veilguard.ocr")

GEMINI_API_URL = os.environ.get(
    "GEMINI_API_URL", "https://generativelanguage.googleapis.com"
).rstrip("/")
# Same env fallback chain as the TCMM AIStudio NLP adapter.
_API_KEY = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
DEFAULT_MODEL = os.environ.get("GEMINI_OCR_MODEL", "gemini-2.5-flash")

# Gemini inline_data caps at ~20MB per request; leave headroom for the
# prompt + envelope.  Decoded-size estimate from base64 length.
MAX_DECODED_BYTES = int(os.environ.get("OCR_MAX_BYTES", str(18 * 1024 * 1024)))

# Mimetypes Gemini accepts as inline_data.  Everything else should fall
# back to LibreChat's document_parser (it handles docx/xlsx/epub natively).
_SUPPORTED_MIME = re.compile(
    r"^(image/(png|jpeg|jpg|gif|bmp|webp|tiff?|heic|heif)|application/pdf)$", re.I
)

_DATA_URI = re.compile(r"^data:(?P<mime>[\w.+/-]+);base64,(?P<b64>.+)$", re.S)

_OCR_PROMPT = (
    "Transcribe ALL text in this document/image, verbatim, as GitHub-flavored "
    "markdown. Preserve reading order, headings, lists, and table structure "
    "(render tables as markdown tables). Do not correct, translate, summarize, "
    "or omit anything. For charts, figures, or photos without text, include a "
    "one-line description in square brackets, e.g. [Figure: bar chart of "
    "monthly revenue]. If the document contains no readable text at all, "
    "describe what it shows in one short paragraph. Output ONLY the markdown "
    "content - no preamble, no commentary, no code fences around the whole "
    "output."
)


class OCRShimError(Exception):
    """Upstream (Gemini) failure or malformed request."""


class OCRUnsupportedType(OCRShimError):
    """Mimetype Gemini can't take inline — caller should let LibreChat
    fall back to its document_parser."""


def parse_document(document: dict) -> tuple[str, str]:
    """Extract (mime, base64_payload) from the Mistral-shape document obj."""
    if not isinstance(document, dict):
        raise OCRShimError("missing document object")
    doc_type = document.get("type") or "document_url"
    url = document.get("image_url") if doc_type == "image_url" else document.get("document_url")
    # Be lenient — some clients put the URI under the other key.
    url = url or document.get("image_url") or document.get("document_url") or ""
    if not isinstance(url, str) or not url:
        raise OCRShimError("document has no image_url/document_url")
    m = _DATA_URI.match(url)
    if not m:
        # We only accept data URIs — this shim never fetches remote URLs
        # (no SSRF surface, and LibreChat's azure strategy always inlines).
        raise OCRShimError("only data: URIs are supported")
    mime = m.group("mime").lower()
    b64 = m.group("b64").strip()
    if not _SUPPORTED_MIME.match(mime):
        raise OCRUnsupportedType(f"unsupported mimetype for Gemini OCR: {mime}")
    approx_bytes = (len(b64) * 3) // 4
    if approx_bytes > MAX_DECODED_BYTES:
        raise OCRShimError(
            f"file too large for OCR ({approx_bytes // (1024 * 1024)}MB > "
            f"{MAX_DECODED_BYTES // (1024 * 1024)}MB limit)"
        )
    return mime, b64


async def gemini_ocr(document: dict, model: str | None = None) -> dict:
    """Run Gemini vision OCR; return a Mistral-OCR-shaped response dict."""
    if not _API_KEY:
        raise OCRShimError("GOOGLE_API_KEY/GEMINI_API_KEY not configured on proxy")
    mime, b64 = parse_document(document)
    # LibreChat passes the yaml `mistralModel` through; ignore anything that
    # isn't a Gemini model id so a stale mistral-* default can't 404 us.
    use_model = (model or "").strip()
    if not use_model.startswith("gemini"):
        use_model = DEFAULT_MODEL

    url = f"{GEMINI_API_URL}/v1beta/models/{use_model}:generateContent"
    body = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": mime, "data": b64}},
                    {"text": _OCR_PROMPT},
                ]
            }
        ],
        "generationConfig": {"temperature": 0},
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": _API_KEY}

    last_err = "unknown"
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
        for attempt in range(3):
            try:
                resp = await client.post(url, json=body, headers=headers)
            except httpx.HTTPError as e:
                last_err = f"network error: {e}"
                await asyncio.sleep(2 * (attempt + 1))
                continue
            if resp.status_code == 200:
                data = resp.json()
                break
            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
            # 429 here is either rate limit or depleted prepaid credits —
            # same signature as the NLP adapter's billing incident.
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                await asyncio.sleep(2 * (attempt + 1))
                continue
            raise OCRShimError(f"Gemini OCR failed: {last_err}")
        else:
            raise OCRShimError(f"Gemini OCR failed after retries: {last_err}")

    candidates = data.get("candidates") or []
    parts = (candidates[0].get("content") or {}).get("parts") if candidates else None
    text = "".join(p.get("text", "") for p in (parts or [])).strip()
    if not text:
        block = (data.get("promptFeedback") or {}).get("blockReason")
        raise OCRShimError(
            f"Gemini returned no text (blockReason={block or 'none'})"
        )

    usage = data.get("usageMetadata") or {}
    logger.info(
        "[OCR] model=%s mime=%s in~%dKB out=%dc tokens=%s",
        use_model, mime, ((len(b64) * 3) // 4) // 1024, len(text),
        usage.get("totalTokenCount", "?"),
    )

    return {
        "pages": [
            {
                "index": 0,
                "markdown": text,
                "images": [],
                "dimensions": {"dpi": 0, "height": 0, "width": 0},
            }
        ],
        "model": use_model,
        "document_annotation": None,
        "usage_info": {
            "pages_processed": 1,
            "doc_size_bytes": (len(b64) * 3) // 4,
        },
        # [OCR_AUDIT_2026-06-11] internal metadata for the pii_audit row —
        # the /v1/ocr route pops this before returning to the caller.
        "_veilguard": {
            "mime": mime,
            "approx_bytes": (len(b64) * 3) // 4,
            "model": use_model,
            "chars": len(text),
            "prompt_tokens": usage.get("promptTokenCount"),
            "output_tokens": usage.get("candidatesTokenCount"),
        },
    }
