"""Shared OCR client — calls the pii-proxy /v1/ocr Gemini-vision shim.

[OCR_GEMINI_2026-06-11] One implementation for every MCP server that needs
to read scanned PDFs / images: container servers (documents, image) reach
the proxy via compose DNS, host servers (sub-agents) via the published
port. The shim does the actual Gemini call (see pii-proxy/app/ocr_gemini.py)
so this stays stdlib-only like everything in _shared.

The extracted text returns to the agent prompt and flows through the
normal redaction pipeline — same boundary as the LibreChat OCR path.
"""

import base64
import json
import os
import urllib.error
import urllib.request

# Tried in order on connection failure (NOT on HTTP errors — those mean we
# reached the service and retrying another host won't help):
#   1. explicit override            (OCR_BASEURL)
#   2. compose-network DNS          (servers inside the api/proxy network)
#   3. published port on localhost  (host processes: sub-agents, forge)
#   4. Docker Desktop host gateway  (containers without compose DNS)
_CANDIDATES = [
    os.environ.get("OCR_BASEURL", "").rstrip("/"),
    "http://pii-proxy:4000/v1",
    "http://localhost:4000/v1",
    "http://host.docker.internal:4000/v1",
]


def ocr_bytes(data: bytes, mime: str, timeout: float = 180.0) -> str:
    """OCR a PDF or image; return the markdown transcription.

    Raises RuntimeError with an actionable message on failure.
    """
    api_key = os.environ.get("OCR_API_KEY", "")
    doc_key = "image_url" if mime.startswith("image/") else "document_url"
    payload = json.dumps(
        {
            "model": os.environ.get("GEMINI_OCR_MODEL", "gemini-2.5-flash"),
            "image_limit": 0,
            "include_image_base64": False,
            "document": {
                "type": doc_key,
                doc_key: f"data:{mime};base64," + base64.b64encode(data).decode(),
            },
        }
    ).encode()

    last = "no OCR endpoint configured"
    for base in dict.fromkeys(c for c in _CANDIDATES if c):
        req = urllib.request.Request(
            base + "/ocr",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = json.loads(resp.read())
            text = "\n\n".join(
                p.get("markdown", "") for p in out.get("pages", [])
            ).strip()
            if not text:
                raise RuntimeError("OCR service returned no text")
            return text
        except urllib.error.HTTPError as e:
            body = e.read()[:200]
            raise RuntimeError(
                f"OCR service error HTTP {e.code}: {body!r} "
                f"(429/502 can mean AI Studio prepaid credits are depleted)"
            ) from None
        except (urllib.error.URLError, OSError) as e:
            last = f"{base}: {e}"
            continue
    raise RuntimeError(
        f"OCR service unreachable (last attempt {last}). "
        f"Is the pii-proxy running and OCR_BASEURL/OCR_API_KEY set?"
    )
