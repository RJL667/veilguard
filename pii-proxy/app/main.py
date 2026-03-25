"""
VEILGUARD PII Gateway
======================
Multi-LLM PII redaction gateway. Sits between LibreChat and any LLM API.

Routes:
  /anthropic/*  → https://api.anthropic.com/*
  /openai/*     → https://api.openai.com/*
  /gemini/*     → https://generativelanguage.googleapis.com/*

All user-authored content is scanned for PII before forwarding.
All responses are rehydrated (PII tokens → original values) before returning.

LibreChat config:
  ANTHROPIC_BASE_URL=http://pii-proxy:4000/anthropic
  OPENAI_BASE_URL=http://pii-proxy:4000/openai/v1
  (Google: custom endpoint with baseURL http://pii-proxy:4000/gemini)
"""

import json
import logging
import os
import sys
import uuid

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from .redactor import get_redactor
from .session import pii_store

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [VEILGUARD] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pii-proxy")

PORT = int(os.environ.get("PII_PROXY_PORT", "4000"))
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.7"))

# Backend routing table
BACKENDS = {
    "anthropic": os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com"),
    "openai": os.environ.get("OPENAI_API_URL", "https://api.openai.com"),
    "gemini": os.environ.get("GEMINI_API_URL", "https://generativelanguage.googleapis.com"),
}

app = FastAPI(title="Veilguard PII Gateway")


@app.on_event("startup")
async def startup():
    logger.info("Loading Presidio NLP models...")
    get_redactor(min_score=MIN_SCORE)
    logger.info("=" * 50)
    logger.info(f"Veilguard PII Gateway ready on port {PORT}")
    for name, url in BACKENDS.items():
        logger.info(f"  /{name}/* → {url}")
    logger.info("=" * 50)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "veilguard-pii-gateway",
        "presidio": "active",
        "backends": list(BACKENDS.keys()),
    }


def extract_conversation_id(data: dict, headers: dict) -> str:
    """Extract or generate a conversation ID for PII session tracking."""
    conv_id = (
        headers.get("x-conversation-id")
        or headers.get("x-request-id")
        or data.get("conversation_id")
        or data.get("parent_message_id")
    )
    return conv_id or str(uuid.uuid4())


def resolve_backend(path: str) -> tuple[str | None, str, str]:
    """Parse path to find backend and remaining path.

    Returns: (backend_url, remaining_path, backend_name)
    """
    parts = path.strip("/").split("/", 1)
    backend_name = parts[0].lower()
    remaining = parts[1] if len(parts) > 1 else ""

    if backend_name in BACKENDS:
        return BACKENDS[backend_name], remaining, backend_name

    return None, path, ""


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def gateway(request: Request, path: str):
    """Universal PII-redacting gateway.

    Routes /<backend>/... to the appropriate LLM API.
    Redacts PII from requests, rehydrates PII in responses.
    """
    backend_url, remaining_path, backend_name = resolve_backend(path)

    if not backend_url:
        return JSONResponse(
            {
                "error": f"Unknown backend: '{path.split('/')[0]}'. Available: {list(BACKENDS.keys())}",
                "usage": "Use /<backend>/... where backend is: anthropic, openai, gemini",
            },
            status_code=404,
        )

    target_url = f"{backend_url}/{remaining_path}" if remaining_path else backend_url

    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    redactor = get_redactor()
    is_stream = False
    conversation_id = None

    # Redact PII from JSON request body
    if body:
        content_type = headers.get("content-type", "")
        if "json" in content_type or body.strip()[:1] == b"{":
            try:
                data = json.loads(body)
                is_stream = data.get("stream", False)
                conversation_id = extract_conversation_id(data, headers)

                logger.info(
                    f">>> {request.method} [{backend_name}] /{remaining_path} "
                    f"(stream={is_stream}, conv={conversation_id[:8]}...)"
                )

                # Redact PII
                redacted = redactor.redact_json(data, conversation_id)
                body = json.dumps(redacted, ensure_ascii=False).encode("utf-8")
                headers["content-length"] = str(len(body))

            except json.JSONDecodeError:
                logger.info(f">>> {request.method} [{backend_name}] /{remaining_path} (non-json)")
        else:
            logger.info(f">>> {request.method} [{backend_name}] /{remaining_path}")

    if is_stream:
        # For streaming: create client that lives as long as the generator
        client = httpx.AsyncClient(timeout=300)
        req = client.build_request(
            method=request.method, url=target_url,
            content=body, headers=headers,
        )
        response = await client.send(req, stream=True)

        # If upstream returned an error, read the body and log it
        if response.status_code >= 400:
            error_body = await response.aread()
            await response.aclose()
            await client.aclose()
            logger.error(f"<<< [{backend_name}] {response.status_code}: {error_body[:500]}")
            return JSONResponse(
                json.loads(error_body) if error_body else {"error": f"Upstream {response.status_code}"},
                status_code=response.status_code,
            )

        async def stream_with_rehydration():
            """Stream through, rehydrating PII tokens in each chunk."""
            try:
                async for chunk in response.aiter_bytes():
                    if conversation_id:
                        try:
                            text = chunk.decode("utf-8")
                            text = redactor.rehydrate_text(text, conversation_id)
                            yield text.encode("utf-8")
                        except UnicodeDecodeError:
                            yield chunk
                    else:
                        yield chunk
            except (httpx.ReadError, httpx.RemoteProtocolError) as e:
                logger.warning(f"Stream ended: {e}")
            finally:
                try:
                    await response.aclose()
                    await client.aclose()
                except Exception:
                    pass

        resp_headers = {
            k: v for k, v in response.headers.items()
            if k.lower() not in ("content-encoding", "transfer-encoding", "content-length")
        }
        return StreamingResponse(
            stream_with_rehydration(),
            status_code=response.status_code,
            headers=resp_headers,
            media_type=response.headers.get("content-type", "text/event-stream"),
        )
    else:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.request(
                method=request.method, url=target_url,
                content=body if body else None, headers=headers,
            )

            logger.info(f"<<< [{backend_name}] {response.status_code} /{remaining_path}")

            # Rehydrate PII in response
            resp_body = response.content
            if conversation_id:
                try:
                    resp_text = resp_body.decode("utf-8")
                    resp_text = redactor.rehydrate_text(resp_text, conversation_id)
                    resp_body = resp_text.encode("utf-8")
                except UnicodeDecodeError:
                    pass

            resp_headers = {
                k: v for k, v in response.headers.items()
                if k.lower() not in ("content-encoding", "transfer-encoding", "content-length")
            }
            return StreamingResponse(
                iter([resp_body]),
                status_code=response.status_code,
                headers=resp_headers,
            )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
