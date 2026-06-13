"""Veilguard Sub-Agents: TCMM memory integration."""

import httpx

from config import TCMM_URL


async def tcmm_recall(query: str, conversation_id: str = "") -> str:
    """Call TCMM to recall memory context for a query.

    Uses the active conversation ID from the MCP request headers (set by LibreChat
    via x-conversation-id). This ensures sub-agents recall from the parent chat's
    memory, not a random session — even with multiple concurrent users.
    """
    try:
        # Use provided ID, or get from MCP request state (set by x-conversation-id header)
        cid = conversation_id
        if not cid:
            from core import state
            cid = getattr(state, "active_conversation_id", "")

        if not cid:
            return ""

        async with httpx.AsyncClient(timeout=10) as client:
            # Use recall-only mode: don't create a new session or ingest the query.
            # Just recall from the existing conversation's archive.
            resp = await client.post(f"{TCMM_URL}/pre_request", json={
                "user_message": query,
                "conversation_id": cid,
                "recall_only": True,
            })
            if resp.status_code == 200:
                data = resp.json()
                prompt = data.get("prompt", "")
                if "--- MEMORY CONTEXT" in prompt and "--- END MEMORY CONTEXT ---" in prompt:
                    return prompt.split("--- END MEMORY CONTEXT ---")[0]
                elif "--- MEMORY CONTEXT" in prompt:
                    return prompt[:2000]
            return ""
    except Exception:
        return ""


async def ingest_text(
    *,
    chunks: list,
    conversation_id: str,
    user_id: str = "",
    title: str = "",
    source: str = "",
    channel: str = "team_knowledge",
    doc_id: str = "",
    provenance: dict | None = None,
) -> dict:
    """POST a pre-chunked document to TCMM ``/ingest_text`` (ingest
    architecture Phase 1 endpoint). Bulk-writes the chunks into the archive
    as a cohesive, recall-only knowledge unit — NOT a conversation turn.

    Returns the service response dict (``{"doc_id", "chunks_added", ...}``)
    or ``{"chunks_added": 0, "error": ...}`` on failure. Never raises.
    """
    if not chunks or not conversation_id:
        return {"chunks_added": 0, "error": "missing chunks or conversation_id"}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{TCMM_URL}/ingest_text", json={
                "conversation_id": conversation_id,
                "user_id": user_id,
                "doc_id": doc_id,
                "title": title,
                "source": source,
                "channel": channel,
                "chunks": chunks,
                "provenance": provenance or {},
            })
            if resp.status_code == 200:
                return resp.json()
            return {"chunks_added": 0, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"chunks_added": 0, "error": str(e)}
