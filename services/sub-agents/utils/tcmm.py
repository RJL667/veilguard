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
