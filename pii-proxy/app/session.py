"""Session-scoped PII token mapping for rehydration.

Stores {conversation_id: {token: original_value}} with TTL eviction.
"""

import threading
import time
from typing import Optional


class PIISessionStore:
    """Thread-safe in-memory store for PII token mappings."""

    def __init__(self, ttl_seconds: int = 3600):
        self._store: dict[str, dict] = {}  # conv_id -> {"mapping": {}, "reverse": {}, "counters": {}, "ts": float}
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def _evict_expired(self):
        now = time.time()
        expired = [k for k, v in self._store.items() if now - v["ts"] > self._ttl]
        for k in expired:
            del self._store[k]

    def get_or_create(self, conversation_id: str) -> dict:
        with self._lock:
            self._evict_expired()
            if conversation_id not in self._store:
                self._store[conversation_id] = {
                    "mapping": {},      # token -> original
                    "reverse": {},      # original -> token
                    "counters": {},     # entity_type -> count
                    "ts": time.time(),
                }
            else:
                self._store[conversation_id]["ts"] = time.time()
            return self._store[conversation_id]

    def add_mapping(self, conversation_id: str, entity_type: str, original: str) -> str:
        """Add a PII mapping and return the token. Reuses existing token for same value.

        Case-insensitive for PERSON entities: "sarah" and "Sarah" get the same token.
        The original casing of the FIRST occurrence is preserved for rehydration.
        """
        session = self.get_or_create(conversation_id)

        # Case-insensitive lookup for person names
        lookup_key = original
        if entity_type == "PERSON":
            lookup_key = original.lower()

        # If we've seen this value before (case-insensitive for names), reuse its token
        if lookup_key in session["reverse"]:
            return session["reverse"][lookup_key]

        # Generate new token
        counter = session["counters"].get(entity_type, 0) + 1
        session["counters"][entity_type] = counter
        # Use a neutral format that won't trigger LLM safety filters
        short_type = entity_type.replace("_ADDRESS", "").replace("_NUMBER", "").replace("SA_", "")
        token = f"REF_{short_type}_{counter}"
        session["mapping"][token] = original
        session["reverse"][lookup_key] = token
        return token

    def rehydrate(self, conversation_id: str, text: str) -> str:
        """Replace all PII tokens in text with original values."""
        with self._lock:
            if conversation_id not in self._store:
                return text
            mapping = self._store[conversation_id]["mapping"]
        for token, original in mapping.items():
            text = text.replace(token, original)
        return text

    def get_mapping(self, conversation_id: str) -> Optional[dict]:
        with self._lock:
            if conversation_id not in self._store:
                return None
            return dict(self._store[conversation_id]["mapping"])


# Global singleton
pii_store = PIISessionStore()
