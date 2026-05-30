"""veilguard.pii — shared PII redaction package.

Used by:
  - agent-proxy (was pii-proxy): redacts user-message before LLM call,
    rehydrates response before LibreChat sees it
  - agent-runtime: applies the same redaction inside Agent.run_turn's
    step 2, between TCMM render and the AnthropicGenerationAdapter call

Both consumers use the SAME SessionId (per-conv mapping) so the same
PII gets the same REF token regardless of who's asking — that's the
cache-stability property for Anthropic's prompt caching.

Public surface:
  SessionId           — typed key (tenant_id, conv_id) with parent-cid
                        resolution for sub-cids
  redactor()          — singleton PIIRedactor (Presidio + Lance store)
  PIIRedactor         — class with redact_text / redact_blocks /
                        rehydrate_text / rehydrate_blocks
"""

from .session_store import SessionId, PIISessionStore, get_store
from .redactor import (
    PIIRedactor,
    get_redactor,
    PII_ENTITIES,
    RedactionUnavailable,
)

__all__ = [
    "SessionId",
    "PIISessionStore",
    "get_store",
    "PIIRedactor",
    "get_redactor",
    "PII_ENTITIES",
    "RedactionUnavailable",
]
