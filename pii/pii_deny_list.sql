-- pii_deny_list: the LLM-fed BLACKLIST of high-confidence missed PII.
-- Terms here are FORCE-redacted by the boundary redactor even when the default
-- Presidio recognizers miss them (the safe direction: over-redacting is a
-- quality bug, not a leak, so the LLM's flags auto-apply with no confirm gate).
-- Symmetric to pii_allow_list (the NLP-fed, gated whitelist).
-- The redactor (pii/redactor.py _load_deny_list) reads active terms and builds
-- a Presidio deny-list recognizer (score 1.0), refreshed on the same TTL daemon.
CREATE TABLE IF NOT EXISTS pii_deny_list (
    term        text PRIMARY KEY,
    entity_type text,                       -- the LLM's PII type (PERSON/ACCOUNT/...) — informational
    source      text,                       -- 'llm-audit' | 'manual' | ...
    added_by    text,
    added_at    timestamptz DEFAULT now(),
    active      boolean DEFAULT true
);
CREATE INDEX IF NOT EXISTS pii_deny_list_active_idx ON pii_deny_list (active);
