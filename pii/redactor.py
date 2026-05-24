"""PII redaction + rehydration — byte-stable, cache-safe.

Ported from agent-proxy/app/redactor.py with these changes:
  - Session store is Lance-backed (veilguard.pii.session_store) so
    pii-proxy AND agent-runtime see the same mapping for the same conv.
  - SessionId is now a typed key (tenant_id, conv_id) — not just a string.
  - Added `redact_blocks` + `rehydrate_blocks` for the post-TCMM-render
    pipeline (block-level redaction preserves cache_control markers).

Hard invariants (tested in tests/test_byte_stability.py):
  A. Same input + same SessionId → same output bytes
  B. Existing REF tokens NEVER change across turns
  C. Block dicts mutate only `.text`; cache_control / type / other
     keys are preserved unchanged
  D. No whitespace normalization, no Unicode normalization — pure
     N-byte substring replacement
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine

from .recognizers import (
    SouthAfricanBankAccountRecognizer,
    SouthAfricanIDRecognizer,
    SouthAfricanPhoneRecognizer,
)
from .session_store import SessionId, get_store as _get_store

logger = logging.getLogger("veilguard.pii.redactor")


# ── Config ──────────────────────────────────────────────────────────────


PII_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "PERSON",
    "IBAN_CODE",
    "IP_ADDRESS",
    "SA_ID_NUMBER",
    "SA_PHONE_NUMBER",
    "SA_BANK_ACCOUNT",
]


def _load_allow_list() -> list[str]:
    """Load brand/tech allow_list so Presidio doesn't redact `Docker`,
    `LibreChat`, etc. as PERSON.  spaCy en_core_web_lg false-positives
    those at ~0.85 confidence.

    Search order:
      1. ${PII_ALLOW_LIST_PATH} env override
      2. veilguard/pii/allow_list.txt (this dir)
      3. agent-proxy/app/allow_list.txt (compat with the old location)
    """
    candidates = [
        Path(__file__).parent / "allow_list.txt",
        Path(__file__).parent.parent / "agent-proxy" / "app" / "allow_list.txt",
    ]
    env_override = os.environ.get("PII_ALLOW_LIST_PATH", "")
    if env_override:
        candidates.insert(0, Path(env_override))
    for p in candidates:
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                terms = [
                    line.strip() for line in f
                    if line.strip() and not line.strip().startswith("#")
                ]
            logger.info(f"[pii] loaded {len(terms)} allow_list terms from {p}")
            return terms
    logger.warning("[pii] no allow_list.txt found — brand names may false-positive")
    return []


# ── Redactor ────────────────────────────────────────────────────────────


class PIIRedactor:
    """Presidio-backed analyzer + Lance-backed session mapping.

    Singleton per process.  Holds the analyzer state (model weights) so
    the per-request overhead is just the analyze() call.
    """

    _instance: Optional["PIIRedactor"] = None

    def __init__(self, min_score: float = 0.7):
        self.min_score = min_score

        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()
        registry.add_recognizer(SouthAfricanIDRecognizer())
        registry.add_recognizer(SouthAfricanPhoneRecognizer())
        registry.add_recognizer(SouthAfricanBankAccountRecognizer())

        self.analyzer = AnalyzerEngine(registry=registry)
        self.anonymizer = AnonymizerEngine()
        self.allow_list = _load_allow_list()
        self.store = _get_store()
        logger.info(
            f"[pii] redactor ready (Presidio + SA recognizers + "
            f"{len(self.allow_list)} allow_list terms)"
        )

    @classmethod
    def get(cls) -> "PIIRedactor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Text-level ────────────────────────────────────────────────────

    def redact_text(self, text: str, sid: SessionId) -> str:
        """Replace PII spans in `text` with REF tokens.

        Pure substring replacement — no whitespace normalization, no
        Unicode normalization.  Output bytes are deterministic given
        (text, sid).
        """
        if not text or len(text.strip()) < 5:
            return text

        try:
            results = self.analyzer.analyze(
                text=text,
                entities=PII_ENTITIES,
                language="en",
                score_threshold=self.min_score,
                allow_list=self.allow_list if self.allow_list else None,
            )
            if not results:
                return text

            # Don't re-redact our own REF tokens.
            results = [
                r for r in results
                if not text[r.start:r.end].startswith("REF_")
            ]
            if not results:
                return text

            # Replace end-to-start so later positions aren't shifted.
            results.sort(key=lambda r: r.start, reverse=True)

            redacted = text
            for r in results:
                original = text[r.start:r.end]
                token = self.store.add_mapping(sid, r.entity_type, original)
                redacted = redacted[:r.start] + token + redacted[r.end:]
            return redacted

        except Exception as e:
            logger.error(f"[pii] analyzer error (returning raw): {e}")
            return text

    def rehydrate_text(self, text: str, sid: SessionId) -> str:
        return self.store.rehydrate(sid, text)

    # ── Block-level (the new pipeline) ────────────────────────────────

    def redact_blocks(self, blocks: list[dict], sid: SessionId) -> list[dict]:
        """Redact each text block in place-shape, preserving structure.

        Each block is a dict like:
          {"type": "text", "text": "...", "cache_control": {...}}

        ONLY the `.text` field is mutated.  Every other key — type,
        cache_control, _veilguard, etc. — is copied through unchanged.
        This is the cache-stability contract: the block's structural
        hash (everything but text bytes) stays identical, so TCMM's
        cache_control placement still hits.

        Non-text blocks (images, tool_use, tool_result) are returned
        unchanged.
        """
        out: list[dict] = []
        for blk in blocks:
            if not isinstance(blk, dict):
                out.append(blk)
                continue
            btype = blk.get("type")
            if btype != "text":
                out.append(blk)
                continue
            text = blk.get("text", "")
            redacted = self.redact_text(text, sid) if isinstance(text, str) else text
            new_blk = dict(blk)        # shallow copy preserves all other keys
            new_blk["text"] = redacted
            out.append(new_blk)
        return out

    def rehydrate_blocks(self, blocks: list[dict], sid: SessionId) -> list[dict]:
        """Reverse of redact_blocks.  Also rehydrates tool_use input args
        (a tool call may have been invoked with REF tokens that need
        swapping back before dispatch sees them).
        """
        out: list[dict] = []
        for blk in blocks:
            if not isinstance(blk, dict):
                out.append(blk)
                continue
            btype = blk.get("type")
            new_blk = dict(blk)
            if btype == "text" and isinstance(blk.get("text"), str):
                new_blk["text"] = self.rehydrate_text(blk["text"], sid)
            elif btype == "tool_use" and isinstance(blk.get("input"), dict):
                new_blk["input"] = self._rehydrate_dict(blk["input"], sid)
            out.append(new_blk)
        return out

    def _rehydrate_dict(self, d: dict, sid: SessionId) -> dict:
        """Walk a dict recursively, rehydrating any string values."""
        out = {}
        for k, v in d.items():
            if isinstance(v, str):
                out[k] = self.rehydrate_text(v, sid)
            elif isinstance(v, dict):
                out[k] = self._rehydrate_dict(v, sid)
            elif isinstance(v, list):
                out[k] = [
                    self._rehydrate_dict(x, sid) if isinstance(x, dict)
                    else (self.rehydrate_text(x, sid) if isinstance(x, str) else x)
                    for x in v
                ]
            else:
                out[k] = v
        return out

    # ── Message-level (Anthropic-shape conversation) ──────────────────

    def redact_messages(
        self, messages: list[dict], sid: SessionId
    ) -> list[dict]:
        """Redact a list of {role, content} messages.

        Content may be a plain string OR a list of blocks.  Other dict
        keys (e.g. `name`, `tool_use_id`) are passed through.
        """
        out: list[dict] = []
        for m in messages:
            if not isinstance(m, dict):
                out.append(m)
                continue
            new_m = dict(m)
            content = m.get("content")
            if isinstance(content, str):
                new_m["content"] = self.redact_text(content, sid)
            elif isinstance(content, list):
                new_m["content"] = self.redact_blocks(content, sid)
            out.append(new_m)
        return out


# ── Module convenience ──────────────────────────────────────────────────


def get_redactor() -> PIIRedactor:
    return PIIRedactor.get()


__all__ = [
    "PIIRedactor",
    "get_redactor",
    "PII_ENTITIES",
]
