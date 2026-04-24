"""PII redaction and rehydration engine using Microsoft Presidio."""

import logging
import os
from pathlib import Path
from typing import List, Optional

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from .recognizers import SouthAfricanIDRecognizer, SouthAfricanPhoneRecognizer
from .session import pii_store

logger = logging.getLogger("pii-proxy")


def _load_allow_list() -> List[str]:
    """Load the Presidio allow_list from config/allow_list.txt (one term per line).

    Presidio's built-in ``allow_list`` parameter tells the analyzer to skip
    any span whose text matches a term on the list. We use it to suppress
    false-positive PERSON tags on known brand / product / technology names
    like ``Docker``, ``Veilguard``, ``LibreChat`` etc. The spaCy ``en_core_web_lg``
    model routinely classifies these as PERSON (0.85 confidence) because
    they look like surnames to the statistical NER.

    Without this list, Petrus's Pipedrive traffic redacts every ``Docker`` /
    ``Python`` / ``Pipedrive`` mention, destroying the LLM's ability to
    reason about technical topics.

    File format: one term per line, blank lines and ``#`` comments ignored.
    Case-sensitive (Presidio does exact-text match). Fall back to an empty
    list if the file is missing — all brand names will be redacted but
    the system still works.
    """
    # Search order (first hit wins):
    #   1. ${PII_ALLOW_LIST_PATH} env override
    #   2. /app/app/allow_list.txt — same dir as this module (mounted)
    #   3. /app/config/allow_list.txt — /app/config is NOT mounted
    #      but falls back to baked-in image if someone rebuilds with COPY
    #   4. ../config/allow_list.txt — dev-mode layout
    candidates = [
        Path(__file__).parent / "allow_list.txt",
        Path("/app/config/allow_list.txt"),
        Path(__file__).parent.parent / "config" / "allow_list.txt",
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
            logger.info(f"Loaded {len(terms)} allow_list terms from {p}")
            return terms

    logger.warning(
        "No allow_list.txt found — brand/tech names may be false-positive "
        "redacted as PERSON. Create config/allow_list.txt to fix."
    )
    return []

# Entity types to detect
PII_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "PERSON",
    "IBAN_CODE",
    "IP_ADDRESS",
    "SA_ID_NUMBER",
    "SA_PHONE_NUMBER",
]

# JSON keys containing user-authored content to scan
USER_CONTENT_KEYS = {"text", "content", "query", "prompt", "input", "message", "value", "system"}

# JSON keys to skip (metadata, not user content)
SKIP_KEYS = {
    "model", "type", "role", "stop_reason", "id", "object",
    "created", "usage", "index", "finish_reason", "stream",
    "max_tokens", "temperature", "top_p", "anthropic_version",
    "name", "source", "media_type", "cache_control", "tool_use_id",
    "signature", "thinking",  # Anthropic thinking blocks — signature invalidated if content changes
}


class PIIRedactor:
    def __init__(self, min_score: float = 0.7):
        self.min_score = min_score

        # Register custom SA recognizers
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()
        registry.add_recognizer(SouthAfricanIDRecognizer())
        registry.add_recognizer(SouthAfricanPhoneRecognizer())

        self.analyzer = AnalyzerEngine(registry=registry)
        self.anonymizer = AnonymizerEngine()
        # Load the allow_list once at init so per-request redact calls
        # don't pay the file IO. Reload requires a container restart —
        # acceptable because the list changes rarely.
        self.allow_list = _load_allow_list()
        logger.info(
            f"Presidio engine ready with SA recognizers + "
            f"{len(self.allow_list)} allow_list terms"
        )

    def redact_text(self, text: str, conversation_id: str) -> str:
        """Detect PII in text and replace with deterministic tokens."""
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

            # Filter out false positives: our own REF_ tokens should never be re-redacted
            results = [r for r in results if not text[r.start:r.end].startswith("REF_")]

            if not results:
                return text

            # Sort by start position descending so we can replace from end to start
            results.sort(key=lambda r: r.start, reverse=True)

            redacted = text
            for r in results:
                original = text[r.start:r.end]
                token = pii_store.add_mapping(conversation_id, r.entity_type, original)
                logger.info(f"  REDACTED {r.entity_type}: {original[:8]}... -> {token}")
                redacted = redacted[:r.start] + token + redacted[r.end:]

            return redacted

        except Exception as e:
            logger.error(f"Presidio error: {e}")
            return text

    def redact_json(self, obj, conversation_id: str, depth: int = 0, in_user_content: bool = False):
        """Recursively redact PII from JSON object."""
        if depth > 20:
            return obj

        if isinstance(obj, str):
            if in_user_content and len(obj) > 5:
                return self.redact_text(obj, conversation_id)
            return obj
        elif isinstance(obj, list):
            return [self.redact_json(item, conversation_id, depth + 1, in_user_content) for item in obj]
        elif isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if key in SKIP_KEYS:
                    result[key] = value
                elif key in USER_CONTENT_KEYS:
                    result[key] = self.redact_json(value, conversation_id, depth + 1, in_user_content=True)
                else:
                    result[key] = self.redact_json(value, conversation_id, depth + 1, in_user_content)
            return result

        return obj

    def rehydrate_text(self, text: str, conversation_id: str) -> str:
        """Replace PII tokens with original values."""
        return pii_store.rehydrate(conversation_id, text)


# Global singleton
redactor: Optional[PIIRedactor] = None


def get_redactor(min_score: float = 0.7) -> PIIRedactor:
    global redactor
    if redactor is None:
        redactor = PIIRedactor(min_score=min_score)
    return redactor
