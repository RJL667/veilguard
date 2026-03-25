"""PII redaction and rehydration engine using Microsoft Presidio."""

import logging
from typing import Optional

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from .recognizers import SouthAfricanIDRecognizer, SouthAfricanPhoneRecognizer
from .session import pii_store

logger = logging.getLogger("pii-proxy")

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
USER_CONTENT_KEYS = {"text", "content", "query", "prompt", "input", "message", "value"}

# JSON keys to skip (metadata, not user content)
SKIP_KEYS = {
    "model", "type", "role", "stop_reason", "id", "object",
    "created", "usage", "index", "finish_reason", "stream",
    "max_tokens", "temperature", "top_p", "anthropic_version",
    "name", "source", "media_type", "cache_control", "tool_use_id",
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
        logger.info("Presidio engine ready with SA recognizers")

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
            )
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
