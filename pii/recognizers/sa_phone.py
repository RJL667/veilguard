"""South African phone number recognizer for Presidio.

Formats:
- +27 XX XXX XXXX
- 0XX XXX XXXX
- 27XXXXXXXXX
"""

from presidio_analyzer import Pattern, PatternRecognizer


class SouthAfricanPhoneRecognizer(PatternRecognizer):
    PATTERNS = [
        Pattern(
            "SA_PHONE_INTL",
            r"\b\+?27[\s-]?\d{2}[\s-]?\d{3}[\s-]?\d{4}\b",
            0.7,
        ),
        # Previously 0.5 — which is below the PIIRedactor.min_score
        # default (0.7). Presidio's context-word boost (``phone``,
        # ``cell``, etc.) can push 0.5 → 0.75, but only when the
        # context token shares a sentence with the number. In
        # structured JSON like ``{"phone": "0828002292"}`` the
        # tokenizer doesn't always carry the boost across the
        # punctuation, so the match stays at 0.5 and gets filtered.
        # Audit (Apr 23 2026) found 77 raw SA phone numbers leaking
        # through Petrus's Pipedrive traffic for exactly this reason.
        # Bumping to 0.7 so the pattern fires on its regex alone.
        # Regex is tight (10 digits, valid SA prefix 0[1-8]), so
        # false-positive risk is low. 10-digit Unix timestamps start
        # with 1, not 0, so they don't collide.
        Pattern(
            "SA_PHONE_LOCAL",
            r"\b0[1-8]\d[\s-]?\d{3}[\s-]?\d{4}\b",
            0.7,
        ),
    ]

    CONTEXT = [
        "phone", "cell", "mobile", "tel", "telephone", "contact",
        "whatsapp", "call", "sms", "number",
    ]

    def __init__(self):
        super().__init__(
            supported_entity="SA_PHONE_NUMBER",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="en",
            name="South African Phone Recognizer",
        )
