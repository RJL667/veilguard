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
        Pattern(
            "SA_PHONE_LOCAL",
            r"\b0[1-8]\d[\s-]?\d{3}[\s-]?\d{4}\b",
            0.5,
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
