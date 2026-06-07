"""Passport number recognizer for Presidio.

Passport numbers have no universal checksum and look like generic
alphanumeric reference codes, so this recognizer is CONTEXT-GATED: the
base pattern score (0.4) only clears the 0.7 redact threshold when a
"passport" context word is nearby (+0.35 boost). This biases toward
redaction when the text is clearly about a passport, while avoiding
false-positives on order numbers / reference codes elsewhere.

Covers SA (1 letter + 8 digits, e.g. M00012345) and common international
formats (1-2 letters + 6-9 digits). No checksum exists, so there is no
``validate_result`` — context is the precision mechanism.
"""

from presidio_analyzer import Pattern, PatternRecognizer


class PassportRecognizer(PatternRecognizer):
    PATTERNS = [
        Pattern(
            "PASSPORT_ALNUM",
            r"\b[A-Za-z]{1,2}\d{6,9}\b",
            0.4,
        ),
    ]

    CONTEXT = [
        "passport", "passport no", "passport number", "passport #",
        "passport-no", "passportno", "travel document", "passport details",
    ]

    def __init__(self):
        super().__init__(
            supported_entity="PASSPORT_NUMBER",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="en",
            name="Passport Recognizer",
        )
