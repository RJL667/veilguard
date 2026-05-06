"""South African ID Number recognizer for Presidio.

SA ID format: YYMMDD SSSS C A Z (13 digits)
- YYMMDD: date of birth
- SSSS: sequence number (gender: 0000-4999 female, 5000-9999 male)
- C: citizenship (0=SA citizen, 1=permanent resident)
- A: usually 8 (was used for race classification, now deprecated)
- Z: Luhn checksum digit
"""

from presidio_analyzer import Pattern, PatternRecognizer


class SouthAfricanIDRecognizer(PatternRecognizer):
    PATTERNS = [
        Pattern(
            "SA_ID_13_DIGIT",
            r"\b(\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{4}[01]\d{2}\b",
            0.6,
        ),
    ]

    CONTEXT = [
        "id", "identity", "id number", "id no", "identification",
        "sa id", "south african", "rsa", "identity number",
        "id-number", "idnumber",
    ]

    def __init__(self):
        super().__init__(
            supported_entity="SA_ID_NUMBER",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="en",
            name="South African ID Recognizer",
        )

    def validate_result(self, pattern_text):
        """Validate using the SA-ID Luhn variant.

        Presidio treats ``True`` as confirmed, ``False`` as rejected,
        and ``None`` as "not validated — keep with reduced score". We
        used to return ``None`` on Luhn failure, which let bad-checksum
        13-digit numbers through whenever a context word ("id", "id
        no", etc.) was nearby — the +0.35 context boost pushed the
        unvalidated 0.6 base score above the 0.7 redact threshold.
        Returning ``False`` is the correct Presidio idiom for "this is
        not a real SA ID, drop the match."
        """
        if not pattern_text or len(pattern_text) != 13:
            return False

        try:
            digits = [int(d) for d in pattern_text]
        except ValueError:
            return False

        # SA-ID Luhn variant: sum odd-position digits, concatenate
        # even-position digits and double the resulting number, then
        # sum its digits. Check digit closes the total to a multiple
        # of 10.
        odd_sum = sum(digits[i] for i in range(0, 12, 2))
        even_str = "".join(str(digits[i]) for i in range(1, 12, 2))
        even_doubled = int(even_str) * 2
        even_sum = sum(int(d) for d in str(even_doubled))
        total = odd_sum + even_sum
        check = (10 - (total % 10)) % 10

        return check == digits[12]
