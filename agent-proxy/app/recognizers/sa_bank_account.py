"""South African bank account number recognizer for Presidio.

SA bank accounts are 9-11 digits with no checksum, so a bare regex would
false-positive against VAT numbers, policy numbers, partial IDs, etc.
Two patterns:

1. SA_ACCT_BARE — 9-11 digits, low base score (0.45). Only crosses the
   PIIRedactor 0.7 threshold when Presidio's +0.35 context boost lands
   (a CONTEXT word in the same sentence). In structured payloads
   (e.g. spreadsheet rows where the column header is far from the
   value) the boost won't carry, and the match stays below threshold.
   That's the price of avoiding VAT-number false positives.

2. SA_ACCT_WITH_BANK — bank brand directly adjacent to a 9-11 digit
   run. Always fires (0.85). Bulletproof against false positives.

Caveat: free-floating account numbers in serialised tables won't be
caught. Spreadsheet data with a "Bank Account" column needs structural
handling (column-aware redaction) — out of scope here.
"""

from presidio_analyzer import Pattern, PatternRecognizer


class SouthAfricanBankAccountRecognizer(PatternRecognizer):
    PATTERNS = [
        Pattern(
            "SA_ACCT_WITH_BANK",
            r"(?i)\b(?:fnb|absa|nedbank|capitec|investec|standard\s*bank|"
            r"discovery\s*bank|tymebank|tyme\s*bank|bidvest|african\s*bank)"
            r"[\s\-:,]+(?:bank\s+)?(?:acc(?:ount|t)?|a\/c)?[\s\-:#]*"
            r"(\d{9,11})\b",
            0.85,
        ),
        Pattern(
            "SA_ACCT_BARE",
            r"\b\d{9,11}\b",
            0.45,
        ),
    ]

    CONTEXT = [
        "account", "acct", "a/c", "bank account", "bank acc",
        "bank", "branch", "branch code", "swift", "bic",
        "debit order", "eft", "payment", "deposit",
        "fnb", "absa", "nedbank", "capitec", "investec",
        "standard bank", "discovery bank", "tymebank", "bidvest",
    ]

    def __init__(self):
        super().__init__(
            supported_entity="SA_BANK_ACCOUNT",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="en",
            name="South African Bank Account Recognizer",
        )
