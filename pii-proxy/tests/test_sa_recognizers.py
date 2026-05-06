"""Local smoke test for the SA recognizer set, including the new
SA_BANK_ACCOUNT recognizer. Run from repo root:

    python -m pytest pii-proxy/tests/test_sa_recognizers.py -v

Or just `python pii-proxy/tests/test_sa_recognizers.py` for a printed
report (no pytest required).

The cases exercise the three things we need to verify:

1. True positives — bank-account, ID, phone, card, IBAN, person — all
   get redacted to deterministic REF_* tokens.
2. Context-boost behaviour — bare 10-digit numbers redact only when a
   banking-context word is in the same string.
3. False-positive guards — VAT number, order ref, naked digits without
   banking context — must NOT be redacted.
"""
import sys, os, re, json
# Make the pii-proxy directory the package root so `app` is importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault(
    "PII_ALLOW_LIST_PATH",
    os.path.join(os.path.dirname(__file__), "..", "app", "allow_list.txt"),
)

from app.redactor import PIIRedactor, PII_ENTITIES  # noqa: E402

REDACTOR = PIIRedactor(min_score=0.7)
TOK_RE = re.compile(r"REF_[A-Z_]+_\d+")


def run(label, text, conv, expect_redact, expect_clean=()):
    out = REDACTOR.redact_text(text, conv)
    tokens = TOK_RE.findall(out)
    redacted_kinds = sorted({re.sub(r"_\d+$", "", t) for t in tokens})
    # Anything in expect_redact must appear in the output as some REF_ token,
    # with the matching expected entity kind.
    fails = []
    for kind in expect_redact:
        wanted = f"REF_{kind}_"
        if not any(t.startswith(wanted) for t in tokens):
            fails.append(f"missing {kind}")
    # Anything in expect_clean must STILL be in the output verbatim.
    for needle in expect_clean:
        if needle not in out:
            fails.append(f"unexpectedly redacted: {needle!r}")
    status = "PASS" if not fails else "FAIL"
    print(f"[{status}] {label}")
    print(f"   in : {text}")
    print(f"   out: {out}")
    print(f"   tokens: {tokens}  kinds: {redacted_kinds}")
    if fails:
        print(f"   issues: {fails}")
    print()
    return not fails


def main():
    print(f"Entities enabled: {PII_ENTITIES}")
    print(f"Allow_list size: {len(REDACTOR.allow_list)}")
    print()

    results = []

    # NOTE: token short names — the redactor strips "SA_" and
    # "_NUMBER"/"_ADDRESS" suffixes when minting tokens. So
    # SA_BANK_ACCOUNT → REF_BANK_ACCOUNT_*, SA_ID_NUMBER →
    # REF_ID_*, SA_PHONE_NUMBER → REF_PHONE_*. Test assertions use
    # the short forms.

    # === SA bank account — high-confidence brand patterns ===
    results.append(run(
        "FNB inline brand+number",
        "Pay R5000 to FNB account 62112233445 by Friday.",
        "t1",
        expect_redact=["BANK_ACCOUNT"],
    ))
    results.append(run(
        "ABSA with a/c shorthand",
        "ABSA a/c 4067123456 — debit Friday",
        "t2",
        expect_redact=["BANK_ACCOUNT"],
    ))
    results.append(run(
        "Capitec brand prefix",
        "My Capitec account is 1234567890.",
        "t3",
        expect_redact=["BANK_ACCOUNT"],
    ))
    results.append(run(
        "Standard Bank with space",
        "Standard Bank: 012345678",
        "t4",
        expect_redact=["BANK_ACCOUNT"],
    ))

    # === SA bank account — context-boost on bare digits ===
    results.append(run(
        "bare digits + 'bank account' context",
        "Please transfer R5000 to bank account 9876543210.",
        "t5",
        expect_redact=["BANK_ACCOUNT"],
    ))
    results.append(run(
        "bare digits + 'Account:' label",
        "Account: 1234567890. Branch: 250655.",
        "t6",
        expect_redact=["BANK_ACCOUNT"],
    ))

    # === SA bank account — false-positive guards ===
    results.append(run(
        "VAT number, no banking context",
        "VAT No: 4800270268 (Phishield UMA Pty Ltd)",
        "t7",
        expect_redact=[],
        expect_clean=["4800270268"],
    ))
    results.append(run(
        "order reference, no banking context",
        "Order #9876543210 confirmed for collection.",
        "t8",
        expect_redact=[],
        expect_clean=["9876543210"],
    ))
    results.append(run(
        "policy number, no banking context",
        "Policy 12345678901 was renewed last month.",
        "t9",
        expect_redact=[],
        expect_clean=["12345678901"],
    ))

    # === SA ID number ===
    # 8001015000086 — valid checksum (1980-01-01, seq 5000, citizen 0, A=8,
    # computed check digit 6). 8001015000088 was incorrect.
    results.append(run(
        "SA ID with valid Luhn",
        "ID: 8001015000086",
        "t10",
        expect_redact=["ID"],
    ))
    results.append(run(
        "SA ID with bad Luhn (must NOT redact)",
        "ID: 8001015000080",
        "t11",
        expect_redact=[],
        expect_clean=["8001015000080"],
    ))

    # === SA phone ===
    results.append(run(
        "SA mobile +27 international",
        "Call me on +27 82 800 2292",
        "t12",
        expect_redact=["PHONE"],
    ))
    results.append(run(
        "SA mobile local 0 prefix",
        "WhatsApp 0828002292",
        "t13",
        expect_redact=["PHONE"],
    ))

    # === Presidio built-ins still firing ===
    results.append(run(
        "Visa card (Luhn-valid)",
        "Charge to 4111111111111111",
        "t14",
        expect_redact=["CREDIT_CARD"],
    ))
    results.append(run(
        "IBAN (German)",
        "Pay to DE89370400440532013000 at Commerzbank.",
        "t15",
        expect_redact=["IBAN"],
    ))
    results.append(run(
        "Person name + email",
        "Email Sarah Connor at sarah@example.com to confirm.",
        "t16",
        expect_redact=["PERSON", "EMAIL"],
    ))

    # === Combined realistic spreadsheet-like row ===
    results.append(run(
        "combined PERSON + bank account in one line",
        "John Smith — FNB 62112233445 — debit order on the 1st.",
        "t17",
        expect_redact=["PERSON", "BANK_ACCOUNT"],
    ))

    # === Known structural limitation: bare value with key only ===
    results.append(run(
        "EXPECTED LIMITATION: JSON-style bare value",
        "62112233445",
        "t18",
        expect_redact=[],
        expect_clean=["62112233445"],
    ))

    passed = sum(results)
    total = len(results)
    print(f"=== {passed}/{total} passed ===")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
