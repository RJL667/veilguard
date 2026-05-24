"""Unit tests for utils.text_sanitize.

The sanitizer is the front-line defense against CRLF + Windows-1252
smart-quote crashes in user-editable config files (CONSTITUTION.md,
agents/*.md frontmatter).  Memory documents two prior .env incidents;
this is the test suite that prevents the third.
"""

from app.utils.text_sanitize import (
    sanitize_bytes,
    sanitize_text,
    SanitizeMode,
)


class TestSanitizeBytes:
    def test_plain_utf8_is_unchanged(self):
        raw = b"hello world\n"
        clean, issues = sanitize_bytes(raw, mode="env")
        assert clean == "hello world\n"
        assert issues == []

    def test_crlf_normalized_to_lf(self):
        raw = b"line1\r\nline2\r\nline3\n"
        clean, issues = sanitize_bytes(raw, mode="env")
        assert clean == "line1\nline2\nline3\n"
        assert any(i.code == "crlf" for i in issues)

    def test_bare_cr_normalized_to_lf(self):
        raw = b"line1\rline2\n"
        clean, issues = sanitize_bytes(raw, mode="env")
        assert clean == "line1\nline2\n"
        assert any(i.code == "crlf" for i in issues)

    def test_utf8_bom_stripped(self):
        raw = b"\xef\xbb\xbfhello"
        clean, issues = sanitize_bytes(raw, mode="env")
        assert clean == "hello"
        assert any(i.code == "bom" for i in issues)

    def test_cp1252_emdash_env_mode_lossy(self):
        # 0x97 (cp1252 em-dash) standalone → invalid UTF-8.
        raw = b"a \x97 b"
        clean, issues = sanitize_bytes(raw, mode="env")
        assert clean == "a -- b"
        assert any(i.code == "cp1252-emdash" for i in issues)

    def test_cp1252_emdash_text_mode_unicode(self):
        raw = b"a \x97 b"
        clean, issues = sanitize_bytes(raw, mode="text")
        assert clean == "a — b"
        assert any(i.code == "cp1252-emdash" for i in issues)

    def test_cp1252_smartquotes_env_mode(self):
        raw = b"\x91hello\x92"  # left + right single quote
        clean, issues = sanitize_bytes(raw, mode="env")
        assert clean == "'hello'"

    def test_cp1252_smartquotes_text_mode_preserves_unicode(self):
        raw = b"\x91hello\x92"
        clean, issues = sanitize_bytes(raw, mode="text")
        assert "‘" in clean and "’" in clean

    def test_nul_byte_stripped(self):
        raw = b"a\x00b\x00c\n"
        clean, issues = sanitize_bytes(raw, mode="env")
        assert clean == "abc\n"
        assert any(i.code == "nul" for i in issues)

    def test_bom_then_cp1252_combination(self):
        # Notepad sometimes does this: BOM + smart quote in body.
        raw = b"\xef\xbb\xbfhello \x97 world"
        clean, issues = sanitize_bytes(raw, mode="env")
        assert clean == "hello -- world"
        codes = {i.code for i in issues}
        assert "bom" in codes
        assert "cp1252-emdash" in codes

    def test_invalid_mode_raises(self):
        try:
            sanitize_bytes(b"x", mode="bogus")  # type: ignore[arg-type]
        except ValueError as e:
            assert "mode" in str(e)
        else:
            raise AssertionError("expected ValueError on invalid mode")

    def test_sanitize_mode_enum_accepted(self):
        clean, _ = sanitize_bytes(b"x\r\n", mode=SanitizeMode.ENV)
        assert clean == "x\n"

    def test_idempotent_on_clean_input(self):
        raw = b"already clean utf-8\n"
        once, _ = sanitize_bytes(raw, mode="env")
        twice, _ = sanitize_bytes(once.encode("utf-8"), mode="env")
        assert once == twice


class TestSanitizeText:
    def test_string_crlf_normalized(self):
        clean, issues = sanitize_text("a\r\nb\r\n", mode="text")
        assert clean == "a\nb\n"
        assert any(i.code == "crlf" for i in issues)

    def test_string_nul_stripped(self):
        clean, issues = sanitize_text("a\x00b", mode="text")
        assert clean == "ab"

    def test_clean_string_passthrough(self):
        clean, issues = sanitize_text("hello\n", mode="text")
        assert clean == "hello\n"
        assert issues == []
