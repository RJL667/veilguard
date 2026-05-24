"""CRLF + Windows-1252 text sanitizer.

Two modes:
  - "env"  → lossy ASCII rescue (em-dash → "--").  For .env values,
             config keys, constraint rule strings — anything where the
             raw bytes must round-trip to ASCII-only tooling.
  - "text" → Unicode-preserving rescue (em-dash → "—" / U+2014).  For
             markdown body content, system prompts, prose — anything
             where the original author intent matters.

Why this module exists: Veilguard has been bitten by CRLF + Windows-1252
smart-quote bytes in `.env` (twice).  The new `agents/CONSTITUTION.md`
and `agents/*.md` frontmatter are the same class of user-editable text.
Without this sanitizer at the load boundary, a single em-dash from a
copy-paste off Word will crash python-yaml or our regex parser at
startup → 503 cascade.

Contract:
  sanitize_bytes(raw, mode) -> (clean_text, issues)
  sanitize_text(s, mode)    -> (clean_text, issues)

  `issues` is a list of Issue records describing each substitution.
  Caller decides whether to log loudly, fail CI, or write_back the file.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class SanitizeMode(str, Enum):
    ENV = "env"
    TEXT = "text"


@dataclass(frozen=True)
class Issue:
    """One substitution / decode rescue.  Carries enough context for a
    helpful log line ("CRLF on line 3" / "smart quote at byte 142")."""

    code: str        # "crlf" | "bom" | "cp1252-smartquote" | "cp1252-emdash" | "cp1252-ellipsis" | "nul"
    detail: str
    byte_offset: int | None = None
    line: int | None = None


# ── Cp1252 rescue maps ───────────────────────────────────────────────────
# These are bytes that survive UTF-8 decode in 'strict' mode badly.
# Source: Windows-1252 codepage; commonly pasted from Word/Outlook.
#
# Two policy maps so the same sanitizer handles both env-value strings
# (ASCII targets) and prose text (proper Unicode codepoints).

_CP1252_RESCUE_ENV: dict[int, str] = {
    0x91: "'",   # left single quote
    0x92: "'",   # right single quote
    0x93: '"',   # left double quote
    0x94: '"',   # right double quote
    0x96: "-",   # en dash
    0x97: "--",  # em dash
    0x85: "...", # horizontal ellipsis
    0x82: "'",   # low single quote
    0xA0: " ",   # no-break space
}

_CP1252_RESCUE_TEXT: dict[int, str] = {
    0x91: "‘",  # ‘
    0x92: "’",  # ’
    0x93: "“",  # “
    0x94: "”",  # ”
    0x96: "–",  # –
    0x97: "—",  # —
    0x85: "…",  # …
    0x82: "‚",  # ‚
    0xA0: " ",       # nbsp → regular space; this is intentional, the
                     # invisible-space variant breaks parser regexes
                     # the same way 0xA0-as-raw-byte does.
}

_RESCUE_CODES: dict[int, str] = {
    0x91: "cp1252-smartquote",
    0x92: "cp1252-smartquote",
    0x93: "cp1252-smartquote",
    0x94: "cp1252-smartquote",
    0x96: "cp1252-emdash",
    0x97: "cp1252-emdash",
    0x85: "cp1252-ellipsis",
    0x82: "cp1252-smartquote",
    0xA0: "cp1252-nbsp",
}


def sanitize_bytes(
    raw: bytes,
    mode: Literal["env", "text"] | SanitizeMode,
) -> tuple[str, list[Issue]]:
    """Decode + normalise raw bytes.

    Algorithm:
      1. Strip UTF-8 BOM if present.
      2. Try UTF-8 strict decode.  If that fails, rescue cp1252 bytes
         per the chosen mode and try again, falling back to
         errors="replace" for anything unrecoverable (logs U+FFFD).
      3. Normalise line endings: CRLF + CR → LF.
      4. Strip NUL bytes (never legal in our config files).
    """
    if isinstance(mode, str):
        mode_str = mode
    else:
        mode_str = mode.value

    if mode_str not in ("env", "text"):
        raise ValueError(
            f"sanitize_bytes: mode must be 'env' or 'text', got {mode!r}"
        )

    rescue = _CP1252_RESCUE_ENV if mode_str == "env" else _CP1252_RESCUE_TEXT
    issues: list[Issue] = []

    # ── BOM ──────────────────────────────────────────────────────────────
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
        issues.append(Issue(code="bom", detail="UTF-8 BOM stripped", byte_offset=0))

    # ── Decode ───────────────────────────────────────────────────────────
    # First pass: strict UTF-8.  If that works the file is already clean
    # (no cp1252 surprises); we just need to normalise line endings.
    try:
        s = raw.decode("utf-8")
    except UnicodeDecodeError:
        # Per-byte rescue: any cp1252 byte we know about gets substituted.
        # Anything else falls through to 'replace' (U+FFFD).
        ba = bytearray(raw)
        out: bytearray = bytearray()
        i = 0
        while i < len(ba):
            b = ba[i]
            if b in rescue:
                replacement = rescue[b].encode("utf-8")
                out.extend(replacement)
                issues.append(Issue(
                    code=_RESCUE_CODES.get(b, "cp1252-rescue"),
                    detail=f"byte 0x{b:02X} → {rescue[b]!r}",
                    byte_offset=i,
                ))
            else:
                out.append(b)
            i += 1
        # Second decode attempt; we may still have isolated non-UTF-8
        # bytes that didn't match any rescue (rare but possible).
        s = bytes(out).decode("utf-8", errors="replace")

    # ── Line endings ────────────────────────────────────────────────────
    # CRLF and bare CR → LF.  These are silent killers in YAML / regex
    # parsers when the value ends with a stray \r that gets captured.
    if "\r\n" in s or "\r" in s:
        crlf_count = s.count("\r\n")
        bare_cr = s.count("\r") - crlf_count
        if crlf_count:
            issues.append(Issue(
                code="crlf",
                detail=f"{crlf_count} CRLF sequence(s) → LF",
            ))
        if bare_cr:
            issues.append(Issue(
                code="crlf",
                detail=f"{bare_cr} bare CR → LF",
            ))
        s = s.replace("\r\n", "\n").replace("\r", "\n")

    # ── NUL ─────────────────────────────────────────────────────────────
    if "\x00" in s:
        nul_count = s.count("\x00")
        issues.append(Issue(
            code="nul",
            detail=f"{nul_count} NUL byte(s) stripped (never legal in config)",
        ))
        s = s.replace("\x00", "")

    return s, issues


def sanitize_text(
    s: str,
    mode: Literal["env", "text"] | SanitizeMode = "text",
) -> tuple[str, list[Issue]]:
    """Sanitize an already-decoded string.  Same line-ending + NUL pass
    as sanitize_bytes, minus the decode step.

    Use this when a caller already has `str` (e.g. user-input from an
    HTTP body that FastAPI parsed) but wants the same line-ending +
    control-char guarantees.
    """
    issues: list[Issue] = []
    if "\r\n" in s or "\r" in s:
        crlf_count = s.count("\r\n")
        bare_cr = s.count("\r") - crlf_count
        if crlf_count:
            issues.append(Issue(code="crlf", detail=f"{crlf_count} CRLF → LF"))
        if bare_cr:
            issues.append(Issue(code="crlf", detail=f"{bare_cr} CR → LF"))
        s = s.replace("\r\n", "\n").replace("\r", "\n")
    if "\x00" in s:
        issues.append(Issue(code="nul", detail=f"{s.count(chr(0))} NUL stripped"))
        s = s.replace("\x00", "")
    return s, issues


__all__ = [
    "SanitizeMode",
    "Issue",
    "sanitize_bytes",
    "sanitize_text",
]
