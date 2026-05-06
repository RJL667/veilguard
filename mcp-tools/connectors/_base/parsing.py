"""Shared file parsing + chunking for all connectors.

Connectors fetch document bytes from their source system (SharePoint
download, Slack file URL, Jira attachment, etc.) and need to extract
plain text + split it into chunks for TCMM ingest. Doing this per
connector means re-solving format-routing, parser quirks, and
chunking strategy for every new integration. We do it once here.

Implementation:

  * :func:`parse_to_text` wraps LlamaIndex's ``SimpleDirectoryReader``,
    which auto-routes by file extension to the right reader (PDF →
    ``PDFReader``, ``.docx`` → ``DocxReader``, ``.pptx`` →
    ``PptxReader``, ``.xlsx`` → ``PandasExcelReader``, ``.csv`` →
    ``CSVReader``, ``.html`` → ``HTMLTagReader``, ``.epub`` →
    ``EpubReader``, ``.ipynb`` → ``IPYNBReader``, …). The full list
    lives at https://docs.llamaindex.ai/en/stable/api_reference/readers/file/.

    A plain-text-family fallback (``.txt``, ``.md``, ``.csv`` when
    LlamaIndex's CSV reader isn't appropriate, anything matching
    ``mime_type=text/*``) decodes via UTF-8 with latin-1 fallback.

    Empty input returns ``""``. Truly unknown extensions return a
    human-readable placeholder so the connector still produces
    non-empty Content.text.

  * :func:`chunk_text` wraps LlamaIndex's ``SentenceSplitter`` —
    sentence-aware, token-bounded with overlap. Defaults
    (512 tokens / 64 overlap) match the chunk size most embedding
    models are trained on (BGE, MiniLM, mpnet) and align with
    typical RAG practice.

LlamaIndex is a hard dependency at runtime; install via the
connector's ``requirements.txt`` (``llama-index-core`` +
``llama-index-readers-file``). Imports are lazy so this module can
be imported in tests / contexts that don't actually call the parsers.
"""
from __future__ import annotations

import logging
import os
import tempfile


logger = logging.getLogger("connectors.parsing")


_TEXT_EXTENSIONS = frozenset(
    {"txt", "log", "yaml", "yml", "json", "xml"}
)


def _ext(filename: str) -> str:
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


def parse_to_text(
    raw: bytes,
    filename: str,
    *,
    mime_type: str | None = None,
) -> str:
    """Extract plain text from arbitrary file bytes.

    Returns ``""`` when ``raw`` is empty. Returns a human-readable
    placeholder for unknown formats so the caller can still emit
    a Content with non-empty text.
    """
    if not raw:
        return ""

    ext = _ext(filename)

    # Plain-text family — decode directly without invoking a parser.
    # LlamaIndex's flat-file reader works for these too, but we get
    # cleaner output via direct decode (preserves original whitespace).
    if ext in _TEXT_EXTENSIONS or (mime_type and mime_type.startswith("text/")):
        return _decode_text(raw)

    # Markdown / CSV: prefer LlamaIndex's structured readers — they do
    # the right thing with frontmatter and cell-level layout
    # respectively. For .md without frontmatter the result is
    # equivalent to a direct decode.
    return _parse_via_llama_index(raw, filename) or _placeholder(filename, ext)


def _decode_text(raw: bytes) -> str:
    """UTF-8 first, latin-1 lossless fallback. No errors='replace' —
    latin-1 maps every byte to a codepoint without loss, so we never
    introduce U+FFFD into otherwise-clean text."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _parse_via_llama_index(raw: bytes, filename: str) -> str:
    """Hand the bytes to LlamaIndex's SimpleDirectoryReader.

    SimpleDirectoryReader expects file paths, not bytes. We write a
    temp file, point it at that file, then delete the file regardless
    of outcome. The temp file inherits the original ``filename``'s
    extension so the reader's auto-routing kicks in.
    """
    try:
        from llama_index.core import SimpleDirectoryReader  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "llama-index-core not installed — install via the connector's "
            "requirements.txt to enable rich document parsing"
        )
        return ""

    ext = _ext(filename)
    suffix = f".{ext}" if ext else ""

    tmp_path: str | None = None
    try:
        # delete=False because Windows can't reopen a still-open
        # NamedTemporaryFile from another reader. We unlink in the
        # finally block instead.
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        try:
            documents = SimpleDirectoryReader(
                input_files=[tmp_path]
            ).load_data()
        except (ValueError, ImportError) as e:
            # ValueError: extension not supported by any registered reader.
            # ImportError: a specific reader's optional dep is missing
            # (e.g. python-pptx for .pptx). Fall back to the placeholder.
            logger.info(
                f"LlamaIndex could not parse {filename!r}: "
                f"{type(e).__name__}: {e}"
            )
            return ""
        except Exception as e:
            logger.warning(
                f"LlamaIndex failed on {filename!r}: "
                f"{type(e).__name__}: {e}"
            )
            return ""

        parts: list[str] = []
        for doc in documents:
            text = (doc.text or "").strip()
            if text:
                parts.append(text)
        return "\n\n".join(parts)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def chunk_text(
    text: str,
    *,
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 64,
) -> list[str]:
    """Split ``text`` into overlapping, sentence-aware chunks.

    Uses LlamaIndex's ``SentenceSplitter``. Defaults match the chunk
    size most modern embedding models are trained on (BGE, MiniLM,
    mpnet — all expect ~512-token windows). Overlap (~12% of chunk
    size) preserves cross-boundary context for retrieval.

    Returns ``[]`` for empty or whitespace-only input, never raises.
    Falls back to a paragraph split if LlamaIndex isn't installed
    (so this module remains usable in test contexts without the dep).
    """
    if not text or not text.strip():
        return []

    try:
        from llama_index.core.node_parser import (  # type: ignore[import-not-found]
            SentenceSplitter,
        )
    except ImportError:
        logger.warning(
            "llama-index-core not installed — falling back to paragraph split"
        )
        return _fallback_paragraph_split(text)

    try:
        splitter = SentenceSplitter(
            chunk_size=chunk_size_tokens,
            chunk_overlap=chunk_overlap_tokens,
        )
        return [c for c in splitter.split_text(text) if c.strip()]
    except Exception as e:
        logger.warning(
            f"SentenceSplitter failed ({type(e).__name__}: {e}) — "
            "falling back to paragraph split"
        )
        return _fallback_paragraph_split(text)


def _fallback_paragraph_split(text: str) -> list[str]:
    """Naive paragraph split — only used when LlamaIndex isn't available.
    Keeps the connector framework smoke-testable in environments without
    the heavy deps installed (e.g., the framework's own contract tests)."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


def _placeholder(filename: str, ext: str) -> str:
    """Human-readable placeholder for files we couldn't parse.

    Surfaces filename + extension to the LLM so it knows what's behind
    the link even when text extraction failed — better than empty
    string, which would look like an empty doc."""
    pretty_ext = ext or "unknown"
    return (
        f"[binary content — type={pretty_ext}, filename={filename!r} — "
        "text not extracted by any registered parser]"
    )
