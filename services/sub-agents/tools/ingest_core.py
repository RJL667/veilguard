"""Pure orchestration for file → document ingestion (ingest architecture
Phase 2 — the Veilguard ``file → text`` layer; see INGEST_ARCHITECTURE.md).

Stdlib-only and free of any sub-agents-stack imports, so it unit-tests
standalone:

  * ``kind_of`` / ``ext_of`` — format classification by extension
  * ``chunk_text``           — sentence-aware chunker (overlap, no deps)
  * ``expand``               — recursively flatten a file / zip / tar into
                               leaf ``(path, bytes)`` entries, applying
                               zip-bomb guards and returning the skips

The impure bits live in ``tools/host_docs.py``: the actual PDF/XLSX/OCR
extraction (fitz / openpyxl / the pii-proxy OCR shim) and the POST to
TCMM ``/ingest_text``. This module decides *what* to ingest and *how to
chunk it*; the host layer does the I/O.
"""
from __future__ import annotations

import io
import re
import tarfile
import zipfile
from dataclasses import dataclass

# ── Format classification ────────────────────────────────────────────────
# Archives we recurse into.
_ARCHIVE_EXTS = {"zip", "tar", "tgz", "tar.gz"}
# Formats we can turn into text with the libs the sub-agents host has
# (PyMuPDF / openpyxl / the OCR shim) plus plain decode.
_PDF_EXTS = {"pdf"}
_XLSX_EXTS = {"xlsx", "xlsm"}
_DOCX_EXTS = {"docx"}        # OOXML Word (python-docx); legacy .doc not covered
_PPTX_EXTS = {"pptx"}        # OOXML PowerPoint (python-pptx); legacy .ppt not covered
_IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff", "tif", "heic", "heif"}
_TEXT_EXTS = {
    "txt", "md", "markdown", "csv", "tsv", "json", "jsonl", "log", "xml",
    "html", "htm", "yaml", "yml", "ini", "cfg", "conf", "rst", "tex",
    "py", "js", "ts", "tsx", "jsx", "java", "go", "rs", "c", "h", "cpp",
    "sh", "sql", "toml", "env",
}
# Known document formats with NO parser installed in sub-agents — surfaced as
# "unsupported" (ask the user to export to PDF/DOCX) rather than silently
# dropped. .doc/.ppt are the LEGACY binary formats python-docx/-pptx can't read.
_UNSUPPORTED_EXTS = {"doc", "ppt", "epub", "odt", "ods", "odp", "rtf", "pages"}


def ext_of(name: str) -> str:
    """Lowercased final extension, with ``.tar.gz`` recognised as one unit."""
    base = name.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if base.endswith(".tar.gz"):
        return "tar.gz"
    return base.rsplit(".", 1)[-1] if "." in base else ""


def is_archive(name: str) -> bool:
    return ext_of(name) in _ARCHIVE_EXTS


def kind_of(name: str) -> str:
    """Coarse category the host extractor dispatches on:
    ``archive`` | ``pdf`` | ``xlsx`` | ``image`` | ``text`` | ``unsupported``."""
    e = ext_of(name)
    if e in _ARCHIVE_EXTS:
        return "archive"
    if e in _PDF_EXTS:
        return "pdf"
    if e in _XLSX_EXTS:
        return "xlsx"
    if e in _DOCX_EXTS:
        return "docx"
    if e in _PPTX_EXTS:
        return "pptx"
    if e in _IMAGE_EXTS:
        return "image"
    if e in _TEXT_EXTS:
        return "text"
    if e in _UNSUPPORTED_EXTS:
        return "unsupported"
    return "unsupported"


def is_skippable_entry(name: str) -> bool:
    """Archive junk we never ingest: directory markers, dotfiles, macOS
    resource forks."""
    norm = name.replace("\\", "/")
    if norm.endswith("/"):
        return True
    if "__MACOSX" in norm:
        return True
    base = norm.rsplit("/", 1)[-1]
    if not base or base.startswith("."):
        return True
    return False


# ── Chunking ─────────────────────────────────────────────────────────────
# Sentence boundary: end punctuation followed by whitespace + a capital /
# digit / opening quote. Good enough for prose without an NLP dependency.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def chunk_text(text: str, *, target_words: int = 300, overlap_words: int = 50) -> list:
    """Split ``text`` into overlapping, sentence-aware chunks.

    Packs whole sentences up to ``target_words`` (~400 tokens — within the
    embedding model's window) and carries ~``overlap_words`` of trailing
    context into the next chunk so a fact split across a boundary still
    retrieves. Returns ``[]`` for empty/whitespace input; never raises.
    """
    text = (text or "").strip()
    if not text:
        return []

    # Paragraph then sentence granularity.
    sentences = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        for s in _SENT_SPLIT.split(para):
            s = s.strip()
            if s:
                sentences.append(s)
    if not sentences:
        return []

    chunks = []
    cur, cur_words = [], 0
    for s in sentences:
        w = len(s.split())
        if cur and cur_words + w > target_words:
            chunks.append(" ".join(cur))
            # Carry the trailing ~overlap_words sentences into the next chunk.
            tail, tw = [], 0
            for ss in reversed(cur):
                tail.insert(0, ss)
                tw += len(ss.split())
                if tw >= overlap_words:
                    break
            cur, cur_words = list(tail), tw
        cur.append(s)
        cur_words += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# ── Recursive expansion (archives → leaf files), with guards ─────────────
@dataclass
class Limits:
    max_depth: int = 3                                  # nested-archive depth
    max_files: int = 500                                # leaf-file count
    max_total_uncompressed: int = 200 * 1024 * 1024     # cumulative bytes
    max_file_bytes: int = 25 * 1024 * 1024              # per-leaf cap
    max_ratio: int = 200                                # uncompressed/compressed bomb guard


def expand(name: str, data: bytes, limits: "Limits | None" = None):
    """Recursively flatten ``data`` (a file, or a zip/tar of files) into a
    flat list of leaf ``(path_in_archive, bytes)`` entries.

    Returns ``(entries, skips)`` where ``skips`` is a list of
    ``(path, reason)`` for everything dropped by a guard (oversized entry,
    suspicious compression ratio, depth/count/total caps, unreadable
    archive). Nested archives recurse up to ``limits.max_depth``. Pure: no
    extraction or network — the caller turns each leaf's bytes into text.
    """
    limits = limits or Limits()
    entries: list = []
    skips: list = []
    state = {"files": 0, "total": 0}
    _expand_into(name, data, 0, limits, state, entries, skips)
    return entries, skips


def _expand_into(name, data, depth, limits, state, entries, skips):
    if is_archive(name):
        if depth >= limits.max_depth:
            skips.append((name, f"max recursion depth {limits.max_depth} reached"))
            return
        try:
            members = _read_archive_members(data, limits)
        except ValueError as e:
            skips.append((name, f"unreadable archive: {e}"))
            return
        for inner_name, inner_bytes, note in members:
            full = f"{name}/{inner_name}"
            if note:
                skips.append((full, note))
                continue
            _expand_into(full, inner_bytes, depth + 1, limits, state, entries, skips)
        return

    # Leaf file.
    if state["files"] >= limits.max_files:
        skips.append((name, f"file-count cap ({limits.max_files}) reached"))
        return
    if len(data) > limits.max_file_bytes:
        skips.append((name, f"file too large ({len(data)} bytes > {limits.max_file_bytes})"))
        return
    if state["total"] + len(data) > limits.max_total_uncompressed:
        skips.append((name, "cumulative-size cap reached"))
        return
    state["files"] += 1
    state["total"] += len(data)
    entries.append((name, data))


def _read_archive_members(data, limits):
    """Return ``[(inner_name, inner_bytes_or_None, note)]`` for a zip/tar.
    ``note`` is "" for accepted entries, else a skip reason (bytes None).
    Raises ``ValueError`` if the bytes aren't a readable archive."""
    out = []
    bio = io.BytesIO(data)

    if zipfile.is_zipfile(bio):
        bio.seek(0)
        with zipfile.ZipFile(bio) as zf:
            for info in zf.infolist():
                if info.is_dir() or is_skippable_entry(info.filename):
                    continue
                if info.file_size > limits.max_file_bytes:
                    out.append((info.filename, None,
                                f"entry too large ({info.file_size} bytes)"))
                    continue
                # Zip-bomb guard: a tiny compressed entry that balloons.
                if info.compress_size > 0:
                    ratio = info.file_size / info.compress_size
                    if ratio > limits.max_ratio and info.file_size > 1_000_000:
                        out.append((info.filename, None,
                                    f"suspicious compression ratio {ratio:.0f}x (zip-bomb guard)"))
                        continue
                try:
                    out.append((info.filename, zf.read(info), ""))
                except Exception as e:
                    out.append((info.filename, None, f"read failed: {e}"))
        return out

    bio.seek(0)
    try:
        with tarfile.open(fileobj=bio, mode="r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile() or is_skippable_entry(m.name):
                    continue
                if m.size > limits.max_file_bytes:
                    out.append((m.name, None, f"entry too large ({m.size} bytes)"))
                    continue
                f = tf.extractfile(m)
                if f is None:
                    out.append((m.name, None, "unreadable tar member"))
                    continue
                out.append((m.name, f.read(), ""))
        return out
    except tarfile.TarError as e:
        raise ValueError(str(e))
