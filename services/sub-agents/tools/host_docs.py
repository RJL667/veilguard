"""Host document parsers — read + extract text from documents that live on
the sub-agents host (the user's Windows machine in a local deployment).

Why this exists: the container-based ``documents`` MCP server (read_pdf etc.)
runs in Linux and CANNOT see Windows paths like
``C:\\Users\\rudol\\Downloads\\report.pdf``. The sub-agents service runs on
the host and has PyMuPDF (fitz), so it can read + parse those files directly.
This closes the gap that previously forced the model to FORGE a one-off
pdf-extraction tool every time.

[HOST_DOC_READ_2026_06_01]  Scope: reads are allowed anywhere under the
user's home dir (covers Downloads / Documents / Desktop) — host-exec/daemon
already expose comparable host access. Override the root with
HOST_DOC_READ_ROOT. These are READ-ONLY parsers; no writes.
"""

import asyncio
import os
import re
import sys
from pathlib import Path

# [OCR_GEMINI_2026-06-11] shared OCR client (pii-proxy Gemini shim) — same
# 2-line _shared import shim the container servers use; resolves to
# services/_shared locally and mcp-tools/_shared on the VM.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from ocr_client import ocr_bytes  # noqa: E402

_DOC_READ_ROOT = os.environ.get("HOST_DOC_READ_ROOT", os.path.expanduser("~"))

# [VM_PATH_GUARD_2026-06-12] On the cloud VM this service runs on Linux, but
# the model — seeing files like 'C:\Users\...\x.pdf' mentioned in chat —
# calls these tools with Windows paths. Path("C:/...") is NOT absolute on
# POSIX, so it used to get joined onto $HOME and produce nonsense like
# /home/rudol/C:/Users/... The guard converts that into an instruction the
# model can act on instead of retrying path spellings forever.
_WIN_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


def _reject_foreign_windows_path(path: str, _posix: bool = (os.name != "nt")):
    """Non-empty error string if `path` is a Windows drive path on Linux."""
    if _posix and _WIN_DRIVE.match(path.strip()):
        return (
            f"Error: {path!r} is a Windows path, but this tool runs on the "
            "Veilguard SERVER (Linux) and cannot see the user's local drive. "
            "Do NOT retry other path spellings — no path will work. To READ "
            "a file from the user's machine, ask the user to attach it in "
            "chat via the '+' menu -> 'Upload as Text' (it is OCR'd and its "
            "full text arrives in the conversation). To WRITE a file, use a "
            "server-side path (relative to home) and tell the user where it "
            "was saved."
        )
    return ""


def _resolve_read(path: str):
    """Return (resolved_path, "") or (None, error_str)."""
    if not isinstance(path, str) or not path.strip():
        return None, "Error: path is required"
    if err := _reject_foreign_windows_path(path):
        return None, err
    p = Path(path.replace("\\", "/"))
    if not p.is_absolute():
        p = Path(_DOC_READ_ROOT) / p
    rp = p.resolve()
    root = Path(_DOC_READ_ROOT).resolve()
    if not str(rp).startswith(str(root)):
        return None, (
            f"Error: read denied — {path!r} is outside the allowed root "
            f"({root}). Set HOST_DOC_READ_ROOT to widen."
        )
    if not rp.exists():
        return None, f"Error: File not found: {rp}"
    if not rp.is_file():
        return None, f"Error: Not a file: {rp}"
    return rp, ""


def _resolve_write(path: str):
    """Resolve a path for WRITING a document on the host. Same root rules as
    reads, but the file need not exist; parent dirs are created."""
    if not isinstance(path, str) or not path.strip():
        return None, "Error: path is required"
    if err := _reject_foreign_windows_path(path):
        return None, err
    p = Path(path.replace("\\", "/"))
    if not p.is_absolute():
        p = Path(_DOC_READ_ROOT) / p
    rp = p.resolve()
    root = Path(_DOC_READ_ROOT).resolve()
    if not str(rp).startswith(str(root)):
        return None, (
            f"Error: write denied — {path!r} is outside the allowed root "
            f"({root}). Set HOST_DOC_READ_ROOT to widen."
        )
    try:
        rp.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return None, f"Error: cannot create parent dir: {e}"
    return rp, ""


def register(mcp):
    @mcp.tool()
    async def read_pdf(path: str, pages: str = "") -> str:
        """Read and extract text from a PDF on the machine RUNNING this
        sub-agents service.

        LOCAL install: that is the user's Windows machine, so real Windows
        paths work (e.g. 'C:/Users/<name>/Downloads/report.pdf'); forward or
        back slashes both work. CLOUD/VM deployment: that is the Veilguard
        server — the user's local files are NOT reachable by any path. If the
        user mentions a file on THEIR machine, do not guess paths; ask them to
        attach it in chat ('+' menu -> 'Upload as Text') and the text will be
        OCR'd into the conversation.

        Args:
            path: PDF path (absolute, or relative to the service home dir).
            pages: Page range like "1-5" or "3". Empty = all pages.
        """
        rp, err = _resolve_read(path)
        if err:
            return err
        try:
            import fitz  # PyMuPDF
        except Exception as e:
            return f"Error: PDF parser (PyMuPDF) unavailable on host: {e}"
        try:
            doc = fitz.open(str(rp))
        except Exception as e:
            return f"Error opening PDF: {e}"
        try:
            total = len(doc)
            if pages.strip():
                parts = pages.strip().split("-")
                start = max(0, int(parts[0]) - 1)
                end = int(parts[-1]) if len(parts) > 1 else start + 1
                rng = range(start, min(end, total))
            else:
                rng = range(total)
            out = [f"# {rp.name} ({total} pages, reading {len(rng)})\n"]
            text_chars = 0
            for i in rng:
                text = doc[i].get_text().strip()
                text_chars += len(text)
                out.append(f"--- Page {i + 1} ---\n{text}\n")
            # [OCR_GEMINI_2026-06-11] No text layer => scanned/image PDF.
            # OCR the selected pages via the pii-proxy Gemini shim instead
            # of returning empty pages. to_thread keeps the blocking HTTP
            # call off the server's event loop.
            if rng and text_chars < 25 * len(rng):
                if len(rng) >= total:
                    data = rp.read_bytes()
                else:
                    sub = fitz.open()
                    sub.insert_pdf(doc, from_page=rng[0], to_page=rng[-1])
                    data = sub.tobytes()
                    sub.close()
                try:
                    md = await asyncio.to_thread(ocr_bytes, data, "application/pdf")
                    return (
                        f"# {rp.name} ({total} pages — no text layer on the "
                        f"selected {len(rng)} page(s); OCR'd via Gemini "
                        f"vision)\n\n{md}"
                    )
                except Exception as e:
                    out.append(
                        f"\n[Note: these pages have no text layer (likely a "
                        f"scan) and the OCR fallback failed: {e}]"
                    )
            return "\n".join(out)
        except Exception as e:
            return f"Error reading PDF: {e}"
        finally:
            doc.close()

    @mcp.tool()
    async def read_xlsx(path: str, sheet: str = "", max_rows: int = 500) -> str:
        """Read an Excel spreadsheet on the machine RUNNING this sub-agents
        service, as text.

        LOCAL install: the user's Windows machine (real Windows paths work).
        CLOUD/VM deployment: the Veilguard server — the user's local files
        are NOT reachable; ask the user to attach the file in chat instead
        of guessing paths.

        Args:
            path: XLSX path (absolute, or relative to the service home dir).
            sheet: Sheet name. Empty = first/active sheet.
            max_rows: Max rows to return (default 500).
        """
        rp, err = _resolve_read(path)
        if err:
            return err
        try:
            from openpyxl import load_workbook
        except Exception as e:
            return f"Error: Excel parser (openpyxl) unavailable on host: {e}"
        try:
            wb = load_workbook(str(rp), read_only=True, data_only=True)
            ws = wb[sheet] if sheet and sheet in wb.sheetnames else wb.active
            if sheet and sheet not in wb.sheetnames:
                return f"Error: Sheet {sheet!r} not found. Available: {wb.sheetnames}"
            out = [f"# {rp.name} — Sheet: {ws.title}", f"# Sheets: {wb.sheetnames}\n"]
            n = 0
            for row in ws.iter_rows(values_only=True):
                if n >= max_rows:
                    out.append(f"\n... truncated at {max_rows} rows")
                    break
                out.append(" | ".join(str(c) if c is not None else "" for c in row))
                n += 1
            wb.close()
            return "\n".join(out)
        except Exception as e:
            return f"Error reading XLSX: {e}"

    @mcp.tool()
    async def create_xlsx(path: str, data: str, sheet_name: str = "Sheet1") -> str:
        """Create an Excel spreadsheet on the machine RUNNING this
        sub-agents service.

        LOCAL install: writes to the user's Windows machine (e.g.
        'C:/Users/<name>/Downloads/report.xlsx'). CLOUD/VM deployment: writes
        land on the Veilguard SERVER, not the user's machine — use a path
        relative to home and tell the user where the file was saved. The
        container 'documents' create_xlsx only writes inside its /workspace.

        Args:
            path: Output XLSX path (absolute, or relative to the service home dir).
            data: JSON string — a list of dicts [{"col":"val",...}] OR a list of
                  lists [["h1","h2"],["v1","v2"]] (first row = header).
            sheet_name: Worksheet name.
        """
        import json as _json
        rp, err = _resolve_write(path)
        if err:
            return err
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font
        except Exception as e:
            return f"Error: Excel writer (openpyxl) unavailable on host: {e}"
        try:
            parsed = _json.loads(data)
        except Exception as e:
            return f"Error: `data` is not valid JSON: {e}"
        try:
            wb = Workbook()
            ws = wb.active
            ws.title = sheet_name
            if not parsed:
                wb.save(str(rp))
                return f"Created empty XLSX: {rp}"
            if isinstance(parsed[0], dict):
                headers = list(parsed[0].keys())
                ws.append(headers)
                for cell in ws[1]:
                    cell.font = Font(bold=True)
                for r in parsed:
                    ws.append([r.get(h, "") for h in headers])
            elif isinstance(parsed[0], (list, tuple)):
                for i, r in enumerate(parsed):
                    ws.append(list(r))
                    if i == 0:
                        for cell in ws[1]:
                            cell.font = Font(bold=True)
            for col in ws.columns:
                mx = max((len(str(c.value or "")) for c in col), default=0)
                ws.column_dimensions[col[0].column_letter].width = min(mx + 2, 50)
            wb.save(str(rp))
            return f"Created XLSX {rp} ({ws.max_row} rows, sheet {sheet_name!r})"
        except Exception as e:
            return f"Error creating XLSX: {e}"

    @mcp.tool()
    async def create_pdf(path: str, content: str, title: str = "") -> str:
        """Create a PDF document on the machine RUNNING this sub-agents
        service.

        LOCAL install: writes to the user's Windows machine (e.g.
        'C:/Users/<name>/Downloads/report.pdf'). CLOUD/VM deployment: writes
        land on the Veilguard SERVER, not the user's machine — use a path
        relative to home and tell the user where the file was saved. Runs
        with reportlab; PDF counterpart to create_xlsx.

        Args:
            path: Output PDF path (absolute, or relative to the service home dir).
            content: Body as light markdown, parsed per line: '# '/'## '/'### '
                     headings, '- ' or '* ' bullets, blank line = spacer,
                     '**bold**' inline; anything else = a paragraph.
            title: Optional title rendered as a large heading on page 1.
        """
        import html as _html
        import re as _re2
        rp, err = _resolve_write(path)
        if err:
            return err
        if rp.suffix.lower() != ".pdf":
            rp = rp.with_suffix(".pdf")
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.lib import colors
            from reportlab.platypus import (SimpleDocTemplate, Paragraph,
                                            Spacer, HRFlowable)
        except Exception as e:
            return f"Error: PDF writer (reportlab) unavailable on host: {e}"

        def _inline(s: str) -> str:
            s = _html.escape(s, quote=False)
            return _re2.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)

        styles = getSampleStyleSheet()
        body = ParagraphStyle("CRBody", parent=styles["BodyText"], fontSize=10.5, leading=15, spaceAfter=6)
        h1 = ParagraphStyle("CRH1", parent=styles["Heading1"], fontSize=16, spaceBefore=10, spaceAfter=6)
        h2 = ParagraphStyle("CRH2", parent=styles["Heading2"], fontSize=13, spaceBefore=8, spaceAfter=4)
        h3 = ParagraphStyle("CRH3", parent=styles["Heading3"], fontSize=11.5, spaceBefore=6, spaceAfter=3)
        bul = ParagraphStyle("CRBul", parent=body, leftIndent=14, bulletIndent=2, spaceAfter=3)
        ttl = ParagraphStyle("CRTitle", parent=styles["Title"], fontSize=22, spaceAfter=4)

        story = []
        if title.strip():
            story.append(Paragraph(_inline(title.strip()), ttl))
            story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#444444"), spaceAfter=10))
        for raw in (content or "").replace("\r\n", "\n").split("\n"):
            s = raw.strip()
            if not s:
                story.append(Spacer(1, 6)); continue
            if s.startswith("### "):
                story.append(Paragraph(_inline(s[4:]), h3))
            elif s.startswith("## "):
                story.append(Paragraph(_inline(s[3:]), h2))
            elif s.startswith("# "):
                story.append(Paragraph(_inline(s[2:]), h1))
            elif s[:2] in ("- ", "* "):
                story.append(Paragraph(_inline(s[2:]), bul, bulletText="•"))
            else:
                story.append(Paragraph(_inline(s), body))
        if not story:
            story.append(Paragraph("(empty document)", body))

        def _footer(canvas, doc_):
            canvas.saveState()
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.HexColor("#888888"))
            canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc_.page}")
            canvas.restoreState()

        try:
            doc = SimpleDocTemplate(str(rp), pagesize=A4, leftMargin=2 * cm,
                                    rightMargin=2 * cm, topMargin=2 * cm,
                                    bottomMargin=2 * cm, title=(title.strip() or rp.stem))
            doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
        except Exception as e:
            return f"Error creating PDF: {e}"
        try:
            size = rp.stat().st_size
        except Exception:
            size = 0
        return f"Created PDF {rp} ({size} bytes)"
