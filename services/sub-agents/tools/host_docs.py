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

import os
from pathlib import Path

_DOC_READ_ROOT = os.environ.get("HOST_DOC_READ_ROOT", os.path.expanduser("~"))


def _resolve_read(path: str):
    """Return (resolved_path, "") or (None, error_str)."""
    if not isinstance(path, str) or not path.strip():
        return None, "Error: path is required"
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
        """Read and extract text from a PDF on the user's Windows machine.

        USE THIS for PDFs at real Windows paths, e.g.
        'C:/Users/rudol/Downloads/report.pdf'. The container 'documents'
        read_pdf cannot see Windows paths — this runs on the host and parses
        with PyMuPDF. Forward or back slashes both work.

        Args:
            path: PDF path (absolute Windows path, or relative to the user's home dir).
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
            for i in rng:
                out.append(f"--- Page {i + 1} ---\n{doc[i].get_text().strip()}\n")
            return "\n".join(out)
        except Exception as e:
            return f"Error reading PDF: {e}"
        finally:
            doc.close()

    @mcp.tool()
    async def read_xlsx(path: str, sheet: str = "", max_rows: int = 500) -> str:
        """Read an Excel spreadsheet on the user's WINDOWS machine as text.

        USE THIS for .xlsx at real Windows paths (e.g.
        'C:/Users/rudol/test_excel.xlsx'). The container 'documents' read_xlsx
        cannot see Windows paths; this runs on the host.

        Args:
            path: XLSX path (absolute Windows path or relative to the user's home).
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
        """Create an Excel spreadsheet ON the user's WINDOWS machine.

        USE THIS (not the container 'documents' create_xlsx) to create .xlsx
        files on the user's actual machine, e.g.
        'C:/Users/rudol/Downloads/report.xlsx'. The container tool only writes
        inside its own /workspace and never touches the user's filesystem.

        Args:
            path: Output XLSX path (absolute Windows path or relative to the user's home).
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
