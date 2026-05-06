"""SharePoint file parsing — thin shim over the shared parsing layer.

The actual parsing work happens in
:mod:`mcp_tools.connectors._base.parsing`, which uses LlamaIndex's
``SimpleDirectoryReader`` to handle the long tail of office /
ebook / notebook / image formats out of the box. Keeping a per-
connector shim here means future connectors (Slack file uploads,
Jira attachments, Confluence attachments) reuse the exact same
parsing pipeline by importing the same shared helper.
"""
from __future__ import annotations

import pathlib
import sys


_THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))


from _base.parsing import parse_to_text  # noqa: E402, F401  (re-export)


__all__ = ["parse_to_text"]
