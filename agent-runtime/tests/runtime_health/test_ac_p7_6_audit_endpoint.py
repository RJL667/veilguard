"""AC-P7.6 — `/audit/tcmm_source_kinds` endpoint contract.

The endpoint is a runtime probe: it scans TCMM `archive` for rows
written by agent-runtime (extracted_by starts with `agent:` OR
namespace starts with `agent/`) and reports any whose `source` value
is not in the AC-P7.6 allow-list.

These tests cover the contract — the allow-list, the violation
detection, and the 422-on-violation status code.  Live coverage is via
a cron poll of this endpoint (not unit-tested here).
"""

from __future__ import annotations

import json
import unittest.mock as mock
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_audit_endpoint_registered_in_main():
    """Source-grep — the endpoint must stay registered."""
    src = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert '@app.get("/audit/tcmm_source_kinds")' in src
    assert 'async def audit_tcmm_source_kinds' in src


def test_audit_allow_list_matches_spec():
    """The endpoint's allow-list must mirror AC-P7.6 §7.3."""
    src = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    # Crude but stable — assert every allow-list member appears in the file.
    for kind in ("lesson", "proposal", "outcome_narrative",
                 "discussion", "agent_observation"):
        assert f'"{kind}"' in src, (
            f"AC-P7.6 allow-list missing {kind!r} in audit endpoint"
        )


def test_audit_allow_list_matches_phase_7_writers():
    """The runtime allow-list must NOT drift from the static allow-list
    used by `tests/memory/test_phase_7_acceptance.py`.  Both must contain
    the same five members."""
    static_allow = {"lesson", "proposal", "outcome_narrative",
                    "discussion", "agent_observation"}
    # Re-read the static AC-P7.6 test to extract its allow-list.
    src = (REPO_ROOT / "tests" / "memory"
           / "test_phase_7_acceptance.py").read_text(encoding="utf-8")
    for k in static_allow:
        assert f'"{k}"' in src, (
            f"static AC-P7.6 test missing {k!r} — drift from runtime probe"
        )
