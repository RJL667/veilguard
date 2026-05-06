"""End-to-end dry run: TCMM service ↔ connector framework ↔ SharePoint
with a mocked Graph client. Run from `mcp-tools/tcmm-service`:

    cd mcp-tools/tcmm-service && python ../connectors/tests/test_e2e_tcmm_sharepoint.py

The goal is to exercise `_augment_with_connector_hints` exactly as the
recall path will, without hitting Microsoft Graph.
"""
from __future__ import annotations
import asyncio
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_CONNECTORS = _HERE.parent
_MCP_TOOLS = _CONNECTORS.parent
_TCMM_SERVICE = _MCP_TOOLS / "tcmm-service"

sys.path.insert(0, str(_CONNECTORS))
sys.path.insert(0, str(_CONNECTORS / "sharepoint"))
sys.path.insert(0, str(_TCMM_SERVICE))

# Force the connector flag ON before importing tcmm-service so its
# `_connectors_enabled()` returns True.
os.environ["VEILGUARD_CONNECTORS_ENABLED"] = "1"

# A short deadline keeps the test snappy without changing the codepath.
os.environ["VEILGUARD_CONNECTOR_HINT_DEADLINE_MS"] = "1000"


def main() -> int:
    import server as tcmm_server  # noqa: E402
    from _base import (  # noqa: E402
        Capability,
        UserContext,
        default_registry,
    )
    from _base.credentials import OAuthToken, StaticCredentialResolver  # noqa: E402
    from connector import SharePointConnector  # noqa: E402
    from graph import SearchHit  # noqa: E402

    # ── Mock graph client ────────────────────────────────────────
    class _StubGraph:
        async def search(self, access_token: str, query: str, top: int = 10):
            return [
                SearchHit(
                    item_id="01ABCDEF",
                    drive_id="b!drive123",
                    name="Q1-2026-Group-Scheme-Premiums.xlsx",
                    summary=(
                        "Group scheme premium reconciliation across "
                        "broker partners. Includes columns: Bank Account, "
                        "ID Number, Premium Collected. Last updated by "
                        "Petrus Schroeder."
                    ),
                    score=0.92,
                    last_modified="2026-04-30T08:14:22Z",
                    web_url="https://phishield.sharepoint.com/sites/ops/...",
                ),
                SearchHit(
                    item_id="01GHIJKL",
                    drive_id="b!drive123",
                    name="Broker-onboarding-2026.docx",
                    summary=(
                        "Onboarding workflow for new brokers — Pipedrive "
                        "configuration, broker tier, debit order setup."
                    ),
                    score=0.71,
                    last_modified="2026-04-12T10:02:01Z",
                    web_url="https://phishield.sharepoint.com/sites/ops/...",
                ),
            ]

    # ── Register connector with the same default_registry the
    #    TCMM service will read from. ───────────────────────────
    tok = OAuthToken(access_token="stub-token", expires_at=1e12)
    creds = StaticCredentialResolver({("test-user", "sharepoint"): tok})
    sp = SharePointConnector(credentials=creds, graph=_StubGraph())
    default_registry.register(sp)

    print(f"connectors registered: {[c.name for c in default_registry.all()]}")
    print(f"hint-capable:          {[c.name for c in default_registry.all_hint_capable()]}")
    print(f"VEILGUARD_CONNECTORS_ENABLED env: {os.environ.get('VEILGUARD_CONNECTORS_ENABLED')!r}")
    print(f"_connectors_enabled():   {tcmm_server._connectors_enabled()}")
    print(f"deadline_ms:             {tcmm_server._connector_hint_deadline_ms()}")
    print()

    # ── Drive the augment path ──────────────────────────────────
    base_prompt = "[SYSTEM]\nYou are Veilguard. Memory blocks follow.\n"
    user_msg = "What broker premium reconciliations do we have for Q1 2026?"

    augmented, n_blocks = asyncio.run(
        tcmm_server._augment_with_connector_hints(
            prompt=base_prompt,
            user_message=user_msg,
            user_id="test-user",
            tenant_id="phishield",
        )
    )

    print(f"shadow blocks emitted: {n_blocks}")
    print(f"augmented length:      {len(augmented)} chars")
    print()
    print("=" * 70)
    print("AUGMENTED PROMPT")
    print("=" * 70)
    print(augmented)
    print("=" * 70)

    if n_blocks == 0:
        print("FAIL: no shadow blocks produced")
        return 1

    # Sanity: the rendered text should reference the SharePoint connector
    # (so the LLM knows where the snippet came from) and the file titles.
    must_contain = ["sharepoint", "Q1-2026-Group-Scheme-Premiums"]
    missing = [s for s in must_contain if s not in augmented]
    if missing:
        print(f"FAIL: missing markers in augmented prompt: {missing}")
        return 1

    print(f"PASS: {n_blocks} shadow block(s) rendered with expected markers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
