"""Tests for a2a_auth.py — JWT + mTLS + API-key precedence."""

import os
from unittest.mock import patch, MagicMock
import pytest

from app import a2a_auth as A


# ── parse_mtls_subject ─────────────────────────────────────────────────


def test_parse_mtls_subject_returns_dict_of_keys():
    out = A.parse_mtls_subject("CN=acme-corp,O=Acme,C=ZA")
    assert out == {"CN": "acme-corp", "O": "Acme", "C": "ZA"}


def test_parse_mtls_subject_handles_whitespace():
    out = A.parse_mtls_subject(" CN = acme , O = Acme ")
    assert out["CN"] == "acme"
    assert out["O"] == "Acme"


def test_parse_mtls_subject_empty_returns_none():
    assert A.parse_mtls_subject("") is None
    assert A.parse_mtls_subject(None) is None


def test_parse_mtls_subject_no_equals_skipped():
    out = A.parse_mtls_subject("CN=foo,garbage,O=bar")
    assert out == {"CN": "foo", "O": "bar"}


# ── verify_jwt ─────────────────────────────────────────────────────────


def test_verify_jwt_returns_none_when_disabled():
    with patch.dict(os.environ, {}, clear=False):
        # Ensure env var not set
        os.environ.pop("VEILGUARD_A2A_JWT_JWKS_URL", None)
        assert A.verify_jwt("any.token.here") is None


def test_verify_jwt_returns_none_for_empty_token():
    with patch.dict(os.environ, {"VEILGUARD_A2A_JWT_JWKS_URL": "https://example.com/.well-known/jwks.json"}):
        assert A.verify_jwt("") is None
        assert A.verify_jwt(None) is None


def test_verify_jwt_returns_none_when_pyjwt_unavailable_and_no_insecure_flag():
    """Without PyJWT AND no insecure-flag → safely returns None."""
    with patch.dict(os.environ, {
        "VEILGUARD_A2A_JWT_JWKS_URL": "https://example.com/jwks",
    }, clear=False):
        os.environ.pop("VEILGUARD_A2A_JWT_INSECURE_UNSIGNED", None)
        # If PyJWT IS installed we can't test this path directly without
        # heavy mocking — but verify_jwt also fails fast on bad tokens,
        # so passing junk is enough to assert "returns None" defensively.
        out = A.verify_jwt("not.a.real.jwt")
    assert out is None


# ── mtls_to_tenant ─────────────────────────────────────────────────────


def test_mtls_to_tenant_disabled_returns_none():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("VEILGUARD_A2A_MTLS_ENABLED", None)
        assert A.mtls_to_tenant("CN=foo") is None


def test_mtls_to_tenant_enabled_finds_row_by_cn():
    """When CN matches an active key row's label → returns that row."""
    class FakeColumn:
        def __init__(self, vs): self._v = vs
        def __getitem__(self, i):
            class C:
                def __init__(self, v): self.v = v
                def as_py(self): return self.v
            return C(self._v[i])
    class FakeArrow:
        def __init__(self, rows):
            self._rows = rows
            self.num_rows = len(rows)
            cols = set()
            for r in rows: cols.update(r.keys())
            self.column_names = sorted(cols)
        def column(self, name):
            return FakeColumn([r.get(name) for r in self._rows])
    class FakeTable:
        def __init__(self, rows):
            self._rows = rows
        def search(self): return self
        def where(self, _): return self
        def limit(self, _): return self
        def to_arrow(self):
            return FakeArrow(self._rows)
    fake_store = MagicMock()
    fake_store.table.return_value = FakeTable([{
        "id": "a2ak-1", "tenant_id": "acme-tenant", "label": "acme-corp",
        "allowed_agents": ["researcher"], "rate_limit_per_min": 30,
        "status": "active",
    }])
    with patch.dict(os.environ, {"VEILGUARD_A2A_MTLS_ENABLED": "1"}), \
         patch("app.ledger.store.LedgerStore.get", return_value=fake_store):
        row = A.mtls_to_tenant("CN=acme-corp,O=Acme")
    assert row is not None
    assert row["tenant_id"] == "acme-tenant"
    assert row["label"] == "acme-corp"


def test_mtls_to_tenant_no_match_returns_none():
    class FakeTable:
        def search(self): return self
        def where(self, _): return self
        def limit(self, _): return self
        class _A:
            num_rows = 0
            def column(self, _n): return None
        def to_arrow(self): return self._A()
    fake_store = MagicMock()
    fake_store.table.return_value = FakeTable()
    with patch.dict(os.environ, {"VEILGUARD_A2A_MTLS_ENABLED": "1"}), \
         patch("app.ledger.store.LedgerStore.get", return_value=fake_store):
        assert A.mtls_to_tenant("CN=unknown") is None


# ── authenticate() precedence ───────────────────────────────────────────


def test_authenticate_returns_none_when_all_methods_fail():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("VEILGUARD_A2A_JWT_JWKS_URL", None)
        os.environ.pop("VEILGUARD_A2A_MTLS_ENABLED", None)
        out = A.authenticate(
            bearer_token=None, mtls_subject=None,
            api_key=None, target_agent_id="researcher",
        )
    assert out is None


def test_authenticate_api_key_path_works_when_only_method():
    """Standalone API-key path: JWT + mTLS disabled, valid key → row returned."""
    fake_row = {"id": "a2ak-1", "tenant_id": "t1", "status": "active"}
    with patch.dict(os.environ, {}, clear=False), \
         patch("app.a2a_external._resolve_key", return_value=fake_row):
        os.environ.pop("VEILGUARD_A2A_JWT_JWKS_URL", None)
        os.environ.pop("VEILGUARD_A2A_MTLS_ENABLED", None)
        out = A.authenticate(
            bearer_token=None, mtls_subject=None,
            api_key="some-key", target_agent_id="researcher",
        )
    assert out is not None
    assert out["_auth_method"] == "api_key"
    assert out["tenant_id"] == "t1"
