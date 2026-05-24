"""Tests for the in-process connector registry."""
from __future__ import annotations

import pathlib
import sys

import pytest

_CONNECTORS_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CONNECTORS_DIR))

from _base import (  # noqa: E402
    Capability,
    Chunk,
    Connector,
    ConnectorRegistry,
    Content,
)


class _FakeConnector(Connector):
    def __init__(self, name: str, capabilities: set[Capability] | None = None):
        self.name = name
        self.version = "0.0.0"
        self.capabilities = capabilities or set()

    def chunk(self, content: Content) -> list[Chunk]:
        return []


class TestConnectorRegistry:
    def test_empty_registry(self):
        r = ConnectorRegistry()
        assert len(r) == 0
        assert r.all() == []
        assert r.all_hint_capable() == []
        assert r.get("anything") is None

    def test_register_and_get(self):
        r = ConnectorRegistry()
        c = _FakeConnector("alpha")
        r.register(c)
        assert len(r) == 1
        assert "alpha" in r
        assert r.get("alpha") is c

    def test_register_replaces_by_name(self):
        r = ConnectorRegistry()
        c1 = _FakeConnector("alpha")
        c2 = _FakeConnector("alpha")
        r.register(c1)
        r.register(c2)
        assert len(r) == 1
        assert r.get("alpha") is c2

    def test_unregister(self):
        r = ConnectorRegistry()
        c = _FakeConnector("alpha")
        r.register(c)
        assert r.unregister("alpha") is True
        assert r.unregister("alpha") is False  # already gone
        assert "alpha" not in r

    def test_all_returns_snapshot(self):
        r = ConnectorRegistry()
        r.register(_FakeConnector("a"))
        r.register(_FakeConnector("b"))
        snapshot = r.all()
        # Mutating the snapshot doesn't affect the registry
        snapshot.clear()
        assert len(r) == 2

    def test_all_hint_capable_filters(self):
        r = ConnectorRegistry()
        r.register(_FakeConnector("hinter", {Capability.HINT, Capability.READ}))
        r.register(_FakeConnector("writer", {Capability.WRITE}))
        r.register(_FakeConnector("reader", {Capability.READ}))
        names = [c.name for c in r.all_hint_capable()]
        assert names == ["hinter"]

    def test_register_rejects_non_connector(self):
        r = ConnectorRegistry()
        with pytest.raises(TypeError):
            r.register("not a connector")  # type: ignore[arg-type]

    def test_register_rejects_empty_name(self):
        r = ConnectorRegistry()
        c = _FakeConnector("")
        with pytest.raises(ValueError):
            r.register(c)

    def test_default_registry_exists(self):
        from _base import default_registry
        assert isinstance(default_registry, ConnectorRegistry)
