"""In-process connector registry for the recall pipeline.

A registry holds the connector instances active in the current process.
The recall layer queries it to know which `hint()` calls to fan out.

For v1 this is flat (one set of connectors for the whole process).
Per-tenant enablement is a future addition — handled at the recall
layer by passing a filtered subset of `registry.all_hint_capable()`,
not by maintaining N registries.

Thread-safety: methods take an internal lock so a connector can be
registered or removed while a recall is in flight without races. The
recall layer copies the connector list under the lock and runs
`hint()` outside it.
"""
from __future__ import annotations

import threading

from .base import Connector
from .types import Capability


class ConnectorRegistry:
    """Process-wide registry of connector instances by name."""

    def __init__(self) -> None:
        self._connectors: dict[str, Connector] = {}
        self._lock = threading.Lock()

    def register(self, connector: Connector) -> None:
        """Register or replace a connector. Replacing by name is intentional
        so a connector can be hot-swapped without process restart."""
        if not isinstance(connector, Connector):
            raise TypeError(
                f"register() expects a Connector instance, got {type(connector).__name__}"
            )
        if not connector.name:
            raise ValueError("Connector.name must be non-empty to register")
        with self._lock:
            self._connectors[connector.name] = connector

    def unregister(self, name: str) -> bool:
        """Remove a connector by name. Returns True if it was present."""
        with self._lock:
            return self._connectors.pop(name, None) is not None

    def get(self, name: str) -> Connector | None:
        with self._lock:
            return self._connectors.get(name)

    def all(self) -> list[Connector]:
        """Snapshot of all registered connectors (any capability set)."""
        with self._lock:
            return list(self._connectors.values())

    def all_hint_capable(self) -> list[Connector]:
        """Snapshot of registered connectors that advertise HINT.

        The recall layer iterates this list. Connectors without HINT
        are still callable as MCP tools but never fire on recall
        fan-out — useful for connectors that only expose write
        operations or where native search quality is too poor for
        live hints.
        """
        with self._lock:
            return [
                c for c in self._connectors.values()
                if Capability.HINT in c.capabilities
            ]

    def __len__(self) -> int:
        with self._lock:
            return len(self._connectors)

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._connectors


# Process-wide default registry. Most callers register on this.
# A different registry can be constructed for tests.
default_registry = ConnectorRegistry()
