"""Test-only loader for agent-proxy/app/chat_agent_handler.py.

We can't `from app.chat_agent_handler import ...` because:
  - From agent-proxy/ cwd, `app` resolves correctly.
  - From repo root, `app` collides with `agent-runtime/app/` (which
    gets added to sys.path when agent/tests run their persona loader).
  - The collision turns `app.chat_agent_handler` into a ModuleNotFoundError
    when the test runs after agent-runtime modules have been imported.

Importlib by file path bypasses the ambiguous package lookup.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HANDLER_PATH = (
    Path(__file__).resolve().parent.parent / "app" / "chat_agent_handler.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "veilguard_pii_proxy_chat_agent_handler",  # uniquely-named
        str(_HANDLER_PATH),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {_HANDLER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load()
is_enabled = _mod.is_enabled
handle_chat_request = _mod.handle_chat_request
