"""Pytest fixtures shared across tests.

Setup:
  - Add the project root to sys.path so `from app.X import Y` works
    when pytest runs from agent-runtime/.
  - Set minimal env vars so config.py doesn't complain during import.
"""

import os
import sys
from pathlib import Path

# Make `from app.* import ...` resolve without an editable install.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Make `from agent import ...` (the unified agent harness, lives outside
# the agent-runtime/ package) resolve too — needed by Phase 6.1 critic
# tests + Phase 6.11 Director interface tests.
_REPO_ROOT = _ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Minimal env so config.validate() doesn't fail spuriously on test import.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")
os.environ.setdefault("AGENTS_DIR", str(_ROOT / "tests" / "fixtures"))
