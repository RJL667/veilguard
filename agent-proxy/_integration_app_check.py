"""Integration check: load the REAL FastAPI proxy app in-process and
exercise the unified redactor wiring (import bootstrap + /rehydrate route
+ startup prewarm).  No network/forward needed.

Run from agent-proxy/:  python _integration_app_check.py
"""
import os, sys, tempfile, asyncio

os.environ.setdefault("VEILGUARD_PII_DB_PATH", tempfile.mkdtemp(prefix="pii_app_"))
os.environ.setdefault("TCMM_ENABLED", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `app` package imports

ok = True
def check(n, c):
    global ok; print(("PASS" if c else "FAIL"), n); ok = ok and c

# 1. The app module imports cleanly with the new `from pii import ...`.
from app.main import app, startup  # noqa: E402
check("1 app.main imports (pii bootstrap resolves in-process)", app is not None)

# 2. startup() prewarm runs (constructs the shared redactor, fail-closed).
asyncio.get_event_loop().run_until_complete(startup())
check("2 startup() prewarm ran without error", True)

# 3. Seed a mapping via the shared redactor, then round-trip through the
#    REAL /rehydrate ASGI route.
from pii import get_redactor  # noqa: E402
r = get_redactor()
sid = "pii-inttest"
red = r.redact_text("Contact Dana Reyes at dana.reyes@example.com today.", sid)
check("3a seeded redaction produced REF tokens", "REF_" in red and "dana.reyes@example.com" not in red)

from fastapi.testclient import TestClient  # noqa: E402
client = TestClient(app)
resp = client.post("/rehydrate", json={"text": red, "conversation_id": sid, "user_id": "inttest"})
check("3b /rehydrate route returns 200", resp.status_code == 200)
body = resp.json()
check("3c /rehydrate restored the email", "dana.reyes@example.com" in body.get("text", ""))
check("3d /rehydrate restored the person", "Dana Reyes" in body.get("text", ""))

# 4. Session-less /rehydrate (no conv/user) still works via rehydrate_any.
resp2 = client.post("/rehydrate", json={"text": red})
check("4 session-less /rehydrate restored values", "dana.reyes@example.com" in resp2.json().get("text", ""))

print("\nINTEGRATION:", "ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
