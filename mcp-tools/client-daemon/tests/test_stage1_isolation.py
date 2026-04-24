"""Stage 1 multi-tenancy integration tests.

Exercises the live veilguard.phishield.com stack — no mocks — to verify:
  1. Unauthenticated HTTP proxy rejects with 401.
  2. /api/client/register (direct to sub-agents via Caddy) needs x-user-id.
  3. Same user_id always gets the same token (persistence).
  4. Different user_ids get different tokens.
  5. WebSocket rejects without user_id.
  6. WebSocket rejects with wrong token.
  7. WebSocket accepts valid (user_id, token) pair and registers bridge.
  8. Second user's WebSocket gets its own independent bridge.
  9. /api/client/status returns per-user connection state.
 10. Daemon disconnect removes only its bridge.

Run:  python tests/test_stage1_isolation.py

Requires: httpx websockets
"""

import asyncio
import json
import sys
import time

import httpx
import websockets


BASE = "https://veilguard.phishield.com"
SUB_AGENTS = f"{BASE}/api/sub-agents"
WS_URL = "wss://veilguard.phishield.com/ws/client"

USER_A = f"test-user-a-{int(time.time())}"
USER_B = f"test-user-b-{int(time.time())}"


RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, passed, detail))
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))


async def http_json(client: httpx.AsyncClient, method: str, path: str, **kw):
    resp = await client.request(method, path, **kw)
    try:
        body = resp.json()
    except Exception:
        body = {"_raw": resp.text[:200]}
    return resp.status_code, body


async def ws_auth(user_id: str, token: str, client_id: str = "test-client"):
    """Attempt WebSocket auth, return (accepted: bool, response: dict)."""
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({
            "jsonrpc": "2.0",
            "method": "auth",
            "params": {"user_id": user_id, "token": token, "client_id": client_id},
        }))
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        except asyncio.TimeoutError:
            return False, {"error": "timeout"}
        resp = json.loads(raw)
        if "error" in resp:
            return False, resp
        return True, resp


async def ws_connect_and_hold(user_id: str, token: str, client_id: str, duration: float):
    """Connect, auth, and hold the connection for `duration` seconds — lets
    status/bridge endpoints observe a live daemon without race.
    """
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({
            "jsonrpc": "2.0",
            "method": "auth",
            "params": {"user_id": user_id, "token": token, "client_id": client_id},
        }))
        # Drain the auth ack.
        await asyncio.wait_for(ws.recv(), timeout=5.0)
        await asyncio.sleep(duration)


async def main():
    print(f"USER_A = {USER_A}")
    print(f"USER_B = {USER_B}\n")

    # ── HTTP: unauthenticated LibreChat proxy → 401 ────────────────
    async with httpx.AsyncClient(timeout=10.0) as client:
        sc, _ = await http_json(client, "GET", f"{BASE}/api/veilguard-client/register")
        record(
            "LibreChat proxy rejects unauthenticated",
            sc == 401,
            f"status={sc}",
        )

    # ── HTTP direct to sub-agents via Caddy ────────────────────────
    async with httpx.AsyncClient(timeout=10.0) as client:
        # No x-user-id → 400
        sc, body = await http_json(client, "GET", f"{SUB_AGENTS}/api/client/register")
        record(
            "sub-agents /register rejects missing x-user-id",
            sc == 400,
            f"status={sc} body={body}",
        )

        # User A registers → 200, token
        sc, body_a = await http_json(
            client, "GET", f"{SUB_AGENTS}/api/client/register",
            headers={"x-user-id": USER_A},
        )
        token_a = body_a.get("token", "")
        record(
            "User A register mints token",
            sc == 200 and bool(token_a) and body_a.get("user_id") == USER_A,
            f"status={sc} token={token_a[:8]}…",
        )

        # User A registers again → same token (persistence)
        sc, body_a2 = await http_json(
            client, "GET", f"{SUB_AGENTS}/api/client/register",
            headers={"x-user-id": USER_A},
        )
        record(
            "User A register is idempotent (same token)",
            sc == 200 and body_a2.get("token") == token_a,
            f"same={body_a2.get('token') == token_a}",
        )

        # User B registers → different token
        sc, body_b = await http_json(
            client, "GET", f"{SUB_AGENTS}/api/client/register",
            headers={"x-user-id": USER_B},
        )
        token_b = body_b.get("token", "")
        record(
            "User B token differs from User A",
            sc == 200 and token_b and token_b != token_a,
            f"a={token_a[:8]} b={token_b[:8]}",
        )

    # ── WebSocket auth rejection cases ─────────────────────────────
    try:
        accepted, resp = await ws_auth(user_id="", token=token_a)
        record(
            "WS rejects missing user_id",
            not accepted and "user_id" in json.dumps(resp).lower(),
            f"accepted={accepted} resp={resp}",
        )
    except websockets.exceptions.ConnectionClosedError as e:
        record("WS rejects missing user_id", True, f"closed={e}")

    try:
        accepted, resp = await ws_auth(user_id=USER_A, token="deadbeef" * 4)
        record(
            "WS rejects wrong token",
            not accepted,
            f"accepted={accepted} resp={resp}",
        )
    except websockets.exceptions.ConnectionClosedError as e:
        record("WS rejects wrong token", True, f"closed={e}")

    # Cross-user token swap: user A presents user B's token
    try:
        accepted, resp = await ws_auth(user_id=USER_A, token=token_b)
        record(
            "WS rejects cross-user token swap (A with B's token)",
            not accepted,
            f"accepted={accepted} resp={resp}",
        )
    except websockets.exceptions.ConnectionClosedError as e:
        record("WS rejects cross-user token swap", True, f"closed={e}")

    # ── WebSocket happy path + per-user status ─────────────────────
    # Launch both daemons concurrently, then probe /status for each.
    async def daemon_a():
        await ws_connect_and_hold(USER_A, token_a, "daemon-a", duration=6.0)

    async def daemon_b():
        await ws_connect_and_hold(USER_B, token_b, "daemon-b", duration=6.0)

    da = asyncio.create_task(daemon_a())
    db = asyncio.create_task(daemon_b())
    # Give both time to complete the auth handshake before probing.
    await asyncio.sleep(1.5)

    async with httpx.AsyncClient(timeout=10.0) as client:
        sc, status_a = await http_json(
            client, "GET", f"{SUB_AGENTS}/api/client/status",
            headers={"x-user-id": USER_A},
        )
        record(
            "User A status shows connected daemon-a",
            sc == 200
            and status_a.get("connected") is True
            and status_a.get("client_id") == "daemon-a",
            f"status_a={status_a}",
        )

        sc, status_b = await http_json(
            client, "GET", f"{SUB_AGENTS}/api/client/status",
            headers={"x-user-id": USER_B},
        )
        record(
            "User B status shows connected daemon-b (independent)",
            sc == 200
            and status_b.get("connected") is True
            and status_b.get("client_id") == "daemon-b",
            f"status_b={status_b}",
        )

        # Cross-check: third user who hasn't registered shouldn't see either daemon.
        sc, status_c = await http_json(
            client, "GET", f"{SUB_AGENTS}/api/client/status",
            headers={"x-user-id": "never-registered-user"},
        )
        record(
            "Unregistered user sees no daemon",
            sc == 200 and status_c.get("connected") is False,
            f"status_c={status_c}",
        )

    # Wait for daemons to exit cleanly.
    await asyncio.gather(da, db, return_exceptions=True)

    # After disconnect, status should flip back.
    await asyncio.sleep(0.5)
    async with httpx.AsyncClient(timeout=10.0) as client:
        sc, status_a_post = await http_json(
            client, "GET", f"{SUB_AGENTS}/api/client/status",
            headers={"x-user-id": USER_A},
        )
        record(
            "User A status clears on daemon disconnect",
            sc == 200 and status_a_post.get("connected") is False,
            f"post={status_a_post}",
        )

    # ── Download URL still serves the installer ────────────────────
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.head(f"{SUB_AGENTS}/download/VeilguardSetup.exe")
        record(
            "Installer download URL returns 200 + attachment",
            resp.status_code == 200
            and "attachment" in (resp.headers.get("Content-Disposition") or ""),
            f"status={resp.status_code} cd={resp.headers.get('Content-Disposition')}",
        )

    # ── Summary ────────────────────────────────────────────────────
    passed = sum(1 for _, p, _ in RESULTS if p)
    total = len(RESULTS)
    print(f"\n{'=' * 60}")
    print(f"  {passed}/{total} passed")
    print("=" * 60)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
