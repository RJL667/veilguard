"""
Local setup web server for Veilguard client daemon.

After install, serves a page on localhost:9090 where the user can:
1. Scan the QR code from LibreChat (webcam)
2. Or paste the connection token manually

Once configured, saves to ~/.veilguard/config.yaml and starts the daemon.
"""

import http.server
import json
import logging
import os
import socket
import threading
import webbrowser

logger = logging.getLogger("veilguard-setup")

SETUP_PORT = 9090

# Paste-only setup page — no external CDN, no camera permissions, no
# tabs.  Scanning your own screen with your own webcam was never a
# realistic flow on a laptop; "Click to copy" in LibreChat + paste
# here is what everyone actually does.  Removing the QR scanner also
# eliminates a class of dead-page bugs (CDN blocked, camera stuck
# on permission prompt, etc.).
#
# IMPORTANT — raw string ``r"""..."""``.  Regular Python triple-quoted
# strings process ``\u200B``, ``\uFEFF``, ``\r``, ``\n`` and similar
# escapes *before* writing to HTML.  The regex below
# (``/[\u200B-\u200D\uFEFF\r\n]/``) would therefore emit real unicode
# zero-width chars + an actual CRLF inside a JS regex literal, which
# is a JS syntax error that kills every statement after it — including
# the button click handler.  Raw string leaves the backslash-escapes
# literal so the JS engine parses them itself.  This bug dated back
# to the original file; it only showed up once the paste flow became
# the only flow.
SETUP_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Veilguard Setup</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0f; color: #e0e0e0; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
  .container { max-width: 560px; width: 100%; padding: 40px; }
  .logo { text-align: center; margin-bottom: 32px; }
  .logo h1 { font-size: 28px; font-weight: 700; color: #fff; }
  .logo span { color: #7c3aed; }
  .logo p { color: #888; font-size: 14px; margin-top: 8px; }
  .card { background: #141420; border: 1px solid #2a2a3a; border-radius: 12px; padding: 24px; margin-bottom: 20px; }
  .card h2 { font-size: 14px; margin-bottom: 12px; color: #fff; font-weight: 600; }
  textarea { width: 100%; min-height: 120px; padding: 12px 16px; background: #0a0a0f; border: 1px solid #2a2a3a; border-radius: 8px; color: #fff; font-size: 13px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; outline: none; resize: vertical; }
  textarea:focus { border-color: #7c3aed; }
  textarea::placeholder { color: #555; }
  .btn { width: 100%; padding: 14px; background: #7c3aed; color: #fff; border: none; border-radius: 8px; font-size: 15px; font-weight: 600; cursor: pointer; margin-top: 12px; transition: background 0.2s; }
  .btn:hover { background: #6d28d9; }
  .btn:disabled { background: #333; cursor: not-allowed; }
  .status { text-align: center; padding: 16px; border-radius: 8px; margin-top: 16px; font-size: 14px; display: none; }
  .status.success { display: block; background: #0a2e1a; border: 1px solid #166534; color: #4ade80; }
  .status.error { display: block; background: #2e0a0a; border: 1px solid #991b1b; color: #f87171; }
  .status.loading { display: block; background: #1a1a2e; border: 1px solid #312e81; color: #a78bfa; }
  .help { color: #666; font-size: 12px; line-height: 1.6; padding: 8px 4px; }
  .help ol { padding-left: 20px; margin: 6px 0; }
  .help code { background: #1a1a2e; padding: 1px 6px; border-radius: 3px; color: #a78bfa; font-size: 11px; }
</style>
</head>
<body>
<div class="container">
  <div class="logo">
    <h1><span>Veilguard</span> Setup</h1>
    <p>Connect this machine to your Veilguard workspace</p>
  </div>

  <div class="card">
    <h2>Paste the connection string from LibreChat</h2>
    <textarea id="tokenInput" placeholder='{"server":"wss://veilguard.phishield.com/ws/client","token":"...","user_id":"..."}'></textarea>
    <button class="btn" id="connectBtn" type="button">Connect</button>
  </div>

  <div id="status" class="status"></div>

  <div class="help">
    <strong style="color:#888;">How to get the string:</strong>
    <ol>
      <li>Open <code>veilguard.phishield.com</code> in your browser, log in</li>
      <li>Open the <strong>Workspace</strong> side panel</li>
      <li>Under "Scan QR code in the installer", click the grey box labelled <strong>Click to copy</strong></li>
      <li>Paste above and click Connect</li>
    </ol>
  </div>
</div>

<script>
// Paste-only flow.  Wiring the click handler in JS (rather than inline
// onclick) so this works even if something weird happens with HTML
// attribute parsing.
document.getElementById('connectBtn').addEventListener('click', function () {
  var val = document.getElementById('tokenInput').value.trim();
  if (!val) return;
  processConnectionString(val);
});
// Also submit on Ctrl/Cmd+Enter from the textarea — quality-of-life.
document.getElementById('tokenInput').addEventListener('keydown', function (e) {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    document.getElementById('connectBtn').click();
  }
});

function processConnectionString(rawStr) {
  const status = document.getElementById('status');
  status.className = 'status loading';
  status.textContent = 'Connecting...';

  // Clean input: trim whitespace, remove zero-width chars, newlines
  const str = rawStr.trim().replace(/[\u200B-\u200D\uFEFF\r\n]/g, '');

  let server = '', token = '', user_id = '';

  // Method 1: JSON format {"server":"...","token":"...","user_id":"..."}
  // Extract JSON if embedded in other text
  const jsonMatch = str.match(/\{[^}]*"(server|ws_url)"\s*:\s*"[^"]+[^}]*\}/);
  if (jsonMatch) {
    try {
      const obj = JSON.parse(jsonMatch[0]);
      server = obj.server || obj.ws_url || '';
      token = obj.token || '';
      user_id = obj.user_id || '';
    } catch { /* fall through */ }
  }

  // Method 2: URL format ws://host/ws/client#token=abc&user_id=xyz
  if (!server && str.match(/^wss?:\/\//)) {
    const hashIdx = str.indexOf('#');
    const queryIdx = str.indexOf('?');
    let paramStr = '';
    if (hashIdx > 0) {
      server = str.substring(0, hashIdx);
      paramStr = str.substring(hashIdx + 1);
    } else if (queryIdx > 0) {
      server = str.substring(0, queryIdx);
      paramStr = str.substring(queryIdx + 1);
    } else {
      server = str;
    }
    if (paramStr) {
      for (const part of paramStr.split('&')) {
        const [k, v] = part.split('=');
        if (k === 'token') token = decodeURIComponent(v || '');
        else if (k === 'user_id') user_id = decodeURIComponent(v || '');
      }
    }
  }

  // Method 3: Just a hex token (32+ chars) — legacy, no user_id
  if (!server && !token && str.match(/^[a-f0-9]{24,}$/i)) {
    token = str;
  }

  if (!server && !token) {
    status.className = 'status error';
    status.textContent = 'Invalid input. Paste the JSON string or scan the QR code.';
    return;
  }

  if (!user_id) {
    status.className = 'status error';
    status.textContent = 'Missing user_id — re-copy the QR code from LibreChat (must be a per-user token).';
    return;
  }

  // Show what we parsed
  status.textContent = 'Connecting to ' + (server || 'server') + '...';

  // POST to local daemon setup endpoint
  fetch('/setup', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ server, token, user_id })
  })
  .then(r => r.json())
  .then(data => {
    if (data.status === 'ok') {
      status.className = 'status success';
      status.textContent = 'Connected! You can close this page. Veilguard daemon is running.';
    } else {
      status.className = 'status error';
      status.textContent = 'Error: ' + (data.error || 'Unknown error');
    }
  })
  .catch(err => {
    status.className = 'status error';
    status.textContent = 'Failed to connect: ' + err.message;
  });
}
</script>
</body>
</html>"""


class SetupHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for the local setup page."""

    # Callback set by the daemon when setup completes
    on_setup_complete = None

    def do_GET(self):
        if self.path == "/" or self.path == "/setup":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(SETUP_HTML.encode())
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/setup":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            server = body.get("server", "")
            token = body.get("token", "")
            user_id = body.get("user_id", "")

            if not server:
                self._json_response(400, {"status": "error", "error": "Missing server URL"})
                return
            if not user_id:
                self._json_response(
                    400,
                    {"status": "error", "error": "Missing user_id — re-copy the QR from LibreChat"},
                )
                return

            # Save config
            try:
                config = save_config(server, token, user_id)
                self._json_response(200, {"status": "ok", "config": config})

                # Signal daemon to start
                if SetupHandler.on_setup_complete:
                    SetupHandler.on_setup_complete(config)

            except Exception as e:
                self._json_response(500, {"status": "error", "error": str(e)})
        else:
            self.send_error(404)

    def _json_response(self, code, data):
        # Explicit Content-Length + Connection: close.  Without these,
        # Python's BaseHTTPRequestHandler emits an HTTP response with
        # neither ``Content-Length`` nor ``Transfer-Encoding: chunked``,
        # and the browser keeps reading past the JSON body looking for
        # more bytes — producing a parse error like "Unexpected
        # non-whitespace character after JSON at position 292".
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        try:
            self.wfile.flush()
        except Exception:
            pass

    def log_message(self, format, *args):
        logger.debug(f"[SETUP] {args[0]}")


def save_config(server: str, token: str, user_id: str = "") -> dict:
    """Save connection config to ~/.veilguard/config.yaml."""
    import platform

    client_id = f"{os.getenv('USER', os.getenv('USERNAME', 'client'))}-{platform.node()}"
    config_dir = os.path.join(os.path.expanduser("~"), ".veilguard")
    os.makedirs(config_dir, exist_ok=True)

    config = {
        "server": server,
        "token": token,
        "user_id": user_id,
        "client_id": client_id,
        "project_root": os.path.expanduser("~"),
        "timeout": 60,
        "reconnect_delay": 5,
        "max_reconnect_delay": 300,
    }

    config_path = os.path.join(config_dir, "config.yaml")
    try:
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    except ImportError:
        with open(config_path, "w") as f:
            for k, v in config.items():
                f.write(f"{k}: {json.dumps(v)}\n")

    logger.info(f"Config saved to {config_path}")
    return config


def is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def run_setup_server(on_complete=None) -> threading.Thread:
    """Start the setup web server in a background thread.

    Args:
        on_complete: callback(config_dict) called when user completes setup

    Returns:
        The server thread (daemon thread, dies with main process)
    """
    if not is_port_free(SETUP_PORT):
        logger.warning(f"Port {SETUP_PORT} already in use — setup server skipped")
        return None

    SetupHandler.on_setup_complete = on_complete

    server = http.server.HTTPServer(("127.0.0.1", SETUP_PORT), SetupHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    logger.info(f"Setup server running at http://localhost:{SETUP_PORT}")
    return thread


def open_setup_page():
    """Open the setup page in the default browser."""
    webbrowser.open(f"http://localhost:{SETUP_PORT}")
