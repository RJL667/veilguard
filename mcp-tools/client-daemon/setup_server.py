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

# HTML with embedded QR scanner (html5-qrcode CDN) + manual paste
SETUP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Veilguard Setup</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0f; color: #e0e0e0; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
  .container { max-width: 480px; width: 100%; padding: 40px; }
  .logo { text-align: center; margin-bottom: 32px; }
  .logo h1 { font-size: 28px; font-weight: 700; color: #fff; }
  .logo span { color: #7c3aed; }
  .logo p { color: #888; font-size: 14px; margin-top: 8px; }
  .card { background: #141420; border: 1px solid #2a2a3a; border-radius: 12px; padding: 24px; margin-bottom: 20px; }
  .card h2 { font-size: 16px; margin-bottom: 16px; color: #fff; }
  #reader { width: 100%; border-radius: 8px; overflow: hidden; }
  .divider { text-align: center; color: #555; font-size: 13px; margin: 20px 0; position: relative; }
  .divider::before, .divider::after { content: ''; position: absolute; top: 50%; width: 40%; height: 1px; background: #2a2a3a; }
  .divider::before { left: 0; }
  .divider::after { right: 0; }
  input[type="text"] { width: 100%; padding: 12px 16px; background: #0a0a0f; border: 1px solid #2a2a3a; border-radius: 8px; color: #fff; font-size: 14px; font-family: monospace; outline: none; }
  input[type="text"]:focus { border-color: #7c3aed; }
  input[type="text"]::placeholder { color: #555; }
  .btn { width: 100%; padding: 14px; background: #7c3aed; color: #fff; border: none; border-radius: 8px; font-size: 15px; font-weight: 600; cursor: pointer; margin-top: 12px; transition: background 0.2s; }
  .btn:hover { background: #6d28d9; }
  .btn:disabled { background: #333; cursor: not-allowed; }
  .status { text-align: center; padding: 16px; border-radius: 8px; margin-top: 16px; font-size: 14px; display: none; }
  .status.success { display: block; background: #0a2e1a; border: 1px solid #166534; color: #4ade80; }
  .status.error { display: block; background: #2e0a0a; border: 1px solid #991b1b; color: #f87171; }
  .status.loading { display: block; background: #1a1a2e; border: 1px solid #312e81; color: #a78bfa; }
  .help { text-align: center; color: #555; font-size: 12px; margin-top: 24px; }
  .help a { color: #7c3aed; text-decoration: none; }
  .tabs { display: flex; gap: 4px; margin-bottom: 16px; }
  .tab { flex: 1; padding: 10px; text-align: center; font-size: 13px; font-weight: 500; border: 1px solid #2a2a3a; border-radius: 8px; cursor: pointer; color: #888; background: transparent; }
  .tab.active { background: #7c3aed22; border-color: #7c3aed; color: #a78bfa; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
</style>
</head>
<body>
<div class="container">
  <div class="logo">
    <h1><span>Veilguard</span> Setup</h1>
    <p>Connect this machine to your Veilguard workspace</p>
  </div>

  <div class="card">
    <div class="tabs">
      <div class="tab active" onclick="switchTab('scan')">Scan QR Code</div>
      <div class="tab" onclick="switchTab('paste')">Paste Token</div>
    </div>

    <div id="tab-scan" class="tab-content active">
      <div id="reader"></div>
      <p style="color:#888; font-size:12px; margin-top:12px; text-align:center;">
        Point your camera at the QR code shown in LibreChat
      </p>
    </div>

    <div id="tab-paste" class="tab-content">
      <label style="font-size:13px; color:#888; display:block; margin-bottom:8px;">
        Connection string from LibreChat:
      </label>
      <input type="text" id="tokenInput" placeholder="wss://server/ws/client#token=abc123..." />
      <button class="btn" onclick="submitManual()">Connect</button>
    </div>
  </div>

  <div id="status" class="status"></div>

  <div class="help">
    Open LibreChat &rarr; Cowork panel &rarr; "Connect Machine" to see your QR code
  </div>
</div>

<script src="https://unpkg.com/html5-qrcode@2.3.8/html5-qrcode.min.js"></script>
<script>
let scanner = null;

function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelector(`.tab:nth-child(${tab === 'scan' ? 1 : 2})`).classList.add('active');
  document.getElementById(`tab-${tab}`).classList.add('active');

  if (tab === 'scan' && !scanner) { startScanner(); }
}

function startScanner() {
  scanner = new Html5Qrcode("reader");
  scanner.start(
    { facingMode: "environment" },
    { fps: 10, qrbox: { width: 250, height: 250 } },
    onScanSuccess,
    () => {}
  ).catch(err => {
    document.getElementById('reader').innerHTML =
      '<p style="color:#f87171; padding:20px; text-align:center;">Camera not available. Use the Paste Token tab instead.</p>';
  });
}

function onScanSuccess(decoded) {
  if (scanner) { scanner.stop().catch(() => {}); }
  processConnectionString(decoded);
}

function submitManual() {
  const val = document.getElementById('tokenInput').value.trim();
  if (!val) return;
  processConnectionString(val);
}

function processConnectionString(rawStr) {
  const status = document.getElementById('status');
  status.className = 'status loading';
  status.textContent = 'Connecting...';

  // Clean input: trim whitespace, remove zero-width chars, newlines
  const str = rawStr.trim().replace(/[\u200B-\u200D\uFEFF\r\n]/g, '');

  let server = '', token = '';

  // Method 1: JSON format {"server":"...","token":"..."}
  // Extract JSON if embedded in other text
  const jsonMatch = str.match(/\{[^}]*"server"\s*:\s*"[^"]+[^}]*\}/);
  if (jsonMatch) {
    try {
      const obj = JSON.parse(jsonMatch[0]);
      server = obj.server || '';
      token = obj.token || '';
    } catch { /* fall through */ }
  }

  // Method 2: URL format ws://host/ws/client#token=abc or ?token=abc
  if (!server && str.match(/^wss?:\/\//)) {
    const hashIdx = str.indexOf('#token=');
    const queryIdx = str.indexOf('?token=');
    if (hashIdx > 0) {
      server = str.substring(0, hashIdx);
      token = str.substring(hashIdx + 7);
    } else if (queryIdx > 0) {
      server = str.substring(0, queryIdx);
      token = str.substring(queryIdx + 7);
    } else {
      server = str;
    }
  }

  // Method 3: Just a hex token (32+ chars)
  if (!server && !token && str.match(/^[a-f0-9]{24,}$/i)) {
    token = str;
  }

  if (!server && !token) {
    status.className = 'status error';
    status.textContent = 'Invalid input. Paste the JSON string or scan the QR code.';
    return;
  }

  // Show what we parsed
  status.textContent = 'Connecting to ' + (server || 'server') + '...';

  // POST to local daemon setup endpoint
  fetch('/setup', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ server, token })
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

// Auto-start scanner
startScanner();
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

            if not server:
                self._json_response(400, {"status": "error", "error": "Missing server URL"})
                return

            # Save config
            try:
                config = save_config(server, token)
                self._json_response(200, {"status": "ok", "config": config})

                # Signal daemon to start
                if SetupHandler.on_setup_complete:
                    SetupHandler.on_setup_complete(config)

            except Exception as e:
                self._json_response(500, {"status": "error", "error": str(e)})
        else:
            self.send_error(404)

    def _json_response(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        logger.debug(f"[SETUP] {args[0]}")


def save_config(server: str, token: str) -> dict:
    """Save connection config to ~/.veilguard/config.yaml."""
    import platform

    client_id = f"{os.getenv('USER', os.getenv('USERNAME', 'client'))}-{platform.node()}"
    config_dir = os.path.join(os.path.expanduser("~"), ".veilguard")
    os.makedirs(config_dir, exist_ok=True)

    config = {
        "server": server,
        "token": token,
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
