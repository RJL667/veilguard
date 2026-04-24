"""
Gemini Embedding Adapter - Concrete implementation of EmbeddingProvider for Google Gemini.

This is the ONLY file that should import google.generativeai.
Keep all Gemini-specific code isolated here.

Supports three modes:
  1. API key (AI Studio)  — set GEMINI_API_KEY
  2. Vertex AI SSO        — set GEMINI_SSO=1, uses gcloud auth token + Vertex AI endpoint
  3. Google OAuth         — set GEMINI_OAUTH=1, uses Google Sign-In + refresh tokens
"""

import os
import sys
import json
import google.generativeai as genai
from typing import List
from pathlib import Path

try:
    from src.memory.core.interfaces import EmbeddingProvider
    from src.memory.core.perf import perf_scope
    from src.memory.core.tcmm_logging import log_adapter, log_exception
except ImportError:
    from core.interfaces import EmbeddingProvider
    from core.perf import perf_scope
    from core.tcmm_logging import log_adapter, log_exception


# Vertex AI model name mapping: AI Studio uses "models/gemini-..." prefix,
# Vertex AI uses just the model ID without "models/" prefix.
def _strip_models_prefix(name):
    return name.replace("models/", "") if name.startswith("models/") else name


class GeminiOAuthManager:
    """
    Manage Google OAuth tokens for Gemini API access.

    Uses the same OAuth client as Google's official Gemini CLI:
    - No GCP project required for personal accounts
    - No API key needed
    - Free tier: 60 req/min, 1000 req/day
    - Stores refresh token in ~/.gemini/oauth_creds.json
    """

    # OAuth client — set via env vars or defaults to Gemini CLI public client
    CLIENT_ID = os.environ.get("GEMINI_OAUTH_CLIENT_ID", "")
    CLIENT_SECRET = os.environ.get("GEMINI_OAUTH_CLIENT_SECRET", "")
    SCOPES = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
    ]
    AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    TOKEN_FILE = Path.home() / ".gemini" / "oauth_creds.json"

    def __init__(self):
        self.TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)

    def login(self) -> str:
        """Open browser for Google OAuth login. Returns access token."""
        import webbrowser
        import urllib.parse
        import urllib.request
        import http.server
        import hashlib
        import base64
        import secrets

        # PKCE: generate code verifier and challenge
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).rstrip(b"=").decode()
        state = secrets.token_urlsafe(32)

        # Find an open port
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        redirect_uri = f"http://127.0.0.1:{port}/oauth2callback"
        auth_code_holder = {"code": None, "error": None}

        class AuthHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                query = urllib.parse.urlparse(self.path).query
                params = urllib.parse.parse_qs(query)

                if "code" in params:
                    auth_code_holder["code"] = params["code"][0]
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(
                        b"<html><body><h2>Authenticated!</h2>"
                        b"<p>You can close this tab and return to the terminal.</p>"
                        b"</body></html>"
                    )
                elif "error" in params:
                    auth_code_holder["error"] = params.get("error", ["unknown"])[0]
                    self.send_response(400)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    msg = params.get("error_description", [auth_code_holder["error"]])[0]
                    self.wfile.write(f"<html><body><h2>Error: {msg}</h2></body></html>".encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass

        # Build auth URL
        auth_params = {
            "client_id": self.CLIENT_ID,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.SCOPES),
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        auth_url = f"{self.AUTH_URL}?{urllib.parse.urlencode(auth_params)}"

        # Start local server and open browser
        import socketserver
        with socketserver.TCPServer(("127.0.0.1", port), AuthHandler) as httpd:
            httpd.timeout = 300  # 5 min timeout
            print(f"Opening browser for Google Sign-In...")
            webbrowser.open(auth_url)
            print(f"Waiting for authentication (port {port})...")
            httpd.handle_request()

        if auth_code_holder["error"]:
            raise RuntimeError(f"OAuth error: {auth_code_holder['error']}")
        if not auth_code_holder["code"]:
            raise RuntimeError("No authorization code received (timed out?)")

        # Exchange code for tokens (with PKCE verifier)
        token_params = {
            "client_id": self.CLIENT_ID,
            "client_secret": self.CLIENT_SECRET,
            "code": auth_code_holder["code"],
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        }

        req = urllib.request.Request(
            self.TOKEN_URL,
            data=urllib.parse.urlencode(token_params).encode(),
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        token_data = json.loads(resp.read().decode())

        if "error" in token_data:
            raise RuntimeError(f"Token exchange failed: {token_data}")

        # Save credentials
        creds = {
            "refresh_token": token_data["refresh_token"],
            "access_token": token_data["access_token"],
            "expires_at": __import__("time").time() + token_data.get("expires_in", 3600),
            "client_id": self.CLIENT_ID,
            "client_secret": self.CLIENT_SECRET,
            "type": "authorized_user",
        }
        with open(self.TOKEN_FILE, "w") as f:
            json.dump(creds, f, indent=2)
        # Restrictive permissions on non-Windows
        if sys.platform != "win32":
            self.TOKEN_FILE.chmod(0o600)

        return token_data["access_token"]

    def get_access_token(self) -> str:
        """Get a valid access token, refreshing if needed."""
        import urllib.request
        import urllib.parse
        import time

        if not self.TOKEN_FILE.exists():
            raise RuntimeError("Not logged in. Run: python gemini_oauth_login.py")

        with open(self.TOKEN_FILE) as f:
            creds = json.load(f)

        # Return cached token if still valid (with 60s buffer)
        if creds.get("access_token") and creds.get("expires_at", 0) > time.time() + 60:
            return creds["access_token"]

        # Refresh
        refresh_token = creds.get("refresh_token")
        if not refresh_token:
            raise RuntimeError("No refresh token. Run login again.")

        token_params = {
            "client_id": creds.get("client_id", self.CLIENT_ID),
            "client_secret": creds.get("client_secret", self.CLIENT_SECRET),
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }

        req = urllib.request.Request(
            self.TOKEN_URL,
            data=urllib.parse.urlencode(token_params).encode(),
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        token_data = json.loads(resp.read().decode())

        # Update cached credentials
        creds["access_token"] = token_data["access_token"]
        creds["expires_at"] = time.time() + token_data.get("expires_in", 3600)
        with open(self.TOKEN_FILE, "w") as f:
            json.dump(creds, f, indent=2)

        return token_data["access_token"]

    def is_logged_in(self) -> bool:
        """Check if user has stored credentials."""
        if not self.TOKEN_FILE.exists():
            return False
        try:
            with open(self.TOKEN_FILE) as f:
                creds = json.load(f)
            return bool(creds.get("refresh_token"))
        except Exception:
            return False


class GeminiGenerationAdapter:
    """
    Gemini generation provider for LLM calls.

    Kept separate from TCMM core - only used by console/orchestration layer.

    Three modes:
      1. API key mode (default)    — uses google.generativeai SDK with GEMINI_API_KEY
      2. Vertex AI SSO mode        — set GEMINI_SSO=1, uses gcloud auth + Vertex AI REST API
      3. Google OAuth mode         — set GEMINI_OAUTH=1, uses Google Sign-In + refresh tokens
    """

    def __init__(self, api_key: str = "", model_name: str = "models/gemini-3-flash-preview"):
        self.model_name = model_name
        self._mode = "Gemini API key"
        self._use_vertex = os.environ.get("GEMINI_SSO", "").strip() == "1"
        self._use_oauth = os.environ.get("GEMINI_OAUTH", "").strip() == "1"

        if self._use_oauth:
            self._oauth_manager = GeminiOAuthManager()
            if not self._oauth_manager.is_logged_in():
                raise RuntimeError("Not logged in. Run: python -c 'from TCMM.adapters.gemini_adapter import GeminiOAuthManager; GeminiOAuthManager().login()'")
            self._mode = "Gemini OAuth (Google Sign-In)"
            self.model = None  # Will use REST API instead
        elif self._use_vertex:
            self._project = os.environ.get("GCLOUD_PROJECT", "gen-lang-client-0085142705")
            self._region = os.environ.get("GCLOUD_REGION", "us-central1")
            self._vertex_model = _strip_models_prefix(model_name)
            self._mode = f"Vertex AI SSO ({self._project}/{self._region})"
            self.model = None  # Not using SDK
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)

    def _get_access_token(self):
        """Get fresh OAuth token from gcloud CLI."""
        import subprocess
        import sys
        # Windows needs shell=True for .cmd scripts (gcloud.cmd)
        result = subprocess.run(
            "gcloud auth print-access-token",
            capture_output=True, text=True, timeout=15,
            shell=(sys.platform == "win32"),
        )
        if result.returncode != 0:
            raise RuntimeError(f"gcloud auth failed: {result.stderr.strip()}")
        return result.stdout.strip()

    def _generate_via_oauth(self, prompt: str) -> str:
        """Generate via Gemini API using OAuth token. Tries AI Studio first, falls back to Vertex AI."""
        import json
        import urllib.request
        import urllib.error

        token = self._oauth_manager.get_access_token()
        model_id = _strip_models_prefix(self.model_name)
        project = os.environ.get("GCLOUD_PROJECT", "tcmm-cloud")

        payload = json.dumps({
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0}
        }).encode("utf-8")

        # Try AI Studio endpoint first (has latest preview models)
        ai_studio_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
        try:
            req = urllib.request.Request(
                ai_studio_url, data=payload, method="POST",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "x-goog-user-project": project,
                }
            )
            resp = urllib.request.urlopen(req, timeout=300)
            body = json.loads(resp.read().decode("utf-8"))
            candidates = body.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            return str(body)
        except urllib.error.HTTPError as e:
            if e.code not in (403, 404):
                raise
            # Fall through to Vertex AI

        # Fallback: Vertex AI endpoint
        region = os.environ.get("GCLOUD_REGION", "us-central1")
        vertex_url = (
            f"https://{region}-aiplatform.googleapis.com/v1/"
            f"projects/{project}/locations/{region}/"
            f"publishers/google/models/{model_id}:generateContent"
        )
        req = urllib.request.Request(
            vertex_url, data=payload, method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
        )
        resp = urllib.request.urlopen(req, timeout=300)
        body = json.loads(resp.read().decode("utf-8"))

        candidates = body.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "")
        return str(body)

    def _generate_via_vertex(self, prompt: str) -> str:
        """Generate via Vertex AI REST API using gcloud OAuth token."""
        import json
        import urllib.request
        import urllib.error

        token = self._get_access_token()
        model_id = _strip_models_prefix(self.model_name)
        url = (
            f"https://{self._region}-aiplatform.googleapis.com/v1/"
            f"projects/{self._project}/locations/{self._region}/"
            f"publishers/google/models/{model_id}:generateContent"
        )
        payload = json.dumps({
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0}
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload, method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
        )
        resp = urllib.request.urlopen(req, timeout=300)
        body = json.loads(resp.read().decode("utf-8"))

        # Extract text from response
        candidates = body.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "")
        return str(body)

    def generate(self, prompt: str, label: str = None) -> str:
        """Generate text completion."""
        import time

        with perf_scope("llm.generate_text"):
            max_retries = 3
            base_delay = 2.0

            for attempt in range(max_retries):
                try:
                    if self._use_oauth:
                        text = self._generate_via_oauth(prompt)
                    elif self._use_vertex:
                        text = self._generate_via_vertex(prompt)
                    else:
                        try:
                            resp = self.model.generate_content(
                                prompt,
                                generation_config={"temperature": 0.2},
                                request_options={"timeout": 300}
                            )
                        except (ValueError, TypeError):
                            # Newer google-generativeai removed request_options
                            resp = self.model.generate_content(
                                prompt,
                                generation_config={"temperature": 0.2},
                            )
                        if hasattr(resp, "text") and resp.text:
                            text = resp.text
                        else:
                            text = str(resp)

                    # ---- DEBUG TAP (restore old behaviour) ----
                    try:
                        log_adapter(f"Raw LLM output (first 800 chars): {text[:800]}")
                        if len(text) > 800:
                             log_adapter(f"Raw LLM output (last 800 chars): {text[-800:]}")
                    except Exception:
                        pass
                    # ------------------------------------------

                    return text
                except Exception as e:
                    err_str = str(e).lower()
                    retryable = (
                        "429" in err_str or "exhausted" in err_str
                        or "504" in err_str or "deadline" in err_str
                        or "503" in err_str or "unavailable" in err_str
                    )
                    if retryable and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        log_adapter(f"Gemini API error (retryable). Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        log_exception("Gemini generate failed", e)
                        return ""
            return ""
