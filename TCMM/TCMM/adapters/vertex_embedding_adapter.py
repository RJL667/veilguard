"""
Vertex AI Embedding Adapter for TCMM.

Drop-in replacement for LocalEmbeddingAdapter. Uses Google's
text-embedding-005 API instead of a local SentenceTransformer model.

No GPU required. Pay per API call (~$0.025 per million tokens).

Usage:
    from adapters.vertex_embedding_adapter import VertexEmbeddingAdapter
    embedder = VertexEmbeddingAdapter(project_id="my-project", region="us-central1")
    vec = embedder.embed("Hello world")
"""

import json
import logging
import os
import time
import threading
from typing import List, Optional

logger = logging.getLogger("tcmm.vertex-embedding")


class VertexEmbeddingAdapter:
    """Vertex AI text-embedding-005 adapter. Same interface as LocalEmbeddingAdapter."""

    def __init__(
        self,
        project_id: str = "",
        region: str = "us-central1",
        model: str = "text-embedding-005",
        api_key: str = "",
    ):
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self.region = region
        self.model = model
        self.api_key = api_key or os.environ.get("VERTEX_API_KEY", "")
        self.dimension = 768  # text-embedding-005 outputs 768 dimensions
        self._base_url = (
            f"https://{region}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{region}/"
            f"publishers/google/models/{model}"
        )

        self._token_lock = threading.Lock()

        logger.info(f"[VertexEmbedding] project={self.project_id} model={model} dim={self.dimension} auth={'api_key' if self.api_key else 'oauth'}")

    def _get_oauth_token(self) -> str:
        """Get OAuth token from ADC. Thread-safe."""
        with self._token_lock:
            if hasattr(self, '_oauth_token') and self._oauth_token and time.time() < self._oauth_expiry - 60:
                return self._oauth_token
            import google.auth
            import google.auth.transport.requests
            creds, _ = google.auth.default()
            creds.refresh(google.auth.transport.requests.Request())
            self._oauth_token = creds.token
            self._oauth_expiry = creds.expiry.timestamp() if creds.expiry else time.time() + 3600
            return self._oauth_token

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Call the Vertex AI embedding API. Retries on 429 with backoff."""
        import httpx
        instances = [{"content": t[:2048]} for t in texts]

        url = f"{self._base_url}:predict"
        if self.api_key:
            url += f"?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._get_oauth_token()}",
            }

        body = {"instances": instances}

        for attempt in range(4):
            resp = httpx.post(url, headers=headers, json=body, timeout=30)
            if resp.status_code == 429:
                wait = 2 ** attempt + 1
                logger.warning(f"[VertexEmbedding] 429 rate limit, retrying in {wait}s (attempt {attempt+1}/4)")
                time.sleep(wait)
                if not self.api_key:
                    headers["Authorization"] = f"Bearer {self._get_oauth_token()}"
                continue
            break

        if resp.status_code != 200:
            logger.error(f"[VertexEmbedding] API error {resp.status_code}: {resp.text[:200]}")
            return [[] for _ in texts]

        data = resp.json()
        predictions = data.get("predictions", [])
        return [p["embeddings"]["values"] for p in predictions]

    def validate_embedding_model(self):
        """No-op for API-based adapter. Local adapter uses this to verify GPU model."""
        pass

    def embed(self, text: str) -> List[float]:
        """Embed a single passage."""
        if not text:
            return []
        results = self._call_api([text])
        return results[0] if results else []

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of passages. Handles Vertex AI's 250-instance limit."""
        if not texts:
            return []
        all_results = []
        # Vertex AI: 250 instance limit but 20K token limit — use 50 to stay safe
        for i in range(0, len(texts), 50):
            chunk = texts[i:i + 250]
            results = self._call_api(chunk)
            all_results.extend(results)
        return all_results

    def embed_query(self, text: str) -> List[float]:
        """Embed a query (same as passage for this model)."""
        return self.embed(text)

    def embed_query_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of queries."""
        return self.embed_batch(texts)
