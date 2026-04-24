"""
LangChain Adapter for TCMM.
Allows using any LangChain compatible embedding model as a TCMM provider.
"""
from typing import List, Any
try:
    from src.memory.core.interfaces import EmbeddingProvider
    from src.memory.core.tcmm_logging import log_warning
except ImportError:
    from core.interfaces import EmbeddingProvider
    from core.tcmm_logging import log_warning

class LangChainAdapter:
    """
    Wraps a LangChain Embeddings model to satisfy the TCMM EmbeddingProvider protocol.
    """
    def __init__(self, embeddings_model: Any):
        """
        Args:
            embeddings_model: A LangChain Embeddings object (has embed_query/embed_documents).
                              Typed as Any to avoid hard dependency on langchain package.
        """
        self.model = embeddings_model
        
        # Duck-typing check
        if not hasattr(self.model, "embed_query") or not hasattr(self.model, "embed_documents"):
            log_warning("LangChainAdapter: Model provided does not look like a LangChain Embeddings model.")

    def embed(self, text: str) -> List[float]:
        """Generate single embedding."""
        # LangChain: embed_query(text) -> List[float]
        return self.model.embed_query(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings."""
        # LangChain: embed_documents(texts) -> List[List[float]]
        return self.model.embed_documents(texts)
