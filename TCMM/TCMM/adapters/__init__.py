# TCMM Adapters Package
# Framework-agnostic memory adapters for TCMM integration

from .base import TCMMMemoryAdapter
from .gemini_adapter import GeminiGenerationAdapter
from .langchain_adapter import LangChainAdapter

__all__ = [
    "TCMMMemoryAdapter",

    "GeminiGenerationAdapter"
]

# Lazy imports to avoid mandatory dependencies
def get_langchain_adapter():
    from .langchain import TCMMLangChainMemory
    return TCMMLangChainMemory

def get_llamaindex_adapter():
    from .llamaindex import TCMMLlamaIndexContextProvider
    return TCMMLlamaIndexContextProvider
