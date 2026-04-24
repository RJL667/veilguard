"""
TCMM LlamaIndex Adapter - Context provider for LlamaIndex integration.

Inject build_context() directly into your prompt template in LlamaIndex.
Do NOT use any LlamaIndex retriever - TCMM handles all recall internally.
"""


class TCMMLlamaIndexContextProvider:
    """
    LlamaIndex-compatible TCMM context provider.
    
    Usage:
        from src.memory.adapters.llamaindex import TCMMLlamaIndexContextProvider
        
        provider = TCMMLlamaIndexContextProvider(tcmm)
        
        # In your prompt template:
        context = provider.build_context(query)
        prompt = f"MEMORY:\\n{context}\\n\\nQUERY: {query}"
    
    IMPORTANT: Do not use any LlamaIndex retriever alongside this provider.
    """

    def __init__(self, tcmm):
        """
        Initialize LlamaIndex adapter with a TCMM instance.
        
        Args:
            tcmm: A fully initialized TCMM instance
        """
        self.tcmm = tcmm

    def build_context(self, query: str) -> str:
        """
        Build context string for prompt injection.
        
        TCMM handles recall, reinjection, and eviction internally.
        
        Args:
            query: The current user query
            
        Returns:
            str: Prompt-ready memory context
        """
        return self.tcmm.export_live_context(task_query=query)

    def observe_turn(self, user_text: str, assistant_text: str):
        """
        Observe a conversation turn after LLM response.
        
        Args:
            user_text: The user's input
            assistant_text: The assistant's response
        """
        if user_text:
            self.tcmm.observe(user_text, source="user")
        if assistant_text:
            self.tcmm.observe(assistant_text, source="assistant")

    def observe_retrieval(self, retrieved_text: str):
        """
        Observe externally retrieved content (if any).
        
        Note: TCMM's internal recall is preferred. Only use this for
        content from external sources (e.g., web search).
        
        Args:
            retrieved_text: Externally retrieved content
        """
        if retrieved_text:
            self.tcmm.observe(f"[EXTERNAL] {retrieved_text}", source="tool")

    def get_memory_stats(self) -> dict:
        """
        Get current memory statistics for debugging.
        
        Returns:
            dict: Memory statistics
        """
        return {
            "live_blocks": len(self.tcmm.live_blocks),
            "archive_size": len(self.tcmm.archive),
            "current_step": self.tcmm.current_step,
            "live_tokens": sum(b.token_count for b in self.tcmm.live_blocks)
        }
