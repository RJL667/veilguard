"""
TCMM LangChain Adapter - Drop-in replacement for LangChain memory.

IMPORTANT: Do NOT attach any LangChain memory, summary memory, or retrievers.
TCMM handles all recall and lifecycle logic internally.

Bug label: BUG-INTEGRATION-RECALL-LEAK
"""

try:
    from langchain.memory import BaseMemory
except ImportError:
    # Fallback for environments without LangChain
    class BaseMemory:
        """Stub BaseMemory for type checking when LangChain is not installed."""
        memory_variables = []
        def load_memory_variables(self, inputs): return {}
        def save_context(self, inputs, outputs): pass


class TCMMLangChainMemory(BaseMemory):
    """
    LangChain-compatible TCMM memory adapter.
    
    Usage:
        from src.memory.adapters.langchain import TCMMLangChainMemory
        
        memory = TCMMLangChainMemory(tcmm)
        chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    
    IMPORTANT: Do not attach any other LangChain memory alongside this.
    """

    def __init__(self, tcmm):
        """
        Initialize LangChain adapter with a TCMM instance.
        
        Args:
            tcmm: A fully initialized TCMM instance
        """
        self.tcmm = tcmm

    @property
    def memory_variables(self):
        """Return the list of memory variables provided by this memory."""
        return ["history"]

    def load_memory_variables(self, inputs):
        """
        Load memory context for the current chain invocation.
        
        Args:
            inputs: Dict containing chain inputs (looks for 'input' or 'query')
            
        Returns:
            Dict with 'history' key containing TCMM memory context
        """
        query = (
            inputs.get("input")
            or inputs.get("query")
            or ""
        )

        return {
            "history": self.tcmm.export_live_context(task_query=query)
        }

    def save_context(self, inputs, outputs):
        """
        Save chain context to TCMM memory.
        
        Args:
            inputs: Dict containing chain inputs
            outputs: Dict containing chain outputs
        """
        user = (
            inputs.get("input")
            or inputs.get("query")
        )

        assistant = (
            outputs.get("output")
            or outputs.get("result")
        )

        if user:
            self.tcmm.observe(user, source="user")

        if assistant:
            self.tcmm.observe(assistant, source="assistant")

    def clear(self):
        """
        Clear is a no-op for TCMM - lifecycle is managed internally.
        
        TCMM uses thermodynamic decay for automatic memory management.
        """
        pass
