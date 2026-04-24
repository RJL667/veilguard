"""
TCMM Base Adapter - Framework-neutral memory controller wrapper.

This is the canonical adapter that all framework-specific adapters wrap.
It exposes TCMM's public API without leaking any internal lifecycle logic.
"""


class TCMMMemoryAdapter:
    """
    Framework-neutral TCMM adapter.
    
    Usage:
        adapter = TCMMMemoryAdapter(tcmm)
        context = adapter.load_context(query="user question")
        adapter.save_turn("user message", "assistant response")
    """

    def __init__(self, tcmm):
        """
        Initialize adapter with a TCMM instance.
        
        Args:
            tcmm: A fully initialized TCMM instance
        """
        self.tcmm = tcmm

    def load_context(self, query: str = None) -> str:
        """
        Load prompt-ready memory context.
        
        TCMM handles all recall, reinjection, and eviction internally.
        
        Args:
            query: Optional task query to guide recall
            
        Returns:
            str: Concatenated live memory blocks
        """
        return self.tcmm.export_live_context(task_query=query)

    def save_turn(self, user_text: str, assistant_text: str):
        """
        Save a conversation turn to TCMM memory.
        
        Args:
            user_text: The user's input
            assistant_text: The assistant's response
        """
        if user_text:
            self.tcmm.observe(user_text, source="user")
        if assistant_text:
            self.tcmm.observe(assistant_text, source="assistant")

    def save_tool_output(self, tool_name: str, output: str):
        """
        Save tool output to TCMM memory.
        
        Args:
            tool_name: Name of the tool that produced the output
            output: The tool's output text
        """
        if output:
            self.tcmm.observe(f"[TOOL:{tool_name}] {output}", source="tool")

    def save_system_message(self, message: str):
        """
        Save a system message to TCMM memory.
        
        Args:
            message: The system message text
        """
        if message:
            self.tcmm.observe(message, source="system")
