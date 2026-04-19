"""AskUserQuestion tool — agent asks the user a question mid-execution.

In LibreChat, this returns the question as the tool result, prompting the
LLM to present it to the user. The user's next message becomes the answer.
This mimics Claude Code's AskUserQuestion tool.
"""


def register(mcp):
    @mcp.tool()
    async def ask_user(question: str, options: str = "") -> str:
        """Ask the user a question and wait for their response.

        WHEN TO USE: When you need clarification, a decision, or user input before proceeding.
        WHEN NOT TO USE: For rhetorical questions or when you can make a reasonable assumption.

        Args:
            question: The question to ask the user
            options: Optional comma-separated choices (e.g. "Option A, Option B, Option C")
        """
        result = f"**Question for you:**\n\n{question}"
        if options:
            opts = [o.strip() for o in options.split(",") if o.strip()]
            result += "\n\nOptions:\n"
            for i, opt in enumerate(opts, 1):
                result += f"  {i}. {opt}\n"
            result += "\nPlease reply with your choice."
        else:
            result += "\n\nPlease reply with your answer."
        return result
