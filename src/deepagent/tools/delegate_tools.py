"""Delegate tool — spawn sub-agents for parallel task execution.

When the main agent needs to handle multiple independent tasks simultaneously,
it can call delegate() multiple times in one turn. Since delegate is READONLY,
calls are executed in parallel (limited by SubAgentRunner's semaphore).
"""

from deepagent.tools.protocol import SafetyLevel
from deepagent.tools.registry import ToolRegistry, tool


def create_delegate_tools(registry: ToolRegistry, sub_agent_runner) -> list:
    """Create and register the delegate tool.

    Args:
        registry: ToolRegistry to register with
        sub_agent_runner: SubAgentRunner instance that executes delegated tasks

    Returns:
        List containing the delegate tool
    """
    runner = sub_agent_runner  # closure capture

    @tool(
        registry=registry,
        description="Delegate a complex subtask to a sub-agent that works autonomously. Use for independent tasks that can run in parallel (e.g., researching two different modules, refactoring separate files). The sub-agent has access to all tools. Returns the sub-agent's final response.",
        safety_level=SafetyLevel.READONLY,
    )
    async def delegate(description: str, prompt: str) -> dict:
        """Spawn a sub-agent to handle a delegated task.

        Args:
            description: Short label for what this sub-agent is doing (e.g., "Refactor auth module")
            prompt: Detailed instructions for the sub-agent, including what to do and what to return
        """
        return await runner.run(description, prompt)

    return [delegate]
