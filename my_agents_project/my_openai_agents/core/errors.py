"""Exceptions for the agents package."""

class AgentError(Exception):
    """Base class for all agent errors."""
    pass

class HandoffError(AgentError):
    """Error during agent handoff."""
    pass

class GuardrailError(AgentError):
    """Error during guardrail validation."""
    pass

class ToolError(AgentError):
    """Error during tool execution."""
    pass
