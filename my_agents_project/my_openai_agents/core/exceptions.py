"""Custom exceptions for the agents package."""

class AgentsException(Exception):
    """Base exception for all agents errors."""
    pass

class ModelBehaviorError(AgentsException):
    """Raised when the model does something unexpected."""
    pass

class UserError(AgentsException):
    """Raised when there's an error in user-provided code or configuration."""
    pass

class MaxTurnsExceeded(AgentsException):
    """Raised when the maximum number of turns is exceeded."""
    pass

class GuardrailTripwireTriggered(AgentsException):
    """Raised when a guardrail tripwire is triggered."""
    def __init__(self, guardrail_result):
        self.guardrail_result = guardrail_result
        super().__init__("Guardrail tripwire triggered")
