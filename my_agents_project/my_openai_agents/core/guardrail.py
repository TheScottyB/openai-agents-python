"""Guardrail implementation."""
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from ._utils import MaybeAwaitable
from .errors import GuardrailError

class Guardrail(BaseModel):
    """A guardrail for validating agent inputs and outputs."""
    name: str
    description: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)

    def validate(self) -> None:
        """Validate the guardrail configuration."""
        if not self.name:
            raise GuardrailError("Guardrail name is required")
