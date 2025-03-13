"""Agent handoff functionality."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .exceptions import ModelBehaviorError, UserError
from .items import RunItem, TResponseInputItem

class Handoff(BaseModel):
    """Configuration for an agent handoff."""
    name: str
    description: str
    output_schema: Optional[Any] = None
    config: Dict[str, Any] = Field(default_factory=dict)

    def validate(self) -> None:
        """Validate the handoff configuration."""
        if not self.name:
            raise UserError("Handoff name is required")
        if not self.description:
            raise UserError("Handoff description is required")

    def matches(self, name: str) -> bool:
        """Check if this handoff matches the given name."""
        return self.name.lower() == name.lower()
