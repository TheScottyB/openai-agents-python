"""Agent output schema handling."""
from typing import Any, Type, get_origin

class AgentOutputSchema:
    """Schema for agent outputs."""
    def __init__(self, output_type: Type[Any]):
        self.output_type = output_type

    def output_type_name(self) -> str:
        """Get the name of the output type."""
        origin = get_origin(self.output_type)
        if origin:
            return str(origin)
        return self.output_type.__name__
