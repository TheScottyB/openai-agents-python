"""Tool functionality."""
from typing import Optional, Type, Any, Callable, Dict
from pydantic import BaseModel

class ToolCall(BaseModel):
    """A tool call with parameters."""
    tool_name: str
    parameters: Optional[Dict[str, Any]] = None

class Tool:
    """A tool that can be used by an agent."""
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[Type[BaseModel]] = None,
        callback: Optional[Callable] = None
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.callback = callback

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        if self.callback is None:
            return {"error": "No callback defined"}
        
        if self.parameters:
            params = self.parameters(**kwargs)
            return await self.callback(params)
        return await self.callback()
