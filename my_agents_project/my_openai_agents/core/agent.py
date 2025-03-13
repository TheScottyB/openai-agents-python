"""Agent functionality."""
from typing import Optional, List, Type, Any
from pydantic import BaseModel

from .tool import Tool
from .models.openai_model import OpenAIModel

class Agent:
    """An agent that can perform tasks."""
    def __init__(
        self,
        name: str,
        instructions: str,
        tools: Optional[List[Tool]] = None,
        handoffs: Optional[List[str]] = None,
        output_type: Optional[Type[BaseModel]] = None,
        model: Optional[OpenAIModel] = None,
    ):
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        self.handoffs = handoffs or []
        self.output_type = output_type
        self.model = model

    def __str__(self) -> str:
        return f"Agent name: {self.name}\nAgent instructions: {self.instructions}\nAgent config: {self.to_dict()}"

    def to_dict(self) -> dict:
        """Convert agent to dictionary."""
        return {
            "name": self.name,
            "instructions": self.instructions,
            "tools": self.tools,
            "handoffs": self.handoffs,
            "output_type": self.output_type
        }
