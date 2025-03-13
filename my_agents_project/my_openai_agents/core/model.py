from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, List, Optional
from pydantic import BaseModel

from .tool import Tool
from .handoffs import Handoff
from .agent_output import AgentOutputSchema

class ModelSettings(BaseModel):
    """Settings for model execution."""
    temperature: float = 0.7
    max_tokens: Optional[int] = None

class ModelResponse(BaseModel):
    """Response from a model."""
    output: str
    usage: Optional[dict[str, int]] = None
    id: Optional[str] = None

class Model(ABC):
    """Abstract base class for models."""
    
    @abstractmethod
    async def get_response(
        self,
        system_instructions: Optional[str],
        input: List[Any],
        model_settings: ModelSettings,
        tools: List[Tool],
        output_schema: Optional[AgentOutputSchema],
        handoffs: List[Handoff],
        tracing: bool = True,
    ) -> ModelResponse:
        """Get a response from the model."""
        pass

    @abstractmethod
    async def stream_response(
        self,
        system_instructions: Optional[str],
        input: List[Any],
        model_settings: ModelSettings,
        tools: List[Tool],
        output_schema: Optional[AgentOutputSchema],
        handoffs: List[Handoff],
        tracing: bool = True,
    ) -> AsyncIterator[Any]:
        """Stream a response from the model."""
        pass
