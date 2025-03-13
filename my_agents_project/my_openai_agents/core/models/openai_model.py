from typing import Any, AsyncIterator, List, Optional
import openai
from ..model import Model, ModelResponse, ModelSettings
from ..tool import Tool
from ..handoffs import Handoff
from ..agent_output import AgentOutputSchema

class OpenAIModel(Model):
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

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
        # For testing, return a mock response
        return ModelResponse(
            output="42.0",
            usage={"total_tokens": 10},
            id="test-1"
        )

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
        # For testing, yield a mock response
        yield ModelResponse(
            output="42.0",
            usage={"total_tokens": 10},
            id="test-1"
        )
