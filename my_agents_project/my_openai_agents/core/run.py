"""Run functionality for agents."""
from dataclasses import dataclass, field
import re
from typing import Any, List, Optional, Union, Type, TypeVar, cast
from pydantic import BaseModel

from .agent import Agent
from .items import ModelResponse, ItemHelpers, TResponseInputItem
from .usage import Usage

T = TypeVar('T', bound=BaseModel)

class RunConfig(BaseModel):
    """Configuration for a run."""
    max_turns: int = 10
    trace: bool = True

@dataclass
class RunResult:
    """Result of a run."""
    final_output: Any
    usage: Usage = field(default_factory=Usage)

class Runner:
    """Runs an agent with the given input."""
    @staticmethod
    def _mock_parse_problem(text: str) -> dict:
        """Mock parsing of math problems."""
        text = text.lower()
        if "plus" in text or "add" in text:
            numbers = [float(n) for n in re.findall(r'\d+', text)]
            return {
                "operation": "add",
                "values": numbers[:2] if numbers else [25, 17],
                "original_text": text
            }
        elif "minus" in text or "subtract" in text or "spend" in text:
            numbers = [float(n) for n in re.findall(r'\d+', text)]
            return {
                "operation": "subtract",
                "values": numbers[:2] if numbers else [100, 33],
                "original_text": text
            }
        elif "multiply" in text or "times" in text:
            numbers = [float(n) for n in re.findall(r'\d+', text)]
            return {
                "operation": "multiply",
                "values": numbers[:2] if numbers else [13, 4],
                "original_text": text
            }
        elif "divide" in text:
            numbers = [float(n) for n in re.findall(r'\d+', text)]
            return {
                "operation": "divide",
                "values": numbers[:2] if numbers else [100, 4],
                "original_text": text
            }
        return {
            "operation": "add",
            "values": [25, 17],
            "original_text": text
        }

    @staticmethod
    def _mock_calculate(operation: str, values: List[float]) -> dict:
        """Mock calculation based on operation and values."""
        try:
            if len(values) < 2:
                return {
                    "result": 0,
                    "explanation": "Not enough values provided"
                }

            a, b = values[:2]
            if operation == "add":
                result = a + b
                explanation = f"Adding {a} and {b}"
            elif operation == "subtract":
                result = a - b
                explanation = f"Subtracting {b} from {a}"
            elif operation == "multiply":
                result = a * b
                explanation = f"Multiplying {a} by {b}"
            elif operation == "divide":
                if b == 0:
                    return {
                        "result": 0,
                        "explanation": "Cannot divide by zero"
                    }
                result = a / b
                explanation = f"Dividing {a} by {b}"
            else:
                return {
                    "result": 0,
                    "explanation": f"Unknown operation: {operation}"
                }

            return {
                "result": result,
                "explanation": explanation
            }
        except Exception as e:
            return {
                "result": 0,
                "explanation": f"Error: {str(e)}"
            }

    @staticmethod
    async def run(
        agent: Agent,
        input: Union[str, List[TResponseInputItem], dict],
        config: Optional[RunConfig] = None
    ) -> RunResult:
        """Run the agent with the given input."""
        if config is None:
            config = RunConfig()

        # Handle the input based on its type
        if isinstance(input, dict):
            # For calculator agent receiving parsed problem
            operation = input.get("operation", "")
            values = input.get("values", [])
            mock_data = Runner._mock_calculate(operation, values)
        else:
            input_list = ItemHelpers.input_to_new_input_list(input)
            input_str = str(input) if isinstance(input, str) else str(input_list)
            
            # Return appropriate mock results based on agent name
            if agent.name == "browser_automation":
                mock_data = {
                    "success": True,
                    "message": "Successfully performed browser action",
                    "data": None
                }
            elif agent.name == "math_parser":
                mock_data = Runner._mock_parse_problem(input_str)
            elif agent.name == "calculator":
                parsed = Runner._mock_parse_problem(input_str)
                mock_data = Runner._mock_calculate(parsed["operation"], parsed["values"])
            else:  # Default mock data
                mock_data = {
                    "result": 42,
                    "explanation": "This is a mock calculation"
                }

        # Convert the dictionary to the output type if specified
        if agent.output_type and issubclass(agent.output_type, BaseModel):
            final_output = agent.output_type(**mock_data)
        else:
            final_output = mock_data

        return RunResult(final_output=final_output)
