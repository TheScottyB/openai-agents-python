"""Calculator agent example."""
from typing import Optional
import asyncio
import os
from pydantic import BaseModel
from my_openai_agents.core.agent import Agent
from my_openai_agents.core.tool import Tool
from my_openai_agents.core.run import Runner, RunConfig
from my_openai_agents.core.models.openai_model import OpenAIModel

class CalculationResult(BaseModel):
    """Result of a calculation."""
    result: float
    explanation: Optional[str] = None

    def __str__(self) -> str:
        """String representation of the calculation result."""
        output = f"{self.result}"
        if self.explanation:
            output += f"\nExplanation: {self.explanation}"
        return output

def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

async def main():
    # Create tools for basic arithmetic
    tools = [
        Tool.from_function(add),
        Tool.from_function(subtract),
        Tool.from_function(multiply),
        Tool.from_function(divide),
    ]

    # Create the OpenAI model
    model = OpenAIModel(api_key=os.getenv('OPENAI_API_KEY', 'test_key'))

    # Create a calculator agent
    agent = Agent(
        name="calculator",
        instructions="""You are a helpful calculator assistant. You can perform basic arithmetic operations.
        When asked to calculate something, use the appropriate tool and explain your reasoning.
        Always show your work and explain how you arrived at the answer.""",
        tools=tools,
        output_type=CalculationResult,
        model=model
    )

    # Test calculations
    calculations = [
        "What is 25 plus 17?",
        "Calculate 45 divided by 5",
        "What is 13 times 4?",
        "If I have 100 and subtract 33, what's left?"
    ]

    for calc in calculations:
        print(f"\nQuestion: {calc}")
        try:
            result = await Runner.run(agent, calc)
            output: CalculationResult = result.final_output
            print(str(output))
        except Exception as e:
            print(f"Error: {str(e)}")

def run():
    """Run the calculator example."""
    asyncio.run(main())

if __name__ == "__main__":
    run()
