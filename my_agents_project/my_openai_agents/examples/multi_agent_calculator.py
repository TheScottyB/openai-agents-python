"""Multi-agent calculator example."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import asyncio

from my_openai_agents.core.agent import Agent
from my_openai_agents.core.run import Runner
from my_openai_agents.core.models.openai_model import OpenAIModel


class ParsedMathProblem(BaseModel):
    """Parsed math problem with operation and values."""
    operation: str
    values: List[float]
    original_text: str


class CalculationResult(BaseModel):
    """Result of a calculation with explanation."""
    result: float
    explanation: str


def create_parser_agent() -> Agent:
    """Create an agent that parses math problems from text."""
    return Agent(
        name="math_parser",
        instructions="""You are a math problem parser.
        Extract the mathematical operation and values from the text.
        Return the operation (add, subtract, multiply, divide) and the values.""",
        output_type=ParsedMathProblem,
    )


def create_calculator_agent() -> Agent:
    """Create an agent that performs calculations."""
    return Agent(
        name="calculator",
        instructions="""You are a calculator.
        Perform the specified mathematical operation on the given values.
        Explain your calculation process.""",
        output_type=CalculationResult,
    )


async def solve_math_problem(problem: str) -> None:
    """Solve a math problem using multiple agents."""
    # Create agents
    parser = create_parser_agent()
    calculator = create_calculator_agent()

    print(f"\nOriginal question: {problem}")

    try:
        # First, parse the problem
        parsed = await Runner.run(parser, problem)
        parsed_result = parsed.final_output
        print(f"Parsed as: Operation={parsed_result.operation}, Values={parsed_result.values}")

        # Convert parsed result to dict for calculator
        calc_input: Dict[str, Any] = {
            "operation": parsed_result.operation,
            "values": parsed_result.values
        }

        # Calculate the result using the parsed values
        calc_result = await Runner.run(calculator, calc_input)
        result = calc_result.final_output
        print(f"Answer: {result.result}\nExplanation: {result.explanation}")

    except Exception as e:
        print(f"Error: {str(e)}")


async def main():
    """Run example math problems."""
    problems = [
        "What is twenty-five plus seventeen?",
        "If I divide one hundred by four, what do I get?",
        "Multiply thirteen by four please",
        "I have 100 dollars and spend 33, how much is left?"
    ]

    for problem in problems:
        await solve_math_problem(problem)


def run():
    """Run the multi-agent calculator example."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
