#!/usr/bin/env python3

"""
Combined Agents Demo
===================

This example demonstrates how to use multiple specialized agents together:
1. A Calculator Agent with unit conversion and constants
2. A Customer Service Agent that uses the calculator as a tool

It showcases:
- Using one agent as a tool for another agent
- Input and output guardrails
- Lifecycle hooks for monitoring
- Proper error handling

To run:
    python combined_agents_demo.py

Requirements:
    - OpenAI API key set as OPENAI_API_KEY environment variable
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

# Import from the OpenAI Agents SDK
from agents import (
    Agent,
    AgentHooks,
    ModelSettings,
    Runner,
    function_tool,
    input_guardrail,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    GuardrailFunctionOutput,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("combined_agents")


# -----------------------------------------------------------------------------
# SECTION 1: Context and Output Types
# -----------------------------------------------------------------------------

@dataclass
class CustomerContext:
    """Context for tracking customer interactions."""

    customer_id: str
    username: str = field(default="User")
    premium_user: bool = False
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    session_start_time: datetime = field(default_factory=datetime.now)

    def add_to_history(self, role: str, content: str) -> None:
        """Add an interaction to the conversation history."""
        self.conversation_history.append(
            {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        )

    def log_interaction(
        self, tool_name: str, input_data: Any, output_data: Any
    ) -> None:
        """Log an interaction with a tool to the conversation history."""
        self.conversation_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "tool": tool_name,
                "input": input_data,
                "output": output_data,
            }
        )


class SupportCategory(str, Enum):
    """Categories for support tickets."""

    CALCULATION = "calculation"
    UNIT_CONVERSION = "unit_conversion"
    CONSTANT_INFO = "constant_info"
    GENERAL = "general"


# -----------------------------------------------------------------------------
# SECTION 2: Unit Conversion and Math Constants
# -----------------------------------------------------------------------------

class UnitConverter:
    """Tool for converting between different units of measurement."""
    
    @staticmethod
    def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
        """Convert temperature between Celsius, Fahrenheit and Kelvin."""
        conversions = {
            ("c", "f"): lambda x: x * 9 / 5 + 32,
            ("f", "c"): lambda x: (x - 32) * 5 / 9,
            ("c", "k"): lambda x: x + 273.15,
            ("k", "c"): lambda x: x - 273.15,
            ("f", "k"): lambda x: (x - 32) * 5 / 9 + 273.15,
            ("k", "f"): lambda x: (x - 273.15) * 9 / 5 + 32,
        }
        key = (from_unit.lower()[0], to_unit.lower()[0])
        if key not in conversions:
            raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")
        return conversions[key](value)

    @staticmethod
    def convert_length(value: float, from_unit: str, to_unit: str) -> float:
        """Convert length between meters, feet, inches, and miles."""
        units = {"m": 1.0, "ft": 0.3048, "in": 0.0254, "mi": 1609.34}
        try:
            return value * units[from_unit] / units[to_unit]
        except KeyError:
            raise ValueError(f"Invalid units: {from_unit} to {to_unit}")

    @staticmethod
    def convert_weight(value: float, from_unit: str, to_unit: str) -> float:
        """Convert weight between kilograms, pounds, and ounces."""
        units = {"kg": 1.0, "lb": 0.453592, "oz": 0.0283495}
        if from_unit not in units or to_unit not in units:
            raise ValueError(f"Invalid weight units: {from_unit} to {to_unit}")
        return value * units[from_unit] / units[to_unit]


class MathConstant:
    """Tool providing mathematical constants and their explanations."""

    CONSTANTS: Dict[str, Tuple[float, str]] = {
        "pi": (3.14159265359, "The ratio of a circle's circumference to its diameter"),
        "e": (2.71828182846, "The base of the natural logarithm"),
        "golden_ratio": (
            1.61803398875,
            "The ratio where the ratio of the sum of quantities to the larger quantity "
            "is equal to the ratio of the larger quantity to the smaller one",
        ),
        "avogadro": (
            6.02214076e23,
            "The number of constituent particles in one mole of a substance",
        ),
        "plank": (
            6.62607015e-34,
            "The fundamental quantum of action in quantum mechanics",
        ),
    }

    @classmethod
    def get_constant(cls, name: str) -> Tuple[float, str]:
        """Retrieve a mathematical constant and its explanation."""
        name = name.lower()
        if name not in cls.CONSTANTS:
            raise ValueError(f"Unknown constant: {name}")
        return cls.CONSTANTS[name]


# -----------------------------------------------------------------------------
# SECTION 3: Agent Tools
# -----------------------------------------------------------------------------

@function_tool
def convert_temperature(
    context: RunContextWrapper[CustomerContext], value: float, from_unit: str, to_unit: str
) -> Dict[str, Any]:
    """Convert temperature between Celsius, Fahrenheit and Kelvin."""
    try:
        converter = UnitConverter()
        result = converter.convert_temperature(value, from_unit, to_unit)

        # Log the interaction
        context.context.log_interaction(
            "convert_temperature",
            {"value": value, "from_unit": from_unit, "to_unit": to_unit},
            result,
        )

        return {
            "original_value": value,
            "original_unit": from_unit,
            "converted_value": result,
            "converted_unit": to_unit,
            "formula_used": f"Conversion formula from {from_unit} to {to_unit}",
            "success": True,
        }
    except ValueError as e:
        return {"success": False, "error": str(e)}


@function_tool
def convert_length(
    context: RunContextWrapper[CustomerContext], value: float, from_unit: str, to_unit: str
) -> Dict[str, Any]:
    """Convert length between meters, feet, inches, and miles."""
    try:
        converter = UnitConverter()
        result = converter.convert_length(value, from_unit, to_unit)

        # Log the interaction
        context.context.log_interaction(
            "convert_length",
            {"value": value, "from_unit": from_unit, "to_unit": to_unit},
            result,
        )

        return {
            "original_value": value,
            "original_unit": from_unit,
            "converted_value": result,
            "converted_unit": to_unit,
            "success": True,
        }
    except ValueError as e:
        return {"success": False, "error": str(e)}


@function_tool
def convert_weight(
    context: RunContextWrapper[CustomerContext], value: float, from_unit: str, to_unit: str
) -> Dict[str, Any]:
    """Convert weight between kilograms, pounds, and ounces."""
    try:
        converter = UnitConverter()
        result = converter.convert_weight(value, from_unit, to_unit)

        # Log the interaction
        context.context.log_interaction(
            "convert_weight",
            {"value": value, "from_unit": from_unit, "to_unit": to_unit},
            result,
        )

        return {
            "original_value": value,
            "original_unit": from_unit,
            "converted_value": result,
            "converted_unit": to_unit,
            "success": True,
        }
    except ValueError as e:
        return {"success": False, "error": str(e)}


@function_tool
def get_math_constant(
    context: RunContextWrapper[CustomerContext], name: str
) -> Dict[str, Any]:
    """Get information about a mathematical constant."""
    try:
        math_constants = MathConstant()
        value, description = math_constants.get_constant(name)

        # Log the interaction
        context.context.log_interaction(
            "get_math_constant",
            {"name": name},
            {"value": value, "description": description},
        )

        return {
            "constant_name": name,
            "value": value,
            "description": description,
            "success": True,
        }
    except ValueError as e:
        return {"success": False, "error": str(e)}


@function_tool
def get_user_details(context: RunContextWrapper[CustomerContext]) -> Dict[str, Any]:
    """Get information about the current user."""
    user = context.context

    return {
        "customer_id": user.customer_id,
        "username": user.username,
        "is_premium": user.premium_user,
        "conversation_count": len(user.conversation_history),
    }


# -----------------------------------------------------------------------------
# SECTION 4: Lifecycle Hooks
# -----------------------------------------------------------------------------

class AgentLifecycleHooks(AgentHooks):
    """Custom hooks for monitoring agent lifecycle."""

    def __init__(self):
        self.start_time = None
        self.tools_called = 0

    async def on_start(
        self, context: RunContextWrapper[CustomerContext], agent: Agent[CustomerContext]
    ) -> None:
        """Called when an agent run starts."""
        self.start_time = datetime.now()
        logger.info(f"Agent run started: {agent.name}")
        logger.info(
            f"User: {context.context.username} (Premium: {context.context.premium_user})"
        )

    async def on_end(
        self,
        context: RunContextWrapper[CustomerContext],
        agent: Agent[CustomerContext],
        output: Any,
    ) -> None:
        """Called when an agent run completes."""
        duration = datetime.now() - self.start_time if self.start_time else None
        logger.info(f"Agent run completed: {agent.name}")
        logger.info(f"Duration: {duration}")
        logger.info(f"Tools called: {self.tools_called}")

    async def on_tool_start(
        self,
        context: RunContextWrapper[CustomerContext],
        agent: Agent[CustomerContext],
        tool: Any,
    ) -> None:
        """Called when a tool is invoked."""
        self.tools_called += 1
        logger.info(f"Tool called: {tool.name}")


# -----------------------------------------------------------------------------
# SECTION 5: Guardrails
# -----------------------------------------------------------------------------

@input_guardrail
async def calculation_guardrail(
    context: RunContextWrapper[CustomerContext],
    agent: Agent[Any],
    input_text: Union[str, List[Any]],
) -> GuardrailFunctionOutput:
    # Define keywords for valid calculation queries
    calculation_keywords = [
        "convert",
        "calculate",
        "compute",
        "solve",
        "math",
        "constant",
        "pi",
        "celsius",
        "fahrenheit",
        "kelvin",
        "meter",
        "foot",
        "feet",
        "inch",
        "mile",
        "kilogram",
        "pound",
        "ounce",
    ]

    # Check if any calculation keyword is in the input
    input_str = input_text if isinstance(input_text, str) else str(input_text)
    if any(keyword in input_str.lower() for keyword in calculation_keywords):
        return GuardrailFunctionOutput(
            output_info="Input contains calculation-related keywords",
            tripwire_triggered=False,
        )

    # Minimal query length check
    if len(input_str.split()) < 3:
        return GuardrailFunctionOutput(
            output_info="Please provide a more detailed query for calculation or conversion.",
            tripwire_triggered=True,
        )

    # Default to valid if we're not sure
    return GuardrailFunctionOutput(
        output_info="Input passed default validation", tripwire_triggered=False
    )


# -----------------------------------------------------------------------------
# SECTION 6: Agent Implementations
# -----------------------------------------------------------------------------

def create_calculator_agent() -> Agent[CustomerContext]:
    """Create the calculator agent."""
    return Agent[CustomerContext](
        name="Calculator Helper",
        instructions="""You are a specialized calculator agent that helps with:
1. Temperature conversions (Celsius, Fahrenheit, Kelvin)
2. Length conversions (meters, feet, inches, miles)
3. Weight conversions (kilograms, pounds, ounces)
4. Math constants (Pi, E, Golden Ratio, etc.)

Always provide precise calculations and include units in your responses.
If a conversion or calculation isn't possible, clearly explain why.
For constants, include both the value and a brief explanation of its significance.
""",
        model="gpt-4",
        model_settings=ModelSettings(temperature=0.1),
        tools=[
            convert_temperature,
            convert_length,
            convert_weight,
            get_math_constant,
            get_user_details,
        ],
        hooks=AgentLifecycleHooks(),
        input_guardrails=[calculation_guardrail],
    )


def create_customer_service_agent() -> Agent[CustomerContext]:
    """Create the customer service agent."""
    return Agent[CustomerContext](
        name="Customer Service Assistant",
        instructions="""You are a helpful customer service assistant who can help with various queries including calculations and conversions.

For calculation and conversion requests, use your calculator tools to provide accurate results.
Always be polite, clear, and professional when responding to customer inquiries.

When helping customers:
- Use appropriate units in all conversions
- Explain mathematical concepts clearly
- Check if the user is a premium user and provide more detailed responses if they are
- Be concise but thorough in your explanations
""",
        model="gpt-4",
        model_settings=ModelSettings(temperature=0.7),
        tools=[
            convert_temperature,
            convert_length,
            convert_weight,
            get_math_constant,
            get_user_details,
        ],
        hooks=AgentLifecycleHooks(),
        input_guardrails=[calculation_guardrail],
    )


# -----------------------------------------------------------------------------
# SECTION 7: Demo Scenarios
# -----------------------------------------------------------------------------

async def run_conversion_inquiry(agent: Agent[CustomerContext]) -> None:
    """Run a scenario where a customer asks about unit conversions."""
    context = CustomerContext(customer_id="cust_12345", username="Alice")

    # Temperature conversion inquiry
    user_message = "I need to convert 100 degrees Celsius to Fahrenheit for my recipe. Can you help me?"
    print(f"\n=== Temperature Conversion Inquiry ===\nCustomer: {user_message}")

    try:
        result = await Runner.run(agent, user_message, context=context)
        print(f"Agent: {result.final_output}")
    except Exception as e:
        print(f"Error: {e}")

    # Length conversion inquiry
    user_message = "How many feet are in 10 meters? I'm trying to figure out if my new couch will fit in my living room."
    print(f"\n=== Length Conversion Inquiry ===\nCustomer: {user_message}")

    try:
        result = await Runner.run(agent, user_message, context=context)
        print(f"Agent: {result.final_output}")
    except Exception as e:
        print(f"Error: {e}")


async def run_constant_inquiry(agent: Agent[CustomerContext]) -> None:
    """Run a scenario where a customer asks about mathematical constants."""
    context = CustomerContext(customer_id="cust_67890", username="Bob")

    # Math constant inquiry
    user_message = "I'm working on a math project and need the value of Pi. Can you tell me what it is and why it's important?"
    print(f"\n=== Mathematical Constant Inquiry ===\nCustomer: {user_message}")

    try:
        result = await Runner.run(agent, user_message, context=context)
        print(f"Agent: {result.final_output}")
    except Exception as e:
        print(f"Error: {e}")

    # Multiple constants inquiry
    user_message = "Can you tell me about the golden ratio and how it relates to the Fibonacci sequence? Also, what is Avogadro's number used for?"
    print(f"\n=== Multiple Constants Inquiry ===\nCustomer: {user_message}")

    try:
        result = await Runner.run(agent, user_message, context=context)
        print(f"Agent: {result.final_output}")
    except Exception as e:
        print(f"Error: {e}")


async def run_mixed_inquiry(agent: Agent[CustomerContext]) -> None:
    """Run a scenario with mixed inquiries about conversions and constants."""
    context = CustomerContext(customer_id="cust_54321", username="Charlie", premium_user=True)

    user_message = """I'm working on a physics project and need some help:
    1. What is Planck's constant and why is it significant?
    2. I need to convert 212°F to Kelvin for my experiment.
    3. Also, how many kilograms are in 150 pounds?
    Thanks for your help!"""

    print(f"\n=== Mixed Inquiry (Premium User) ===\nCustomer: {user_message}")

    try:
        result = await Runner.run(agent, user_message, context=context)
        print(f"Agent: {result.final_output}")
    except Exception as e:
        print(f"Error: {e}")


# -----------------------------------------------------------------------------
# SECTION 8: Main Execution
# -----------------------------------------------------------------------------

async def main() -> None:
    """Main execution function."""
    print("=== Math-Enabled Customer Service Agent Demo ===")

    # Create the agents
    calculator_agent = create_calculator_agent()
    customer_service_agent = create_customer_service_agent()

    # Run demonstration scenarios with the customer service agent
    await run_conversion_inquiry(customer_service_agent)
    await run_constant_inquiry(customer_service_agent)
    await run_mixed_inquiry(customer_service_agent)

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())