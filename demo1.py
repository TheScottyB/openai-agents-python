"""
Combined Agents Demo - Calculator + Customer Service

This example demonstrates how to create a Customer Service Agent that uses
a Calculator Agent as a tool. The Calculator Agent provides unit conversions
and mathematical constants, which the Customer Service Agent can use to answer
customer queries.

Usage:
    python combined_agents_demo.pyw

Requirements:
    - OpenAI API key set as OPENAI_API_KEY environment variable
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Union

from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai: pip install openai")

try:
    from agents import Agent, function_tool, InlineToolDefinition
except ImportError:
    raise ImportError("Please install openai-agents: pip install openai-agents")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Check for OpenAI API key
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# ----- CALCULATOR AGENT ----- #


class UnitConverter:
    """Tool for handling unit conversions."""

    def convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert temperature between Celsius, Fahrenheit, and Kelvin."""
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()

        # Convert to Kelvin first (base unit)
        if from_unit == "c":
            value_k = value + 273.15
        elif from_unit == "f":
            value_k = (value - 32) * 5 / 9 + 273.15
        elif from_unit == "k":
            value_k = value
        else:
            raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")

        # Convert from Kelvin to target unit
        if to_unit == "c":
            return value_k - 273.15
        elif to_unit == "f":
            return (value_k - 273.15) * 9 / 5 + 32
        elif to_unit == "k":
            return value_k
        else:
            raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")

    def convert_length(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert length between meters, feet, inches, and miles."""
        # Conversion rates to meters (base unit)
        to_m = {"m": 1.0, "ft": 0.3048, "in": 0.0254, "mi": 1609.344}

        # Check if units are supported
        if from_unit not in to_m or to_unit not in to_m:
            raise ValueError(
                f"Cannot convert from {from_unit} to {to_unit}. Supported units: m, ft, in, mi"
            )

        # Convert to meters then to target unit
        meters = value * to_m[from_unit]
        return meters / to_m[to_unit]


class MathConstants:
    """Tool for providing mathematical constants."""

    def get_constant(self, name: str) -> dict:
        """Get a mathematical constant and its description."""
        constants = {
            "pi": {
                "value": 3.14159265359,
                "description": "The ratio of a circle's circumference to its diameter",
            },
            "e": {
                "value": 2.71828182846,
                "description": "The base of the natural logarithm",
            },
            "golden_ratio": {
                "value": 1.61803398875,
                "description": "The ratio where the ratio of the sum of quantities to the larger quantity equals the ratio of the larger quantity to the smaller one",
            },
            "avogadro": {
                "value": 6.02214076e23,
                "description": "The number of constituent particles in one mole of a substance",
            },
            "plank": {
                "value": 6.62607015e-34,
                "description": "The fundamental quantum of action in quantum mechanics",
            },
        }

        name = name.lower()
        if name not in constants:
            raise ValueError(f"Unknown constant: {name}")

        return constants[name]


# Define tools for the Calculator Agent
@function_tool
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert temperature between units (c=Celsius, f=Fahrenheit, k=Kelvin).

    Args:
        value: The temperature value to convert
        from_unit: The unit to convert from (c, f, or k)
        to_unit: The unit to convert to (c, f, or k)

    Returns:
        The converted temperature as a formatted string
    """
    converter = UnitConverter()
    try:
        result = converter.convert_temperature(value, from_unit, to_unit)
        return f"{value}{from_unit} is {result}{to_unit}"
    except ValueError as e:
        return f"Error: {str(e)}"


@function_tool
def convert_length(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert length between units (m=meters, ft=feet, in=inches, mi=miles).

    Args:
        value: The length value to convert
        from_unit: The unit to convert from (m, ft, in, mi)
        to_unit: The unit to convert to (m, ft, in, mi)

    Returns:
        The converted length as a formatted string
    """
    converter = UnitConverter()
    try:
        result = converter.convert_length(value, from_unit, to_unit)
        return f"{value}{from_unit} is {result}{to_unit}"
    except ValueError as e:
        return f"Error: {str(e)}"


@function_tool
def get_math_constant(name: str) -> str:
    """
    Get the value and description of a mathematical constant.

    Args:
        name: The name of the constant (pi, e, golden_ratio, avogadro, plank)

    Returns:
        The constant value and its description
    """
    constants = MathConstants()
    try:
        result = constants.get_constant(name)
        return f"{name}: {result['value']}\n    {result['description']}"
    except ValueError as e:
        return f"Error: {str(e)}"


# Create Calculator Agent
calculator_agent = Agent(
    name="Calculator Assistant",
    instructions="""
    You are a helpful Calculator Assistant that can:
    1. Convert between different units (temperature, length)
    2. Provide mathematical constants and their explanations
    
    Respond to user queries about calculations, conversions, and mathematical constants.
    Always show your work for calculations and explain how you arrived at the answer.
    """,
    tools=[convert_temperature, convert_length, get_math_constant],
    model="gpt-3.5-turbo",
)

# ----- CUSTOMER SERVICE AGENT ----- #


@dataclass
class CustomerContext:
    """Context for the customer service agent."""

    customer_id: str
    is_premium: bool
    history: List[str] = Field(default_factory=list)


class CustomerResponse(BaseModel):
    """Structured output for customer service responses."""

    message: str
    requires_manager_review: bool = False
    follow_up_needed: bool = False
    additional_information: Optional[str] = None


# Define the Customer Service tools
@function_tool
def use_calculator(query: str) -> str:
    """
    Use the Calculator Assistant to answer math-related questions.

    Args:
        query: Math-related question to ask the Calculator Assistant

    Returns:
        The answer from the Calculator Assistant
    """
    client = OpenAI()
    messages = [
        {"role": "system", "content": calculator_agent.instructions},
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, max_tokens=300
    )

    return response.choices[0].message.content


@function_tool
def get_account_status(customer_context: CustomerContext) -> str:
    """
    Check the customer's account status.

    Args:
        customer_context: The customer's context information

    Returns:
        A description of the customer's account status
    """
    if customer_context.is_premium:
        return f"Customer {customer_context.customer_id} has a premium account with priority support."
    else:
        return f"Customer {customer_context.customer_id} has a standard account."


# Create Customer Service Agent
customer_service_agent = Agent[CustomerContext](
    name="Customer Service Assistant",
    instructions="""
    You are a helpful Customer Service Assistant for a technology company.
    You can help customers with:
    1. Account status inquiries
    2. Mathematical calculations and unit conversions
    3. Information about mathematical constants
    
    For any math-related queries, use the calculator tool to provide accurate answers.
    Always be polite, professional, and helpful to customers.
    If you don't know the answer, say so honestly and offer alternative help.
    """,
    tools=[use_calculator, get_account_status],
    model="gpt-3.5-turbo",
    output_type=CustomerResponse,
)

# ----- DEMONSTRATION ----- #


async def run_demo():
    """Run demonstration of the combined agents."""
    logger.info("Starting Combined Agents Demo")

    # Create customer context
    context = CustomerContext(customer_id="CUST12345", is_premium=True, history=[])

    # Test scenarios
    test_queries = [
        "Can you tell me what temperature 32°F is in Celsius?",
        "How many meters are in 5 miles?",
        "What is the value of pi and why is it important?",
        "What's the status of my account?",
        "Can you convert 100 Celsius to Fahrenheit and tell me if I'm a premium customer?",
    ]

    for query in test_queries:
        logger.info(f"\n\nCustomer Query: {query}")
        context.history.append(query)

        try:
            response = await customer_service_agent.run(context=context, message=query)
            logger.info(f"Response: {response.message}")

            if response.additional_information:
                logger.info(
                    f"Additional Information: {response.additional_information}"
                )

            if response.requires_manager_review:
                logger.info("This response requires manager review")

            if response.follow_up_needed:
                logger.info("Follow-up is needed for this customer")

        except Exception as e:
            logger.error(f"Error processing query: {e}")


if __name__ == "__main__":
    asyncio.run(run_demo())
