"""
Combined Agents Demo - Calculator + Customer Service

This example demonstrates how to create a Customer Service Agent that uses
a Calculator Agent as a tool. The Calculator Agent provides unit conversions
and mathematical constants, which the Customer Service Agent can use to answer
customer queries.

Usage:
    python combined_agents_demo.py

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
            value_k = (value - 32) * 5/9 + 273.15
        elif from_unit == "k":
            value_k = value
        else:
            raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")
            
        # Convert from Kelvin to target unit
        if to_unit == "c":
            return value_k - 273.15
        elif to_unit == "f":
            return (value_k - 273.15) * 9/5 + 32
        elif to_unit == "k":
            return value_k
        else:
            raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")
    
    def convert_length(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert length between meters, feet, inches, and miles."""
        # Conversion rates to meters (base unit)
        to_m = {
            "m": 1.0,
            "ft": 0.3048,
            "in": 0.0254,
            "mi": 1609.344
        }
        
        # Check if units are supported
        if from_unit not in to_m or to_unit not in to_m:
            raise ValueError(f"Cannot convert from {from_unit} to {to_unit}. Supported units: m, ft, in, mi")
            
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
                "description": "The ratio of a circle's circumference to its diameter"
            },
            "e": {
                "value": 2.71828182846,
                "description": "The base of the natural logarithm"
            },
            "golden_ratio": {
                "value": 1.61803398875,
                "description": "The ratio where the ratio of the sum of quantities to the larger quantity equals the ratio of the larger quantity to the smaller one"
            },
            "avogadro": {
                "value": 6.02214076e23,
                "description": "The number of constituent particles in one mole of a substance"
            },
            "plank": {
                "value": 6.62607015e-34,
                "description": "The fundamental quantum of action in quantum mechanics"
            }
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
    model="gpt-3.5-turbo"
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
        {"role": "user", "content": query}
    ]
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=300
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
    output_type=CustomerResponse
)

# ----- DEMONSTRATION ----- #

async def run_demo():
    """Run demonstration of the combined agents."""
    logger.info("Starting Combined Agents Demo")
    
    # Create customer context
    context = CustomerContext(
        customer_id="CUST12345",
        is_premium=True,
        history=[]
    )
    
    # Test scenarios
    test_queries = [
        "Can you tell me what temperature 32°F is in Celsius?",
        "How many meters are in 5 miles?",
        "What is the value of pi and why is it important?",
        "What's the status of my account?",
        "Can you convert 100 Celsius to Fahrenheit and tell me if I'm a premium customer?"
    ]
    
    for query in test_queries:
        logger.info(f"\n\nCustomer Query: {query}")
        context.history.append(query)
        
        try:
            response = await customer_service_agent.run(context=context, message=query)
            logger.info(f"Response: {response.message}")
            
            if response.additional_information:
                logger.info(f"Additional Information: {response.additional_information}")
                
            if response.requires_manager_review:
                logger.info("This response requires manager review")
                
            if response.follow_up_needed:
                logger.info("Follow-up is needed for this customer")
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")


if __name__ == "__main__":
    asyncio.run(run_demo())

#!/usr/bin/env python3
"""
Combined Agents Demo

This script demonstrates how to combine a calculator agent with a customer service agent,
allowing the customer service agent to use the calculator as a tool for handling
numerical inquiries alongside standard customer support functions.

Features:
1. Calculator agent with unit conversion and mathematical constants
2. Customer service agent with support for billing and technical questions
3. Integration between agents using tool interfaces
4. Test scenarios demonstrating the combined functionality

Usage:
    python combined_agents_demo.py

Requirements:
    - OpenAI API key set as environment variable OPENAI_API_KEY
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

# Import from the Agents SDK
from agents import (
    Agent,
    AgentResponse,
    Guardrail,
    GuardrailResponse,
    ModelSettings,
    RunContext,
    Runner,
    function_tool,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ========== CALCULATOR AGENT ==========

class UnitType(str, Enum):
    """Enum for different types of units that can be converted."""
    TEMPERATURE = "temperature"
    LENGTH = "length"
    WEIGHT = "weight"

@dataclass
class CalculatorContext:
    """Context for the Calculator Agent to maintain state and history."""
    calculation_history: List[str] = field(default_factory=list)
    current_conversion_type: Optional[UnitType] = None
    error_count: int = 0

class CalculatorOutput(BaseModel):
    """Structured output for calculator responses."""
    result: str = Field(description="The calculated result or converted value")
    explanation: Optional[str] = Field(None, description="Explanation of the calculation or conversion")
    error: Optional[str] = Field(None, description="Error message if calculation failed")

class CalculatorTools:
    """Tools for the Calculator Agent."""
    
    @function_tool
    def add(a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b
    
    @function_tool
    def subtract(a: float, b: float) -> float:
        """Subtract b from a."""
        return a - b
    
    @function_tool
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers together."""
        return a * b
    
    @function_tool
    def divide(a: float, b: float) -> float:
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    @function_tool
    def convert_temperature(from_value: float, from_unit: str, to_unit: str) -> str:
        """
        Convert temperature between different units (Celsius, Fahrenheit, Kelvin).
        
        Args:
            from_value: The temperature value to convert
            from_unit: The unit to convert from (c, f, k)
            to_unit: The unit to convert to (c, f, k)
            
        Returns:
            The converted temperature value with unit
        """
        # Normalize units to lowercase
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        # Valid units
        valid_units = {"c", "f", "k", "celsius", "fahrenheit", "kelvin"}
        
        # Check if units are valid
        if from_unit not in valid_units or to_unit not in valid_units:
            raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")
        
        # Normalize units to single character
        if from_unit in {"celsius", "c"}:
            from_unit = "c"
        elif from_unit in {"fahrenheit", "f"}:
            from_unit = "f"
        elif from_unit in {"kelvin", "k"}:
            from_unit = "k"
            
        if to_unit in {"celsius", "c"}:
            to_unit = "c"
        elif to_unit in {"fahrenheit", "f"}:
            to_unit = "f"
        elif to_unit in {"kelvin", "k"}:
            to_unit = "k"
        
        # Convert to Celsius first
        if from_unit == "c":
            celsius = from_value
        elif from_unit == "f":
            celsius = (from_value - 32) * 5/9
        elif from_unit == "k":
            celsius = from_value - 273.15
        
        # Convert from Celsius to target unit
        if to_unit == "c":
            result = celsius
            unit_symbol = "°C"
        elif to_unit == "f":
            result = celsius * 9/5 + 32
            unit_symbol = "°F"
        elif to_unit == "k":
            result = celsius + 273.15
            unit_symbol = "K"
        
        return f"{from_value}°{from_unit.upper()} is {result}{unit_symbol}"
    
    @function_tool
    def convert_length(from_value: float, from_unit: str, to_unit: str) -> str:
        """
        Convert length between different units (meters, feet, inches, miles).
        
        Args:
            from_value: The length value to convert
            from_unit: The unit to convert from (m, ft, in, mi)
            to_unit: The unit to convert to (m, ft, in, mi)
            
        Returns:
            The converted length value with unit
        """
        # Conversion factors to meters
        to_meters = {
            "m": 1,
            "meter": 1,
            "meters": 1,
            "ft": 0.3048,
            "foot": 0.3048,
            "feet": 0.3048,
            "in": 0.0254,
            "inch": 0.0254,
            "inches": 0.0254,
            "mi": 1609.34,
            "mile": 1609.34,
            "miles": 1609.34
        }
        
        # Unit symbols for output
        unit_symbols = {
            "m": "m",
            "meter": "m",
            "meters": "m",
            "ft": "ft",
            "foot": "ft",
            "feet": "ft",
            "in": "in",
            "inch": "in",
            "inches": "in",
            "mi": "mi",
            "mile": "mi",
            "miles": "mi"
        }
        
        # Check if units are valid
        if from_unit not in to_meters or to_unit not in to_meters:
            raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")
        
        # Convert to meters first
        meters = from_value * to_meters[from_unit]
        
        # Convert from meters to target unit
        result = meters / to_meters[to_unit]
        
        return f"{from_value} {unit_symbols[from_unit]} is {result} {unit_symbols[to_unit]}"
    
    @function_tool
    def convert_weight(from_value: float, from_unit: str, to_unit: str) -> str:
        """
        Convert weight between different units (kg, lb, oz).
        
        Args:
            from_value: The weight value to convert
            from_unit: The unit to convert from (kg, lb, oz)
            to_unit: The unit to convert to (kg, lb, oz)
            
        Returns:
            The converted weight value with unit
        """
        # Conversion factors to kilograms
        to_kg = {
            "kg": 1,
            "kilogram": 1,
            "kilograms": 1,
            "lb": 0.453592,
            "pound": 0.453592,
            "pounds": 0.453592,
            "oz": 0.0283495,
            "ounce": 0.0283495,
            "ounces": 0.0283495
        }
        
        # Unit symbols for output
        unit_symbols = {
            "kg": "kg",
            "kilogram": "kg",
            "kilograms": "kg",
            "lb": "lb",
            "pound": "lb",
            "pounds": "lb",
            "oz": "oz",
            "ounce": "oz",
            "ounces": "oz"
        }
        
        # Check if units are valid
        if from_unit not in to_kg or to_unit not in to_kg:
            raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")
        
        # Convert to kilograms first
        kg = from_value * to_kg[from_unit]
        
        # Convert from kilograms to target unit
        result = kg / to_kg[to_unit]
        
        return f"{from_value} {unit_symbols[from_unit]} is {result} {unit_symbols[to_unit]}"
    
    @function_tool
    def get_math_constant(constant_name: str) -> str:
        """
        Get the value and description of a mathematical constant.
        
        Args:
            constant_name: Name of the constant (pi, e, golden_ratio, avogadro, plank)
            
        Returns:
            The value and description of the constant
        """
        constants = {
            "pi": {
                "value": 3.14159265359,
                "description": "The ratio of a circle's circumference to its diameter"
            },
            "e": {
                "value": 2.71828182846,
                "description": "The base of the natural logarithm"
            },
            "golden_ratio": {
                "value": 1.61803398875,
                "description": "The ratio where the ratio of the sum of quantities to the larger quantity is equal to the ratio of the larger quantity to the smaller one"
            },
            "avogadro": {
                "value": 6.02214076e23,
                "description": "The number of constituent particles in one mole of a substance"
            },
            "plank": {
                "value": 6.62607015e-34,
                "description": "The fundamental quantum of action in quantum mechanics"
            }
        }
        
        constant_name = constant_name.lower()
        if constant_name not in constants:
            raise ValueError(f"Unknown constant: {constant_name}")
        
        constant = constants[constant_name]
        return f"{constant_name}: {constant['value']}\n    {constant['description']}"

# Create the Calculator Agent
calculator_agent = Agent[CalculatorContext](
    name="Calculator Assistant",
    instructions="""
    You are a helpful calculator assistant that can perform mathematical operations,
    unit conversions, and provide mathematical constants.
    
    For calculations, show your work and explain the steps.
    For unit conversions, provide the result with appropriate units.
    For mathematical constants, provide the value and a brief explanation.
    
    Be precise with numerical values and always check your work.
    """,
    model="gpt-4-turbo",
    model_settings=ModelSettings(
        temperature=0.1,  # Low temperature for deterministic, precise responses
    ),
    tools=[
        CalculatorTools.add,
        CalculatorTools.subtract,
        CalculatorTools.multiply,
        CalculatorTools.divide,
        CalculatorTools.convert_temperature,
        CalculatorTools.convert_length,
        CalculatorTools.convert_weight,
        CalculatorTools.get_math_constant,
    ],
    output_type=CalculatorOutput,
)

# ========== CUSTOMER SERVICE AGENT ==========

class CustomerQuery(Enum):
    """Types of customer queries."""
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"
    MATHEMATICAL = "mathematical"

@dataclass
class CustomerContext:
    """Context for the Customer Service Agent."""
    customer_id: str = "anonymous"
    premium_tier: bool = False
    query_history: List[str] = field(default_factory=list)
    query_type: Optional[CustomerQuery] = None

class CustomerServiceOutput(BaseModel):
    """Structured output for customer service responses."""
    response: str = Field(description="The response to the customer's query")
    requires_followup: bool = Field(False, description="Whether a follow-up is needed")
    followup_question: Optional[str] = Field(None, description="Follow-up question if needed")

class InputGuardrail(Guardrail):
    """Guardrail for checking input appropriateness."""
    
    async def check(self, input_text: str, context: RunContext) -> GuardrailResponse:
        """Check if the input is appropriate."""
        # List of forbidden terms
        forbidden_terms = ["profanity", "offensive", "inappropriate"]
        
        # Check if any forbidden terms are in the input
        for term in forbidden_terms:
            if term in input_text.lower():
                return GuardrailResponse(
                    compliant=False,
                    message=f"Input contains inappropriate content: {term}",
                )
                
        return GuardrailResponse(compliant=True)

# Create tools for the Customer Service Agent
class CustomerServiceTools:
    """Tools for the Customer Service Agent."""
    
    @function_tool
    def get_billing_info(context: RunContext[CustomerContext], account_id: Optional[str] = None) -> str:
        """Get billing information for the customer."""
        customer_id = account_id or context.context.customer_id
        premium = context.context.premium_tier
        
        if premium:
            return f"Customer {customer_id} is on the Premium plan ($99/month), last payment received on 2023-03-15, next payment due on 2023-04-15."
        else:
            return f"Customer {customer_id} is on the Basic plan ($9/month), last payment received on 2023-03-10, next payment due on 2023-04-10."
    
    @function_tool
    def check_service_status() -> str:
        """Check the current service status."""
        return "All services are operational. No outages reported in the last 24 hours."
    
    @function_tool
    async def perform_calculation(context: RunContext[CustomerContext], query: str) -> str:
        """
        Use the calculator agent to perform calculations or conversions.
        
        Args:
            query: The mathematical query to process
            
        Returns:
            The result from the calculator agent
        """
        # Create a new context for the calculator agent
        calc_context = CalculatorContext()
        
        # Run the calculator agent with the query
        runner = Runner(calculator_agent)
        calc_response = await runner.run(calc_context, query)
        
        # Save the query to history
        context.context.query_history.append(query)
        context.context.query_type = CustomerQuery.MATHEMATICAL
        
        # Return the calculation result
        if calc_response.output.error:
            return f"Calculation error: {calc_response.output.error}"
        else:
            return f"{calc_response.output.result}" + (f"\n{calc_response.output.explanation}" if calc_response.output.explanation else "")

# Create the Customer Service Agent
customer_service_agent = Agent[CustomerContext](
    name="Customer Support Assistant",
    instructions="""
    You are a helpful customer support assistant.
    
    Help customers with:
    1. Billing inquiries - use the get_billing_info tool

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
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

# Import the Agents SDK components
from agents import (
    Agent,
    AgentHooks,
    AgentResponse,
    ModelSettings,
    RunContext,
    Runner,
    function_tool,
    handoff,
)
from agents.guardrails import (
    InputGuardrail,
    OutputGuardrail,
    ValidationResult,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("combined_agents")

# -----------------------------------------------------------------------------
# SECTION 1: Context and Output Types
# -----------------------------------------------------------------------------

@dataclass
class AgentContext:
    """Custom context with user information and history."""
    user_id: str
    username: str
    is_premium: bool
    conversation_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
    
    def log_interaction(self, tool_name: str, input_data: Any, output_data: Any) -> None:
        """Log an interaction with a tool to the conversation history."""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "input": input_data,
            "output": output_data
        })


class SupportCategory(str, Enum):
    """Categories for support tickets."""
    CALCULATION = "calculation"
    UNIT_CONVERSION = "unit_conversion"
    CONSTANT_INFO = "constant_info"
    GENERAL = "general"


class SupportQuery(BaseModel):
    """Structured output for the support triage agent."""
    category: SupportCategory = Field(description="The type of support request")
    query_details: Dict[str, Any] = Field(description="Details of the query")
    requires_calculation: bool = Field(description="Whether this requires calculation")
    priority: str = Field(description="Priority level (low, medium, high)")


# -----------------------------------------------------------------------------
# SECTION 2: Calculator Tools
# -----------------------------------------------------------------------------

class UnitConverter:
    """Tool for converting between different units of measurement."""
    
    @staticmethod
    def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
        """Convert temperature between Celsius, Fahrenheit and Kelvin."""
        conversions = {
            ('c', 'f'): lambda x: x * 9/5 + 32,
            ('f', 'c'): lambda x: (x - 32) * 5/9,
            ('c', 'k'): lambda x: x + 273.15,
            ('k', 'c'): lambda x: x - 273.15,
            ('f', 'k'): lambda x: (x - 32) * 5/9 + 273.15,
            ('k', 'f'): lambda x: (x - 273.15) * 9/5 + 32
        }
        key = (from_unit.lower()[0], to_unit.lower()[0])
        if key not in conversions:
            raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")
        return conversions[key](value)

    @staticmethod
    def convert_length(value: float, from_unit: str, to_unit: str) -> float:
        """Convert length between meters, feet, inches, and miles."""
        units = {
            'm': 1.0,
            'ft': 0.3048,
            'in': 0.0254,
            'mi': 1609.34
        }
        try:
            return value * units[from_unit] / units[to_unit]
        except KeyError:
            raise ValueError(f"Invalid units: {from_unit} to {to_unit}")

    @staticmethod
    def convert_weight(value: float, from_unit: str, to_unit: str) -> float:
        """Convert weight between kilograms, pounds, and ounces."""
        units = {
            'kg': 1.0,
            'lb': 0.453592,
            'oz': 0.0283495
        }
        if from_unit not in units or to_unit not in units:
            raise ValueError(f"Invalid weight units: {from_unit} to {to_unit}")
        return value * units[from_unit] / units[to_unit]


class MathConstant:
    """Tool providing mathematical constants and their explanations."""
    
    CONSTANTS: Dict[str, Tuple[float, str]] = {
        'pi': (3.14159265359, "The ratio of a circle's circumference to its diameter"),
        'e': (2.71828182846, "The base of the natural logarithm"),
        'golden_ratio': (1.61803398875, 
                        "The ratio where the ratio of the sum of quantities to the larger quantity "
                        "is equal to the ratio of the larger quantity to the smaller one"),
        'avogadro': (6.02214076e23, 
                    "The number of constituent particles in one mole of a substance"),
        'plank': (6.62607015e-34, 
                 "The fundamental quantum of action in quantum mechanics")
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
def convert_temperature(value: float, from_unit: str, to_unit: str, 
                        context: RunContext[AgentContext]) -> Dict[str, Any]:
    """Convert temperature between Celsius, Fahrenheit and Kelvin."""
    try:
        converter = UnitConverter()
        result = converter.convert_temperature(value, from_unit, to_unit)
        
        # Log the interaction
        context.context.log_interaction(
            "convert_temperature", 
            {"value": value, "from_unit": from_unit, "to_unit": to_unit}, 
            result
        )
        
        return {
            "original_value": value,
            "original_unit": from_unit,
            "converted_value": result,
            "converted_unit": to_unit,
            "formula_used": f"Conversion formula from {from_unit} to {to_unit}",
            "success": True
        }
    except ValueError as e:
        return {
            "success": False,
            "error": str(e)
        }


@function_tool
def convert_length(value: float, from_unit: str, to_unit: str, 
                   context: RunContext[AgentContext]) -> Dict[str, Any]:
    """Convert length between meters, feet, inches, and miles."""
    try:
        converter = UnitConverter()
        result = converter.convert_length(value, from_unit, to_unit)
        
        # Log the interaction
        context.context.log_interaction(
            "convert_length", 
            {"value": value, "from_unit": from_unit, "to_unit": to_unit}, 
            result
        )
        
        return {
            "original_value": value,
            "original_unit": from_unit,
            "converted_value": result,
            "converted_unit": to_unit,
            "success": True
        }
    except ValueError as e:
        return {
            "success": False,
            "error": str(e)
        }


@function_tool
def convert_weight(value: float, from_unit: str, to_unit: str, 
                   context: RunContext[AgentContext]) -> Dict[str, Any]:
    """Convert weight between kilograms, pounds, and ounces."""
    try:
        converter = UnitConverter()
        result = converter.convert_weight(value, from_unit, to_unit)
        
        # Log the interaction
        context.context.log_interaction(
            "convert_weight", 
            {"value": value, "from_unit": from_unit, "to_unit": to_unit}, 
            result
        )
        
        return {
            "original_value": value,
            "original_unit": from_unit,
            "converted_value": result,
            "converted_unit": to_unit,
            "success": True
        }
    except ValueError as e:
        return {
            "success": False,
            "error": str(e)
        }


@function_tool
def get_math_constant(name: str, context: RunContext[AgentContext]) -> Dict[str, Any]:
    """Get information about a mathematical constant."""
    try:
        math_constants = MathConstant()
        value, description = math_constants.get_constant(name)
        
        # Log the interaction
        context.context.log_interaction(
            "get_math_constant", 
            {"name": name}, 
            {"value": value, "description": description}
        )
        
        return {
            "constant_name": name,
            "value": value,
            "description": description,
            "success": True
        }
    except ValueError as e:
        return {
            "success": False,
            "error": str(e)
        }


@function_tool
def get_user_details(context: RunContext[AgentContext]) -> Dict[str, Any]:
    """Get information about the current user."""
    user = context.context
    
    return {
        "user_id": user.user_id,
        "username": user.username,
        "is_premium": user.is_premium,
        "conversation_count": len(user.conversation_history)
    }


# -----------------------------------------------------------------------------
# SECTION 4: Lifecycle Hooks
# -----------------------------------------------------------------------------

class AgentLifecycleHooks(AgentHooks):
    """Custom hooks for monitoring agent lifecycle."""
    
    def __init__(self):
        self.start_time = None
        self.tools_called = 0
        
    async def on_start(self, context: RunContext[AgentContext], agent: Agent[AgentContext]) -> None:
        """Called when an agent run starts."""
        self.start_time = datetime.now()
        logger.info(f"Agent run started: {agent.name}")
        logger.info(f"User: {context.context.username} (Premium: {context.context.is_premium})")
        
    async def on_end(self, context: RunContext[AgentContext], agent: Agent[AgentContext], output: Any) -> None:
        """Called when an agent run completes."""
        duration = datetime.now() - self.start_time if self.start_time else None
        logger.info(f"Agent run completed: {agent.name}")
        logger.info(f"Duration: {duration}")
        logger.info(f"Tools called: {self.tools_called}")
    
    async def on_tool_start(self, context: RunContext[AgentContext], agent: Agent[AgentContext], tool: Any) -> None:
        """Called when a tool is invoked."""
        self.tools_called += 1
        logger.info(f"Tool called: {tool.name}")


# -----------------------------------------------------------------------------
# SECTION 5: Guardrails
# -----------------------------------------------------------------------------

class CalculationGuardrail(InputGuardrail):
    """Input guardrail to validate calculation-related queries."""
    
    async def validate(self, input_text: str, context: RunContext[AgentContext]) -> ValidationResult:
        # Define keywords for valid calculation queries
        calculation_keywords = [
            "convert", "calculate", "math", "formula", "equation", "value", "constant",
            "celsius", "fahrenheit", "kelvin", "meter", "foot", "feet", "inch", "mile",
            "kilogram", "pound", "ounce", "pi", "e", "golden ratio", "temperature", "length", "weight"
        ]
        
        # Check if any calculation keyword is in the input
        if any(keyword in input_text.lower() for keyword in calculation_keywords):
            return ValidationResult(valid=True)
        
        # Minimal query length check
        if len(input_text.split()) < 3:
            return ValidationResult(
                valid=False,
                message="Please provide a more detailed query for calculation or conversion."
            )
        
        # Default to valid if we're not sure
        return ValidationResult(valid=True)


class ResponseFormatGuardrail(OutputGuardrail):
    """Output guardrail to ensure responses include proper unit information."""
    
    async def validate(self, output_text: str, context: RunContext[AgentContext]) -> ValidationResult:
        # For conversion results, ensure they include unit information
        conversion_keywords = ["convert", "conversion", "calculated", "result", "value"]
        
        if any(keyword in output_text.lower() for keyword in conversion_keywords):
            unit_keywords = ["celsius", "fahrenheit", "kelvin", "meter", "foot", "feet", 
                            "inch", "mile", "kilogram", "pound", "ounce"]
            
            if not any(unit in output_text.lower() for unit in unit_keywords):
                return ValidationResult(
                    valid=False,
                    message="Please include the units in your response for clarity."
                )
        
        return ValidationResult(valid=True)


# -----------------------------------------------------------------------------
# SECTION 6: Agent Implementations
# -----------------------------------------------------------------------------

# Calculator Agent for handling calculations and conversions
calculator_agent = Agent[AgentContext](
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
    model="o3-mini",
    tools=[
        convert_temperature,
        convert_length,
        convert_weight,
        get_math_constant,
        get_user_details
    ],
    hooks=AgentLifecycleHooks(),
    guardrails=[
        CalculationGuardrail(),
        ResponseFormatGuardrail()
    ]
)

# Main Customer Service Agent that uses the calculator as a tool
assistant_agent = Agent[AgentContext](
    name="Customer Service Assistant",
    instructions="""You are a helpful customer service assistant who can help with various queries including calculations and conversions.

For calculation and conversion requests, use your calculator tools to provide accurate results.
Always be polite, clear,

#!/usr/bin/env python3
"""
Combined Agents Demo

This script demonstrates how to use one agent as a tool for another agent.
It combines the calculator capabilities from core_concepts_demo.py
with the customer service agent from agent_features_demo.py.

Usage:
    python combined_agents_demo.py

Requirements:
    - OpenAI API key set as OPENAI_API_KEY environment variable
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime

from pydantic import BaseModel

# Import from the OpenAI Agents SDK
from agents import Agent, ModelSettings, function_tool, AgentHooks, AgentOutput

# Import components from the calculator demo
from my_examples.demo_agents.core_concepts_demo import (
    UnitConverter,
    MathConstant,
    error_handler
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- CONTEXT DEFINITION ---
@dataclass
class CustomerContext:
    """Context for tracking customer interactions."""
    customer_id: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    session_start_time: datetime = field(default_factory=datetime.now)
    premium_user: bool = False

    def add_to_history(self, role: str, content: str) -> None:
        """Add an interaction to the conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

# --- OUTPUT DEFINITIONS ---
class CustomerResponse(BaseModel):
    """Structured output for customer service responses."""
    answer: str
    sentiment: str
    requires_escalation: bool = False
    reference_materials: Optional[List[str]] = None
    calculation_results: Optional[Dict[str, Any]] = None

# --- CALCULATOR TOOL IMPLEMENTATION ---
class CalculatorTool:
    """Tool that provides calculator functionality to the agent."""
    
    def __init__(self):
        self.unit_converter = UnitConverter()
        self.math_constant = MathConstant()
    
    @function_tool
    def convert_temperature(self, value: float, from_unit: str, to_unit: str) -> str:
        """
        Convert temperature between Celsius (c), Fahrenheit (f), and Kelvin (k).
        
        Args:
            value: The temperature value to convert
            from_unit: The unit to convert from (c, f, k)
            to_unit: The unit to convert to (c, f, k)
            
        Returns:
            The converted temperature as a string with units
        """
        try:
            result = self.unit_converter.convert_temperature(value, from_unit, to_unit)
            return f"{value}°{from_unit.upper()} is {result}°{to_unit.upper()}"
        except ValueError as e:
            return f"Error: {str(e)}"
    
    @function_tool
    def convert_length(self, value: float, from_unit: str, to_unit: str) -> str:
        """
        Convert length between meters (m), feet (ft), inches (in), and miles (mi).
        
        Args:
            value: The length value to convert
            from_unit: The unit to convert from (m, ft, in, mi)
            to_unit: The unit to convert to (m, ft, in, mi)
            
        Returns:
            The converted length as a string with units
        """
        try:
            result = self.unit_converter.convert_length(value, from_unit, to_unit)
            return f"{value}{from_unit} is {result}{to_unit}"
        except ValueError as e:
            return f"Error: {str(e)}"
    
    @function_tool
    def convert_weight(self, value: float, from_unit: str, to_unit: str) -> str:
        """
        Convert weight between kilograms (kg), pounds (lb), and ounces (oz).
        
        Args:
            value: The weight value to convert
            from_unit: The unit to convert from (kg, lb, oz)
            to_unit: The unit to convert to (kg, lb, oz)
            
        Returns:
            The converted weight as a string with units
        """
        try:
            result = self.unit_converter.convert_weight(value, from_unit, to_unit)
            return f"{value}{from_unit} is {result}{to_unit}"
        except ValueError as e:
            return f"Error: {str(e)}"
    
    @function_tool
    def get_math_constant(self, constant_name: str) -> str:
        """
        Get the value and description of a mathematical constant.
        
        Args:
            constant_name: The name of the constant (pi, e, golden_ratio, avogadro, plank)
            
        Returns:
            The value and description of the constant
        """
        try:
            value, description = self.math_constant.get_constant(constant_name)
            return f"{constant_name.capitalize()}: {value}\n    {description}"
        except ValueError as e:
            return f"Error: {str(e)}"

# --- CUSTOMER SERVICE HOOKS ---
class CustomerServiceHooks(AgentHooks):
    """Hooks for tracking customer service agent lifecycle events."""
    
    def __init__(self):
        self.start_time = None
        
    async def on_agent_run_begin(self, agent: Agent, context: CustomerContext) -> None:
        """Called when the agent run begins."""
        self.start_time = datetime.now()
        logger.info(f"Starting agent run for customer {context.customer_id}")
        
    async def on_agent_run_end(self, agent: Agent, context: CustomerContext) -> None:
        """Called when the agent run ends."""
        duration = datetime.now() - self.start_time
        logger.info(f"Agent run completed in {duration.total_seconds():.2f} seconds")
        
    async def on_message_received(self, message: str, context: CustomerContext) -> None:
        """Called when a message is received from the user."""
        context.add_to_history("user", message)
        logger.info(f"Received message from customer {context.customer_id}")
        
    async def on_agent_response(self, response: Union[str, CustomerResponse], context: CustomerContext) -> None:
        """Called when the agent generates a response."""
        if isinstance(response, str):
            context.add_to_history("assistant", response)
        else:
            context.add_to_history("assistant", response.answer)
            
            if response.requires_escalation:
                logger.warning(f"Customer {context.customer_id} issue requires escalation")

# --- AGENT CONFIGURATION ---
def create_customer_service_agent() -> Agent[CustomerContext]:
    """
    Create a customer service agent with calculator tools.
    
    Returns:
        An agent configured for customer service with calculator capabilities
    """
    # Create calculator tool
    calculator_tool = CalculatorTool()
    
    # Create the agent with both basic tools and calculator tools
    agent = Agent[CustomerContext](
        name="Math-Enabled Customer Service Agent",
        instructions="""
        You are a friendly customer service agent for 'TechMath Solutions'.
        You help customers with their inquiries about our products and services,
        especially those related to mathematical calculations, unit conversions, and constants.
        
        When customers ask about unit conversions or mathematical constants, use your tools to help them.
        Always be polite, concise, and accurate in your responses.
        
        For unit conversions:
        - Temperature: Supports Celsius (c), Fahrenheit (f), and Kelvin (k)
        - Length: Supports meters (m), feet (ft), inches (in), and miles (mi)
        - Weight: Supports kilograms (kg), pounds (lb), and ounces (oz)
        
        For mathematical constants, you can provide:
        - Pi
        - E (Euler's number)
        - Golden ratio
        - Avogadro's number
        - Planck constant
        
        Be empathetic and helpful at all times.
        """,
        model="gpt-4-turbo",
        model_settings=ModelSettings(
            temperature=0.7,
            # Other model settings as needed
        ),
        tools=[
            calculator_tool.convert_temperature,
            calculator_tool.convert_length,
            calculator_tool.convert_weight,
            calculator_tool.get_math_constant,
        ],
        output_type=CustomerResponse,
        hooks=CustomerServiceHooks()
    )
    
    return agent

# --- DEMONSTRATION SCENARIOS ---
async def run_conversion_inquiry(agent: Agent[CustomerContext]) -> None:
    """Run a scenario where a customer asks about unit conversions."""
    context = CustomerContext(customer_id="cust_12345")
    
    # Temperature conversion inquiry
    user_message = "I need to convert 100 degrees Celsius to Fahrenheit for my recipe. Can you help me?"
    print(f"\n=== Temperature Conversion Inquiry ===\nCustomer: {user_message}")
    
    response = await agent.run(
        user_message,
        context=context
    )
    
    print(f"Agent: {response.answer}")
    if response.calculation_results:
        print(f"Calculation results: {response.calculation_results}")
    
    # Length conversion inquiry
    user_message = "How many feet are in 10 meters? I'm trying to figure out if my new couch will fit in my living room."
    print(f"\n=== Length Conversion Inquiry ===\nCustomer: {user_message}")
    
    response = await agent.run(
        user_message,
        context=context
    )
    
    print(f"Agent: {response.answer}")

async def run_constant_inquiry(agent: Agent[CustomerContext]) -> None:
    """Run a scenario where a customer asks about mathematical constants."""
    context = CustomerContext(customer_id="cust_67890")
    
    # Math constant inquiry
    user_message = "I'm working on a math project and need the value of Pi. Can you tell me what it is and why it's important?"
    print(f"\n=== Mathematical Constant Inquiry ===\nCustomer: {user_message}")
    
    response = await agent.run(
        user_message,
        context=context
    )
    
    print(f"Agent: {response.answer}")
    
    # Multiple constants inquiry
    user_message = "Can you tell me about the golden ratio and how it relates to the Fibonacci sequence? Also, what is Avogadro's number used for?"
    print(f"\n=== Multiple Constants Inquiry ===\nCustomer: {user_message}")
    
    response = await agent.run(
        user_message,
        context=context
    )
    
    print(f"Agent: {response.answer}")

async def run_mixed_inquiry(agent: Agent[CustomerContext]) -> None:
    """Run a scenario with mixed inquiries about conversions and constants."""
    context = CustomerContext(customer_id="cust_54321", premium_user=True)
    
    user_message = """I'm working on a physics project and need some help:
    1. What is Planck's constant and why is it significant?
    2. I need to convert 212°F to Kelvin for my experiment.
    3. Also, how many kilograms are in 150 pounds?
    Thanks for your help!"""
    
    print(f"\n=== Mixed Inquiry (Premium User) ===\nCustomer: {user_message}")
    
    response = await agent.run(
        user_message,
        context=context
    )
    
    print(f"Agent: {response.answer}")
    if response.reference_materials:
        print(f"Reference materials: {response.reference_materials}")

# --- MAIN EXECUTION ---
async def main() -> None:
    """Main execution function."""
    print("=== Math-Enabled Customer Service Agent Demo ===")
    
    # Create the customer service agent with calculator capabilities
    agent = create_customer_service_agent()
    
    # Run demonstration scenarios
    await run_conversion_inquiry(agent)
    await run_constant_inquiry(agent)
    await run_mixed_inquiry(agent)
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    asyncio.run(main())

