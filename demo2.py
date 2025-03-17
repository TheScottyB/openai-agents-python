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
