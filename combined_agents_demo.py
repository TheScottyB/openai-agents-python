#!/usr/bin/env python3

"""
Combined Agents Demo
===================

This example demonstrates a simple agent system with calculator functionality and 
customer service capabilities using OpenAI Agents SDK best practices.

Features:
- Calculator Agent with unit conversion
- Customer Service Agent that uses Calculator as a tool
- Proper handoffs between agents
- Input and output guardrails
- Structured outputs with Pydantic models
- Lifecycle hooks for monitoring
- Tracing for debugging

To run:
    python combined_agents_demo.py

Requirements:
    - OpenAI API key set as OPENAI_API_KEY environment variable
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

from pydantic import BaseModel, Field

from agents import (
    Agent,
    AgentHooks,
    InputGuardrail,
    ModelSettings,
    OutputGuardrail,
    Runner,
    RunContextWrapper,
    GuardrailFunctionOutput,
    function_tool,
    input_guardrail,
    output_guardrail,
    set_tracing_disabled,
)
from agents.exceptions import InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

# -----------------------------------------------------------------------------
# SECTION 1: Context and Output Types
# -----------------------------------------------------------------------------

@dataclass
class CustomerContext:
    """Context for tracking customer information and interactions."""
    customer_id: str
    username: str
    premium_user: bool = False
    session_start_time: datetime = field(default_factory=datetime.now)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)

    def log_interaction(self, tool_name: str, input_data: Any, output_data: Any) -> None:
        """Log a tool interaction to the conversation history."""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "input": input_data,
            "output": output_data,
        })


class CalculationResult(BaseModel):
    """Structured output for calculation results."""
    result: str = Field(description="The calculation result")
    units: Optional[str] = Field(None, description="The units of the result, if applicable")
    explanation: str = Field(description="Explanation of the calculation")


class CustomerResponse(BaseModel):
    """Structured output for customer service responses."""
    answer: str = Field(description="The response to the customer's query")
    sentiment: str = Field(description="The sentiment of the response (positive, neutral, negative)")
    requires_escalation: bool = Field(description="Whether this query requires escalation to a human agent")
    calculation_result: Optional[CalculationResult] = Field(
        None, description="Results of any calculations performed"
    )


class MathScreenerOutput(BaseModel):
    """Output format for math query screening."""
    is_math_query: bool = Field(description="Whether the query is math-related")
    reasoning: str = Field(description="Reasoning for the determination")


class ResponseQualityOutput(BaseModel):
    """Output format for response quality checking."""
    is_appropriate: bool = Field(description="Whether the response is appropriate")
    reasoning: str = Field(description="Reasoning for the determination")

# -----------------------------------------------------------------------------
# SECTION 2: Unit Conversions
# -----------------------------------------------------------------------------

@function_tool(use_docstring_info=True)
def convert_temperature(
    ctx: RunContextWrapper[CustomerContext], 
    value: float, 
    from_unit: str, 
    to_unit: str
) -> Dict[str, Any]:
    """Convert temperature between Celsius, Fahrenheit and Kelvin.
    
    Args:
        value: The temperature value to convert
        from_unit: The source unit (c/celsius, f/fahrenheit, k/kelvin)
        to_unit: The target unit (c/celsius, f/fahrenheit, k/kelvin)
        
    Returns:
        A dictionary containing the conversion result
    """
    try:
        # Define conversion functions
        conversions = {
            ("c", "f"): lambda x: x * 9/5 + 32,
            ("f", "c"): lambda x: (x - 32) * 5/9,
            ("c", "k"): lambda x: x + 273.15,
            ("k", "c"): lambda x: x - 273.15,
            ("f", "k"): lambda x: (x - 32) * 5/9 + 273.15,
            ("k", "f"): lambda x: (x - 273.15) * 9/5 + 32,
        }
        
        # Convert units to single character
        from_char = from_unit.lower()[0]
        to_char = to_unit.lower()[0]
        key = (from_char, to_char)
        
        if key not in conversions:
            return {
                "success": False,
                "error": f"Cannot convert from {from_unit} to {to_unit}"
            }
            
        result = conversions[key](value)
        
        # Log the interaction
        ctx.context.log_interaction(
            "convert_temperature",
            {"value": value, "from_unit": from_unit, "to_unit": to_unit},
            result
        )
        
        return {
            "success": True,
            "original_value": value,
            "original_unit": from_unit,
            "converted_value": result,
            "converted_unit": to_unit,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@function_tool(use_docstring_info=True)
def convert_length(
    ctx: RunContextWrapper[CustomerContext], 
    value: float, 
    from_unit: str, 
    to_unit: str
) -> Dict[str, Any]:
    """Convert length between meters, feet, inches, and miles.
    
    Args:
        value: The length value to convert
        from_unit: The source unit (m, ft, in, mi)
        to_unit: The target unit (m, ft, in, mi)
        
    Returns:
        A dictionary containing the conversion result
    """
    try:
        # Define unit conversions to meters
        units = {"m": 1.0, "ft": 0.3048, "in": 0.0254, "mi": 1609.34}
        
        if from_unit not in units or to_unit not in units:
            return {
                "success": False,
                "error": f"Invalid units: {from_unit} to {to_unit}. Supported units: m, ft, in, mi"
            }
            
        # Convert to target unit
        result = value * units[from_unit] / units[to_unit]
        
        # Log the interaction
        ctx.context.log_interaction(
            "convert_length",
            {"value": value, "from_unit": from_unit, "to_unit": to_unit},
            result
        )
        
        return {
            "success": True,
            "original_value": value,
            "original_unit": from_unit,
            "converted_value": result,
            "converted_unit": to_unit,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@function_tool(use_docstring_info=True)
def get_math_constant(
    ctx: RunContextWrapper[CustomerContext], 
    name: str
) -> Dict[str, Any]:
    """Get information about a mathematical constant.
    
    Args:
        name: The name of the constant (pi, e, golden_ratio, avogadro, planck)
        
    Returns:
        A dictionary containing the constant's value and description
    """
    try:
        # Define constants with their values and descriptions
        constants = {
            "pi": (3.14159265359, "The ratio of a circle's circumference to its diameter"),
            "e": (2.71828182846, "The base of the natural logarithm"),
            "golden_ratio": (
                1.61803398875,
                "The ratio where the ratio of the sum of quantities to the larger quantity "
                "is equal to the ratio of the larger quantity to the smaller one"
            ),
            "avogadro": (
                6.02214076e23,
                "The number of constituent particles in one mole of a substance"
            ),
            "planck": (
                6.62607015e-34,
                "The fundamental quantum of action in quantum mechanics"
            ),
        }
        
        name_lower = name.lower()
        if name_lower not in constants:
            return {
                "success": False,
                "error": f"Unknown constant: {name}. Supported constants: pi, e, golden_ratio, avogadro, planck"
            }
            
        value, description = constants[name_lower]
        
        # Log the interaction
        ctx.context.log_interaction(
            "get_math_constant",
            {"name": name},
            {"value": value, "description": description}
        )
        
        return {
            "success": True,
            "constant_name": name,
            "value": value,
            "description": description,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@function_tool(use_docstring_info=True)
def get_user_details(
    ctx: RunContextWrapper[CustomerContext]
) -> Dict[str, Any]:
    """Get information about the current user.
    
    Returns:
        A dictionary containing user information
    """
    user = ctx.context
    
    return {
        "customer_id": user.customer_id,
        "username": user.username,
        "is_premium": user.premium_user,
        "session_started": user.session_start_time.isoformat(),
        "conversation_count": len(user.conversation_history),
    }

# -----------------------------------------------------------------------------
# SECTION 3: Lifecycle Hooks
# -----------------------------------------------------------------------------

class AgentLifecycleHooks(AgentHooks):
    """Custom hooks for monitoring agent lifecycle."""

    async def on_start(
        self, context: RunContextWrapper[CustomerContext], agent: Agent[CustomerContext]
    ) -> None:
        """Called when an agent run starts."""
        print(f"Agent '{agent.name}' started with {context.context.username}")

    async def on_end(
        self,
        context: RunContextWrapper[CustomerContext],
        agent: Agent[CustomerContext],
        output: Any,
    ) -> None:
        """Called when an agent run completes."""
        print(f"Agent '{agent.name}' completed")

    async def on_tool_start(
        self,
        context: RunContextWrapper[CustomerContext],
        agent: Agent[CustomerContext],
        tool: Any,
    ) -> None:
        """Called when a tool is invoked."""
        print(f"Tool '{tool.name}' called by agent '{agent.name}'")

# -----------------------------------------------------------------------------
# SECTION 4: Guardrails
# -----------------------------------------------------------------------------

@input_guardrail
async def math_query_guardrail(
    ctx: RunContextWrapper[CustomerContext],
    agent: Agent[CustomerContext],
    input_data: Union[str, List[Any]]
) -> GuardrailFunctionOutput:
    """Guardrail to verify math queries have enough information."""
    # Create a simple agent for screening math queries
    math_screener = Agent(
        name="Math Query Screener",
        instructions="Determine if the user query is related to math calculations or unit conversions, and if it contains sufficient information to provide a useful response.",
        model="gpt-3.5-turbo",
        model_settings=ModelSettings(temperature=0.1),
        output_type=MathScreenerOutput,
    )
    
    input_str = input_data if isinstance(input_data, str) else str(input_data)
    result = await Runner.run(math_screener, input_str, context=ctx.context)
    output = result.final_output_as(MathScreenerOutput)
    
    # If it's a math query, check for minimum required info
    if output.is_math_query and len(input_str.split()) < 4:
        return GuardrailFunctionOutput(
            output_info=output,
            tripwire_triggered=True,
        )
    
    return GuardrailFunctionOutput(
        output_info=output,
        tripwire_triggered=False,
    )


@output_guardrail
async def response_quality_guardrail(
    ctx: RunContextWrapper[CustomerContext],
    agent: Agent[CustomerContext],
    output: CustomerResponse
) -> GuardrailFunctionOutput:
    """Guardrail to verify response quality and appropriateness."""
    # Create a simple agent for checking response quality
    quality_checker = Agent(
        name="Response Quality Checker",
        instructions=(
            "Verify that the customer service response is appropriate, helpful, and accurate. "
            "Check that calculations are correct if present and that the response addresses "
            "the user's query completely."
        ),
        model="gpt-3.5-turbo",
        model_settings=ModelSettings(temperature=0.1),
        output_type=ResponseQualityOutput,
    )
    
    result = await Runner.run(quality_checker, output.answer, context=ctx.context)
    quality_output = result.final_output_as(ResponseQualityOutput)
    
    return GuardrailFunctionOutput(
        output_info=quality_output,
        tripwire_triggered=not quality_output.is_appropriate,
    )

# -----------------------------------------------------------------------------
# SECTION 5: Dynamic Instructions
# -----------------------------------------------------------------------------

def calculator_instructions(
    ctx: RunContextWrapper[CustomerContext], 
    agent: Agent[CustomerContext]
) -> str:
    """Dynamic instructions for the calculator agent."""
    premium_content = ""
    if ctx.context.premium_user:
        premium_content = " As this is a premium user, provide extra detail in your explanations."
    
    base_instructions = f"""You are a specialized calculator agent that performs conversions and provides mathematical constants.

Your capabilities include:
- Temperature conversions (Celsius, Fahrenheit, Kelvin)
- Length conversions (meters, feet, inches, miles)
- Math constants (Pi, e, Golden Ratio, etc.)

Always show your work and include appropriate units in your results.{premium_content}

Current user: {ctx.context.username}
"""
    return prompt_with_handoff_instructions(base_instructions)


def customer_service_instructions(
    ctx: RunContextWrapper[CustomerContext], 
    agent: Agent[CustomerContext]
) -> str:
    """Dynamic instructions for the customer service agent."""
    premium_content = ""
    if ctx.context.premium_user:
        premium_content = " As this is a premium user, provide priority service with more detailed explanations."
    
    base_instructions = f"""You are a helpful customer service assistant who can handle calculation requests.

For math-related questions, you can:
1. Use built-in conversion tools directly
2. Use the calculator agent for more complex requests

Always be professional, clear, and concise in your responses.{premium_content}

Current user: {ctx.context.username}
"""
    return prompt_with_handoff_instructions(base_instructions)

# -----------------------------------------------------------------------------
# SECTION 6: Agent Implementations
# -----------------------------------------------------------------------------

def create_calculator_agent() -> Agent[CustomerContext]:
    """Create the calculator agent."""
    return Agent[CustomerContext](
        name="Calculator Agent",
        handoff_description="Specialized agent for handling complex math calculations and conversions",
        instructions=calculator_instructions,
        model="gpt-4o",
        model_settings=ModelSettings(temperature=0.1),
        tools=[
            convert_temperature,
            convert_length,
            get_math_constant,
            get_user_details,
        ],
        hooks=AgentLifecycleHooks(),
        input_guardrails=[InputGuardrail(guardrail_function=math_query_guardrail)],
        output_type=CalculationResult,
    )


def create_customer_service_agent() -> Agent[CustomerContext]:
    """Create the customer service agent with calculator as tool."""
    # Create the calculator agent
    calculator_agent = create_calculator_agent()
    
    # Create a tool from the calculator agent
    calculator_tool = calculator_agent.as_tool(
        tool_name="use_calculator",
        tool_description="Use the specialized calculator to handle complex math calculations and conversions",
    )
    
    # Create the main customer service agent
    return Agent[CustomerContext](
        name="Customer Service Agent",
        instructions=customer_service_instructions,
        model="gpt-4o",
        model_settings=ModelSettings(temperature=0.3),
        tools=[
            convert_temperature,
            convert_length,
            get_math_constant,
            get_user_details,
            calculator_tool,
        ],
        output_type=CustomerResponse,
        hooks=AgentLifecycleHooks(),
        input_guardrails=[InputGuardrail(guardrail_function=math_query_guardrail)],
        output_guardrails=[OutputGuardrail(guardrail_function=response_quality_guardrail)],
        handoffs=[calculator_agent],
    )

# -----------------------------------------------------------------------------
# SECTION 7: Demo Execution
# -----------------------------------------------------------------------------

async def run_demo(agent: Agent[CustomerContext], query: str, user_context: CustomerContext) -> None:
    """Run a demo with the given query and context."""
    print(f"\n=== New Query ===\nCustomer: {query}")
    
    try:
        result = await Runner.run(agent, query, context=user_context)
        response = result.final_output_as(CustomerResponse)
        
        print(f"Agent: {response.answer}")
        print(f"Sentiment: {response.sentiment}")
        print(f"Requires escalation: {response.requires_escalation}")
        
        if response.calculation_result:
            print(f"Calculation: {response.calculation_result.result}")
            if response.calculation_result.units:
                print(f"Units: {response.calculation_result.units}")
            print(f"Explanation: {response.calculation_result.explanation}")
    
    except InputGuardrailTripwireTriggered as e:
        print(f"Input guardrail triggered: Please provide more information for your calculation request.")
    except OutputGuardrailTripwireTriggered as e:
        print(f"Output guardrail triggered: Response did not meet quality standards.")
    except Exception as e:
        print(f"Error: {e}")


async def main() -> None:
    """Main execution function."""
    # Enable or disable tracing as needed
    # set_tracing_disabled(False)  # Enable for debugging/monitoring
    
    print("=== Combined Agents Demo ===")
    
    # Create the customer service agent (which includes the calculator agent)
    customer_service_agent = create_customer_service_agent()
    
    # Create user contexts
    regular_user = CustomerContext(
        customer_id="user123",
        username="Alice",
        premium_user=False
    )
    
    premium_user = CustomerContext(
        customer_id="premium456",
        username="Bob",
        premium_user=True
    )
    
    # Run demonstration scenarios
    await run_demo(
        customer_service_agent,
        "Can you convert 30 degrees Celsius to Fahrenheit?",
        regular_user
    )
    
    await run_demo(
        customer_service_agent,
        "What is the value of Pi and why is it important in mathematics?",
        regular_user
    )
    
    await run_demo(
        customer_service_agent,
        "I need to convert 5 meters to feet for my home renovation project.",
        premium_user
    )
    
    await run_demo(
        customer_service_agent,
        "Can you help me with these conversions: 100°F to Celsius, 10 miles to kilometers, and what is the golden ratio?",
        premium_user
    )
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())