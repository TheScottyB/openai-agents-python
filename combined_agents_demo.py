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
- Unit testing and error handling examples

To run:
    python combined_agents_demo.py

For testing:
    uv run pytest tests/test_combined_agents.py -v

Requirements:
    - OpenAI API key set as OPENAI_API_KEY environment variable
"""

import asyncio
import os
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, cast

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
from agents.exceptions import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    UserError,
    ModelBehaviorError,
)
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("combined_agents")

# -----------------------------------------------------------------------------
# SECTION 1: Context and Output Types
# -----------------------------------------------------------------------------

class UserType(str, Enum):
    """User type enumeration for stronger typing."""
    REGULAR = "regular"
    PREMIUM = "premium"
    ADMIN = "admin"


@dataclass
class CustomerContext:
    """Context for tracking customer information and interactions."""
    customer_id: str
    username: str
    user_type: UserType = UserType.REGULAR
    session_start_time: datetime = field(default_factory=datetime.now)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def premium_user(self) -> bool:
        """Check if the user has premium status."""
        return self.user_type in (UserType.PREMIUM, UserType.ADMIN)

    def log_interaction(self, tool_name: str, input_data: Any, output_data: Any) -> None:
        """Log a tool interaction to the conversation history."""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "input": input_data,
            "output": output_data,
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            "customer_id": self.customer_id,
            "username": self.username,
            "user_type": self.user_type,
            "premium_user": self.premium_user,
            "session_start": self.session_start_time.isoformat(),
            "interaction_count": len(self.conversation_history),
        }


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

class ConversionError(UserError):
    """Error raised when a conversion cannot be performed."""
    pass


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
    
    Raises:
        ConversionError: If the conversion cannot be performed
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
        
        # Validate input
        if not isinstance(value, (int, float)):
            raise ConversionError(f"Temperature value must be a number, got {type(value)}")
        
        # Convert units to single character
        from_char = from_unit.lower()[0]
        to_char = to_unit.lower()[0]
        
        if from_char not in "cfk" or to_char not in "cfk":
            raise ConversionError(
                f"Invalid temperature units. Supported units: celsius/c, fahrenheit/f, kelvin/k"
            )
        
        key = (from_char, to_char)
        if key not in conversions:
            raise ConversionError(f"Cannot convert from {from_unit} to {to_unit}")
            
        result = conversions[key](value)
        
        # Log the interaction
        ctx.context.log_interaction(
            "convert_temperature",
            {"value": value, "from_unit": from_unit, "to_unit": to_unit},
            result
        )
        
        logger.info(f"Temperature conversion: {value}{from_unit} → {result}{to_unit}")
        
        return {
            "success": True,
            "original_value": value,
            "original_unit": from_unit,
            "converted_value": result,
            "converted_unit": to_unit,
        }
    except ConversionError as e:
        logger.warning(f"Conversion error: {str(e)}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error in temperature conversion: {str(e)}", exc_info=True)
        return {"success": False, "error": f"An unexpected error occurred: {str(e)}"}


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
    
    Raises:
        ConversionError: If the conversion cannot be performed
    """
    try:
        # Define unit conversions to meters
        units = {"m": 1.0, "ft": 0.3048, "in": 0.0254, "mi": 1609.34, "km": 1000.0}
        
        # Validate input
        if not isinstance(value, (int, float)):
            raise ConversionError(f"Length value must be a number, got {type(value)}")
        
        if from_unit not in units:
            raise ConversionError(f"Invalid source unit: {from_unit}. Supported units: {', '.join(units.keys())}")
            
        if to_unit not in units:
            raise ConversionError(f"Invalid target unit: {to_unit}. Supported units: {', '.join(units.keys())}")
            
        # Convert to target unit
        result = value * units[from_unit] / units[to_unit]
        
        # Log the interaction
        ctx.context.log_interaction(
            "convert_length",
            {"value": value, "from_unit": from_unit, "to_unit": to_unit},
            result
        )
        
        logger.info(f"Length conversion: {value}{from_unit} → {result}{to_unit}")
        
        return {
            "success": True,
            "original_value": value,
            "original_unit": from_unit,
            "converted_value": result,
            "converted_unit": to_unit,
        }
    except ConversionError as e:
        logger.warning(f"Conversion error: {str(e)}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error in length conversion: {str(e)}", exc_info=True)
        return {"success": False, "error": f"An unexpected error occurred: {str(e)}"}


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
    
    Raises:
        ValueError: If the constant name is not recognized
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
        
        if not name or not isinstance(name, str):
            raise ValueError("Constant name must be a non-empty string")
            
        name_lower = name.lower()
        if name_lower not in constants:
            supported = ", ".join(constants.keys())
            raise ValueError(f"Unknown constant: {name}. Supported constants: {supported}")
            
        value, description = constants[name_lower]
        
        # Log the interaction
        ctx.context.log_interaction(
            "get_math_constant",
            {"name": name},
            {"value": value, "description": description}
        )
        
        logger.info(f"Math constant retrieved: {name}")
        
        return {
            "success": True,
            "constant_name": name,
            "value": value,
            "description": description,
        }
    except ValueError as e:
        logger.warning(f"Invalid constant request: {str(e)}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error retrieving constant: {str(e)}", exc_info=True)
        return {"success": False, "error": f"An unexpected error occurred: {str(e)}"}


@function_tool(use_docstring_info=True)
def get_user_details(
    ctx: RunContextWrapper[CustomerContext]
) -> Dict[str, Any]:
    """Get information about the current user.
    
    Returns:
        A dictionary containing user information
    """
    try:
        user = ctx.context
        
        user_info = {
            "customer_id": user.customer_id,
            "username": user.username,
            "user_type": user.user_type,
            "is_premium": user.premium_user,
            "session_started": user.session_start_time.isoformat(),
            "conversation_count": len(user.conversation_history),
        }
        
        logger.info(f"User details retrieved for {user.username}")
        
        return user_info
    except Exception as e:
        logger.error(f"Error retrieving user details: {str(e)}", exc_info=True)
        return {"error": "Could not retrieve user details"}

# -----------------------------------------------------------------------------
# SECTION 3: Lifecycle Hooks
# -----------------------------------------------------------------------------

class AgentLifecycleHooks(AgentHooks[CustomerContext]):
    """Custom hooks for monitoring agent lifecycle."""
    
    def __init__(self):
        self.start_time: Optional[datetime] = None
        self.tools_called: Dict[str, int] = {}

    async def on_start(
        self, context: RunContextWrapper[CustomerContext], agent: Agent[CustomerContext]
    ) -> None:
        """Called when an agent run starts."""
        self.start_time = datetime.now()
        self.tools_called = {}
        
        logger.info(
            f"Agent '{agent.name}' started for user {context.context.username} "
            f"({context.context.user_type})"
        )

    async def on_end(
        self,
        context: RunContextWrapper[CustomerContext],
        agent: Agent[CustomerContext],
        output: Any,
    ) -> None:
        """Called when an agent run completes."""
        if self.start_time:
            duration = datetime.now() - self.start_time
            logger.info(
                f"Agent '{agent.name}' completed in {duration.total_seconds():.2f}s "
                f"with {sum(self.tools_called.values())} tool calls"
            )
            
            if self.tools_called:
                tool_summary = ", ".join(f"{tool}: {count}" for tool, count in self.tools_called.items())
                logger.info(f"Tools used: {tool_summary}")

    async def on_tool_start(
        self,
        context: RunContextWrapper[CustomerContext],
        agent: Agent[CustomerContext],
        tool: Any,
    ) -> None:
        """Called when a tool is invoked."""
        tool_name = getattr(tool, "name", str(tool))
        self.tools_called[tool_name] = self.tools_called.get(tool_name, 0) + 1
        logger.info(f"Tool '{tool_name}' called by agent '{agent.name}'")

    async def on_tool_error(
        self,
        context: RunContextWrapper[CustomerContext],
        agent: Agent[CustomerContext],
        tool: Any,
        error: Exception,
    ) -> None:
        """Called when a tool execution fails."""
        tool_name = getattr(tool, "name", str(tool))
        logger.error(f"Tool '{tool_name}' failed: {str(error)}", exc_info=True)

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
        instructions=(
            "Determine if the user query is related to math calculations or unit conversions. "
            "Check if it contains enough information to provide a useful response."
        ),
        model="gpt-3.5-turbo",
        model_settings=ModelSettings(temperature=0.1),
        output_type=MathScreenerOutput,
    )
    
    input_str = input_data if isinstance(input_data, str) else str(input_data)
    
    # Skip guardrail for very short inputs (will fail anyway)
    if len(input_str.split()) < 2:
        return GuardrailFunctionOutput(
            output_info=MathScreenerOutput(
                is_math_query=False,
                reasoning="Input is too short to analyze"
            ),
            tripwire_triggered=False,
        )
    
    try:
        result = await Runner.run(math_screener, input_str, context=ctx.context)
        output = result.final_output_as(MathScreenerOutput)
        
        # If it's a math query, check for minimum required info
        if output.is_math_query and len(input_str.split()) < 4:
            logger.warning(f"Math query guardrail triggered: {input_str}")
            return GuardrailFunctionOutput(
                output_info=output,
                tripwire_triggered=True,
            )
        
        return GuardrailFunctionOutput(
            output_info=output,
            tripwire_triggered=False,
        )
    except Exception as e:
        logger.error(f"Error in math query guardrail: {str(e)}", exc_info=True)
        # In case of error, don't block the query but log the issue
        return GuardrailFunctionOutput(
            output_info=MathScreenerOutput(
                is_math_query=False,
                reasoning=f"Error analyzing query: {str(e)}"
            ),
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
    
    try:
        result = await Runner.run(quality_checker, output.answer, context=ctx.context)
        quality_output = result.final_output_as(ResponseQualityOutput)
        
        if not quality_output.is_appropriate:
            logger.warning(f"Response quality guardrail triggered: {quality_output.reasoning}")
        
        return GuardrailFunctionOutput(
            output_info=quality_output,
            tripwire_triggered=not quality_output.is_appropriate,
        )
    except Exception as e:
        logger.error(f"Error in response quality guardrail: {str(e)}", exc_info=True)
        # In case of error, don't block the response but log the issue
        return GuardrailFunctionOutput(
            output_info=ResponseQualityOutput(
                is_appropriate=True,
                reasoning=f"Error checking response quality: {str(e)}"
            ),
            tripwire_triggered=False,
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
- Length conversions (meters, feet, inches, miles, kilometers)
- Math constants (Pi, e, Golden Ratio, etc.)

Always show your work and include appropriate units in your results.{premium_content}

Current user: {ctx.context.username} (Type: {ctx.context.user_type})
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
3. Hand off to the Calculator Agent for specialized assistance

Always be professional, clear, and concise in your responses.{premium_content}

Current user: {ctx.context.username} (Type: {ctx.context.user_type})
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

async def run_demo(
    agent: Agent[CustomerContext], 
    query: str, 
    user_context: CustomerContext
) -> None:
    """Run a demo with the given query and context."""
    print(f"\n=== New Query ===\nCustomer: {query}")
    
    try:
        # Make a copy of the query for testing/debugging
        sanitized_query = query.strip()
        
        result = await Runner.run(agent, sanitized_query, context=user_context)
        response = result.final_output_as(CustomerResponse)
        
        print(f"Agent: {response.answer}")
        print(f"Sentiment: {response.sentiment}")
        print(f"Requires escalation: {response.requires_escalation}")
        
        if response.calculation_result:
            print(f"Calculation: {response.calculation_result.result}")
            if response.calculation_result.units:
                print(f"Units: {response.calculation_result.units}")
            print(f"Explanation: {response.calculation_result.explanation}")
    
    except InputGuardrailTripwireTriggered:
        print("Input guardrail triggered: Please provide more information for your calculation request.")
    except OutputGuardrailTripwireTriggered:
        print("Output guardrail triggered: Response did not meet quality standards.")
    except ModelBehaviorError as e:
        print(f"Model error: {str(e)}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")
        logger.error(f"Unexpected error in demo: {str(e)}", exc_info=True)


async def main() -> None:
    """Main execution function."""
    # Enable or disable tracing based on environment variable
    tracing_enabled = os.environ.get("ENABLE_TRACING", "").lower() in ("true", "1", "yes")
    set_tracing_disabled(not tracing_enabled)
    
    if tracing_enabled:
        print("Tracing enabled - visit the OpenAI platform to view traces")
    
    print("=== Combined Agents Demo ===")
    
    # Create the customer service agent (which includes the calculator agent)
    customer_service_agent = create_customer_service_agent()
    
    # Create user contexts
    regular_user = CustomerContext(
        customer_id="user123",
        username="Alice",
        user_type=UserType.REGULAR
    )
    
    premium_user = CustomerContext(
        customer_id="premium456",
        username="Bob",
        user_type=UserType.PREMIUM
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