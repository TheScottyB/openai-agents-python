#!/usr/bin/env python3
"""
Core Concepts Demo for OpenAI Agents SDK

This file demonstrates the fundamental concepts of the OpenAI Agents SDK including:
- Basic agent configuration
- Tool implementation and usage
- Context and state management
- Guardrails for input/output validation
- Hooks for lifecycle events
- Basic logging and tracing

Usage:
    python core_concepts_demo.py [--verbose]

Requirements:
    - OpenAI Agents SDK
    - Valid OpenAI API key set as OPENAI_API_KEY environment variable
    - Python 3.10+

Author: Your Name
Created: YYYY-MM-DD
"""

import os
import sys
import logging
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union

# Import the OpenAI Agents SDK components
try:
    from agents import (
        Agent, 
        ModelSettings, 
        RunContext,
        AgentHooks,
        function_tool,
        Guardrail
    )
except ImportError:
    print("Error: OpenAI Agents SDK not found. Please install it with: pip install openai-agents")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("core_concepts_demo")

# Enable more detailed logging if verbose flag is provided
if "--verbose" in sys.argv:
    logger.setLevel(logging.DEBUG)
    logger.debug("Verbose logging enabled")

# Check for OpenAI API key
if not os.environ.get("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY environment variable not set. Please set it before running this demo.")
    sys.exit(1)


# --------------------------------
# CONTEXT AND STATE MANAGEMENT
# --------------------------------

@dataclass
class CalculatorContext:
    """
    Context object for the Calculator agent.
    
    This represents the state and configuration passed to the agent during operation.
    It can be extended with additional fields as needed.
    """
    user_id: str
    session_id: str
    history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []
    
    def add_to_history(self, operation: str, result: float):
        """Add an operation and its result to the calculation history."""
        self.history.append({
            "operation": operation,
            "result": result,
            "timestamp": asyncio.get_event_loop().time()
        })
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the full calculation history."""
        return self.history


# --------------------------------
# TOOL IMPLEMENTATION
# --------------------------------

class CalculatorTool:
    """
    A simple calculator tool that provides basic mathematical operations.
    
    This demonstrates how to implement tools that agents can use to perform tasks.
    """
    
    @staticmethod
    @function_tool
    def add(a: float, b: float) -> float:
        """Add two numbers and return the result."""
        return a + b
    
    @staticmethod
    @function_tool
    def subtract(a: float, b: float) -> float:
        """Subtract b from a and return the result."""
        return a - b
    
    @staticmethod
    @function_tool
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers and return the result."""
        return a * b
    
    @staticmethod
    @function_tool
    def divide(a: float, b: float) -> float:
        """
        Divide a by b and return the result.
        
        Args:
            a: The dividend
            b: The divisor (must not be zero)
            
        Returns:
            The result of the division
            
        Raises:
            ValueError: If b is zero
        """
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    @staticmethod
    @function_tool
    def calculate(expression: str) -> float:
        """
        Evaluate a mathematical expression and return the result.
        
        Args:
            expression: A string containing a mathematical expression
            
        Returns:
            The result of evaluating the expression
            
        Raises:
            ValueError: If the expression is invalid or contains unauthorized functions
        """
        # This is a simplified and unsafe implementation for demonstration
        # In a real application, you would use a secure math expression evaluator
        allowed_chars = set("0123456789+-*/() .")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Expression contains invalid characters")
        
        try:
            # WARNING: eval is dangerous and should not be used in production
            # This is just for demonstration purposes
            return eval(expression)
        except Exception as e:
            raise ValueError(f"Invalid expression: {e}")


# --------------------------------
# HOOKS FOR LIFECYCLE EVENTS
# --------------------------------

class CalculatorHooks(AgentHooks[CalculatorContext]):
    """
    Hooks to monitor and interact with the agent lifecycle.
    
    These hooks allow us to log events, update state, and perform actions
    at different points in the agent's execution.
    """
    
    async def on_run_start(self, context: RunContext[CalculatorContext]) -> None:
        """Called when an agent run starts."""
        logger.debug(f"Starting calculator session for user: {context.context.user_id}")
    
    async def on_run_end(self, context: RunContext[CalculatorContext]) -> None:
        """Called when an agent run completes."""
        logger.debug(f"Calculator session completed for user: {context.context.user_id}")
        logger.debug(f"Session history: {context.context.get_history()}")
    
    async def on_tool_call_start(
        self, 
        context: RunContext[CalculatorContext], 
        tool_name: str, 
        tool_args: Dict[str, Any]
    ) -> None:
        """Called when a tool is about to be used."""
        logger.debug(f"Using tool: {tool_name} with args: {tool_args}")
    
    async def on_tool_call_end(
        self, 
        context: RunContext[CalculatorContext], 
        tool_name: str, 
        tool_args: Dict[str, Any], 
        result: Any
    ) -> None:
        """Called after a tool has been used."""
        logger.debug(f"Tool {tool_name} returned: {result}")
        
        # Add operation to history if it's a calculation
        if tool_name in ["add", "subtract", "multiply", "divide", "calculate"]:
            operation = f"{tool_name}: {tool_args}"
            context.context.add_to_history(operation, result)


# --------------------------------
# GUARDRAILS
# --------------------------------

class InputGuardrail(Guardrail[CalculatorContext]):
    """
    A guardrail to validate user input before processing.
    
    This demonstrates how to implement safety checks and validations
    to ensure agents behave appropriately.
    """
    
    async def run(
        self, 
        context: RunContext[CalculatorContext], 
        input_text: str
    ) -> Optional[str]:
        """
        Check if the input is appropriate.
        
        Args:
            context: The run context
            input_text: The user's input text
            
        Returns:
            None if the input is valid, or an error message if not
        """
        # Check for empty input
        if not input_text.strip():
            return "Please provide a calculation or question."
        
        # Check for excessively long input
        if len(input_text) > 500:
            return "Input is too long. Please keep your request under 500 characters."
        
        # Check for valid character set (for a calculator)
        valid_chars = set("0123456789+-*/() .abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ?,= ")
        if not all(c in valid_chars for c in input_text):
            return "Input contains invalid characters. Please use only numbers, basic operators, and letters."
        
        return None  # Input is valid

class OutputGuardrail(Guardrail[CalculatorContext]):
    \"\"\"
    A guardrail to validate agent output before returning to the user.
    
    This demonstrates how to implement safety checks on agent responses.
    \"\"\"
    
    async def run(
        self, 
        context: RunContext[CalculatorContext], 
        output_text: str
    ) -> Optional[str]:
        \"\"\"
        Check if the output is appropriate.
        
        Args:
            context: The run context
            output_text: The agent's output text
            
        Returns:
            None if the output is valid, or a replacement message if not
        \"\"\"
        # Check for empty output
        if not output_text.strip():
            return \"I couldn't generate a response. Please try again with a clearer request.\"
        
        # Check for excessively long output
        if len(output_text) > 1000:
            logger.warning(\"Output exceeded length limit, truncating...\")
            return output_text[:997] + \"...\"
        
        return None  # Output is valid


# --------------------------------
# CALCULATOR AGENT IMPLEMENTATION
# --------------------------------

class CalculatorAgent:
    \"\"\"
    The main CalculatorAgent class that brings together all components.
    This demonstrates how to orchestrate tools, guardrails, and hooks.
    \"\"\"
    
    def __init__(self):
        self.calculator_tool = CalculatorTool()
        self.hooks = CalculatorHooks()
        self.input_guardrail = InputGuardrail()
        self.output_guardrail = OutputGuardrail()
        
        self.agent = Agent[CalculatorContext](
            name="Calculator Agent",
            instructions=\"\"\"
                You are a helpful calculator assistant. 
                Perform calculations and explain your steps clearly.
                Always validate input using available tools before proceeding.
            \"\"\",
            tools=[
                self.calculator_tool.add,
                self.calculator_tool.subtract,
                self.calculator_tool.multiply,
                self.calculator_tool.divide,
                self.calculator_tool.calculate
            ],
            hooks=self.hooks,
            guardrails=[self.input_guardrail, self.output_guardrail],
            model="gpt-4",
            model_settings=ModelSettings(temperature=0.3)
        )
    
    async def run(self, context: CalculatorContext, query: str) -> str:
        \"\"\"
        Run the calculator agent with the given query.
        
        Args:
            context: The calculator context with user and session info
            query: The user's calculation request
            
        Returns:
            The result of the calculation with explanation
        \"\"\"
        try:
            result = await self.agent.run(context, query)
            return result
        except Exception as e:
            logger.error(f"Error during calculation: {e}")
            return "Sorry, something went wrong with the calculation."


async def demonstrate_calculator() -> None:
    \"\"\"
    Demonstrate core calculator functionality.
    
    This function shows:
    - Basic calculator operations
    - Guardrails in action
    - Lifecycle hooks
    \"\"\"
    # Create context for demonstration
    context = CalculatorContext(
        user_id="demo_user",
        session_id="demo_session"
    )
    
    # Initialize agent
    calculator = CalculatorAgent()
    
    # Demonstrate basic operations
    operations = [
        "What is 2 + 2?",
        "Calculate 15 * (3 + 7)",
        "Help me compute 100 / 0",
        "This input is way too long and should be rejected by the guardrail"
    ]
    
    for operation in operations:
        print(f"\n[User Query: {operation}]")
        result = await calculator.run(context, operation)
        print(f"[Agent Response: {result}]")
        print(f"[Session History: {context.get_history()}]")
        
        # Add small delay for clarity
        await asyncio.sleep(1)


def main():
    \"\"\"
    Main entry point for the calculator demonstration.
    \"\"\"
    parser = argparse.ArgumentParser(description="Calculator Agent Demo")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    asyncio.run(demonstrate_calculator())


if __name__ == "__main__":
    main()

