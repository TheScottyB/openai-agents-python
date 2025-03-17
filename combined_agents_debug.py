#!/usr/bin/env python3

"""
Combined Agents - Debugging Example
==================================

This script demonstrates debugging techniques for OpenAI Agents SDK applications.
Run this with a debugger to explore how errors are handled and traced.

To use in VSCode:
1. Set breakpoints on the marked lines below
2. Run with debugging (F5) after selecting the "Python: Combined Agents Debug" configuration
3. Step through code and inspect variables

Features:
- Error handling examples
- Debugging and tracing
- Break on exception demonstrations
"""

import asyncio
import logging
import sys
from typing import Optional, Dict, Any, cast

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("agent_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("agent_debug")

# Import from our application
from combined_agents_demo import (
    UserType,
    CustomerContext,
    ConversionError,
    create_customer_service_agent,
)
from agents import Runner, ModelSettings, Agent, set_tracing_disabled
from agents.exceptions import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    ModelBehaviorError,
    UserError,
)

# Create a customer context for testing
def create_test_user() -> CustomerContext:
    """Create a test user for debugging."""
    # DEBUGGING TIP: Set a breakpoint here to inspect the created user
    return CustomerContext(
        customer_id="debug_user_123",
        username="DebugUser",
        user_type=UserType.REGULAR
    )

# Example of a function that will trigger an error for debugging
async def trigger_error_for_debugging(agent: Agent[CustomerContext], user: CustomerContext) -> None:
    """Deliberately trigger errors for debugging practice."""
    logger.info("Attempting to trigger an error for debugging...")
    
    try:
        # This will likely fail due to invalid inputs
        # DEBUGGING TIP: Set a breakpoint here and step into this function
        result = await Runner.run(
            agent, 
            "convert temp", # Too short to pass the guardrail
            context=user
        )
        print(f"Result: {result.final_output}")
    
    except InputGuardrailTripwireTriggered as e:
        # DEBUGGING TIP: Set a breakpoint here to examine the exception
        logger.warning(f"Input guardrail triggered as expected: {e}")
        print(f"Guardrail triggered: {e}")
    
    except Exception as e:
        # DEBUGGING TIP: Set exception breakpoints in VSCode to break on exceptions
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"Error: {type(e).__name__}: {e}")

# Test function to demonstrate proper error handling
async def run_with_error_handling(
    agent: Agent[CustomerContext], 
    query: str, 
    user: CustomerContext
) -> Optional[Dict[str, Any]]:
    """Run an agent with comprehensive error handling."""
    logger.info(f"Running query: {query}")
    
    try:
        # Make a copy of the query for testing/debugging
        sanitized_query = query.strip()
        
        # DEBUGGING TIP: Set a breakpoint here to examine inputs before the API call
        result = await Runner.run(agent, sanitized_query, context=user)
        
        # DEBUGGING TIP: Set a breakpoint here to examine the response
        return {
            "status": "success",
            "result": result.final_output,
            "response_type": type(result.final_output).__name__
        }
    
    except InputGuardrailTripwireTriggered as e:
        logger.warning(f"Input guardrail triggered: {e}")
        return {
            "status": "guardrail_input",
            "error": "Input validation failed. Please provide more information."
        }
    
    except OutputGuardrailTripwireTriggered as e:
        logger.warning(f"Output guardrail triggered: {e}")
        return {
            "status": "guardrail_output",
            "error": "The response did not meet quality standards."
        }
    
    except ModelBehaviorError as e:
        # Issues with the LLM response
        logger.error(f"Model behavior error: {e}", exc_info=True)
        return {
            "status": "model_error",
            "error": f"AI model error: {str(e)}"
        }
    
    except UserError as e:
        # Application-level errors (like conversion errors)
        logger.error(f"User error: {e}", exc_info=True)
        return {
            "status": "user_error",
            "error": str(e)
        }
    
    except Exception as e:
        # Catch-all for unexpected errors
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        return {
            "status": "error",
            "error": f"An unexpected error occurred: {type(e).__name__}"
        }

async def main() -> None:
    """Main debugging demo function."""
    # Disable tracing to reduce API calls during debugging
    set_tracing_disabled(True)
    
    logger.info("Starting debugging demo")
    print("=== Combined Agents Debugging Demo ===")
    
    # Create test user
    user = create_test_user()
    logger.info(f"Created test user: {user.username} ({user.user_type})")
    
    # Create the agent with debug settings
    try:
        # DEBUGGING TIP: Set a breakpoint here to inspect the agent creation
        print("Creating agent...")
        agent = create_customer_service_agent()
        agent.model_settings = ModelSettings(temperature=0.1)  # Lower temp for more predictable debugging
        
        print("\n=== Testing Error Triggering ===")
        await trigger_error_for_debugging(agent, user)
        
        print("\n=== Testing Proper Error Handling ===")
        # Test with valid query
        valid_result = await run_with_error_handling(
            agent,
            "What's the value of Pi?",
            user
        )
        print(f"Valid query result: {valid_result}")
        
        # Test with invalid query
        invalid_result = await run_with_error_handling(
            agent,
            "???",  # Likely to fail with various errors
            user
        )
        print(f"Invalid query result: {invalid_result}")
        
    except Exception as e:
        logger.critical(f"Fatal error in demo: {e}", exc_info=True)
        print(f"Fatal error: {e}")
    
    logger.info("Debugging demo completed")
    print("\n=== Demo Complete ===")
    print("See agent_debug.log for detailed logs")

if __name__ == "__main__":
    asyncio.run(main())