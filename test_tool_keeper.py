#!/usr/bin/env python3

"""
Tool Keeper Demo
===============

This script demonstrates the enhanced ToolKeeper multi-agent system with guardrails,
parallel processing, LLM-as-judge, and multimodal capabilities.

To run:
    python test_tool_keeper.py
"""

import asyncio
import json
import sys
from agents import trace
from tool_keeper import ToolKeeper, EXAMPLE_TOOL


async def demo_with_streaming(tool_keeper: ToolKeeper, prompt: str) -> None:
    """Demonstrate streaming response for a given prompt."""
    print(f"\n=== Streaming Response Demo ===\nQuery: {prompt}")
    print("\nResponse: ", end="", flush=True)
    
    async for chunk in tool_keeper.process_request_stream(prompt):
        print(chunk, end="", flush=True)
        await asyncio.sleep(0.01)  # Small delay to make streaming more visible
    
    print("\n")


async def demo_comprehensive_processing(tool_keeper: ToolKeeper, tool_definition: dict) -> None:
    """Demonstrate comprehensive parallel processing of a tool definition."""
    print(f"\n=== Comprehensive Tool Processing Demo ===")
    print("This demonstrates parallel execution of all specialized agents for maximum efficiency.")
    
    tool_json = json.dumps(tool_definition, indent=2)
    print(f"\nTool Definition:\n{tool_json}\n")
    
    print("Processing with all agents in parallel...\n")
    with trace("Comprehensive Tool Processing Demo"):
        results = await tool_keeper.process_tool_comprehensive(tool_json)
    
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print("\n=== VALIDATION RESULTS ===\n")
        print(results["validation"])
        
        print("\n=== ANALYSIS RESULTS ===\n")
        print(results["analysis"])
        
        print("\n=== QUALITY EVALUATION ===\n")
        print(results["evaluation"])
        
        print("\n=== DOCUMENTATION ===\n")
        print(results["documentation"])
        
        print("\n=== IMPLEMENTATION ===\n")
        print(results["implementation"])


async def demo_judge_evaluation(tool_keeper: ToolKeeper, tool_definition: dict) -> None:
    """Demonstrate LLM-as-judge pattern with tool evaluation."""
    print(f"\n=== LLM-as-Judge Pattern Demo ===")
    print("This demonstrates using the judge agent to evaluate tool quality.")
    
    tool_json = json.dumps(tool_definition, indent=2)
    print(f"\nTool Definition:\n{tool_json}\n")
    
    print("Evaluating tool...\n")
    evaluation = await tool_keeper.evaluate_tool(tool_json)
    print(f"Evaluation Results:\n{evaluation}")


async def demo_guardrail_protection(tool_keeper: ToolKeeper) -> None:
    """Demonstrate input and output guardrails."""
    print(f"\n=== Guardrails Protection Demo ===")
    print("This demonstrates how guardrails protect against invalid or inappropriate content.")
    
    # Example of invalid tool structure to test input guardrail
    invalid_tool = {
        "name": "test_tool",
        # Missing required fields like description and parameters
    }
    
    print("\nTesting input guardrail with invalid tool schema:")
    result = await tool_keeper.analyze_tool(json.dumps(invalid_tool))
    print(f"\nResult: {result}")
    
    # Example of tool with potentially sensitive data to test output guardrail
    sensitive_tool = {
        "name": "connect_database",
        "description": "Connect to a database using credentials",
        "parameters": {
            "connection_string": {
                "type": "string",
                "description": "Database connection string with credentials"
            },
            "password": {
                "type": "string",
                "description": "Database password"
            }
        }
    }
    
    print("\nTesting output guardrail with potentially sensitive tool:")
    implementation = await tool_keeper.generate_implementation(json.dumps(sensitive_tool))
    print(f"\nImplementation (should be protected from generating actual credentials):\n{implementation}")


async def main():
    """Main demonstration function."""
    print("\n=== Enhanced Tool Keeper Multi-Agent System Demo ===")
    print("This demo shows the improved ToolKeeper system with advanced capabilities:")
    print("- Input and output guardrails for content validation")
    print("- Parallel processing for maximum efficiency")
    print("- LLM-as-judge pattern for quality evaluation")
    print("- Enhanced streaming responses with agent handoffs")
    
    # Create the Tool Keeper
    tool_keeper = ToolKeeper()
    
    # Example tool definitions
    browser_tool = {
        "name": "goto_url",
        "description": "Navigate to a specific URL in the browser",
        "parameters": {
            "url": {
                "type": "string",
                "description": "The URL to navigate to"
            },
            "new_tab": {
                "type": "boolean",
                "description": "Whether to open the URL in a new tab",
                "required": False
            }
        }
    }
    
    # Demo 1: Streaming response with agent handoffs
    await demo_with_streaming(
        tool_keeper,
        f"Validate this tool definition against OpenAI Agents SDK requirements: {json.dumps(browser_tool)}"
    )
    
    # Demo 2: Judge evaluation
    await demo_judge_evaluation(tool_keeper, browser_tool)
    
    # Demo 3: Guardrails protection
    await demo_guardrail_protection(tool_keeper)
    
    # Demo 4: Comprehensive parallel processing
    await demo_comprehensive_processing(tool_keeper, EXAMPLE_TOOL)
    
    # Try a complex request that demonstrates router handoffs
    await demo_with_streaming(
        tool_keeper,
        f"""I have this tool definition:
        
        {json.dumps(EXAMPLE_TOOL, indent=2)}
        
        Could you validate it, suggest any improvements, and then generate Python implementation?"""
    )
    
    print("\nWould you like to start an interactive chat session? (y/n)")
    response = input()
    
    if response.lower() in ("y", "yes"):
        await tool_keeper.chat()
    else:
        print("\n=== Demo Complete ===")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo interrupted.")
        sys.exit(0)