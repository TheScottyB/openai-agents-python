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
    print("\n=== Tool Keeper Demo ===")
    print("This demo shows the functionality of the enhanced ToolKeeper system.")
    
    # Create the Tool Keeper
    tool_keeper = ToolKeeper()
    
    # Example tool definition
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
    
    # Test the basic analyze functionality first
    print("\nAnalyzing tool definition...")
    result = await tool_keeper.analyze_tool(json.dumps(browser_tool))
    print(f"Analysis result:\n{result}\n")
    
    # Test the validation functionality
    print("\nValidating tool definition...")
    result = await tool_keeper.validate_tool(json.dumps(browser_tool))
    print(f"Validation result:\n{result}\n")
    
    # Test the judge evaluation
    print("\nEvaluating tool definition with judge agent...")
    result = await tool_keeper.evaluate_tool(json.dumps(browser_tool))
    print(f"Evaluation result:\n{result}\n")
    
    # Test the comprehensive parallel processing
    print("\nProcessing tool definition comprehensively in parallel...")
    print("(This may take a moment as it runs all agents in parallel)")
    results = await tool_keeper.process_tool_comprehensive(json.dumps(EXAMPLE_TOOL))
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print("\nComprehensive processing completed successfully!")
        print("Results include validation, analysis, documentation, implementation, and evaluation.")
    
    # Skip the interactive part which causes EOF errors during testing
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo interrupted.")
        sys.exit(0)