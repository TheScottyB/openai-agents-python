#!/usr/bin/env python3

"""
Tool Keeper Demo
===============

This script demonstrates the usage of the ToolKeeper agent for validating,
analyzing, and documenting tool definitions.

To run:
    python test_tool_keeper.py
"""

import asyncio
import json
from typing import Dict, Any

from tool_keeper import ToolKeeper


async def test_tool_analysis(tool_keeper: ToolKeeper, tool_def: Dict[str, Any]) -> None:
    """Test the tool analysis functionality.
    
    Args:
        tool_keeper: The ToolKeeper instance
        tool_def: The tool definition to analyze
    """
    print("\n=== Testing Tool Analysis ===")
    print(f"Analyzing tool: {tool_def['name']}")
    
    # Use the agent for analysis
    print("\nRunning analysis...")
    analysis = await tool_keeper.analyze_tool_directly(json.dumps(tool_def))
    print("Analysis Result:")
    print(json.dumps(analysis, indent=2))
    
    # Get general feedback using the main agent
    print("\nGetting general feedback...")
    result = await tool_keeper.run(f"Provide feedback on this tool definition: {json.dumps(tool_def)}")
    print("\nGeneral Feedback:")
    print(result)


async def test_tool_validation(tool_keeper: ToolKeeper, tool_def: Dict[str, Any]) -> None:
    """Test the tool validation functionality.
    
    Args:
        tool_keeper: The ToolKeeper instance
        tool_def: The tool definition to validate
    """
    print("\n=== Testing Tool Validation ===")
    print(f"Validating tool: {tool_def['name']}")
    
    # Run validation
    validation = await tool_keeper.validate_tool_directly(json.dumps(tool_def))
    print("\nValidation Result:")
    print(json.dumps(validation, indent=2))


async def test_tool_documentation(tool_keeper: ToolKeeper, tool_def: Dict[str, Any]) -> None:
    """Test the tool documentation functionality.
    
    Args:
        tool_keeper: The ToolKeeper instance
        tool_def: The tool definition to document
    """
    print("\n=== Testing Tool Documentation ===")
    print(f"Generating documentation for tool: {tool_def['name']}")
    
    # Generate documentation
    docs = await tool_keeper.document_tool_directly(json.dumps(tool_def))
    print("\nGenerated Documentation:")
    print(docs)


async def test_improvement_suggestions(tool_keeper: ToolKeeper, tool_def: Dict[str, Any]) -> None:
    """Test the improvement suggestions functionality.
    
    Args:
        tool_keeper: The ToolKeeper instance
        tool_def: The tool definition to improve
    """
    print("\n=== Testing Improvement Suggestions ===")
    print(f"Getting improvement suggestions for tool: {tool_def['name']}")
    
    # Ask for specific improvement suggestions
    prompt = f"""
    Please suggest improvements for this tool definition to make it follow OpenAI Agents SDK best practices:
    
    ```json
    {json.dumps(tool_def, indent=2)}
    ```
    
    Focus on:
    1. Error handling
    2. Documentation quality
    3. Parameter typing
    4. SDK integration
    """
    
    result = await tool_keeper.run(prompt)
    print("\nImprovement Suggestions:")
    print(result)


async def main():
    """Main test function."""
    print("=== Starting Tool Keeper Demo ===")
    
    # Create the Tool Keeper instance
    tool_keeper = ToolKeeper()
    
    # Example tool definitions
    browser_tool = {
        "name": "goto",
        "description": "Navigate to a specific URL",
        "parameters": {
            "url": {
                "type": "string",
                "description": "The URL to navigate to"
            }
        },
        "function": "computer.goto"
    }
    
    # Incomplete tool to test validation
    incomplete_tool = {
        "name": "search",
        "parameters": {
            "query": {
                "type": "string"
            }
        }
    }
    
    # Run tests with the complete tool
    await test_tool_analysis(tool_keeper, browser_tool)
    await test_tool_validation(tool_keeper, browser_tool)
    await test_tool_documentation(tool_keeper, browser_tool)
    await test_improvement_suggestions(tool_keeper, browser_tool)
    
    # Run tests with the incomplete tool
    print("\n\n=== Testing with Incomplete Tool ===")
    await test_tool_validation(tool_keeper, incomplete_tool)
    await test_improvement_suggestions(tool_keeper, incomplete_tool)
    
    print("\n=== Tool Keeper Demo Completed ===")


if __name__ == "__main__":
    asyncio.run(main())