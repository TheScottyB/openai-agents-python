#!/usr/bin/env python3

"""
Tool Keeper Test
===============

Tests for the ToolKeeper class which validates, analyzes, and documents
tool definitions for the OpenAI Agents SDK.

To run this test:
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
    
    # Direct method approach
    analysis = await tool_keeper.analyze_tool_directly(json.dumps(tool_def))
    print("\nDirect Analysis Result:")
    print(json.dumps(analysis, indent=2))
    
    # Agent-based approach
    result = await tool_keeper.run(f"Please analyze this tool definition: {json.dumps(tool_def)}")
    print("\nAgent Analysis Result:")
    print(result)


async def test_tool_validation(tool_keeper: ToolKeeper, tool_def: Dict[str, Any]) -> None:
    """Test the tool validation functionality.
    
    Args:
        tool_keeper: The ToolKeeper instance
        tool_def: The tool definition to validate
    """
    print("\n=== Testing Tool Validation ===")
    print(f"Validating tool: {tool_def['name']}")
    
    # Direct method approach
    validation = await tool_keeper.validate_tool_directly(json.dumps(tool_def))
    print("\nDirect Validation Result:")
    print(json.dumps(validation, indent=2))
    
    # Agent-based approach
    result = await tool_keeper.run(f"Please validate this tool definition: {json.dumps(tool_def)}")
    print("\nAgent Validation Result:")
    print(result)


async def test_tool_documentation(tool_keeper: ToolKeeper, tool_def: Dict[str, Any]) -> None:
    """Test the tool documentation functionality.
    
    Args:
        tool_keeper: The ToolKeeper instance
        tool_def: The tool definition to document
    """
    print("\n=== Testing Tool Documentation ===")
    print(f"Generating documentation for tool: {tool_def['name']}")
    
    # Direct method approach
    docs = await tool_keeper.document_tool(None, json.dumps(tool_def))
    print("\nGenerated Documentation:")
    print(docs)
    
    # Agent-based approach (shortened for brevity)
    result = await tool_keeper.run(f"Generate documentation for this tool: {json.dumps(tool_def)}")
    print("\nAgent Documentation (Preview):")
    print(result[:200] + "..." if len(result) > 200 else result)


async def test_tool_improvement(tool_keeper: ToolKeeper, tool_def: Dict[str, Any]) -> None:
    """Test the tool improvement suggestions functionality.
    
    Args:
        tool_keeper: The ToolKeeper instance
        tool_def: The tool definition to improve
    """
    print("\n=== Testing Tool Improvement Suggestions ===")
    print(f"Getting improvement suggestions for tool: {tool_def['name']}")
    
    result = await tool_keeper.run(
        f"Please suggest improvements for this tool definition: {json.dumps(tool_def)}"
    )
    print("\nImprovement Suggestions:")
    print(result)


async def main():
    """Main test function."""
    print("Starting Tool Keeper Tests")
    
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
    
    # Run tests
    await test_tool_analysis(tool_keeper, browser_tool)
    await test_tool_validation(tool_keeper, browser_tool)
    await test_tool_documentation(tool_keeper, browser_tool)
    await test_tool_improvement(tool_keeper, browser_tool)
    
    # Test with incomplete tool
    print("\n\n=== Testing with Incomplete Tool ===")
    await test_tool_validation(tool_keeper, incomplete_tool)
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    asyncio.run(main())