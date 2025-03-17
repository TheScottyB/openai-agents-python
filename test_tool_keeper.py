#!/usr/bin/env python3

"""
Tool Keeper Demo
===============

This script demonstrates the ToolKeeper agent which analyzes, validates,
and generates documentation for tool definitions.

To run:
    python test_tool_keeper.py
"""

import asyncio
import json
from tool_keeper import ToolKeeper, EXAMPLE_TOOL


async def demo_tool_analysis(tool_keeper, tool_definition):
    """Demo the tool analysis functionality."""
    print("\n=== Tool Analysis ===")
    
    result = await tool_keeper.run(
        f"Analyze this tool definition in detail and recommend improvements: ```{json.dumps(tool_definition, indent=2)}```"
    )
    
    print(result)


async def demo_tool_validation(tool_keeper, tool_definition):
    """Demo the tool validation functionality."""
    print("\n=== Tool Validation ===")
    
    result = await tool_keeper.run(
        f"Validate this tool definition against OpenAI Agents SDK requirements: ```{json.dumps(tool_definition, indent=2)}```"
    )
    
    print(result)


async def demo_tool_documentation(tool_keeper, tool_definition):
    """Demo the tool documentation generation."""
    print("\n=== Tool Documentation ===")
    
    result = await tool_keeper.run(
        f"Generate comprehensive documentation for this tool: ```{json.dumps(tool_definition, indent=2)}```"
    )
    
    print(result)


async def demo_with_incomplete_tool(tool_keeper):
    """Demo with an incomplete tool definition."""
    incomplete_tool = {
        "name": "search_web",
        "parameters": {
            "query": {
                "type": "string"
            }
        }
    }
    
    print("\n=== Analyzing Incomplete Tool ===")
    result = await tool_keeper.run(
        f"Analyze this incomplete tool definition and provide specific improvement suggestions: ```{json.dumps(incomplete_tool, indent=2)}```"
    )
    print(result)


async def main():
    """Main demo function."""
    # Create the Tool Keeper with the default model
    tool_keeper = ToolKeeper()
    
    print("=== Tool Keeper Demo ===")
    print("This agent analyzes, validates, and documents tool definitions")
    
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
    
    # Run demos with the various functionalities
    await demo_tool_analysis(tool_keeper, browser_tool)
    await demo_tool_validation(tool_keeper, browser_tool)
    await demo_tool_documentation(tool_keeper, browser_tool)
    
    # Demo with incomplete tool
    await demo_with_incomplete_tool(tool_keeper)
    
    # Demo with the example tool from the module
    print("\n=== Using Built-in Example Tool ===")
    print(f"Example tool: {json.dumps(EXAMPLE_TOOL, indent=2)}")
    
    result = await tool_keeper.run(
        "Create a Python function implementation for the built-in example tool"
    )
    print("\nGenerated Implementation:")
    print(result)
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())