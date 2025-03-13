from tool_keeper import ToolKeeper
import json
import asyncio

async def main():
    # Create a Tool Keeper instance
    tool_keeper = ToolKeeper()

    # Define one of our browser automation tools for analysis
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

    # Get tool analysis
    print("Analyzing browser navigation tool...")
    result = await tool_keeper.run(f"Analyze this browser automation tool: {json.dumps(browser_tool)}")
    print("\nAnalysis Result:")
    print(result)

    # Get validation results
    print("\nValidating tool definition...")
    result = await tool_keeper.run(f"Validate this tool definition: {json.dumps(browser_tool)}")
    print("\nValidation Results:")
    print(result)

    # Get documentation
    print("\nGenerating documentation for the tool...")
    result = await tool_keeper.run(f"Generate documentation for this tool: {json.dumps(browser_tool)}")
    print("\nDocumentation Generated:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
