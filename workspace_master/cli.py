#!/usr/bin/env python3
import os
import sys
import argparse
import asyncio
import asyncio
from agents import Agent, Runner
from agents.tool import function_tool
from computers.computer import Computer
from typing import Any, Dict, List, Optional, Union
def parse_args():
    parser = argparse.ArgumentParser(description='Computer Using Agent CLI')
    parser.add_argument('--computer', default='local-playwright',
                      help='Computer environment to use (default: local-playwright)')
    parser.add_argument('--input', help='Initial input to the agent', default=None)
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--show', action='store_true', help='Show images during execution')
    parser.add_argument('--start-url', default='https://bing.com',
                      help='Start URL for browser environments')
    return parser.parse_args()

async def setup_computer(computer_type: str, start_url: str):
    # Import the appropriate computer class based on type
    if computer_type == 'local-playwright':
        from computers.local_playwright import LocalPlaywright
        computer = LocalPlaywright(start_url=start_url)
        await computer.setup()
        return computer
    elif computer_type == 'docker':
        from computers.docker import Docker
        computer = Docker()
        await computer.setup()
        return computer
    elif computer_type == 'browserbase':
        from computers.browserbase import Browserbase
        computer = Browserbase()
        await computer.setup()
        return computer
    elif computer_type == 'scrapybara-browser':
        from computers.scrapybara import ScrapybaraBrowser
        computer = ScrapybaraBrowser()
        await computer.setup()
        return computer
    elif computer_type == 'scrapybara-ubuntu':
        from computers.scrapybara import ScrapybaraUbuntu
        computer = ScrapybaraUbuntu()
        await computer.setup()
        return computer
    else:
        raise ValueError(f"Unknown computer type: {computer_type}")

# Define function schemas for the agent
def get_tools(computer: Computer) -> List[Dict[str, Any]]:
    tools = []
    
    @function_tool(name_override="goto", description_override="Navigate to a specific URL")
    async def goto(url: str):
        """Navigate to a specific URL in the browser.
        
        Args:
            url: The URL to navigate to
            
        Returns:
            str: Success message or error description
        """
        try:
            await computer.goto(url)
            return f"Successfully navigated to {url}"
        except ValueError as e:
            return f"Invalid URL error: {str(e)}"
        except TimeoutError as e:
            return f"Navigation timeout: {str(e)}"
        except Exception as e:
            return f"Navigation failed: {str(e)}"
    
    @function_tool(name_override="click_element", description_override="Click an element using a CSS selector")
    async def click_element(selector: str):
        """Click an element on the page using a CSS selector."""
        return await computer.click_element(selector)
    
    @function_tool(name_override="type_into", description_override="Type text into a specific element")
    async def type_into(selector: str, text: str):
        """Type text into an element on the page using a CSS selector."""
        return await computer.type_into(selector, text)
    
    @function_tool(name_override="get_screenshot", description_override="Take a screenshot of the current page")
    async def get_screenshot():
        """Take a screenshot of the current page and save it for reference."""
        screenshot_data = await computer.get_screenshot()
        # Instead of returning the raw bytes, return a message about the screenshot
        return "Screenshot taken of the current page."
    
    @function_tool(name_override="get_current_url", description_override="Get the current page URL")
    async def get_current_url():
        """Get the URL of the current page."""
        return await computer.get_current_url()
    
    @function_tool(name_override="back", description_override="Go back to the previous page")
    async def back():
        """Navigate back to the previous page."""
        return await computer.back()
    
    @function_tool(name_override="forward", description_override="Go forward to the next page")
    async def forward():
        """Navigate forward to the next page."""
        return await computer.forward()
    
    return [
        goto,
        click_element,
        type_into,
        get_screenshot,
        get_current_url,
        back,
        forward
    ]

async def run_agent(agent: Agent, runner: Runner, query: str, debug: bool = False, computer_type: str = None, show: bool = False):
    result = await runner.run(agent, query)
    print("\nAgent:", result.final_output)
    
    # If debug mode is enabled, show additional information
    if debug:
        print("\nDebug info:")
        print(f"Computer type: {computer_type}")
        print(f"Show images: {show}")
    
    return result

async def main_async():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set")
        sys.exit(1)

    args = parse_args()
    
    # Setup the computer environment
    try:
        computer = await setup_computer(args.computer, args.start_url)
    except Exception as e:
        print(f"Error setting up computer environment: {e}")
        sys.exit(1)

    # Create a workspace-aware agent with computer capabilities
    # Create a workspace-aware agent with computer capabilities
    agent = Agent(
        name="Computer Using Agent",
        instructions="""You are a computer-using assistant that can interact with the user's browser.
        Available actions:
        - Navigate to URLs with goto(url)
        - Click elements with click_element(selector)
        - Type text with type_into(selector, text)
        - Take screenshots (use sparingly)
        - Navigate with back() and forward()
        
        When browsing:
        1. Use precise CSS selectors
        2. Take screenshots only when necessary
        3. Handle errors gracefully
        
        Current directory: {}
        """.format(os.getcwd()),
        tools=get_tools(computer)
    )

    # Create a runner instance
    runner = Runner()

    print(f"Computer Using Agent initialized with {args.computer} environment.")
    print("Use Ctrl+C to exit.")

    # If initial input was provided, process it first
    if args.input:
        await run_agent(agent, runner, args.input, args.debug, args.computer, args.show)

    try:
        while True:
            query = input("\nEnter your query (or 'exit' to quit): ")
            if query.lower() in ["exit", "quit"]:
                break
            
            await run_agent(agent, runner, query, args.debug, args.computer, args.show)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Cleanup any resources
        if computer:
            await computer.cleanup()

def main():
    try:
        asyncio.run(main_async())
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
