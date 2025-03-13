"""Interactive browser automation example."""
import asyncio
import os
from pydantic import BaseModel
from typing import Optional, Dict, Any, Tuple
import re

from my_openai_agents.core.agent import Agent
from my_openai_agents.core.run import Runner
from my_openai_agents.core.models.openai_model import OpenAIModel
from my_openai_agents.core.computers import LocalPlaywright, create_computer_tools

class BrowserResult(BaseModel):
    """Result of a browser automation task."""
    success: bool
    message: str
    data: Optional[str] = None

def parse_command(command: str) -> Tuple[str, dict]:
    """Parse command into action and parameters."""
    command = command.strip().lower()
    
    # Regular expressions for command parsing
    goto_pattern = r'^(?:go\s+to\s+|goto\s+)((?:https?://)?[\w.-]+(?:\.[\w.-]+)+[\w\-._~:/?#[\]@!$&\'()*+,;=]*)'
    search_pattern = r'^(?:search\s+for\s+|search\s+|find\s+)(.+)$'
    click_pattern = r'^click\s+(?:at\s+)?(\d+)\s+(\d+)$'
    type_pattern = r'^type\s+(.+)$'
    screenshot_pattern = r'^(?:take\s+)?screenshot$'
    
    if match := re.match(goto_pattern, command):
        return 'goto', {'url': match.group(1)}
    elif match := re.match(search_pattern, command):
        return 'search', {'query': match.group(1)}
    elif match := re.match(click_pattern, command):
        return 'click', {'x': int(match.group(1)), 'y': int(match.group(2))}
    elif match := re.match(type_pattern, command):
        return 'type', {'text': match.group(1)}
    elif re.match(screenshot_pattern, command):
        return 'screenshot', {}
    else:
        return 'unknown', {}

async def create_browser_agent() -> tuple[Agent, LocalPlaywright]:
    """Create and initialize the browser agent."""
    # Initialize the browser
    computer = LocalPlaywright(start_url="https://www.google.com")
    await computer.setup()

    # Create tools for the agent
    tools = create_computer_tools(computer)

    # Create the browser automation agent
    agent = Agent(
        name="browser_automation",
        instructions="""You are a browser automation agent that can control a web browser.
        You can:
        - Navigate to URLs (goto <url>)
        - Search for content (search <query>)
        - Click at coordinates (click <x> <y>)
        - Type text (type <text>)
        - Take screenshots (screenshot)
        
        Always explain what you're doing and handle errors appropriately.
        For URLs, ensure they have a proper scheme (http:// or https://).""",
        tools=tools,
        output_type=BrowserResult,
        model=OpenAIModel(api_key=os.getenv('OPENAI_API_KEY', 'test_key'))
    )

    return agent, computer

async def handle_command(agent: Agent, command: str) -> None:
    """Handle a single command."""
    try:
        action, params = parse_command(command)
        
        if action == 'unknown':
            print("Unknown command. Available commands:")
            print("- goto <url>")
            print("- search [for] <query>")
            print("- click [at] <x> <y>")
            print("- type <text>")
            print("- [take] screenshot")
            return

        tool_map = {
            'goto': (agent.tools[0], {'url': params.get('url', '')}),
            'search': (agent.tools[4], {'query': params.get('query', '')}),
            'click': (agent.tools[1], {'x': params.get('x', 0), 'y': params.get('y', 0)}),
            'type': (agent.tools[2], {'text': params.get('text', '')}),
            'screenshot': (agent.tools[3], {})
        }

        tool, tool_params = tool_map[action]
        result = await tool.execute(**tool_params)
        print(f"\nResult: {result}")
        
    except Exception as e:
        print(f"\nError executing command: {str(e)}")

async def interactive_browser():
    """Run an interactive browser automation session."""
    print("\nInitializing browser automation agent...")
    agent = computer = None

    try:
        agent, computer = await create_browser_agent()
        
        print("\nBrowser Agent Ready!")
        print("Available commands:")
        print("- goto <url>")
        print("- search [for] <query>")
        print("- click [at] <x> <y>")
        print("- type <text>")
        print("- [take] screenshot")
        print("Type 'quit' to exit\n")

        while True:
            try:
                command = input("Enter command: ").strip()
                if command.lower() == 'quit':
                    break
                if command:
                    await handle_command(agent, command)
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit properly")
            except Exception as e:
                print(f"\nError: {str(e)}")

    finally:
        if computer:
            print("\nCleaning up...")
            try:
                await asyncio.wait_for(computer.cleanup(), timeout=5.0)
            except asyncio.TimeoutError:
                print("Cleanup timed out, but continuing...")
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")

def run():
    """Run the interactive browser example."""
    try:
        asyncio.run(interactive_browser())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")

if __name__ == "__main__":
    run()
