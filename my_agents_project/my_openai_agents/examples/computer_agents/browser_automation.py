"""Browser automation example using the computer agent."""
import asyncio
import os
from pydantic import BaseModel
from typing import Optional

from my_openai_agents.core.agent import Agent
from my_openai_agents.core.run import Runner
from my_openai_agents.core.models.openai_model import OpenAIModel
from my_openai_agents.core.computers.local_playwright import LocalPlaywright
from my_openai_agents.core.computers.tools import create_computer_tools

class BrowserResult(BaseModel):
    """Result of a browser automation task."""
    success: bool
    message: str
    data: Optional[str] = None

async def main():
    # Initialize the browser
    computer = LocalPlaywright(start_url="https://www.bing.com")
    await computer.setup()

    try:
        # Create the model
        model = OpenAIModel(api_key=os.getenv('OPENAI_API_KEY', 'test_key'))

        # Create tools for the agent
        tools = create_computer_tools(computer)

        # Create the browser automation agent
        agent = Agent(
            name="browser_automation",
            instructions="""You are a browser automation agent that can control a web browser.
            You can:
            - Navigate to URLs
            - Click on elements
            - Type text
            - Take screenshots
            
            Always explain what you're doing and handle errors appropriately.""",
            tools=tools,
            output_type=BrowserResult,
            model=model
        )

        # Example task
        task = "Search for 'OpenAI news' on Bing"
        print(f"\nTask: {task}")
        try:
            result = await Runner.run(agent, task)
            print(f"Result: {result.final_output}")
        except Exception as e:
            print(f"Error: {str(e)}")

    finally:
        # Clean up
        await computer.cleanup()

def run():
    """Run the browser automation example."""
    asyncio.run(main())

if __name__ == "__main__":
    run()
