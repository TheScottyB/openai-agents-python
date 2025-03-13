#!/usr/bin/env python3
import os
import sys
from agents import Agent

if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable is not set")
    sys.exit(1)

# Create a workspace-aware agent
agent = Agent(
    name="Workspace Caretaker",
    instructions="""You are a workspace management assistant. 
    Help users organize, maintain, and navigate their workspace.
    Current directory: {}
    """.format(os.getcwd())
)

print("Workspace Caretaker initialized. Use Ctrl+C to exit.")
try:
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() in ['exit', 'quit']:
            break
        response = agent.invoke(query)
        print("\nAgent:", response)
except KeyboardInterrupt:
    print("\nExiting...")

#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Optional, List
import asyncio

# Basic error handling function
def handle_error(error_message: str, exit_code: int = 1) -> None:
    """Print error message to stderr and exit with the given code."""
    print(f"Error: {error_message}", file=sys.stderr)
    sys.exit(exit_code)

def setup_environment() -> None:
    """Ensure the required environment variables are set."""
    if not os.environ.get("OPENAI_API_KEY"):
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            handle_error("OPENAI_API_KEY environment variable not set. Please set it before running this script.")

def get_workspace_info() -> dict:
    """Get basic information about the current workspace."""
    try:
        current_path = os.getcwd()
        files = os.listdir(current_path)
        return {
            "current_directory": current_path,
            "files_count": len(files),
            "directories": [f for f in files if os.path.isdir(os.path.join(current_path, f))],
            "python_files": [f for f in files if f.endswith('.py')],
        }
    except Exception as e:
        handle_error(f"Failed to get workspace info: {str(e)}")
        return {}  # This will not be reached due to handle_error exiting

def main() -> None:
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(description="Workspace assistant using OpenAI Agent")
    parser.add_argument("--computer", choices=["local-playwright"], 
                        help="Specify computer interface to use (currently only local-playwright is supported)")
    
    args = parser.parse_args()
    
    # Check for required environment variables
    setup_environment()
    
    try:
        from agents import Agent, Runner
        from agents.tools import FileSearchTool, FileReadTool, ShellTool
        from agents.schema import ModelConfig, RunnerOptions
    except ImportError:
        handle_error("Failed to import required classes from the 'agents' package. Please ensure it's installed correctly.")
    
    # Get workspace information to provide context for the agent
    workspace_info = get_workspace_info()
    
    # Construct enhanced instructions with workspace awareness
    instructions = (
        "You are a workspace management assistant. "
        "Help users organize, maintain, and navigate their workspace. "
        f"Current directory: {workspace_info['current_directory']}. "
        f"Contains {workspace_info['files_count']} files, "
        f"including {len(workspace_info['directories'])} directories "
        f"and {len(workspace_info['python_files'])} Python files."
    )
    
    try:
        # Configure workspace management tools
        tools = [
            FileSearchTool(),
            FileReadTool(),
            ShellTool(allowed_commands=["ls", "pwd", "find", "cat", "grep"])
        ]
        
        # Set up the OpenAI configuration
        model_config = ModelConfig(
            model="gpt-4-turbo-preview"
        )
        
        # Create the agent with workspace management instructions
        agent = Agent(
            name="Workspace Caretaker",
            instructions=instructions,
            model_config=model_config,
            tools=tools
        )
        
        # Set up the Runner to execute the agent
        runner = Runner(
            agent=agent,
            options=RunnerOptions(
                stream=True
            )
        )
        
        if args.computer == "local-playwright":
            print("Starting agent with local-playwright computer interface...")
            print("Agent initialized with workspace awareness.")
            print("Use Ctrl+C to stop the agent.")
            
            # Simple interaction loop
            try:
                while True:
                    user_input = input("\nEnter query (or 'exit' to quit): ")
                    if user_input.lower() in ['exit', 'quit']:
                        break
                    
                    print("\nProcessing your request...")
                    # Use the Runner class to execute the agent
                    response = asyncio.run(runner.run(user_input))
                    print("\nAgent response:")
                    print(response.content)
                    
            except KeyboardInterrupt:
                print("\nAgent terminated by user.")
                
    except Exception as e:
        handle_error(f"Error initializing or running agent: {str(e)}")

if __name__ == "__main__":
    main()

