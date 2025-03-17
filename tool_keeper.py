#!/usr/bin/env python3

"""
Tool Keeper
===========

A CLI and interactive interface for validating, analyzing, and documenting tool definitions.
Combines multiple specialized agents for different aspects of tool management.

Features:
- Analyze tool definitions for best practices
- Validate tool schemas against SDK requirements
- Generate documentation and implementation examples
- Interactive chat interface for tool management
- Streaming support for real-time responses

Example usage:
```python
import asyncio
from tool_keeper import ToolKeeper

async def main():
    # Create the ToolKeeper with default configuration
    tool_keeper = ToolKeeper()
    
    # Use with specific command
    result = await tool_keeper.analyze_tool(tool_json_str)
    print(result)
    
    # Or use the interactive chat interface
    await tool_keeper.chat()

if __name__ == "__main__":
    asyncio.run(main())
```

Command-line usage:
```
python tool_keeper.py [--analyze|--validate|--document] [tool_json_file]
python tool_keeper.py --chat  # For interactive mode
```
"""

import json
import sys
import asyncio
import argparse
from typing import Dict, Any, Optional, List, Union, Callable, AsyncGenerator
from dataclasses import dataclass, field

from agents import (
    Agent,
    Runner,
    ModelSettings,
    MessageOutputItem,
    HandoffCallItem,
    TResponseInputItem,
    trace,
)

from tool_keeper_agents import (
    create_analyzer_agent,
    create_validator_agent,
    create_documenter_agent,
    create_judge_agent,
    tool_schema_guardrail,
    sensitive_data_guardrail,
    offensive_content_guardrail,
)


@dataclass
class ToolKeeperConfig:
    """Configuration for the ToolKeeper interface."""
    analyzer_model: str = "gpt-4o"
    validator_model: str = "gpt-3.5-turbo"  # Validation is simpler, can use faster model
    documenter_model: str = "gpt-4o"
    router_model: str = "gpt-4o"
    temperature: float = 0.2


@dataclass
class ToolKeeperContext:
    """Context for the ToolKeeper session."""
    session_id: str = "default_session"
    tools_analyzed: List[str] = field(default_factory=list)
    last_tool: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = field(default_factory=list)


class ToolKeeperRouter:
    """Router agent that directs requests to specialized agents."""
    
    def __init__(self, config: ToolKeeperConfig = ToolKeeperConfig()):
        """
        Initialize the router agent.
        
        Args:
            config: Configuration for the agent
        """
        self.config = config
        self.agent = Agent(
            name="Tool Keeper Router",
            instructions="""You are a specialized routing agent for tool management.
            
            Your job is to understand user requests related to tool definitions and direct them to the 
            appropriate specialized agent:
            
            1. The Analyzer Agent: For analyzing tools against best practices and suggesting improvements
            2. The Validator Agent: For validating tool schemas against SDK requirements
            3. The Documenter Agent: For generating documentation and implementation examples
            
            Based on the user's request, determine which agent would be most appropriate to handle it.
            If the request doesn't clearly map to one agent, choose the most relevant one.
            
            When a user provides a tool definition, extract and remember it for future use.
            
            Respond in a helpful, clear way and explain which specialized capability you're using to help them.
            """,
            model=config.router_model,
            model_settings=ModelSettings(temperature=config.temperature),
            handoffs=[
                create_analyzer_agent(config.analyzer_model),
                create_validator_agent(config.validator_model),
                create_documenter_agent(config.documenter_model),
            ]
        )


class ToolKeeper:
    """
    Main interface for the Tool Keeper system, combining multiple specialized agents
    for tool validation, analysis, and documentation with advanced guardrails and
    quality control using parallel processing and LLM-as-judge patterns.
    """
    
    def __init__(self, config: ToolKeeperConfig = ToolKeeperConfig()):
        """
        Initialize the ToolKeeper with the specified configuration.
        
        Args:
            config: Configuration for the ToolKeeper system
        """
        self.config = config
        self.context = ToolKeeperContext()
        
        # Initialize specialized agents
        self.analyzer_agent = create_analyzer_agent(config.analyzer_model)
        self.validator_agent = create_validator_agent(config.validator_model)
        self.documenter_agent = create_documenter_agent(config.documenter_model)
        self.judge_agent = create_judge_agent(config.router_model)  # Using router model for judge
        
        # Add guardrails to specialized agents
        self.analyzer_agent.input_guardrails = [tool_schema_guardrail]
        self.validator_agent.input_guardrails = [tool_schema_guardrail]
        self.documenter_agent.output_guardrails = [sensitive_data_guardrail, offensive_content_guardrail]
        
        # Initialize router agent
        self.router = ToolKeeperRouter(config)
    
    async def analyze_tool(self, tool_definition: str) -> str:
        """
        Analyze a tool definition for best practices.
        
        Args:
            tool_definition: The tool definition to analyze as a JSON string
            
        Returns:
            Analysis results as a string
        """
        try:
            # Parse to validate JSON and update context
            tool_dict = json.loads(tool_definition)
            self.context.last_tool = tool_dict
            tool_name = tool_dict.get("name", "Unnamed Tool")
            if tool_name not in self.context.tools_analyzed:
                self.context.tools_analyzed.append(tool_name)
                
            # Run the analyzer agent
            prompt = f"Analyze this tool definition and provide detailed feedback: {tool_definition}"
            result = await Runner.run(self.analyzer_agent, prompt, context=self.context)
            return result.final_output
            
        except json.JSONDecodeError:
            return "Error: The tool definition is not valid JSON. Please check the format and try again."
        except Exception as e:
            return f"Error analyzing tool: {str(e)}"
    
    async def validate_tool(self, tool_definition: str) -> str:
        """
        Validate a tool definition against SDK requirements.
        
        Args:
            tool_definition: The tool definition to validate as a JSON string
            
        Returns:
            Validation results as a string
        """
        try:
            # Parse to validate JSON and update context
            tool_dict = json.loads(tool_definition)
            self.context.last_tool = tool_dict
            
            # Run the validator agent
            prompt = f"Validate this tool definition against SDK requirements: {tool_definition}"
            result = await Runner.run(self.validator_agent, prompt, context=self.context)
            return result.final_output
            
        except json.JSONDecodeError:
            return "Error: The tool definition is not valid JSON. Please check the format and try again."
        except Exception as e:
            return f"Error validating tool: {str(e)}"
    
    async def document_tool(self, tool_definition: str) -> str:
        """
        Generate documentation for a tool definition.
        
        Args:
            tool_definition: The tool definition to document as a JSON string
            
        Returns:
            Documentation as a string
        """
        try:
            # Parse to validate JSON and update context
            tool_dict = json.loads(tool_definition)
            self.context.last_tool = tool_dict
            
            # Run the documenter agent
            prompt = f"Generate documentation for this tool definition: {tool_definition}"
            result = await Runner.run(self.documenter_agent, prompt, context=self.context)
            return result.final_output
            
        except json.JSONDecodeError:
            return "Error: The tool definition is not valid JSON. Please check the format and try again."
        except Exception as e:
            return f"Error documenting tool: {str(e)}"
    
    async def generate_implementation(self, tool_definition: str) -> str:
        """
        Generate a Python implementation for a tool definition.
        
        Args:
            tool_definition: The tool definition to implement as a JSON string
            
        Returns:
            Python implementation as a string
        """
        try:
            # Parse to validate JSON and update context
            tool_dict = json.loads(tool_definition)
            self.context.last_tool = tool_dict
            
            # Run the documenter agent specifically for implementation
            prompt = f"Generate a complete Python implementation for this tool definition: {tool_definition}"
            result = await Runner.run(self.documenter_agent, prompt, context=self.context)
            return result.final_output
            
        except json.JSONDecodeError:
            return "Error: The tool definition is not valid JSON. Please check the format and try again."
        except Exception as e:
            return f"Error generating implementation: {str(e)}"
            
    async def evaluate_tool(self, tool_definition: str) -> str:
        """
        Evaluate a tool definition for quality using the judge agent.
        
        Args:
            tool_definition: The tool definition to evaluate as a JSON string
            
        Returns:
            Evaluation results as a string
        """
        try:
            # Parse to validate JSON and update context
            tool_dict = json.loads(tool_definition)
            self.context.last_tool = tool_dict
            
            # Run the judge agent to evaluate the tool
            prompt = f"Evaluate this tool definition: {tool_definition}"
            result = await Runner.run(self.judge_agent, prompt, context=self.context)
            return result.final_output
            
        except json.JSONDecodeError:
            return "Error: The tool definition is not valid JSON. Please check the format and try again."
        except Exception as e:
            return f"Error evaluating tool: {str(e)}"
    
    async def process_tool_comprehensive(self, tool_definition: str) -> Dict[str, str]:
        """
        Process a tool definition comprehensively with parallel execution of multiple agents.
        Uses the parallelization pattern for improved performance.
        
        Args:
            tool_definition: The tool definition to process as a JSON string
            
        Returns:
            Dictionary containing results from all agents
        """
        try:
            # Parse to validate JSON and update context
            tool_dict = json.loads(tool_definition)
            self.context.last_tool = tool_dict
            
            # Run all agents in parallel with trace for better observability
            with trace("Comprehensive Tool Processing"):
                validation_result, analysis_result, documentation_result, implementation_result, evaluation_result = await asyncio.gather(
                    Runner.run(self.validator_agent, f"Validate this tool definition: {tool_definition}", context=self.context),
                    Runner.run(self.analyzer_agent, f"Analyze this tool definition: {tool_definition}", context=self.context),
                    Runner.run(self.documenter_agent, f"Generate documentation for this tool definition: {tool_definition}", context=self.context),
                    Runner.run(self.documenter_agent, f"Generate a complete Python implementation for this tool definition: {tool_definition}", context=self.context),
                    Runner.run(self.judge_agent, f"Evaluate this tool definition: {tool_definition}", context=self.context)
                )
            
            # Return all results in a dictionary
            return {
                "validation": validation_result.final_output,
                "analysis": analysis_result.final_output,
                "documentation": documentation_result.final_output,
                "implementation": implementation_result.final_output,
                "evaluation": evaluation_result.final_output
            }
            
        except json.JSONDecodeError:
            return {
                "error": "The tool definition is not valid JSON. Please check the format and try again."
            }
        except Exception as e:
            return {
                "error": f"Error processing tool: {str(e)}"
            }
    
    async def evaluate_implementation(self, implementation: str, tool_definition: str) -> str:
        """
        Evaluate a tool implementation against its definition.
        
        Args:
            implementation: The Python implementation code to evaluate
            tool_definition: The original tool definition as a JSON string
            
        Returns:
            Evaluation results as a string
        """
        try:
            # Parse to validate JSON
            tool_dict = json.loads(tool_definition)
            
            # Run the judge agent to evaluate the implementation
            prompt = (
                f"Evaluate this implementation against its tool definition.\n\n"
                f"Tool Definition:\n{tool_definition}\n\n"
                f"Implementation:\n{implementation}"
            )
            result = await Runner.run(self.judge_agent, prompt, context=self.context)
            return result.final_output
            
        except json.JSONDecodeError:
            return "Error: The tool definition is not valid JSON. Please check the format and try again."
        except Exception as e:
            return f"Error evaluating implementation: {str(e)}"
    
    async def process_request(self, query: str) -> str:
        """
        Process a user request using the router agent.
        
        Args:
            query: The user's query
            
        Returns:
            The agent's response
        """
        # Add to history
        self.context.history.append({"role": "user", "content": query})
        
        # Process with router agent which will handoff to specialists
        result = await Runner.run(self.router.agent, query, context=self.context)
        
        # Add to history
        response = result.final_output
        self.context.history.append({"role": "assistant", "content": response})
        
        return response
    
    async def process_request_stream(self, query: str) -> AsyncGenerator[str, None]:
        """
        Process a user request using the router agent with streaming.
        
        Args:
            query: The user's query
            
        Yields:
            Chunks of the agent's response as they're generated
        """
        # Add to history
        self.context.history.append({"role": "user", "content": query})
        
        # Get streaming response
        response_chunks = []
        
        from openai.types.responses import ResponseTextDeltaEvent
        
        result = Runner.run_streamed(self.router.agent, query, context=self.context)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                chunk = event.data.delta or ""
                response_chunks.append(chunk)
                yield chunk
            elif event.type == "run_item_stream_event" and event.name == "handoff_requested":
                handoff_name = event.item.raw_item.function.name
                yield f"\n[Handing off to specialized agent: {handoff_name}]\n"
        
        # Add complete response to history
        full_response = "".join(response_chunks)
        self.context.history.append({"role": "assistant", "content": full_response})
    
    async def chat(self, initial_message: str = "Hello! I'm the Tool Keeper assistant. I can help you analyze, validate, and document tool definitions. What would you like help with today?") -> None:
        """
        Start an interactive chat session with streaming responses.
        
        Args:
            initial_message: The initial greeting message
        """
        print(f"\n{initial_message}\n")
        
        while True:
            try:
                # Get user input
                print("\nYou: ", end="", flush=True)
                user_input = input()
                
                if user_input.lower() in ("exit", "quit", "bye"):
                    print("\nTool Keeper: Goodbye! Feel free to return if you need more help with tool management.")
                    break
                
                # Process and stream response
                print("\nTool Keeper: ", end="", flush=True)
                async for chunk in self.process_request_stream(user_input):
                    print(chunk, end="", flush=True)
                print()
                
            except KeyboardInterrupt:
                print("\n\nTool Keeper: Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")


async def main():
    """Command line interface for ToolKeeper."""
    parser = argparse.ArgumentParser(description="Tool Keeper - Tool Management Assistant")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--analyze", help="Analyze a tool definition file", type=str)
    group.add_argument("--validate", help="Validate a tool definition file", type=str)
    group.add_argument("--document", help="Generate documentation for a tool definition file", type=str)
    group.add_argument("--implement", help="Generate implementation for a tool definition file", type=str)
    group.add_argument("--evaluate", help="Evaluate quality of a tool definition file", type=str)
    group.add_argument("--evaluate-impl", nargs=2, 
                      metavar=('IMPL_FILE', 'DEF_FILE'), 
                      help="Evaluate implementation file against definition file")
    group.add_argument("--comprehensive", help="Process tool definition comprehensively with all agents in parallel", type=str)
    group.add_argument("--chat", help="Start interactive chat mode", action="store_true")
    
    args = parser.parse_args()
    
    # Create the Tool Keeper
    tool_keeper = ToolKeeper()
    
    if args.analyze:
        try:
            with open(args.analyze, 'r') as f:
                tool_json = f.read()
            result = await tool_keeper.analyze_tool(tool_json)
            print(result)
        except FileNotFoundError:
            print(f"Error: File {args.analyze} not found.")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    elif args.validate:
        try:
            with open(args.validate, 'r') as f:
                tool_json = f.read()
            result = await tool_keeper.validate_tool(tool_json)
            print(result)
        except FileNotFoundError:
            print(f"Error: File {args.validate} not found.")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    elif args.document:
        try:
            with open(args.document, 'r') as f:
                tool_json = f.read()
            result = await tool_keeper.document_tool(tool_json)
            print(result)
        except FileNotFoundError:
            print(f"Error: File {args.document} not found.")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    elif args.implement:
        try:
            with open(args.implement, 'r') as f:
                tool_json = f.read()
            result = await tool_keeper.generate_implementation(tool_json)
            print(result)
        except FileNotFoundError:
            print(f"Error: File {args.implement} not found.")
        except Exception as e:
            print(f"Error: {str(e)}")
            
    elif args.evaluate:
        try:
            with open(args.evaluate, 'r') as f:
                tool_json = f.read()
            result = await tool_keeper.evaluate_tool(tool_json)
            print(result)
        except FileNotFoundError:
            print(f"Error: File {args.evaluate} not found.")
        except Exception as e:
            print(f"Error: {str(e)}")
            
    elif args.evaluate_impl:
        try:
            impl_file, def_file = args.evaluate_impl
            with open(impl_file, 'r') as f:
                implementation = f.read()
            with open(def_file, 'r') as f:
                tool_json = f.read()
            result = await tool_keeper.evaluate_implementation(implementation, tool_json)
            print(result)
        except FileNotFoundError as e:
            print(f"Error: File not found - {str(e)}")
        except Exception as e:
            print(f"Error: {str(e)}")
            
    elif args.comprehensive:
        try:
            with open(args.comprehensive, 'r') as f:
                tool_json = f.read()
            
            print("\n=== Running Comprehensive Tool Processing ===\n")
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
        except FileNotFoundError:
            print(f"Error: File {args.comprehensive} not found.")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    elif args.chat:
        await tool_keeper.chat()
    
    else:
        # No arguments, start chat mode
        await tool_keeper.chat()


# Example tool definition for testing
EXAMPLE_TOOL = {
    "name": "fetch_weather",
    "description": "Fetch weather information for a location",
    "parameters": {
        "location": {
            "type": "string",
            "description": "The location to get weather for (city, address, etc.)"
        },
        "units": {
            "type": "string",
            "description": "Temperature units (metric/imperial)",
            "required": False
        }
    }
}


if __name__ == "__main__":
    asyncio.run(main())