"""
Agent Template - A standardized approach to creating agents with the OpenAI Agents Python SDK

This template demonstrates a comprehensive approach to creating agents with:
1. Standard imports
2. Type definitions for context and output
3. Custom hooks implementation
4. Tool definitions
5. Agent configuration
6. Main runner implementation

Use this as a starting point for new agent development to maintain consistency across
implementations.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agents import (
    Agent,
    AgentHooks,
    GuardrailFunctionOutput,
    ItemHelpers,
    Runner,
    RunContextWrapper,
    Tool,
    function_tool,
    input_guardrail,
    output_guardrail,
    trace,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# SECTION 1: Context and Output Type Definitions
# ------------------------------------------------------------------------------

@dataclass
class AgentContext:
    """
    Context object that will be passed to the agent and is accessible by tools
    and guardrails. This allows you to maintain state between interactions.
    
    Customize this class to include any data your agent needs to maintain
    across its lifecycle.
    """
    session_id: str
    user_info: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_to_history(self, role: str, content: str) -> None:
        """Helper to add an entry to the conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })


class AgentOutput(BaseModel):
    """
    Structured output that the agent will return. Using Pydantic models for
    output provides schema validation and clear documentation of the output
    structure.
    """
    reasoning: str = Field(
        description="The reasoning process used to arrive at the response"
    )
    response: str = Field(
        description="The final response to be presented to the user"
    )
    references: List[str] = Field(
        default_factory=list, 
        description="Any references or sources used in generating the response"
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="A confidence score between 0 and 1 indicating how confident the agent is in the response"
    )

# ------------------------------------------------------------------------------
# SECTION 2: Custom Hooks Implementation
# ------------------------------------------------------------------------------

class StandardAgentHooks(AgentHooks[AgentContext]):
    """
    Hooks for monitoring and responding to agent lifecycle events.
    Implement each hook method to add custom logic at specific points
    in the agent's execution.
    """
    
    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        
    async def on_start(self, context: RunContextWrapper[AgentContext], agent: Agent) -> None:
        """Called when the agent starts processing a request"""
        logger.info(f"Agent '{self.agent_name}' started processing request")
        # You can access and modify the context here
        context.context.add_to_history("system", f"Agent '{agent.name}' started")
        
    async def on_end(self, context: RunContextWrapper[AgentContext], agent: Agent, output: Any) -> None:
        """Called when the agent completes processing a request"""
        logger.info(f"Agent '{self.agent_name}' completed processing with output type: {type(output)}")
        if isinstance(output, AgentOutput):
            logger.info(f"Confidence score: {output.confidence_score}")
        
    async def on_error(self, context: RunContextWrapper[AgentContext], agent: Agent, error: Exception) -> None:
        """Called when an error occurs during processing"""
        logger.error(f"Error in agent '{self.agent_name}': {str(error)}")
        
    async def on_tool_start(self, context: RunContextWrapper[AgentContext], agent: Agent, tool: Tool) -> None:
        """Called when a tool is about to be used"""
        logger.info(f"Agent '{self.agent_name}' is using tool: {tool.name}")
        
    async def on_tool_end(self, context: RunContextWrapper[AgentContext], agent: Agent, tool: Tool, result: str) -> None:
        """Called when a tool completes execution"""
        logger.info(f"Tool '{tool.name}' completed with result length: {len(str(result))}")
        
    async def on_handoff(self, context: RunContextWrapper[AgentContext], agent: Agent, source: Agent) -> None:
        """Called when control is handed off from one agent to another"""
        logger.info(f"Handoff from agent '{source.name}' to '{agent.name}'")

# ------------------------------------------------------------------------------
# SECTION 3: Guardrails Implementation
# ------------------------------------------------------------------------------

@input_guardrail
async def input_validation_guardrail(
    context: RunContextWrapper[AgentContext], 
    agent: Agent[AgentContext], 
    input_text: str
) -> GuardrailFunctionOutput:
    """
    Input guardrail to validate and potentially modify the user input
    before it reaches the agent.
    """
    # Example implementation: Detect and flag potentially sensitive information
    contains_sensitive_info = any(term in input_text.lower() for term in ["password", "ssn", "credit card"])
    
    return GuardrailFunctionOutput(
        modified_input=input_text if not contains_sensitive_info else "I cannot process messages containing sensitive information.",
        output_info={"contains_sensitive_info": contains_sensitive_info},
        tripwire_triggered=contains_sensitive_info
    )


@output_guardrail
async def output_validation_guardrail(
    context: RunContextWrapper[AgentContext], 
    agent: Agent[AgentContext], 
    output: AgentOutput
) -> GuardrailFunctionOutput:
    """
    Output guardrail to validate and potentially modify the agent's output
    before it is returned to the user.
    """
    # Example implementation: Ensure the response doesn't contain certain phrases
    prohibited_phrases = ["I don't know", "I can't help with that"]
    contains_prohibited = any(phrase in output.response for phrase in prohibited_phrases)
    
    # If problems are detected, you can modify the output
    modified_output = output
    if contains_prohibited:
        modified_output.response = "I'd like to provide more specific information. " + output.response
        
    return GuardrailFunctionOutput(
        modified_output=modified_output if contains_prohibited else None,
        output_info={"contains_prohibited_phrases": contains_prohibited},
        tripwire_triggered=False  # Set to True if you want to block the response
    )

# ------------------------------------------------------------------------------
# SECTION 4: Tool Definitions
# ------------------------------------------------------------------------------

@function_tool
async def search_knowledge_base(query: str, context: RunContextWrapper[AgentContext]) -> str:
    """
    Search the knowledge base for information related to the query.
    
    Args:
        query: The search query string
        context: The run context wrapper containing the agent context
        
    Returns:
        A string containing the search results
    """
    # In a real implementation, this would connect to a knowledge base or vector store
    # This is a simplified example
    logger.info(f"Searching knowledge base for: {query}")
    
    # Access context data as needed
    user_info = context.context.user_info
    
    # Simulate a search with a delay
    await asyncio.sleep(1)
    
    return f"Sample results for query '{query}' (User: {user_info.get('name', 'Anonymous')})"


@function_tool
def format_response(
    content: str, 
    style: str = "standard",
    include_references: bool = False
) -> Dict[str, Any]:
    """
    Format the response according to the specified style.
    
    Args:
        content: The content to format
        style: The formatting style (standard, concise, detailed)
        include_references: Whether to include references in the formatted output
        
    Returns:
        A dictionary with the formatted content and metadata
    """
    logger.info(f"Formatting response with style: {style}")
    
    formatted_content = content
    if style == "concise":
        # In a real implementation, this would summarize the content
        formatted_content = f"Concise: {content[:100]}..."
    elif style == "detailed":
        # In a real implementation, this would add more details
        formatted_content = f"Detailed: {content}\n\nAdditional context would be added here."
    
    result = {
        "formatted_content": formatted_content,
        "style": style,
        "character_count": len(formatted_content),
    }
    
    if include_references:
        result["references"] = ["Reference 1", "Reference 2"]
        
    return result

# ------------------------------------------------------------------------------
# SECTION 5: Agent Configuration
# ------------------------------------------------------------------------------

# Define any sub-agents if needed (for handoffs)
sub_agent = Agent(
    name="DetailAgent",
    instructions="""
    You are a specialized agent that provides detailed information on specific topics.
    When you receive a request:
    1. Analyze what specific details are being requested
    2. Provide comprehensive information on that specific topic
    3. Include relevant references or sources
    """,
    tools=[search_knowledge_base],
    output_type=AgentOutput,
    hooks=StandardAgentHooks(agent_name="DetailAgent"),
    handoff_description="A specialist that provides detailed information on specific topics"
)

# Main agent definition
main_agent = Agent(
    name="StandardAgent",
    instructions="""
    You are a helpful assistant that provides informative and accurate responses.
    
    Follow these guidelines:
    1. First, understand what the user is asking for
    2. Use tools when appropriate to gather information
    3. Be concise and clear in your responses
    4. If you don't know something, admit it rather than making up information
    5. For complex or specialized queries, consider handing off to a specialist agent
    
    When structuring your response:
    - Include your reasoning process
    - Provide a clear, direct answer to the user's query
    - Add references where relevant
    - Assign a confidence score to your response
    """,
    tools=[search_knowledge_base, format_response],
    handoffs=[sub_agent],
    input_guardrails=[input_validation_guardrail],
    output_guardrails=[output_validation_guardrail],
    output_type=AgentOutput,
    hooks=StandardAgentHooks(agent_name="StandardAgent"),
    # Optional model configuration
    model="gpt-4-turbo",  # Specify the model to use
    model_settings={
        "temperature": 0.7,
        "top_p": 1.0,
    }
)

# ------------------------------------------------------------------------------
# SECTION 6: Main Execution Logic
# ------------------------------------------------------------------------------

async def run_agent(user_input: str, session_id: str = "default", user_info: Optional[Dict[str, Any]] = None) -> AgentOutput:
    """
    Run the agent with the given input and context.
    
    Args:
        user_input: The user's input message
        session_id: A unique identifier for the session
        user_info: Optional dictionary with user information
        
    Returns:
        The structured output from the agent
    """
    # Create a context object for this run
    context = AgentContext(
        session_id=session_id,
        user_info=user_info or {}
    )
    
    # Add the user input to the conversation history
    context.add_to_history("user", user_input)
    
    # Use a trace to capture the full execution for debugging/monitoring
    with trace(f"agent_run_{session_id}"):
        # Run the agent with the provided input and context
        result = await Runner.run(
            starting_agent=main_agent,
            input=user_input,
            context=context
        )
    
    # Extract the final output
    output = result.final_output
    
    # Log the completion
    logger.info(f"Agent run completed for session {session_id}")
    
    # Add the agent's response to the conversation history
    if isinstance(output, AgentOutput):
        context.add_to_history("assistant", output.response)
    
    return output


async def main():
    """
    Main function to run the agent interactively
    """
    print("Agent Template Demo - Enter 'exit' to quit")
    
    session_id = f"session_{asyncio.get_event_loop().time()}"
    user_info = {
        "name": "Test User",
        "preferences": {
            "response_style": "concise"
        }
    }
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        try:
            output = await run_agent(user_input, session_id, user_info)
            
            print("\nAgent's Reasoning:")
            print(f"{output.reasoning}\n")
            
            print("Agent's Response:")
            print(f"{output.response}\n")
            
            if output.references:
                print("References:")
                for ref in output.references:
                    print(f"- {ref}")
                print()
                
            print(f"Confidence: {output.confidence_score:.2f}")
            
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    """
    Entry point for running the agent from the command line
    """
    asyncio.run(main())

