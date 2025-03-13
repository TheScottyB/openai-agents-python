#!/usr/bin/env python3
"""
Core Concepts Demo
=================

This file demonstrates the key concepts of the OpenAI Agents SDK:
- Agents: LLMs configured with instructions, tools, guardrails, and handoffs
- Handoffs: Transferring control between specialized agents
- Guardrails: Safety checks for input and output validation
- Tracing: Tracking agent runs for debugging and optimization

The example implements a workflow with a Coordinator agent that delegates to specialized
agents (Researcher, Analyst, and Writer) to complete a comprehensive research task.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Annotated
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from agents import Agent, AgentResponse
from agents.hooks import AgentHooks
from agents.guardrails import InputGuardrail, OutputGuardrail, GuardrailException
from agents.schema import AgentOutputParser, LazyStr
from agents.tools import BaseTool, ToolOutput

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================================
# SECTION 1: Context and Output Types
# =====================================================================

@dataclass
class AgentContext:
    """Context object to maintain state across agent calls."""
    user_query: str
    research_results: Dict[str, Any] = field(default_factory=dict)
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    final_content: str = ""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    def add_to_history(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})

class ResearchOutput(BaseModel):
    """Structured output for the Research agent."""
    sources: List[Dict[str, str]] = Field(..., description="List of sources with URLs and titles")
    key_facts: List[str] = Field(..., description="Key facts discovered in the research")
    summary: str = Field(..., description="Brief summary of the research findings")

class AnalysisOutput(BaseModel):
    """Structured output for the Analysis agent."""
    insights: List[str] = Field(..., description="Key insights from analyzing the research")
    recommendations: List[str] = Field(..., description="Actionable recommendations based on analysis")
    limitations: List[str] = Field(..., description="Limitations or gaps in the current research")

class WriterOutput(BaseModel):
    """Structured output for the Writer agent."""
    title: str = Field(..., description="Title for the content")
    content: str = Field(..., description="The fully written content")
    audience_suitability: str = Field(..., description="Assessment of how well the content suits the target audience")

class CoordinatorOutput(BaseModel):
    """Structured output for the Coordinator agent."""
    next_step: str = Field(..., description="The next step in the workflow")
    reasoning: str = Field(..., description="Reasoning behind the decision")
    completed_steps: List[str] = Field(default_factory=list, description="Steps completed so far")
    final_output: Optional[str] = Field(None, description="Final output when the workflow is complete")

# =====================================================================
# SECTION 2: Custom Hooks for Tracing and Logging
# =====================================================================

class DemoAgentHooks(AgentHooks):
    """Custom hooks to trace and log agent activities."""
    
    def on_agent_run_start(self, agent_name: str, instructions: str, model: str) -> None:
        """Called when an agent run starts."""
        logger.info(f"Starting agent: {agent_name} using model: {model}")
        
    def on_agent_run_end(self, agent_name: str, response: Any) -> None:
        """Called when an agent run completes."""
        logger.info(f"Agent {agent_name} completed run")
        
    def on_tool_call_start(self, agent_name: str, tool_name: str, tool_input: Any) -> None:
        """Called when a tool call starts."""
        logger.info(f"Agent {agent_name} calling tool: {tool_name} with input: {tool_input}")
        
    def on_tool_call_end(self, agent_name: str, tool_name: str, output: Any) -> None:
        """Called when a tool call completes."""
        logger.info(f"Tool {tool_name} returned output")
        
    def on_agent_message(self, agent_name: str, message: str) -> None:
        """Called when an agent produces a message."""
        logger.info(f"Agent {agent_name} message: {message[:100]}...")
        
    def on_handoff_start(self, from_agent: str, to_agent: str, reason: str) -> None:
        """Called when a handoff between agents starts."""
        logger.info(f"Handoff from {from_agent} to {to_agent}. Reason: {reason}")
        
    def on_handoff_end(self, from_agent: str, to_agent: str, result: Any) -> None:
        """Called when a handoff between agents completes."""
        logger.info(f"Handoff from {from_agent} to {to_agent} completed")

# =====================================================================
# SECTION 3: Guardrails
# =====================================================================

class SensitiveInfoGuardrail(InputGuardrail):
    """Guardrail that blocks requests containing sensitive information."""
    
    def __init__(self, sensitive_terms: List[str]):
        self.sensitive_terms = sensitive_terms
        
    def run(self, input_data: str) -> str:
        """Check if input contains sensitive information."""
        for term in self.sensitive_terms:
            if term.lower() in input_data.lower():
                raise GuardrailException(f"Input contains sensitive term: {term}")
        return input_data

class ContentQualityGuardrail(OutputGuardrail):
    """Guardrail that ensures output content meets quality standards."""
    
    def __init__(self, min_length: int = 50):
        self.min_length = min_length
        
    def run(self, output_data: str) -> str:
        """Check if output meets quality standards."""
        if len(output_data) < self.min_length:
            raise GuardrailException(f"Output is too short: {len(output_data)} chars, minimum: {self.min_length}")
        return output_data

# =====================================================================
# SECTION 4: Tool Definitions
# =====================================================================

class SearchTool(BaseTool):
    """Tool for performing web searches."""
    
    name: str = "search"
    description: str = "Search the web for information on a topic"
    
    async def _run(self, query: Annotated[str, "The search query"]) -> ToolOutput:
        """Run a web search (simulated)."""
        # In a real implementation, this would call a search API
        logger.info(f"Searching for: {query}")
        
        # Simulate search results
        results = [
            {"title": f"Result 1 for {query}", "url": f"https://example.com/1?q={query}", "snippet": f"This is a sample result about {query}..."},
            {"title": f"Result 2 for {query}", "url": f"https://example.com/2?q={query}", "snippet": f"Another informative page about {query}..."},
            {"title": f"Result 3 for {query}", "url": f"https://example.com/3?q={query}", "snippet": f"More detailed information regarding {query}..."}
        ]
        
        return ToolOutput(output=json.dumps(results))

class RetrieveDocumentTool(BaseTool):
    """Tool for retrieving content from a document or URL."""
    
    name: str = "retrieve_document"
    description: str = "Retrieve the content of a specific document or URL"
    
    async def _run(self, url: Annotated[str, "The URL or document identifier to retrieve"]) -> ToolOutput:
        """Retrieve document content (simulated)."""
        # In a real implementation, this would fetch actual content
        logger.info(f"Retrieving content from: {url}")
        
        # Simulate document content
        content = f"This is the content of the document at {url}. It contains information about the requested topic."
        content += " The document discusses key concepts, methodologies, and findings related to the subject matter."
        content += " Various experts have contributed to this field with research spanning several decades."
        
        return ToolOutput(output=content)

class AnalyzeDataTool(BaseTool):
    """Tool for analyzing numerical or structured data."""
    
    name: str = "analyze_data"
    description: str = "Analyze numerical or structured data to extract insights"
    
    async def _run(self, 
                  data: Annotated[str, "JSON string containing the data to analyze"],
                  analysis_type: Annotated[str, "Type of analysis to perform (e.g., statistical, trend, comparison)"]) -> ToolOutput:
        """Analyze data (simulated)."""
        # In a real implementation, this would perform actual data analysis
        logger.info(f"Analyzing data with method: {analysis_type}")
        
        # Parse the data
        try:
            parsed_data = json.loads(data)
        except json.JSONDecodeError:
            return ToolOutput(output="Error: Invalid JSON data provided")
        
        # Simulate analysis results
        analysis_result = {
            "analysis_type": analysis_type,
            "data_points_analyzed": len(parsed_data) if isinstance(parsed_data, list) else 1,
            "insights": [
                "The data shows a clear trend in the specified domain",
                "Several outliers were identified that may warrant further investigation",
                "The correlation between key variables suggests a strong relationship"
            ],
            "confidence_score": 0.85
        }
        
        return ToolOutput(output=json.dumps(analysis_result))

class SaveContentTool(BaseTool):
    """Tool for saving generated content to a specified format or location."""
    
    name: str = "save_content"
    description: str = "Save generated content to a specified format or location"
    
    async def _run(self, 
                  content: Annotated[str, "The content to save"],
                  format: Annotated[str, "Format to save in (e.g., markdown, pdf, txt)"],
                  title: Annotated[str, "Title for the content"]) -> ToolOutput:
        """Save content to a file (simulated)."""
        # In a real implementation, this would save to actual files or databases
        logger.info(f"Saving content with title '{title}' in {format} format")
        
        # Simulate saving
        filename = f"{title.replace(' ', '_').lower()}.{format}"
        
        # In a real implementation, you would actually write to a file
        # with open(filename, 'w') as f:
        #     f.write(content)
        
        return ToolOutput(output=f"Content saved successfully to {filename}")

# =====================================================================
# SECTION 5: Agent Implementations
# =====================================================================

# Coordinator Agent - Orchestrates the overall workflow
coordinator_instructions = """
You are a Workflow Coordinator agent responsible for managing a research and content creation process.
Your job is to:
1. Understand the user's request
2. Delegate research tasks to the Research Agent
3. Send research results to the Analysis Agent
4. Forward analysis to the Content Writer Agent
5. Present the final output to the user

For each interaction, determine the next appropriate step in the workflow and provide clear reasoning.
Once the entire workflow is complete, present the final content to the user with a summary of the process.

IMPORTANT: Do not attempt to perform research, analysis, or content writing yourself.
Instead, use the appropriate handoff to delegate these tasks to the specialized agents.
"""
You are a Research Agent specialized in gathering comprehensive information on any topic.
Your job is to:
1. Search for relevant information using the search tool
2. Retrieve detailed content from the most promising sources
3. Organize the information into a structured format
4. Provide source attribution for all information

Focus on gathering factual, accurate, and comprehensive information.
Be thorough but prioritize quality and relevance over quantity.
Always provide proper citation to sources.

Return your findings in a structured format with:
- List of sources with URLs and titles
- Key facts discovered in the research
- A brief summary of the research findings
"""

# Analysis Agent - Specializes in analyzing information
analyst_instructions = """
You are an Analysis Agent specialized in extracting insights from research data.
Your job is to:
1. Carefully review the research provided
2. Identify key patterns, trends, and relationships
3. Provide meaningful insights that go beyond the obvious
4. Suggest limitations of the current research
5. Recommend potential actions based on the analysis

Focus on depth rather than breadth in your analysis.
Consider multiple perspectives and potential implications.
Be honest about limitations and uncertainties in the data.

Return your analysis in a structured format with:
- Key insights from analyzing the research
- Actionable recommendations based on analysis
- Limitations or gaps in the current research
"""

# Content Writer Agent - Specializes in creating high-quality content
writer_instructions = """
You are a Content Writer Agent specialized in creating high-quality, engaging content.
Your job is to:
1. Review the research and analysis provided
2. Organize the information into a coherent structure
3. Write clear, engaging, and accessible content
4. Tailor the content to be appropriate for the intended audience
5. Ensure the content is factually accurate based on the research

Focus on clarity, engagement, and accuracy in your writing.
Use an appropriate tone and style for the intended audience.
Incorporate the key insights from the analysis.

Return your content in a structured format with:
- A compelling title
- The fully written content
- An assessment of how well the content suits the target audience
"""

# Create the agents with appropriate configuration
coordinator_agent = Agent(
    name="Coordinator",
    instructions=coordinator_instructions,
    model="gpt-4",
    tools=[],  # Coordinator doesn't need tools, just orchestrates
    output_parser=AgentOutputParser(pydantic_object=CoordinatorOutput),
    hooks=DemoAgentHooks(),
    input_guardrails=[SensitiveInfoGuardrail(sensitive_terms=["password", "social security", "credit card"])],
)

researcher_agent = Agent(
    name="Researcher",
    instructions=researcher_instructions,
    model="gpt-4",
    tools=[SearchTool(), RetrieveDocumentTool()],
    output_parser=AgentOutputParser(pydantic_object=ResearchOutput),
    hooks=DemoAgentHooks(),
)

analyst_agent = Agent(
    name="Analyst",
    instructions=analyst_instructions,
    model="gpt-4",
    tools=[AnalyzeDataTool()],
    output_parser=AgentOutputParser(pydantic_object=AnalysisOutput),
    hooks=DemoAgentHooks(),
)

writer_agent = Agent(
    name="Writer",
    instructions=writer_instructions,
    model="gpt-4",
    tools=[SaveContentTool()],
    output_parser=AgentOutputParser(pydantic_object=WriterOutput),
    hooks=DemoAgentHooks(),
    output_guardrails=[ContentQualityGuardrail(min_length=200)],
)

# Configure agent handoffs
coordinator_agent.handoffs = [researcher_agent, analyst_agent, writer_agent]

# =====================================================================
# SECTION 6: Main Execution Logic
# =====================================================================

async def run_research_workflow(query: str) -> str:
    """
    Run the complete research workflow with the given query.
    
    This function demonstrates the full workflow:
    1. Coordinator understands the query and plans the process
    2. Research Agent gathers information
    3. Analysis Agent extracts insights
    4. Writer Agent creates the final content
    5. Coordinator returns the complete result to the user
    
    Args:
        query: The research query from the user
        
    Returns:
        The final content generated by the workflow
    """
    logger.info(f"Starting research workflow with query: {query}")
    
    # Initialize the context with the user query
    context = AgentContext(user_query=query)
    context.add_to_history("user", query)
    
    # Start with the coordinator to plan the workflow
    response = await coordinator_agent.run(query, context=context)
    
    if not isinstance(response, CoordinatorOutput):
        raise ValueError(f"Expected CoordinatorOutput but got {type(response)}")
    
    # The coordinator will handle the workflow through handoffs
    # We'll interact with it until the workflow is complete
    while response.next_step != "complete" and not response.final_output:
        # The coordinator should have already made handoffs to the specialized agents
        # We just need to provide any additional input to continue the workflow
        follow_up = f"Continue with the next step: {response.next_step}"
        response = await coordinator_agent.run(follow_up, context=context)
        
        if not isinstance(response, CoordinatorOutput):
            raise ValueError(f"Expected CoordinatorOutput but got {type(response)}")
            
        logger.info(f"Workflow progress - Next step: {response.next_step}")
        logger.info(f"Completed steps: {', '.join(response.completed_steps)}")
        
    # Return the final output from the workflow
    return response.final_output or "Workflow completed but no final output was provided."

async def demonstrate_workflow():
    """Run a demonstration of the research workflow with a sample research task."""
    
    # Sample research task
    sample_query = "What are the emerging trends in renewable energy storage technologies?"
    
    print("=" * 80)
    print("CORE CONCEPTS DEMO - RESEARCH WORKFLOW")
    print("=" * 80)
    print(f"\nSample research task: {sample_query}\n")
    print("=" * 80)
    
    try:
        # Run the workflow
        result = await run_research_workflow(sample_query)
        
        # Display the result
        print("\nWORKFLOW RESULT:")
        print("=" * 80)
        print(result)
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during workflow execution: {str(e)}")
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    """
    Main entry point for the core concepts demo.
    
    Run this file directly to see a demonstration of the research workflow
    with handoffs between specialized agents.
    """
    asyncio.run(demonstrate_workflow())
    hooks=DemoAgentHooks(),
    output_guardrails=[ContentQualityGuardrail(min_length=200)],
)

#!/usr/bin/env python3
"""
Core Concepts Demo

This file demonstrates the key concepts of the OpenAI Agents Python SDK:
1. Main agent with clear instructions
2. Multiple specialized sub-agents for handoffs
3. Input and output guardrails
4. Custom tracing implementation
5. Different agent patterns (deterministic, iterative)

The demo creates a system with a main coordinator agent that can hand off tasks
to specialized sub-agents for specific tasks, all with appropriate guardrails
and tracing.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ValidationError, validator
from src.agents import Agent, AgentFinish, AgentStep, tool

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#############################################################
# SECTION 1: Context and Output Types
#############################################################

class DomainType(str, Enum):
    """Enum defining the domains for specialized agents."""
    RESEARCH = "research"
    CODING = "coding"
    CUSTOMER_SERVICE = "customer_service"
    DATA_ANALYSIS = "data_analysis"


@dataclass
class AgentContext:
    """
    Shared context for agent state management.
    
    This context will be passed between agents during handoffs and
    is used to maintain state throughout the conversation.
    """
    user_id: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: f"session_{int(time.time())}")
    metadata: Dict[str, Any] = field(default_factory=dict)
    current_domain: Optional[DomainType] = None
    
    def add_to_history(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
    
    def get_history_as_string(self) -> str:
        """Get the conversation history as a formatted string."""
        result = []
        for msg in self.conversation_history:
            result.append(f"{msg['role'].upper()}: {msg['content']}")
        return "\n".join(result)


class AgentInput(BaseModel):
    """
    Standardized input format for all agents.
    
    Using Pydantic for automatic validation and type checking.
    """
    query: str = Field(..., description="The user's query or request")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for the request")
    
    @validator("query")
    def query_not_empty(cls, v):
        """Validate that the query is not empty."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v
    
    @validator("query")
    def query_length(cls, v):
        """Validate that the query is within a reasonable length."""
        if len(v) > 1000:
            raise ValueError("Query is too long (max 1000 characters)")
        return v


class AgentOutput(BaseModel):
    """
    Standardized output format for all agents.
    
    Provides a consistent structure for agent responses.
    """
    answer: str = Field(..., description="The agent's response to the user's query")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the answer")
    sources: List[str] = Field(default_factory=list, description="Sources used to generate the answer")
    followup_questions: List[str] = Field(default_factory=list, description="Suggested follow-up questions")
    handoff_recommendation: Optional[DomainType] = Field(None, description="Recommended agent for handoff")
    
    @validator("answer")
    def answer_not_empty(cls, v):
        """Validate that the answer is not empty."""
        if not v.strip():
            raise ValueError("Answer cannot be empty")
        return v


#############################################################
# SECTION 2: Custom Tracing Implementation
#############################################################

class TracingManager:
    """
    Custom tracing implementation to track agent activities.
    
    The TracingManager provides hooks into the agent lifecycle to collect
    metrics, log interactions, and export traces to external systems.
    """
    
    def __init__(self, session_id: str, output_path: Optional[str] = None, external_service_url: Optional[str] = None):
        self.session_id = session_id
        self.start_time = time.time()
        self.spans = []
        self.current_span_id = 0
        self.output_path = output_path
        self.external_service_url = external_service_url
        
    def start_span(self, name: str, parent_id: Optional[int] = None, attributes: Optional[Dict[str, Any]] = None) -> int:
        """Start a new tracing span."""
        span_id = self.current_span_id
        self.current_span_id += 1
        
        span = {
            "id": span_id,
            "name": name,
            "parent_id": parent_id,
            "start_time": time.time(),
            "end_time": None,
            "attributes": attributes or {},
            "events": []
        }
        
        self.spans.append(span)
        logger.info(f"Started span: {name} (ID: {span_id})")
        return span_id
    
    def end_span(self, span_id: int, status: str = "success") -> None:
        """End an existing tracing span."""
        for span in self.spans:
            if span["id"] == span_id:
                span["end_time"] = time.time()
                span["status"] = status
                logger.info(f"Ended span: {span['name']} (ID: {span_id}) with status: {status}")
                return
                
        logger.warning(f"Attempted to end nonexistent span with ID {span_id}")
    
    def add_event(self, span_id: int, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to an existing span."""
        for span in self.spans:
            if span["id"] == span_id:
                event = {
                    "name": name,
                    "timestamp": time.time(),
                    "attributes": attributes or {}
                }
                span["events"].append(event)
                logger.debug(f"Added event: {name} to span ID {span_id}")
                return
                
        logger.warning(f"Attempted to add event to nonexistent span with ID {span_id}")
    
    def export_trace(self) -> Dict[str, Any]:
        """Export the complete trace as a dictionary."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": time.time(),
            "duration": time.time() - self.start_time,
            "spans": self.spans
        }
    
    def save_trace(self) -> None:
        """Save the trace to a file if output_path is specified."""
        if not self.output_path:
            logger.warning("No output path specified, trace not saved")
            return
            
        trace = self.export_trace()
        try:
            with open(self.output_path, 'w') as f:
                json.dump(trace, f, indent=2)
            logger.info(f"Saved trace to {self.output_path}")
        except Exception as e:
            logger.error(f"Failed to save trace: {e}")
    
    def send_to_external_service(self) -> bool:
        """Send trace data to an external monitoring service."""
        if not self.external_service_url:
            logger.warning("No external service URL specified, trace not sent")
            return False
            
        # This would normally use a proper HTTP client
        logger.info(f"Sending trace to {self.external_service_url}")
        # Mock implementation - in real code this would make an API call
        logger.info("Trace sent successfully")
        return True

    async def trace_agent_run(self, agent_name: str, run_id: str, func):
        """Decorator-like function to trace an entire agent run."""
        span_id = self.start_span(f"agent_run:{agent_name}", attributes={"run_id": run_id})
        
        try:
            result = await func()
            self.end_span(span_id, "success")
            return result
        except Exception as e:
            self.add_event(span_id, "exception", {"error_type": type(e).__name__, "message": str(e)})
            self.end_span(span_id, "error")
            raise


#############################################################
# SECTION 3: Input and Output Guardrails
#############################################################

class InputGuardrail:
    """
    Input validation and sanitization guardrail.
    
    Applies various checks to ensure inputs are safe and appropriate.
    """
    
    def __init__(self, banned_patterns: Optional[List[str]] = None):
        self.banned_patterns = banned_patterns or [
            r'\b(?:password|api[_\s]?key|secret|credentials)\b',  # Sensitive information
            r'(?:\b|\d+\.)+\d{1,3}\.(?:\d{1,3}\.){2}\d{1,3}\b',  # IP addresses
        ]
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.banned_patterns]
    
    def validate(self, input_data: AgentInput) -> tuple[bool, Optional[str]]:
        """
        Validate the input against all guardrails.
        
        Returns (is_valid, reason) where is_valid is a boolean indicating
        if the input passed all guardrails, and reason is an optional string
        explaining why the input was rejected.
        """
        # Check for banned patterns
        for pattern in self.compiled_patterns:
            if pattern.search(input_data.query):
                return False, "Input contains potentially sensitive information"
        
        # Check for overall length (prevent prompt injection via long inputs)
        if len(input_data.query) > 1000:
            return False, "Input exceeds maximum allowed length"
        
        return True, None
    
    def sanitize(self, input_data: AgentInput) -> AgentInput:
        """
        Sanitize the input by removing or replacing problematic content.
        
        This is applied after validation as a secondary defense.
        """
        query = input_data.query
        
        # Replace potential HTML/script tags
        query = re.sub(r'<[^>]*>', '[TAG]', query)
        
        # Create a sanitized copy
        sanitized_input = AgentInput(
            query=query,
            context=input_data.context
        )
        
        return sanitized_input


class OutputGuardrail:
    """
    Output validation and sanitization guardrail.
    
    Ensures that agent outputs meet quality and safety standards.
    """
    
    def __init__(
        self, 
        min_confidence: float = 0.6,
        max_answer_length: int = 2000,
        require_sources: bool = False,
        banned_patterns: Optional[List[str]] = None
    ):
        self.min_confidence = min_confidence
        self.max_answer_length = max_answer_length
        self.require_sources = require_sources
        self.banned_patterns = banned_patterns or [
            r'\b(?:I don\'t know|unable to help|insufficient information)\b',  # Refusal patterns
            r'\b(?:password|api[_\s]?key|secret|credentials)\b',  # Sensitive information
        ]
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.banned_patterns]
    
    def validate(self, output_data: AgentOutput) -> tuple[bool, Optional[str]]:
        """
        Validate the output against all guardrails.
        
        Returns (is_valid, reason) where is_valid is a boolean indicating
        if the output passed all guardrails, and reason is an optional string
        explaining why the output was rejected.
        """
        # Check confidence threshold
        if output_data.confidence < self.min_confidence:
            return False, f"Confidence score ({output_data.confidence}) below minimum threshold ({self.min_confidence})"
        
        # Check for banned patterns
        for pattern in self.compiled_patterns:
            if pattern.search(output_data.answer):
                return False, "Output contains problematic content"
        
        # Check answer length
        if len(output_data.answer) > self.max_answer_length:
            return False, f"Answer exceeds maximum allowed length ({self.max_answer_length})"
        
        # Check for required sources
        if self.require_sources and not output_data.sources:
            return False, "Output missing required sources"
        
        return True, None
    
    def improve_output(self, output_data: AgentOutput) -> AgentOutput:
        """
        Attempt to improve output quality if it falls below standards.
        
        This can be called when validation fails to try to fix issues.
        """
        answer = output_data.answer
        
        # Truncate if too long
        if len(answer) > self.max_answer_length:
            answer = answer[:self.max_answer_length - 3] + "..."
        
        # Remove harmful patterns
        for pattern in self.compiled_patterns:
            answer = pattern.sub("[REDACTED]", answer)
        
        # Create improved output
        improved_output = AgentOutput(
            answer=answer,
            confidence=output_data.confidence,
            sources=output_data.sources,
            followup_questions=output_data.followup_questions,
            handoff_recommendation=output_data.handoff_recommendation
        )
        
        return improved_output


#############################################################
# SECTION 4: Tool Definitions
#############################################################

@tool("search_knowledge_base")
async def search_knowledge_base(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the knowledge base for relevant information.
    
    Args:
        query: The search query.
        max_results: Maximum number of results to return.
        
    Returns:
        A list of matching documents with their content and metadata.
    """
    # This would normally interface with a real search system
    # Here we'll return mock results
    await asyncio.sleep(0.5)  # Simulate network delay
    
    mock_results = [
        {
            "id": "doc1",
            "title": "Introduction to AI Agents",
            "content": "AI agents are systems that can perceive their environment and take actions to achieve goals.",
            "relevance_score": 0.92
        },
        {
            "id": "doc2",
            "title": "Handoffs Between Agents",
            "content": "Agent handoffs allow for specialized processing by transferring control between agents.",
            "relevance_score": 0.87
        },
        {
            "id": "doc3",
            "title": "Guardrails in AI Systems",
            "content":

