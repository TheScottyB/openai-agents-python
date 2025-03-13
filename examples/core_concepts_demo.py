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

from examples.shared.utils import setup_logger

# Configure logging
logger = setup_logger(__name__)

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
Your job is

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

# Research Agent - Specializes in gathering information
researcher_instructions = """
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
    

