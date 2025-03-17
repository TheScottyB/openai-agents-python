#!/usr/bin/env python3

"""
Tool Analyzer Agent
==================

Specialized agent for analyzing tool definitions against best practices.
"""

import json
from typing import Dict, Any, Optional
from agents import Agent, function_tool, ModelSettings, RunContextWrapper


@function_tool(use_docstring_info=True)
async def analyze_tool(ctx: RunContextWrapper, tool_definition: str) -> str:
    """Analyze a tool definition for improvements and best practices.
    
    Args:
        tool_definition: The tool definition to analyze in JSON string format.
                         Should include name, description, and parameters.
    
    Returns:
        A JSON string containing analysis results and recommendations.
    """
    try:
        # Parse the tool definition
        tool_dict = json.loads(tool_definition)
        
        # Create the analysis structure
        analysis = {
            "schema_check": "Valid" if all(k in tool_dict for k in ["name", "description", "parameters"]) else "Invalid",
            "docstring_check": "Present" if "description" in tool_dict and tool_dict["description"] else "Missing",
            "error_handling": "Implemented" if "failure_error_function" in tool_dict else "Missing",
            "recommendations": []
        }
        
        # Add recommendations based on analysis
        if analysis["schema_check"] == "Invalid":
            analysis["recommendations"].append("Add missing required fields (name, description, parameters)")
        if analysis["docstring_check"] == "Missing":
            analysis["recommendations"].append("Add proper documentation with clear description")
        if analysis["error_handling"] == "Missing":
            analysis["recommendations"].append("Implement error handling with failure_error_function")
        
        # Check parameter descriptions
        if "parameters" in tool_dict:
            for param_name, param_info in tool_dict["parameters"].items():
                if "description" not in param_info or not param_info["description"]:
                    analysis["recommendations"].append(f"Add description for parameter '{param_name}'")
                if "type" not in param_info:
                    analysis["recommendations"].append(f"Add type for parameter '{param_name}'")
        
        return json.dumps(analysis, indent=2)
    
    except json.JSONDecodeError:
        return json.dumps({
            "error": "Invalid JSON format",
            "message": "The tool definition must be a valid JSON string"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "message": "Error analyzing tool"
        }, indent=2)


def create_analyzer_agent(model: str = "gpt-4o") -> Agent:
    """
    Create an agent specialized in analyzing tool definitions.
    
    Args:
        model: The model to use for the agent
        
    Returns:
        An Agent configured for tool analysis
    """
    return Agent(
        name="Tool Analyzer",
        instructions="""You are a specialized agent for analyzing tool definitions and providing recommendations for improvements.

Your key responsibilities:
1. Analyze tool schemas against best practices
2. Identify missing or incomplete documentation
3. Suggest improvements for error handling
4. Check parameter definitions for completeness

When analyzing tools, focus on:
- Schema completeness and correctness 
- Documentation quality and completeness
- Error handling mechanisms
- Parameter typing and validation

Always use specific examples in your recommendations, and reference the OpenAI Agents SDK documentation when appropriate.
""",
        model=model,
        model_settings=ModelSettings(temperature=0.1),
        tools=[analyze_tool]
    )