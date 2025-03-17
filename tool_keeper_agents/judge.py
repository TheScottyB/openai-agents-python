#!/usr/bin/env python3

"""
Tool Judge Agent
===============

Specialized agent for evaluating and scoring tool definitions and implementations.
Acts as a quality control mechanism for other agents' outputs.
"""

import json
from typing import Dict, Any, List
from dataclasses import dataclass
from agents import Agent, function_tool, ModelSettings, RunContextWrapper


@dataclass
class ToolEvaluation:
    """Evaluation results for a tool definition or implementation."""
    score: int  # 1-10
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    overall_assessment: str


@function_tool(use_docstring_info=True)
async def evaluate_tool_definition(ctx: RunContextWrapper, tool_definition: str) -> str:
    """Evaluate the quality of a tool definition.
    
    Args:
        tool_definition: The tool definition to evaluate in JSON string format.
        
    Returns:
        A JSON string containing the evaluation results.
    """
    try:
        # Parse the tool definition
        tool_dict = json.loads(tool_definition)
        
        # Perform evaluation
        evaluation = {
            "score": 0,
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "overall_assessment": ""
        }
        
        # Check completeness
        has_name = "name" in tool_dict and tool_dict["name"]
        has_description = "description" in tool_dict and tool_dict["description"]
        has_parameters = "parameters" in tool_dict and tool_dict["parameters"]
        
        score = 5  # Start with a middle score
        
        # Evaluate name
        if has_name:
            if len(tool_dict["name"]) < 3:
                evaluation["weaknesses"].append("Tool name is too short")
                evaluation["recommendations"].append("Use a more descriptive name")
                score -= 1
            elif len(tool_dict["name"]) > 30:
                evaluation["weaknesses"].append("Tool name is too long")
                evaluation["recommendations"].append("Consider a more concise name")
                score -= 1
            else:
                evaluation["strengths"].append("Tool name has appropriate length")
                score += 1
                
        # Evaluate description
        if has_description:
            if len(tool_dict["description"]) < 20:
                evaluation["weaknesses"].append("Description is too brief")
                evaluation["recommendations"].append("Provide a more detailed description")
                score -= 1
            elif len(tool_dict["description"]) > 500:
                evaluation["weaknesses"].append("Description is excessively long")
                evaluation["recommendations"].append("Make description more concise")
                score -= 1
            else:
                evaluation["strengths"].append("Description is informative and concise")
                score += 1
                
        # Evaluate parameters
        if has_parameters:
            has_param_issues = False
            for param_name, param_info in tool_dict["parameters"].items():
                if "description" not in param_info or not param_info["description"]:
                    evaluation["weaknesses"].append(f"Parameter '{param_name}' lacks description")
                    has_param_issues = True
                if "type" not in param_info:
                    evaluation["weaknesses"].append(f"Parameter '{param_name}' lacks type information")
                    has_param_issues = True
                    
            if not has_param_issues:
                evaluation["strengths"].append("Parameters are well-defined with proper types and descriptions")
                score += 2
            else:
                evaluation["recommendations"].append("Ensure all parameters have types and descriptions")
                score -= 2
        
        # Set final score
        evaluation["score"] = max(1, min(10, score))
        
        # Overall assessment
        if evaluation["score"] >= 8:
            evaluation["overall_assessment"] = "Excellent tool definition that follows best practices"
        elif evaluation["score"] >= 6:
            evaluation["overall_assessment"] = "Good tool definition with minor improvements needed"
        elif evaluation["score"] >= 4:
            evaluation["overall_assessment"] = "Adequate tool definition that needs several improvements"
        else:
            evaluation["overall_assessment"] = "Poor tool definition requiring significant improvements"
            
        return json.dumps(evaluation, indent=2)
        
    except json.JSONDecodeError:
        return json.dumps({
            "error": "Invalid JSON format",
            "message": "The tool definition must be a valid JSON string"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "message": "Error evaluating tool"
        }, indent=2)


@function_tool(use_docstring_info=True)
async def evaluate_implementation(ctx: RunContextWrapper, implementation: str, tool_definition: str) -> str:
    """Evaluate the quality of a tool implementation against its definition.
    
    Args:
        implementation: The Python implementation code as a string.
        tool_definition: The original tool definition in JSON string format.
        
    Returns:
        A JSON string containing the evaluation results.
    """
    try:
        # Parse the tool definition
        tool_dict = json.loads(tool_definition)
        
        # Perform evaluation
        evaluation = {
            "score": 0,
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "overall_assessment": ""
        }
        
        # Check if implementation has basic elements
        has_function = "@function_tool" in implementation
        has_async_def = "async def" in implementation
        has_docstring = '"""' in implementation or "'''" in implementation
        has_error_handling = "try:" in implementation and "except" in implementation
        
        score = 5  # Start with a middle score
        
        # Evaluate structure
        if has_function and has_async_def:
            evaluation["strengths"].append("Implementation uses proper function_tool decorator and async definition")
            score += 1
        else:
            evaluation["weaknesses"].append("Implementation is missing proper function_tool decorator or async definition")
            evaluation["recommendations"].append("Use @function_tool decorator and async def for implementation")
            score -= 1
            
        # Evaluate docstring
        if has_docstring:
            evaluation["strengths"].append("Implementation includes docstring")
            score += 1
        else:
            evaluation["weaknesses"].append("Implementation is missing docstring")
            evaluation["recommendations"].append("Add comprehensive docstring with Args and Returns sections")
            score -= 1
            
        # Evaluate error handling
        if has_error_handling:
            evaluation["strengths"].append("Implementation includes error handling")
            score += 1
        else:
            evaluation["weaknesses"].append("Implementation lacks error handling")
            evaluation["recommendations"].append("Add try/except blocks for robust error handling")
            score -= 1
            
        # Check for parameter matching
        tool_params = []
        if "parameters" in tool_dict:
            tool_params = list(tool_dict["parameters"].keys())
            
        all_params_found = True
        for param in tool_params:
            if param not in implementation:
                evaluation["weaknesses"].append(f"Implementation is missing parameter '{param}' from definition")
                all_params_found = False
                
        if all_params_found and tool_params:
            evaluation["strengths"].append("Implementation includes all parameters from definition")
            score += 2
        elif tool_params:
            evaluation["recommendations"].append("Ensure implementation handles all parameters from definition")
            score -= 2
            
        # Set final score
        evaluation["score"] = max(1, min(10, score))
        
        # Overall assessment
        if evaluation["score"] >= 8:
            evaluation["overall_assessment"] = "Excellent implementation that follows best practices"
        elif evaluation["score"] >= 6:
            evaluation["overall_assessment"] = "Good implementation with minor improvements needed"
        elif evaluation["score"] >= 4:
            evaluation["overall_assessment"] = "Adequate implementation that needs several improvements"
        else:
            evaluation["overall_assessment"] = "Poor implementation requiring significant improvements"
            
        return json.dumps(evaluation, indent=2)
        
    except json.JSONDecodeError:
        return json.dumps({
            "error": "Invalid JSON format",
            "message": "The tool definition must be a valid JSON string"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "message": "Error evaluating implementation"
        }, indent=2)


def create_judge_agent(model: str = "gpt-4o") -> Agent:
    """
    Create an agent specialized in evaluating tool definitions and implementations.
    
    Args:
        model: The model to use for the agent
        
    Returns:
        An Agent configured for tool evaluation
    """
    return Agent(
        name="Tool Judge",
        instructions="""You are a specialized judge agent for evaluating tool definitions and implementations.

Your key responsibilities:
1. Evaluate tool definitions for quality, completeness, and clarity
2. Evaluate tool implementations against their definitions
3. Provide detailed feedback with strengths and weaknesses
4. Score tools on a 1-10 scale with specific recommendations for improvement

When evaluating, focus on:
- Adherence to SDK standards and best practices
- Proper documentation and error handling
- Completeness of parameter definitions
- Clarity and usability
- Security considerations

Always provide balanced feedback with both strengths and areas for improvement.
""",
        model=model,
        tools=[evaluate_tool_definition, evaluate_implementation]
    )