#!/usr/bin/env python3

"""
Tool Validator Agent
==================

Specialized agent for validating tool definitions against SDK requirements.
"""

import json
from typing import Dict, Any, Optional
from agents import Agent, function_tool, ModelSettings, RunContextWrapper


@function_tool(use_docstring_info=True)
async def validate_tool(ctx: RunContextWrapper, tool_definition: str) -> str:
    """Validate a tool definition against SDK requirements.
    
    Args:
        tool_definition: The tool definition to validate in JSON string format.
    
    Returns:
        A JSON string containing validation results, with any errors or warnings.
    """
    try:
        # Parse the tool definition
        tool_dict = json.loads(tool_definition)
        
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required fields
        required_fields = ["name", "description", "parameters"]
        for field in required_fields:
            if field not in tool_dict:
                validation["is_valid"] = False
                validation["errors"].append(f"Missing required field: {field}")
        
        # Check parameters structure
        if "parameters" in tool_dict:
            if not isinstance(tool_dict["parameters"], dict):
                validation["is_valid"] = False
                validation["errors"].append("Parameters must be a dictionary")
            else:
                # Check each parameter
                for param_name, param_info in tool_dict["parameters"].items():
                    if not isinstance(param_info, dict):
                        validation["is_valid"] = False
                        validation["errors"].append(f"Parameter '{param_name}' must be a dictionary")
                    else:
                        if "type" not in param_info:
                            validation["warnings"].append(f"Parameter '{param_name}' is missing type information")
                        if "description" not in param_info:
                            validation["warnings"].append(f"Parameter '{param_name}' is missing description")
        
        # Check name format
        if "name" in tool_dict:
            if not isinstance(tool_dict["name"], str) or not tool_dict["name"]:
                validation["is_valid"] = False
                validation["errors"].append("Tool name must be a non-empty string")
            elif " " in tool_dict["name"]:
                validation["warnings"].append("Tool name should not contain spaces")
        
        return json.dumps(validation, indent=2)
        
    except json.JSONDecodeError:
        return json.dumps({
            "is_valid": False,
            "errors": ["Invalid JSON format"],
            "warnings": []
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "is_valid": False,
            "errors": [f"Error validating tool: {str(e)}"],
            "warnings": []
        }, indent=2)


def create_validator_agent(model: str = "gpt-4o") -> Agent:
    """
    Create an agent specialized in validating tool definitions.
    
    Args:
        model: The model to use for the agent
        
    Returns:
        An Agent configured for tool validation
    """
    return Agent(
        name="Tool Validator",
        instructions="""You are a specialized agent for validating tool definitions against OpenAI Agents SDK requirements.

Your key responsibilities:
1. Validate that tool schemas conform to SDK requirements
2. Identify missing required fields
3. Check for proper parameter formatting
4. Flag invalid or problematic definitions

When validating tools, specifically check:
- That all required fields (name, description, parameters) are present
- That parameters have proper type and description information
- That naming conventions follow SDK best practices
- That the structure follows the expected JSON schema format

Be specific about validation errors and provide clear guidance on how to fix issues.
""",
        model=model,
        model_settings=ModelSettings(temperature=0.1),
        tools=[validate_tool]
    )