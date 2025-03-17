#!/usr/bin/env python3

"""
Tool Keeper Guardrails
=====================

Input and output guardrails for filtering, validating, and checking tool definitions
and generated content.
"""

import json
import re
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

from agents import (
    Agent,
    RunContextWrapper,
    GuardrailFunctionOutput,
    input_guardrail,
    output_guardrail,
    TResponseInputItem,
    Runner,
)


class ToolSchemaValidation(BaseModel):
    """Validation results for tool schema structure."""
    is_valid: bool
    missing_fields: List[str] = Field(default_factory=list)
    invalid_fields: List[str] = Field(default_factory=list)
    reasoning: str


# Tool schema validation guard agent
tool_schema_validation_agent = Agent(
    name="Tool Schema Validator",
    instructions="""You validate if a given input contains a valid tool definition schema.
    A valid tool definition must include at least:
    - A "name" field that's a non-empty string
    - A "description" field that's a non-empty string
    - A "parameters" object with at least one parameter
    
    Check if the input contains JSON that could be a tool definition.
    If there's no JSON or the JSON doesn't look like a tool definition, mark it as invalid.
    """,
    output_type=ToolSchemaValidation,
)


@input_guardrail
async def tool_schema_guardrail(
    context: RunContextWrapper[None], 
    agent: Agent, 
    input: Union[str, List[TResponseInputItem]]
) -> GuardrailFunctionOutput:
    """
    Input guardrail that validates if the input contains a properly structured tool definition.
    Prevents processing invalid schemas that would waste resources.
    """
    # Extract content from input
    content = input if isinstance(input, str) else input[-1]["content"] if input else ""
    
    # Quick preliminary check for JSON structure
    json_pattern = r'\{[\s\S]*\}'
    potential_json = re.search(json_pattern, content)
    
    if not potential_json:
        # No JSON-like structure found
        return GuardrailFunctionOutput(
            output_info={
                "is_valid": False,
                "reason": "No JSON structure found in input"
            },
            tripwire_triggered=False  # Allow processing, but tool will handle validation
        )
        
    # Run validation through validation agent for more thorough check
    validation_result = await Runner.run(
        tool_schema_validation_agent, 
        content,
        context=context.context
    )
    
    validation = validation_result.final_output_as(ToolSchemaValidation)
    
    # Only trigger tripwire if user explicitly submitted a tool definition
    # that's clearly invalid (contains tool schema keywords but fails validation)
    schema_keywords = ["name", "description", "parameters", "tool", "function"]
    explicit_tool_attempt = any(keyword in content.lower() for keyword in schema_keywords)
    
    return GuardrailFunctionOutput(
        output_info=validation.model_dump(),
        tripwire_triggered=explicit_tool_attempt and not validation.is_valid
    )


class SensitiveDataCheck(BaseModel):
    """Results of checking for sensitive data in content."""
    contains_sensitive_data: bool
    detected_types: List[str] = Field(default_factory=list)
    reasoning: str


# Sensitive data check agent
sensitive_data_check_agent = Agent(
    name="Sensitive Data Detector",
    instructions="""You check if text contains sensitive data that shouldn't be included in tool definitions.
    
    Sensitive data includes:
    - API keys, tokens, or credentials
    - Passwords or authentication details
    - Personal information (addresses, phone numbers, etc.)
    - Internal server paths or database connection strings
    - Environment variables with sensitive values
    
    Carefully examine the content and determine if it contains any sensitive information.
    """,
    output_type=SensitiveDataCheck,
)


@output_guardrail
async def sensitive_data_guardrail(
    context: RunContextWrapper, 
    agent: Agent, 
    output: Any
) -> GuardrailFunctionOutput:
    """
    Output guardrail that checks if generated content contains sensitive data.
    Prevents leaking credentials, API keys, or personal information.
    """
    # Convert output to string for checking
    output_str = str(output)
    
    # Run check through specialized agent
    check_result = await Runner.run(
        sensitive_data_check_agent, 
        f"Check this content for sensitive data: {output_str}",
        context=context.context
    )
    
    check = check_result.final_output_as(SensitiveDataCheck)
    
    return GuardrailFunctionOutput(
        output_info=check.model_dump(),
        tripwire_triggered=check.contains_sensitive_data
    )


class ContentOffensivenessCheck(BaseModel):
    """Results of checking for offensive or inappropriate content."""
    is_offensive: bool
    categories: List[str] = Field(default_factory=list)
    reasoning: str


# Offensive content check agent
offensive_content_check_agent = Agent(
    name="Content Appropriateness Checker",
    instructions="""You check if content contains offensive, inappropriate, or harmful material.
    
    Look for:
    - Offensive language or slurs
    - Harmful instructions or malicious code
    - Content that promotes illegal activities
    - Privacy violations
    - Discriminatory or hateful content
    
    Carefully examine the content and determine if it's appropriate for a professional tool definition.
    """,
    output_type=ContentOffensivenessCheck,
)


@output_guardrail
async def offensive_content_guardrail(
    context: RunContextWrapper, 
    agent: Agent, 
    output: Any
) -> GuardrailFunctionOutput:
    """
    Output guardrail that checks if generated content contains offensive or inappropriate material.
    Ensures all tool documentation and implementations remain professional and appropriate.
    """
    # Convert output to string for checking
    output_str = str(output)
    
    # Run check through specialized agent
    check_result = await Runner.run(
        offensive_content_check_agent, 
        f"Check if this content is appropriate: {output_str}",
        context=context.context
    )
    
    check = check_result.final_output_as(ContentOffensivenessCheck)
    
    return GuardrailFunctionOutput(
        output_info=check.model_dump(),
        tripwire_triggered=check.is_offensive
    )