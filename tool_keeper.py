#!/usr/bin/env python3

"""
Tool Keeper
===========

A specialized agent for validating, analyzing, and documenting tool definitions
according to OpenAI Agents SDK best practices.

Features:
- Validates tool schemas against SDK requirements
- Analyzes tools for improvements and adherence to best practices
- Generates proper documentation for tools
- Suggests optimizations and improvements

Usage:
    from tool_keeper import ToolKeeper
    
    # Create a Tool Keeper instance
    tool_keeper = ToolKeeper()
    
    # Run the agent with a tool analysis request
    result = await tool_keeper.run("Analyze this tool: {tool_json}")
"""

import json
from typing import Dict, Any, Optional, List, cast
from pydantic import BaseModel, Field

from agents import (
    Agent,
    Runner,
    function_tool,
    ModelSettings,
    RunContextWrapper,
)


# Data models for validation and responses
class ToolParameter(BaseModel):
    """Model for a tool parameter."""
    type: str = Field(description="The data type of the parameter")
    description: Optional[str] = Field(None, description="Description of the parameter")
    required: Optional[bool] = Field(None, description="Whether the parameter is required")


class ToolDefinition(BaseModel):
    """Model for a tool definition."""
    name: str = Field(description="The name of the tool")
    description: str = Field(description="Description of what the tool does")
    parameters: Dict[str, Dict[str, Any]] = Field(description="Parameters the tool accepts")
    function: Optional[str] = Field(None, description="Function reference for the tool (optional)")


class AnalysisResult(BaseModel):
    """Model for tool analysis results."""
    schema_check: str = Field(description="Whether the schema is valid")
    docstring_check: str = Field(description="Whether documentation is present")
    error_handling: str = Field(description="Whether error handling is implemented")
    recommendations: List[str] = Field(description="Improvement recommendations")


class ValidationResult(BaseModel):
    """Model for tool validation results."""
    is_valid: bool = Field(description="Whether the tool definition is valid")
    errors: List[str] = Field(description="List of validation errors")
    warnings: List[str] = Field(description="List of validation warnings")


class ToolKeeper:
    """
    A specialized agent for validating, analyzing, and documenting tool definitions.
    Provides functionality to ensure tools follow OpenAI Agents SDK best practices.
    """

    def __init__(self):
        """Initialize the Tool Keeper agent with specialized tools."""
        self.agent = Agent(
            name="Tool Keeper",
            instructions="""You are an expert agent specialized in tool management and implementation.
            Your responsibilities include:
            
            1. Tool Implementation Review:
               - Validate tool schemas and implementations
               - Suggest improvements for tool definitions
               - Ensure proper error handling
            
            2. Tool Documentation:
               - Review and improve tool documentation
               - Ensure docstrings follow best practices (Google style)
               - Maintain clear parameter descriptions
            
            3. Tool Optimization:
               - Analyze tool performance and usage patterns
               - Suggest optimizations for commonly used tools
               - Identify opportunities for new tools
            
            4. Tool Integration:
               - Help integrate new tools with existing systems
               - Ensure proper typing and schema validation
               - Maintain consistency across tool implementations
            
            When providing feedback or suggestions:
            1. Be specific about implementation details
            2. Include code examples where appropriate
            3. Reference relevant documentation from the OpenAI Agents SDK
            4. Consider error handling and edge cases
            5. Focus on maintainability and clarity
            
            Always use your tools to perform the core operations of analyzing, validating and documenting tools.
            """,
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.2),
            tools=[
                self._analyze_tool,
                self._validate_tool,
                self._document_tool,
            ]
        )
        
        # Analysis agent for direct method access
        self.analysis_agent = Agent(
            name="Analysis Agent",
            instructions="Analyze the given tool definition for improvements based on OpenAI Agents SDK best practices.",
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.1),
            tools=[self._analyze_tool]
        )
        
        # Validation agent for direct method access
        self.validation_agent = Agent(
            name="Validation Agent",
            instructions="Validate the given tool definition against SDK requirements.",
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.1),
            tools=[self._validate_tool]
        )
        
        # Documentation agent for direct method access
        self.documentation_agent = Agent(
            name="Documentation Agent",
            instructions="Generate proper documentation for the given tool definition.",
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.1),
            tools=[self._document_tool]
        )

    @function_tool(use_docstring_info=True)
    async def _analyze_tool(self, ctx: RunContextWrapper, tool_definition: str) -> str:
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
            analysis = AnalysisResult(
                schema_check="Valid" if all(k in tool_dict for k in ["name", "description", "parameters"]) else "Invalid",
                docstring_check="Present" if "description" in tool_dict and tool_dict["description"] else "Missing",
                error_handling="Implemented" if "failure_error_function" in tool_dict else "Missing",
                recommendations=[]
            )
            
            # Add recommendations based on analysis
            if analysis.schema_check == "Invalid":
                analysis.recommendations.append("Add missing required fields (name, description, parameters)")
            if analysis.docstring_check == "Missing":
                analysis.recommendations.append("Add proper documentation with clear description")
            if analysis.error_handling == "Missing":
                analysis.recommendations.append("Implement error handling with failure_error_function")
            
            # Check parameter descriptions
            if "parameters" in tool_dict:
                for param_name, param_info in tool_dict["parameters"].items():
                    if "description" not in param_info or not param_info["description"]:
                        analysis.recommendations.append(f"Add description for parameter '{param_name}'")
                    if "type" not in param_info:
                        analysis.recommendations.append(f"Add type for parameter '{param_name}'")
            
            return json.dumps(analysis.model_dump(), indent=2)
        
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

    @function_tool(use_docstring_info=True)
    async def _validate_tool(self, ctx: RunContextWrapper, tool_definition: str) -> str:
        """Validate a tool definition against SDK requirements.
        
        Args:
            tool_definition: The tool definition to validate in JSON string format.
        
        Returns:
            A JSON string containing validation results, with any errors or warnings.
        """
        try:
            # Parse the tool definition
            tool_dict = json.loads(tool_definition)
            
            validation = ValidationResult(
                is_valid=True,
                errors=[],
                warnings=[]
            )
            
            # Check required fields
            required_fields = ["name", "description", "parameters"]
            for field in required_fields:
                if field not in tool_dict:
                    validation.is_valid = False
                    validation.errors.append(f"Missing required field: {field}")
            
            # Check parameters structure
            if "parameters" in tool_dict:
                if not isinstance(tool_dict["parameters"], dict):
                    validation.is_valid = False
                    validation.errors.append("Parameters must be a dictionary")
                else:
                    # Check each parameter
                    for param_name, param_info in tool_dict["parameters"].items():
                        if not isinstance(param_info, dict):
                            validation.is_valid = False
                            validation.errors.append(f"Parameter '{param_name}' must be a dictionary")
                        else:
                            if "type" not in param_info:
                                validation.warnings.append(f"Parameter '{param_name}' is missing type information")
                            if "description" not in param_info:
                                validation.warnings.append(f"Parameter '{param_name}' is missing description")
            
            # Check name format
            if "name" in tool_dict:
                if not isinstance(tool_dict["name"], str) or not tool_dict["name"]:
                    validation.is_valid = False
                    validation.errors.append("Tool name must be a non-empty string")
                elif " " in tool_dict["name"]:
                    validation.warnings.append("Tool name should not contain spaces")
            
            return json.dumps(validation.model_dump(), indent=2)
            
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

    @function_tool(use_docstring_info=True)
    async def _document_tool(self, ctx: RunContextWrapper, tool_definition: str) -> str:
        """Generate proper documentation for a tool in markdown format.
        
        Args:
            tool_definition: JSON string containing the tool definition with name, 
                             description, and parameters.
        
        Returns:
            A markdown-formatted string with documentation for the tool.
        """
        try:
            # Parse the tool definition
            tool_dict = json.loads(tool_definition)
            
            # Extract tool information
            tool_name = tool_dict.get("name", "Unnamed Tool")
            description = tool_dict.get("description", "No description provided")
            parameters = tool_dict.get("parameters", {})
            
            # Generate documentation
            doc = f"# {tool_name}\n\n"
            doc += f"{description}\n\n"
            
            if parameters:
                doc += "## Parameters\n\n"
                for param_name, param_info in parameters.items():
                    param_type = param_info.get("type", "unknown")
                    param_desc = param_info.get("description", "No description provided")
                    required = param_info.get("required", False)
                    required_str = "Required" if required else "Optional"
                    
                    doc += f"- `{param_name}` ({param_type}, {required_str}): {param_desc}\n"
            
            doc += "\n## Usage Example\n\n"
            doc += "```python\n"
            doc += f"@function_tool\n"
            doc += f"async def {tool_name.lower().replace(' ', '_')}("
            
            # Add parameters to example
            params = ["self", "ctx: RunContextWrapper"]
            for param_name, param_info in parameters.items():
                param_type = param_info.get("type", "Any")
                python_type = {
                    "string": "str",
                    "number": "float",
                    "integer": "int",
                    "boolean": "bool",
                    "object": "Dict[str, Any]",
                    "array": "List[Any]"
                }.get(param_type, "Any")
                
                default = "" if param_info.get("required", False) else " = None"
                params.append(f"{param_name}: {python_type}{default}")
            
            doc += ", ".join(params)
            doc += "):\n"
            doc += f'    """{description}\n\n'
            
            # Add docstring parameters
            if parameters:
                doc += "    Args:\n"
                for param_name, param_info in parameters.items():
                    param_desc = param_info.get("description", "No description provided")
                    doc += f"        {param_name}: {param_desc}\n"
            
            doc += '\n    Returns:\n        The result\n    """\n'
            doc += "    # Implementation goes here\n"
            doc += "    pass\n"
            doc += "```\n"
            
            return doc
            
        except json.JSONDecodeError:
            return "Error: Invalid JSON format in tool definition"
        except Exception as e:
            return f"Error generating documentation: {str(e)}"

    async def run(self, query: str) -> str:
        """Run the Tool Keeper agent with a query.
        
        Args:
            query: The query or request for the Tool Keeper
        
        Returns:
            The agent's response as a string
        """
        result = await Runner.run(self.agent, query)
        return result.final_output
    
    async def analyze_tool_directly(self, tool_definition: str) -> Dict[str, Any]:
        """Analyze a tool definition directly using a dedicated agent.
        
        Args:
            tool_definition: The tool definition as a JSON string
            
        Returns:
            The analysis results as a dictionary
        """
        # Run the analysis through the agent loop
        result = await Runner.run(
            self.analysis_agent, 
            f"Analyze this tool definition and return the results in JSON format: {tool_definition}"
        )
        
        # Extract the JSON from the response
        try:
            # Find JSON in the response
            response_text = result.final_output
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                # If no JSON found, try direct tool call
                context_wrapper = RunContextWrapper(None)
                direct_result = await self._analyze_tool(context_wrapper, tool_definition)
                return json.loads(direct_result)
        except Exception:
            # Fallback to direct tool call if JSON extraction fails
            context_wrapper = RunContextWrapper(None)
            direct_result = await self._analyze_tool(context_wrapper, tool_definition)
            return json.loads(direct_result)
    
    async def validate_tool_directly(self, tool_definition: str) -> Dict[str, Any]:
        """Validate a tool definition directly using a dedicated agent.
        
        Args:
            tool_definition: The tool definition as a JSON string
            
        Returns:
            The validation results as a dictionary
        """
        # Run the validation through the agent loop
        result = await Runner.run(
            self.validation_agent, 
            f"Validate this tool definition and return the results in JSON format: {tool_definition}"
        )
        
        # Extract the JSON from the response
        try:
            # Find JSON in the response
            response_text = result.final_output
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                # If no JSON found, try direct tool call
                context_wrapper = RunContextWrapper(None)
                direct_result = await self._validate_tool(context_wrapper, tool_definition)
                return json.loads(direct_result)
        except Exception:
            # Fallback to direct tool call if JSON extraction fails
            context_wrapper = RunContextWrapper(None)
            direct_result = await self._validate_tool(context_wrapper, tool_definition)
            return json.loads(direct_result)
    
    async def document_tool_directly(self, tool_definition: str) -> str:
        """Generate documentation for a tool definition using a dedicated agent.
        
        Args:
            tool_definition: The tool definition as a JSON string
            
        Returns:
            The generated documentation as a string
        """
        # Run the documentation through the agent loop
        result = await Runner.run(
            self.documentation_agent, 
            f"Generate documentation for this tool definition: {tool_definition}"
        )
        
        return result.final_output


if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Example tool definition
        example_tool = {
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
        
        # Create the Tool Keeper
        tool_keeper = ToolKeeper()
        
        # Analyze the tool using the agent
        print("Analyzing tool via agent...")
        analysis = await tool_keeper.analyze_tool_directly(json.dumps(example_tool))
        print(json.dumps(analysis, indent=2))
        
        # Validate the tool using the agent
        print("\nValidating tool via agent...")
        validation = await tool_keeper.validate_tool_directly(json.dumps(example_tool))
        print(json.dumps(validation, indent=2))
        
        # Generate documentation using the agent
        print("\nGenerating documentation via agent...")
        docs = await tool_keeper.document_tool_directly(json.dumps(example_tool))
        print(docs)
    
    asyncio.run(main())