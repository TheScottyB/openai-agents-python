#!/usr/bin/env python3

"""
Tool Documenter Agent
==================

Specialized agent for generating documentation for tool definitions.
"""

import json
from typing import Dict, Any, Optional
from agents import Agent, function_tool, ModelSettings, RunContextWrapper


@function_tool(use_docstring_info=True)
async def document_tool(ctx: RunContextWrapper, tool_definition: str) -> str:
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


@function_tool(use_docstring_info=True)
async def generate_implementation(ctx: RunContextWrapper, tool_definition: str) -> str:
    """Generate a Python implementation for a tool definition.
    
    Args:
        tool_definition: JSON string containing the tool definition with name, description, and parameters.
        
    Returns:
        A Python code implementation for the tool.
    """
    try:
        # Parse the tool definition
        tool_dict = json.loads(tool_definition)
        
        # Extract tool information
        tool_name = tool_dict.get("name", "unnamed_tool")
        description = tool_dict.get("description", "No description provided")
        parameters = tool_dict.get("parameters", {})
        
        # Generate the implementation
        code = f"""from typing import Dict, Any, Optional
from agents import function_tool, RunContextWrapper

@function_tool(use_docstring_info=True)
async def {tool_name.lower().replace(' ', '_')}(
    ctx: RunContextWrapper"""
        
        # Add parameters
        for param_name, param_info in parameters.items():
            param_type = param_info.get("type", "Any")
            required = param_info.get("required", False)
            
            python_type = {
                "string": "str",
                "number": "float",
                "integer": "int",
                "boolean": "bool",
                "object": "Dict[str, Any]",
                "array": "List[Any]"
            }.get(param_type, "Any")
            
            default = "" if required else " = None"
            code += f",\n    {param_name}: {python_type}{default}"
        
        # Close parameters and add return type
        code += "\n) -> Dict[str, Any]:\n"
        
        # Add docstring
        code += f'    """{description}\n\n'
        
        if parameters:
            code += "    Args:\n"
            for param_name, param_info in parameters.items():
                param_desc = param_info.get("description", "No description provided")
                code += f"        {param_name}: {param_desc}\n"
        
        code += """
    Returns:
        A dictionary with the result
    """
        
        # Add error handling and examples
        code += f'    """\n    try:\n'
        code += '        # Implement the tool functionality here\n'
        code += '        result = {\n'
        code += '            "success": True,\n'
        
        # Add example result based on parameters
        for param_name in parameters:
            code += f'            "{param_name}_processed": {param_name},\n'
        
        code += '            "result": "Implementation needed"\n'
        code += '        }\n\n'
        code += '        return result\n'
        code += '    except Exception as e:\n'
        code += '        return {\n'
        code += '            "success": False,\n'
        code += '            "error": str(e)\n'
        code += '        }\n'
        
        return code
        
    except json.JSONDecodeError:
        return "Error: Invalid JSON format in tool definition"
    except Exception as e:
        return f"Error generating implementation: {str(e)}"


def create_documenter_agent(model: str = "gpt-4o") -> Agent:
    """
    Create an agent specialized in documenting tool definitions.
    
    Args:
        model: The model to use for the agent
        
    Returns:
        An Agent configured for tool documentation
    """
    return Agent(
        name="Tool Documenter",
        instructions="""You are a specialized agent for creating high-quality documentation for tool definitions.

Your key responsibilities:
1. Generate clear, comprehensive documentation for tools
2. Create code examples that demonstrate proper usage
3. Document parameters with their types and requirements
4. Provide implementation examples that follow best practices

When documenting tools, focus on:
- Clear explanations of what the tool does
- Complete parameter documentation with types and requirements
- Proper Python type hints and docstring formats
- Error handling best practices in examples
- Following the Google docstring style convention

Make your documentation developer-friendly and provide both usage examples and complete implementation suggestions.
""",
        model=model,
        model_settings=ModelSettings(temperature=0.2),
        tools=[document_tool, generate_implementation]
    )