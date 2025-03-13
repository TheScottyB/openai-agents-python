from agents import Agent, Runner, function_tool
from typing import Dict, List, Any, Optional
import json

class ToolKeeper:
    def __init__(self):
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
               - Ensure docstrings follow best practices
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
            3. Reference relevant documentation
            4. Consider error handling and edge cases
            5. Focus on maintainability and clarity
            """,
            tools=[
                self._create_analyze_tool(),
                self._create_validate_tool(),
                self._create_document_tool()
            ]
        )

    @staticmethod
    @function_tool(
        name_override="analyze_tool",
        description_override="Analyze a tool definition for improvements and best practices"
    )
    async def analyze_tool(tool_definition: str) -> str:
        """Analyze a tool definition for improvements and best practices.
        
        Args:
            tool_definition: The tool definition to analyze in string format. Should be a valid JSON string.
        
        Returns:
            str: Analysis results and recommendations in JSON format
        """
        try:
            tool_dict = json.loads(tool_definition)
            analysis = {
                "schema_check": "Valid" if all(k in tool_dict for k in ["name", "description", "parameters"]) else "Invalid",
                "docstring_check": "Present" if "description" in tool_dict and tool_dict["description"] else "Missing",
                "error_handling": "Implemented" if "failure_error_function" in tool_dict else "Missing",
                "recommendations": []
            }
            
            if analysis["schema_check"] == "Invalid":
                analysis["recommendations"].append("Add missing required fields (name, description, parameters)")
            if analysis["docstring_check"] == "Missing":
                analysis["recommendations"].append("Add proper documentation with clear description")
            if analysis["error_handling"] == "Missing":
                analysis["recommendations"].append("Implement error handling with failure_error_function")
            
            return json.dumps(analysis, indent=2)
        except Exception as e:
            return f"Error analyzing tool: {str(e)}"

    @staticmethod
    @function_tool(
        name_override="validate_tool",
        description_override="Validate a tool definition against SDK requirements"
    )
    async def validate_tool(tool_definition: str) -> str:
        """Validate a tool definition against SDK requirements.
        
        Args:
            tool_definition: The tool definition to validate. Should be a valid JSON string.
        
        Returns:
            str: Validation results in JSON format
        """
        try:
            tool_dict = json.loads(tool_definition)
            validation = {
                "is_valid": True,
                "errors": [],
                "warnings": []
            }
            
            required_fields = ["name", "description", "parameters"]
            for field in required_fields:
                if field not in tool_dict:
                    validation["is_valid"] = False
                    validation["errors"].append(f"Missing required field: {field}")
            
            if "parameters" in tool_dict and not isinstance(tool_dict["parameters"], dict):
                validation["is_valid"] = False
                validation["errors"].append("Parameters must be a dictionary")
            
            return json.dumps(validation, indent=2)
        except Exception as e:
            return f"Error validating tool: {str(e)}"

    @staticmethod
    @function_tool(
        name_override="document_tool",
        description_override="Generate proper documentation for a tool"
    )
    async def document_tool(tool_definition: str) -> str:
        """Generate proper documentation for a tool.
        
        Args:
            tool_definition: JSON string containing the tool definition with name, description, and parameters.
        
        Returns:
            str: Generated documentation in markdown format
        """
        try:
            tool_dict = json.loads(tool_definition)
            tool_name = tool_dict.get("name", "Unnamed Tool")
            description = tool_dict.get("description", "No description provided")
            parameters = tool_dict.get("parameters", {})

            doc_template = f"""# {tool_name}

            {description}

            ## Parameters

            """
            
            for param_name, param_info in parameters.items():
                doc_template += f"- `{param_name}`: {param_info.get('description', 'No description provided')}\n"
            
            return doc_template
        except Exception as e:
            return f"Error generating documentation: {str(e)}"

    def _create_analyze_tool(self):
        return self.analyze_tool

    def _create_validate_tool(self):
        return self.validate_tool

    def _create_document_tool(self):
        return self.document_tool

    async def run(self, query: str) -> str:
        """Run the Tool Keeper agent with a query.
        
        Args:
            query: The query or request for the Tool Keeper
        
        Returns:
            str: The agent's response
        """
        result = await Runner.run(self.agent, query)
        return result.final_output
