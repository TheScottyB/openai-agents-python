#!/usr/bin/env python3

"""
Tool Keeper Unit Tests
=====================

Unit tests for the ToolKeeper class using pytest.

To run:
    uv run pytest tests/test_tool_keeper_unit.py -v
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from tool_keeper import ToolKeeper, ValidationResult, AnalysisResult


@pytest.fixture
def valid_tool() -> Dict[str, Any]:
    """Fixture providing a valid tool definition."""
    return {
        "name": "fetch_data",
        "description": "Fetch data from an API endpoint",
        "parameters": {
            "url": {
                "type": "string",
                "description": "The URL to fetch data from",
                "required": True
            },
            "format": {
                "type": "string",
                "description": "Response format (json/xml)",
                "required": False
            }
        }
    }


@pytest.fixture
def invalid_tool() -> Dict[str, Any]:
    """Fixture providing an invalid tool definition."""
    return {
        "name": "incomplete_tool",
        "parameters": {
            "query": {
                "type": "string"
            }
        }
    }


@pytest.fixture
def tool_keeper() -> ToolKeeper:
    """Fixture providing a ToolKeeper instance."""
    return ToolKeeper()


class TestToolKeeperUnit:
    """Unit tests for the ToolKeeper class."""

    @pytest.mark.asyncio
    async def test_validate_tool_valid(self, tool_keeper: ToolKeeper, valid_tool: Dict[str, Any]) -> None:
        """Test validation with a valid tool."""
        result = await tool_keeper.validate_tool(None, json.dumps(valid_tool))
        result_dict = json.loads(result)
        
        assert result_dict["is_valid"] is True
        assert not result_dict["errors"]
        
        # Check the direct method
        direct_result = await tool_keeper.validate_tool_directly(json.dumps(valid_tool))
        assert direct_result["is_valid"] is True

    @pytest.mark.asyncio
    async def test_validate_tool_invalid(self, tool_keeper: ToolKeeper, invalid_tool: Dict[str, Any]) -> None:
        """Test validation with an invalid tool."""
        result = await tool_keeper.validate_tool(None, json.dumps(invalid_tool))
        result_dict = json.loads(result)
        
        assert result_dict["is_valid"] is False
        assert len(result_dict["errors"]) > 0
        assert "description" in "".join(result_dict["errors"])

    @pytest.mark.asyncio
    async def test_analyze_tool(self, tool_keeper: ToolKeeper, valid_tool: Dict[str, Any]) -> None:
        """Test tool analysis functionality."""
        result = await tool_keeper.analyze_tool(None, json.dumps(valid_tool))
        result_dict = json.loads(result)
        
        assert "schema_check" in result_dict
        assert "docstring_check" in result_dict
        assert "error_handling" in result_dict
        assert "recommendations" in result_dict
        
        # Valid tool should have Valid schema check
        assert result_dict["schema_check"] == "Valid"

    @pytest.mark.asyncio
    async def test_document_tool(self, tool_keeper: ToolKeeper, valid_tool: Dict[str, Any]) -> None:
        """Test documentation generation."""
        result = await tool_keeper.document_tool(None, json.dumps(valid_tool))
        
        # Check that the documentation contains key elements
        assert valid_tool["name"] in result
        assert valid_tool["description"] in result
        assert "Parameters" in result
        assert "Usage Example" in result
        
        # Check that each parameter is documented
        for param_name in valid_tool["parameters"]:
            assert param_name in result

    @pytest.mark.asyncio
    async def test_json_error_handling(self, tool_keeper: ToolKeeper) -> None:
        """Test error handling with invalid JSON."""
        # Test with invalid JSON
        invalid_json = "{name: 'invalid json'}"
        
        # Validation should handle the error gracefully
        validation_result = await tool_keeper.validate_tool(None, invalid_json)
        validation_dict = json.loads(validation_result)
        
        assert validation_dict["is_valid"] is False
        assert "Invalid JSON format" in "".join(validation_dict["errors"])
        
        # Analysis should handle the error gracefully
        analysis_result = await tool_keeper.analyze_tool(None, invalid_json)
        analysis_dict = json.loads(analysis_result)
        
        assert "error" in analysis_dict
        assert "Invalid JSON format" in analysis_dict["error"]

    @pytest.mark.asyncio
    async def test_agent_integration(self, tool_keeper: ToolKeeper, valid_tool: Dict[str, Any]) -> None:
        """Test agent integration with mocked Runner."""
        with patch("tool_keeper.Runner") as mock_runner:
            # Setup mock runner
            mock_result = MagicMock()
            mock_result.final_output = "Mocked agent response"
            mock_runner.run = AsyncMock(return_value=mock_result)
            
            # Run the agent
            result = await tool_keeper.run("Test query")
            
            # Verify the agent was called
            mock_runner.run.assert_called_once()
            assert result == "Mocked agent response"


if __name__ == "__main__":
    pytest.main(["-v", __file__])