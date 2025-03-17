#!/usr/bin/env python3

"""
Tool Keeper Unit Tests
=====================

Unit tests for the ToolKeeper class and its function tools.

To run:
    uv run pytest tests/test_tool_keeper_unit.py -v
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from tool_keeper import ToolKeeper, analyze_tool, validate_tool, document_tool


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


class TestToolFunctions:
    """Test the individual tool functions directly."""

    @pytest.mark.asyncio
    async def test_analyze_tool_valid(self, valid_tool: Dict[str, Any]) -> None:
        """Test analyzing a valid tool definition."""
        ctx_mock = MagicMock()
        
        result = await analyze_tool(ctx_mock, json.dumps(valid_tool))
        result_dict = json.loads(result)
        
        assert "schema_check" in result_dict
        assert result_dict["schema_check"] == "Valid"
        assert "docstring_check" in result_dict
        assert "error_handling" in result_dict
        assert "recommendations" in result_dict

    @pytest.mark.asyncio
    async def test_analyze_tool_invalid(self, invalid_tool: Dict[str, Any]) -> None:
        """Test analyzing an invalid tool definition."""
        ctx_mock = MagicMock()
        
        result = await analyze_tool(ctx_mock, json.dumps(invalid_tool))
        result_dict = json.loads(result)
        
        assert "schema_check" in result_dict
        assert result_dict["schema_check"] == "Invalid"
        assert len(result_dict["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_validate_tool_valid(self, valid_tool: Dict[str, Any]) -> None:
        """Test validating a valid tool definition."""
        ctx_mock = MagicMock()
        
        result = await validate_tool(ctx_mock, json.dumps(valid_tool))
        result_dict = json.loads(result)
        
        assert "is_valid" in result_dict
        assert result_dict["is_valid"] is True
        assert "errors" in result_dict
        assert len(result_dict["errors"]) == 0

    @pytest.mark.asyncio
    async def test_validate_tool_invalid(self, invalid_tool: Dict[str, Any]) -> None:
        """Test validating an invalid tool definition."""
        ctx_mock = MagicMock()
        
        result = await validate_tool(ctx_mock, json.dumps(invalid_tool))
        result_dict = json.loads(result)
        
        assert "is_valid" in result_dict
        assert result_dict["is_valid"] is False
        assert "errors" in result_dict
        assert len(result_dict["errors"]) > 0
        assert "description" in "".join(result_dict["errors"])

    @pytest.mark.asyncio
    async def test_document_tool(self, valid_tool: Dict[str, Any]) -> None:
        """Test documenting a tool definition."""
        ctx_mock = MagicMock()
        
        result = await document_tool(ctx_mock, json.dumps(valid_tool))
        
        assert valid_tool["name"] in result
        assert valid_tool["description"] in result
        assert "Parameters" in result
        assert "Usage Example" in result
        
        # Verify each parameter is documented
        for param_name in valid_tool["parameters"]:
            assert param_name in result
    
    @pytest.mark.asyncio
    async def test_error_handling(self) -> None:
        """Test error handling in tool functions."""
        ctx_mock = MagicMock()
        invalid_json = "{name: 'invalid json'}"
        
        # Test analyze_tool
        result = await analyze_tool(ctx_mock, invalid_json)
        result_dict = json.loads(result)
        assert "error" in result_dict
        assert "Invalid JSON format" in result_dict["error"]
        
        # Test validate_tool
        result = await validate_tool(ctx_mock, invalid_json)
        result_dict = json.loads(result)
        assert "is_valid" in result_dict
        assert result_dict["is_valid"] is False
        assert "Invalid JSON format" in "".join(result_dict["errors"])
        
        # Test document_tool
        result = await document_tool(ctx_mock, invalid_json)
        assert "Error" in result


class TestToolKeeper:
    """Test the ToolKeeper class."""

    @pytest.mark.asyncio
    async def test_run_method(self, tool_keeper: ToolKeeper) -> None:
        """Test the run method of ToolKeeper."""
        with patch("tool_keeper.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "Analysis complete!"
            mock_runner.run = AsyncMock(return_value=mock_result)
            
            result = await tool_keeper.run("Analyze this tool")
            
            # Verify Runner.run was called with the right agent
            mock_runner.run.assert_called_once_with(tool_keeper.agent, "Analyze this tool", context=None)
            assert result == "Analysis complete!"
    
    @pytest.mark.asyncio
    async def test_run_with_context(self, tool_keeper: ToolKeeper) -> None:
        """Test running with custom context."""
        with patch("tool_keeper.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "Analysis with context!"
            mock_runner.run = AsyncMock(return_value=mock_result)
            
            custom_context = {"user_id": "123", "settings": {"verbose": True}}
            result = await tool_keeper.run("Analyze this tool", context=custom_context)
            
            # Verify context was passed to Runner.run
            mock_runner.run.assert_called_once_with(tool_keeper.agent, "Analyze this tool", context=custom_context)
            assert result == "Analysis with context!"


if __name__ == "__main__":
    pytest.main(["-v", __file__])