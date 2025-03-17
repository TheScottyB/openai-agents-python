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


@pytest.fixture
def mock_runner_result():
    """Fixture providing a mock runner result."""
    mock_result = MagicMock()
    mock_result.final_output = json.dumps({
        "schema_check": "Valid",
        "docstring_check": "Present",
        "error_handling": "Missing",
        "recommendations": ["Implement error handling with failure_error_function"]
    })
    return mock_result


class TestToolKeeperUnit:
    """Unit tests for the ToolKeeper class."""

    @pytest.mark.asyncio
    async def test_validate_tool_valid(self, tool_keeper: ToolKeeper, valid_tool: Dict[str, Any], mock_runner_result) -> None:
        """Test validation with a valid tool."""
        with patch("tool_keeper.Runner") as mock_runner:
            # Set up the mock
            mock_runner_result.final_output = json.dumps({
                "is_valid": True,
                "errors": [],
                "warnings": []
            })
            mock_runner.run = AsyncMock(return_value=mock_runner_result)
            
            # Call the method
            result = await tool_keeper.validate_tool_directly(json.dumps(valid_tool))
            
            # Verify results
            assert result["is_valid"] is True
            assert not result["errors"]
            
            # Verify that Runner.run was called
            mock_runner.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_tool_invalid(self, tool_keeper: ToolKeeper, invalid_tool: Dict[str, Any], mock_runner_result) -> None:
        """Test validation with an invalid tool."""
        with patch("tool_keeper.Runner") as mock_runner:
            # Set up the mock
            mock_runner_result.final_output = json.dumps({
                "is_valid": False,
                "errors": ["Missing required field: description"],
                "warnings": ["Parameter 'query' is missing description"]
            })
            mock_runner.run = AsyncMock(return_value=mock_runner_result)
            
            # Call the method
            result = await tool_keeper.validate_tool_directly(json.dumps(invalid_tool))
            
            # Verify results
            assert result["is_valid"] is False
            assert len(result["errors"]) > 0
            assert "description" in result["errors"][0]
            
            # Verify that Runner.run was called
            mock_runner.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_tool(self, tool_keeper: ToolKeeper, valid_tool: Dict[str, Any], mock_runner_result) -> None:
        """Test tool analysis functionality."""
        with patch("tool_keeper.Runner") as mock_runner:
            # Set up the mock
            mock_runner.run = AsyncMock(return_value=mock_runner_result)
            
            # Call the method
            result = await tool_keeper.analyze_tool_directly(json.dumps(valid_tool))
            
            # Verify results
            assert "schema_check" in result
            assert "docstring_check" in result
            assert "error_handling" in result
            assert "recommendations" in result
            
            # Verify that Runner.run was called
            mock_runner.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_document_tool(self, tool_keeper: ToolKeeper, valid_tool: Dict[str, Any]) -> None:
        """Test documentation generation."""
        with patch("tool_keeper.Runner") as mock_runner:
            # Set up the mock
            mock_result = MagicMock()
            mock_result.final_output = f"# {valid_tool['name']}\n\n{valid_tool['description']}"
            mock_runner.run = AsyncMock(return_value=mock_result)
            
            # Call the method
            result = await tool_keeper.document_tool_directly(json.dumps(valid_tool))
            
            # Verify results
            assert valid_tool["name"] in result
            assert valid_tool["description"] in result
            
            # Verify that Runner.run was called
            mock_runner.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_json_extraction(self, tool_keeper: ToolKeeper, valid_tool: Dict[str, Any]) -> None:
        """Test JSON extraction from agent responses."""
        with patch("tool_keeper.Runner") as mock_runner:
            # Set up the mock - agent response with text before and after JSON
            mock_result = MagicMock()
            mock_result.final_output = f"""
            Here is the analysis of the tool:
            
            {json.dumps({
                "schema_check": "Valid",
                "docstring_check": "Present",
                "error_handling": "Missing",
                "recommendations": ["Add error handling"]
            })}
            
            Let me know if you need anything else!
            """
            mock_runner.run = AsyncMock(return_value=mock_result)
            
            # Call the method
            result = await tool_keeper.analyze_tool_directly(json.dumps(valid_tool))
            
            # Verify JSON was correctly extracted
            assert "schema_check" in result
            assert result["schema_check"] == "Valid"
            assert "recommendations" in result
            assert "Add error handling" in result["recommendations"]

    @pytest.mark.asyncio
    async def test_fallback_to_direct_call(self, tool_keeper: ToolKeeper, valid_tool: Dict[str, Any]) -> None:
        """Test fallback to direct tool call when JSON extraction fails."""
        with patch("tool_keeper.Runner") as mock_runner:
            # Set up the mock - agent response with no JSON
            mock_result = MagicMock()
            mock_result.final_output = "I've analyzed the tool but couldn't format the output as JSON."
            mock_runner.run = AsyncMock(return_value=mock_result)
            
            # Set up the direct tool call mock
            with patch.object(tool_keeper, '_analyze_tool') as mock_analyze:
                mock_analyze.return_value = json.dumps({
                    "schema_check": "Valid",
                    "docstring_check": "Present",
                    "error_handling": "Missing",
                    "recommendations": []
                })
                
                # Call the method
                result = await tool_keeper.analyze_tool_directly(json.dumps(valid_tool))
                
                # Verify direct tool call was used
                assert "schema_check" in result
                mock_analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_agent_run(self, tool_keeper: ToolKeeper) -> None:
        """Test the main agent run method."""
        with patch("tool_keeper.Runner") as mock_runner:
            # Set up the mock
            mock_result = MagicMock()
            mock_result.final_output = "Analysis complete: The tool looks good!"
            mock_runner.run = AsyncMock(return_value=mock_result)
            
            # Call the method
            result = await tool_keeper.run("Analyze this tool")
            
            # Verify results
            assert result == "Analysis complete: The tool looks good!"
            
            # Verify that Runner.run was called with the main agent
            mock_runner.run.assert_called_once_with(tool_keeper.agent, "Analyze this tool")


if __name__ == "__main__":
    pytest.main(["-v", __file__])