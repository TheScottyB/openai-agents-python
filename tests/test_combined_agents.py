#!/usr/bin/env python3

"""
Test suite for combined_agents_demo.py

Shows best practices for testing OpenAI Agents SDK implementations
including mocking, fixture usage, and testing guardrails.
"""

import asyncio
import pytest
from datetime import datetime
from typing import Any, Dict, cast
from unittest.mock import AsyncMock, MagicMock, patch

# Import from the code being tested
from combined_agents_demo import (
    UserType,
    CustomerContext,
    CalculationResult,
    CustomerResponse,
    ConversionError,
    convert_temperature,
    convert_length,
    get_math_constant,
    get_user_details,
    math_query_guardrail,
    response_quality_guardrail,
    create_calculator_agent,
    create_customer_service_agent,
)

from agents import (
    Agent, 
    RunContextWrapper,
    GuardrailFunctionOutput,
)
from agents.exceptions import ModelBehaviorError

# pytest fixtures
@pytest.fixture
def regular_user_context() -> CustomerContext:
    """Test fixture providing a regular user context."""
    return CustomerContext(
        customer_id="test_user_123",
        username="TestUser",
        user_type=UserType.REGULAR
    )

@pytest.fixture
def premium_user_context() -> CustomerContext:
    """Test fixture providing a premium user context."""
    return CustomerContext(
        customer_id="test_premium_456",
        username="PremiumUser",
        user_type=UserType.PREMIUM
    )

@pytest.fixture
def wrapped_context(regular_user_context: CustomerContext) -> RunContextWrapper[CustomerContext]:
    """Test fixture providing a wrapped context for function tools."""
    mock_wrapper = MagicMock(spec=RunContextWrapper)
    mock_wrapper.context = regular_user_context
    return cast(RunContextWrapper[CustomerContext], mock_wrapper)

# Test core functionality
class TestUnitConversions:
    """Test unit conversion functions."""
    
    async def test_convert_temperature_valid(self, wrapped_context: RunContextWrapper[CustomerContext]) -> None:
        """Test valid temperature conversion."""
        # Test C to F conversion
        result = await convert_temperature(wrapped_context, 100.0, "Celsius", "Fahrenheit")
        assert result["success"] is True
        assert result["original_value"] == 100.0
        assert result["original_unit"] == "Celsius"
        assert abs(result["converted_value"] - 212.0) < 0.01  # Allow small floating point error
        
        # Test F to K conversion
        result = await convert_temperature(wrapped_context, 32.0, "f", "k")
        assert result["success"] is True
        assert abs(result["converted_value"] - 273.15) < 0.01
    
    async def test_convert_temperature_invalid(self, wrapped_context: RunContextWrapper[CustomerContext]) -> None:
        """Test error handling for invalid temperature conversion."""
        # Test invalid units
        result = await convert_temperature(wrapped_context, 100.0, "Celsius", "Miles")
        assert result["success"] is False
        assert "Invalid temperature units" in result["error"]
    
    async def test_convert_length_valid(self, wrapped_context: RunContextWrapper[CustomerContext]) -> None:
        """Test valid length conversion."""
        result = await convert_length(wrapped_context, 1.0, "m", "ft")
        assert result["success"] is True
        assert abs(result["converted_value"] - 3.28084) < 0.01
    
    async def test_get_math_constant_valid(self, wrapped_context: RunContextWrapper[CustomerContext]) -> None:
        """Test retrieving a valid math constant."""
        result = await get_math_constant(wrapped_context, "pi")
        assert result["success"] is True
        assert abs(result["value"] - 3.14159265359) < 0.00001
        assert "circle's circumference" in result["description"]
    
    async def test_get_math_constant_invalid(self, wrapped_context: RunContextWrapper[CustomerContext]) -> None:
        """Test error handling for invalid math constant."""
        result = await get_math_constant(wrapped_context, "not_a_constant")
        assert result["success"] is False
        assert "Unknown constant" in result["error"]

class TestUserContext:
    """Test the CustomerContext class."""
    
    def test_premium_status(self) -> None:
        """Test checking premium status based on user type."""
        regular = CustomerContext("test1", "user1", UserType.REGULAR)
        premium = CustomerContext("test2", "user2", UserType.PREMIUM)
        admin = CustomerContext("test3", "user3", UserType.ADMIN)
        
        assert regular.premium_user is False
        assert premium.premium_user is True
        assert admin.premium_user is True
    
    def test_log_interaction(self, regular_user_context: CustomerContext) -> None:
        """Test logging tool interactions."""
        regular_user_context.log_interaction("test_tool", {"input": "test"}, {"output": "result"})
        
        assert len(regular_user_context.conversation_history) == 1
        assert regular_user_context.conversation_history[0]["tool"] == "test_tool"
        assert regular_user_context.conversation_history[0]["input"] == {"input": "test"}

# Test guardrails with mocks
class TestGuardrails:
    """Test guardrails with mocked Runner to avoid actual model calls."""
    
    @pytest.mark.asyncio
    async def test_math_query_guardrail(self) -> None:
        """Test math query input guardrail."""
        ctx_mock = MagicMock()
        agent_mock = MagicMock()
        
        # Mock the Runner
        with patch("combined_agents_demo.Runner") as mock_runner:
            # Set up mock result
            result_mock = MagicMock()
            result_mock.final_output_as.return_value = MagicMock(is_math_query=True, reasoning="It's a math query")
            mock_runner.run = AsyncMock(return_value=result_mock)
            
            # Test with a math query that's too short
            output = await math_query_guardrail(ctx_mock, agent_mock, "convert 5")
            
            # Verify the guardrail triggered
            assert output.tripwire_triggered is True
    
    @pytest.mark.asyncio
    async def test_response_quality_guardrail(self) -> None:
        """Test response quality output guardrail."""
        ctx_mock = MagicMock()
        agent_mock = MagicMock()
        response = CustomerResponse(
            answer="I'll convert that for you: 100C = 212F",
            sentiment="positive",
            requires_escalation=False,
            calculation_result=CalculationResult(
                result="212 degrees Fahrenheit",
                units="°F",
                explanation="Multiply by 9/5 and add 32"
            )
        )
        
        # Mock the Runner
        with patch("combined_agents_demo.Runner") as mock_runner:
            # Set up mock result
            result_mock = MagicMock()
            result_mock.final_output_as.return_value = MagicMock(
                is_appropriate=True, 
                reasoning="Good response"
            )
            mock_runner.run = AsyncMock(return_value=result_mock)
            
            # Test with a valid response
            output = await response_quality_guardrail(ctx_mock, agent_mock, response)
            
            # Verify the guardrail didn't trigger
            assert output.tripwire_triggered is False

# Test agent creation
class TestAgentCreation:
    """Test agent creation and configuration."""
    
    def test_calculator_agent_creation(self) -> None:
        """Test creating a calculator agent."""
        agent = create_calculator_agent()
        
        assert agent.name == "Calculator Agent"
        assert agent.model == "gpt-4o"
        assert agent.model_settings.temperature == 0.1
        assert len(agent.tools) == 4
        assert agent.output_type == CalculationResult
    
    def test_customer_service_agent_creation(self) -> None:
        """Test creating a customer service agent."""
        agent = create_customer_service_agent()
        
        assert agent.name == "Customer Service Agent"
        assert agent.model == "gpt-4o"
        assert agent.model_settings.temperature == 0.3
        assert len(agent.tools) == 5  # 4 regular tools + 1 calculator tool
        assert agent.output_type == CustomerResponse
        assert len(agent.handoffs) == 1  # Has calculator agent as handoff

# Integration-style tests that would require actual model calls
# These would be marked to skip unless explicitly enabled
@pytest.mark.skip(reason="Integration test requiring actual model calls")
class TestIntegration:
    """Integration tests that would call actual OpenAI models."""
    
    @pytest.mark.asyncio
    async def test_full_agent_workflow(self, regular_user_context: CustomerContext) -> None:
        """Test a full agent workflow with a real model."""
        from agents import Runner
        
        agent = create_customer_service_agent()
        result = await Runner.run(
            agent, 
            "What is the value of Pi?", 
            context=regular_user_context
        )
        
        response = result.final_output_as(CustomerResponse)
        assert response.answer is not None
        assert len(response.answer) > 0
        assert "3.14" in response.answer