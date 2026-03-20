"""
Tests for Tool Use module.

Tests both the tool implementations (unit tests) and the
agentic loop logic (integration tests against the API).
"""

import os

import anthropic
import pytest
from dotenv import load_dotenv

from tools import TOOL_DEFINITIONS, execute_tool

load_dotenv()


# ---------------------------------------------------------------------------
# Unit tests — tool implementations (no API calls needed)
# ---------------------------------------------------------------------------
class TestToolImplementations:
    """Test that tools return correct results for known inputs."""

    def test_get_weather_known_city(self):
        result = execute_tool("get_weather", {"city": "Tokyo"})
        assert "22°C" in result
        assert "Partly Cloudy" in result

    def test_get_weather_fahrenheit(self):
        result = execute_tool("get_weather", {"city": "Tokyo", "unit": "fahrenheit"})
        assert "°F" in result
        # 22°C = 71.6°F
        assert "71.6" in result

    def test_get_weather_unknown_city(self):
        result = execute_tool("get_weather", {"city": "Atlantis"})
        assert "Weather in Atlantis" in result

    def test_search_orders_found(self):
        result = execute_tool("search_orders", {"email": "alice@example.com"})
        assert "ORD-1001" in result
        assert "Mechanical Keyboard" in result

    def test_search_orders_not_found(self):
        result = execute_tool("search_orders", {"email": "nobody@example.com"})
        assert "No orders found" in result

    def test_search_orders_with_filter(self):
        result = execute_tool(
            "search_orders",
            {"email": "bob@example.com", "status_filter": "pending"},
        )
        assert "Standing Desk" in result
        assert "Monitor Arm" not in result  # delivered, not pending

    def test_calculate_basic(self):
        result = execute_tool("calculate", {"expression": "2 + 2"})
        assert "4" in result

    def test_calculate_complex(self):
        result = execute_tool("calculate", {"expression": "(15 * 3) + 7.5"})
        assert "52.5" in result

    def test_calculate_with_functions(self):
        result = execute_tool("calculate", {"expression": "sqrt(144)"})
        assert "12" in result

    def test_calculate_rejects_unsafe(self):
        result = execute_tool("calculate", {"expression": "__import__('os').system('ls')"})
        assert "Error" in result or "Unsafe" in result

    def test_unknown_tool(self):
        result = execute_tool("nonexistent_tool", {})
        assert "Error" in result


# ---------------------------------------------------------------------------
# Tool definition validation
# ---------------------------------------------------------------------------
class TestToolDefinitions:
    """Validate tool schemas are well-formed."""

    def test_all_tools_have_required_fields(self):
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool {tool['name']} missing 'description'"
            assert "input_schema" in tool, f"Tool {tool['name']} missing 'input_schema'"

    def test_descriptions_are_descriptive(self):
        """Descriptions should be detailed enough for the model to choose correctly."""
        for tool in TOOL_DEFINITIONS:
            desc = tool["description"]
            assert len(desc) > 30, (
                f"Tool '{tool['name']}' description is too short ({len(desc)} chars). "
                "Short descriptions lead to poor tool selection."
            )

    def test_schemas_have_required_fields(self):
        for tool in TOOL_DEFINITIONS:
            schema = tool["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema


# ---------------------------------------------------------------------------
# Integration tests — require ANTHROPIC_API_KEY
# ---------------------------------------------------------------------------
@pytest.fixture
def client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


MODEL = "claude-sonnet-4-20250514"


class TestAgenticLoop:
    """Test the full agentic loop against the real API."""

    def _run_agentic_loop(self, client, user_message, max_iterations=5):
        """Helper: runs the same agentic loop as the /chat endpoint."""
        messages = [{"role": "user", "content": user_message}]
        tool_calls = []

        for _ in range(max_iterations):
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                text = "".join(
                    b.text for b in response.content if b.type == "text"
                )
                return {"text": text, "tool_calls": tool_calls}

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = execute_tool(block.name, block.input)
                        tool_calls.append({"tool": block.name, "input": block.input})
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                messages.append({"role": "user", "content": results})

        return {"text": "max iterations", "tool_calls": tool_calls}

    def test_weather_query_triggers_tool(self, client):
        """Model should call get_weather for weather questions."""
        result = self._run_agentic_loop(client, "What's the weather in Tokyo?")
        tool_names = [tc["tool"] for tc in result["tool_calls"]]
        assert "get_weather" in tool_names
        # Response should include actual weather data
        assert "22" in result["text"] or "Partly Cloudy" in result["text"]

    def test_order_query_triggers_tool(self, client):
        """Model should call search_orders for order lookups."""
        result = self._run_agentic_loop(
            client, "Can you check orders for alice@example.com?"
        )
        tool_names = [tc["tool"] for tc in result["tool_calls"]]
        assert "search_orders" in tool_names

    def test_math_query_triggers_tool(self, client):
        """Model should use calculate tool for arithmetic."""
        result = self._run_agentic_loop(
            client, "What is 1847 * 293 + 15?"
        )
        tool_names = [tc["tool"] for tc in result["tool_calls"]]
        assert "calculate" in tool_names

    def test_no_tool_needed(self, client):
        """Model should NOT use tools for simple chat."""
        result = self._run_agentic_loop(client, "Hello, how are you?")
        assert len(result["tool_calls"]) == 0

    def test_multi_tool_query(self, client):
        """Model should handle questions requiring multiple tools."""
        result = self._run_agentic_loop(
            client,
            "What's the weather in Tokyo and London? "
            "Also calculate 15 * 23.",
        )
        tool_names = [tc["tool"] for tc in result["tool_calls"]]
        assert "get_weather" in tool_names
        assert "calculate" in tool_names
