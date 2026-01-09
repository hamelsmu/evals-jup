"""Tests for tool loop termination and server-side tool execution.

Tests the max_steps limit and tool execution flow in PromptHandler.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class MockChoice:
    """Mock choice object for LiteLLM streaming."""

    def __init__(self, delta):
        self.delta = delta


class MockDelta:
    """Mock delta object for LiteLLM streaming events."""

    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class MockToolCall:
    """Mock tool call object."""

    def __init__(self, index, tool_id=None, function_name=None, arguments=None):
        self.index = index
        self.id = tool_id
        self.function = MagicMock()
        self.function.name = function_name
        self.function.arguments = arguments


class MockChunk:
    """Mock streaming chunk."""

    def __init__(self, delta):
        self.choices = [MockChoice(delta)]


class MockRequest:
    """Mock tornado request."""

    def __init__(self):
        self.connection = MagicMock()


class MockApplication:
    """Mock tornado application."""

    def __init__(self):
        self.ui_modules = {}
        self.ui_methods = {}


class MockHandler:
    """Mock handler with required tornado attributes."""

    def __init__(self):
        self.request = MockRequest()
        self.application = MockApplication()
        self._headers_written = False
        self._finished = False
        self._status_code = 200
        self._headers = {}
        self._buffer = []
        self.log = MagicMock()
        self.settings = {"base_url": "/"}
        self._json_body = {}
        self.current_user = "test_user"

    def set_header(self, name, value):
        self._headers[name] = value

    def set_status(self, code):
        self._status_code = code

    def write(self, data):
        if isinstance(data, dict):
            self._buffer.append(json.dumps(data))
        else:
            self._buffer.append(data)

    def finish(self, data=None):
        if data:
            self.write(data)
        self._finished = True

    async def flush(self):
        pass

    def get_json_body(self):
        return self._json_body


@pytest.fixture
def handler():
    """Create a mock handler with PromptHandler methods bound."""
    from jupyvibe.handlers import PromptHandler

    h = MockHandler()
    h._build_system_prompt = PromptHandler._build_system_prompt.__get__(h, MockHandler)
    h._build_tools = PromptHandler._build_tools.__get__(h, MockHandler)
    h._build_messages = PromptHandler._build_messages.__get__(h, MockHandler)
    h._python_type_to_json_schema = PromptHandler._python_type_to_json_schema.__get__(h, MockHandler)
    h._write_sse = PromptHandler._write_sse.__get__(h, MockHandler)
    h.post = PromptHandler.post.__get__(h, MockHandler)
    return h


class TestUnknownToolRejected:
    """Tests for unknown tool name rejection."""

    @pytest.mark.asyncio
    async def test_unknown_tool_rejected_with_error(self, handler):
        """Tool not in functions dict should produce error SSE event."""
        mock_kernel = MagicMock()
        mock_kernel_manager = MagicMock()
        mock_kernel_manager.get_kernel.return_value = mock_kernel
        handler.settings["kernel_manager"] = mock_kernel_manager

        handler._json_body = {
            "prompt": "test",
            "context": {"functions": {"calc": {"signature": "()", "docstring": "calc", "parameters": {}}}},
            "kernel_id": "k1",
            "max_steps": 5,
        }

        async def mock_stream():
            yield MockChunk(MockDelta(tool_calls=[
                MockToolCall(0, tool_id="tool_1", function_name="unknown_tool", arguments="{}")
            ]))
            yield MockChunk(MockDelta(content=None, tool_calls=None))

        mock_execute_tool = AsyncMock(return_value={"status": "success", "result": {"type": "text", "content": "42"}})

        with (
            patch("jupyvibe.handlers.HAS_LITELLM", True),
            patch("jupyvibe.handlers.litellm") as mock_litellm,
            patch("jupyvibe.handlers.PromptHandler._execute_tool_in_kernel", mock_execute_tool),
        ):
            mock_litellm.acompletion = AsyncMock(return_value=mock_stream())
            await handler.post()

        response = "".join(handler._buffer)
        assert "Unknown tool: unknown_tool" in response
        assert '{"done": true}' in response
        mock_execute_tool.assert_not_called()


class TestInvalidToolInputJSON:
    """Tests for invalid tool input JSON handling."""

    @pytest.mark.asyncio
    async def test_invalid_json_produces_error(self, handler):
        """Invalid JSON in tool input should produce error SSE event."""
        mock_kernel = MagicMock()
        mock_kernel_manager = MagicMock()
        mock_kernel_manager.get_kernel.return_value = mock_kernel
        handler.settings["kernel_manager"] = mock_kernel_manager

        handler._json_body = {
            "prompt": "test",
            "context": {"functions": {"calc": {"signature": "()", "docstring": "calc", "parameters": {}}}},
            "kernel_id": "k1",
            "max_steps": 5,
        }

        async def mock_stream():
            yield MockChunk(MockDelta(tool_calls=[
                MockToolCall(0, tool_id="tool_1", function_name="calc", arguments="not valid json")
            ]))
            yield MockChunk(MockDelta(content=None, tool_calls=None))

        mock_execute_tool = AsyncMock(return_value={"status": "success", "result": {"type": "text", "content": "42"}})

        with (
            patch("jupyvibe.handlers.HAS_LITELLM", True),
            patch("jupyvibe.handlers.litellm") as mock_litellm,
            patch("jupyvibe.handlers.PromptHandler._execute_tool_in_kernel", mock_execute_tool),
        ):
            mock_litellm.acompletion = AsyncMock(return_value=mock_stream())
            await handler.post()

        response = "".join(handler._buffer)
        assert "Invalid tool input JSON" in response
        assert '{"done": true}' in response
        mock_execute_tool.assert_not_called()


class TestInvalidToolArgumentName:
    """Tests for invalid tool argument names (prevents injection via kwargs)."""

    @pytest.mark.asyncio
    async def test_invalid_argument_key_produces_error(self, handler):
        mock_kernel = MagicMock()
        mock_kernel_manager = MagicMock()
        mock_kernel_manager.get_kernel.return_value = mock_kernel
        handler.settings["kernel_manager"] = mock_kernel_manager

        handler._json_body = {
            "prompt": "test",
            "context": {"functions": {"calc": {"signature": "()", "docstring": "calc", "parameters": {}}}},
            "kernel_id": "k1",
            "max_steps": 5,
        }

        tool_input = json.dumps({'x); __import__("os").system("echo injected"); #': 1})

        async def mock_stream():
            yield MockChunk(MockDelta(tool_calls=[
                MockToolCall(0, tool_id="tool_1", function_name="calc", arguments=tool_input)
            ]))
            yield MockChunk(MockDelta(content=None, tool_calls=None))

        mock_execute_tool = AsyncMock(
            return_value={"status": "success", "result": {"type": "text", "content": "42"}}
        )

        with (
            patch("jupyvibe.handlers.HAS_LITELLM", True),
            patch("jupyvibe.handlers.litellm") as mock_litellm,
            patch("jupyvibe.handlers.PromptHandler._execute_tool_in_kernel", mock_execute_tool),
        ):
            mock_litellm.acompletion = AsyncMock(return_value=mock_stream())
            await handler.post()

        response = "".join(handler._buffer)
        assert "Invalid tool argument name" in response
        assert '{"done": true}' in response
        mock_execute_tool.assert_not_called()
