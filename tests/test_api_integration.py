"""Integration tests for the backend API endpoints.

Tests the PromptHandler in jupyvibe/handlers.py with mocked LiteLLM client.
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


class TestMissingLiteLLMPackage:
    """Tests for missing litellm package."""

    @pytest.mark.asyncio
    async def test_missing_litellm_package_returns_500(self, handler):
        """When HAS_LITELLM is False, should return HTTP 500 with error JSON."""
        handler._json_body = {"prompt": "test", "context": {}}

        with patch("jupyvibe.handlers.HAS_LITELLM", False):
            await handler.post()

        assert handler._status_code == 500
        assert handler._finished
        response = json.loads(handler._buffer[0])
        assert "error" in response
        assert "litellm" in response["error"].lower()


class TestStreamingSSE:
    """Tests for streaming SSE responses."""

    @pytest.mark.asyncio
    async def test_streaming_sse_with_mock(self, handler):
        """Mock LiteLLM client should produce multiple data: lines with final done."""
        handler._json_body = {"prompt": "hi", "context": {}}

        async def mock_stream():
            yield MockChunk(MockDelta(content="Hello "))
            yield MockChunk(MockDelta(content="world"))

        with patch("jupyvibe.handlers.HAS_LITELLM", True):
            with patch("jupyvibe.handlers.litellm") as mock_litellm:
                mock_litellm.acompletion = AsyncMock(return_value=mock_stream())
                await handler.post()

        response = "".join(handler._buffer)
        assert 'data: {"text": "Hello "}' in response
        assert 'data: {"text": "world"}' in response
        assert 'data: {"done": true}' in response


class TestModelParameter:
    """Tests for model parameter handling."""

    @pytest.mark.asyncio
    async def test_model_parameter_passed_to_client(self, handler):
        """Specified model should be passed to LiteLLM client."""
        handler._json_body = {
            "prompt": "test",
            "context": {},
            "model": "gpt-4o",
        }

        async def mock_stream():
            yield MockChunk(MockDelta(content="Done"))

        with patch("jupyvibe.handlers.HAS_LITELLM", True):
            with patch("jupyvibe.handlers.litellm") as mock_litellm:
                mock_litellm.acompletion = AsyncMock(return_value=mock_stream())
                await handler.post()

                call_kwargs = mock_litellm.acompletion.call_args.kwargs
                assert call_kwargs["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_default_model_used_when_omitted(self, handler):
        """Default model should be used when not specified in request."""
        handler._json_body = {"prompt": "test", "context": {}}

        async def mock_stream():
            yield MockChunk(MockDelta(content="Done"))

        with patch("jupyvibe.handlers.HAS_LITELLM", True):
            with patch("jupyvibe.handlers.litellm") as mock_litellm:
                mock_litellm.acompletion = AsyncMock(return_value=mock_stream())
                await handler.post()

                call_kwargs = mock_litellm.acompletion.call_args.kwargs
                assert call_kwargs["model"] == "claude-sonnet-4-20250514"


class TestSystemPromptConstruction:
    """Tests for system prompt construction."""

    @pytest.mark.asyncio
    async def test_system_prompt_contains_instructions(self, handler):
        """System prompt should contain assistant instructions."""
        handler._json_body = {"prompt": "test", "context": {}}

        async def mock_stream():
            yield MockChunk(MockDelta(content="Done"))

        with patch("jupyvibe.handlers.HAS_LITELLM", True):
            with patch("jupyvibe.handlers.litellm") as mock_litellm:
                mock_litellm.acompletion = AsyncMock(return_value=mock_stream())
                await handler.post()

                call_kwargs = mock_litellm.acompletion.call_args.kwargs
                messages = call_kwargs["messages"]
                system_msg = next((m for m in messages if m["role"] == "system"), None)
                assert system_msg is not None
                assert "AI assistant" in system_msg["content"]


class TestToolCalling:
    """Tests for tool calling with LiteLLM."""

    @pytest.fixture
    def handler_with_kernel(self, handler):
        """Handler with a mock kernel manager."""
        mock_kernel = MagicMock()
        mock_kernel_manager = MagicMock()
        mock_kernel_manager.get_kernel.return_value = mock_kernel
        handler.settings["kernel_manager"] = mock_kernel_manager
        return handler

    @pytest.mark.asyncio
    async def test_tool_call_streamed(self, handler_with_kernel):
        """Tool calls should be streamed to the client."""
        handler = handler_with_kernel

        handler._json_body = {
            "prompt": "calculate 2+2",
            "context": {
                "functions": {
                    "calculate": {
                        "signature": "(x: int) -> int",
                        "docstring": "Calculate",
                        "parameters": {"x": {"type": "int"}}
                    }
                }
            },
            "kernel_id": "test-kernel",
            "max_steps": 0,  # Don't loop
        }

        async def mock_stream():
            # First chunk: tool call starts
            tc = MockToolCall(0, "call_123", "calculate", '{"x": 4}')
            yield MockChunk(MockDelta(tool_calls=[tc]))

        with patch("jupyvibe.handlers.HAS_LITELLM", True):
            with patch("jupyvibe.handlers.litellm") as mock_litellm:
                mock_litellm.acompletion = AsyncMock(return_value=mock_stream())
                await handler.post()

        response = "".join(handler._buffer)
        assert "tool_call" in response
        assert "calculate" in response


class TestBadInput:
    """Tests for bad input handling."""

    @pytest.mark.asyncio
    async def test_none_json_body_treated_as_empty(self, handler):
        """When JSON body is None, should be treated as empty dict."""
        handler.get_json_body = lambda: None

        async def mock_stream():
            yield MockChunk(MockDelta(content="Hello"))

        with patch("jupyvibe.handlers.HAS_LITELLM", True):
            with patch("jupyvibe.handlers.litellm") as mock_litellm:
                mock_litellm.acompletion = AsyncMock(return_value=mock_stream())
                await handler.post()

        assert handler._status_code == 200

    @pytest.mark.asyncio
    async def test_list_json_body_returns_400(self, handler):
        """When JSON body is a list instead of dict, should return HTTP 400."""
        handler.get_json_body = lambda: ["prompt", "test"]

        await handler.post()

        assert handler._status_code == 400
        assert handler._finished
        response = json.loads(handler._buffer[0])
        assert "Invalid JSON body" in response["error"]


class TestStreamClosedError:
    """Tests for StreamClosedError handling."""

    @pytest.mark.asyncio
    async def test_stream_closed_during_write_handled_gracefully(self, handler):
        """StreamClosedError during SSE write should not cause unhandled exception."""
        from tornado.iostream import StreamClosedError

        handler._json_body = {"prompt": "test", "context": {}}

        async def failing_flush():
            raise StreamClosedError()

        handler.flush = failing_flush

        async def mock_stream():
            yield MockChunk(MockDelta(content="Hello"))

        with patch("jupyvibe.handlers.HAS_LITELLM", True):
            with patch("jupyvibe.handlers.litellm") as mock_litellm:
                mock_litellm.acompletion = AsyncMock(return_value=mock_stream())
                # Should not raise - StreamClosedError should be caught
                await handler.post()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
