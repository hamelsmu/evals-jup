"""Tests for the server handlers."""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_build_system_prompt():
    """Test the system prompt building logic."""
    from jupyvibe.handlers import PromptHandler
    
    class MockHandler:
        pass
    
    handler = MockHandler()
    handler._build_system_prompt = PromptHandler._build_system_prompt.__get__(handler, MockHandler)
    
    # Test with empty context - returns tuple of (string, image_content_list)
    result = handler._build_system_prompt("", {}, {})
    assert isinstance(result, tuple)
    assert len(result) == 2
    prompt, images = result
    assert isinstance(prompt, str)
    assert isinstance(images, list)
    assert "AI assistant" in prompt
    assert "Jupyter notebook" in prompt
    assert len(images) == 0
    print("✓ Basic system prompt works")
    
    # Test with preceding code
    prompt, _ = handler._build_system_prompt("import pandas as pd\nx = 5", {}, {})
    assert "Preceding Code" in prompt
    assert "import pandas as pd" in prompt
    print("✓ System prompt with preceding code works")
    
    # Test with variables
    variables = {
        "df": {"type": "DataFrame", "repr": "   A  B\n0  1  2"},
        "x": {"type": "int", "repr": "42"}
    }
    prompt, _ = handler._build_system_prompt("", variables, {})
    assert "Available Variables" in prompt
    assert "df" in prompt
    assert "DataFrame" in prompt
    print("✓ System prompt with variables works")
    
    # Test with functions
    functions = {
        "calculate": {
            "signature": "(x: int, y: int) -> int",
            "docstring": "Add two numbers",
            "parameters": {"x": {"type": "int"}, "y": {"type": "int"}}
        }
    }
    prompt, _ = handler._build_system_prompt("", {}, functions)
    assert "Available Functions" in prompt
    assert "calculate" in prompt
    print("✓ System prompt with functions works")
    
    # Test with images - should return image content for user message
    images_input = [
        {"data": "base64data", "mimeType": "image/png", "source": "output", "cellIndex": 0}
    ]
    prompt, image_content = handler._build_system_prompt("", {}, {}, images=images_input)
    assert "1 image(s)" in prompt
    assert len(image_content) == 2  # text description + image
    assert image_content[1]["type"] == "image_url"
    assert "data:image/png;base64,base64data" in image_content[1]["image_url"]["url"]
    print("✓ System prompt with images works (OpenAI format)")


def test_build_tools():
    """Test the tool building logic (OpenAI format)."""
    from jupyvibe.handlers import PromptHandler
    
    class MockHandler:
        pass
    
    handler = MockHandler()
    handler._build_tools = PromptHandler._build_tools.__get__(handler, MockHandler)
    handler._python_type_to_json_schema = PromptHandler._python_type_to_json_schema.__get__(handler, MockHandler)
    
    # Test with empty functions
    tools = handler._build_tools({})
    assert tools == []
    print("✓ Empty tools works")
    
    # Test with functions - should use OpenAI format
    functions = {
        "calculate": {
            "signature": "(x: int, y: int) -> int",
            "docstring": "Add two numbers",
            "parameters": {"x": {"type": "int"}, "y": {"type": "int"}}
        }
    }
    tools = handler._build_tools(functions)
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "calculate"
    assert "parameters" in tools[0]["function"]
    print("✓ Tool building works (OpenAI format)")


def test_build_messages():
    """Test the conversation history message building logic."""
    from jupyvibe.handlers import PromptHandler
    
    class MockHandler:
        pass
    
    handler = MockHandler()
    handler._build_messages = PromptHandler._build_messages.__get__(handler, MockHandler)
    
    # Test with empty history - includes system message
    system_prompt = ("You are an assistant.", [])
    messages = handler._build_messages([], "Hello", system_prompt)
    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": "You are an assistant."}
    assert messages[1] == {"role": "user", "content": "Hello"}
    print("✓ Empty conversation history works")
    
    # Test with single turn history
    history = [{"prompt": "Write a poem", "response": "Roses are red..."}]
    messages = handler._build_messages(history, "Continue the poem", system_prompt)
    assert len(messages) == 4  # system + history (2) + current
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Write a poem"}
    assert messages[2] == {"role": "assistant", "content": "Roses are red..."}
    assert messages[3] == {"role": "user", "content": "Continue the poem"}
    print("✓ Single turn conversation history works")
    
    # Test with multiple turn history
    history = [
        {"prompt": "Hello", "response": "Hi there!"},
        {"prompt": "How are you?", "response": "I'm doing well, thanks!"}
    ]
    messages = handler._build_messages(history, "That's great", system_prompt)
    assert len(messages) == 6  # system + 4 history + current
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[3]["role"] == "user"
    assert messages[4]["role"] == "assistant"
    assert messages[5] == {"role": "user", "content": "That's great"}
    print("✓ Multiple turn conversation history works")
    
    # Test with empty prompt/response in history (should skip empty)
    history = [{"prompt": "", "response": ""}]
    messages = handler._build_messages(history, "Hello", system_prompt)
    assert len(messages) == 2  # system + current
    assert messages[1] == {"role": "user", "content": "Hello"}
    print("✓ Empty turns are skipped")
    
    # Test with images in system prompt
    system_prompt_with_images = ("You are an assistant.", [
        {"type": "text", "text": "## Image 1"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
    ])
    messages = handler._build_messages([], "Describe this", system_prompt_with_images)
    assert len(messages) == 2  # system + user with images
    # User message should have content as list with images + text
    assert isinstance(messages[1]["content"], list)
    assert len(messages[1]["content"]) == 3  # 2 image items + prompt text
    assert messages[1]["content"][2] == {"type": "text", "text": "Describe this"}
    print("✓ Messages with images work (OpenAI multimodal format)")


def test_provider_display_names():
    """Test provider display names (fast, no model iteration)."""
    from jupyvibe.models import get_provider_display_names
    
    providers = get_provider_display_names()
    assert providers == {
        'openai': 'OpenAI',
        'anthropic': 'Anthropic',
        'gemini': 'Google (Gemini)',
    }
    print("✓ get_provider_display_names works")


if __name__ == '__main__':
    test_build_system_prompt()
    test_build_tools()
    test_build_messages()
    test_provider_display_names()
    print("\n✅ All handler tests passed!")
