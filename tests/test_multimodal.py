"""Tests for multimodal image support in prompts (OpenAI/LiteLLM format)."""
import json
import sys
import os
import base64

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_build_system_prompt_with_images():
    """Test system prompt building with image context (OpenAI format)."""
    from jupyvibe.handlers import PromptHandler
    
    class MockHandler:
        pass
    
    handler = MockHandler()
    handler._build_system_prompt = PromptHandler._build_system_prompt.__get__(handler, MockHandler)
    
    images = [
        {
            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ",
            "mimeType": "image/png",
            "source": "output",
            "cellIndex": 0
        }
    ]
    
    system_text, image_content = handler._build_system_prompt("", {}, {}, images)
    
    # Should contain mention of images in system text
    assert "1 image(s)" in system_text
    assert "outputs from" in system_text.lower() or "cell output" in system_text.lower()
    print("✓ System prompt mentions images when present")
    
    # Image content should be in OpenAI format (for user message)
    assert len(image_content) == 2  # text description + image_url
    assert image_content[0]["type"] == "text"
    assert "cell output" in image_content[0]["text"].lower()
    assert image_content[1]["type"] == "image_url"
    assert "data:image/png;base64," in image_content[1]["image_url"]["url"]
    print("✓ Image block correctly formatted for OpenAI/LiteLLM API")


def test_build_system_prompt_with_multiple_images():
    """Test system prompt with multiple images from different sources."""
    from jupyvibe.handlers import PromptHandler
    
    class MockHandler:
        pass
    
    handler = MockHandler()
    handler._build_system_prompt = PromptHandler._build_system_prompt.__get__(handler, MockHandler)
    
    images = [
        {"data": "png_data_1", "mimeType": "image/png", "source": "output", "cellIndex": 0},
        {"data": "jpeg_data", "mimeType": "image/jpeg", "source": "attachment", "cellIndex": 1},
        {"data": "png_data_2", "mimeType": "image/png", "source": "output", "cellIndex": 2}
    ]
    
    system_text, image_content = handler._build_system_prompt("", {}, {}, images)
    
    # Should mention 3 images
    assert "3 image(s)" in system_text
    print("✓ System prompt correctly counts multiple images")
    
    # Should have 6 content items (3 descriptions + 3 images)
    assert len(image_content) == 6
    
    # Check structure: alternating text and image_url
    assert image_content[0]["type"] == "text"
    assert image_content[1]["type"] == "image_url"
    assert "data:image/png;base64,png_data_1" in image_content[1]["image_url"]["url"]
    
    assert image_content[2]["type"] == "text"
    assert image_content[3]["type"] == "image_url"
    assert "data:image/jpeg;base64,jpeg_data" in image_content[3]["image_url"]["url"]
    
    assert image_content[4]["type"] == "text"
    assert image_content[5]["type"] == "image_url"
    assert "data:image/png;base64,png_data_2" in image_content[5]["image_url"]["url"]
    print("✓ Multiple images correctly formatted")


def test_build_system_prompt_image_descriptions():
    """Test that image content has descriptions with source and cell info."""
    from jupyvibe.handlers import PromptHandler
    
    class MockHandler:
        pass
    
    handler = MockHandler()
    handler._build_system_prompt = PromptHandler._build_system_prompt.__get__(handler, MockHandler)
    
    images = [
        {"data": "output_image", "mimeType": "image/png", "source": "output", "cellIndex": 5},
        {"data": "attachment_image", "mimeType": "image/jpeg", "source": "attachment", "cellIndex": 3}
    ]
    
    _, image_content = handler._build_system_prompt("", {}, {}, images)
    
    # Get the text descriptions
    text_items = [item for item in image_content if item["type"] == "text"]
    all_text = " ".join(item["text"] for item in text_items)
    
    # Check for image descriptions
    assert "cell output" in all_text.lower()
    assert "5" in all_text
    assert "markdown attachment" in all_text.lower() or "attachment" in all_text.lower()
    assert "3" in all_text
    print("✓ Image descriptions include source type and cell index")


def test_build_system_prompt_no_images():
    """Test that system prompt works normally without images."""
    from jupyvibe.handlers import PromptHandler
    
    class MockHandler:
        pass
    
    handler = MockHandler()
    handler._build_system_prompt = PromptHandler._build_system_prompt.__get__(handler, MockHandler)
    
    # No images passed
    system_text, image_content = handler._build_system_prompt("import pandas", {"x": {"type": "int", "repr": "5"}}, {}, [])
    
    # Should not mention images
    assert "image(s)" not in system_text.lower()
    
    # Should have normal content
    assert "import pandas" in system_text
    assert "Available Variables" in system_text
    print("✓ System prompt works without images")
    
    # Should have no image content
    assert len(image_content) == 0
    print("✓ No image content when no images provided")


def test_build_system_prompt_images_with_none():
    """Test that system prompt handles None images parameter."""
    from jupyvibe.handlers import PromptHandler
    
    class MockHandler:
        pass
    
    handler = MockHandler()
    handler._build_system_prompt = PromptHandler._build_system_prompt.__get__(handler, MockHandler)
    
    # None images (backwards compatibility)
    result = handler._build_system_prompt("", {}, {}, None)
    
    # Should work without error and return tuple
    assert isinstance(result, tuple)
    assert len(result) == 2
    system_text, image_content = result
    assert isinstance(system_text, str)
    assert isinstance(image_content, list)
    assert len(image_content) == 0
    print("✓ System prompt handles None images gracefully")


def test_image_block_structure_openai_format():
    """Test the exact structure of image content for OpenAI/LiteLLM API."""
    from jupyvibe.handlers import PromptHandler
    
    class MockHandler:
        pass
    
    handler = MockHandler()
    handler._build_system_prompt = PromptHandler._build_system_prompt.__get__(handler, MockHandler)
    
    # Test with a realistic base64 PNG (1x1 red pixel)
    red_pixel_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    
    images = [
        {"data": red_pixel_png, "mimeType": "image/png", "source": "output", "cellIndex": 0}
    ]
    
    _, image_content = handler._build_system_prompt("", {}, {}, images)
    
    assert len(image_content) == 2
    
    # First should be text description
    assert image_content[0]["type"] == "text"
    
    # Second should be image_url
    img_item = image_content[1]
    assert img_item["type"] == "image_url"
    assert "image_url" in img_item
    assert img_item["image_url"]["url"] == f"data:image/png;base64,{red_pixel_png}"
    print("✓ Image block structure matches OpenAI/LiteLLM API format")


def test_gif_image_support():
    """Test that GIF images are correctly handled."""
    from jupyvibe.handlers import PromptHandler
    
    class MockHandler:
        pass
    
    handler = MockHandler()
    handler._build_system_prompt = PromptHandler._build_system_prompt.__get__(handler, MockHandler)
    
    gif_data = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
    images = [
        {"data": gif_data, "mimeType": "image/gif", "source": "output", "cellIndex": 0}
    ]
    
    _, image_content = handler._build_system_prompt("", {}, {}, images)
    
    assert len(image_content) == 2
    assert image_content[1]["type"] == "image_url"
    assert f"data:image/gif;base64,{gif_data}" in image_content[1]["image_url"]["url"]
    print("✓ GIF images are correctly handled")


if __name__ == "__main__":
    test_build_system_prompt_with_images()
    test_build_system_prompt_with_multiple_images()
    test_build_system_prompt_image_descriptions()
    test_build_system_prompt_no_images()
    test_build_system_prompt_images_with_none()
    test_image_block_structure_openai_format()
    test_gif_image_support()
    print("\n✅ All multimodal tests passed!")
