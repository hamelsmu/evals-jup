# LiteLLM Integration Brainstorming

## Goal

Replace direct Anthropic SDK usage with LiteLLM to support multiple vision-capable model providers (OpenAI, Anthropic, Gemini) while maintaining a clean model picker UX.

## Model Picker Design

### UI Flow

1. **Provider dropdown** (first selection):
   - OpenAI
   - Anthropic
   - Gemini

2. **Model dropdown** (filtered by provider):
   - Sorted by release date (newest first)
   - `-latest` aliases at top
   - Only vision-capable models shown

### Backend: Getting Available Models

```python
import re
import litellm

TRUSTED_PROVIDERS = {
    'openai': 'OpenAI',
    'anthropic': 'Anthropic', 
    'gemini': 'Google (Gemini)',
}

def extract_date(model_name: str) -> str:
    """Extract date from model name for sorting."""
    match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', model_name)
    if match:
        return f'{match.group(1)}-{match.group(2)}-{match.group(3)}'
    if 'latest' in model_name:
        return '9999-99-99'  # Sort to top
    return '0000-00-00'

def get_vision_models_by_provider() -> dict[str, list[dict]]:
    """Get vision-capable models grouped by provider, sorted by date."""
    by_provider = {p: [] for p in TRUSTED_PROVIDERS}
    
    for model_name, info in litellm.model_cost.items():
        provider = info.get('litellm_provider', '')
        
        # Skip non-trusted or non-vision models
        if not litellm.supports_vision(model_name):
            continue
        
        # Map to our provider keys
        matched_provider = None
        for key in TRUSTED_PROVIDERS:
            if key in provider and 'bedrock' not in provider and 'azure' not in provider:
                matched_provider = key
                break
        
        if not matched_provider:
            continue
        
        by_provider[matched_provider].append({
            'name': model_name,
            'date': extract_date(model_name),
            'provider_raw': provider,
        })
    
    # Sort each provider's models by date descending
    for provider in by_provider:
        by_provider[provider].sort(key=lambda x: x['date'], reverse=True)
    
    return by_provider
```

### API Endpoint

New endpoint: `GET /jupyvibe/models`

```python
class ModelsHandler(APIHandler):
    @authenticated
    async def get(self):
        """Return available vision models grouped by provider."""
        models = get_vision_models_by_provider()
        self.finish(json.dumps(models))
```

### Frontend Component

```typescript
interface ModelPickerProps {
  value: string;
  onChange: (model: string) => void;
}

// Two dropdowns:
// 1. Provider selector
// 2. Model selector (filtered by provider)
```

## Message Format Translation

### Current Format (Anthropic-specific)

```python
# Images in system prompt
{
    "type": "image",
    "source": {
        "type": "base64",
        "media_type": "image/png",
        "data": "<base64>"
    }
}
```

### LiteLLM Format (OpenAI-style, auto-translated)

```python
# LiteLLM expects OpenAI format, translates to each provider
{
    "type": "image_url",
    "image_url": {
        "url": "data:image/png;base64,<base64>",
        "detail": "auto"  # optional
    }
}
```

### Required Changes in handlers.py

1. Replace `anthropic.Anthropic()` with `litellm.completion()`
2. Convert image blocks from Anthropic format to OpenAI format
3. Use `litellm.supports_vision()` to validate model selection

```python
def convert_image_to_litellm_format(anthropic_image: dict) -> dict:
    """Convert Anthropic image block to LiteLLM/OpenAI format."""
    source = anthropic_image.get('source', {})
    media_type = source.get('media_type', 'image/png')
    data = source.get('data', '')
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{media_type};base64,{data}"
        }
    }
```

## Mid-Session Model Switching Considerations

### What Could Break

1. **Message History Format**
   - Currently not an issue: each prompt cell is independent
   - No conversation history maintained between cells
   - ✅ Safe to switch models between cells

2. **Tool Calling Behavior**
   - Different models have different tool-calling syntax/behavior
   - LiteLLM normalizes this, but quality of tool use varies
   - ⚠️ May see different results with same `&function` references

3. **Context Window Limits**
   - Models have different max input tokens
   - Large notebooks with many preceding cells could exceed limits
   - ⚠️ Should validate context size before sending

4. **Image Handling Differences**
   - Some models handle images better than others
   - Gemini supports video, others don't
   - ✅ LiteLLM handles format translation

5. **Streaming Behavior**
   - All three providers support streaming
   - LiteLLM normalizes streaming events
   - ✅ Should work consistently

### Recommendations

1. **Per-cell model selection**: Already in metadata, keep this pattern
2. **Validate on execution**: Check `supports_vision()` before sending
3. **Context size check**: Warn if context exceeds model's limit
4. **No breaking changes expected**: Cells are independent

## Implementation Plan

### Phase 1: Backend LiteLLM Integration

1. Add `litellm` to dependencies in `pyproject.toml`
2. Create `get_vision_models_by_provider()` utility
3. Add `/jupyvibe/models` endpoint
4. Modify `PromptHandler` to use `litellm.completion()` instead of `anthropic.Anthropic()`
5. Convert image format in `_build_system_prompt()`

### Phase 2: Frontend Model Picker

1. Create `ModelPicker` component with provider + model dropdowns
2. Fetch models from `/jupyvibe/models` on mount
3. Store selected model in notebook metadata
4. Wire up to prompt cell execution

### Phase 3: Validation & Polish

1. Add context size validation
2. Handle API key errors gracefully (different env vars per provider)
3. Cache model list (LiteLLM's model_cost is static per version)

## Environment Variables

LiteLLM uses standard env vars:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...  # or GOOGLE_API_KEY
```

No changes needed - already using `ANTHROPIC_API_KEY`.

## Decisions

1. **Allowed providers**: OpenAI, Anthropic, and Gemini (not Vertex AI - simpler API key auth, no GCP service accounts)
2. **Model picker location**: In settings (global default), not per-cell
3. **Test connection**: Yes, add a "Test Connection" button per provider

### Test Connection Implementation

Backend endpoint: `POST /jupyvibe/test-connection`

```python
class TestConnectionHandler(APIHandler):
    @authenticated
    async def post(self):
        """Test connection to a provider with a minimal request."""
        data = self.get_json_body() or {}
        model = data.get("model", "")
        
        if not model:
            self.set_status(400)
            self.finish({"error": "model is required", "success": False})
            return
        
        try:
            # Minimal request to verify API key works
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
            )
            self.finish({"success": True, "model": model})
        except litellm.AuthenticationError as e:
            self.finish({"success": False, "error": "Invalid API key", "details": str(e)})
        except litellm.NotFoundError as e:
            self.finish({"success": False, "error": "Model not found", "details": str(e)})
        except Exception as e:
            self.finish({"success": False, "error": str(e)})
```

Frontend: Button next to model dropdown that shows ✓ or ✗ with error message.
