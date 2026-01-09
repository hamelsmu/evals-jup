"""Model discovery and filtering for LiteLLM providers."""

import re
import litellm

TRUSTED_PROVIDERS = {
    'openai': 'OpenAI',
    'anthropic': 'Anthropic',
    'gemini': 'Google (Gemini)',
}


def extract_date(model_name: str) -> str:
    """Extract date from model name for sorting (newest first)."""
    match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', model_name)
    if match:
        return f'{match.group(1)}-{match.group(2)}-{match.group(3)}'
    if 'latest' in model_name.lower():
        return '9999-99-99'
    return '0000-00-00'


def get_vision_models_by_provider() -> dict[str, list[dict]]:
    """Get vision-capable models grouped by provider, sorted by date (newest first)."""
    by_provider = {p: [] for p in TRUSTED_PROVIDERS}

    for model_name, info in litellm.model_cost.items():
        provider = info.get('litellm_provider', '')

        if not info.get('supports_vision', False):
            continue

        matched_provider = None
        for key in TRUSTED_PROVIDERS:
            if key in provider and 'bedrock' not in provider and 'azure' not in provider:
                matched_provider = key
                break

        if not matched_provider:
            continue

        by_provider[matched_provider].append({
            'id': model_name,
            'name': model_name,
            'date': extract_date(model_name),
        })

    for provider in by_provider:
        by_provider[provider].sort(key=lambda x: x['date'], reverse=True)

    return by_provider


def get_provider_display_names() -> dict[str, str]:
    """Return provider key -> display name mapping."""
    return TRUSTED_PROVIDERS.copy()
