"""Model pricing configuration and cost calculation utilities."""

from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Model pricing information (per 1M tokens) - updated as of 2024
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    'gpt-5': {'input': 1.25, 'cached_input': 0.125, 'output': 10.00},
    'gpt-5-mini': {'input': 0.25, 'cached_input': 0.025, 'output': 2.00},
    'gpt-4.1': {'input': 2.00, 'cached_input': 0.50, 'output': 8.00},
    'gpt-4.1-mini': {'input': 0.8, 'cached_input': 0.10, 'output': 1.6},
    'o3': {'input': 2.0, 'cached_input': 0.50, 'output': 8.0},
    'o4-mini': {'input': 1.10, 'cached_input': 0.275, 'output': 4.40},
    'deepseek/deepseek-reasoner': {'input': 0.28, 'cached_input': 0.028, 'output': 0.42},
    'deepseek/deepseek-chat': {'input': 0.28, 'cached_input': 0.028, 'output': 0.42},
    'openrouter/deepseek/deepseek-v3.2': {'input': 0.269, 'cached_input': 0.1345, 'output': 0.4},
    'openrouter/z-ai/glm-5': {'input': 0.95, 'cached_input': 0.475, 'output': 2.55},
    'openrouter/openai/gpt-oss-120b': {'input': 0.05, 'cached_input': 0.025, 'output': 0.45},
    'openrouter/qwen/qwen3-235b-a22b-2507': {'input': 0.10, 'cached_input': 0.10, 'output': 0.10},
    'qwen3-235b-a22b-2507': {'input': 0.10, 'cached_input': 0.10, 'output': 0.10},
    'qwen/qwen3-14b': {'input': 0, 'cached_input': 0, 'output': 0},
    'qwen/qwen3-8b': {'input': 0, 'cached_input': 0, 'output': 0},
}

DEFAULT_PRICING = MODEL_PRICING['gpt-5']


def get_model_pricing(model_name: str) -> Dict[str, float]:
    """Get pricing info for a model, with fallback to default."""
    normalized_name = model_name.lower()
    for model_key in sorted(MODEL_PRICING.keys(), key=len, reverse=True):
        if model_key in normalized_name:
            return MODEL_PRICING[model_key]

    logger.warning(f"No pricing found for model {model_name}, using gpt-5 pricing")
    return DEFAULT_PRICING


def calculate_api_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0
) -> float:
    """Calculate the API cost for given token usage.

    Args:
        model_name: Name of the model used
        input_tokens: Number of input tokens (non-cached)
        output_tokens: Number of output tokens
        cached_input_tokens: Number of cached input tokens (cheaper rate)

    Returns:
        Total cost in USD
    """
    pricing = get_model_pricing(model_name)

    # Calculate cost (pricing is per 1M tokens)
    input_cost = (input_tokens / 1_000_000) * pricing['input']
    cached_input_cost = (cached_input_tokens / 1_000_000) * pricing.get('cached_input', pricing['input'] * 0.5)
    output_cost = (output_tokens / 1_000_000) * pricing['output']

    return input_cost + cached_input_cost + output_cost
