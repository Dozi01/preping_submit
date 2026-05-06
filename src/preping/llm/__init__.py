"""
LLM Interface Module

This package provides a unified interface for various LLM services with a clean
inheritance hierarchy:

Submodules:
- constants: Shared defaults (timeouts, retry limits, etc.)
- pricing: Model pricing tables and cost calculation
- token_tracking: TokenUsage / TokenTracker for API call accounting
- openrouter_utils: OpenRouter provider routing, reasoning toggle, extra_body builder
- base: BaseLLMModel abstract class
- providers: ChatGPT, LiteLLMModel, OpenRouterModel, vLLM
- factory: LLMManager factory and API key helpers

Common public symbols are re-exported here so callers can import from
``preping.llm`` without depending on provider-specific modules.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

# Re-exports: constants
from preping.llm.constants import (  # noqa: F401
    DEFAULT_LLM_PARALLEL_WORKERS,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_RETRY_DELAY,
    DEFAULT_RETRY_LIMIT,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEEPSEEK_DEFAULT_FINGERPRINT,
    NON_RETRYABLE_ERROR_MARKERS,
)

# Re-exports: pricing
from preping.llm.pricing import (  # noqa: F401
    DEFAULT_PRICING,
    MODEL_PRICING,
    calculate_api_cost,
    get_model_pricing,
)

# Re-exports: token tracking
from preping.llm.token_tracking import TokenTracker, TokenUsage  # noqa: F401

# Re-exports: OpenRouter utilities
from preping.llm.openrouter_utils import (  # noqa: F401
    build_openrouter_extra_body,
    deep_merge_dicts,
    get_openrouter_provider_override,
    is_openrouter_qwen_non_thinking_model,
    openrouter_supports_reasoning_toggle,
)

# Re-exports: base
from preping.llm.base import BaseLLMModel  # noqa: F401

# Re-exports: providers
from preping.llm.providers import (  # noqa: F401
    ChatGPT,
    LiteLLMModel,
    OpenRouterModel,
    vLLM,
)

# Re-exports: factory
from preping.llm.factory import (  # noqa: F401
    API_KEY_ENV_VARS,
    LLMManager,
    get_api_key,
)


# Data classes

@dataclass
class LLMOutput:
    """Holds the output from an agent's forward method."""
    action: Any
    response: str
    metadata: Optional[Dict[str, Any]] = None


# Module exports

__all__ = [
    # Constants
    'DEFAULT_MAX_TOKENS',
    'DEFAULT_TEMPERATURE',
    'DEFAULT_LLM_PARALLEL_WORKERS',
    'MODEL_PRICING',
    # Classes
    'BaseLLMModel',
    'ChatGPT',
    'LiteLLMModel',
    'OpenRouterModel',
    'vLLM',
    'LLMManager',
    'TokenTracker',
    'TokenUsage',
    'LLMOutput',
    # Functions
    'get_api_key',
    'calculate_api_cost',
    'build_openrouter_extra_body',
    'deep_merge_dicts',
    'get_openrouter_provider_override',
    'openrouter_supports_reasoning_toggle',
    'is_openrouter_qwen_non_thinking_model',
    'get_model_pricing',
]
