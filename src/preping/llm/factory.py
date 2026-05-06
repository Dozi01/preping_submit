"""Factory for creating LLM instances based on model name."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from preping.llm.base import BaseLLMModel
from preping.llm.constants import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from preping.llm.openrouter_utils import (
    is_openrouter_qwen_non_thinking_model,
    openrouter_supports_reasoning_toggle,
)
from preping.llm.providers import ChatGPT, LiteLLMModel, OpenRouterModel, vLLM

logger = logging.getLogger(__name__)


# Provider to environment variable mapping
API_KEY_ENV_VARS: Dict[str, str] = {
    'openai': 'OPENAI_API_KEY',
    'gemini': 'GEMINI_API_KEY',
    'deepseek': 'DEEPSEEK_API_KEY',
    'azure': 'AZURE_OPENAI_API_KEY',
    'openrouter': 'OPENROUTER_API_KEY',
}


def get_api_key(provider: str = 'openai') -> Optional[str]:
    """Get API key from environment variable for the specified provider.

    Supported providers: openai, gemini, deepseek, azure
    """
    env_var = API_KEY_ENV_VARS.get(provider.lower(), 'OPENAI_API_KEY')
    return os.getenv(env_var)


class LLMManager:
    """Factory for creating LLM instances based on model name."""

    # Model name patterns for provider detection
    OPENAI_PATTERNS = ('gpt', 'o1', 'o3', 'o4')
    OPENROUTER_PATTERNS = ('openrouter',)

    @staticmethod
    def _require_api_key(
        key: Optional[str],
        provider: str,
        required_env_var: str,
    ) -> str:
        """Resolve API key from argument or environment, with provider-specific error."""
        api_key = key or get_api_key(provider)
        if not api_key:
            raise ValueError(f"{required_env_var} environment variable is required")
        return api_key

    @staticmethod
    def _ensure_model_prefix(model_name: str, prefix: str) -> str:
        """Ensure model name has provider prefix expected by LiteLLM."""
        return model_name if model_name.startswith(prefix) else f"{prefix}{model_name}"

    @classmethod
    def _create_litellm_provider_model(
        cls,
        model_name: str,
        key: Optional[str],
        system_message: str,
        temperature: float,
        use_thinking: bool,
        max_tokens: int,
        *,
        provider: str,
        required_env_var: str,
        model_prefix: str,
        model_cls: Any,
    ) -> LiteLLMModel:
        """Create LiteLLM-based provider model with shared validation and normalization."""
        api_key = cls._require_api_key(key, provider, required_env_var)
        litellm_name = cls._ensure_model_prefix(model_name, model_prefix)
        return model_cls(
            litellm_name,
            api_key=api_key,
            system_message=system_message,
            temperature=temperature,
            use_thinking=use_thinking,
            max_tokens=max_tokens,
        )

    @classmethod
    def create_llm(
        cls,
        model_name: str,
        key: Optional[str] = None,
        system_message: str = '',
        temperature: float = DEFAULT_TEMPERATURE,
        use_thinking: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> BaseLLMModel:
        """Create appropriate LLM instance based on model name.

        Args:
            model_name: Name of the model to use
            key: API key (optional, falls back to environment variables)
            system_message: System message to use
            temperature: Sampling temperature
            use_thinking: Enable thinking mode for reasoning models

        Returns:
            Appropriate LLM model instance

        Raises:
            ValueError: If required API key is not found
        """
        name_lower = model_name.lower()

        # OpenRouter models (check before openai/deepseek since model names can overlap)
        if any(pattern in name_lower for pattern in cls.OPENROUTER_PATTERNS):
            return cls._create_openrouter_model(model_name, key, system_message, temperature, use_thinking, max_tokens)

        # OpenAI models
        if any(pattern in name_lower for pattern in cls.OPENAI_PATTERNS):
            return cls._create_openai_model(model_name, key, system_message, temperature, use_thinking, max_tokens)

        # Gemini models
        if 'gemini' in name_lower:
            return cls._create_gemini_model(model_name, key, system_message, temperature, use_thinking, max_tokens)

        # DeepSeek models
        if 'deepseek' in name_lower:
            return cls._create_deepseek_model(model_name, key, system_message, temperature, use_thinking, max_tokens)

        # Default to local vLLM
        return vLLM(
            model_name,
            system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=key,
        )

    @staticmethod
    def _create_openai_model(
        model_name: str,
        key: Optional[str],
        system_message: str,
        temperature: float,
        use_thinking: bool,
        max_tokens: int,
    ) -> ChatGPT:
        """Create OpenAI ChatGPT model."""
        api_key = key or get_api_key('openai')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return ChatGPT(
            model_name,
            api_key,
            system_message,
            temperature=temperature,
            use_thinking=use_thinking,
            max_tokens=max_tokens,
        )

    @staticmethod
    def _create_gemini_model(
        model_name: str,
        key: Optional[str],
        system_message: str,
        temperature: float,
        use_thinking: bool,
        max_tokens: int,
    ) -> LiteLLMModel:
        """Create Gemini model via LiteLLM."""
        return LLMManager._create_litellm_provider_model(
            model_name=model_name,
            key=key,
            system_message=system_message,
            temperature=temperature,
            use_thinking=use_thinking,
            max_tokens=max_tokens,
            provider='gemini',
            required_env_var='GEMINI_API_KEY',
            model_prefix='gemini/',
            model_cls=LiteLLMModel,
        )

    @staticmethod
    def _create_deepseek_model(
        model_name: str,
        key: Optional[str],
        system_message: str,
        temperature: float,
        use_thinking: bool,
        max_tokens: int,
    ) -> LiteLLMModel:
        """Create DeepSeek model via LiteLLM."""
        return LLMManager._create_litellm_provider_model(
            model_name=model_name,
            key=key,
            system_message=system_message,
            temperature=temperature,
            use_thinking=use_thinking,
            max_tokens=max_tokens,
            provider='deepseek',
            required_env_var='DEEPSEEK_API_KEY',
            model_prefix='deepseek/',
            model_cls=LiteLLMModel,
        )

    @staticmethod
    def _create_openrouter_model(
        model_name: str,
        key: Optional[str],
        system_message: str,
        temperature: float,
        use_thinking: bool,
        max_tokens: int,
    ) -> OpenRouterModel:
        """Create OpenRouter model via the native OpenRouter API."""
        api_key = LLMManager._require_api_key(key, 'openrouter', 'OPENROUTER_API_KEY')
        normalized_name = LLMManager._ensure_model_prefix(model_name, 'openrouter/')
        if use_thinking and is_openrouter_qwen_non_thinking_model(normalized_name):
            logger.warning(
                "Model %s may ignore use_thinking on OpenRouter because it is a non-thinking Qwen instruct variant.",
                normalized_name,
            )
        reasoning_config = None
        if openrouter_supports_reasoning_toggle(normalized_name):
            reasoning_config = {"enabled": bool(use_thinking)}
        return OpenRouterModel(
            normalized_name,
            api_key=api_key,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_config=reasoning_config,
        )
