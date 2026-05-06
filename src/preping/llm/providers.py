"""Concrete LLM provider implementations.

Classes:
    - ChatGPT: OpenAI API with API key authentication
    - LiteLLMModel: LiteLLM-based model for multiple providers
    - OpenRouterModel: OpenRouter API via OpenAI-compatible client
    - vLLM: Local vLLM server interface
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

from preping.llm.base import BaseLLMModel
from preping.llm.constants import (
    DEEPSEEK_DEFAULT_FINGERPRINT,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
)
from preping.llm.openrouter_utils import build_openrouter_extra_body
from preping.llm.token_tracking import TokenTracker

logger = logging.getLogger(__name__)


# OpenAI ChatGPT

class ChatGPT(BaseLLMModel):
    """OpenAI ChatGPT API implementation."""

    def __init__(
        self,
        model_name: str,
        key: str,
        system_message: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        use_thinking: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        super().__init__(model_name, system_message, temperature, max_tokens=max_tokens)
        self.key = key
        self.use_thinking = use_thinking
        self.default_reasoning_effort = 'medium' if use_thinking else 'low'
        self.client = OpenAI(api_key=key)

    @property
    def _supports_reasoning_effort(self) -> bool:
        """Check if model supports OpenAI reasoning effort parameter."""
        name_lower = self.model_name.lower()
        return any(x in name_lower for x in ['o1', 'o3', 'o4', 'gpt-5'])

    def _resolve_reasoning_effort(
        self,
        reasoning_effort: Optional[str]
    ) -> str:
        """Resolve reasoning effort from explicit effort or cached default."""
        return reasoning_effort or self.default_reasoning_effort

    def generate(
        self,
        prompt: Union[str, List],
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        stop: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        flex_mode: Optional[bool] = True,
        **kwargs
    ) -> str:
        """Generate response using OpenAI API."""
        messages = self._build_messages(prompt)
        options = self.get_model_options(
            temperature=temperature,
            n=1,
            seed=seed if seed is not None else DEFAULT_SEED
        )

        if stop:
            options['stop'] = stop

        if self._supports_reasoning_effort:
            options['reasoning_effort'] = self._resolve_reasoning_effort(
                reasoning_effort=reasoning_effort,
            )

        if flex_mode:
            options['service_tier'] = "flex"

        try:
            response = self._execute_request_with_tracking(
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    timeout=DEFAULT_LLM_TIMEOUT,
                    **options,
                )
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"ERROR: {e}"


# LiteLLM

class LiteLLMModel(BaseLLMModel):
    """LiteLLM-based model supporting multiple providers (Gemini, DeepSeek, Anthropic, etc.)

    Uses litellm package for unified API access to various LLM providers.
    Model names should follow litellm naming convention:
    - Gemini: "gemini/gemini-1.5-pro", "gemini/gemini-2.0-flash-exp"
    - DeepSeek: "deepseek/deepseek-chat", "deepseek/deepseek-reasoner"
    - Anthropic: "anthropic/claude-3-opus", "anthropic/claude-3-sonnet"
    - OpenRouter: "openrouter/model-name"
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        system_message: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        base_url: Optional[str] = None,
        use_thinking: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        super().__init__(model_name, system_message, temperature, max_tokens=max_tokens)

        self.api_key = api_key
        self.base_url = base_url
        self.use_thinking = use_thinking

        try:
            import litellm
            self.litellm = litellm
            litellm.cache = None  # Disable caching
        except ImportError:
            raise ImportError("litellm package is required. Install with: pip install litellm")

    @property
    def _is_reasoning_model(self) -> bool:
        """Check if this is a reasoning model that supports thinking mode."""
        name_lower = self.model_name.lower()
        return 'reasoner' in name_lower or 'deepseek-r1' in name_lower

    def _get_extra_body(self) -> Optional[Dict[str, Any]]:
        """Return provider-specific extra body params for LiteLLM."""
        return None

    def _build_litellm_api_params(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        stop: Optional[Union[str, List[str]]],
        seed: Optional[int],
    ) -> Dict[str, Any]:
        """Build common LiteLLM completion parameters."""
        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        if self.api_key:
            api_params["api_key"] = self.api_key
        if self.base_url:
            api_params["base_url"] = self.base_url
        if stop:
            api_params["stop"] = stop
        if seed:
            api_params["seed"] = seed

        if self._is_reasoning_model:
            api_params["thinking"] = {"type": "enabled" if self.use_thinking else "disabled"}

        extra_body = self._get_extra_body()
        if extra_body:
            api_params["extra_body"] = extra_body

        return api_params

    def _log_reasoning_content(self, response: Any) -> None:
        """Log reasoning content when available."""
        message = response.choices[0].message
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            logger.debug(f"Reasoning content: {message.reasoning_content[:500]}...")

    def _handle_post_response(self, response: Any) -> None:
        """Hook for provider-specific response handling."""
        if hasattr(response, 'system_fingerprint') and response.system_fingerprint != DEEPSEEK_DEFAULT_FINGERPRINT:
            logger.warning(f"System fingerprint changed!: {response.system_fingerprint}")

    def generate(
        self,
        prompt: Union[str, List],
        temperature: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate response using LiteLLM.

        Args:
            prompt: Input prompt (string or list of messages)
            temperature: Sampling temperature
            stop: Stop sequences
            seed: Random seed for reproducibility
        """
        messages = self._build_messages(prompt)
        api_params = self._build_litellm_api_params(
            messages=messages,
            temperature=temperature,
            stop=stop,
            seed=seed,
        )

        try:
            response = self._execute_request_with_tracking(
                lambda: self.litellm.completion(**api_params, timeout=DEFAULT_LLM_TIMEOUT)
            )

            self._log_reasoning_content(response)
            self._handle_post_response(response)

            return response.choices[0].message.content
        except Exception as e:
            return f"ERROR: {e}"


# OpenRouter

class OpenRouterModel(BaseLLMModel):
    """OpenRouter model via the OpenAI-compatible OpenRouter API."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    # Fixed provider configuration for selected OpenRouter models
    DEEPSEEK_PROVIDER = {
        "order": ["deepseek"],
        "allow_fallbacks": False
    }
    GPT_OSS_PROVIDER = {
        "order": ["chutes/bf16", "atlas-cloud/fp8"],
        "allow_fallbacks": True,
    }
    QWEN3_235B_PROVIDER = {
        "order": ["wandb/bf16", "deepinfra/fp8"],
        "allow_fallbacks": False,
    }

    def __init__(
        self,
        model_name: str,
        api_key: str,
        system_message: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        app_url: Optional[str] = None,
        app_title: Optional[str] = None,
        provider_override: Optional[Dict[str, Any]] = None,
        reasoning_config: Optional[Dict[str, Any]] = None,
        extra_body_override: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_name, system_message, temperature, max_tokens=max_tokens)
        self.api_key = api_key
        self.api_model_name = self._normalize_model_name(model_name)
        self.app_url = app_url or os.getenv("OPENROUTER_HTTP_REFERER")
        self.app_title = app_title or os.getenv("OPENROUTER_APP_TITLE")
        self.provider_override = provider_override
        self.reasoning_config = reasoning_config
        self.extra_body_override = extra_body_override
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.DEFAULT_BASE_URL,
            default_headers=self._build_default_headers(),
        )

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        """Strip the local routing prefix before sending the model name to OpenRouter."""
        prefix = "openrouter/"
        return model_name[len(prefix):] if model_name.startswith(prefix) else model_name

    def _build_default_headers(self) -> Optional[Dict[str, str]]:
        """Build optional OpenRouter attribution headers."""
        headers: Dict[str, str] = {}
        if self.app_url:
            headers["HTTP-Referer"] = self.app_url
        if self.app_title:
            headers["X-Title"] = self.app_title
        return headers or None

    @staticmethod
    def _get_provider_override(model_name: str) -> Optional[Dict[str, Any]]:
        """Return fixed OpenRouter provider routing for selected models."""
        from preping.llm.openrouter_utils import get_openrouter_provider_override
        return get_openrouter_provider_override(model_name)

    def _get_extra_body(self) -> Optional[Dict[str, Any]]:
        """Build OpenRouter extra_body from provider pinning and optional overrides."""
        return build_openrouter_extra_body(
            self.model_name,
            provider_override=self.provider_override,
            reasoning_config=self.reasoning_config,
            extra_body_override=self.extra_body_override,
        )

    def _handle_post_response(self, response: Any) -> None:
        """OpenRouter response post-processing hook."""
        return

    def generate(
        self,
        prompt: Union[str, List],
        temperature: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate response using the OpenRouter Chat Completions API."""
        messages = self._build_messages(prompt)
        options = self.get_model_options(
            temperature=temperature,
            n=1,
            seed=seed if seed is not None else DEFAULT_SEED,
        )

        if stop:
            options["stop"] = stop

        extra_body = self._get_extra_body()
        if extra_body:
            options["extra_body"] = extra_body

        try:
            response = self._execute_request_with_tracking(
                lambda: self.client.chat.completions.create(
                    model=self.api_model_name,
                    messages=messages,
                    timeout=DEFAULT_LLM_TIMEOUT,
                    **options,
                ),
                increment_request_on_empty_usage=True,
            )
            self._handle_post_response(response)
            return response.choices[0].message.content
        except Exception as e:
            return f"ERROR: {e}"


# vLLM (local server)

class vLLM(BaseLLMModel):
    """Local vLLM server interface."""

    DEFAULT_BASE_URL = "http://localhost:8000/v1"
    DEFAULT_API_KEY = "token-abc"
    DEFAULT_PRESENCE_PENALTY = 0.5

    def __init__(
        self,
        model_name: str,
        system_message: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        super().__init__(model_name, system_message, temperature, max_tokens=max_tokens)

        resolved_base_url = base_url or os.getenv("VLLM_BASE_URL", self.DEFAULT_BASE_URL)
        resolved_api_key = api_key or os.getenv("VLLM_API_KEY", self.DEFAULT_API_KEY)
        self.client = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)

    def generate(
        self,
        prompt: Union[str, List],
        n: int = 1,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """Generate response using vLLM server.

        Args:
            prompt: Input prompt
            n: Number of completions to generate
            temperature: Sampling temperature
            seed: Random seed

        Returns:
            Single response string or list of strings if n > 1
        """
        messages = self._build_messages(prompt)
        options = self.get_model_options(
            temperature=temperature,
            n=n,
            seed=seed if seed is not None else DEFAULT_SEED
        )

        options["extra_body"] = {
            "presence_penalty": self.DEFAULT_PRESENCE_PENALTY,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        try:
            response = self._execute_request_with_tracking(
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **options,
                ),
                increment_request_on_empty_usage=True,
            )

            if n > 1:
                return [choice.message.content for choice in response.choices]
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"vLLM generation Error: {e}")
            return f"ERROR: {e}"
