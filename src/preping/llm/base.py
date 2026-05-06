"""Abstract base class for all LLM models with common functionality."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Union

from preping.llm.constants import (
    DEFAULT_LLM_PARALLEL_WORKERS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_RETRY_DELAY,
    DEFAULT_RETRY_LIMIT,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    NON_RETRYABLE_ERROR_MARKERS,
)
from preping.llm.token_tracking import TokenTracker, TokenUsage

logger = logging.getLogger(__name__)


class BaseLLMModel(ABC):
    """Abstract base class for all LLM models with common functionality."""

    def __init__(
        self,
        model_name: str,
        system_message: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.model_name = model_name
        self.system_message = system_message
        self.temperature = self._resolve_temperature(temperature)
        self.max_tokens = max_tokens
        self.token_tracker = TokenTracker(model_name)

    def _resolve_temperature(self, temperature: float) -> float:
        """Get appropriate temperature for model type."""
        # Force temperature 1.0 for OpenAI reasoning models
        name_lower = self.model_name.lower()
        if any(x in name_lower for x in ['o1', 'o3', 'o4', 'gpt-5']):
            logger.info(f"Forcing temperature to 1.0 for reasoning model {self.model_name}")
            return 1.0
        return temperature

    @property
    def _max_token_key(self) -> str:
        """Get appropriate max token parameter key for model type."""
        name_lower = self.model_name.lower()
        if any(x in name_lower for x in ['o1', 'o3', 'o4', 'gpt-5']):
            return 'max_completion_tokens'
        return 'max_tokens'

    @property
    def _supports_system_message(self) -> bool:
        """Check if model supports system messages directly."""
        # o1 models don't support system messages
        return 'o1' not in self.model_name

    def _build_messages(self, prompt: Union[str, List]) -> List[Dict[str, str]]:
        """Build messages array for API call."""
        messages = []

        # Add system message for supported models
        if self._supports_system_message and self.system_message:
            messages.append({
                "role": "system",
                "content": self.system_message
            })

        # Handle different prompt types
        if isinstance(prompt, str):
            user_content = prompt
            # For o1 models, prepend system message to user content
            if not self._supports_system_message and self.system_message:
                user_content = f"{self.system_message}\n\n{prompt}"

            messages.append({"role": "user", "content": user_content})

        elif isinstance(prompt, list):
            if prompt and isinstance(prompt[0], str):
                # Alternating user/assistant format
                for i, msg in enumerate(prompt):
                    role = "user" if i % 2 == 0 else "assistant"
                    messages.append({"role": role, "content": msg})
            elif prompt and isinstance(prompt[0], dict):
                messages.extend(prompt)
            else:
                raise ValueError("Prompt list must contain strings or dicts")
        else:
            raise ValueError("Prompt must be a string or a list")

        return messages

    def _should_retry(self, error: Exception, retry_num: int) -> bool:
        """Log error and determine if should retry. Returns True if should retry."""
        logger.error(f"{self.__class__.__name__}: {error}")
        if self._is_non_retryable_error(error):
            return False
        if retry_num >= DEFAULT_RETRY_LIMIT:
            return False
        time.sleep(DEFAULT_RETRY_DELAY)
        return True

    def _is_non_retryable_error(self, error: Exception) -> bool:
        """Return True for request errors that retries cannot resolve."""
        error_text = f"{type(error).__name__}: {error}".lower()
        return any(marker in error_text for marker in NON_RETRYABLE_ERROR_MARKERS)

    def _execute_request(self, request_fn: Callable[[], Any]) -> Any:
        """Execute request with retry policy."""
        for retry_num in range(DEFAULT_RETRY_LIMIT + 1):
            try:
                return request_fn()
            except Exception as error:
                if not self._should_retry(error, retry_num):
                    raise

    def _track_response_usage(
        self,
        response: Any,
        increment_request_on_empty_usage: bool = False,
    ) -> None:
        """Extract and track token usage from response."""
        usage = TokenTracker.extract_from_response(response)
        if usage.total > 0:
            self._update_token_usage(usage)
        elif increment_request_on_empty_usage:
            self.token_tracker.increment_request_count()

    def _execute_request_with_tracking(
        self,
        request_fn: Callable[[], Any],
        increment_request_on_empty_usage: bool = False,
    ) -> Any:
        """Execute request with retries and token tracking."""
        response = self._execute_request(request_fn)
        self._track_response_usage(
            response,
            increment_request_on_empty_usage=increment_request_on_empty_usage,
        )
        return response

    def get_model_options(
        self,
        temperature: Optional[float] = None,
        top_p: float = 1.0,
        n: int = 1,
        seed: int = DEFAULT_SEED
    ) -> Dict[str, Any]:
        """Get model options for API call."""
        return {
            self._max_token_key: self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": top_p,
            "n": n,
            "seed": seed,
        }

    @abstractmethod
    def generate(self, prompt: Union[str, List], **kwargs) -> str:
        """Generate response - to be implemented by subclasses."""
        pass

    def generate_batch(
        self,
        prompts: List[Union[str, List]],
        max_workers: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        """Generate responses for multiple prompts in parallel while preserving input order."""
        if not prompts:
            return []

        worker_count = min(
            len(prompts),
            max_workers if max_workers is not None else DEFAULT_LLM_PARALLEL_WORKERS,
        )

        responses: List[str] = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_idx = {
                executor.submit(self.generate, prompt, **kwargs): idx
                for idx, prompt in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    response = future.result()
                    responses[idx] = response if isinstance(response, str) else str(response)
                except Exception as e:
                    responses[idx] = f"ERROR: {e}"

        return responses

    def set_system_message(self, system_message: str) -> None:
        """Set system message."""
        self.system_message = system_message

    # Token tracking delegation methods
    def _update_token_usage(self, usage: TokenUsage) -> None:
        """Update token usage statistics."""
        self.token_tracker.update(usage)

    def get_total_tokens(self) -> int:
        """Get total tokens used."""
        return self.token_tracker.get_total_tokens()

    def get_token_usage_stats(self) -> Dict[str, Any]:
        """Get detailed token usage statistics."""
        return self.token_tracker.get_stats()

    def get_total_cost(self) -> float:
        """Get total API cost for this model instance."""
        return self.token_tracker.get_total_cost()

    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Get detailed cost breakdown."""
        return self.token_tracker.get_cost_breakdown()
