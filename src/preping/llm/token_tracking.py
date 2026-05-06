"""Token usage tracking for LLM API calls."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict

from preping.llm.pricing import calculate_api_cost

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token usage for a single API call."""
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.cached_input_tokens + self.output_tokens


class TokenTracker:
    """Tracks token usage across multiple API calls."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.total_input_tokens: int = 0
        self.total_cached_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_reasoning_tokens: int = 0
        self.total_requests: int = 0
        self._lock = Lock()

    def update(self, usage: TokenUsage) -> None:
        """Update tracking with new usage data."""
        with self._lock:
            self.total_input_tokens += usage.input_tokens
            self.total_cached_input_tokens += usage.cached_input_tokens
            self.total_output_tokens += usage.output_tokens
            self.total_reasoning_tokens += usage.reasoning_tokens
            self.total_requests += 1

        self._log_usage(usage)

    def increment_request_count(self) -> None:
        """Increment total request count for calls without usage details."""
        with self._lock:
            self.total_requests += 1

    def _log_usage(self, usage: TokenUsage) -> None:
        """Log token usage details."""
        log_parts = [f"Input: {usage.input_tokens}"]
        if usage.cached_input_tokens > 0:
            log_parts.append(f"Cached Input: {usage.cached_input_tokens}")
        log_parts.append(f"Output: {usage.output_tokens}")
        if usage.reasoning_tokens > 0:
            log_parts.append(f"Reasoning: {usage.reasoning_tokens}")
        log_parts.append(f"Total: {self.get_total_tokens()}")
        logger.debug(f"Token usage - {', '.join(log_parts)}")

    def get_total_tokens(self) -> int:
        """Get total tokens used (including cached)."""
        with self._lock:
            return self.total_input_tokens + self.total_cached_input_tokens + self.total_output_tokens

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed token usage statistics."""
        with self._lock:
            total_tokens = self.total_input_tokens + self.total_cached_input_tokens + self.total_output_tokens
            return {
                'total_input_tokens': self.total_input_tokens,
                'total_cached_input_tokens': self.total_cached_input_tokens,
                'total_output_tokens': self.total_output_tokens,
                'total_reasoning_tokens': self.total_reasoning_tokens,
                'total_tokens': total_tokens,
                'total_requests': self.total_requests,
                'model_name': self.model_name
            }

    def get_total_cost(self) -> float:
        """Get total API cost."""
        return calculate_api_cost(
            self.model_name,
            self.total_input_tokens,
            self.total_output_tokens,
            self.total_cached_input_tokens
        )

    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Get detailed cost breakdown."""
        return {
            **self.get_stats(),
            'total_cost_usd': self.get_total_cost(),
            'input_cost_usd': calculate_api_cost(self.model_name, self.total_input_tokens, 0, 0),
            'cached_input_cost_usd': calculate_api_cost(self.model_name, 0, 0, self.total_cached_input_tokens),
            'output_cost_usd': calculate_api_cost(self.model_name, 0, self.total_output_tokens, 0),
        }

    @staticmethod
    def extract_from_response(response: Any) -> TokenUsage:
        """Extract token usage from API response."""
        if not hasattr(response, 'usage') or not response.usage:
            return TokenUsage()

        usage = response.usage
        input_tokens = getattr(usage, 'prompt_tokens', 0) or 0
        output_tokens = getattr(usage, 'completion_tokens', 0) or 0

        # Extract cached input tokens
        cached_input_tokens = 0
        if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
            cached_input_tokens = getattr(usage.prompt_tokens_details, 'cached_tokens', 0) or 0

        # Extract reasoning tokens
        reasoning_tokens = 0
        if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
            reasoning_tokens = getattr(usage.completion_tokens_details, 'reasoning_tokens', 0) or 0

        # Adjust input_tokens to exclude cached tokens
        non_cached_input_tokens = input_tokens - cached_input_tokens

        return TokenUsage(
            input_tokens=non_cached_input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens
        )
