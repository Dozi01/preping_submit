"""Core memory interface used across benchmarks."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MemoryManager(ABC):
    """Abstract interface for memory implementations."""

    @abstractmethod
    def get_memory(
        self,
        task_description: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """Get formatted memory/context for prompt injection."""
        ...

    @abstractmethod
    def process_episode(
        self,
        task_description: str,
        trajectory: List[Dict[str, Any]],
        **kwargs,
    ) -> Any:
        """Process a completed episode and update memory."""
        ...

    def get_token_usage_summary(self) -> Dict[str, Any]:
        """Get summary of token usage and costs for memory-related LLM calls."""
        if hasattr(self.llm_client, "get_cost_breakdown"):
            return self.llm_client.get_cost_breakdown()
        return {}
