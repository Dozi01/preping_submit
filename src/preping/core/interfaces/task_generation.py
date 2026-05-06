"""Core task-generation interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol


@dataclass
class GeneratedTask:
    """Normalized generated task payload."""

    task_id: str
    instruction: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskGeneratorEngine(Protocol):
    """Benchmark-agnostic task generation engine protocol."""

    def generate(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Generate a list of task dictionaries."""


class TaskSource(Protocol):
    """Interface to enumerate benchmark tasks by split."""

    def load_task_ids(self, split: str) -> List[str]:
        """Load task IDs for the requested dataset split."""
