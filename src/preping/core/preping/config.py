"""Shared PrePing experiment configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TaskManagerConfig:
    """Shared task generation / validation configuration for PrePing."""

    model_name: str = "deepseek/deepseek-chat"
    temperature: float = 1.0
    use_thinking: bool = False

    use_environment_info: bool = True
    include_dataset_examples: bool = False
    env_info_max_entities_per_type: Optional[int] = 3
    grounded_env_max_summaries: int = 10
    use_validator: bool = True
    memory_guided_generation: bool = False
    use_proposer_memory: bool = True
    embedding_model: Optional[str] = None
    embedding_base_url: Optional[str] = "http://localhost:8201/v1"
    semantic_oversample_multiplier: int = 3
    validator_min_feasibility_score: int = 5
    validator_min_task_completion_score: int = 4
    complexity_schedule: Optional[List[int]] = None
    execution_workers: int = 1
    runs_per_task: int = 1
    repeat_eval_min_feasibility_score: int = 5
    repeat_eval_require_mixed_outcomes: bool = True
    memory_selection_mode: str = "feasible_only"
