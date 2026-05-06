"""
Experiment configuration for AppWorld experiments.

This module contains configuration dataclasses for running experiments 
with run_all_parallel.py.
"""

from dataclasses import dataclass, field
from typing import Optional

from preping.appworld import AppWorldAgentConfig
from preping.appworld import AppWorldEnvConfig
from preping.core.preping.config import TaskManagerConfig


@dataclass
class MemoryConfig:
    """Memory/context configuration for agent context.
    """
    # LLM settings for memory (Reflector/Curator)
    model_name: str = "deepseek/deepseek-chat"
    temperature: float = 0.7
    use_thinking: bool = False

    # Memory type selection
    memory_type: Optional[str] = None  # "playbook"
    memory_path: Optional[str] = None  # Path to existing memory file

    # Playbook configuration (PrePing format)
    playbook_path: Optional[str] = None  # Path to playbook JSON file
    
    # Embedding model configuration for playbook semantic retrieval
    embedding_model: Optional[str] = None
    embedding_base_url: Optional[str] = None
    
    # Playbook retrieval settings 
    playbook_max_bullets: Optional[int] = None

    # Memory construction
    use_ground_truth: bool = True  # Whether to use GT labels (final_reward) during playbook induction
    include_task_context: bool = True  # If False, exclude task descriptions from prompts (for exploration trajectories)
    use_reflector_curator_examples: bool = False  # Keep few-shot examples in reflector/curator prompts

    # Task-informed adaptation baselines
    adaptation_mode: str = "offline"  # "offline" (train then freeze) or "online" (keep updating)
    save_intermediate: bool = True  # Save playbook after each task


@dataclass
class ExperimentConfig:
    """Configuration for running experiments (run_all_parallel.py).
    
    This config contains:
    - Dataset/experiment settings (split, tag, output_dir)
    - Agent configuration (agent_config): model and prompts
    - Memory configuration (memory_config): playbook, embedding, induction, and adaptation
    - Execution (num_workers, timeout, retries)
    """
    
    # === Shared settings (Single Source of Truth) ===
    split: str = "train"  # train, dev, test_normal, test_challenge
    debug_mode: bool = False
    verbose: bool = True
    
    # Experiment settings
    tag: str = "base"
    output_dir: str = "./outputs"
    seed: int = 42
    
    # Parallel execution settings
    num_workers: int = 30
    task_timeout_seconds: int = 1800
    max_retries: int = 2
    
    # Agent configuration (model, prompts, etc. - agent-specific only)
    agent_config: AppWorldAgentConfig = field(default_factory=lambda: AppWorldAgentConfig(
        model_name="deepseek/deepseek-chat",
        prompt_file="./prompts/instruction.jinja",
        seed=42
    ))
    
    # Environment configuration (max_iter, env-specific only)
    env_config: AppWorldEnvConfig = field(default_factory=AppWorldEnvConfig)
    
    # Memory/context configuration (playbook, embedding, induction, adaptation)
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    
    # Task manager configuration (task generation, trajectory validation)
    task_manager_config: TaskManagerConfig = field(default_factory=TaskManagerConfig)
    
    def set_model_name(self, model_name: str) -> None:
        """Set model_name across all component configs (Agent, Memory, TaskManager)."""
        self.agent_config.model_name = model_name
        self.memory_config.model_name = model_name
        self.task_manager_config.model_name = model_name

    def set_component_model_names(
        self,
        *,
        agent_model_name: Optional[str] = None,
        memory_model_name: Optional[str] = None,
        task_manager_model_name: Optional[str] = None,
    ) -> None:
        """Set component model names individually when provided."""
        if agent_model_name is not None:
            self.agent_config.model_name = agent_model_name
        if memory_model_name is not None:
            self.memory_config.model_name = memory_model_name
        if task_manager_model_name is not None:
            self.task_manager_config.model_name = task_manager_model_name
