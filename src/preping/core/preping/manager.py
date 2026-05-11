"""PrePing coordination logic."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from preping.core.memory import MemoryManager

from .proposer import PrePingProposer
from .cycle_logging import PrePingCycleLogger
from .grounded_environment import PrePingGroundedEnvironmentSummaries
from .validation import PrePingValidationGate
from .validation_policy import PrePingValidationPolicy


logger = logging.getLogger(__name__)


class PrePingManager:
    """Coordinate PrePing generation, validation, and memory updates.

    Serves as the benchmark-extensible facade. Benchmark-specific adapters
    override hooks such as
    ``generate_tasks``, ``validate_results``, ``prepare_generated_tasks``,
    ``build_worker_task_args``, and ``_add_result_to_memory`` to inject
    benchmark-specific behaviour.
    """

    def __init__(
        self,
        model_name: str = "deepseek/deepseek-chat",
        temperature: float = 0.7,
        use_thinking: bool = False,
        verbose: bool = False,
        *,
        use_validator: bool = False,
        min_feasibility_score: int = 5,
        min_task_completion_score: int = 4,
        trajectory_file_pattern: str = "**/trajectory.json",
        complexity_level: int = 1,
        task_generation_prompt: Optional[str] = None,
        environment_extraction_prompt: Optional[str] = None,
        grounded_environment_summary_prompt: Optional[str] = None,
        complexity_level_descriptions: Optional[Dict[int, str]] = None,
        validation_prompt: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        semantic_oversample_multiplier: int = 3,
        include_dataset_examples: bool = True,
        prompt_context_provider: Optional[Callable[[Optional[List[str]]], str]] = None,
        dataset_examples_section: Optional[str] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.use_thinking = use_thinking
        self.verbose = verbose
        self.use_validator = use_validator

        self._proposer = PrePingProposer(
            model_name=self.model_name,
            temperature=self.temperature,
            use_thinking=self.use_thinking,
            verbose=self.verbose,
            trajectory_file_pattern=trajectory_file_pattern,
            complexity_level=complexity_level,
            task_generation_prompt=task_generation_prompt,
            environment_extraction_prompt=environment_extraction_prompt,
            grounded_environment_summary_prompt=grounded_environment_summary_prompt,
            complexity_level_descriptions=complexity_level_descriptions,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            semantic_oversample_multiplier=semantic_oversample_multiplier,
            include_dataset_examples=include_dataset_examples,
            prompt_context_provider=prompt_context_provider,
            dataset_examples_section=dataset_examples_section,
        )
        self._validation_gate: Optional[PrePingValidationGate] = None
        if use_validator:
            self._validation_gate = PrePingValidationGate(
                model_name=self.model_name,
                temperature=self.temperature,
                use_thinking=self.use_thinking,
                verbose=self.verbose,
                min_feasibility_score=min_feasibility_score,
                min_task_completion_score=min_task_completion_score,
                validation_prompt=validation_prompt,
            )
        self._cycle_logger = PrePingCycleLogger()

    # -- Task generation (overridable) ------------------------------------

    def generate_tasks(
        self,
        environment_info: Optional[Dict[str, Any] | str] = None,
        memory_context: Optional[List[str]] = None,
        task_history_summary: Optional[Dict[str, Any]] = None,
        num_tasks: int = 5,
        target_apps: Optional[List[str]] = None,
        complexity_level: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return self._proposer.generate_tasks(
            environment_info=environment_info,
            num_tasks=num_tasks,
            target_apps=target_apps,
            memory_context=memory_context,
            task_history_summary=task_history_summary,
            complexity_level=complexity_level,
        )

    # -- Validation (overridable) -----------------------------------------

    def validate_results(self, results: List[Dict]) -> Dict[str, Any]:
        validator = self._validation_gate
        if not validator:
            return {}

        validation_results = validator.validate_batch(results)
        validation_stats = validator.get_validation_summary(validation_results)
        for result, validation in zip(results, validation_results):
            result["validation"] = validation.to_dict()
            self._cycle_logger.save_validation_result(result)
        logger.info("Validation: %s/%s valid", validation_stats["valid_count"], validation_stats["total_validated"])
        return validation_stats

    def post_process_validation_results(
        self,
        *,
        cycle_results: List[Dict[str, Any]],
        validation_stats: Dict[str, Any],
        runs_per_task: int,
        repeat_eval_min_feasibility_score: int,
        repeat_eval_require_mixed_outcomes: bool,
        memory_selection_mode: str = "feasible_only",
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Apply validation policy to select solver-memory candidates."""
        return PrePingValidationPolicy.post_process_validation_results(
            cycle_results=cycle_results,
            validation_stats=validation_stats,
            runs_per_task=runs_per_task,
            repeat_eval_min_feasibility_score=repeat_eval_min_feasibility_score,
            repeat_eval_require_mixed_outcomes=repeat_eval_require_mixed_outcomes,
            memory_selection_mode=memory_selection_mode,
        )

    # -- Memory ingestion (overridable) -----------------------------------

    def add_results_to_memory(self, results: List[Dict], memory_manager: MemoryManager) -> int:
        for result in results:
            self._add_result_to_memory(memory_manager, result)

        logger.info("Added %s trajectories to memory", len(results))
        return len(results)

    @staticmethod
    def _add_result_to_memory(memory_manager: MemoryManager, result: Dict) -> None:
        task_dict = result.get("original_task") or {}
        llm_history = result.get("llm_history") or []
        validation = result.get("validation") or {}
        if llm_history:
            trajectory_data = llm_history
            trajectory_format = "llm_history"
        else:
            trajectory_data = result.get("trajectory") or []
            trajectory_format = "action_output"

        # Use validator decision when available; execution success
        # can disagree with the validator's task-completion judgment.
        validation_result = str(validation.get("validation_result", "")).strip().lower()
        if validation_result == "success":
            episode_result = "success"
        elif validation_result == "failure":
            episode_result = "failure"
        elif validation_result == "invalid":
            logger.warning(
                "Skipping invalid trajectory during memory ingestion for task_id=%s",
                result.get("task_id"),
            )
            return
        else:
            episode_result = "success" if result.get("success", False) else "failure"

        memory_manager.process_episode(
            task_description=(
                result.get("task_text")
                or task_dict.get("instruction")
                or task_dict.get("task_description")
                or ""
            ),
            trajectory=trajectory_data,
            result=episode_result,
            task_id=result.get("task_id"),
            trajectory_format=trajectory_format,
            debug_output_dir=(
                str(Path(result["task_output_dir"]) / "memory_debug")
                if result.get("task_output_dir")
                else None
            ),
        )

    # -- Grounded environment summaries (overridable) ---------------------

    def extract_and_merge_grounded_environment_summaries(
        self,
        *,
        cycle_index: int,
        cycle_results: List[Dict[str, Any]],
        all_grounded_environment_summaries: List[Dict[str, Any]],
        use_env_info: bool,
    ) -> List[Dict[str, Any]]:
        if not use_env_info:
            return all_grounded_environment_summaries

        representative_results = PrePingValidationPolicy.select_representative_trajectory_results(cycle_results)
        if not representative_results:
            return all_grounded_environment_summaries

        summary_entries = self._proposer.summarize_environment_from_results(
            cycle_index=cycle_index,
            representative_results=representative_results,
        )
        if not summary_entries:
            return all_grounded_environment_summaries
        return PrePingGroundedEnvironmentSummaries.merge_summary_entries(
            all_grounded_environment_summaries,
            summary_entries,
        )

    @staticmethod
    def prepare_grounded_environment_summaries_for_generation(
        *,
        all_grounded_environment_summaries: List[Dict[str, Any]],
        use_environment_info: bool,
        seed: int,
        cycle_num: int,
        max_summaries: int = 10,
    ) -> Optional[str]:
        return PrePingGroundedEnvironmentSummaries.prepare_for_generation(
            summary_entries=all_grounded_environment_summaries,
            use_environment_info=use_environment_info,
            seed=seed,
            cycle_num=cycle_num,
            max_summaries=max_summaries,
        )

    # -- Benchmark hooks (override in subclasses) -------------------------

    def prepare_generated_tasks(
        self,
        *,
        tasks: List[Dict[str, Any]],
        cycle_dir: str,
        cycle_index: int,
    ) -> List[Dict[str, Any]]:
        """Allow benchmark-specific task materialization before execution."""
        return tasks

    def build_worker_task_args(
        self,
        *,
        task: Dict[str, Any],
        cycle_dir: str,
        context: Any,
    ) -> Optional[Dict[str, Any]]:
        """Allow benchmark-specific worker args without changing the orchestrator."""
        return None

    # -- Bookkeeping helpers ----------------------------------------------

    def assign_task_ids_and_collect(
        self,
        current_tasks: List[Dict],
        cycle_num: int,
        global_task_id_counter: int,
        all_tasks: List[Dict],
    ) -> int:
        for task in current_tasks:
            task["task_id"] = str(global_task_id_counter)
            task["cycle"] = cycle_num + 1
            all_tasks.append(task.copy())
            global_task_id_counter += 1
        return global_task_id_counter

    def get_generation_debug_artifact(self) -> Dict[str, Any]:
        return self._proposer.get_last_generation_debug_artifact()

    def get_generator_token_usage(self) -> Dict[str, Any]:
        return self._proposer.llm.get_token_usage_stats()

    def get_generator_cost(self) -> Dict[str, Any]:
        return self._proposer.llm.get_cost_breakdown()

    def get_validator_token_usage(self) -> Dict[str, Any]:
        if self._validation_gate is None:
            return {}
        return self._validation_gate.get_token_usage_stats()

    def get_validator_cost(self) -> Dict[str, Any]:
        if self._validation_gate is None:
            return {}
        return self._validation_gate.get_cost_breakdown()


# -- Factory registry (merged from registry.py) ----------------------------

PrePingFactory = Callable[..., PrePingManager]

_REGISTRY: Dict[str, PrePingFactory] = {}


def register_preping_factory(benchmark_id: str):
    """Decorator to register a PrePing factory for a benchmark."""

    def decorator(factory: PrePingFactory) -> PrePingFactory:
        _REGISTRY[benchmark_id] = factory
        return factory

    return decorator


def get_preping_factory(benchmark_id: str) -> PrePingFactory:
    """Return registered PrePing factory for ``benchmark_id``."""
    if benchmark_id not in _REGISTRY:
        raise KeyError(f"PrePing factory '{benchmark_id}' is not registered")
    return _REGISTRY[benchmark_id]


def create_preping_for_benchmark(benchmark_id: str, *args: Any, **kwargs: Any) -> PrePingManager:
    """Instantiate a benchmark-specific PrePing manager via registry."""
    factory = get_preping_factory(benchmark_id)
    return factory(*args, **kwargs)


def available_preping_factories() -> Dict[str, PrePingFactory]:
    """Return a copy of currently registered task-manager factories."""
    return dict(_REGISTRY)
