"""Orchestrator for PrePing cycles."""

from __future__ import annotations

import copy
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from preping.core.memory import MemoryManager

from .proposer_memory import PrePingProposerMemory
from .cycle_logging import PrePingCycleLogger
from .manager import PrePingManager
from .validation_policy import PrePingValidationPolicy


logger = logging.getLogger(__name__)


@dataclass
class CycleExecutionResult:
    """Aggregated outputs produced by a trajectory cycle run."""

    all_cycle_results: List[Dict[str, Any]]
    all_grounded_environment_summaries: List[Dict[str, Any]]
    all_tasks: List[Dict[str, Any]]
    total_tasks_executed: int
    proposer_memory: Dict[str, Any]


@dataclass
class CycleExecutionContext:
    """Serializable execution context shared with worker processes."""

    cycle_dir: str
    base_task_id: str
    split: str
    agent_config: Any
    env_config: Any
    experiment_name: str
    context_sections: Optional[List[str]]
    debug_mode: bool
    verbose: bool
    benchmark_config: str | None = None
    model_name_override: str | None = None
    context_env: Optional[Dict[str, str]] = None


def _build_failed_result(task: Dict[str, Any], task_output_dir: str, error: Exception | str) -> Dict[str, Any]:
    return {
        "success": False,
        "task_id": str(task.get("task_id", "0")),
        "error": str(error),
        "task_text": task.get("instruction", task.get("task_description", "")),
        "trajectory": [],
        "llm_history": [],
        "task_output_dir": task_output_dir,
        "original_task": task,
    }


class PrePingOrchestrator:
    """Main orchestrator for generation -> execution -> validation -> memory cycles."""

    def __init__(
        self,
        *,
        task_manager: PrePingManager,
        config: Any,
        experiment_name: str,
        output_root_dir: str,
        base_task_id: str,
        max_cycles: int,
        tasks_per_cycle: int,
        target_apps: Optional[List[str]] = None,
        memory_manager: Optional[MemoryManager] = None,
        proposer_memory_path: Optional[str] = None,
        proposer_history: Optional[PrePingProposerMemory] = None,
        task_executor: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> None:
        self.task_manager = task_manager
        self.config = config
        self.experiment_name = experiment_name
        self.output_root_dir = output_root_dir
        self.base_task_id = base_task_id
        self.max_cycles = max_cycles
        self.tasks_per_cycle = tasks_per_cycle
        self.target_apps = target_apps
        self.memory_manager = memory_manager
        self.proposer_memory_path = proposer_memory_path
        self.task_executor = task_executor
        self.cycle_logger = PrePingCycleLogger()
        self.proposer_history = proposer_history or PrePingProposerMemory()

    def run(self, *, resolve_cycle_complexity: Callable[[int], Optional[int]]) -> CycleExecutionResult:
        all_cycle_results: List[Dict[str, Any]] = []
        all_grounded_environment_summaries: List[Dict[str, Any]] = []
        all_tasks: List[Dict[str, Any]] = []
        total_tasks_executed = 0
        global_task_id_counter = 0

        for cycle_num in range(self.max_cycles):
            cycle_data = self._run_single_cycle(
                cycle_num=cycle_num,
                all_grounded_environment_summaries=all_grounded_environment_summaries,
                all_tasks=all_tasks,
                global_task_id_counter=global_task_id_counter,
                resolve_cycle_complexity=resolve_cycle_complexity,
            )
            if cycle_data is None:
                break

            all_cycle_results.append(cycle_data["cycle_result_entry"])
            total_tasks_executed += len(cycle_data["cycle_results"])
            global_task_id_counter = cycle_data["global_task_id_counter"]
            all_grounded_environment_summaries = cycle_data["all_grounded_environment_summaries"]

        return CycleExecutionResult(
            all_cycle_results=all_cycle_results,
            all_grounded_environment_summaries=all_grounded_environment_summaries,
            all_tasks=all_tasks,
            total_tasks_executed=total_tasks_executed,
            proposer_memory=self.proposer_history.to_dict(),
        )

    def _run_single_cycle(
        self,
        *,
        cycle_num: int,
        all_grounded_environment_summaries: List[Dict[str, Any]],
        all_tasks: List[Dict[str, Any]],
        global_task_id_counter: int,
        resolve_cycle_complexity: Callable[[int], Optional[int]],
    ) -> Optional[Dict[str, Any]]:
        cycle_index = cycle_num + 1
        logger.info("Cycle %s/%s", cycle_index, self.max_cycles)

        cycle_dir = os.path.join(self.output_root_dir, f"cycle_{cycle_index}")
        os.makedirs(cycle_dir, exist_ok=True)

        current_tasks = self._generate_tasks_for_cycle(
            cycle_num=cycle_num,
            all_grounded_environment_summaries=all_grounded_environment_summaries,
            resolve_cycle_complexity=resolve_cycle_complexity,
        )
        if not current_tasks:
            logger.warning("No tasks generated at cycle %s", cycle_index)
            return None

        current_tasks = self.task_manager.prepare_generated_tasks(
            tasks=current_tasks,
            cycle_dir=cycle_dir,
            cycle_index=cycle_index,
        )
        if not current_tasks:
            logger.warning("No executable tasks remained after task preparation at cycle %s", cycle_index)
            return None

        self._save_generation_artifacts(cycle_dir=cycle_dir)

        execution_tasks = self._build_execution_tasks(current_tasks)
        previous_all_tasks_count = len(all_tasks)
        global_task_id_counter = self.task_manager.assign_task_ids_and_collect(
            current_tasks=execution_tasks,
            cycle_num=cycle_num,
            global_task_id_counter=global_task_id_counter,
            all_tasks=all_tasks,
        )
        cycle_task_records = all_tasks[previous_all_tasks_count:]

        cycle_results = self._execute_cycle_tasks(
            tasks=execution_tasks,
            cycle_dir=cycle_dir,
            context_sections=[],
        )

        validation_stats = self.task_manager.validate_results(cycle_results)
        task_diagnostics, memory_input_results = PrePingValidationPolicy.post_process_validation_results(
            cycle_results=cycle_results,
            validation_stats=validation_stats,
            runs_per_task=self.config.task_manager_config.runs_per_task,
            repeat_eval_min_feasibility_score=self.config.task_manager_config.repeat_eval_min_feasibility_score,
            repeat_eval_require_mixed_outcomes=self.config.task_manager_config.repeat_eval_require_mixed_outcomes,
            memory_selection_mode=self.config.task_manager_config.memory_selection_mode,
        )
        if self.memory_manager:
            added_count = self.task_manager.add_results_to_memory(memory_input_results, self.memory_manager)
            self.cycle_logger.save_memory_update(
                cycle_dir=cycle_dir,
                added_count=added_count,
                candidate_count=len(memory_input_results),
            )

        PrePingValidationPolicy.attach_validation_to_tasks(
            tasks_to_update=cycle_task_records,
            cycle_results=cycle_results,
            task_diagnostics=task_diagnostics,
            validation_stats=validation_stats,
        )
        proposer_history_entries = self.proposer_history.update_from_cycle(
            cycle_index=cycle_index,
            cycle_results=cycle_results,
            task_diagnostics=task_diagnostics,
        )
        self._save_proposer_memory_artifacts(cycle_dir=cycle_dir)

        successful = sum(1 for result in cycle_results if result.get("success", False))
        cycle_result_entry = {
            "cycle": cycle_index,
            "unique_tasks_generated": len(current_tasks),
            "tasks_executed": len(cycle_results),
            "successful": successful,
            "results": cycle_results,
        }
        if validation_stats:
            cycle_result_entry["validation_stats"] = validation_stats
        if task_diagnostics:
            cycle_result_entry["task_diagnostics_stats"] = task_diagnostics.get("summary", {})

        use_env_info = self.config.task_manager_config.use_environment_info
        if cycle_num < self.max_cycles - 1:
            all_grounded_environment_summaries = self.task_manager.extract_and_merge_grounded_environment_summaries(
                cycle_index=cycle_index,
                cycle_results=cycle_results,
                all_grounded_environment_summaries=all_grounded_environment_summaries,
                use_env_info=use_env_info,
            )
            self.cycle_logger.save_grounded_environment_summaries(
                cycle_dir=cycle_dir,
                payload=all_grounded_environment_summaries,
            )

        logger.info(
            "Cycle %s done: %s/%s successful",
            cycle_index,
            successful,
            len(cycle_results),
        )
        return {
            "cycle_result_entry": cycle_result_entry,
            "cycle_results": cycle_results,
            "global_task_id_counter": global_task_id_counter,
            "all_grounded_environment_summaries": all_grounded_environment_summaries,
            "proposer_history_entries": proposer_history_entries,
        }

    def _save_proposer_memory_artifacts(self, *, cycle_dir: str) -> None:
        payload = self.proposer_history.to_dict()
        self.cycle_logger.save_proposer_memory(
            path=str(Path(cycle_dir) / "proposer_memory.json"),
            payload=payload,
        )
        if self.proposer_memory_path:
            self.cycle_logger.save_proposer_memory(
                path=self.proposer_memory_path,
                payload=payload,
            )

    def _generate_tasks_for_cycle(
        self,
        *,
        cycle_num: int,
        all_grounded_environment_summaries: List[Dict[str, Any]],
        resolve_cycle_complexity: Callable[[int], Optional[int]],
    ) -> List[Dict[str, Any]]:
        env_info_for_generation = self._prepare_grounded_environment_summaries_for_generation(
            cycle_num=cycle_num,
            all_grounded_environment_summaries=all_grounded_environment_summaries,
        )

        memory_for_generation: Optional[List[str]] = None
        if self.config.task_manager_config.memory_guided_generation and self.memory_manager:
            memory_for_generation = self.memory_manager.get_memory()
        task_history_summary: Optional[Dict[str, Any]] = None
        if self.config.task_manager_config.use_proposer_memory:
            task_history_summary = self.proposer_history.get_generation_summary()

        return self.task_manager.generate_tasks(
            environment_info=env_info_for_generation,
            num_tasks=self.tasks_per_cycle,
            target_apps=self.target_apps,
            memory_context=memory_for_generation,
            task_history_summary=task_history_summary,
            complexity_level=resolve_cycle_complexity(cycle_num),
        )

    def _prepare_grounded_environment_summaries_for_generation(
        self,
        *,
        cycle_num: int,
        all_grounded_environment_summaries: List[Dict[str, Any]],
    ) -> Optional[str]:
        task_manager_config = self.config.task_manager_config
        return self.task_manager.prepare_grounded_environment_summaries_for_generation(
            all_grounded_environment_summaries=all_grounded_environment_summaries,
            use_environment_info=task_manager_config.use_environment_info,
            seed=getattr(self.config, "seed", 0),
            cycle_num=cycle_num,
            max_summaries=getattr(task_manager_config, "grounded_env_max_summaries", 10),
        )

    def _save_generation_artifacts(self, *, cycle_dir: str) -> None:
        generation_debug_artifact = self.task_manager.get_generation_debug_artifact()
        self.cycle_logger.save_generation_debug_artifact(
            cycle_dir=cycle_dir,
            payload=generation_debug_artifact,
        )

    def _build_execution_tasks(self, current_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        runs_per_task = self.config.task_manager_config.runs_per_task
        execution_tasks: List[Dict[str, Any]] = []
        for synthetic_idx, task in enumerate(current_tasks):
            for run_idx in range(runs_per_task):
                task_instance = copy.deepcopy(task)
                task_instance["synthetic_task_index"] = synthetic_idx
                task_instance["repeat_index"] = run_idx + 1
                task_instance["repeat_total"] = runs_per_task
                if self.memory_manager:
                    task_query = task_instance.get("instruction") or task_instance.get("task_description")
                    task_instance["context_sections"] = self.memory_manager.get_memory(
                        task_description=task_query,
                        max_bullets=self.config.memory_config.playbook_max_bullets,
                    )
                execution_tasks.append(task_instance)
        return execution_tasks

    def _execute_cycle_tasks(
        self,
        *,
        tasks: List[Dict[str, Any]],
        cycle_dir: str,
        context_sections: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        context = CycleExecutionContext(
            cycle_dir=cycle_dir,
            base_task_id=self.base_task_id,
            split=self.config.split,
            agent_config=getattr(self.config, "agent_config", None),
            env_config=getattr(self.config, "env_config", None),
            experiment_name=self.experiment_name,
            context_sections=context_sections,
            debug_mode=bool(getattr(self.config, "debug_mode", False)),
            verbose=bool(getattr(self.config, "verbose", False)),
            benchmark_config=getattr(self.config, "benchmark_config", None),
            model_name_override=getattr(self.config, "model_name_override", None),
            context_env=getattr(self.config, "context_env", None),
        )
        workers = max(1, int(self.config.task_manager_config.execution_workers))
        worker_task_args = [self._build_worker_task_args(task, cycle_dir, context) for task in tasks]

        if workers <= 1:
            results: List[Dict[str, Any]] = []
            for idx, task_args in enumerate(worker_task_args):
                task = tasks[idx]
                task_output_dir = str(task_args["task_output_dir"])
                try:
                    results.append(self.task_executor(task_args))
                except Exception as error:  # pragma: no cover - defensive runtime path
                    logger.error("Task %s failed in direct worker path: %s", task.get("task_id", "0"), error)
                    results.append(_build_failed_result(task, task_output_dir, error))
            return results

        results: List[Optional[Dict[str, Any]]] = [None] * len(tasks)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(
                    self.task_executor,
                    task_args,
                ): idx
                for idx, task_args in enumerate(worker_task_args)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                task = tasks[idx]
                task_id = str(task.get("task_id", "0"))
                task_output_dir = os.path.join(cycle_dir, f"task_{task_id}")
                try:
                    results[idx] = future.result()
                except Exception as error:  # pragma: no cover - defensive runtime path
                    logger.error("Task %s failed in process pool: %s", task_id, error)
                    results[idx] = _build_failed_result(task, task_output_dir, error)

        return [result or {} for result in results]

    def _build_worker_task_args(
        self,
        task: Dict[str, Any],
        cycle_dir: str,
        context: CycleExecutionContext,
    ) -> Dict[str, Any]:
        custom_task_args = self.task_manager.build_worker_task_args(
            task=task,
            cycle_dir=cycle_dir,
            context=context,
        )
        if custom_task_args is not None:
            return custom_task_args

        task_id = str(task.get("task_id", "0"))
        task_output_dir = os.path.join(cycle_dir, f"task_{task_id}")
        cycle_name = os.path.basename(cycle_dir.rstrip(os.sep))
        execution_experiment_name = f"{context.experiment_name}/{cycle_name}/task_{task_id}"

        return {
            "task_id": task_id,
            "execution_task_id": context.base_task_id,
            "split": context.split,
            "task_output_dir": task_output_dir,
            "agent_config": context.agent_config,
            "env_config": copy.deepcopy(context.env_config),
            "experiment_name": execution_experiment_name,
            "agent_type": "task",
            "override_instruction": task.get("instruction", ""),
            "context_sections": task.get("context_sections", context.context_sections),
            "debug_mode": context.debug_mode,
            "verbose": context.verbose,
            "original_task": task,
        }
