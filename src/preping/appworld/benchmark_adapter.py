"""AppWorld benchmark adapter for the shared runner."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Mapping, Sequence

from preping.appworld import AppWorldTaskSource, run_single_task_worker
from preping.core.benchmark_base import BaseBenchmark
from preping.core.interfaces.benchmark import BenchmarkAdapter, BenchmarkResult, BenchmarkTask, RunnerConfig


try:
    from experiments.appworld.config import ExperimentConfig
    from experiments.appworld.experiment_utils import (
        aggregate_costs,
        get_task_instruction,
        initialize_experiment_runtime,
        log_cost_summary,
    )
except ModuleNotFoundError:
    # Support execution from experiments/appworld where modules are imported as local files.
    from config import ExperimentConfig
    from experiment_utils import (  # type: ignore[no-redef]
        aggregate_costs,
        get_task_instruction,
        initialize_experiment_runtime,
        log_cost_summary,
    )


logger = logging.getLogger(__name__)


class AppWorldBenchmark(BaseBenchmark, BenchmarkAdapter):
    name = "appworld"

    def __init__(
        self,
        *,
        config: ExperimentConfig | None = None,
        selected_task_ids: Sequence[str] | None = None,
        agent_type: str = "task",
        memory_manager: Any | None = None,
    ):
        self.config = config or ExperimentConfig()
        self.selected_task_ids = list(selected_task_ids) if selected_task_ids else None
        self.agent_type = agent_type
        self.memory_manager = memory_manager
        self.experiment_name: str | None = None
        self.output_root_dir: str | None = None
        self._start_time: float | None = None
        self._task_costs: dict[str, float] = {}
        self._all_token_usages: List[dict] = []
        self.task_source = AppWorldTaskSource()

    def prepare(self, *, runner_config: RunnerConfig) -> Mapping[str, str]:
        exp_name, output_root_dir = initialize_experiment_runtime(
            output_dir=self.config.output_dir,
            model_name=self.config.agent_config.model_name,
            tag=self.config.tag,
            split=self.config.split,
            debug_mode=self.config.debug_mode,
        )
        self._start_time = time.time()
        self.experiment_name = exp_name
        self.output_root_dir = output_root_dir
        return {"experiment_name": exp_name, "output_root_dir": output_root_dir}

    def iter_tasks(self, *, runner_config: RunnerConfig, context: Mapping[str, Any] | None = None):
        task_ids = self.selected_task_ids or self.task_source.load_task_ids(self.config.split)
        for task_id in task_ids:
            yield BenchmarkTask(task_id=task_id, split=self.config.split)

    def _get_task_context_sections(self, task_id: str) -> List[str] | None:
        if self.memory_manager is None:
            return None

        task_description = get_task_instruction(task_id)
        max_bullets = self.config.memory_config.playbook_max_bullets
        if hasattr(self.memory_manager, "get_context_sections"):
            return self.memory_manager.get_context_sections(
                task_description=task_description,
                max_bullets=max_bullets,
            )
        if hasattr(self.memory_manager, "get_memory"):
            return self.memory_manager.get_memory(
                task_description=task_description,
                max_bullets=max_bullets,
            )
        logger.warning("Memory manager has no supported retrieval method; expected get_context_sections/get_memory")
        return None

    def build_task_args(
        self,
        task: BenchmarkTask,
        *,
        runner_config: RunnerConfig,
        context: Mapping[str, Any] | None = None,
        position: int = 0,
        total: int = 0,
    ) -> Mapping[str, Any]:
        task_context = self._get_task_context_sections(task.task_id)

        return {
            "task_id": task.task_id,
            "task_index": position,
            "total_tasks": total,
            "split": task.split,
            "output_root_dir": self.output_root_dir,
            "task_output_dir": os.path.join(str(self.output_root_dir), f"task_{task.task_id}"),
            "agent_config": self.config.agent_config,
            "env_config": self.config.env_config,
            "experiment_name": self.experiment_name,
            "agent_type": self.agent_type,
            "context_sections": task_context or None,
            "debug_mode": self.config.debug_mode,
            "verbose": self.config.verbose,
            "timeout_seconds": self.config.task_timeout_seconds,
        }

    def should_skip(
        self,
        task_args: Mapping[str, Any],
        *,
        runner_config: RunnerConfig,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        task_output_dir = Path(str(task_args["task_output_dir"]))
        return not runner_config.continue_existing and (task_output_dir / "results.json").exists()

    def get_worker(self):
        return run_single_task_worker

    def build_result(
        self,
        raw_result: Any,
        task_args: Mapping[str, Any],
        *,
        runner_config: RunnerConfig,
        context: Mapping[str, Any] | None = None,
    ) -> BenchmarkResult:
        result = super().build_result(raw_result, task_args, runner_config=runner_config, context=context)
        task_output_dir = Path(str(task_args["task_output_dir"]))
        task_output_dir.mkdir(parents=True, exist_ok=True)

        payload = dict(result.payload)
        payload.setdefault("task_id", result.task_id)
        payload.setdefault("split", result.split)
        payload.setdefault("task_output_dir", str(task_args["task_output_dir"]))
        payload.setdefault("attempt_index", task_args.get("attempt_index"))
        payload.setdefault("success", result.success)
        payload.setdefault("iterations", 0)
        payload.setdefault("done", False)
        payload.setdefault("info", {})
        payload.setdefault("final_reward", False)
        payload.setdefault("termination_reason", "timeout" if payload.get("error") == "timeout" else "error")
        with (task_output_dir / "results.json").open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        result.payload = payload

        if result.skipped:
            return result

        token_usage = payload.get("token_usage")
        if token_usage:
            self._all_token_usages.append(token_usage)
            self._task_costs[result.task_id] = token_usage.get("total_cost_usd", 0.0)
        return result

    def summarize(
        self,
        results: Sequence[BenchmarkResult],
        *,
        runner_config: RunnerConfig,
        context: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        successful = [r.task_id for r in results if r.success and not r.skipped]
        failed = [r.task_id for r in results if not r.success and not r.skipped]
        skipped = [r.task_id for r in results if r.skipped]
        total_time = (time.time() - self._start_time) if self._start_time else 0.0

        all_costs = aggregate_costs(total_tasks=len(results), agent=self._all_token_usages)
        all_costs["task_costs"] = self._task_costs
        log_cost_summary(all_costs, total_tasks=len(results))

        summary = {
            "experiment_config": asdict(self.config),
            "success_rate": len(successful) / max(1, len(results)),
            "total_tasks": len(results),
            "successful_tasks": successful,
            "failed_tasks": failed,
            "skipped_tasks": skipped,
            "total_time_seconds": total_time,
            "run_timestamp": time.time(),
            "cost_summary": all_costs,
        }

        if self.output_root_dir:
            summary_path = Path(self.output_root_dir) / "experiment_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("w", encoding="utf-8") as file:
                json.dump(summary, file, indent=2)
        else:
            logger.warning("output_root_dir is not set; experiment summary was not written")

        return summary
