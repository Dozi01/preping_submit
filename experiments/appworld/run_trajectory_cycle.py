#!/usr/bin/env python3
"""Trajectory-driven task generation cycle runner."""

import argparse
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from config import ExperimentConfig
from experiment_utils import (
    aggregate_costs,
    apply_optional_overrides,
    initialize_experiment_runtime,
    log_cost_summary,
)
from rich.console import Console
from rich.table import Table

import preping.appworld.task_manager_factory  # noqa: F401  # Register appworld task-manager factory.
from preping.appworld import PrePingMemoryManager, run_single_task_worker
from preping.core.preping import PrePingOrchestrator, create_preping_for_benchmark
from preping.core.preping.proposer_memory import PrePingProposerMemory
from preping.core.memory import MemoryManager, create_memory_store


logger = logging.getLogger(__name__)
console = Console()


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(description="Run Trajectory-Driven Task Generation Cycle")
    parser.add_argument("--max_cycles", type=int, default=3, help="Maximum number of cycles to run")
    parser.add_argument(
        "--tasks_per_cycle",
        type=int,
        default=5,
        help="Number of unique synthetic tasks to generate per cycle",
    )
    parser.add_argument(
        "--runs_per_task",
        type=int,
        default=1,
        help="Execute each generated synthetic task N times (N trajectories per task)",
    )
    parser.add_argument("--num_workers", type=int, default=1, help="Parallel workers for task execution in each cycle")
    parser.add_argument(
        "--repeat_eval_min_feasibility",
        type=int,
        default=5,
        help="Repeated-run aggregate criterion: each run must have feasibility >= this score",
    )
    parser.add_argument(
        "--repeat_eval_require_mixed_outcomes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require mixed success/failure outcomes across repeated runs",
    )
    parser.add_argument(
        "--memory_selection_mode",
        type=str,
        choices=["default", "single_run_include_failure"],
        default=None,
        help="Research mode for validator-based memory selection: default or single_run_include_failure",
    )
    parser.add_argument("--base_task_id", type=str, default=None, help="Base task ID for environment initialization")
    parser.add_argument(
        "--target_apps",
        type=str,
        nargs="+",
        default=None,
        help="Focus on specific apps for task generation",
    )
    parser.add_argument("--global_model_name", type=str, default=None, help="Global model name for all components")
    parser.add_argument("--agent_model_name", type=str, default=None, help="Model name for agent")
    parser.add_argument("--memory_model_name", type=str, default=None, help="Model name for memory components")
    parser.add_argument("--task_manager_model_name", type=str, default=None, help="Model name for task manager")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--tag", type=str, default=None, help="Tag for experiment")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=None, help="Debug mode")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=None, help="Verbose output")
    parser.add_argument("--build_playbook", action="store_true", help="Enable Playbook building during task execution")
    parser.add_argument(
        "--playbook_path",
        type=str,
        default=None,
        help="Path to save Playbook (default: output_dir/playbook.json)",
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Whether to use ground truth for playbook induction (true/false)",
    )
    parser.add_argument(
        "--use_reflector_curator_examples",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep few-shot examples in reflector/curator memory prompts",
    )
    parser.add_argument(
        "--validate_trajectory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable LLM-based trajectory validation before Playbook construction",
    )
    parser.add_argument(
        "--use_environment_info",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Extract/use trajectory-grounded environment summaries for task generation"
        ),
    )
    parser.add_argument(
        "--include_dataset_examples",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include the dataset example section in the task-generation prompt",
    )
    parser.add_argument(
        "--memory_guided_generation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use current memory/playbook to guide task generation: analyze memory gaps "
            "and generate tasks targeting weak/missing areas"
        ),
    )
    parser.add_argument(
        "--use_proposer_memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use Proposer memory summary to condition next-cycle task generation",
    )
    parser.add_argument(
        "--task_embedding_model",
        type=str,
        default=None,
        help="Embedding model for task-manager semantic de-duplication",
    )
    parser.add_argument(
        "--task_embedding_base_url",
        type=str,
        default=None,
        help="Base URL for task-manager embedding server",
    )
    parser.add_argument(
        "--task_semantic_oversample_multiplier",
        type=int,
        default=None,
        help="Oversampling multiplier before semantic filtering when proposer memory + embeddings are enabled",
    )
    parser.add_argument(
        "--complexity_schedule",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Complexity levels per cycle (e.g. 1 1 2 2 3). If shorter than "
            "max_cycles, last value repeats. If omitted, uses generator default."
        ),
    )
    parser.add_argument(
        "--complexity_start",
        type=int,
        default=None,
        help="Starting complexity level for linear schedule (use with --complexity_end)",
    )
    parser.add_argument(
        "--complexity_end",
        type=int,
        default=None,
        help="Ending complexity level for linear schedule (use with --complexity_start)",
    )
    parser.add_argument(
        "--proposer_memory_path",
        type=str,
        default=None,
        help="Path to persist Proposer memory (default: output_dir/proposer_memory.json)",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    summary = run_cycle(args)
    print(
        f"\nCycle completed! {summary['cycles_completed']} cycles, "
        f"{summary['total_tasks_executed']} tasks executed."
    )


def init_config(args) -> ExperimentConfig:
    """Initialize ExperimentConfig from CLI args."""
    config = ExperimentConfig()
    config.split = "train"
    config.tag = "trajectory_cycle"
    apply_optional_overrides(
        args,
        config,
        {
            "tag": "tag",
            "output_dir": "output_dir",
        },
    )
    apply_optional_overrides(args, config, {"debug": "debug_mode", "verbose": "verbose"})
    config.memory_config.include_task_context = True
    apply_optional_overrides(
        args,
        config,
        {
            "use_reflector_curator_examples": "memory_config.use_reflector_curator_examples",
            "use_environment_info": "task_manager_config.use_environment_info",
            "include_dataset_examples": "task_manager_config.include_dataset_examples",
            "validate_trajectory": "task_manager_config.use_validator",
            "memory_guided_generation": "task_manager_config.memory_guided_generation",
            "use_proposer_memory": "task_manager_config.use_proposer_memory",
            "task_embedding_model": "task_manager_config.embedding_model",
            "task_embedding_base_url": "task_manager_config.embedding_base_url",
            "task_semantic_oversample_multiplier": "task_manager_config.semantic_oversample_multiplier",
            "repeat_eval_require_mixed_outcomes": "task_manager_config.repeat_eval_require_mixed_outcomes",
            "memory_selection_mode": "task_manager_config.memory_selection_mode",
        },
    )
    if args.ground_truth is not None:
        config.memory_config.use_ground_truth = args.ground_truth == "true"

    config.task_manager_config.execution_workers = max(1, args.num_workers)
    config.task_manager_config.runs_per_task = max(1, args.runs_per_task)
    config.task_manager_config.repeat_eval_min_feasibility_score = max(1, min(5, args.repeat_eval_min_feasibility))
    config.task_manager_config.semantic_oversample_multiplier = max(
        1, config.task_manager_config.semantic_oversample_multiplier
    )

    if args.complexity_schedule:
        config.task_manager_config.complexity_schedule = args.complexity_schedule
    elif args.complexity_start is not None and args.complexity_end is not None:
        config.task_manager_config.complexity_schedule = build_linear_complexity_schedule(
            args.complexity_start,
            args.complexity_end,
            args.max_cycles,
        )

    if args.global_model_name:
        config.set_model_name(args.global_model_name)
    config.set_component_model_names(
        agent_model_name=args.agent_model_name,
        memory_model_name=args.memory_model_name,
        task_manager_model_name=args.task_manager_model_name,
    )
    return config


def build_linear_complexity_schedule(start: int, end: int, num_cycles: int) -> List[int]:
    """Build a linearly interpolated complexity schedule."""
    if num_cycles <= 1:
        return [start]
    return [round(start + (end - start) * i / (num_cycles - 1)) for i in range(num_cycles)]


def get_complexity_for_cycle(schedule: Optional[List[int]], cycle_index: int) -> Optional[int]:
    """Resolve complexity level for a cycle index."""
    if not schedule:
        return None
    idx = min(cycle_index, len(schedule) - 1)
    return schedule[idx]


def init_task_manager(args, config: ExperimentConfig):
    """Initialize task manager (task generation + optional trajectory validation)."""
    return create_preping_for_benchmark("appworld", config=config)


def init_memory_manager(args, config: ExperimentConfig, output_root_dir: str) -> Optional[MemoryManager]:
    """Initialize MemoryManager for Playbook building if enabled."""
    if not args.build_playbook:
        return None

    playbook_path = Path(args.playbook_path) if args.playbook_path else Path(output_root_dir) / "playbook.json"
    config.memory_config.memory_type = "playbook"
    config.memory_config.memory_path = str(playbook_path)
    memory_manager = create_memory_store(
        memory_type=config.memory_config.memory_type,
        memory_path=config.memory_config.memory_path,
        model_name=config.memory_config.model_name,
        temperature=config.memory_config.temperature,
        use_thinking=config.memory_config.use_thinking,
        embedding_model=config.memory_config.embedding_model,
        embedding_base_url=config.memory_config.embedding_base_url,
        include_task_context=config.memory_config.include_task_context,
        use_ground_truth=config.memory_config.use_ground_truth,
        playbook_max_bullets=config.memory_config.playbook_max_bullets,
        use_reflector_curator_examples=config.memory_config.use_reflector_curator_examples,
    )
    if not isinstance(memory_manager, PrePingMemoryManager):
        raise RuntimeError("Failed to initialize PrePingMemoryManager for playbook mode.")
    logger.info("Playbook building enabled: %s", playbook_path)
    logger.info("Initial playbook: %s bullets", memory_manager.playbook.get_bullet_count())
    logger.info("Use ground truth: %s", config.memory_config.use_ground_truth)
    return memory_manager


def run_cycle(args) -> dict:
    """Run trajectory-driven cycle experiment."""
    if args.runs_per_task < 1:
        raise ValueError("--runs_per_task must be >= 1")
    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1")

    config = init_config(args)
    experiment_name, output_root_dir = initialize_experiment_runtime(
        output_dir=config.output_dir,
        model_name=config.agent_config.model_name,
        tag=config.tag,
        debug_mode=config.debug_mode,
    )

    logger.info("=" * 60)
    logger.info("TRAJECTORY-DRIVEN TASK GENERATION CYCLE")
    logger.info("Max cycles: %s Tasks per cycle: %s", args.max_cycles, args.tasks_per_cycle)
    logger.info("Runs per task: %s", config.task_manager_config.runs_per_task)
    logger.info("Execution workers: %s", config.task_manager_config.execution_workers)
    logger.info("Include dataset examples in generation prompt: %s", config.task_manager_config.include_dataset_examples)
    logger.info("Use Proposer memory for generation: %s", config.task_manager_config.use_proposer_memory)
    logger.info(
        "Repeat-eval criteria: feasibility>=%s, require_mixed_outcomes=%s",
        config.task_manager_config.repeat_eval_min_feasibility_score,
        config.task_manager_config.repeat_eval_require_mixed_outcomes,
    )
    logger.info("Output: %s", output_root_dir)
    logger.info("=" * 60)

    from appworld import load_task_ids

    base_task_id = args.base_task_id or load_task_ids("train")[0] # Use first task from train split as default base task for environment initialization
    logger.info("Base task ID: %s", base_task_id)

    task_manager = init_task_manager(args, config)
    memory_manager = init_memory_manager(args, config, output_root_dir)
    proposer_memory_path = args.proposer_memory_path or str(Path(output_root_dir) / "proposer_memory.json")
    proposer_history = _load_proposer_history(proposer_memory_path)

    orchestrator = PrePingOrchestrator(
        task_manager=task_manager,
        config=config,
        experiment_name=experiment_name,
        output_root_dir=output_root_dir,
        base_task_id=base_task_id,
        max_cycles=args.max_cycles,
        tasks_per_cycle=args.tasks_per_cycle,
        target_apps=args.target_apps,
        memory_manager=memory_manager,
        proposer_memory_path=proposer_memory_path,
        proposer_history=proposer_history,
        task_executor=run_single_task_worker,
    )
    execution_result = orchestrator.run(
        resolve_cycle_complexity=lambda cycle_index: get_complexity_for_cycle(
            config.task_manager_config.complexity_schedule,
            cycle_index,
        )
    )

    _print_cycle_summary_table(execution_result.all_cycle_results)
    logger.info("\nTotal tasks executed: %s", execution_result.total_tasks_executed)
    logger.info(
        "Grounded environment summaries available: %s",
        len(execution_result.all_grounded_environment_summaries),
    )

    all_costs = aggregate_costs(
        total_tasks=execution_result.total_tasks_executed,
        agent=[
            result.get("token_usage")
            for cycle_result in execution_result.all_cycle_results
            for result in cycle_result.get("results", [])
        ],
        task_generator=task_manager.get_generator_cost(),
        task_validator=task_manager.get_validator_cost(),
        memory=memory_manager.get_token_usage_summary() if isinstance(memory_manager, PrePingMemoryManager) else None,
    )
    log_cost_summary(all_costs, total_tasks=execution_result.total_tasks_executed)

    if isinstance(memory_manager, PrePingMemoryManager):
        _log_playbook_summary(memory_manager)

    summary = {
        "experiment_config": asdict(config),
        "run_type": "trajectory_cycle",
        "output_root_dir": output_root_dir,
        "playbook_path": config.memory_config.memory_path if args.build_playbook else None,
        "total_tasks_executed": execution_result.total_tasks_executed,
        "cycles_completed": len(execution_result.all_cycle_results),
        "cost_summary": all_costs,
        "playbook_stats": memory_manager.get_stats() if isinstance(memory_manager, PrePingMemoryManager) else None,
        "validation_enabled": task_manager.use_validator,
        "proposer_memory_path": proposer_memory_path,
        "proposer_memory_stats": {
            "num_entries": execution_result.proposer_memory.get("num_entries", 0),
        },
        "proposer_memory_enabled_for_generation": config.task_manager_config.use_proposer_memory,
    }

    summary_path = os.path.join(output_root_dir, "experiment_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False, default=str)

    _save_all_tasks(
        output_root_dir=output_root_dir,
        config=config,
        all_cycle_results=execution_result.all_cycle_results,
        all_tasks=execution_result.all_tasks,
        target_apps=args.target_apps,
        proposer_memory=execution_result.proposer_memory,
    )

    logger.info("\nSummary saved to: %s", summary_path)
    return summary


def _print_cycle_summary_table(all_cycle_results) -> None:
    table = Table(title="Cycle Results")
    table.add_column("Cycle", style="cyan")
    table.add_column("Tasks", justify="right")
    table.add_column("Success", justify="right", style="green")
    table.add_column("Rate", justify="right")

    for cycle_result in all_cycle_results:
        tasks_executed = cycle_result["tasks_executed"]
        success_rate = (cycle_result["successful"] / tasks_executed * 100) if tasks_executed > 0 else 0
        table.add_row(
            str(cycle_result["cycle"]),
            str(tasks_executed),
            str(cycle_result["successful"]),
            f"{success_rate:.1f}%",
        )

    console.print(table)


def _log_playbook_summary(memory_manager: PrePingMemoryManager) -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info("PLAYBOOK SUMMARY")
    logger.info("=" * 60)
    logger.info("Final bullet count: %s", memory_manager.playbook.get_bullet_count())
    playbook_stats = memory_manager.get_stats()
    for section, count in playbook_stats.get("section_counts", {}).items():
        logger.info("  %s: %s", section, count)
    logger.info("=" * 60)


def _save_all_tasks(
    *,
    output_root_dir: str,
    config: ExperimentConfig,
    all_cycle_results,
    all_tasks,
    target_apps,
    proposer_memory,
) -> None:
    all_tasks_output = {
        "metadata": {
            "model_name": config.task_manager_config.model_name,
            "temperature": config.task_manager_config.temperature,
            "runs_per_task": config.task_manager_config.runs_per_task,
            "execution_workers": config.task_manager_config.execution_workers,
            "include_dataset_examples": config.task_manager_config.include_dataset_examples,
            "use_proposer_memory": config.task_manager_config.use_proposer_memory,
            "total_tasks": len(all_tasks),
            "total_cycles": len(all_cycle_results),
            "target_apps": target_apps,
        },
        "proposer_memory": proposer_memory,
        "tasks": all_tasks,
    }

    all_tasks_path = os.path.join(output_root_dir, "all_tasks.json")
    with open(all_tasks_path, "w", encoding="utf-8") as file:
        json.dump(all_tasks_output, file, indent=2, ensure_ascii=False)

    logger.info("All tasks saved to: %s", all_tasks_path)


def _load_proposer_history(proposer_memory_path: str) -> PrePingProposerMemory:
    path = Path(proposer_memory_path)
    if not path.exists():
        logger.info("Starting with empty Proposer memory: %s", path)
        return PrePingProposerMemory()

    history = PrePingProposerMemory.load(path)
    logger.info("Loaded Proposer memory: %s (%s entries)", path, history.to_dict()["num_entries"])
    return history


if __name__ == "__main__":
    main()
