#!/usr/bin/env python3
"""Entry point for running AppWorld experiments via the shared benchmark runner."""

from __future__ import annotations

import argparse
import logging

from config import ExperimentConfig
from experiment_utils import apply_optional_overrides

from preping.appworld import AppWorldTaskSource, create_appworld_memory_adapter
from preping.appworld.benchmark_adapter import AppWorldBenchmark
from preping.core.interfaces.benchmark import RunnerConfig
from preping.runner import run_benchmark


logger = logging.getLogger(__name__)


def _build_parser(config: ExperimentConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AppWorld experiments")
    parser.add_argument("--split", type=str, default=None, help=f"Dataset split (default: {config.split})")
    parser.add_argument("--task_id", type=str, default=None, help="Specific task ID to run")
    parser.add_argument("--task_ids", type=str, nargs="+", default=None, help="List of specific task IDs to run")
    parser.add_argument("--tag", type=str, default=None, help=f"Tag for the experiment (default: {config.tag})")
    parser.add_argument(
        "--global_model_name",
        type=str,
        default=None,
        help=f"Global model name to use (default: {config.agent_config.model_name})",
    )
    parser.add_argument("--agent_model_name", type=str, default=None, help="Model name for agent")
    parser.add_argument("--memory_model_name", type=str, default=None, help="Model name for memory components")
    parser.add_argument("--task_manager_model_name", type=str, default=None, help="Model name for task manager")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=f"Temperature for LLM generation (default: {config.agent_config.temperature})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=f"Output directory (default: {config.output_dir})",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=f"Enable debug mode (default: {config.debug_mode})",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=None,
        help=f"Maximum iterations per task (default: {config.env_config.max_iter})",
    )
    parser.add_argument("--continue_existing", action="store_true", help="Continue from existing results")
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=f"Verbose output (default: {config.verbose})",
    )
    parser.add_argument("--seed", type=int, default=None, help=f"Random seed for LLM generation (default: {config.seed})")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help=f"Number of parallel workers (default: {config.num_workers})",
    )

    parser.add_argument(
        "--memory_type",
        type=str,
        default=None,
        choices=["playbook"],
        help="Type of memory to use",
    )
    parser.add_argument("--memory_path", type=str, default=None, help="Path to existing memory file")
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help=f"Prompt template file path (default: {config.agent_config.prompt_file})",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=None,
        help="Embedding model for semantic retrieval",
    )
    parser.add_argument(
        "--embedding_base_url",
        type=str,
        default=None,
        help="Base URL for embedding server",
    )
    parser.add_argument(
        "--playbook_max_bullets",
        type=int,
        default=None,
        help=f"Maximum bullets to retrieve for each task (default: {config.memory_config.playbook_max_bullets})",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="task",
        choices=["task", "random_explorer", "guided_explorer"],
        help="Agent type: task (default), random_explorer, or guided_explorer",
    )
    return parser


def _apply_cli_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    apply_optional_overrides(
        args,
        config,
        {
            "split": "split",
            "tag": "tag",
            "output_dir": "output_dir",
            "seed": "seed",
            "num_workers": "num_workers",
        },
    )
    apply_optional_overrides(args, config, {"debug": "debug_mode", "verbose": "verbose"})

    if args.global_model_name is not None:
        config.set_model_name(args.global_model_name)
    config.set_component_model_names(
        agent_model_name=args.agent_model_name,
        memory_model_name=args.memory_model_name,
        task_manager_model_name=args.task_manager_model_name,
    )
    apply_optional_overrides(
        args,
        config,
        {
            "temperature": "agent_config.temperature",
            "prompt_file": "agent_config.prompt_file",
            "max_iter": "env_config.max_iter",
            "memory_type": "memory_config.memory_type",
            "memory_path": "memory_config.memory_path",
            "embedding_model": "memory_config.embedding_model",
            "embedding_base_url": "memory_config.embedding_base_url",
            "playbook_max_bullets": "memory_config.playbook_max_bullets",
        },
    )

    # Keep seed synchronized with agent config.
    config.agent_config.seed = config.seed


def _resolve_task_ids(args: argparse.Namespace, split: str) -> list[str]:
    task_source = AppWorldTaskSource()
    if args.task_id:
        task_ids = [args.task_id]
        print(f"Running single task: {args.task_id}")
        return task_ids
    if args.task_ids:
        print(f"Running specified tasks: {args.task_ids}")
        return list(args.task_ids)
    task_ids = task_source.load_task_ids(split)
    print(f"Running all tasks from {split} split: {len(task_ids)} tasks")
    return task_ids


def _print_completion_summary(config: ExperimentConfig, summary: dict) -> None:
    total_time = summary.get("total_time_seconds", 0.0)
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"\n{'=' * 60}")
    print("APPWORLD EXPERIMENT COMPLETED")
    print(f"{'=' * 60}")
    print(f"Split: {config.split}")
    print(f"Model: {config.agent_config.model_name}")
    print(f"Tag: {config.tag}")
    print(f"Total tasks: {summary.get('total_tasks', 0)}")
    print(f"Success rate: {summary.get('success_rate', 0.0) * 100:.1f}%")
    print(f"Total experiment time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"{'=' * 60}")

    failed_tasks = summary.get("failed_tasks", [])
    if failed_tasks:
        print(f"Failed tasks: {failed_tasks}")


def run_appworld_experiment(
    config: ExperimentConfig,
    *,
    task_id: str | None = None,
    task_ids: list[str] | None = None,
    agent_type: str = "task",
    continue_existing: bool = False,
) -> dict:
    """Run an AppWorld benchmark with an already constructed config."""
    memory_manager = create_appworld_memory_adapter(
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
    if memory_manager is not None:
        logger.info("Memory loaded: %s from %s", config.memory_config.memory_type, config.memory_config.memory_path)

    if task_id is not None:
        selected_task_ids = [task_id]
        print(f"Running single task: {task_id}")
    elif task_ids is not None:
        selected_task_ids = list(task_ids)
        print(f"Running specified tasks: {selected_task_ids}")
    else:
        task_source = AppWorldTaskSource()
        selected_task_ids = task_source.load_task_ids(config.split)
        print(f"Running all tasks from {config.split} split: {len(selected_task_ids)} tasks")

    if config.debug_mode and len(selected_task_ids) > 1:
        selected_task_ids = selected_task_ids[:1]
        print(f"Debug mode enabled: running only first task ({selected_task_ids[0]})")

    benchmark = AppWorldBenchmark(
        config=config,
        selected_task_ids=selected_task_ids,
        agent_type=agent_type,
        memory_manager=memory_manager,
    )
    runner_config = RunnerConfig(
        num_workers=config.num_workers,
        debug=config.debug_mode,
        continue_existing=continue_existing,
        max_retries=config.max_retries,
        task_timeout_seconds=config.task_timeout_seconds,
    )
    run_output = run_benchmark(benchmark, runner_config=runner_config)
    summary = run_output["summary"]
    _print_completion_summary(config, summary)
    print(f"Experiment summary saved to: {benchmark.output_root_dir}/experiment_summary.json")
    return {
        "run_output": run_output,
        "summary": summary,
        "output_root_dir": str(benchmark.output_root_dir),
        "selected_task_ids": selected_task_ids,
    }


def main_runner() -> None:
    config = ExperimentConfig()
    parser = _build_parser(config)
    args = parser.parse_args()
    _apply_cli_overrides(config, args)
    task_ids = None
    task_id = None
    if args.task_id is not None:
        task_id = args.task_id
    elif args.task_ids is not None:
        task_ids = list(args.task_ids)
    run_appworld_experiment(
        config,
        task_id=task_id,
        task_ids=task_ids,
        agent_type=args.agent_type,
        continue_existing=args.continue_existing,
    )


if __name__ == "__main__":
    main_runner()
