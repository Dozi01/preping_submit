#!/usr/bin/env python3
"""Run an AppWorld trajectory cycle and immediately evaluate the generated playbook."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import ExperimentConfig
from run_all_parallel import run_appworld_experiment
from run_trajectory_cycle import init_config as init_cycle_config
from run_trajectory_cycle import run_cycle


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_cycles", type=int, default=3, help="Maximum number of cycles to run")
    parser.add_argument("--tasks_per_cycle", type=int, default=5, help="Number of unique synthetic tasks to generate per cycle")
    parser.add_argument("--runs_per_task", type=int, default=1, help="Execute each generated synthetic task N times")
    parser.add_argument("--num_workers", type=int, default=1, help="Parallel workers for synthetic task execution")
    parser.add_argument("--repeat_eval_min_feasibility", type=int, default=5)
    parser.add_argument("--repeat_eval_require_mixed_outcomes", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--memory_selection_mode",
        type=str,
        choices=["default", "single_run_include_failure"],
        default=None,
        help="Research mode for validator-based memory selection",
    )
    parser.add_argument("--base_task_id", type=str, default=None, help="Base task ID for environment initialization")
    parser.add_argument("--target_apps", type=str, nargs="+", default=None, help="Focus on specific apps for task generation")
    parser.add_argument("--global_model_name", type=str, default=None, help="Global model name for all components")
    parser.add_argument("--agent_model_name", type=str, default=None, help="Model name for agent")
    parser.add_argument("--memory_model_name", type=str, default=None, help="Model name for memory components")
    parser.add_argument("--task_manager_model_name", type=str, default=None, help="Model name for task manager")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--tag", type=str, default=None, help="Cycle/evaluation tag")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=None, help="Debug mode")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=None, help="Verbose output")
    parser.add_argument("--playbook_path", type=str, default=None, help="Optional explicit playbook output path")
    parser.add_argument("--ground_truth", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--validate_trajectory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_environment_info", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_reflector_curator_examples", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include_dataset_examples", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--memory_guided_generation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_proposer_memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--task_embedding_model", type=str, default=None)
    parser.add_argument("--task_embedding_base_url", type=str, default=None)
    parser.add_argument("--task_semantic_oversample_multiplier", type=int, default=None)
    parser.add_argument("--complexity_schedule", type=int, nargs="+", default=None)
    parser.add_argument("--complexity_start", type=int, default=None)
    parser.add_argument("--complexity_end", type=int, default=None)
    parser.add_argument("--proposer_memory_path", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default="test_normal", help="Split for follow-up playbook evaluation")
    return parser


def build_eval_config(cycle_config: ExperimentConfig, playbook_path: Path, args: argparse.Namespace) -> ExperimentConfig:
    """Build evaluation config from the cycle config and generated playbook."""
    eval_config = ExperimentConfig()
    eval_config.set_component_model_names(
        agent_model_name=cycle_config.agent_config.model_name,
        memory_model_name=cycle_config.memory_config.model_name,
        task_manager_model_name=cycle_config.task_manager_config.model_name,
    )
    eval_config.split = args.eval_split
    eval_config.tag = cycle_config.tag
    eval_config.output_dir = cycle_config.output_dir
    eval_config.seed = cycle_config.seed
    eval_config.num_workers = cycle_config.num_workers
    eval_config.task_timeout_seconds = cycle_config.task_timeout_seconds
    eval_config.max_retries = cycle_config.max_retries
    eval_config.debug_mode = cycle_config.debug_mode
    eval_config.verbose = cycle_config.verbose
    eval_config.agent_config.prompt_file = cycle_config.agent_config.prompt_file
    eval_config.agent_config.temperature = cycle_config.agent_config.temperature
    eval_config.agent_config.seed = cycle_config.agent_config.seed
    eval_config.env_config.max_iter = cycle_config.env_config.max_iter
    eval_config.memory_config.memory_type = "playbook"
    eval_config.memory_config.memory_path = str(playbook_path)
    eval_config.memory_config.embedding_model = cycle_config.memory_config.embedding_model
    eval_config.memory_config.embedding_base_url = cycle_config.memory_config.embedding_base_url
    eval_config.memory_config.playbook_max_bullets = cycle_config.memory_config.playbook_max_bullets
    eval_config.memory_config.use_ground_truth = cycle_config.memory_config.use_ground_truth
    eval_config.memory_config.include_task_context = cycle_config.memory_config.include_task_context
    eval_config.memory_config.use_reflector_curator_examples = (
        cycle_config.memory_config.use_reflector_curator_examples
    )
    return eval_config


def resolve_playbook_path(cycle_summary: dict) -> Path:
    """Resolve generated playbook path from cycle summary."""
    playbook_path = cycle_summary.get("playbook_path")
    if not playbook_path:
        raise ValueError("Cycle summary does not contain playbook_path")
    resolved = Path(playbook_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Generated playbook not found: {resolved}")
    return resolved


def run_cycle_then_playbook_eval(args: argparse.Namespace) -> dict:
    """Run the full AppWorld cycle -> playbook evaluation pipeline."""
    args.build_playbook = True
    cycle_summary = run_cycle(args)
    playbook_path = resolve_playbook_path(cycle_summary)
    cycle_config = init_cycle_config(args)
    eval_config = build_eval_config(cycle_config, playbook_path, args)
    eval_result = run_appworld_experiment(
        eval_config,
    )
    return {
        "cycle": cycle_summary,
        "playbook_path": str(playbook_path),
        "eval_output_root": eval_result["output_root_dir"],
        "eval_summary": eval_result["summary"],
    }


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    pipeline_result = run_cycle_then_playbook_eval(args)
    print(f"Cycle output: {pipeline_result['cycle']['output_root_dir']}")
    print(f"Playbook: {pipeline_result['playbook_path']}")
    print(f"Eval output: {pipeline_result['eval_output_root']}")


if __name__ == "__main__":
    main()
