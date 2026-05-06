#!/usr/bin/env python3
"""Run task-informed offline/online playbook adaptation on AppWorld."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from config import ExperimentConfig
from experiment_utils import (
    aggregate_costs,
    apply_optional_overrides,
    build_experiment_paths,
    get_task_instruction,
    load_task_ids_from_split,
    load_tasks_from_json,
    log_cost_summary,
    setup_logging,
)

from preping.appworld import PrePingMemoryManager, run_single_task
from preping.core.memory import create_memory_store


logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=str, default=None, help="Dataset split, e.g. train or test_normal")
    parser.add_argument("--mode", type=str, choices=["offline", "online"], default=None)
    parser.add_argument("--playbook_path", type=str, default=None, help="Path to save/load playbook")
    parser.add_argument("--task_id", type=str, default=None, help="Run one task")
    parser.add_argument("--task_ids", type=str, nargs="+", default=None, help="Run specific task IDs")
    parser.add_argument("--max_tasks", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--tag", type=str, default=None, help="Experiment tag")
    parser.add_argument("--global_model_name", type=str, default=None, help="Global model name for all components")
    parser.add_argument("--agent_model_name", type=str, default=None, help="Model name for task execution")
    parser.add_argument("--memory_model_name", type=str, default=None, help="Model name for memory updates")
    parser.add_argument("--task_manager_model_name", type=str, default=None, help="Accepted for shared scripts")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=None, help="Debug mode")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=None, help="Verbose output")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for task ordering")
    parser.add_argument("--ground_truth", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--use_reflector_curator_examples", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--embedding_model", type=str, default=None, help="Embedding model for semantic retrieval")
    parser.add_argument("--embedding_base_url", type=str, default=None, help="Embedding server base URL")
    parser.add_argument("--playbook_max_bullets", type=int, default=None, help="Maximum retrieved bullets per task")
    parser.add_argument("--synthetic_file", type=str, default=None, help="Synthetic tasks JSON file")
    parser.add_argument("--synthetic_idx", type=int, default=None, help="Run one synthetic task by index")
    parser.add_argument("--base_task_id", type=str, default=None, help="Environment task_id for synthetic tasks")
    return parser


def init_config(args: argparse.Namespace) -> ExperimentConfig:
    """Build an experiment config from CLI options."""
    config = ExperimentConfig()
    config.tag = "offline_pb"
    config.memory_config.adaptation_mode = "offline"

    apply_optional_overrides(
        args,
        config,
        {
            "split": "split",
            "tag": "tag",
            "output_dir": "output_dir",
            "mode": "memory_config.adaptation_mode",
            "debug": "debug_mode",
            "verbose": "verbose",
            "seed": "seed",
        },
    )
    if args.debug is True:
        config.verbose = True
    if args.global_model_name:
        config.set_model_name(args.global_model_name)
    config.set_component_model_names(
        agent_model_name=args.agent_model_name,
        memory_model_name=args.memory_model_name,
        task_manager_model_name=args.task_manager_model_name,
    )
    if args.ground_truth is not None:
        config.memory_config.use_ground_truth = args.ground_truth == "true"
    apply_optional_overrides(
        args,
        config,
        {
            "embedding_model": "memory_config.embedding_model",
            "embedding_base_url": "memory_config.embedding_base_url",
            "playbook_max_bullets": "memory_config.playbook_max_bullets",
            "use_reflector_curator_examples": "memory_config.use_reflector_curator_examples",
        },
    )
    if args.seed is not None:
        config.agent_config.seed = args.seed
    return config


def load_adaptation_tasks(args: argparse.Namespace, config: ExperimentConfig) -> List[Dict[str, Any]]:
    """Load real AppWorld tasks or synthetic task records."""
    if args.synthetic_file:
        synthetic_tasks = load_tasks_from_json(args.synthetic_file)
        logger.info("Loaded %s synthetic tasks from %s", len(synthetic_tasks), args.synthetic_file)
        if args.synthetic_idx is not None:
            if args.synthetic_idx >= len(synthetic_tasks):
                raise ValueError(f"synthetic_idx {args.synthetic_idx} out of range")
            synthetic_tasks = [synthetic_tasks[args.synthetic_idx]]
        if args.max_tasks:
            synthetic_tasks = synthetic_tasks[: args.max_tasks]

        base_split = "train" if config.split == "synthetic" else config.split
        base_task_id = args.base_task_id or load_task_ids_from_split(base_split)[0]
        return [
            {
                "task_id": base_task_id,
                "synthetic_task_id": task.get("task_id", ""),
                "task_description": task.get("instruction", ""),
                "override_instruction": task.get("instruction", ""),
                "source_app": task.get("source_app", ""),
                "involved_apis": task.get("involved_apis", []),
            }
            for task in synthetic_tasks
        ]

    if args.task_id:
        task_ids = [args.task_id]
    elif args.task_ids:
        task_ids = args.task_ids
    else:
        task_ids = load_task_ids_from_split(config.split, config.seed)
    if args.max_tasks:
        task_ids = task_ids[: args.max_tasks]
    return [{"task_id": task_id, "task_description": get_task_instruction(task_id)} for task_id in task_ids]


def create_playbook_manager(config: ExperimentConfig, playbook_path: Path) -> PrePingMemoryManager:
    """Create the playbook memory manager used for adaptation."""
    mem_cfg = config.memory_config
    mem_cfg.memory_type = "playbook"
    mem_cfg.memory_path = str(playbook_path)
    memory_manager = create_memory_store(
        memory_type=mem_cfg.memory_type,
        memory_path=mem_cfg.memory_path,
        model_name=mem_cfg.model_name,
        temperature=mem_cfg.temperature,
        use_thinking=mem_cfg.use_thinking,
        embedding_model=mem_cfg.embedding_model,
        embedding_base_url=mem_cfg.embedding_base_url,
        include_task_context=mem_cfg.include_task_context,
        use_ground_truth=mem_cfg.use_ground_truth,
        playbook_max_bullets=mem_cfg.playbook_max_bullets,
        use_reflector_curator_examples=mem_cfg.use_reflector_curator_examples,
    )
    if not isinstance(memory_manager, PrePingMemoryManager):
        raise RuntimeError("Failed to initialize PrePingMemoryManager for sequential adaptation.")
    return memory_manager


def run_task_with_memory(
    *,
    task: Dict[str, Any],
    index: int,
    total: int,
    config: ExperimentConfig,
    output_root_dir: str,
    experiment_name: str,
    memory_manager: PrePingMemoryManager,
) -> Dict[str, Any]:
    """Run one task, then update memory unless the playbook is frozen."""
    task_id = task.get("task_id", f"task_{index}")
    override_instruction = task.get("override_instruction")
    folder_name = task.get("synthetic_task_id") or task_id
    task_output_dir = os.path.join(output_root_dir, f"task_{folder_name}")
    task_query = task.get("task_description") or override_instruction
    context_sections = memory_manager.get_memory(
        task_description=task_query,
        max_bullets=config.memory_config.playbook_max_bullets,
    )

    logger.info("[%s/%s] Processing task: %s", index + 1, total, task_id)
    try:
        result = run_single_task(
            task_id=task_id,
            split=config.split,
            output_dir=task_output_dir,
            agent_config=config.agent_config,
            env_config=config.env_config,
            experiment_name=experiment_name,
            agent_type="task",
            override_instruction=override_instruction,
            context_sections=context_sections,
            debug_mode=config.debug_mode,
            verbose=config.verbose,
        )
    except Exception as error:  # noqa: BLE001
        logger.error("Task %s failed: %s", task_id, error)
        result = {"success": False, "task_id": task_id, "error": str(error), "llm_history": []}

    result["task_id"] = task_id
    if memory_manager.is_frozen():
        return result

    try:
        result_label = "success" if result.get("success", False) else "failure"
        memory_manager.process_episode(
            task_description=task.get("task_description", ""),
            trajectory=result.get("llm_history", []),
            result=result_label if config.memory_config.use_ground_truth else "None",
            task_id=task_id,
            ground_truth_code=task.get("ground_truth_code") if config.memory_config.use_ground_truth else None,
            unit_test_results=result.get("unit_test_results") if config.memory_config.use_ground_truth else None,
            trajectory_format="llm_history",
            debug_output_dir=os.path.join(task_output_dir, "memory_debug"),
        )
        if config.memory_config.save_intermediate:
            memory_manager.playbook.save(Path(config.memory_config.memory_path))
    except Exception as error:  # noqa: BLE001
        logger.error("Failed to update memory for %s: %s", task_id, error)
    return result


def summarize_results(
    *,
    config: ExperimentConfig,
    results: List[Dict[str, Any]],
    output_root_dir: str,
    playbook_path: Path,
    memory_manager: PrePingMemoryManager,
    elapsed_seconds: float,
) -> Dict[str, Any]:
    """Build and persist a run summary."""
    successful = sum(1 for result in results if result.get("success", False))
    failed = len(results) - successful
    cost_summary = aggregate_costs(
        total_tasks=len(results),
        agent=[result.get("token_usage") for result in results],
        memory=memory_manager.get_token_usage_summary(),
    )
    summary = {
        "experiment_config": asdict(config),
        "successful": successful,
        "failed": failed,
        "success_rate": successful / len(results) if results else 0,
        "total_time_seconds": elapsed_seconds,
        "playbook_path": str(playbook_path),
        "final_bullet_count": memory_manager.playbook.get_bullet_count(),
        "playbook_frozen": memory_manager.playbook.is_frozen(),
        "playbook_stats": memory_manager.get_stats(),
        "cost_summary": cost_summary,
    }
    summary_path = Path(output_root_dir) / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_cost_summary(cost_summary, total_tasks=len(results))
    logger.info("Summary saved to: %s", summary_path)
    return summary


def run_sequential_adaptation(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the full sequential adaptation baseline."""
    config = init_config(args)
    os.environ["APPWORLD_OUTPUT_DIR"] = os.path.abspath(config.output_dir)
    paths = build_experiment_paths(
        output_dir=config.output_dir,
        model_name=config.agent_config.model_name,
        tag=config.tag,
        split=config.split,
    )
    output_root_dir = paths["output_root_dir"]
    experiment_name = paths["experiment_name"]
    setup_logging(output_dir=output_root_dir, debug_mode=config.debug_mode)

    playbook_path = Path(args.playbook_path) if args.playbook_path else Path(output_root_dir) / "playbook.json"
    memory_manager = create_playbook_manager(config, playbook_path)
    tasks = load_adaptation_tasks(args, config)
    logger.info("Sequential adaptation mode: %s", config.memory_config.adaptation_mode)
    logger.info("Split: %s", config.split)
    logger.info("Tasks: %s", len(tasks))
    logger.info("Playbook: %s", playbook_path)

    start_time = time.time()
    results = [
        run_task_with_memory(
            task=task,
            index=index,
            total=len(tasks),
            config=config,
            output_root_dir=output_root_dir,
            experiment_name=experiment_name,
            memory_manager=memory_manager,
        )
        for index, task in enumerate(tasks)
    ]

    if config.memory_config.adaptation_mode == "offline":
        memory_manager.freeze()
        logger.info("Offline adaptation complete. Playbook frozen.")
    else:
        logger.info("Online adaptation complete. Playbook remains updatable.")
    memory_manager.playbook.save(playbook_path)

    summary = summarize_results(
        config=config,
        results=results,
        output_root_dir=output_root_dir,
        playbook_path=playbook_path,
        memory_manager=memory_manager,
        elapsed_seconds=time.time() - start_time,
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """CLI entry point."""
    run_sequential_adaptation(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
