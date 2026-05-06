#!/usr/bin/env python3
"""
Common utilities for AppWorld experiments.

This module contains shared functionality used across multiple experiment scripts:
- Logging setup with Rich formatting
- Task loading utilities
"""

import json
import logging
import os
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

from rich.logging import RichHandler


logger = logging.getLogger(__name__)
KST = timezone(timedelta(hours=9))

# Loggers to suppress (noisy third-party libraries)
NOISY_LOGGERS = ["httpx", "httpcore", "azure.identity", "azure.core", "LiteLLM"]


def _kst_timestamp() -> str:
    """Return run directory timestamp in KST."""
    return datetime.now(UTC).astimezone(KST).strftime("%y%m%d_%H%M%S")


def _set_attr_path(target: Any, attr_path: str, value: Any) -> None:
    """Set a potentially nested attribute path on ``target``.

    Example:
        _set_attr_path(config, "agent_config.temperature", 0.7)
    """
    parts = attr_path.split(".")
    obj = target
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def apply_optional_overrides(args: Any, target: Any, arg_to_attr: Mapping[str, str]) -> None:
    """Apply overrides for optional args whose value is not ``None``."""
    for arg_name, attr_path in arg_to_attr.items():
        if not hasattr(args, arg_name):
            continue
        value = getattr(args, arg_name)
        if value is not None:
            _set_attr_path(target, attr_path, value)


def setup_logging(
    output_dir: str = None,
    debug_mode: bool = False,
    log_filename: str = "experiment.log"
) -> None:
    """Setup logging configuration with rich formatting.

    Args:
        output_dir: If provided, logs will also be saved to {output_dir}/{log_filename}
        debug_mode: If True, set log level to DEBUG instead of INFO
        log_filename: Name of the log file (default: experiment.log)
    """
    root_logger = logging.getLogger()
    log_level = logging.DEBUG if debug_mode else logging.INFO

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    root_logger.setLevel(log_level)

    # Rich console handler - beautiful colored output
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
        markup=True,
    )
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # File handler if output_dir is provided (plain text for file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, log_filename)
        file_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(file_format))
        root_logger.addHandler(file_handler)

    # Suppress noisy loggers (always, even in debug mode)
    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_task_instruction(task_id: str) -> str:
    """Load task instruction for a given AppWorld task_id (for use as memory/retrieval query).

    Uses AppWorld data layout: data/tasks/{task_id}/specs.json with key "instruction".
    Requires APPWORLD_ROOT (or appworld data) to be set so that data/tasks/ exists.

    Args:
        task_id: AppWorld task ID (e.g. from load_task_ids).

    Returns:
        The task instruction string, or empty string if specs.json is not found.
    """
    try:
        from appworld.common.io import read_json
        from appworld.common.path_store import path_store
        path = os.path.join(path_store.data, "tasks", task_id, "specs.json")
        specs = read_json(path)
        return specs.get("instruction", "").strip()
    except Exception as e:
        logger.error(f"Error getting task instruction for {task_id}: {e}")
        return ""

def load_task_ids_from_split(split: str, seed: int = None) -> List[str]:
    """Load task IDs for a given dataset split.

    Args:
        split: Dataset split ('train', 'dev', 'test_normal', 'test_challenge')

    Returns:
        List of task IDs for the split
    """
    from appworld import load_task_ids

    task_ids = load_task_ids(split)
    if seed is not None:
        import random
        random.seed(seed)
        random.shuffle(task_ids)

    return task_ids


def load_tasks_from_json(file_path: str) -> List[Dict]:
    """Load synthetic task records from a JSON file with a top-level tasks list."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data.get("tasks", [])


def build_experiment_paths(
    output_dir: str,
    model_name: str,
    tag: str,
    split: str = None
) -> Dict[str, str]:
    """Build standardized experiment name and output directory.

    Args:
        output_dir: Base output directory
        model_name: Full model name (e.g., 'deepseek/deepseek-chat')
        tag: Experiment tag
        split: Optional dataset split to include in path

    Returns:
        Dict with 'experiment_name' and 'output_root_dir' keys
    """
    import uuid

    timestamp = _kst_timestamp()
    unique_id = uuid.uuid4().hex[:6]
    model_short_name = model_name.split("/")[-1] if "/" in model_name else model_name

    if split:
        experiment_name = f'{model_short_name}/{tag}/{split}/{timestamp}_{unique_id}'
    else:
        experiment_name = f'{model_short_name}/{tag}/{timestamp}_{unique_id}'

    output_root_dir = f'{output_dir}/{experiment_name}/'

    return {
        "experiment_name": experiment_name,
        "output_root_dir": output_root_dir
    }


def initialize_experiment_runtime(
    *,
    output_dir: str,
    model_name: str,
    tag: str,
    debug_mode: bool = False,
    split: Optional[str] = None,
) -> Tuple[str, str]:
    """Build experiment paths, configure logging, and set AppWorld output environment variable."""
    paths = build_experiment_paths(
        output_dir=output_dir,
        model_name=model_name,
        tag=tag,
        split=split,
    )
    experiment_name = paths["experiment_name"]
    output_root_dir = paths["output_root_dir"]

    setup_logging(output_dir=output_root_dir, debug_mode=debug_mode)
    os.environ["APPWORLD_OUTPUT_DIR"] = os.path.abspath(output_dir)
    return experiment_name, output_root_dir


# Token / cost aggregation utilities

_COST_FIELDS = (
    'total_cost_usd',
    'total_input_tokens',
    'total_cached_input_tokens',
    'total_output_tokens',
    'total_reasoning_tokens',
    'total_requests',
)


def _sum_cost_dicts(*dicts) -> Dict:
    """Sum cost fields across multiple cost dicts (from TokenTracker.get_cost_breakdown()).
    None / empty values are silently skipped."""
    totals = {f: 0 for f in _COST_FIELDS}
    for d in dicts:
        if not d:
            continue
        for f in _COST_FIELDS:
            totals[f] += d.get(f, 0) or 0
    totals['total_tokens'] = (
        totals['total_input_tokens']
        + totals['total_cached_input_tokens']
        + totals['total_output_tokens']
    )
    return totals


def aggregate_costs(total_tasks: int = 0, **named_sources) -> Dict:
    """Aggregate token/cost dicts into a per-component + total summary.

    Each kwarg value can be a single cost dict or a list/iterable of cost dicts.
    Returns ``{'agent': {...}, 'task_manager': {...}, ..., 'total': {...}}``.

    Usage::

        cost = aggregate_costs(
            total_tasks=10,
            agent=[r.get('token_usage') for r in results],
            task_manager_generator=task_manager.get_generator_cost(),
            task_manager_validator=task_manager.get_validator_cost(),
            memory=memory_manager.get_token_usage_summary(),
        )
        # cost['agent'], cost['total'], etc.
    """
    components = {}
    all_dicts = []

    for name, src in named_sources.items():
        if src is None:
            continue
        # Normalise to list of dicts
        if isinstance(src, dict):
            items = [src]
        else:
            items = [s for s in src if s]
        component_total = _sum_cost_dicts(*items)
        components[name] = component_total
        all_dicts.extend(items)

    total = _sum_cost_dicts(*all_dicts)
    if total_tasks > 0:
        total['average_cost_per_task'] = total['total_cost_usd'] / total_tasks

    return {**components, 'total': total}


def log_cost_summary(cost: Dict, total_tasks: int = 0) -> None:
    """Log token usage and cost summary."""
    # Support both {'total': {...}} and flat dict
    totals = cost.get('total', cost)
    usd = totals.get('total_cost_usd', 0)
    if usd <= 0:
        return
    total_tokens = totals.get('total_tokens', 0)
    logger.info("")
    logger.info("=" * 60)
    logger.info("TOKEN USAGE AND COST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Requests: {totals.get('total_requests', 0):,}")
    logger.info(f"Total Input Tokens: {totals.get('total_input_tokens', 0):,}")
    logger.info(f"Total Cached Input Tokens: {totals.get('total_cached_input_tokens', 0):,}")
    logger.info(f"Total Reasoning Tokens: {totals.get('total_reasoning_tokens', 0):,}")
    logger.info(f"Total Output Tokens: {totals.get('total_output_tokens', 0):,}")
    logger.info(f"Total Tokens: {total_tokens:,}")
    logger.info(f"TOTAL COST: ${usd:.6f}")

    logger.info("=" * 60)
