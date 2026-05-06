"""Canonical single-task execution for AppWorld."""

from __future__ import annotations

import copy
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

from preping.appworld.agent import AppWorldAgent, AppWorldAgentConfig
from preping.appworld.env import AppWorldEnv, AppWorldEnvConfig
from preping.appworld.exploration import GuidedExplorerAgent, RandomExplorerAgent


_NOISY_LOGGERS = ("httpx", "httpcore", "azure.identity", "azure.core", "LiteLLM")


def setup_appworld_task_logging(*, debug_mode: bool = False) -> None:
    """Configure logging for single-task execution."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    # Keep behavior deterministic in repeated worker calls.
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(handler)

    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _create_agent(
    *,
    agent_type: str,
    agent_config: AppWorldAgentConfig,
    env: AppWorldEnv,
    task_config: Dict[str, Any],
    debug_mode: bool,
    max_iter: int,
) -> Any:
    """Instantiate the requested agent class with shared constructor arguments."""
    api_key = ""
    if agent_type == "random_explorer":
        return RandomExplorerAgent(
            model_name=agent_config.model_name,
            key=api_key,
            env=env,
            task_config=task_config,
            exp_config=agent_config,
            debug_mode=debug_mode,
            max_steps=max_iter,
            temperature=agent_config.temperature,
        )
    if agent_type == "guided_explorer":
        return GuidedExplorerAgent(
            model_name=agent_config.model_name,
            key=api_key,
            env=env,
            task_config=task_config,
            exp_config=agent_config,
            debug_mode=debug_mode,
            max_steps=max_iter,
            temperature=agent_config.temperature,
        )
    return AppWorldAgent(
        model_name=agent_config.model_name,
        key=api_key,
        env=env,
        task_config=task_config,
        exp_config=agent_config,
        debug_mode=debug_mode,
    )


def run_single_task(
    task_id: str,
    split: str = "train",
    output_dir: str = "outputs",
    agent_config: Optional[AppWorldAgentConfig] = None,
    env_config: Optional[AppWorldEnvConfig] = None,
    experiment_name: str = "minimal_test",
    agent_type: str = "task",
    override_instruction: Optional[str] = None,
    context_sections: Optional[List[str]] = None,
    debug_mode: bool = False,
    verbose: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Run one AppWorld task and return execution results."""
    if agent_config is None:
        agent_config = AppWorldAgentConfig()

    setup_appworld_task_logging(debug_mode=debug_mode)
    logger = logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)

    if env_config is None:
        env_config = AppWorldEnvConfig(experiment_name=experiment_name)
    else:
        env_config.experiment_name = experiment_name

    logger.info(
        "Starting AppWorld task: %s Split: %s, Output Dir: %s, Agent Type: %s",
        task_id,
        split,
        output_dir,
        agent_type,
    )
    env = AppWorldEnv(config=env_config, verbose=verbose)
    env.reset(task_id=task_id, override_instruction=override_instruction)

    if override_instruction:
        logger.info("Running with override instruction: %s...", override_instruction[:100])

    task_config = {
        "task_id": task_id,
        "split": split,
        "experiment_name": experiment_name,
        "username": "user",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "weekday": datetime.now().strftime("%A"),
        "time": datetime.now().strftime("%H:%M:%S"),
    }

    agent = _create_agent(
        agent_type=agent_type,
        agent_config=agent_config,
        env=env,
        task_config=task_config,
        debug_mode=debug_mode,
        max_iter=env_config.max_iter,
    )

    results = agent.run(env, max_iter=env_config.max_iter, context_sections=context_sections)
    token_summary = agent.get_token_usage_summary()

    results.update(
        {
            "task_id": task_id,
            "split": split,
            "model_name": agent_config.model_name,
            "experiment_name": experiment_name,
            "token_usage": token_summary,
        }
    )

    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    agent.dump_history(output_dir)
    env.dump_history(output_dir)

    results["trajectory"] = list(env.trajectory)
    results["llm_history"] = list(getattr(agent, "conversation_history", []))

    logger.info(
        "Task completed!, Success: %s, Iterations: %s, Termination reason: %s, Final reward: %s",
        results["success"],
        results["iterations"],
        results["termination_reason"],
        results["final_reward"],
    )

    env.close()
    return results


def _resolve_task_output_dir(task_args: Mapping[str, Any], task_id: str) -> str:
    if task_args.get("task_output_dir"):
        return str(task_args["task_output_dir"])
    output_root_dir = str(task_args["output_root_dir"])
    return os.path.join(output_root_dir, f"task_{task_id}")


def _resolve_attempt_output_dir(task_args: Mapping[str, Any], task_id: str) -> str:
    canonical_output_dir = _resolve_task_output_dir(task_args, task_id)
    attempt_index = task_args.get("attempt_index")
    if attempt_index in (None, 0):
        return canonical_output_dir
    return os.path.join(canonical_output_dir, f"attempt_{attempt_index}")


def _build_failure_result(
    *,
    task_id: str,
    execution_task_id: str,
    task_output_dir: str,
    attempt_output_dir: str,
    error: Exception,
    original_task: Dict[str, Any] | None,
    attempt_index: int | None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "task_id": task_id,
        "execution_task_id": execution_task_id,
        "success": False,
        "task_output_dir": task_output_dir,
        "attempt_output_dir": attempt_output_dir,
        "attempt_index": attempt_index,
        "termination_reason": "Exception",
        "error": str(error),
    }
    if original_task is not None:
        payload["original_task"] = original_task
    return payload


def run_single_task_worker(task_args: Mapping[str, Any]) -> Dict[str, Any]:
    """Process-pool friendly worker for one AppWorld task."""
    task_id = str(task_args["task_id"])
    execution_task_id = str(task_args.get("execution_task_id", task_id))
    task_output_dir = _resolve_task_output_dir(task_args, task_id)
    attempt_output_dir = _resolve_attempt_output_dir(task_args, task_id)
    attempt_index = task_args.get("attempt_index")

    task_index = task_args.get("task_index")
    total_tasks = task_args.get("total_tasks")
    if task_index is not None and total_tasks is not None:
        print(f"[Worker {os.getpid()}] Running task {task_id} ({int(task_index) + 1}/{total_tasks})")
    else:
        print(f"[Worker {os.getpid()}] Running task {task_id}")

    try:
        result = run_single_task(
            task_id=execution_task_id,
            split=task_args["split"],
            output_dir=attempt_output_dir,
            agent_config=task_args["agent_config"],
            env_config=copy.deepcopy(task_args.get("env_config")),
            experiment_name=task_args["experiment_name"],
            agent_type=task_args.get("agent_type", "task"),
            override_instruction=task_args.get("override_instruction"),
            context_sections=task_args.get("context_sections"),
            debug_mode=task_args.get("debug_mode", False),
            verbose=task_args.get("verbose", True),
        )
        result["appworld_task_id"] = str(result.get("task_id", execution_task_id))
        result["task_id"] = task_id
        result["execution_task_id"] = execution_task_id
        result["task_output_dir"] = task_output_dir
        result["attempt_output_dir"] = attempt_output_dir
        result["attempt_index"] = attempt_index
        if task_args.get("original_task") is not None:
            result["original_task"] = task_args["original_task"]
        return dict(result)
    except Exception as exc:  # noqa: PERF203 - worker exception path
        print(f"[Worker {os.getpid()}] Task {task_id} failed: {exc}")
        return _build_failure_result(
            task_id=task_id,
            execution_task_id=execution_task_id,
            task_output_dir=task_output_dir,
            attempt_output_dir=attempt_output_dir,
            error=exc,
            original_task=task_args.get("original_task"),
            attempt_index=attempt_index,
        )
