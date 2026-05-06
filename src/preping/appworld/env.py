"""AppWorld environment and configuration (canonical module)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from appworld import AppWorld

from preping.core.env import BaseEnvConfig, BaseLanguageBasedEnv
from preping.utils import all_seed


@dataclass
class AppWorldEnvConfig(BaseEnvConfig):
    """Configuration for the AppWorld environment."""

    env_id: str = "appworld"
    experiment_name: str = "minimal_react_agent"
    task_id: Optional[str] = None
    max_iter: int = 50
    max_observation_output_chars: int = 10000

    def __post_init__(self):
        super().__init__()
        self.invalid_act = "Invalid action"
        self.invalid_act_score = -1.0


def create_appworld_config(**kwargs) -> AppWorldEnvConfig:
    """Create AppWorld environment config from parameters."""
    return AppWorldEnvConfig(**kwargs)


class AppWorldTask:
    """Simple task class that mimics AppWorld Task interface."""

    def __init__(self, task_id: str, instruction: str, supervisor: str = ""):
        self.task_id = task_id
        self.instruction = instruction
        self.supervisor = supervisor


class AppWorldEnv(BaseLanguageBasedEnv):
    """
    AppWorld environment implementation following the preping framework.
    """

    name = "appworld"

    def __init__(self, config: Optional[AppWorldEnvConfig] = None, verbose: bool = True, **kwargs):
        super().__init__()

        self.config = config or AppWorldEnvConfig()
        self.kwargs = kwargs
        self.verbose = verbose

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        if not self.verbose:
            self.logger.disabled = True

        self.task: Optional[AppWorldTask] = None
        self.task_id: Optional[str] = None
        self.observation: str = ""
        self.done: bool = False
        self.reward: float = 0.0
        self.info: Dict[str, Any] = {}
        self.trajectory: List[Dict[str, Any]] = []
        self.num_interactions: int = 0

        self.world: Optional[AppWorld] = None
        self._override_instruction: Optional[str] = None

        self.experiment_name = getattr(config, "experiment_name", "minimal_test")
        self.max_observation_output_chars = getattr(self.config, "max_observation_output_chars", 10000)

        self.logger.debug(f"AppWorld environment initialized with config: {self.config}")

    def reset(
        self,
        seed: Optional[int] = None,
        task_id: Optional[str] = None,
        override_instruction: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Reset the environment for a new task."""
        if seed is not None:
            all_seed(seed)

        if task_id is None:
            raise ValueError("task_id is required for AppWorld environment")

        self.task_id = task_id
        self.observation = ""
        self.done = False
        self.reward = 0.0
        self.info = {}
        self.trajectory = []
        self.num_interactions = 0

        self.world = AppWorld(task_id=self.task_id, experiment_name=self.experiment_name)

        if override_instruction:
            self._override_instruction = override_instruction
            self.task = AppWorldTask(
                task_id=task_id,
                instruction=override_instruction,
                supervisor=self.world.task.supervisor if hasattr(self.world.task, "supervisor") else "",
            )
            self.logger.info("Instruction overridden for synthetic task execution")
        else:
            self._override_instruction = None
            self.task = self.world.task

        self.observation = (
            f"Task: {self.task.instruction}\n\n"
            "You can execute Python code to complete this task. What would you like to do?"
        )

        self.logger.debug(f"Environment reset with task: {self.task_id}")
        self.logger.debug(f"Task instruction: {self.task.instruction}")
        return self.observation

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        self.num_interactions += 1

        if self.num_interactions >= self.config.max_iter:
            self.done = True
            self.observation = f"Maximum interactions ({self.config.max_iter}) reached."
            self.reward = 0.0
            self.info = {"success": False, "reason": "max_interactions"}
            return self.observation, self.reward, self.done, self.info

        output = None
        execution_error = None
        execution_traceback = None

        try:
            output = self.world.execute(action)
        except Exception as e:
            import traceback

            execution_error = str(e)
            execution_traceback = traceback.format_exc()
            self.logger.info(f"Error executing action: {e}")
            self.logger.info(f"Traceback:\n{execution_traceback}")

        if execution_error is None:
            try:
                self.done = self.world.task_completed()
            except Exception as e:
                self.logger.error(f"Error checking task completion: {e}")
                self.done = False

            if self.done:
                try:
                    eval_result = self.world.evaluate()
                    self.reward = eval_result.success if eval_result else 0.0
                except Exception as e:
                    import traceback

                    eval_traceback = traceback.format_exc()
                    self.logger.error(f"Error during evaluation: {e}")
                    self.logger.error(f"Evaluation traceback:\n{eval_traceback}")
                    self.reward = 0.0
                    self.info["evaluation_error"] = str(e)
            else:
                self.reward = 0.0

            output = self._intercept_show_active_task_output(action, output)
            if len(output) > self.max_observation_output_chars:
                output = output[: self.max_observation_output_chars] + "\n...[truncated]..."
                self.logger.warning(
                    f"Output truncated to {self.max_observation_output_chars} characters for observation."
                )
            self.observation = f"Output: {output}"
            self.info = {"success": self.reward, "reason": "task_completed" if self.done else "in_progress"}

            self.trajectory.append(
                {
                    "step": self.num_interactions,
                    "action": action,
                    "output": output,
                    "reward": self.reward,
                    "done": self.done,
                }
            )
        else:
            self.observation = f"Error executing code: {execution_error}"
            self.reward = 0
            self.done = False
            self.info = {
                "success": False,
                "reason": "execution_error",
                "error": execution_error,
                "traceback": execution_traceback,
            }

            self.trajectory.append(
                {
                    "step": self.num_interactions,
                    "action": action,
                    "output": self.observation,
                    "reward": self.reward,
                    "done": self.done,
                    "error": execution_error,
                    "traceback": execution_traceback,
                }
            )

        return self.observation, self.reward, self.done, self.info

    def _intercept_show_active_task_output(self, action: str, output: str) -> str:
        """Intercept show_active_task output for synthetic tasks."""
        if not self._override_instruction:
            return output

        patterns = [
            r"apis\.supervisor\.show_active_task\s*\(",
            r"show_active_task\s*\(",
        ]
        is_task_query = any(re.search(p, action) for p in patterns)

        if is_task_query and output:
            try:
                task_data = json.loads(output)
                if isinstance(task_data, dict) and "instruction" in task_data:
                    task_data["instruction"] = self._override_instruction
                    output = json.dumps(task_data, indent=2)
            except Exception:
                self.logger.debug("Could not parse show_active_task output as JSON, skipping interception.")

        return output

    def task_completed(self) -> bool:
        """Check if the current task has been completed."""
        return self.done

    def render(self, mode: str = "text") -> str:
        """Render the current environment state."""
        return self.observation

    def close(self):
        """Close environment and cleanup resources."""
        if self.world is not None:
            self.world.close()
            self.world = None

    def dump_history(self, output_dir: str):
        """Dump trajectory history to file."""
        import os

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "appworld_trajectory.json"), "w", encoding="utf-8") as file:
            json.dump(
                {
                    "task_id": self.task_id if self.task else None,
                    "task_instruction": self.task.instruction if self.task else None,
                    "num_interactions": self.num_interactions,
                    "completed": self.done,
                    "final_reward": self.reward,
                    "trajectory": self.trajectory,
                },
                file,
                indent=2,
            )
        with open(os.path.join(output_dir, "env_history.json"), "w", encoding="utf-8") as file:
            json.dump(self.trajectory, file, indent=2)
