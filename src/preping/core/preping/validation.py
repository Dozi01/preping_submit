"""PrePing validation gate.

LLM-based validation of Task-Trajectory pairs to determine:
1. Feasibility: Is this task executable in the current environment?
2. Task Completion: Did the trajectory achieve the task goal?

This validator runs BEFORE Playbook construction, filtering out
invalid trajectories to ensure only valuable experiences are learned.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from rich.console import Console

from preping.llm import LLMManager


logger = logging.getLogger(__name__)
console = Console()


@dataclass
class PrePingValidationResult:
    """Result of trajectory validation using Likert scale (1-5)."""

    validation_result: str
    feasibility_score: int
    task_completion_score: int
    feasibility_reason: str
    task_completion_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PrePingValidationGate:
    """LLM-based validation gate for Task-Trajectory pairs."""

    def __init__(
        self,
        model_name: str = "deepseek/deepseek-chat",
        temperature: float = 0.0,
        max_tokens: int = 3000,
        use_thinking: bool = False,
        verbose: bool = False,
        min_feasibility_score: int = 5,
        min_task_completion_score: int = 4,
        validation_prompt: str | None = None,
    ):
        if validation_prompt is None:
            from preping.core.preping.prompts.prompt_validator import (
                VALIDATION_PROMPT as default_validation_prompt,
            )
            validation_prompt = default_validation_prompt
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_thinking = use_thinking
        self.verbose = verbose
        self.min_feasibility_score = min_feasibility_score
        self.min_task_completion_score = min_task_completion_score
        self.validation_prompt = validation_prompt
        self._llm = None

    @property
    def llm(self):
        """Lazy initialization of LLM."""
        if self._llm is None:
            self._llm = LLMManager.create_llm(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                use_thinking=self.use_thinking,
            )
        return self._llm

    def _format_trajectory(self, trajectory: List[Dict]) -> str:
        """Format trajectory for LLM prompt."""
        lines = []
        for step in trajectory:
            step_num = step.get("step", "?")
            action = step.get("action", "")
            output = step.get("output", "")

            lines.append(f"**Step {step_num}**")
            lines.append(f"Action: {action}")
            lines.append(f"Output: {output}")
            lines.append("")

        return "\n".join(lines)

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        else:
            json_str = response_text.strip()

        return json.loads(json_str)

    def _build_validation_prompt(self, task: Dict, trajectory: List[Dict], result: Dict[str, Any] | None = None) -> str:
        """Build validation prompt from task and trajectory."""
        task_instruction = task.get("instruction") or task.get("task_description", "")
        validation_trajectory_text = ""
        if isinstance(result, dict):
            validation_trajectory_text = str(result.get("validation_trajectory_text") or "").strip()
        formatted_trajectory = validation_trajectory_text or self._format_trajectory(trajectory)
        return self.validation_prompt.format(
            task_instruction=task_instruction,
            trajectory=formatted_trajectory,
        )

    def _validation_from_response(self, response_text: str) -> PrePingValidationResult:
        """Parse response text and map it to ValidationResult."""
        parsed = self._parse_json_response(response_text)

        feasibility_score = int(parsed.get("feasibility_score", 1))
        task_completion_score = int(parsed.get("task_completion_score", 1))

        feasibility_score = max(1, min(5, feasibility_score))
        task_completion_score = max(1, min(5, task_completion_score))

        if feasibility_score < self.min_feasibility_score:
            validation_result = "invalid"
        elif task_completion_score >= self.min_task_completion_score:
            validation_result = "success"
        else:
            validation_result = "failure"

        return PrePingValidationResult(
            validation_result=validation_result,
            feasibility_score=feasibility_score,
            task_completion_score=task_completion_score,
            feasibility_reason=parsed.get("feasibility_reason", ""),
            task_completion_reason=parsed.get("task_completion_reason", ""),
        )

    def validate(
        self,
        task: Dict,
        trajectory: List[Dict],
    ) -> PrePingValidationResult:
        """Validate a single Task-Trajectory pair."""
        prompt = self._build_validation_prompt(task, trajectory)

        if self.verbose:
            logger.info("\n%s\n[VALIDATION PROMPT]\n%s\n%s...\n%s", "=" * 60, "=" * 60, prompt[:2000], "=" * 60)

        try:
            response = self.llm.generate(prompt)
            response_text = response if isinstance(response, str) else str(response)

            if self.verbose:
                logger.info("\n[VALIDATION RESPONSE]\n%s\n%s", response_text, "=" * 60)

            return self._validation_from_response(response_text)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Trajectory validation failed while the validator was enabled. "
                "This is a validator execution/parsing failure, not an infeasible-task judgment."
            ) from exc

    def validate_batch(
        self,
        results: List[Dict],
        show_progress: bool = True,
    ) -> List[PrePingValidationResult]:
        """Validate multiple Task-Trajectory pairs."""
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

        if not results:
            return []

        prompts: List[str] = []
        for result in results:
            task_dict = result.get("original_task", {})
            trajectory = result.get("trajectory", [])
            prompt = self._build_validation_prompt(task_dict, trajectory, result)
            prompts.append(prompt)
            result["validation_prompt"] = prompt

        responses = self.llm.generate_batch(prompts)

        validation_results: List[PrePingValidationResult] = []
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("[cyan]Validating trajectories...", total=len(responses))
                for response_index, response in enumerate(responses):
                    response_text = response if isinstance(response, str) else str(response)
                    if self.verbose:
                        logger.info("\n[VALIDATION RESPONSE]\n%s\n%s", response_text, "=" * 60)
                    try:
                        validation_results.append(self._validation_from_response(response_text))
                    except Exception as exc:  # noqa: BLE001
                        raise RuntimeError(
                            "Trajectory validation failed while parsing validator response "
                            f"at batch index {response_index}."
                        ) from exc
                    progress.update(task, advance=1)
            return validation_results

        for response_index, response in enumerate(responses):
            response_text = response if isinstance(response, str) else str(response)
            if self.verbose:
                logger.info("\n[VALIDATION RESPONSE]\n%s\n%s", response_text, "=" * 60)
            try:
                validation_results.append(self._validation_from_response(response_text))
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "Trajectory validation failed while parsing validator response "
                    f"at batch index {response_index}."
                ) from exc

        return validation_results

    def get_validation_summary(self, validation_results: List[PrePingValidationResult]) -> Dict[str, Any]:
        """Generate summary statistics from validation results."""
        total = len(validation_results)
        valid_count = sum(1 for v in validation_results if v.validation_result != "invalid")

        feasibility_counts = {}
        for result in validation_results:
            key = str(result.feasibility_score)
            feasibility_counts[key] = feasibility_counts.get(key, 0) + 1

        task_completion_counts = {}
        for result in validation_results:
            key = str(result.task_completion_score)
            task_completion_counts[key] = task_completion_counts.get(key, 0) + 1

        avg_feasibility = sum(v.feasibility_score for v in validation_results) / total if total > 0 else 0.0
        avg_task_completion = sum(v.task_completion_score for v in validation_results) / total if total > 0 else 0.0

        return {
            "total_validated": total,
            "valid_count": valid_count,
            "invalid_count": total - valid_count,
            "valid_rate": valid_count / total if total > 0 else 0.0,
            "avg_feasibility_score": avg_feasibility,
            "avg_task_completion_score": avg_task_completion,
            "validation_result_breakdown": {
                "invalid": sum(1 for v in validation_results if v.validation_result == "invalid"),
                "success": sum(1 for v in validation_results if v.validation_result == "success"),
                "failure": sum(1 for v in validation_results if v.validation_result == "failure"),
            },
            "feasibility_score_breakdown": feasibility_counts,
            "task_completion_score_breakdown": task_completion_counts,
        }

    def get_token_usage_stats(self) -> Dict[str, Any]:
        """Get token usage statistics from LLM."""
        if self._llm is None:
            return {}
        return self._llm.get_token_usage_stats()

    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Get cost breakdown from LLM."""
        if self._llm is None:
            return {}
        return self._llm.get_cost_breakdown()
