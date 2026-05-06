"""
PrePing Reflector Module

The Reflector is the "Critic" component that analyzes execution trajectories
to extract actionable insights (Delta Entries) for the Playbook.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from .prompts import REFLECTOR_SYSTEM_PROMPT
from .trajectory_formatters import TrajectoryPromptFormatter, format_trajectory_for_prompt
from .types import Bullet, BulletTag, PlaybookSection


logger = logging.getLogger(__name__)


@dataclass
class ReflectorInput:
    """Input data for the Reflector."""
    task_description: str
    trajectory: List[Dict[str, str]]  # List of trajectory entries
    result: Optional[str] = None  # "success" or "failure"
    ground_truth_code: Optional[str] = None
    unit_test_results: Optional[str] = None
    playbook_used: Optional[str] = None  # Formatted playbook string
    trajectory_format: str = "action_output"  # 'action_output', 'llm_history', or a custom formatter key


@dataclass
class ReflectorOutput:
    """Output from the Reflector analysis."""
    reasoning: str
    error_identification: str
    root_cause_analysis: str
    correct_approach: str
    key_insight: str
    bullet_tags: Dict[str, BulletTag]  # bullet_id -> tag
    raw_response: Dict[str, Any]
    prompt: str
    response_text: str


class Reflector:
    """
    Analyzes execution trajectories to extract insights.

    The Reflector performs "Delta Generation" - instead of rewriting the
    whole playbook, it outputs only new insights or corrections specific
    to the recent episode.
    """

    def __init__(
        self,
        llm_client: Any,
        include_task_context: bool = True,
        system_prompt: str = REFLECTOR_SYSTEM_PROMPT,
        trajectory_formatters: Mapping[str, TrajectoryPromptFormatter] | None = None,
    ):
        """
        Initialize the Reflector.

        Args:
            llm_client: An LLM client with a `generate` method that accepts
                       messages and returns a response string.
            include_task_context: If False, exclude task descriptions from prompts.
                Useful for static doc playbook building or exploration agents.
        """
        self.llm_client = llm_client
        self.include_task_context = include_task_context
        self.system_prompt = system_prompt
        self.trajectory_formatters = dict(trajectory_formatters or {})

    def _build_trajectory_string(self, trajectory: List[Dict[str, str]], trajectory_format: str = "action_output") -> str:
        """Format trajectory as a string for the prompt."""
        return format_trajectory_for_prompt(
            trajectory,
            trajectory_format,
            custom_formatters=self.trajectory_formatters,
        )

    def _build_prompt(self, input_data: ReflectorInput) -> str:
        """Build the complete prompt for the Reflector LLM call."""
        prompt = self.system_prompt

        # Optionally include task description
        if self.include_task_context and input_data.task_description:
            prompt = prompt.replace("{{task_description}}", f"\n{input_data.task_description}")

        if input_data.result is not None:
            prompt = prompt.replace("{{ground_truth_result}}", f"[GROUND TRUTH RESULT]\n{input_data.result}")
            logger.info(f'[Reflector] use ground truth for reflector input. GT: {input_data.result}')

        if input_data.ground_truth_code:
            ground_truth_code_str = f'Ground truth code (reference, known-correct):\nGROUND_TRUTH_CODE_START\n{input_data.ground_truth_code}\nGROUND_TRUTH_CODE_END\n'
        else:
            ground_truth_code_str = ""
        if input_data.unit_test_results:
            unit_test_results_str = f'Test report (unit tests result for the task after the generated code was run):\nTEST_REPORT_START\n{input_data.unit_test_results}\nTEST_REPORT_END\n'
        else:
            unit_test_results_str = ""
        # Replace template variables
        prompt = prompt.replace("{{ground_truth_code}}", ground_truth_code_str)
        prompt = prompt.replace("{{unit_test_results}}", unit_test_results_str)
        prompt = prompt.replace("{{playbook}}", input_data.playbook_used or "Empty playbook")

        # Append the trajectory
        trajectory_str = self._build_trajectory_string(input_data.trajectory, input_data.trajectory_format)
        prompt = prompt.replace("{{trajectory}}", trajectory_str)


        return prompt

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the JSON response from the LLM."""
        # Try to extract JSON from the response
        try:
            # First, try direct JSON parsing
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in the response
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass

        logger.warning("Failed to parse Reflector response as JSON")
        return {
            "reasoning": response,
            "error_identification": "Parse error",
            "root_cause_analysis": "Could not parse LLM response",
            "correct_approach": "N/A",
            "key_insight": "N/A"
        }

    def analyze(self, input_data: ReflectorInput) -> ReflectorOutput:
        """
        Analyze a trajectory and generate insights.

        Args:
            input_data: The ReflectorInput containing trajectory and context.

        Returns:
            ReflectorOutput with analysis results and candidate insights.
        """
        prompt = self._build_prompt(input_data)

        logger.debug(f"[Reflector] PROMPT:\n{'='*60}\n{prompt}\n{'='*60}")

        messages = [
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.generate(messages, flex_mode=False)
        logger.debug(f"[Reflector] RESPONSE:\n{'='*60}\n{response}\n{'='*60}")

        parsed = self._parse_response(response)

        # Parse bullet_tags from response
        bullet_tags = {}
        raw_tags = parsed.get("bullet_tags", {})
        if isinstance(raw_tags, dict):
            for bullet_id, tag_str in raw_tags.items():
                tag_str_lower = str(tag_str).lower()
                if tag_str_lower in ["helpful", "harmful", "neutral"]:
                    bullet_tags[bullet_id] = BulletTag(tag_str_lower)
                else:
                    logger.warning(f"Invalid bullet tag '{tag_str}' for bullet {bullet_id}, skipping")

        return ReflectorOutput(
            reasoning=parsed.get("reasoning", ""),
            error_identification=parsed.get("error_identification", ""),
            root_cause_analysis=parsed.get("root_cause_analysis", ""),
            correct_approach=parsed.get("correct_approach", ""),
            key_insight=parsed.get("key_insight", ""),
            bullet_tags=bullet_tags,
            raw_response=parsed,
            prompt=prompt,
            response_text=response,
        )


    def extract_candidate_bullets(
        self,
        reflector_output: ReflectorOutput,
        task_id: Optional[str] = None
    ) -> List[Bullet]:
        """
        Extract candidate Bullets from the Reflector output.

        This creates new Bullet objects from the key insights identified
        during analysis.

        Args:
            reflector_output: The output from the analyze() method.
            task_id: Optional task ID to record in bullet metadata.

        Returns:
            List of candidate Bullet objects to be curated.
        """
        candidates = []

        # Create a bullet from the key insight if available
        if reflector_output.key_insight and reflector_output.key_insight != "N/A":
            bullet = Bullet(
                content=reflector_output.key_insight,
                section=PlaybookSection.STRATEGIES,
                tag=BulletTag.HELPFUL,
            )
            bullet.metadata.source_task_id = task_id
            candidates.append(bullet)

        # Create a pitfall bullet from error analysis if available
        if (reflector_output.error_identification and
            reflector_output.error_identification != "Parse error"):
            pitfall_content = (
                f"Avoid: {reflector_output.error_identification}\n"
                f"Root cause: {reflector_output.root_cause_analysis}"
            )
            bullet = Bullet(
                content=pitfall_content,
                section=PlaybookSection.PITFALLS,
                tag=BulletTag.HELPFUL,
            )
            bullet.metadata.source_task_id = task_id
            candidates.append(bullet)

        return candidates
