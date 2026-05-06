"""Environment information extraction and merge helpers for PrePing."""

from __future__ import annotations

import json
import logging
import random
import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn


logger = logging.getLogger(__name__)
console = Console()


class PrePingEnvironmentInfo:
    """Extract and merge environment information from trajectories and task results."""

    def __init__(
        self,
        *,
        llm_provider: Callable[[], Any],
        environment_extraction_prompt: str,
        trajectory_file_pattern: str = "**/trajectory.json",
        verbose: bool = False,
    ) -> None:
        self._llm_provider = llm_provider
        self.environment_extraction_prompt = environment_extraction_prompt
        self.trajectory_file_pattern = trajectory_file_pattern
        self.verbose = verbose

    @property
    def llm(self) -> Any:
        return self._llm_provider()

    def load_trajectory(self, trajectory_path: str) -> Dict[str, Any]:
        """Load trajectory from JSON file."""
        with open(trajectory_path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def format_trajectory_for_prompt(trajectory: Dict[str, Any]) -> str:
        """Format trajectory for the extraction prompt."""
        lines = [f"Task: {trajectory.get('task_instruction', 'Unknown')}", ""]

        for step in trajectory.get("trajectory", []):
            lines.append(f"Step {step.get('step', '?')}:")
            lines.append(f"Action: {step.get('action', '')}")
            lines.append(f"Output: {step.get('output', '')[:2000]}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def build_extraction_payload_from_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build a normalized extraction payload from a task result."""
        task = result.get("original_task") or {}
        task_text = result.get("task_text") or task.get("instruction") or task.get("task_description") or "Unknown"

        llm_history = result.get("llm_history")
        if isinstance(llm_history, list) and llm_history:
            steps = []
            for idx, msg in enumerate(llm_history, 1):
                if not isinstance(msg, dict):
                    continue
                steps.append(
                    {
                        "step": idx,
                        "action": f"[{str(msg.get('role', 'unknown')).upper()}]",
                        "output": str(msg.get("content", "")),
                    }
                )
            if steps:
                return {"task_instruction": task_text, "trajectory": steps}

        trajectory = result.get("trajectory")
        if isinstance(trajectory, list) and trajectory:
            normalized_steps = []
            for idx, step in enumerate(trajectory, 1):
                if isinstance(step, dict):
                    normalized_steps.append(
                        {
                            "step": step.get("step", idx),
                            "action": step.get("action", ""),
                            "output": step.get("output", ""),
                        }
                    )
            if normalized_steps:
                return {"task_instruction": task_text, "trajectory": normalized_steps}

        return None

    @staticmethod
    def parse_json_from_response(response_text: str) -> Any:
        """Parse JSON from an LLM response, handling fenced code blocks."""
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

    def _extract_from_formatted_trajectories(
        self,
        trajectory_items: List[Tuple[str, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Extract and merge environment info from normalized trajectories."""
        merged_info: Dict[str, Dict[str, set]] = {}
        if not trajectory_items:
            logger.warning("No valid trajectories provided for environment extraction")
            return {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Extracting environment info...", total=len(trajectory_items))
            prompts: List[str] = []
            prompt_sources: List[str] = []
            for source, trajectory in trajectory_items:
                formatted_trajectory = self.format_trajectory_for_prompt(trajectory)
                prompts.append(self.environment_extraction_prompt.format(trajectory=formatted_trajectory))
                prompt_sources.append(source)

            responses = self.llm.generate_batch(prompts)

            for source, response in zip(prompt_sources, responses):
                try:
                    response_text = response if isinstance(response, str) else str(response)
                    if self.verbose:
                        logger.info(f"\n[EXTRACTION RESPONSE]\n{response_text}\n{'='*60}")
                    env_info = self.parse_json_from_response(response_text)
                    for app, entities in env_info.items():
                        if app not in merged_info:
                            merged_info[app] = {}
                        for entity_type, values in entities.items():
                            if entity_type not in merged_info[app]:
                                merged_info[app][entity_type] = set()

                            def make_hashable(value: Any) -> Any:
                                if isinstance(value, (dict, list)):
                                    return json.dumps(value, sort_keys=True, ensure_ascii=False)
                                return value

                            if isinstance(values, list):
                                for value in values:
                                    merged_info[app][entity_type].add(make_hashable(value))
                            else:
                                merged_info[app][entity_type].add(make_hashable(values))
                except json.JSONDecodeError as exc:
                    logger.warning(f"Failed to parse environment info from {source}: {exc}")
                except Exception as exc:
                    logger.warning(f"Error processing {source}: {exc}")
                finally:
                    progress.update(task, advance=1)

        def try_parse_json(value: Any) -> Any:
            if isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value

        final_info: Dict[str, Dict[str, List[Any]]] = {}
        for app, entities in merged_info.items():
            final_info[app] = {}
            for entity_type, values in entities.items():
                final_info[app][entity_type] = [try_parse_json(value) for value in values]
        return final_info

    def extract_environment_info(self, trajectory_path: str) -> Dict[str, Any]:
        """Extract environment information from a trajectory using the configured LLM."""
        trajectory = self.load_trajectory(trajectory_path)
        formatted_trajectory = self.format_trajectory_for_prompt(trajectory)
        prompt = self.environment_extraction_prompt.format(trajectory=formatted_trajectory)

        response = self.llm.generate(prompt)
        response_text = response if isinstance(response, str) else str(response)

        if self.verbose:
            logger.info(f"\n[EXTRACTION RESPONSE]\n{response_text}\n{'='*60}")

        try:
            return self.parse_json_from_response(response_text)
        except json.JSONDecodeError as exc:
            logger.warning(f"Failed to parse environment info: {exc}")
            return {}

    def extract_from_trajectory_paths(self, trajectory_paths: List[Path]) -> Dict[str, Any]:
        """Extract and merge environment info from selected trajectory files."""
        if not trajectory_paths:
            logger.warning("No trajectory files provided for environment extraction")
            return {}

        logger.info("Found %s trajectory files", len(trajectory_paths))
        trajectory_items: List[Tuple[str, Dict[str, Any]]] = []
        for traj_file in trajectory_paths:
            try:
                trajectory_items.append((str(traj_file), self.load_trajectory(str(traj_file))))
            except Exception as exc:
                logger.warning(f"Error loading {traj_file}: {exc}")
        return self._extract_from_formatted_trajectories(trajectory_items)

    def extract_from_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract and merge environment info from in-memory task results."""
        if not results:
            logger.warning("No results provided for environment extraction")
            return {}

        trajectory_items: List[Tuple[str, Dict[str, Any]]] = []
        for result in results:
            payload = self.build_extraction_payload_from_result(result)
            if payload is None:
                continue
            source = f"task_{result.get('task_id', 'unknown')}"
            trajectory_items.append((source, payload))

        logger.info("Found %s trajectories from results", len(trajectory_items))
        return self._extract_from_formatted_trajectories(trajectory_items)

    def extract_from_multiple_trajectories(self, trajectory_dir: str | Path) -> Dict[str, Any]:
        """Extract and merge environment info from multiple trajectories in a directory."""
        trajectory_dir = Path(trajectory_dir)
        trajectory_files = list(trajectory_dir.glob(self.trajectory_file_pattern))

        if not trajectory_files:
            logger.warning(
                "No trajectory files found in %s with pattern %s",
                trajectory_dir,
                self.trajectory_file_pattern,
            )
            return {}

        return self.extract_from_trajectory_paths(trajectory_files)

    @staticmethod
    def merge_environment_info(all_environment_info: Dict[str, Dict], env_info: Dict[str, Dict]) -> Dict[str, Dict]:
        """Merge newly extracted environment info into the accumulated store."""
        if not env_info:
            return all_environment_info

        if not isinstance(env_info, dict):
            logger.warning(f"Invalid env_info type: {type(env_info)}, expected dict")
            return all_environment_info

        for app, entities in env_info.items():
            try:
                if not isinstance(entities, dict):
                    logger.warning(f"Invalid entities type for app '{app}': {type(entities)}, skipping")
                    continue

                if app not in all_environment_info:
                    all_environment_info[app] = {}

                for entity_type, values in entities.items():
                    try:
                        if entity_type not in all_environment_info[app]:
                            all_environment_info[app][entity_type] = []

                        if not isinstance(values, list):
                            values = [values] if values is not None else []

                        existing_list = all_environment_info[app][entity_type]
                        seen = set()
                        merged = []

                        for item in existing_list + values:
                            try:
                                if isinstance(item, (dict, list)):
                                    key = json.dumps(item, sort_keys=True)
                                else:
                                    key = str(item)

                                if key not in seen:
                                    seen.add(key)
                                    merged.append(item)
                            except (TypeError, ValueError) as exc:
                                logger.warning(f"Failed to process item in {app}.{entity_type}: {exc}")
                                merged.append(item)

                        all_environment_info[app][entity_type] = merged
                    except Exception as exc:
                        logger.warning(f"Error merging entity_type '{entity_type}' for app '{app}': {exc}")
            except Exception as exc:
                logger.warning(f"Error processing app '{app}': {exc}")

        return all_environment_info

    @staticmethod
    def prepare_for_generation(
        *,
        all_environment_info: Dict[str, Any],
        use_environment_info: bool,
        max_entities_per_type: Optional[int],
        seed: int,
        cycle_num: int,
    ) -> Optional[Dict[str, Any]]:
        """Build a prompt-ready environment info view with deterministic shuffle and per-type caps."""
        if not use_environment_info or not all_environment_info:
            return None

        rng = random.Random(f"{seed}:{cycle_num}")
        prepared_env_info: Dict[str, Any] = {}

        for app, entities in all_environment_info.items():
            if not isinstance(entities, dict):
                prepared_env_info[app] = copy.deepcopy(entities)
                continue

            prepared_entities: Dict[str, Any] = {}
            for entity_type, values in entities.items():
                if not isinstance(values, list):
                    prepared_entities[entity_type] = copy.deepcopy(values)
                    continue

                shuffled_values = copy.deepcopy(values)
                rng.shuffle(shuffled_values)
                if max_entities_per_type is not None and max_entities_per_type >= 0:
                    shuffled_values = shuffled_values[:max_entities_per_type]
                prepared_entities[entity_type] = shuffled_values
            prepared_env_info[app] = prepared_entities

        return prepared_env_info
