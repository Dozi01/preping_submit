"""
PrePing Proposer

Builds grounded environment summaries from execution trajectories using LLM,
then generates new grounded tasks based on that information.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from preping.embedding_model import EmbeddingClient
from preping.llm import LLMManager

from .environment_info import PrePingEnvironmentInfo
from .grounded_environment import PrePingGroundedEnvironmentSummaries


logger = logging.getLogger(__name__)


class PrePingProposer:
    """
    Extracts environment information from trajectories and generates grounded tasks.

    Uses LLM for both extraction and generation to be flexible and capture
    relevant information naturally.
    """

    def __init__(
        self,
        trajectory_file_pattern: str = "**/trajectory.json",
        model_name: str = "deepseek/deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 3000,
        use_thinking: bool = False,
        verbose: bool = False,
        complexity_level: int = 1,
        task_generation_prompt: Optional[str] = None,
        environment_extraction_prompt: Optional[str] = None,
        grounded_environment_summary_prompt: Optional[str] = None,
        complexity_level_descriptions: Optional[Dict[int, str]] = None,
        embedding_model: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        semantic_oversample_multiplier: int = 3,
        include_dataset_examples: bool = True,
        prompt_context_provider: Optional[Callable[[Optional[List[str]]], str]] = None,
        dataset_examples_section: Optional[str] = None,
    ):
        default_task_generation_prompt = None
        default_environment_extraction_prompt = None
        default_grounded_environment_summary_prompt = None
        default_complexity_level_descriptions = None
        if (
            task_generation_prompt is None
            or environment_extraction_prompt is None
            or grounded_environment_summary_prompt is None
            or complexity_level_descriptions is None
        ):
            from preping.core.preping.prompts.prompt_task_generator import (
                COMPLEXITY_LEVEL_DESCRIPTIONS as default_complexity_level_descriptions,
            )
            from preping.core.preping.prompts.prompt_task_generator import (
                ENVIRONMENT_EXTRACTION_PROMPT as default_environment_extraction_prompt,
            )
            from preping.core.preping.prompts.prompt_task_generator import (
                GROUNDED_ENVIRONMENT_SUMMARY_PROMPT as default_grounded_environment_summary_prompt,
            )
            from preping.core.preping.prompts.prompt_task_generator import (
                DATASET_EXAMPLES_SECTION as default_dataset_examples_section,
            )
            from preping.core.preping.prompts.prompt_task_generator import (
                TASK_GENERATION_PROMPT as default_task_generation_prompt,
            )
        else:
            default_dataset_examples_section = None
        self.task_generation_prompt = (
            task_generation_prompt if task_generation_prompt is not None else default_task_generation_prompt
        )
        self.environment_extraction_prompt = (
            environment_extraction_prompt
            if environment_extraction_prompt is not None
            else default_environment_extraction_prompt
        )
        self.grounded_environment_summary_prompt = (
            grounded_environment_summary_prompt
            if grounded_environment_summary_prompt is not None
            else default_grounded_environment_summary_prompt
        )
        self.complexity_level_descriptions = (
            complexity_level_descriptions
            if complexity_level_descriptions is not None
            else default_complexity_level_descriptions
        )

        self.trajectory_file_pattern = trajectory_file_pattern

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_thinking = use_thinking
        self.verbose = verbose
        self._llm = None
        self._embedding_client = None
        self.complexity_level = complexity_level
        self.embedding_model = embedding_model
        self.embedding_base_url = embedding_base_url
        self.semantic_oversample_multiplier = semantic_oversample_multiplier
        self.include_dataset_examples = include_dataset_examples
        self.prompt_context_provider = prompt_context_provider
        self.dataset_examples_section = (
            dataset_examples_section if dataset_examples_section is not None else (default_dataset_examples_section or "")
        )
        self._last_generation_prompts: List[Dict[str, Any]] = []
        self._last_generation_candidates: List[Dict[str, Any]] = []
        self._last_selected_tasks: List[Dict[str, Any]] = []
        self._environment_info = PrePingEnvironmentInfo(
            llm_provider=lambda: self.llm,
            environment_extraction_prompt=self.environment_extraction_prompt,
            trajectory_file_pattern=self.trajectory_file_pattern,
            verbose=self.verbose,
        )
        self._grounded_environment_summaries = PrePingGroundedEnvironmentSummaries(
            llm_provider=lambda: self.llm,
            summary_prompt=self.grounded_environment_summary_prompt,
            verbose=self.verbose,
        )

        logger.info(self.task_generation_prompt)
        logger.info(self.environment_extraction_prompt)

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

    @property
    def embedding_client(self) -> Optional[EmbeddingClient]:
        """Lazy initialization of embedding client for semantic de-duplication."""
        if self.embedding_model is None:
            return None
        if self._embedding_client is None:
            self._embedding_client = EmbeddingClient(
                model=self.embedding_model,
                base_url=self.embedding_base_url,
            )
        return self._embedding_client

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)

    @staticmethod
    def _flatten_task_history_instructions(task_history_summary: Optional[Dict[str, Any]]) -> List[str]:
        """Collect proposer-history instructions across buckets."""
        if not task_history_summary:
            return []

        instructions: List[str] = []
        # Flatten all history buckets into one instruction list for similarity checks.
        for key in (
            "success_tasks",
            "failure_tasks",
            "trivial_tasks",
            "boundary_tasks",
            "unsolved_feasible_tasks",
            "infeasible_tasks",
        ):
            for item in task_history_summary.get(key, []) or []:
                instruction = str(item.get("instruction", "")).strip()
                if instruction:
                    instructions.append(instruction)
        return instructions

    @staticmethod
    def _format_task_history_summary_for_prompt(task_history_summary: Optional[Dict[str, Any]]) -> str:
        """Render proposer memory into a compact prompt-friendly text block."""
        if not task_history_summary:
            return "None"

        summary = task_history_summary.get("summary") or {}
        lines: List[str] = []
        if summary:
            summary_bits = []
            for key in (
                "num_entries",
                "success_count",
                "failure_count",
                "infeasible_count",
            ):
                value = summary.get(key)
                if value is None:
                    continue
                summary_bits.append(f"{key}={value}")
            if summary_bits:
                lines.append("Summary: " + ", ".join(summary_bits))

        bucket_specs = (
            ("success_tasks", "Solved tasks", None, None, False),
            ("failure_tasks", "Failure tasks", "representative_failure_reason", 5, True),
            ("infeasible_tasks", "Infeasible tasks", "representative_infeasible_reason", 5, True),
        )

        for bucket_key, bucket_label, reason_key, max_items, use_recent_items in bucket_specs:
            items = task_history_summary.get(bucket_key) or []
            if not items:
                continue
            if lines and lines[-1] != "":
                lines.append("")
            lines.append(f"{bucket_label} ({len(items)}):")
            if max_items is None:
                visible_items = items
            elif use_recent_items:
                visible_items = items[-max_items:]
            else:
                visible_items = items[:max_items]
            for item in visible_items:
                instruction = str(item.get("instruction", "")).strip()
                if not instruction:
                    continue
                instruction = " ".join(instruction.split())

                line = f"- {instruction}"

                reason = str(item.get(reason_key, "")).strip() if reason_key else ""
                if reason:
                    reason = " ".join(reason.split())
                    line += f"\n  reason: {reason}"
                lines.append(line)

        return "\n".join(lines) if lines else "None"

    @classmethod
    def _build_task_history_section(cls, task_history_summary: Optional[Dict[str, Any]]) -> str:
        """Build the optional proposer-memory prompt section."""
        if not task_history_summary:
            return ""

        task_history_summary_str = cls._format_task_history_summary_for_prompt(task_history_summary)
        if task_history_summary_str == "None":
            return ""

        return (
            "## Prior Task History (Optional)\n\n"
            "- Use prior task history as weak guidance for where to explore next, not as templates to imitate.\n"
            "- Push beyond solved tasks with meaningfully harder or more diverse task structures.\n"
            "- Use failure tasks to target weak but still feasible capability regions.\n"
            "- Use infeasible tasks only to avoid invalid setups or impossible assumptions.\n"
            "- Treat failure or infeasibility reasons as short hints about missing skills or bad assumptions, not as instructions to reproduce the same task.\n"
            "- Avoid near-duplicates: do not merely rename entities or tweak dates, numbers, thresholds, or output format while keeping the same task pattern.\n"
            "- Keep the batch diverse across apps, entities, reasoning patterns, and action structures.\n\n"
            f"{task_history_summary_str}\n"
        )

    @staticmethod
    def _build_environment_info_section(environment_info: Optional[Dict[str, Any]] | Optional[str]) -> str:
        """Build the optional environment-info prompt section."""
        if isinstance(environment_info, str):
            env_info_str = environment_info.strip()
        else:
            env_info_str = json.dumps(environment_info, indent=2, ensure_ascii=False) if environment_info else ""

        if not env_info_str or env_info_str == "None":
            return ""

        return f"## Environment Information (Optional)\n{env_info_str}\n"

    @staticmethod
    def _build_memory_context_section(memory_context: Optional[List[str]]) -> str:
        """Build the optional agent-memory prompt section."""
        if not memory_context:
            return ""

        memory_context_str = "\n\n".join(section for section in memory_context if str(section).strip())
        if not memory_context_str.strip():
            return ""

        return f"## Agent Memory (Optional)\n{memory_context_str}\n"

    def _select_semantically_novel_tasks(
        self,
        *,
        tasks: List[Dict[str, Any]],
        task_history_summary: Optional[Dict[str, Any]],
        target_count: int,
    ) -> List[Dict[str, Any]]:
        """Greedily select tasks far from prior history and already selected tasks."""
        embedding_client = self.embedding_client
        if embedding_client is None:
            return tasks[:target_count]

        history_instructions = self._flatten_task_history_instructions(task_history_summary)
        candidate_instructions = [str(task.get("instruction", "")).strip() for task in tasks]
        texts = history_instructions + candidate_instructions
        if not any(candidate_instructions):
            return tasks[:target_count]

        try:
            # Embed both history and current candidates in one batch for consistent comparison.
            embeddings = embedding_client.embed_batch(texts)
        except Exception as exc:
            logger.error("Semantic de-duplication failed for embedding model %s: %s", self.embedding_model, exc)
            raise RuntimeError(
                f"Embedding request failed for semantic de-duplication: model={self.embedding_model}"
            ) from exc

        history_embeddings = embeddings[: len(history_instructions)]
        candidate_embeddings = embeddings[len(history_instructions) :]
        selected_indices: List[int] = []
        remaining_indices = list(range(len(candidate_embeddings)))

        while remaining_indices and len(selected_indices) < target_count:
            reference_embeddings = history_embeddings + [candidate_embeddings[idx] for idx in selected_indices]
            best_idx = remaining_indices[0]
            best_score = float("inf")

            for idx in remaining_indices:
                if not reference_embeddings:
                    max_similarity = 0.0
                else:
                    max_similarity = max(
                        self._cosine_similarity(candidate_embeddings[idx], reference_embedding)
                        for reference_embedding in reference_embeddings
                    )
                if max_similarity < best_score:
                    best_score = max_similarity
                    best_idx = idx

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return [tasks[idx] for idx in selected_indices]

    def extract_environment_info(self, trajectory_path: str) -> Dict[str, Any]:
        """Extract grounded environment information from one trajectory file."""
        return self._environment_info.extract_environment_info(trajectory_path)

    def extract_from_trajectory_paths(self, trajectory_paths: List[Path]) -> Dict[str, Any]:
        """Extract grounded environment information from trajectory files."""
        return self._environment_info.extract_from_trajectory_paths(trajectory_paths)

    def summarize_environment_from_results(
        self,
        *,
        cycle_index: int,
        representative_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build task-local grounded environment summaries from representative results."""
        return self._grounded_environment_summaries.summarize_from_results(
            cycle_index=cycle_index,
            representative_results=representative_results,
        )
    def _load_prompt_context(self, target_apps: Optional[List[str]] = None) -> str:
        """Return benchmark-specific prompt context for task generation."""
        if self.prompt_context_provider is not None:
            try:
                return str(self.prompt_context_provider(target_apps))
            except Exception as exc:  # noqa: BLE001
                logger.error("Prompt context provider failed; aborting task generation: %s", exc)
                raise
        return ""




    def get_last_generation_debug_artifact(self) -> Dict[str, Any]:
        """Return prompts used in the most recent generation call."""
        return {
            "prompts": list(self._last_generation_prompts),
            "generated_candidates": list(self._last_generation_candidates),
            "selected_tasks": list(self._last_selected_tasks),
        }

    def _estimate_generation_max_tokens(self, requested_tasks: int) -> int:
        """Scale output token budget with requested task count to avoid truncated JSON."""
        return max(self.max_tokens, min(12000, 800 + 240 * max(1, requested_tasks)))

    @contextmanager
    def _temporary_llm_max_tokens(self, max_tokens: int):
        """Temporarily override the LLM output token budget for large generation batches."""
        llm = self.llm
        original_max_tokens = getattr(llm, "max_tokens", None)
        if original_max_tokens is None:
            yield
            return

        llm.max_tokens = max_tokens
        try:
            yield
        finally:
            llm.max_tokens = original_max_tokens

    def _request_task_batch(
        self,
        *,
        prompt: str,
        grounded: bool,
        requested_tasks: int,
        max_retries: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """Request a single task batch and parse the JSON response."""
        generation_max_tokens = self._estimate_generation_max_tokens(requested_tasks)
        kind = "grounded" if grounded else "initial"

        if requested_tasks > 10:
            logger.info(
                "Task generation batch size=%s, using max_tokens=%s for %s generation",
                requested_tasks,
                generation_max_tokens,
                kind,
            )

        last_error = None
        for attempt in range(max_retries):
            if attempt > 0:
                logger.info(f"Retry attempt {attempt + 1}/{max_retries} for {kind} task generation...")

            with self._temporary_llm_max_tokens(generation_max_tokens):
                response = self.llm.generate(prompt)
            response_text = response if isinstance(response, str) else str(response)

            if self.verbose:
                logger.info(f"\n[GENERATION RESPONSE]\n{response_text}\n{'='*60}")

            try:
                tasks = PrePingEnvironmentInfo.parse_json_from_response(response_text)
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(f"Failed to parse {kind} tasks (attempt {attempt + 1}/{max_retries}): {e}")
                continue

            if not isinstance(tasks, list):
                last_error = TypeError("Task generation response is not a JSON array.")
                logger.warning(
                    "Failed to parse %s tasks (attempt %s/%s): response is not a JSON array",
                    kind,
                    attempt + 1,
                    max_retries,
                )
                continue
            return tasks

        logger.error(f"Failed to parse {kind} tasks after {max_retries} retries. Last error: {last_error}")
        return None

    def generate_tasks(
        self,
        memory_context: Optional[List[str]] = None,
        task_history_summary: Optional[Dict[str, Any]] = None,
        environment_info: Optional[Dict[str, Any] | str] = None,
        num_tasks: int = 10,
        target_apps: Optional[List[str]] = None,
        max_retries: int = 3,
        complexity_level: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate tasks. If environment_info is None, generates initial tasks with
        placeholders (grounded=False). Otherwise uses discovered entities (grounded=True).

        Args:
            environment_info: Optional grounded environment context for generation.
            num_tasks: Number of tasks to generate
            target_apps: Optional list of apps to focus on
            max_retries: Maximum number of retries on JSON parsing failure
            memory_context: Optional list of memory/playbook sections. When provided,
                the generator analyzes gaps in the memory and produces tasks
                that target under-represented or missing logic.
            task_history_summary: Optional Proposer-memory summary grouped by outcome buckets.

        Returns:
            List of task dictionaries
        """
        grounded = environment_info is not None
        environment_info_section = self._build_environment_info_section(environment_info)
        memory_context_section = self._build_memory_context_section(memory_context)
        task_history_section = self._build_task_history_section(task_history_summary)

        # Use provided complexity_level or fall back to instance default
        effective_complexity = complexity_level if complexity_level is not None else self.complexity_level

        api_docs_str = self._load_prompt_context(target_apps)

        if "{complexity_level_description}" not in self.task_generation_prompt:
            logger.warning(
                "Complexity level is configured but the active task-generation prompt does not use complexity guidance; "
                "provided complexity level will be ignored."
            )

        generation_num_tasks = num_tasks
        if self.embedding_client is not None and task_history_summary is not None:
            # Oversample when semantic filtering is active so final selection has room to prune duplicates.
            generation_num_tasks = min(num_tasks * max(1, self.semantic_oversample_multiplier), 50)
            logger.info(f'[Proposer] Generate {generation_num_tasks} TASKS')

        def build_prompt(requested_tasks: int) -> str:
            return self.task_generation_prompt.format(
                api_docs=api_docs_str,
                environment_info_section=environment_info_section,
                complexity_level_description=self.complexity_level_descriptions.get(effective_complexity, ""),
                num_tasks=requested_tasks,
                memory_context_section=memory_context_section,
                task_history_section=task_history_section,
                dataset_examples_section=self.dataset_examples_section if self.include_dataset_examples else "",
            )

        tasks: Optional[List[Dict[str, Any]]] = []
        remaining = generation_num_tasks
        batch_size = 10
        batch_index = 0
        self._last_generation_prompts = []
        self._last_generation_candidates = []
        self._last_selected_tasks = []
        while remaining > 0:
            requested_batch = min(batch_size, remaining)
            # Large requests are split into smaller LLM calls to keep each batch manageable.
            prompt = build_prompt(requested_batch)
            self._last_generation_prompts.append(
                {
                    "batch_index": batch_index,
                    "requested_tasks": requested_batch,
                    "prompt": prompt,
                }
            )
            if self.verbose:
                kind = "grounded" if grounded else "initial"
                logger.info(
                    f"\n{'='*60}\n[{kind.upper()} TASK GENERATION PROMPT]\n{'='*60}\n{prompt}\n{'='*60}"
                )

            batch_tasks = self._request_task_batch(
                prompt=prompt,
                grounded=grounded,
                requested_tasks=requested_batch,
                max_retries=max_retries,
            )
            if batch_tasks is None:
                tasks = None
                break
            self._last_generation_candidates.extend(
                {
                    "batch_index": batch_index,
                    "candidate_index": candidate_index,
                    "task": copy.deepcopy(task),
                }
                for candidate_index, task in enumerate(batch_tasks)
            )
            tasks.extend(batch_tasks)
            remaining -= requested_batch
            batch_index += 1

        if tasks is None:
            return []

        # Final selection keeps semantically novel tasks relative to history and within-batch duplicates.
        tasks = self._select_semantically_novel_tasks(
            tasks=tasks,
            task_history_summary=task_history_summary,
            target_count=num_tasks,
        )
        self._last_selected_tasks = copy.deepcopy(tasks)
        for idx, task in enumerate(tasks):
            task['task_id'] = str(idx)
            task['grounded'] = grounded
        return tasks
