"""
PrePing Memory Manager Module

The main interface for integrating PrePing with agents. Orchestrates the
Reflector -> Curator pipeline and provides methods for retrieving
relevant Playbook content.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from preping.core.memory import MemoryManager
from preping.llm import LLMManager

from .curator import Curator, CuratorInput
from .prompts import CURATOR_SYSTEM_PROMPT, REFLECTOR_SYSTEM_PROMPT
from .reflector import Reflector, ReflectorInput
from .retriever import PlaybookRetriever
from .trajectory_formatters import TrajectoryPromptFormatter
from .types import Bullet, BulletTag, Playbook, PlaybookSection


logger = logging.getLogger(__name__)


def _truncate_for_log(text: Optional[str], limit: int = 800) -> str:
    if not text:
        return ""
    clean = text.strip()
    if len(clean) <= limit:
        return clean
    return f"{clean[:limit]}..."


def _write_memory_debug_artifacts(
    debug_output_dir: Path,
    *,
    reflector_output,
    curator_output,
) -> None:
    debug_output_dir.mkdir(parents=True, exist_ok=True)
    (debug_output_dir / "reflector_prompt.txt").write_text(reflector_output.prompt, encoding="utf-8")
    (debug_output_dir / "reflector_response.txt").write_text(reflector_output.response_text, encoding="utf-8")
    (debug_output_dir / "reflector_response.json").write_text(
        json.dumps(reflector_output.raw_response, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (debug_output_dir / "curator_prompt.txt").write_text(curator_output.prompt, encoding="utf-8")
    (debug_output_dir / "curator_response.txt").write_text(curator_output.response_text, encoding="utf-8")
    (debug_output_dir / "curator_response.json").write_text(
        json.dumps(curator_output.raw_response, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


class PrePingMemoryManager(MemoryManager):
    """
    Main interface for the PrePing playbook memory system.

    This manager:
    - Maintains the Playbook (persistent knowledge base)
    - Retrieves relevant bullets for task execution
    - Orchestrates the Reflector -> Curator pipeline after episodes
    """

    def __init__(
        self,
        model_name: str,
        playbook_path: Optional[Path] = None,
        temperature: float = 0.7,
        max_tokens: int = 3000,
        use_thinking: bool = False,
        embedding_client: Optional[Any] = None,
        include_task_context: bool = True,
        use_ground_truth: bool = True,
        reflector_system_prompt: Optional[str] = None,
        curator_system_prompt: Optional[str] = None,
        trajectory_formatters: Mapping[str, TrajectoryPromptFormatter] | None = None,
    ):
        """
        Initialize the PrePing Memory Manager.

        Args:
            model_name: Name of the LLM model to use for Reflector and Curator.
            playbook_path: Optional path to save/load the Playbook.
            temperature: Temperature for LLM generation.
            embedding_client: Optional client for semantic retrieval.
            include_task_context: If False, exclude task descriptions and supervisor
                info from prompts. Useful for static doc playbook building or
                exploration agents where no specific task context exists.
            use_ground_truth: If True, use ground truth (final_reward) to determine
                success/failure. If False, use trajectory completion status.
            trajectory_formatters: Optional benchmark-specific prompt renderers
                keyed by trajectory_format name.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_thinking = use_thinking
        self.embedding_client = embedding_client
        self.playbook_path = playbook_path
        self.include_task_context = include_task_context
        self.use_ground_truth = use_ground_truth

        # Initialize LLM client internally
        self.llm_client = LLMManager.create_llm(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            use_thinking=use_thinking,
        )

        # Load or create Playbook (must be before retriever)
        if playbook_path and playbook_path.exists():
            self.playbook = Playbook.load(playbook_path)
            logger.info(f"Loaded Playbook with {self.playbook.get_bullet_count()} bullets")
        else:
            self.playbook = Playbook()
            logger.info("Created new empty Playbook")

        # Initialize retriever first (curator depends on it)
        self.retriever = PlaybookRetriever(
            playbook=self.playbook,
            embedding_client=embedding_client,
        )

        # Initialize components
        self.reflector = Reflector(
            self.llm_client,
            include_task_context=include_task_context,
            system_prompt=reflector_system_prompt or REFLECTOR_SYSTEM_PROMPT,
            trajectory_formatters=trajectory_formatters,
        )
        self.curator = Curator(
            llm_client=self.llm_client,
            retriever=self.retriever,
            include_task_context=include_task_context,
            system_prompt=curator_system_prompt or CURATOR_SYSTEM_PROMPT,
            trajectory_formatters=trajectory_formatters,
        )

    def retrieve_relevant_bullets(
        self,
        task_description: str,
        sections: Optional[List[PlaybookSection]] = None,
        max_bullets: int = 10,
        min_effectiveness: float = -0.5,
    ) -> List[Bullet]:
        """
        Retrieve relevant bullets from the Playbook for the current task.

        Delegates to PlaybookRetriever which uses semantic similarity
        (if embedding_client is available) or keyword-based matching.

        Args:
            task_description: The task description or context to match against.
            sections: Optional list of sections to search. If None, searches all.
            max_bullets: Maximum number of bullets to return.
            min_effectiveness: Minimum effectiveness score to include.

        Returns:
            List of relevant Bullets sorted by relevance score.
        """
        return self.retriever.retrieve(
            task_description=task_description,
            sections=sections,
            max_bullets=max_bullets,
            min_effectiveness=min_effectiveness,
        )

    def get_playbook_for_prompt(
        self,
        task_description: Optional[str] = None,
        sections: Optional[List[PlaybookSection]] = None,
        max_bullets: Optional[int] = None,
        min_effectiveness: float = -0.5,
    ) -> str:
        """
        Get a formatted Playbook string for injection into prompts.

        If task_description is provided, retrieves only relevant bullets using semantic search.
        Otherwise, returns all bullets (optionally filtered by sections).

        Args:
            task_description: Optional task description for semantic retrieval.
                If provided, only relevant bullets are returned.
            sections: Optional list of sections to include.
            max_bullets: Maximum number of bullets to return.
                If None, returns all matching bullets.
            min_effectiveness: Minimum effectiveness score to include (only used with task_description).

        Returns:
            Formatted string representation of the Playbook.
        """
        if task_description is not None and self.embedding_client is not None:
            # Semantic retrieval mode
            result = self.retriever.retrieve_formatted(
                task_description=task_description,
                sections=sections,
                max_bullets=max_bullets if max_bullets else 1000,
                min_effectiveness=min_effectiveness,
            )
            logger.info(f"Retrieved playbook snippet for task_description ({len(result)} chars)")
            return result
        else:
            # Full playbook mode
            return self.playbook.format_for_prompt(
                sections=sections,
                max_per_section=max_bullets,
                sort_by_effectiveness=True,
            )

    def get_memory(
        self,
        task_description: Optional[str] = None,
        max_bullets: Optional[int] = None,
        min_effectiveness: float = -0.5,
    ) -> List[str]:
        """
        Get formatted context sections for prompt injection.

        High-level wrapper around get_playbook_for_prompt() that returns
        ready-to-use context sections with standard header/footer formatting.

        Args:
            task_description: Optional task description or task_id for semantic retrieval.
                If provided and embedding_client is configured, returns only
                relevant bullets. Otherwise returns the full playbook.
            max_bullets: Maximum number of bullets to retrieve.
            min_effectiveness: Minimum effectiveness score to include.

        Returns:
            List of context section strings for prompt injection.
            Empty list if no playbook content is available.
        """
        playbook_str = self.get_playbook_for_prompt(
            task_description=task_description,
            max_bullets=max_bullets,
            min_effectiveness=min_effectiveness,
        )
        if not playbook_str:
            return []

        return [
            "# Playbook\n"
            "The following are insights and strategies from previous tasks. "
            "Use them to guide your approach:\n\n"
            "You are also provided with a curated cheatsheet of strategies, API-specific information, "
            "common mistakes, and proven solutions to help you solve the task effectively.\n"
            "PrePing Playbook: - Read the Playbook first, then execute the task by explicitly leveraging each relevant section:\n"
            f"PLAYBOOK_BEGIN\n{playbook_str}\nPLAYBOOK_END\n\n"
        ]

    def process_episode(
        self,
        task_description: str,
        trajectory: List[Dict[str, str]],
        result: str,
        task_id: Optional[str] = None,
        ground_truth_code: Optional[str] = None,
        unit_test_results: Optional[str] = None,
        trajectory_format: str = "llm_history",  # built-in or custom formatter key
        debug_output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a completed episode through the Reflector -> Curator pipeline.

        This is the main method called after a task completes to learn from
        the experience and update the Playbook.

        Args:
            task_description: Description of the task that was attempted.
            trajectory: List of trajectory entries. Format depends on trajectory_format:
                - 'action_output': List of {"action": ..., "output": ...} pairs
                - 'llm_history': List of {"role": "user"/"assistant", "content": ...} messages
                - custom benchmark-specific formats when registered at manager construction
            result: "success" or "failure".
            task_id: Optional unique identifier for this task.
            ground_truth_code: Optional reference implementation.
            unit_test_results: Optional test results.
            trajectory_format: Format of the trajectory.

        Returns:
            Dictionary with processing results:
            - reflector_output: The Reflector's analysis
            - curator_output: The Curator's decisions
            - bullets_modified: List of modified bullet IDs
        """
        # Get current playbook state for reflection
        playbook_str = self.get_playbook_for_prompt()

        # Step 1: Reflect on the trajectory
        reflector_input = ReflectorInput(
            task_description=task_description if self.include_task_context else "",
            trajectory=trajectory,
            result=result if self.use_ground_truth else None,
            ground_truth_code=ground_truth_code if self.use_ground_truth else None,
            unit_test_results=unit_test_results if self.use_ground_truth else None,
            playbook_used=playbook_str,
            trajectory_format=trajectory_format,
        )

        reflector_output = self.reflector.analyze(reflector_input)
        logger.info(f"Reflector analysis complete: {reflector_output.key_insight[:200]}...")
        logger.info(
            "Reflector details\nreasoning: %s\nerror_identification: %s\nroot_cause_analysis: %s\ncorrect_approach: %s\nkey_insight: %s",
            _truncate_for_log(reflector_output.reasoning),
            _truncate_for_log(reflector_output.error_identification, limit=500),
            _truncate_for_log(reflector_output.root_cause_analysis),
            _truncate_for_log(reflector_output.correct_approach),
            _truncate_for_log(reflector_output.key_insight, limit=500),
        )

        # Step 1.5: Apply bullet-level feedback from Reflector
        feedback_applied = 0
        for bullet_id, tag in reflector_output.bullet_tags.items():
            if tag == BulletTag.HELPFUL:
                if self.update_bullet_feedback(bullet_id, was_helpful=True):
                    feedback_applied += 1
                    logger.debug(f"Marked bullet {bullet_id} as helpful")
            elif tag == BulletTag.HARMFUL:
                if self.update_bullet_feedback(bullet_id, was_helpful=False):
                    feedback_applied += 1
                    logger.debug(f"Marked bullet {bullet_id} as harmful")
            # NEUTRAL bullets are skipped - no feedback update needed

        if feedback_applied > 0:
            logger.info(f"Applied feedback to {feedback_applied} bullets based on Reflector analysis")

        # Step 2: Curate the Playbook
        curator_input = CuratorInput(
            question_context=task_description if self.include_task_context else "",
            current_playbook=playbook_str,
            trajectory=trajectory,
            reflections=reflector_output.key_insight,
            trajectory_format=trajectory_format,
        )

        curator_output = self.curator.curate(curator_input)
        logger.info(f"Curator proposed {len(curator_output.operations)} operations")
        logger.info(
            "Curator reasoning: %s",
            _truncate_for_log(curator_output.reasoning),
        )
        for idx, operation in enumerate(curator_output.operations, start=1):
            logger.info(
                "Curator operation %s: type=%s section=%s content=%s",
                idx,
                operation.operation_type,
                operation.section.value,
                _truncate_for_log(operation.content, limit=500),
            )

        if debug_output_dir:
            _write_memory_debug_artifacts(
                Path(debug_output_dir),
                reflector_output=reflector_output,
                curator_output=curator_output,
            )


        # Step 3: Apply operations to the Playbook
        modified_bullets = self.curator.apply_operations(
            self.playbook,
            curator_output.operations,
            task_id=task_id,
        )

        # Step 4: Save the updated Playbook
        if self.playbook_path:
            self.playbook.save(self.playbook_path)
            logger.info(f"Saved Playbook to {self.playbook_path}")

        return {
            "reflector_output": reflector_output,
            "curator_output": curator_output,
            "bullets_modified": [str(b.id) for b in modified_bullets],
            "feedback_applied": feedback_applied,
        }


    def update_bullet_feedback(
        self,
        bullet_id: str,
        was_helpful: bool,
    ) -> bool:
        """
        Update a bullet's feedback counters based on usage outcome.

        Call this after using a bullet to track whether it helped or harmed.

        Args:
            bullet_id: The string ID of the bullet (e.g., "strategies-001").
            was_helpful: True if the bullet helped, False if it harmed.

        Returns:
            True if the bullet was found and updated, False otherwise.
        """
        bullet = self.playbook.get_bullet(bullet_id)
        if bullet:
            if was_helpful:
                bullet.mark_helpful()
            else:
                bullet.mark_harmful()

            # Update tag based on effectiveness
            score = bullet.get_effectiveness_score()
            if score > 0.3:
                bullet.tag = BulletTag.HELPFUL
            elif score < -0.3:
                bullet.tag = BulletTag.HARMFUL
            else:
                bullet.tag = BulletTag.NEUTRAL

            return True

        logger.warning(f"Bullet not found: {bullet_id}")
        return False

    def prune_playbook(
        self,
        threshold: float = -0.3,
        max_bullets: Optional[int] = None,
    ) -> List[str]:
        """
        Prune low-value bullets from the Playbook.

        This implements "Lazy Pruning" - call when the Playbook is too large
        or context limits are being approached.

        Args:
            threshold: Minimum effectiveness score to keep.
            max_bullets: Optional maximum number of bullets to keep.

        Returns:
            List of removed bullet IDs.
        """
        removed = self.playbook.prune_low_value_bullets(threshold, max_bullets)

        if removed:
            logger.info(f"Pruned {len(removed)} low-value bullets")
            if self.playbook_path:
                self.playbook.save(self.playbook_path)

        return [str(b.id) for b in removed]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current Playbook.

        Returns:
            Dictionary with Playbook statistics.
        """
        all_bullets = self.playbook.get_all_bullets()

        avg_effectiveness = 0.0
        if all_bullets:
            avg_effectiveness = sum(
                b.get_effectiveness_score() for b in all_bullets
            ) / len(all_bullets)

        return {
            "total_bullets": self.playbook.get_bullet_count(),
            "section_counts": self.playbook.get_section_counts(),
            "average_effectiveness": avg_effectiveness,
            "tag_distribution": {
                tag.value: sum(1 for b in all_bullets if b.tag == tag)
                for tag in BulletTag
            },
        }

    def freeze(self) -> None:
        """
        Freeze the playbook (mark as read-only for evaluation).

        After freezing:
        - No new bullets can be added
        - Existing bullets cannot be updated
        - Playbook is saved with '.frozen.json' suffix
        """
        self.playbook.freeze()
        if self.playbook_path:
            frozen_path = self.playbook_path.with_suffix('.frozen.json')
            self.playbook.save(frozen_path)
            logger.info(f"Frozen playbook saved to {frozen_path}")

    def is_frozen(self) -> bool:
        """Check if playbook is frozen."""
        return self.playbook.is_frozen()
