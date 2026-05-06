"""
PrePing Curator Module

The Curator is the "Librarian" component that manages the Playbook by:
- Merging new insights from the Reflector
- De-duplicating semantically similar bullets
- Maintaining the quality and coherence of the knowledge base
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

from .prompts import CURATOR_SYSTEM_PROMPT
from .trajectory_formatters import TrajectoryPromptFormatter, format_trajectory_for_prompt
from .types import Bullet, BulletTag, Playbook, PlaybookSection


if TYPE_CHECKING:
    from .retriever import PlaybookRetriever

logger = logging.getLogger(__name__)


@dataclass
class CuratorOperation:
    """A single operation to perform on the Playbook."""
    operation_type: str  # "ADD", "UPDATE", "REMOVE"
    section: PlaybookSection
    content: str
    target_bullet_id: Optional[str] = None  # For UPDATE/REMOVE operations


@dataclass
class CuratorInput:
    """Input data for the Curator."""
    question_context: str
    current_playbook: str  # Formatted playbook string
    trajectory: List[Dict[str, str]]  # List of trajectory entries
    reflections: str  # Key insights from Reflector
    trajectory_format: str = "llm_history"  # 'action_output', 'llm_history', or a custom formatter key


@dataclass
class CuratorOutput:
    """Output from the Curator."""
    reasoning: str
    operations: List[CuratorOperation]
    raw_response: Dict[str, Any]
    prompt: str
    response_text: str


class Curator:
    """
    Manages the Playbook by merging new insights.

    The Curator implements:
    - Deterministic Merging: Uses non-LLM logic where possible
    - Semantic De-duplication: Finds and merges similar bullets (via Retriever)
    - Grow-and-Refine: Adds new insights while maintaining quality
    """

    def __init__(
        self,
        llm_client: Any,
        retriever: Optional["PlaybookRetriever"] = None,
        similarity_threshold: float = 0.85,
        include_task_context: bool = True,
        system_prompt: str = CURATOR_SYSTEM_PROMPT,
        trajectory_formatters: Mapping[str, TrajectoryPromptFormatter] | None = None,
    ):
        """
        Initialize the Curator.

        Args:
            llm_client: An LLM client with a `generate` method.
            retriever: Optional PlaybookRetriever for semantic deduplication.
            similarity_threshold: Threshold for semantic similarity matching.
            include_task_context: If False, exclude task context from prompts.
                Useful for static doc playbook building or exploration agents.
        """
        self.llm_client = llm_client
        self.retriever = retriever
        self.similarity_threshold = similarity_threshold
        self.include_task_context = include_task_context
        self.system_prompt = system_prompt
        self.trajectory_formatters = dict(trajectory_formatters or {})

    def _build_prompt(self, input_data: CuratorInput) -> str:
        """Build the complete prompt for the Curator LLM call."""
        prompt = self.system_prompt

        # Replace template variables
        # Conditionally include task context
        if self.include_task_context and input_data.question_context:
            prompt = prompt.replace("{question_context}", input_data.question_context)
        else:
            prompt = prompt.replace("{question_context}", "(No specific task - exploring API documentation)")
        prompt = prompt.replace("{current_playbook}", input_data.current_playbook)

        # Format trajectory
        trajectory_str = self._format_trajectory(input_data.trajectory, input_data.trajectory_format)
        prompt = prompt.replace("{trajectory}", trajectory_str)
        prompt = prompt.replace("{guidebook}", input_data.reflections)

        return prompt

    def _format_trajectory(self, trajectory: List[Dict[str, str]], trajectory_format: str = "action_output") -> str:
        """Format trajectory as a string for the prompt."""
        return format_trajectory_for_prompt(
            trajectory,
            trajectory_format,
            custom_formatters=self.trajectory_formatters,
        )

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the JSON response from the LLM."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass

        logger.warning("Failed to parse Curator response as JSON")
        return {"reasoning": response, "operations": []}

    def _section_from_string(self, section_str: str) -> PlaybookSection:
        """Convert a section string to PlaybookSection enum."""
        section_map = {
            "strategies": PlaybookSection.STRATEGIES,
            "strategies_and_hard_rules": PlaybookSection.STRATEGIES,
            "code_snippets": PlaybookSection.CODE_SNIPPETS,
            "pitfalls": PlaybookSection.PITFALLS,
            "apis": PlaybookSection.APIS,
            "apis_to_use_for_specific_information": PlaybookSection.APIS,
            "verification_checklist": PlaybookSection.STRATEGIES,
        }
        return section_map.get(section_str.lower(), PlaybookSection.STRATEGIES)

    def curate(self, input_data: CuratorInput) -> CuratorOutput:
        """
        Analyze reflections and determine what to add to the Playbook.

        Args:
            input_data: The CuratorInput containing context and reflections.

        Returns:
            CuratorOutput with operations to perform on the Playbook.
        """
        prompt = self._build_prompt(input_data)

        logger.debug(f"[Curator] PROMPT:\n{'='*60}\n{prompt}\n{'='*60}")

        messages = [
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.generate(messages, flex_mode=False)
        logger.debug(f"[Curator] RESPONSE:\n{'='*60}\n{response}\n{'='*60}")

        parsed = self._parse_response(response)

        operations = []
        for op_data in parsed.get("operations", []):
            op = CuratorOperation(
                operation_type=op_data.get("type", "ADD"),
                section=self._section_from_string(op_data.get("section", "strategies")),
                content=op_data.get("content", ""),
                target_bullet_id=op_data.get("bullet_id"),
            )
            operations.append(op)

        return CuratorOutput(
            reasoning=parsed.get("reasoning", ""),
            operations=operations,
            raw_response=parsed,
            prompt=prompt,
            response_text=response,
        )

    def apply_operations(
        self,
        playbook: Playbook,
        operations: List[CuratorOperation],
        task_id: Optional[str] = None
    ) -> List[Bullet]:
        """
        Apply curated operations to the Playbook.

        This implements deterministic merging where possible:
        - For ADD: Check for duplicates before adding
        - For UPDATE: Increment version and update content
        - For REMOVE: Remove the specified bullet

        Args:
            playbook: The Playbook to modify.
            operations: List of operations from the curate() method.
            task_id: Optional task ID for tracking.

        Returns:
            List of Bullets that were added or modified.
        """
        modified_bullets = []

        for op in operations:
            if op.operation_type == "ADD":
                # Check for exact duplicates (deterministic)
                existing = self._find_exact_duplicate(playbook, op.content, op.section)
                if existing:
                    # Just increment the helpful count instead of adding
                    existing.mark_helpful()
                    modified_bullets.append(existing)
                    logger.info(f"Reinforced existing bullet: {existing.id}")
                else:
                    # Create new bullet
                    bullet = Bullet(
                        content=op.content,
                        section=op.section,
                        tag=BulletTag.HELPFUL,
                    )
                    bullet.metadata.source_task_id = task_id
                    playbook.add_bullet(bullet)
                    modified_bullets.append(bullet)
                    logger.info(f"Added new bullet: {bullet.id}")

            elif op.operation_type == "UPDATE" and op.target_bullet_id:
                if playbook.update_bullet(op.target_bullet_id, op.content):
                    bullet = playbook.get_bullet(op.target_bullet_id)
                    if bullet:
                        modified_bullets.append(bullet)
                        logger.info(f"Updated bullet: {bullet.id}")
                else:
                    logger.warning(f"Bullet not found for update: {op.target_bullet_id}")

            elif op.operation_type == "REMOVE" and op.target_bullet_id:
                if playbook.remove_bullet(op.target_bullet_id):
                    logger.info(f"Removed bullet: {op.target_bullet_id}")
                else:
                    logger.warning(f"Bullet not found for removal: {op.target_bullet_id}")

        return modified_bullets

    def _find_exact_duplicate(
        self,
        playbook: Playbook,
        content: str,
        section: PlaybookSection
    ) -> Optional[Bullet]:
        """Find an exact content duplicate in the specified section."""
        normalized_content = content.strip().lower()
        for bullet in playbook.get_section(section):
            if bullet.content.strip().lower() == normalized_content:
                return bullet
        return None

    def find_similar_bullets(
        self,
        playbook: Playbook,
        content: str,
        section: Optional[PlaybookSection] = None
    ) -> List[tuple[Bullet, float]]:
        """
        Find semantically similar bullets using the Retriever.

        Args:
            playbook: The Playbook to search.
            content: The content to find similar bullets for.
            section: Optional section to limit the search.

        Returns:
            List of (Bullet, similarity_score) tuples, sorted by score descending.
        """
        if not self.retriever:
            return []

        try:
            return self.retriever.find_similar_bullets(
                content=content,
                section=section,
                threshold=self.similarity_threshold,
            )
        except ValueError:
            # embedding_client not configured
            return []

    def merge_with_deduplication(
        self,
        playbook: Playbook,
        candidate_bullets: List[Bullet],
        task_id: Optional[str] = None
    ) -> List[Bullet]:
        """
        Merge candidate bullets into the Playbook with deduplication.

        This is a convenience method that:
        1. Checks for exact duplicates (reinforces them)
        2. Checks for semantic duplicates (merges if similar)
        3. Adds truly new bullets

        Args:
            playbook: The Playbook to modify.
            candidate_bullets: List of candidate Bullets to potentially add.
            task_id: Optional task ID for tracking.

        Returns:
            List of Bullets that were added or modified.
        """
        modified = []

        for candidate in candidate_bullets:
            # Check exact duplicate first
            exact = self._find_exact_duplicate(
                playbook, candidate.content, candidate.section
            )
            if exact:
                exact.mark_helpful()
                modified.append(exact)
                continue

            # Check semantic similarity
            similar = self.find_similar_bullets(
                playbook, candidate.content, candidate.section
            )
            if similar:
                # Reinforce the most similar existing bullet
                best_match, score = similar[0]
                best_match.mark_helpful()
                logger.info(
                    f"Found similar bullet (score={score:.2f}), "
                    f"reinforcing: {best_match.id}"
                )
                modified.append(best_match)
                continue

            # Add as new bullet
            candidate.metadata.source_task_id = task_id
            playbook.add_bullet(candidate)
            modified.append(candidate)
            logger.info(f"Added new bullet: {candidate.id}")

        return modified
