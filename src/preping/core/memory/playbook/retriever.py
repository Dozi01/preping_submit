"""
PrePing Retriever Module

Handles retrieval of relevant bullets from the Playbook using
semantic similarity search.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .types import Bullet, Playbook, PlaybookSection

logger = logging.getLogger(__name__)


def _to_np(embedding: Any) -> Any:
    """Normalize embedding to numpy array."""
    import numpy as np
    return np.array(embedding) if not isinstance(embedding, np.ndarray) else embedding


class PlaybookRetriever:
    """
    Retrieves relevant bullets from a Playbook based on query similarity.
    
    Uses semantic search with embeddings (requires embedding_client).
    """

    def __init__(
        self,
        playbook: Playbook,
        embedding_client: Optional[Any] = None,
    ):
        """
        Initialize the Retriever.
        
        Args:
            playbook: The Playbook to retrieve from.
            embedding_client: Client for computing embeddings.
                Should have an `embed(text: str) -> List[float]` method.
        """
        self.playbook = playbook
        self.embedding_client = embedding_client

    def _get_candidates(
        self,
        sections: Optional[List[PlaybookSection]],
        min_effectiveness: float,
    ) -> List[Bullet]:
        """Get candidate bullets from sections, filtered by min_effectiveness."""
        secs = sections if sections is not None else list(PlaybookSection)
        candidates = []
        for section in secs:
            candidates.extend(self.playbook.get_section(section))
        return [b for b in candidates if b.get_effectiveness_score() >= min_effectiveness]

    def retrieve(
        self,
        task_description: str,
        sections: Optional[List[PlaybookSection]] = None,
        max_bullets: int = 10,
        min_effectiveness: float = -0.5,
        min_relevance_score: float = 0.0,
    ) -> List[Bullet]:
        """
        Retrieve relevant bullets from the Playbook for the given task description.
        
        Uses semantic similarity search with embeddings.
        
        Args:
            task_description: The task description or context to match against.
            sections: Optional list of sections to search. If None, searches all.
            max_bullets: Maximum number of bullets to return.
            min_effectiveness: Minimum effectiveness score to include.
            min_relevance_score: Minimum relevance score to include in results.
        
        Returns:
            List of relevant Bullets sorted by relevance score (descending).
        
        Raises:
            ValueError: If embedding_client is not configured.
        """
        scored = self.retrieve_with_scores(
            task_description=task_description,
            sections=sections,
            max_bullets=max_bullets,
            min_effectiveness=min_effectiveness,
        )
        relevant = [b for b, s in scored if s > min_relevance_score]
        logger.debug(f"Retrieved {len(relevant)} relevant bullets (min_relevance_score={min_relevance_score})")
        return relevant

    def retrieve_with_scores(
        self,
        task_description: str,
        sections: Optional[List[PlaybookSection]] = None,
        max_bullets: Optional[int] = 10,
        min_effectiveness: Optional[float] = -0.5,
    ) -> List[Tuple[Bullet, float]]:
        """
        Retrieve relevant bullets along with their relevance scores.
        
        Args:
            task_description: The task description or context to match against.
            sections: Optional list of sections to search. If None, searches all.
            max_bullets: Maximum number of bullets to return.
            min_effectiveness: Minimum effectiveness score to include.
        
        Returns:
            List of (Bullet, score) tuples sorted by relevance score (descending).
        
        Raises:
            ValueError: If embedding_client is not configured.
        """
        if self.embedding_client is None:
            raise ValueError("embedding_client is required for retrieval")
        candidates = self._get_candidates(sections, min_effectiveness or -0.5)
        if not candidates:
            return []
        scored = self._semantic_search(task_description, candidates)
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:max_bullets]

    def retrieve_formatted(
        self,
        task_description: str,
        sections: Optional[List[PlaybookSection]] = None,
        max_bullets: Optional[int] = None,
        min_effectiveness: float = -0.5,
    ) -> str:
        """
        Retrieve relevant bullets and return them formatted for prompt injection.
        Groups bullets by section.
        """
        bullets = self.retrieve(
            task_description=task_description,
            sections=sections,
            max_bullets=max_bullets if max_bullets is not None else 1000,
            min_effectiveness=min_effectiveness,
        )
        if not bullets:
            return "No relevant playbook entries found."
        section_bullets: Dict[PlaybookSection, List[Bullet]] = {
            s: [] for s in PlaybookSection
        }
        for b in bullets:
            section_bullets[b.section].append(b)
        lines = []
        for section in PlaybookSection:
            section_list = section_bullets[section]
            if section_list:
                lines.append(f"## {section.value.replace('_', ' ').title()}")
                for b in section_list:
                    lines.append(f"- {b.format_for_prompt()}")
                lines.append("")
        return "\n".join(lines)

    def _semantic_search(
        self,
        task_description: str,
        candidates: List[Bullet],
    ) -> List[Tuple[Bullet, float]]:
        """Perform semantic similarity search using embeddings."""
        query_embedding = _to_np(self.embedding_client.embed(task_description))
        return self._semantic_search_with_embedding(query_embedding, candidates)

    def _cosine_similarity(self, vec1, vec2) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))

    def find_similar_bullets(
        self,
        content: str,
        section: Optional[PlaybookSection] = None,
        threshold: float = 0.85,
    ) -> List[Tuple[Bullet, float]]:
        """
        Find semantically similar bullets using embeddings.
        
        Used for deduplication when adding new bullets to the Playbook.
        
        Args:
            content: The content to find similar bullets for.
            section: Optional section to limit the search.
            threshold: Minimum similarity threshold to include.
        
        Returns:
            List of (Bullet, similarity_score) tuples, sorted by score descending.
        
        Raises:
            ValueError: If embedding_client is not configured.
        """
        if self.embedding_client is None:
            raise ValueError("embedding_client is required for similarity search")
        
        # Get embedding for the new content
        try:
            new_embedding = self.embedding_client.embed(content)
        except Exception as e:
            logger.warning(f"Failed to get embedding: {e}")
            return []
        
        # Get candidates
        if section is not None:
            candidates = self.playbook.get_section(section)
        else:
            candidates = self.playbook.get_all_bullets()
        
        if not candidates:
            return []
        
        # Score all candidates
        scored = self._semantic_search_with_embedding(new_embedding, candidates)
        
        # Filter by threshold and sort
        similar = [(bullet, score) for bullet, score in scored if score >= threshold]
        return sorted(similar, key=lambda x: x[1], reverse=True)

    def _semantic_search_with_embedding(
        self,
        query_embedding,
        candidates: List[Bullet],
    ) -> List[Tuple[Bullet, float]]:
        """
        Perform semantic search with a pre-computed query embedding.
        
        Args:
            query_embedding: Pre-computed embedding vector.
            candidates: List of candidate bullets to search.
        
        Returns:
            List of (bullet, score) tuples.
        """
        query_embedding = _to_np(query_embedding)
        result = []
        for bullet in candidates:
            if bullet.embedding is None:
                bullet.embedding = self.embedding_client.embed(bullet.content)
            result.append((bullet, self._cosine_similarity(query_embedding, _to_np(bullet.embedding))))
        return result
