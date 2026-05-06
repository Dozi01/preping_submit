"""
PrePing Types Module

Defines the fundamental data structures for the PrePing playbook memory system:
- Bullet: An individual insight or strategy with rich metadata
- Playbook: A sectioned container for Bullets
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable
import json
import re
from pathlib import Path


class BulletTag(str, Enum):
    """Classification tag for a Bullet's utility."""
    HELPFUL = "helpful"
    HARMFUL = "harmful"
    NEUTRAL = "neutral"


class PlaybookSection(str, Enum):
    """Organizational sections within a Playbook."""
    STRATEGIES = "strategies"
    CODE_SNIPPETS = "code_snippets"
    PITFALLS = "pitfalls"
    APIS = "apis"


@dataclass
class BulletMetadata:
    """Tracking metadata for a Bullet."""
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None
    usage_count: int = 0
    helpful_count: int = 0
    harmful_count: int = 0
    source_task_id: Optional[str] = None


@dataclass
class Bullet:
    """
    An individual insight, strategy, or knowledge item in the Playbook.
    
    Attributes:
        id: Unique identifier for this bullet (format: "{section}-{number}", e.g., "strategies-001").
        content: The actual text of the strategy/insight.
        section: Which Playbook section this bullet belongs to.
        tag: Classification of the insight (helpful, harmful, neutral).
        version: Integer version tracking updates to this specific bullet.
        embedding: Vector representation for semantic retrieval (optional).
        metadata: Tracking metadata (usage counts, timestamps, etc.).
    """
    content: str
    section: PlaybookSection
    id: str = ""  # Will be set by Playbook.add_bullet() or from_dict()
    tag: BulletTag = BulletTag.NEUTRAL
    version: int = 1
    embedding: Optional[List[float]] = None
    metadata: BulletMetadata = field(default_factory=BulletMetadata)

    def to_dict(self) -> Dict:
        """Serialize Bullet to a dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "section": self.section.value,
            "tag": self.tag.value,
            "version": self.version,
            "embedding": self.embedding,
            "metadata": {
                "created_at": self.metadata.created_at.isoformat(),
                "last_accessed": self.metadata.last_accessed.isoformat() if self.metadata.last_accessed else None,
                "usage_count": self.metadata.usage_count,
                "helpful_count": self.metadata.helpful_count,
                "harmful_count": self.metadata.harmful_count,
                "source_task_id": self.metadata.source_task_id,
            }
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Bullet":
        """Deserialize a Bullet from a dictionary."""
        metadata_data = data.get("metadata", {})
        metadata = BulletMetadata(
            created_at=datetime.fromisoformat(metadata_data["created_at"]) if metadata_data.get("created_at") else datetime.now(),
            last_accessed=datetime.fromisoformat(metadata_data["last_accessed"]) if metadata_data.get("last_accessed") else None,
            usage_count=metadata_data.get("usage_count", 0),
            helpful_count=metadata_data.get("helpful_count", 0),
            harmful_count=metadata_data.get("harmful_count", 0),
            source_task_id=metadata_data.get("source_task_id"),
        )
        return cls(
            id=data["id"],
            content=data["content"],
            section=PlaybookSection(data["section"]),
            tag=BulletTag(data.get("tag", "neutral")),
            version=data.get("version", 1),
            embedding=data.get("embedding"),
            metadata=metadata,
        )

    def increment_usage(self) -> None:
        """Increment usage count and update last accessed time."""
        self.metadata.usage_count += 1
        self.metadata.last_accessed = datetime.now()

    def mark_helpful(self) -> None:
        """Mark this bullet as helpful for the current usage."""
        self.metadata.helpful_count += 1

    def mark_harmful(self) -> None:
        """Mark this bullet as harmful for the current usage."""
        self.metadata.harmful_count += 1

    def get_effectiveness_score(self) -> float:
        """
        Calculate an effectiveness score based on helpful/harmful counts.
        
        Returns:
            A score between -1.0 (always harmful) and 1.0 (always helpful).
            Returns 0.0 if never used.
        """
        total = self.metadata.helpful_count + self.metadata.harmful_count
        if total == 0:
            return 0.0
        return (self.metadata.helpful_count - self.metadata.harmful_count) / total

    def format_for_prompt(self, include_id: bool = True) -> str:
        """
        Format this bullet for inclusion in a prompt.
        
        Args:
            include_id: Whether to include the bullet ID prefix.
        
        Returns:
            Formatted string representation of the bullet.
        """
        if include_id:
            return f"[{self.id}]: {self.content}"
        return self.content


class Playbook:
    """
    A sectioned container for Bullets.
    
    The Playbook organizes insights into logical sections (strategies, code_snippets,
    pitfalls, apis) for more effective retrieval and context injection.
    """

    def __init__(self):
        """Initialize an empty Playbook with all sections."""
        self._sections: Dict[PlaybookSection, List[Bullet]] = {
            section: [] for section in PlaybookSection
        }
        self._bullet_index: Dict[str, Bullet] = {}  # id -> Bullet
        self._section_counters: Dict[PlaybookSection, int] = {
            section: 0 for section in PlaybookSection
        }  # Track next ID number per section
        self._frozen: bool = False

    def freeze(self) -> None:
        """Mark playbook as frozen (read-only for evaluation)."""
        self._frozen = True

    def is_frozen(self) -> bool:
        """Check if playbook is frozen."""
        return self._frozen

    def _raise_if_frozen(self, operation: str) -> None:
        if self._frozen:
            raise RuntimeError(f"Cannot {operation}: playbook is frozen for evaluation.")

    def _generate_bullet_id(self, section: PlaybookSection) -> str:
        """
        Generate a sequential ID for a bullet in the given section.
        
        Format: "{section}-{number}" (e.g., "strategies-001", "apis-012")
        """
        self._section_counters[section] += 1
        return f"{section.value}-{self._section_counters[section]:03d}"

    def add_bullet(self, bullet: Bullet) -> None:
        """
        Add a new bullet to the appropriate section.
        
        If the bullet doesn't have an ID, one will be generated.
        
        Args:
            bullet: The Bullet to add.
        """
        self._raise_if_frozen("add bullet")
        
        # Generate ID if not already set
        if not bullet.id:
            bullet.id = self._generate_bullet_id(bullet.section)
        else:
            # Update counter if loading existing bullet with higher number
            self._update_counter_from_id(bullet.id, bullet.section)
        
        self._sections[bullet.section].append(bullet)
        self._bullet_index[bullet.id] = bullet

    def _update_counter_from_id(self, bullet_id: str, section: PlaybookSection) -> None:
        """
        Update section counter based on existing bullet ID to avoid collisions.
        """
        try:
            # Parse ID like "strategies-001"
            parts = bullet_id.rsplit("-", 1)
            if len(parts) == 2:
                num = int(parts[1])
                if num >= self._section_counters[section]:
                    self._section_counters[section] = num
        except (ValueError, IndexError):
            pass  # Ignore unparseable IDs (e.g., old UUID format)

    def get_bullet(self, bullet_id: str) -> Optional[Bullet]:
        """
        Get a bullet by its ID.
        
        Args:
            bullet_id: The string ID of the bullet to retrieve (e.g., "strategies-001").
        
        Returns:
            The Bullet if found, None otherwise.
        """
        return self._bullet_index.get(bullet_id)

    def update_bullet(self, bullet_id: str, new_content: str) -> bool:
        """
        Update the content of an existing bullet and increment its version.
        
        Args:
            bullet_id: The string ID of the bullet to update.
            new_content: The new content for the bullet.
        
        Returns:
            True if the bullet was found and updated, False otherwise.
        """
        self._raise_if_frozen("update bullet")
        bullet = self._bullet_index.get(bullet_id)
        if bullet:
            bullet.content = new_content
            bullet.version += 1
            return True
        return False

    def remove_bullet(self, bullet_id: str) -> bool:
        """
        Remove a bullet from the Playbook.
        
        Args:
            bullet_id: The string ID of the bullet to remove.
        
        Returns:
            True if the bullet was found and removed, False otherwise.
        """
        self._raise_if_frozen("remove bullet")
        bullet = self._bullet_index.pop(bullet_id, None)
        if bullet:
            self._sections[bullet.section].remove(bullet)
            return True
        return False

    def get_section(self, section: PlaybookSection) -> List[Bullet]:
        """
        Get all bullets in a specific section.
        
        Args:
            section: The section to retrieve.
        
        Returns:
            List of Bullets in that section.
        """
        return self._sections[section]

    def get_all_bullets(self) -> List[Bullet]:
        """
        Get all bullets across all sections.
        
        Returns:
            List of all Bullets in the Playbook.
        """
        all_bullets = []
        for section_bullets in self._sections.values():
            all_bullets.extend(section_bullets)
        return all_bullets

    def get_bullet_count(self) -> int:
        """Get the total number of bullets in the Playbook."""
        return len(self._bullet_index)

    def get_section_counts(self) -> Dict[str, int]:
        """Get the count of bullets per section."""
        return {
            section.value: len(bullets)
            for section, bullets in self._sections.items()
        }

    def format_for_prompt(
        self,
        sections: Optional[List[PlaybookSection]] = None,
        max_per_section: Optional[int] = None,
        sort_by_effectiveness: bool = True,
        max_harmful_count: int = 1,
    ) -> str:
        """
        Format the Playbook (or specific sections) for inclusion in a prompt.
        
        Args:
            sections: Optional list of sections to include. If None, includes all.
            max_per_section: Optional max bullets per section. If None, includes all.
            sort_by_effectiveness: If True, sort bullets by effectiveness score (descending).
            max_harmful_count: Exclude bullets with harmful_count >= this value. Default 1.
        
        Returns:
            Formatted string representation of the Playbook.
        """
        if sections is None:
            sections = list(PlaybookSection)

        lines = []
        for section in sections:
            bullets = self._sections[section]
            if bullets:
                # Filter out highly harmful bullets
                bullets = [b for b in bullets if b.metadata.harmful_count < max_harmful_count]
                
                # Optionally sort by effectiveness
                if sort_by_effectiveness:
                    bullets = sorted(
                        bullets,
                        key=lambda b: b.get_effectiveness_score(),
                        reverse=True
                    )
                
                # Optionally limit count
                if max_per_section is not None:
                    bullets = bullets[:max_per_section]
                
                if bullets:  # Only add section if it has bullets after filtering
                    lines.append(f"## {section.value.replace('_', ' ').title()}")
                    for bullet in bullets:
                        lines.append(f"- {bullet.format_for_prompt()}")
                    lines.append("")  # Empty line between sections

        return "\n".join(lines) if lines else "No playbook entries yet."

    def save(self, filepath: Path) -> None:
        """
        Save the Playbook to a JSON file.
        
        Args:
            filepath: Path to save the JSON file.
        """
        data = {
            section.value: [bullet.to_dict() for bullet in bullets]
            for section, bullets in self._sections.items()
        }
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: Path) -> "Playbook":
        """
        Load a Playbook from a JSON file.
        
        Args:
            filepath: Path to the JSON file.
        
        Returns:
            A Playbook instance loaded from the file.
        """
        playbook = cls()
        if not filepath.exists():
            return playbook

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for section_name, bullets_data in data.items():
            section = PlaybookSection(section_name)
            for bullet_data in bullets_data:
                bullet = Bullet.from_dict(bullet_data)
                bullet.section = section  # Ensure section is set correctly
                playbook.add_bullet(bullet)

        return playbook

    def prune_low_value_bullets(self, threshold: float = -0.3, max_bullets: Optional[int] = None) -> List[Bullet]:
        """
        Remove bullets with low effectiveness scores.
        
        This implements the "Lazy Pruning" strategy where harmful bullets
        are removed when the Playbook grows too large.
        
        Args:
            threshold: Minimum effectiveness score to keep a bullet.
            max_bullets: If set, also prune to keep at most this many bullets.
        
        Returns:
            List of removed bullets.
        """
        removed = []
        
        # First pass: remove by threshold
        for section in PlaybookSection:
            bullets_to_remove = [
                b for b in self._sections[section]
                if b.get_effectiveness_score() < threshold
            ]
            for bullet in bullets_to_remove:
                self.remove_bullet(bullet.id)
                removed.append(bullet)

        # Second pass: if still over max, remove least effective
        if max_bullets and self.get_bullet_count() > max_bullets:
            all_bullets = sorted(
                self.get_all_bullets(),
                key=lambda b: b.get_effectiveness_score()
            )
            excess = self.get_bullet_count() - max_bullets
            for bullet in all_bullets[:excess]:
                self.remove_bullet(bullet.id)
                removed.append(bullet)

        return removed
