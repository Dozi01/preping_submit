# PrePing playbook memory module.
# Implements a self-evolving Playbook of strategies and insights

from ..interface import MemoryManager
from .curator import Curator, CuratorInput, CuratorOutput
from .manager import PrePingMemoryManager
from .prompts import (
    CURATOR_SYSTEM_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
)
from .reflector import Reflector, ReflectorInput, ReflectorOutput
from .retriever import PlaybookRetriever
from .types import Bullet, BulletMetadata, BulletTag, Playbook, PlaybookSection


__all__ = [
    # Types
    "Bullet",
    "BulletMetadata",
    "BulletTag",
    "Playbook",
    "PlaybookSection",
    # Manager
    "MemoryManager",
    "PrePingMemoryManager",
    # Reflector
    "Reflector",
    "ReflectorInput",
    "ReflectorOutput",
    # Curator
    "Curator",
    "CuratorInput",
    "CuratorOutput",
    # Retriever
    "PlaybookRetriever",
    # Prompts
    "REFLECTOR_SYSTEM_PROMPT",
    "CURATOR_SYSTEM_PROMPT",
]
