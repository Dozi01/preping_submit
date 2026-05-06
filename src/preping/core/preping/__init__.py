"""Task generation and orchestration module."""

import logging

from preping.core.preping.proposer import PrePingProposer
from preping.core.preping.manager import (
    PrePingManager,
    register_preping_factory,
    get_preping_factory,
    create_preping_for_benchmark,
    available_preping_factories,
)
from preping.core.preping.orchestrator import PrePingOrchestrator
from preping.core.preping.validation import (
    PrePingValidationGate,
    PrePingValidationResult,
)

__all__ = [
    "PrePingProposer",
    "PrePingManager",
    "PrePingOrchestrator",
    "PrePingValidationGate",
    "PrePingValidationResult",
    "register_preping_factory",
    "get_preping_factory",
    "create_preping_for_benchmark",
    "available_preping_factories",
]

logger = logging.getLogger(__name__)
