"""Shared core interfaces for memory, environment, and task generation."""

from preping.core.env import BaseEnv, BaseEnvConfig, BaseLanguageBasedEnv
from preping.core.memory import MemoryManager
from preping.core.interfaces.benchmark import (
    BenchmarkAdapter,
    BenchmarkResult,
    BenchmarkTask,
    RunnerConfig,
)
from preping.core.interfaces.task_generation import GeneratedTask, TaskGeneratorEngine, TaskSource

LanguageEnvironment = BaseLanguageBasedEnv
MemoryStore = MemoryManager


__all__ = [
    "BenchmarkAdapter",
    "BenchmarkTask",
    "BenchmarkResult",
    "RunnerConfig",
    "BaseEnv",
    "BaseEnvConfig",
    "BaseLanguageBasedEnv",
    "LanguageEnvironment",
    "MemoryManager",
    "MemoryStore",
    "GeneratedTask",
    "TaskGeneratorEngine",
    "TaskSource",
]
