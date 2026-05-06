"""AppWorld-specific adapters and entrypoints.

Exports are loaded lazily to avoid importing heavy benchmark dependencies
unless they are actually requested.
"""

__all__ = [
    "AppWorldBenchmark",
    "AppWorldAgent",
    "AppWorldAgentConfig",
    "create_appworld_agent",
    "create_appworld_agent_config",
    "BaseExplorerAgent",
    "RandomExplorerAgent",
    "GuidedExplorerAgent",
    "TrajectoryMemory",
    "AppWorldEnv",
    "AppWorldEnvConfig",
    "AppWorldMemoryAdapter",
    "create_appworld_memory_adapter",
    "PrePingMemoryManager",
    "Playbook",
    "AppWorldPrePingAdapter",
    "AppWorldTaskGenerationAdapter",
    "AppWorldTaskSource",
    "create_appworld_preping",
    "create_appworld_task_manager",
    "run_single_task",
    "run_single_task_worker",
]


def __getattr__(name: str):
    if name == "AppWorldBenchmark":
        from preping.appworld.benchmark_adapter import AppWorldBenchmark

        return AppWorldBenchmark

    if name in {"AppWorldAgent", "AppWorldAgentConfig", "create_appworld_agent", "create_appworld_agent_config"}:
        from preping.appworld.agent import (
            AppWorldAgent,
            AppWorldAgentConfig,
            create_appworld_agent,
            create_appworld_agent_config,
        )

        mapping = {
            "AppWorldAgent": AppWorldAgent,
            "AppWorldAgentConfig": AppWorldAgentConfig,
            "create_appworld_agent": create_appworld_agent,
            "create_appworld_agent_config": create_appworld_agent_config,
        }
        return mapping[name]

    if name in {"BaseExplorerAgent", "RandomExplorerAgent", "GuidedExplorerAgent", "TrajectoryMemory"}:
        from preping.appworld.exploration import (
            BaseExplorerAgent,
            GuidedExplorerAgent,
            RandomExplorerAgent,
            TrajectoryMemory,
        )

        mapping = {
            "BaseExplorerAgent": BaseExplorerAgent,
            "RandomExplorerAgent": RandomExplorerAgent,
            "GuidedExplorerAgent": GuidedExplorerAgent,
            "TrajectoryMemory": TrajectoryMemory,
        }
        return mapping[name]

    if name in {"AppWorldEnv", "AppWorldEnvConfig"}:
        from preping.appworld.env import AppWorldEnv, AppWorldEnvConfig

        return AppWorldEnv if name == "AppWorldEnv" else AppWorldEnvConfig

    if name in {"AppWorldMemoryAdapter", "create_appworld_memory_adapter"}:
        from preping.appworld.memory_adapter import AppWorldMemoryAdapter, create_appworld_memory_adapter

        return AppWorldMemoryAdapter if name == "AppWorldMemoryAdapter" else create_appworld_memory_adapter

    if name in {"PrePingMemoryManager", "Playbook"}:
        from preping.appworld.memory import PrePingMemoryManager, Playbook

        return PrePingMemoryManager if name == "PrePingMemoryManager" else Playbook

    if name == "AppWorldTaskSource":
        from preping.appworld.task_source import AppWorldTaskSource

        return AppWorldTaskSource

    if name in {"AppWorldPrePingAdapter", "AppWorldTaskGenerationAdapter"}:
        from preping.appworld.task_generation import AppWorldPrePingAdapter, AppWorldTaskGenerationAdapter

        return AppWorldPrePingAdapter if name == "AppWorldPrePingAdapter" else AppWorldTaskGenerationAdapter

    if name in {"create_appworld_preping", "create_appworld_task_manager"}:
        from preping.appworld.task_manager_factory import create_appworld_preping, create_appworld_task_manager

        return create_appworld_preping if name == "create_appworld_preping" else create_appworld_task_manager

    if name == "run_single_task":
        from preping.appworld.execution import run_single_task

        return run_single_task

    if name == "run_single_task_worker":
        from preping.appworld.execution import run_single_task_worker

        return run_single_task_worker

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
