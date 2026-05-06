"""Benchmark-agnostic agent base abstractions."""

from preping.core.agents.base import BaseAgent, load_prompt_dict_from_file, merge_configs

__all__ = ["BaseAgent", "merge_configs", "load_prompt_dict_from_file"]
