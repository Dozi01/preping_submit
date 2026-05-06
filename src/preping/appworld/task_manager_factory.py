"""AppWorld PrePing factory registration."""

from __future__ import annotations

from functools import partial
from typing import Any

from preping.appworld.prompts import (
    APPWORLD_COMPLEXITY_LEVEL_DESCRIPTIONS,
    APPWORLD_DATASET_EXAMPLES_SECTION,
    APPWORLD_ENVIRONMENT_EXTRACTION_PROMPT,
    APPWORLD_GROUNDED_ENVIRONMENT_SUMMARY_PROMPT,
    APPWORLD_TASK_GENERATION_PROMPT,
    APPWORLD_VALIDATION_PROMPT,
)
from preping.appworld.task_generation import (
    APPWORLD_TRAJECTORY_FILE_PATTERN,
    build_appworld_prompt_context,
)
from preping.core.preping import PrePingManager, register_preping_factory


@register_preping_factory("appworld")
def create_appworld_preping(*, config: Any) -> PrePingManager:
    """Create an AppWorld-configured PrePing manager from experiment config."""
    tm_config = config.task_manager_config
    return PrePingManager(
        model_name=tm_config.model_name,
        temperature=tm_config.temperature,
        use_thinking=tm_config.use_thinking,
        verbose=config.verbose,
        use_validator=tm_config.use_validator,
        min_feasibility_score=tm_config.validator_min_feasibility_score,
        min_task_completion_score=tm_config.validator_min_task_completion_score,
        trajectory_file_pattern=APPWORLD_TRAJECTORY_FILE_PATTERN,
        task_generation_prompt=APPWORLD_TASK_GENERATION_PROMPT,
        environment_extraction_prompt=APPWORLD_ENVIRONMENT_EXTRACTION_PROMPT,
        grounded_environment_summary_prompt=APPWORLD_GROUNDED_ENVIRONMENT_SUMMARY_PROMPT,
        complexity_level_descriptions=APPWORLD_COMPLEXITY_LEVEL_DESCRIPTIONS,
        validation_prompt=APPWORLD_VALIDATION_PROMPT,
        embedding_model=tm_config.embedding_model,
        embedding_base_url=tm_config.embedding_base_url,
        semantic_oversample_multiplier=tm_config.semantic_oversample_multiplier,
        include_dataset_examples=tm_config.include_dataset_examples,
        prompt_context_provider=partial(
            build_appworld_prompt_context,
            api_docs_dir=getattr(tm_config, "api_docs_dir", None),
        ),
        dataset_examples_section=APPWORLD_DATASET_EXAMPLES_SECTION,
    )


create_appworld_task_manager = create_appworld_preping
