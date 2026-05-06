"""AppWorld-specific PrePing adapter."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from preping.appworld.prompts import (
    APPWORLD_COMPLEXITY_LEVEL_DESCRIPTIONS,
    APPWORLD_ENVIRONMENT_EXTRACTION_PROMPT,
    APPWORLD_TASK_GENERATION_PROMPT,
)
from preping.core.preping import PrePingProposer


logger = logging.getLogger(__name__)

REPO_APPWORLD_API_DOCS_DIR = (
    Path(__file__).resolve().parents[3] / "experiments" / "appworld" / "data" / "api_docs" / "standard"
)
APPWORLD_TRAJECTORY_FILE_PATTERN = "**/appworld_trajectory.json"


def resolve_appworld_api_docs_dir(api_docs_dir: Optional[str] = None) -> Path:
    """Resolve the AppWorld API documentation directory with a clear failure mode."""
    candidates: List[Path] = []
    if api_docs_dir:
        candidates.append(Path(api_docs_dir))

    env_docs_dir = os.getenv("APPWORLD_API_DOCS_DIR")
    if env_docs_dir:
        candidates.append(Path(env_docs_dir))

    try:
        from appworld.common.path_store import path_store

        candidates.append(Path(path_store.data) / "api_docs" / "standard")
    except Exception:
        pass

    candidates.append(REPO_APPWORLD_API_DOCS_DIR)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "AppWorld API docs directory not found. "
        "Set APPWORLD_API_DOCS_DIR or pass api_docs_dir explicitly.\n"
        f"Tried:\n{tried}"
    )


def build_appworld_prompt_context(
    target_apps: Optional[List[str]] = None,
    *,
    api_docs_dir: Optional[str] = None,
) -> str:
    """Build AppWorld API-doc prompt context for PrePing generation."""
    docs_dir = resolve_appworld_api_docs_dir(api_docs_dir)
    excluded = {"api_docs", "supervisor"}
    lines: List[str] = []

    for json_file in sorted(docs_dir.glob("*.json")):
        app_name = json_file.stem
        if app_name in excluded:
            continue
        if target_apps and app_name not in target_apps:
            continue

        try:
            api_map = json.loads(json_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Failed to load AppWorld API doc file %s: %s", json_file, exc)
            raise RuntimeError(f"Failed to load AppWorld API doc file: {json_file}") from exc
        lines.append(f"# App: {app_name}\n")
        for api_name, api_info in api_map.items():
            lines.append(f"## {app_name}.{api_name}")
            lines.append(f"Description: {api_info.get('description', 'N/A')}")

            params = api_info.get("parameters", [])
            if params:
                lines.append("- Parameters:")
                for param in params:
                    if param.get("name") == "access_token":
                        continue
                    required = "required" if param.get("required") else "optional"
                    param_line = (
                        f"  - {param.get('name')} ({param.get('type')}, {required}): "
                        f"{param.get('description', '')}"
                    )
                    constraints = param.get("constraints", [])
                    if constraints:
                        param_line += f" [Constraints: {', '.join(constraints)}]"
                    lines.append(param_line)

            response_schemas = api_info.get("response_schemas", {})
            success_schema = response_schemas.get("success", {}) if isinstance(response_schemas, dict) else {}
            if success_schema:
                lines.append(f"- Returns: {json.dumps(success_schema, ensure_ascii=False)}")

            lines.append("")

    return "\n".join(lines)


@dataclass
class AppWorldPrePingAdapter:
    """Typed adapter around the PrePing proposer for AppWorld."""

    model_name: str = "deepseek/deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 3000
    use_thinking: bool = False
    verbose: bool = False
    complexity_level: int = 1
    api_docs_dir: Optional[str] = None

    def build_generator(self) -> PrePingProposer:
        """Create the underlying PrePing proposer with AppWorld defaults."""
        return PrePingProposer(
            trajectory_file_pattern=APPWORLD_TRAJECTORY_FILE_PATTERN,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            use_thinking=self.use_thinking,
            verbose=self.verbose,
            complexity_level=self.complexity_level,
            task_generation_prompt=APPWORLD_TASK_GENERATION_PROMPT,
            environment_extraction_prompt=APPWORLD_ENVIRONMENT_EXTRACTION_PROMPT,
            complexity_level_descriptions=APPWORLD_COMPLEXITY_LEVEL_DESCRIPTIONS,
            prompt_context_provider=lambda target_apps: build_appworld_prompt_context(
                target_apps,
                api_docs_dir=self.api_docs_dir,
            ),
        )

    def generate_from_trajectory_files(self, trajectory_paths: List[str], num_tasks: int = 1) -> List[Dict[str, Any]]:
        """Generate synthetic tasks from one or more trajectory JSON files."""
        generator = self.build_generator()
        merged_env_info: Dict[str, Any] = {}
        for trajectory_path in trajectory_paths:
            env_info = generator.extract_environment_info(trajectory_path)
            for app_name, entities in env_info.items():
                app_bucket = merged_env_info.setdefault(app_name, {})
                for entity_name, values in entities.items():
                    if isinstance(values, list):
                        app_bucket.setdefault(entity_name, [])
                        app_bucket[entity_name].extend(values)
                    elif entity_name not in app_bucket:
                        app_bucket[entity_name] = values

        return generator.generate_tasks(environment_info=merged_env_info, num_tasks=num_tasks)


AppWorldTaskGenerationAdapter = AppWorldPrePingAdapter
