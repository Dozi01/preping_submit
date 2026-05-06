"""
Static Document Playbook Builder

Generates a Playbook from API documentation by converting API docs
into virtual trajectories and processing them through the standard
PrePingMemoryManager Reflector -> Curator pipeline.

This is used to build a documentation-only baseline playbook.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console

from preping.llm import LLMManager
from preping.appworld import Playbook
from preping.appworld import PrePingMemoryManager
from preping.appworld.task_generation import resolve_appworld_api_docs_dir
from preping.core.memory.playbook.prompts import (
    CURATOR_SYSTEM_PROMPT_NO_EXAMPLES,
    REFLECTOR_SYSTEM_PROMPT_NO_EXAMPLES,
)
from experiments.appworld.experiment_utils import build_experiment_paths, setup_logging


logger = logging.getLogger(__name__)
console = Console()


API_EXPLORATION_INSTRUCTIONS = """Here are three key APIs that you need to know to get more information

# To get a list of apps that are available to you.

```python
print(apis.api_docs.show_app_descriptions())
```

# To get the list of apis under any app listed above, e.g. spotify

```python
print(apis.api_docs.show_api_descriptions(app_name='spotify'))
```

# To get the specification of a particular api, e.g. spotify app's login api

```python
print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))
```

Each code execution will produce an output that you can use in subsequent calls. Using these APIs, you can now generate code, that I will execute, to solve the task. 

**Key instructions**:
(1) Make sure to end code blocks with ``` followed by a newline(\n).

(2) Remember you can use the variables in your code in subsequent code blocks.

(3) Remember that the email addresses, access tokens and variables (e.g. spotify_password) in the example above are not valid anymore.

(4) You can use the "supervisor" app to get information about my accounts and use the "phone" app to get information about friends and family.

(5) Always look at API specifications (using apis.api_docs.show_api_doc) before calling an API.

(6) Write small chunks of code and only one chunk of code in every step. Make sure everything is working correctly before making any irreversible change.

(7) Many APIs return items in "pages". Make sure to run through all the pages by looping over `page_index`.

(8) Once you have completed the task, make sure to call apis.supervisor.complete_task(). If the task asked for some information, return it as the answer argument, i.e. call apis.supervisor.complete_task(answer=<answer>). Many tasks do not require an answer, so in those cases, just call apis.supervisor.complete_task() i.e. do not pass any argument.
"""


class StaticDocPlaybookBuilder:
    """
    Generates Playbook from API documentation using the standard
    PrePingMemoryManager pipeline.
    
    Converts API documentation into virtual trajectories that simulate
    an agent exploring the API, then processes these through the
    Reflector -> Curator pipeline.
    """
    
    def __init__(
        self,
        api_docs_dir: Optional[str] = None,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 3000,
        use_thinking: bool = False,
        num_iterations: int = 100,
        apps_per_iteration: int = 3,
        verbose: bool = False,
        seed: int | None = None,
    ):
        self.api_docs_dir = resolve_appworld_api_docs_dir(api_docs_dir)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_thinking = use_thinking
        self.num_iterations = num_iterations
        self.apps_per_iteration = apps_per_iteration
        self.verbose = verbose
        self.seed = seed
        self._llm = None
        if self.seed is not None:
            random.seed(self.seed)

    @property
    def llm(self):
        """Lazy initialization of LLM."""
        if self._llm is None:
            self._llm = LLMManager.create_llm(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                use_thinking=self.use_thinking,
            )
        return self._llm
    
    def _load_api_docs(self) -> Dict[str, Dict[str, Any]]:
        """Load all API documentation from JSON files."""
        logger.info(f"Loading API documentation from {self.api_docs_dir}...")
        
        if not self.api_docs_dir.exists():
            raise FileNotFoundError(f"API docs directory not found: {self.api_docs_dir}")
        
        all_apis = {}
        json_files = sorted(self.api_docs_dir.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.api_docs_dir}")
        
        logger.info(f"Found {len(json_files)} API doc files")
        
        for json_file in json_files:
            app_name = json_file.stem
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    api_data = json.load(f)
                all_apis[app_name] = api_data
                logger.info(f"Loaded {len(api_data)} APIs from {app_name}")
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        
        return all_apis
    
    def _format_api_doc_as_output(self, api_name: str, api_info: dict) -> str:
        """Format a single API documentation as exploration output."""
        parts = [f"API: {api_info.get('app_name', 'unknown')}.{api_name}"]
        parts.append(f"Path: {api_info.get('path', 'N/A')}")
        parts.append(f"Method: {api_info.get('method', 'N/A')}")
        parts.append(f"Description: {api_info.get('description', 'N/A')}")
        
        params = api_info.get('parameters', [])
        if params:
            parts.append("Parameters:")
            for param in params:
                required = "required" if param.get('required') else "optional"
                default_val = param.get('default')
                default_str = f", default={default_val}" if default_val is not None else ""
                constraints = param.get('constraints', [])
                constraint_str = f", constraints={constraints}" if constraints else ""
                parts.append(
                    f"  - {param.get('name')} ({param.get('type')}, {required}{default_str}{constraint_str}): "
                    f"{param.get('description', '')}"
                )
        
        response_schemas = api_info.get('response_schemas', {})
        if response_schemas:
            parts.append("Response Schema:")
            if 'success' in response_schemas:
                parts.append(f"  Success: {json.dumps(response_schemas['success'])}")
            if 'failure' in response_schemas:
                parts.append(f"  Failure: {json.dumps(response_schemas['failure'])}")
        
        return "\n".join(parts)
    
    def _create_virtual_trajectory_for_app(
        self, 
        app_name: str, 
        apis: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Create a virtual trajectory simulating API documentation exploration.
        
        Each step simulates an agent action to explore an API and receives
        the documentation as output.
        """
        trajectory = []
        
        # Step 1: Discover available APIs for this app
        api_items = list(apis.items())
        random.shuffle(api_items)
        
        api_names = [name for name, _ in api_items]
        trajectory.append({
            "action": f"apis.api_docs.show_api_descriptions(app_name='{app_name}')",
            "output": f"Available APIs for {app_name}: {', '.join(api_names)}"
        })
        
        # Step 2: Get detailed documentation for each API
        for api_name, api_info in api_items:
            trajectory.append({
                "action": f"apis.api_docs.show_api_doc(app_name='{app_name}', api_name='{api_name}')",
                "output": self._format_api_doc_as_output(api_name, api_info)
            })
        
        return trajectory
    
    def _create_sampled_trajectory(
        self,
        all_apis: Dict[str, Dict[str, Any]],
        num_apps: int = 2
    ) -> List[Dict[str, str]]:
        """
        Create a trajectory by sampling N apps and shuffling their APIs.
        
        This provides diversity across iterations by:
        1. Randomly selecting a subset of apps
        2. Shuffling API order within each app
        3. Interleaving apps for cross-app exploration patterns
        """
        app_names = list(all_apis.keys())
        
        # Sample N apps (or all if fewer available)
        sampled_apps = random.sample(app_names, min(num_apps, len(app_names)))
        random.shuffle(sampled_apps)  # Shuffle app order too
        
        trajectory = []
        doc_exploration_steps = []
        
        for app_name in sampled_apps:
            apis = all_apis[app_name]
            
            # Shuffle API order within this app
            api_items = list(apis.items())
            random.shuffle(api_items)
            
            # Add discovery step
            api_names_list = [name for name, _ in api_items]
            trajectory.append({
                "action": f"apis.api_docs.show_api_descriptions(app_name='{app_name}')",
                "output": f"Available APIs for {app_name}: {', '.join(api_names_list)}"
            })
            
            # Add detailed docs for each API (to be shuffled later)
            for api_name, api_info in api_items:
                doc_exploration_steps.append({
                    "action": f"apis.api_docs.show_api_doc(app_name='{app_name}', api_name='{api_name}')",
                    "output": self._format_api_doc_as_output(api_name, api_info)
                })
        
        # Shuffle detailed documentation steps to interleave exploration
        random.shuffle(doc_exploration_steps)
        trajectory.extend(doc_exploration_steps)
        
        return trajectory, sampled_apps
    
    def build(
        self,
        *,
        output_root_dir: str | Path,
        playbook_path: str | Path,
        tag: str = "direct_memory",
        input_config_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        """
        Build playbook from API documentation using PrePingMemoryManager pipeline.
        
        Args:
            output_root_dir: Output root directory for logs, summary, and debug artifacts
            playbook_path: Explicit playbook JSON output path
            tag: Run tag used when output_path is omitted
            input_config_path: Optional saved input config path
            
        Returns:
            Run summary dictionary
        """
        started_at = time.time()
        output_root = Path(output_root_dir).resolve()
        output_file = Path(playbook_path).resolve()
        memory_debug_root = output_root / "memory_debug"

        output_root.mkdir(parents=True, exist_ok=True)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("Starting Static Doc Playbook Builder")
        logger.info("=" * 60)
        logger.info(f"Output root: {output_root}")
        logger.info(f"Output path: {output_file}")
        logger.info(f"API docs directory: {self.api_docs_dir}")
        logger.info(f"Model: {self.model_name}, Temperature: {self.temperature}")
        
        # Load all API documentation
        logger.info("-" * 40)
        logger.info("Step 1: Loading API documentation...")
        all_apis = self._load_api_docs()
        
        total_apis = sum(len(apis) for apis in all_apis.values())
        logger.info(f"Loaded {len(all_apis)} apps with {total_apis} total APIs")
        for app_name, apis in all_apis.items():
            logger.info(f"  - {app_name}: {len(apis)} APIs")
        
        # Create PrePing Memory Manager
        logger.info("-" * 40)
        logger.info("Step 2: Initializing PrePing Memory Manager...")
        preping_manager = PrePingMemoryManager(
            model_name=self.model_name,
            playbook_path=output_file,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            use_thinking=self.use_thinking,
            include_task_context=False,  # No specific task context for static doc generation
            reflector_system_prompt=REFLECTOR_SYSTEM_PROMPT_NO_EXAMPLES,
            curator_system_prompt=CURATOR_SYSTEM_PROMPT_NO_EXAMPLES,
        )
        preping_manager.playbook.save(output_file)
        logger.info("PrePing Memory Manager initialized")
        
        # Log sampling configuration
        logger.info("-" * 40)
        logger.info(f"Step 3: Will sample {self.apps_per_iteration} apps per iteration (from {len(all_apis)} total apps)...")
        
        # Process through Reflector -> Curator pipeline multiple times
        logger.info(f"\nStep 4: Running Reflector -> Curator pipeline ({self.num_iterations} iterations)...\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("[cyan]Building playbook...", total=self.num_iterations)
            completed_iterations = 0
            
            for iteration in range(1, self.num_iterations + 1):
                try:
                    # Sample different apps each iteration for diversity
                    trajectory, sampled_apps = self._create_sampled_trajectory(
                        all_apis, 
                        num_apps=self.apps_per_iteration
                    )
                    
                    task_description = (
                        f"Explore and understand the API documentation for apps: {', '.join(sampled_apps)}. "
                        f"Learn about available APIs, their parameters, and response formats.\n\n"
                        f"{API_EXPLORATION_INSTRUCTIONS}"
                    )
                    
                    result = preping_manager.process_episode(
                        task_description=task_description,
                        trajectory=trajectory,
                        result="success",  # Documentation exploration is always "successful"
                        task_id=f"static_doc_iter_{iteration}",
                        trajectory_format="action_output",
                        debug_output_dir=str(memory_debug_root / f"iter_{iteration:03d}"),
                    )
                    
                    bullets_modified = result.get('bullets_modified', [])
                    stats = preping_manager.get_stats()
                    progress.update(
                        task, 
                        advance=1, 
                        description=f"[cyan]Iter {iteration} ({','.join(sampled_apps)}): +{len(bullets_modified)} bullets, total: {stats.get('total_bullets', 0)}"
                    )
                    completed_iterations += 1
                    
                except Exception as e:
                    logger.info(f"[red]  Error in iteration {iteration}: {e}[/red]")
                    progress.update(task, advance=1)
                    continue
        
        # Summary
        stats = preping_manager.get_stats()
        
        # Get LLM token usage and cost
        token_stats = preping_manager.llm_client.get_token_usage_stats()
        cost_breakdown = preping_manager.llm_client.get_cost_breakdown()
        
        logger.info("\n" + "=" * 60)
        logger.info("Static Doc Playbook Building Complete!")
        logger.info("=" * 60)
        logger.info(f"Total apps available: {len(all_apis)}")
        logger.info(f"Apps sampled per iteration: {self.apps_per_iteration}")
        logger.info(f"Total APIs available: {total_apis}")
        logger.info(f"Total bullets in playbook: {stats.get('total_bullets', 0)}")
        logger.info(f"Bullets by section:")
        for section, count in stats.get('section_counts', {}).items():
            logger.info(f"  - {section}: {count}")
        logger.info(f"Playbook saved to: {output_file}")
        
        # Print token usage and cost summary
        logger.info("\n" + "-" * 60)
        logger.info("TOKEN USAGE AND COST SUMMARY")
        logger.info("-" * 60)
        logger.info(f"Total Requests: {token_stats.get('total_requests', 0):,}")
        logger.info(f"Total Input Tokens: {token_stats.get('total_input_tokens', 0):,}")
        logger.info(f"Total Cached Input Tokens: {token_stats.get('total_cached_input_tokens', 0):,}")
        logger.info(f"Total Reasoning Tokens: {token_stats.get('total_reasoning_tokens', 0):,}")
        logger.info(f"Total Output Tokens: {token_stats.get('total_output_tokens', 0):,}")
        logger.info(f"Total Tokens: {token_stats.get('total_tokens', 0):,}")
        logger.info(f"TOTAL COST: ${cost_breakdown.get('total_cost', 0):.6f}")
        logger.info("=" * 60)
        
        # Save usage summary to JSON alongside the playbook
        usage_summary = {
            "benchmark": "appworld",
            "run_type": "direct_memory",
            "output_root_dir": str(output_root),
            "playbook_path": str(output_file),
            "memory_debug_dir": str(memory_debug_root),
            "input_config_path": str(input_config_path) if input_config_path else None,
            "playbook_stats": stats,
            "token_usage": token_stats,
            "cost_breakdown": cost_breakdown,
            "num_iterations_requested": self.num_iterations,
            "num_iterations_completed": completed_iterations,
            "apps_per_iteration": self.apps_per_iteration,
            "api_docs_dir": str(self.api_docs_dir),
            "memory_model_name": self.model_name,
            "temperature": self.temperature,
            "tag": tag,
            "seed": self.seed,
            "total_time_seconds": time.time() - started_at,
            "run_timestamp": time.time(),
            "config": {
                "model_name": self.model_name,
                "temperature": self.temperature,
                "num_iterations": self.num_iterations,
                "apps_per_iteration": self.apps_per_iteration,
                "api_docs_dir": str(self.api_docs_dir),
                "tag": tag,
                "seed": self.seed,
            },
        }
        usage_file = output_file.with_name(output_file.stem + '_usage.json')
        with open(usage_file, 'w', encoding='utf-8') as f:
            json.dump(usage_summary, f, indent=2)
        logger.info(f"Usage summary saved to: {usage_file}")

        summary_file = output_root / "experiment_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(usage_summary, f, indent=2)
        logger.info(f"Experiment summary saved to: {summary_file}")
        
        return usage_summary


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate playbook from API documentation using PrePing pipeline"
    )
    parser.add_argument(
        "--output_path", type=str, 
        default=None,
        help="Optional explicit output path for the playbook JSON file"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./outputs",
        help="Base output directory when --output_path is omitted"
    )
    parser.add_argument(
        "--tag", type=str,
        default="direct_memory",
        help="Run tag when --output_path is omitted"
    )
    parser.add_argument(
        "--model_name", type=str, default="deepseek/deepseek-chat",
        help="LLM model to use for generation"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="LLM temperature"
    )
    parser.add_argument(
        "--num_iterations", type=int, default=100,
        help="Number of iterations to run the pipeline (default: 100)"
    )
    parser.add_argument(
        "--apps_per_iteration", type=int, default=2,
        help="Number of apps to sample per iteration for diversity (default: 2)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--api_docs_dir", type=str, default=None,
        help="Directory containing API documentation JSON files"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Optional random seed for reproducible app/API sampling"
    )
    
    args = parser.parse_args()

    if args.output_path:
        playbook_path = Path(args.output_path).resolve()
        output_root_dir = playbook_path.parent
    else:
        paths = build_experiment_paths(
            output_dir=args.output_dir,
            model_name=args.model_name,
            tag=args.tag,
            split=None,
        )
        output_root_dir = Path(paths["output_root_dir"]).resolve()
        playbook_path = output_root_dir / "playbook.json"

    setup_logging(output_dir=str(output_root_dir), debug_mode=args.verbose)
    api_docs_dir = resolve_appworld_api_docs_dir(args.api_docs_dir)

    input_config = {
        "output_root_dir": str(output_root_dir),
        "playbook_path": str(playbook_path),
        "output_dir": args.output_dir,
        "tag": args.tag,
        "model_name": args.model_name,
        "temperature": args.temperature,
        "num_iterations": args.num_iterations,
        "apps_per_iteration": args.apps_per_iteration,
        "seed": args.seed,
        "verbose": args.verbose,
        "api_docs_dir": str(api_docs_dir.resolve()),
    }
    input_config_path = output_root_dir / "input_config.json"
    input_config_path.write_text(json.dumps(input_config, indent=2), encoding="utf-8")

    builder = StaticDocPlaybookBuilder(
        api_docs_dir=str(api_docs_dir),
        model_name=args.model_name,
        temperature=args.temperature,
        num_iterations=args.num_iterations,
        apps_per_iteration=args.apps_per_iteration,
        verbose=args.verbose,
        seed=args.seed,
    )
    
    summary = builder.build(
        output_root_dir=output_root_dir,
        playbook_path=playbook_path,
        tag=args.tag,
        input_config_path=input_config_path,
    )
    print(f"Playbook generated: {summary['playbook_path']}")
    print(f"Experiment summary saved to: {Path(summary['output_root_dir']) / 'experiment_summary.json'}")


if __name__ == "__main__":
    main()
