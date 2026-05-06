# PrePing

This repository contains the core PrePing implementation and AppWorld experiment entry points used for the paper.

PrePing runs a cycle of:

1. generating proposer tasks from prior trajectories and environment summaries,
2. executing the generated tasks in AppWorld,
3. validating the resulting trajectories,
4. updating the playbook memory used by the next cycle.

## Layout

- `src/preping/core/preping/`: task generation, validation, cycle orchestration, and proposer memory.
- `src/preping/core/memory/playbook/`: PrePing playbook memory induction, retrieval, reflection, and curation.
- `src/preping/appworld/`: AppWorld adapters for agents, environments, task execution, and task generation.
- `experiments/appworld/`: runnable AppWorld experiment entry points and prompt templates.
- `servers/`: optional local embedding server helper.

## Entry Points

Run a PrePing trajectory cycle:

```bash
python experiments/appworld/run_trajectory_cycle.py --max_cycles 3 --tasks_per_cycle 5 --build_playbook
```

Run AppWorld evaluation with an existing playbook:

```bash
python experiments/appworld/run_all_parallel.py --memory_type playbook --memory_path outputs/playbook.json
```

The code intentionally excludes local cluster paths, private experiment outputs, and unrelated benchmark-specific scripts.
Set `APPWORLD_API_DOCS_DIR` if AppWorld API documentation is not available from the installed benchmark package.
