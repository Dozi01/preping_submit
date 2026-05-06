# AppWorld experiment helpers for the PrePing paper code.

PYTHON ?= python
export PYTHONPATH := ../../src:../..:$(PYTHONPATH)

MODEL ?= deepseek/deepseek-chat
AGENT_MODEL ?= $(MODEL)
MEMORY_MODEL ?= $(MODEL)
TASK_MANAGER_MODEL ?= $(MODEL)
MODEL_ARGS := --global_model_name $(MODEL) --agent_model_name $(AGENT_MODEL) --memory_model_name $(MEMORY_MODEL) --task_manager_model_name $(TASK_MANAGER_MODEL)

SPLIT ?= test_normal
PROMPT ?= prompts/instruction.jinja
WORKERS ?= 20
TAG ?= preping
MODE ?= offline

MAX_CYCLES ?= 3
TASKS_PER_CYCLE ?= 5
RUNS_PER_TASK ?= 1
TASK_EMBEDDING_MODEL ?= Qwen/Qwen3-Embedding-0.6B
TASK_EMBEDDING_BASE_URL ?= http://localhost:8201/v1

.PHONY: help run run-pb cycle cycle-eval adapt direct-memory filter-playbook

help:
	@echo "PrePing AppWorld commands"
	@echo "  make run                         Run the base AppWorld agent"
	@echo "  make run-pb PB_FILE=path.json    Evaluate with a playbook"
	@echo "  make cycle                       Run PrePing construction cycles"
	@echo "  make cycle-eval                  Run cycles and evaluate the final playbook"
	@echo "  make adapt MODE=offline|online   Run task-informed sequential adaptation"
	@echo "  make direct-memory               Build a static documentation playbook"
	@echo "  make filter-playbook PB_FILE=... Remove answer-format bullets"

run:
	$(PYTHON) run_all_parallel.py \
		--split $(SPLIT) \
		$(MODEL_ARGS) \
		--tag $(TAG) \
		--prompt_file $(PROMPT) \
		--num_workers $(WORKERS)

run-pb:
ifndef PB_FILE
	$(error PB_FILE is required. Usage: make run-pb PB_FILE=outputs/playbook.json)
endif
	$(PYTHON) run_all_parallel.py \
		--split $(SPLIT) \
		$(MODEL_ARGS) \
		--tag $(TAG) \
		--prompt_file $(PROMPT) \
		--memory_type playbook \
		--memory_path $(PB_FILE) \
		--num_workers $(WORKERS)

cycle:
	$(PYTHON) run_trajectory_cycle.py \
		--max_cycles $(MAX_CYCLES) \
		--tasks_per_cycle $(TASKS_PER_CYCLE) \
		--runs_per_task $(RUNS_PER_TASK) \
		--num_workers $(WORKERS) \
		$(MODEL_ARGS) \
		--tag $(TAG) \
		--build_playbook \
		--validate_trajectory \
		--task_embedding_model $(TASK_EMBEDDING_MODEL) \
		--task_embedding_base_url $(TASK_EMBEDDING_BASE_URL)

cycle-eval:
	$(PYTHON) run_trajectory_cycle_playbook_eval.py \
		--max_cycles $(MAX_CYCLES) \
		--tasks_per_cycle $(TASKS_PER_CYCLE) \
		--runs_per_task $(RUNS_PER_TASK) \
		--num_workers $(WORKERS) \
		$(MODEL_ARGS) \
		--tag $(TAG) \
		--validate_trajectory \
		--task_embedding_model $(TASK_EMBEDDING_MODEL) \
		--task_embedding_base_url $(TASK_EMBEDDING_BASE_URL) \
		--eval_split $(SPLIT)

adapt:
	$(PYTHON) run_sequential_adapt.py \
		--split $(SPLIT) \
		--mode $(MODE) \
		$(MODEL_ARGS) \
		--tag $(TAG)

direct-memory:
	$(PYTHON) -m experiments.appworld.scripts.static_doc_playbook_builder \
		--output_dir outputs \
		--tag direct_memory \
		--model_name $(MEMORY_MODEL)

filter-playbook:
ifndef PB_FILE
	$(error PB_FILE is required. Usage: make filter-playbook PB_FILE=outputs/playbook.json)
endif
	$(PYTHON) scripts/filter_playbook.py $(PB_FILE)
