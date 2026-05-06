
VALIDATION_PROMPT = '''You are an expert evaluator of agent task execution.

Evaluate the following Task and Trajectory using a 5-point Likert scale for each criterion.

## Evaluation Criteria

### 1. Task Feasibility
Evaluate whether the task is executable and the entities are grounded in the current environment.
- Do not judge feasibility by API availability alone.
- Verify that every required target entity in the instruction (person, contact, request, file, thread, project, etc.) is present and can be uniquely identified from the trajectory evidence.
- If the entity cannot be identified after reasonable paginated search using relevant APIs, the task is infeasible.

- **5 (Excellent)**: All required entities/constraints are explicitly present and uniquely identified; clearly executable as stated.
- **4 (Good)**: Entities are present with minor ambiguity resolvable by a simple lookup; no core entity missing.
- **3 (Acceptable)**: Plausible, but key grounding is uncertain or search is insufficient.
- **2 (Poor)**: Likely infeasible; required entity not found or not uniquely identifiable after reasonable search.
- **1 (Unacceptable)**: Infeasible/contradictory or relies on non-existent entities/conditions.

### 2. Task Completion
Judge whether the task instruction was successfully completed.

- Evaluate all task instructions are satisfied based on the trajectory, including all constraints.
- Treat a call to `apis.supervisor.complete_task(...)` as a required completion step, not as evidence that the task was successful correctly.
- If the task asked for action, do not treat the task as successful unless the trajectory includes a call to `apis.supervisor.complete_task()` with no answer argument (or with `None`).
- If the task asked for information, success requires a call to `apis.supervisor.complete_task(answer=<answer>)`, and the answer should contain only the exact requested value: a number (int or float), a direct string value, a plain comma-separated list, or `yes`/`no`.
- Do not treat verbose answers, action summaries, explanations, full sentences, or answers with extra words, units, or symbols as correct unless explicitly requested. e.g. `10` is acceptable, but `The answer is 10 songs.` is not. Do not require an answer for action-only tasks.

- **5 (Excellent)**: Every requirement AND every constraint is satisfied, and the trajectory shows clear completion for each required outcome.
- **4 (Good)**: All requirements/constraints appear satisfied with minor ambiguity, and nothing important is missing.
- **3 (Acceptable)**: Some progress, but at least one requirement/constraint is missing, ambiguous, or only asserted without support in the trajectory steps (surface-level success). This includes cases where an object/ID exists but correctness constraints (privacy, filters, counts, formatting, attachment actually uploaded, etc.) are not confirmed.
- **2 (Poor)**: Minor progress only; the main outcome is not achieved or an error/failed response prevents completion.
- **1 (Unacceptable)**: No meaningful progress toward the task goal.

## Task Instruction
{task_instruction}

## Trajectory
{trajectory}

Respond in JSON format:
```json
{{
  "feasibility_reason": "1-2 sentence explanation for feasibility score",
  "feasibility_score": 1-5,

  "task_completion_reason": "1-2 sentence explanation for task completion score",
  "task_completion_score": 1-5
}}
```
'''
