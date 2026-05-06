"""Generic benchmark runner.

This runner is intentionally minimal: it handles task enumeration, optional
parallel execution, retry/timeout handling, and result summarization via the
Benchmark protocol. Benchmarks remain in charge of execution semantics.
"""
from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import List

from preping.core.interfaces.benchmark import BenchmarkAdapter, BenchmarkResult, RunnerConfig


logger = logging.getLogger(__name__)


def _with_attempt(task_args, attempt: int):
    payload = dict(task_args)
    payload["attempt_index"] = attempt
    return payload


def _execute_with_retries(
    *,
    worker,
    benchmark: BenchmarkAdapter,
    task_args,
    runner_config: RunnerConfig,
    context,
) -> BenchmarkResult:
    retries = 0
    while True:
        attempt_args = _with_attempt(task_args, retries)
        try:
            raw = worker(attempt_args)
            return benchmark.build_result(raw, attempt_args, runner_config=runner_config, context=context)
        except Exception as exc:  # noqa: PERF203 - rare path
            retries += 1
            if retries > runner_config.max_retries:
                return benchmark.build_result(
                    {"success": False, "error": str(exc)},
                    attempt_args,
                    runner_config=runner_config,
                    context=context,
                )
            logger.warning(
                "Retrying task %s (%s/%s) after error: %s",
                task_args.get("task_id"),
                retries,
                runner_config.max_retries,
                exc,
            )


def run_benchmark(
    benchmark: BenchmarkAdapter,
    *,
    runner_config: RunnerConfig | None = None,
) -> dict:
    """Run all tasks for ``benchmark`` using the supplied ``RunnerConfig``.

    Returns the benchmark summary augmented with per-task results.
    """

    runner_config = runner_config or RunnerConfig()
    context = benchmark.prepare(runner_config=runner_config)

    tasks = list(benchmark.iter_tasks(runner_config=runner_config, context=context))
    total = len(tasks)
    results: List[BenchmarkResult] = []

    worker = benchmark.get_worker()
    task_args_list = [
        benchmark.build_task_args(task, runner_config=runner_config, context=context, position=i, total=total)
        for i, task in enumerate(tasks)
    ]

    # Skip handling upfront to keep worker picklable
    runnable_args: List = []
    for task_args in task_args_list:
        if benchmark.should_skip(task_args, runner_config=runner_config, context=context):
            results.append(
                benchmark.build_result(
                    {"success": True, "skipped": True},
                    task_args,
                    runner_config=runner_config,
                    context=context,
                )
            )
        else:
            runnable_args.append(task_args)

    if runner_config.debug or runner_config.num_workers == 1:
        for task_args in runnable_args:
            results.append(
                _execute_with_retries(
                    worker=worker,
                    benchmark=benchmark,
                    task_args=task_args,
                    runner_config=runner_config,
                    context=context,
                )
            )
    else:
        pending = {}
        start_times = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=runner_config.num_workers) as executor:
            for task_args in runnable_args:
                attempt_args = _with_attempt(task_args, 0)
                fut = executor.submit(worker, attempt_args)
                pending[fut] = (task_args, 0)
                start_times[fut] = time.time()

            while pending:
                done, _ = concurrent.futures.wait(
                    pending,
                    timeout=30,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                for fut in list(done):
                    task_args, attempt = pending.pop(fut)
                    start_times.pop(fut, None)
                    try:
                        raw = fut.result()
                        attempt_args = _with_attempt(task_args, attempt)
                        results.append(
                            benchmark.build_result(
                                raw,
                                attempt_args,
                                runner_config=runner_config,
                                context=context,
                            )
                        )
                    except Exception as exc:  # noqa: PERF203 - rare path
                        if attempt < runner_config.max_retries:
                            next_attempt = attempt + 1
                            next_attempt_args = _with_attempt(task_args, next_attempt)
                            new_future = executor.submit(worker, next_attempt_args)
                            pending[new_future] = (task_args, attempt + 1)
                            start_times[new_future] = time.time()
                            continue
                        attempt_args = _with_attempt(task_args, attempt)
                        results.append(
                            benchmark.build_result(
                                {"success": False, "error": str(exc)},
                                attempt_args,
                                runner_config=runner_config,
                                context=context,
                            )
                        )

                # Timeout handling
                now = time.time()
                for fut in list(pending):
                    timeout = runner_config.task_timeout_seconds
                    if timeout and now - start_times.get(fut, now) > timeout:
                        task_args, attempt = pending.pop(fut)
                        start_times.pop(fut, None)
                        fut.cancel()
                        if attempt < runner_config.max_retries:
                            next_attempt = attempt + 1
                            next_attempt_args = _with_attempt(task_args, next_attempt)
                            new_future = executor.submit(worker, next_attempt_args)
                            pending[new_future] = (task_args, attempt + 1)
                            start_times[new_future] = time.time()
                        else:
                            attempt_args = _with_attempt(task_args, attempt)
                            results.append(
                                benchmark.build_result(
                                    {"success": False, "error": "timeout"},
                                    attempt_args,
                                    runner_config=runner_config,
                                    context=context,
                                )
                            )

    summary = benchmark.summarize(results, runner_config=runner_config, context=context)
    return {"summary": summary, "results": results}
