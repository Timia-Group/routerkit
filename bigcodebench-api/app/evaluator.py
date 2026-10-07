"""Wrapper over the BigCodeBench sandbox.

`bigcodebench.evaluate` (the official CLI) is designed to run the full
benchmark against the official dataset. To evaluate a single (task, solution)
pair on demand, we use the lower-level pieces exposed by the package
directly:

- `get_bigcodebench()` to resolve an official `bcb_task_id` (e.g.
  "BigCodeBench/12") to the real `test` and `entry_point` of that task.
- `untrusted_check(...)` to run the candidate solution against that test
  inside the sandbox (subprocess with resource limits) already bundled in
  the `bigcodebench/bigcodebench-evaluate` image.
"""

import time

from bigcodebench.data import get_bigcodebench
from bigcodebench.eval import untrusted_check
from bigcodebench.sanitize import sanitize

from app.schemas import EvaluateResult

# Defaults taken from the values documented for the official CLI
# (see bigcodebench's ADVANCED_USAGE.md). Can be overridden per request.
DEFAULT_MAX_AS_LIMIT_MB = 30 * 1024
DEFAULT_MAX_DATA_LIMIT_MB = 30 * 1024
DEFAULT_MAX_STACK_LIMIT_MB = 10
DEFAULT_MIN_TIME_LIMIT_S = 10
DEFAULT_GT_TIME_LIMIT_S = 60

_dataset: dict | None = None


def load_dataset() -> dict:
    """Load and cache the official dataset (the "full" subset) in process memory.

    Called once when the container starts up (see app/main.py). The first
    call may download the dataset from Hugging Face; subsequent calls to
    get_bigcodebench() within the same process reuse the package's internal
    cache. No endpoint is exposed to list/download this dataset:
    bigcodebench-api only uses it to resolve the real `test` when evaluating;
    if the user wants the official catalog to upload to routerkit-api, they
    generate it separately (see the repo's datasets folder).
    """
    global _dataset
    if _dataset is None:
        _dataset = get_bigcodebench(subset="full")
    return _dataset


def get_task(bcb_task_id: str) -> dict:
    """Return the official task dict (with 'test' and 'entry_point').

    Raises KeyError if the bcb_task_id does not exist in the official dataset.
    """
    dataset = _dataset if _dataset is not None else load_dataset()
    return dataset[bcb_task_id]


def evaluate(
    bcb_task_id: str,
    solution: str,
    max_as_limit_mb: float | None = None,
    max_data_limit_mb: float | None = None,
    max_stack_limit_mb: float | None = None,
    min_time_limit_s: float | None = None,
    gt_time_limit_s: float | None = None,
) -> EvaluateResult:
    """Run `untrusted_check` for (bcb_task_id, solution) and return the verdict.

    `solution` arrives exactly as the candidate LLM produced it (markdown
    fences, stray prose, unrelated helper code — see
    routerkit-api/app/services/generation.py, which stores it unprocessed on
    purpose). `sanitize()` is the same post-processing step the official
    BigCodeBench CLI applies before evaluating: it extracts the longest
    syntactically valid block and keeps only the code reachable from
    `entry_point`, dropping everything else (markdown fences included).
    Without this, every solution fails with a syntax error on the fence line
    regardless of whether the underlying code is correct.

    Blocking (forks a subprocess) — must be called from a threadpool, not
    directly on FastAPI's event loop (see app/jobs.py).
    """
    task = get_task(bcb_task_id)
    sanitized_solution = sanitize(solution, entrypoint=task["entry_point"])

    started = time.perf_counter()
    status, details = untrusted_check(
        sanitized_solution,
        task["test"],
        task["entry_point"],
        max_as_limit_mb if max_as_limit_mb is not None else DEFAULT_MAX_AS_LIMIT_MB,
        max_data_limit_mb if max_data_limit_mb is not None else DEFAULT_MAX_DATA_LIMIT_MB,
        max_stack_limit_mb if max_stack_limit_mb is not None else DEFAULT_MAX_STACK_LIMIT_MB,
        min_time_limit_s if min_time_limit_s is not None else DEFAULT_MIN_TIME_LIMIT_S,
        gt_time_limit_s if gt_time_limit_s is not None else DEFAULT_GT_TIME_LIMIT_S,
    )
    exec_ms = (time.perf_counter() - started) * 1000

    passed = status == "pass"
    fail_reason = None if passed else status

    return EvaluateResult(
        passed=passed,
        status=status,
        fail_reason=fail_reason,
        exec_ms=exec_ms,
        details=dict(details) if details else None,
    )
