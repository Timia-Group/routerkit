"""Grading via LLM judge, for custom tasks (routerkit.tasks.bcb_task_id IS NULL).

No code is executed in a sandbox of our own: the LLM judge (the single
active model with role='judge') is sent the task statement, its tests if
they exist, and the candidate solution, and is asked YES/NO on whether it
solves it. Unlike services.features_llm, logprobs are NOT used here and no
calibrated probability is computed — the evaluation is purely binary, just
like the result from bigcodebench-api (a boolean `passed`, nothing more):

- `passed` = answered YES (see app.clients.llm_client.normalize_yes_no).
- `fail_reason` is always None here — only YES/NO matters, there's no
  second call asking the judge to explain itself (keeps this to a single
  LLM call per grade, same cost whether it passes or not).
"""

import time
from dataclasses import dataclass

from app.clients import llm_client
from app.db.models import Model, Solution, Task


class JudgeError(RuntimeError):
    """The judge's response doesn't match the expected case (neither YES nor NO)."""


@dataclass
class JudgeGradeResult:
    passed: bool
    fail_reason: str | None
    exec_ms: float | None


def _verdict_messages(task: Task, solution: Solution) -> list[dict]:
    parts = [f"Task statement:\n{task.prompt}"]
    if task.tests:
        parts.append(f"Known tests / validation criteria:\n{task.tests}")
    parts.append(f"Proposed solution (Python code):\n```python\n{solution.code}\n```")
    parts.append(
        "Does this solution correctly and completely solve the task, meeting "
        'the tests/criteria if any? Answer ONLY with the word "YES" '
        'or the word "NO", with no other text.'
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator of Python code. You will receive the "
                "statement of a programming task, optionally its tests or validation "
                "criteria, and a candidate solution. Your job is to decide whether "
                "that solution correctly solves the task."
            ),
        },
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def grade_with_llm_judge(task: Task, solution: Solution, model: Model) -> JudgeGradeResult:
    """Asks the judge (YES/NO) whether `solution.code` solves `task`.

    Raises JudgeError if the response doesn't match YES/NO, or
    LlmClientError if the call failed; the caller (routers/evaluation.py)
    decides what to do (by convention: it's reported with has_error=true,
    without aborting the rest of the batch).
    """
    started = time.perf_counter()
    response = llm_client.complete(model, _verdict_messages(task, solution))

    answer = llm_client.normalize_yes_no(response.text)
    if answer is None:
        raise JudgeError(f"unexpected judge response (neither YES nor NO): {response.text!r}")

    exec_ms = (time.perf_counter() - started) * 1000
    return JudgeGradeResult(passed=answer == "YES", fail_reason=None, exec_ms=exec_ms)
