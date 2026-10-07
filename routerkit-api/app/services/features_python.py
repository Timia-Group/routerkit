"""Programmatic features: executes `question.script` on `task.prompt`.

`question.script` (when defined) is the text of a complete Python
function, with fixed signature `def task_func(task_prompt: str) -> float`
(the same function name BigCodeBench uses as the entry_point of its
tasks). The text is `exec()`'d in an isolated namespace, the `task_func`
function is taken from it, and it's called with `task.prompt`. No sandbox
or timeout of our own (an explicit decision): whoever writes questions.json
is assumed to be trusted. `question.repetitions` is ignored here — a
script is deterministic, repeating it would just recompute the exact same
value (unlike features_llm.compute_llm_feature, where it guards against
the LLM's answer varying between calls).
"""

import time
from dataclasses import dataclass

from app.db.models import Question, Task


class ScriptError(RuntimeError):
    """The question's script doesn't define `task_func`, or `task_func` raised an exception."""


@dataclass
class PythonFeatureResult:
    value: float
    latency_ms: float


def compute_python_feature(task: Task, question: Question) -> PythonFeatureResult:
    """Executes `question.script` with `task.prompt` and returns the computed value.

    Precondition: `question.script` is not None (if it is, this question
    must be resolved with features_llm.compute_llm_feature instead).
    Raises ScriptError if the script doesn't define `task_func`, or if
    `task_func` raises an exception when executed; the caller
    (routers/features.py) decides what to do with that error (by
    convention: it's reported and processing continues with the others).
    """
    if not question.script:
        raise ScriptError(f"question {question.question_id} has no script defined")

    namespace: dict = {}
    started = time.perf_counter()
    try:
        exec(question.script, namespace)  # noqa: S102 - trusted script, curated by the team
        task_func = namespace.get("task_func")
        if task_func is None:
            raise ScriptError(f"question {question.question_id}: the script doesn't define 'task_func'")
        value = float(task_func(task.prompt))
    except ScriptError:
        raise
    except Exception as exc:  # noqa: BLE001 - reported as ScriptError to the caller
        raise ScriptError(f"question {question.question_id}: {exc}") from exc

    if question.kind == "binary" and value not in (0.0, 1.0):
        raise ScriptError(
            f"question {question.question_id}: kind='binary' requires 0 or 1, the script returned {value}"
        )

    latency_ms = (time.perf_counter() - started) * 1000
    return PythonFeatureResult(value=value, latency_ms=latency_ms)
