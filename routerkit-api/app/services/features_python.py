"""Features programáticas: ejecuta `question.script` sobre `task.prompt`.

`question.script` (cuando está definido) es el texto de una función Python
completa, con firma fija `def task_func(task_prompt: str) -> float` (mismo
nombre de función que usa BigCodeBench como entry_point de sus tareas). Se
hace `exec()` del texto en un namespace aislado, se toma la función
`task_func` de ahí y se la llama con `task.prompt`. Sin sandbox ni timeout
propios (decisión explícita): se asume que quien escribe questions.json es
de confianza.
"""

import time
from dataclasses import dataclass

from app.db.models import Question, Task


class ScriptError(RuntimeError):
    """El script de la question no define `task_func`, o `task_func` lanzó una excepción."""


@dataclass
class PythonFeatureResult:
    value: float
    latency_ms: float


def compute_python_feature(task: Task, question: Question) -> PythonFeatureResult:
    """Ejecuta `question.script` con `task.prompt` y devuelve el valor calculado.

    Precondición: `question.script` no es None (si lo es, esta pregunta debe
    resolverse con features_llm.compute_llm_feature en su lugar).
    Lanza ScriptError si el script no define `task_func`, o si `task_func`
    lanza una excepción al ejecutarse; el caller (routers/features.py) decide
    qué hacer con ese error (por convención: se reporta y se sigue con las demás).
    """
    if not question.script:
        raise ScriptError(f"question {question.question_id} no tiene script definido")

    namespace: dict = {}
    started = time.perf_counter()
    try:
        exec(question.script, namespace)  # noqa: S102 - script de confianza, curado por el equipo
        task_func = namespace.get("task_func")
        if task_func is None:
            raise ScriptError(f"question {question.question_id}: el script no define 'task_func'")
        value = float(task_func(task.prompt))
    except ScriptError:
        raise
    except Exception as exc:  # noqa: BLE001 - se reporta como ScriptError al caller
        raise ScriptError(f"question {question.question_id}: {exc}") from exc

    if question.kind == "binary" and value not in (0.0, 1.0):
        raise ScriptError(
            f"question {question.question_id}: kind='binary' requiere 0 o 1, el script devolvió {value}"
        )

    latency_ms = (time.perf_counter() - started) * 1000
    return PythonFeatureResult(value=value, latency_ms=latency_ms)
