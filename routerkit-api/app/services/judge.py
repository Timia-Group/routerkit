"""Grading vía LLM juez, para tareas custom (routerkit.tasks.bcb_task_id IS NULL).

No se ejecuta código en un sandbox nuestro: se le envía al LLM juez (el
único model con role='judge' activo) el enunciado de la tarea, sus tests si
existen, y la solución candidata, y se le pregunta YES/NO sobre si la
resuelve. A diferencia de services.features_llm, acá NO se usan logprobs ni
se calcula ninguna probabilidad calibrada — la evaluación es puramente
binaria, igual que el resultado de bigcodebench-api (`passed` booleano, sin
más):

- `passed` = respondió YES (ver app.clients.llm_client.normalize_yes_no).
- `fail_reason`: si `passed=False`, se hace una SEGUNDA llamada pidiéndole
  al juez que explique en una oración por qué falla.
"""

import time
from dataclasses import dataclass

from app.clients import llm_client
from app.db.models import Model, Solution, Task


class JudgeError(RuntimeError):
    """La respuesta del juez no matchea la casuística esperada (ni YES ni NO)."""


@dataclass
class JudgeGradeResult:
    passed: bool
    fail_reason: str | None
    exec_ms: float | None


def _verdict_messages(task: Task, solution: Solution) -> list[dict]:
    parts = [f"Enunciado de la tarea:\n{task.prompt}"]
    if task.tests:
        parts.append(f"Tests / criterios de validación conocidos:\n{task.tests}")
    parts.append(f"Solución propuesta (código Python):\n```python\n{solution.code}\n```")
    parts.append(
        "¿Esta solución resuelve correcta y completamente la tarea, cumpliendo "
        'los tests/criterios si los hay? Respondé ÚNICAMENTE con la palabra "YES" '
        'o la palabra "NO", sin ningún otro texto.'
    )
    return [
        {
            "role": "system",
            "content": (
                "Sos un evaluador experto de código Python. Vas a recibir el enunciado "
                "de una tarea de programación, opcionalmente sus tests o criterios de "
                "validación, y una solución candidata. Tu trabajo es decidir si esa "
                "solución resuelve correctamente la tarea."
            ),
        },
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _reason_messages(task: Task, solution: Solution) -> list[dict]:
    messages = _verdict_messages(task, solution)
    messages[-1] = {
        "role": "user",
        "content": (
            messages[-1]["content"].rsplit("\n\n", 1)[0]
            + "\n\nEsa solución NO resuelve correctamente la tarea. Explicá en una "
            "sola oración corta por qué falla."
        ),
    }
    return messages


def grade_with_llm_judge(task: Task, solution: Solution, model: Model) -> JudgeGradeResult:
    """Le pregunta al juez (YES/NO) si `solution.code` resuelve `task`.

    Lanza JudgeError si la respuesta no matchea YES/NO, o LlmClientError si
    falló alguna de las llamadas; el caller (routers/evaluation.py) decide
    qué hacer (por convención: se reporta con has_error=true, sin abortar
    el resto del lote).
    """
    started = time.perf_counter()
    response = llm_client.complete(model, _verdict_messages(task, solution))

    answer = llm_client.normalize_yes_no(response.text)
    if answer is None:
        raise JudgeError(f"respuesta del juez inesperada (ni YES ni NO): {response.text!r}")

    passed = answer == "YES"

    fail_reason = None
    if not passed:
        reason_response = llm_client.complete(model, _reason_messages(task, solution))
        fail_reason = reason_response.text.strip()

    exec_ms = (time.perf_counter() - started) * 1000
    return JudgeGradeResult(passed=passed, fail_reason=fail_reason, exec_ms=exec_ms)
