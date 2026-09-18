"""Cálculo de features haciéndole `questions` a un LLM sobre una `task`.

Se usa solo para preguntas sin `question.script` (ver features_python.py
para las que sí lo tienen), llamando al único model con role='feature'
activo vía app.clients.llm_client.complete(). Una sola llamada por
(task, question) — no hay concepto de "corridas" repetidas a promediar.

Extracción del valor según `question.kind` (probabilidad = exp(logprob),
nunca el logprob crudo):

- binary: se le pide al modelo que responda únicamente "YES" o "NO". El
  valor de la feature es siempre "probabilidad de YES": si respondió YES,
  `exp(logprob)` de ese token; si respondió NO, `1 - exp(logprob)`.
- numeric: se le pide que responda únicamente un número entre `lo` y `hi`.
  Se toman los ≤20 candidatos del primer token (`top_logprobs`) con sus
  probabilidades (`exp(logprob)`, sin normalizar a que sumen 1), y el valor
  es el promedio ponderado: Σ(valor_i · prob_i) / Σ(prob_i).

Solo se controlan las casuísticas esperadas: respuesta binaria YES/NO
(tolerante a mayúsculas/minúsculas, puntuación final, y variantes en
español "sí"/"no" — ver llm_client.normalize_yes_no), y candidatos numéricos
parseables como número. Cualquier otro caso (respuesta binaria ambigua,
candidato no numérico, sin logprobs en la respuesta) se reporta como
LlmFeatureError — no se intenta adivinar más allá de eso.

Los logprobs crudos que devuelve el proveedor (`top_logprobs` en la
respuesta de llm_client.complete) se serializan a texto y se guardan en
`features.answer_logprobs`.
"""

import json
import math
import time
from dataclasses import dataclass

from app.clients import llm_client
from app.db.models import Model, Question, Task


class LlmFeatureError(RuntimeError):
    """La llamada al LLM falló, o su respuesta no matchea ninguna casuística esperada."""


@dataclass
class LlmFeatureResult:
    value: float
    answer_text: str
    answer_logprobs: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float


def _build_messages(task: Task, question: Question) -> list[dict]:
    if question.kind == "binary":
        instruction = 'Respondé ÚNICAMENTE con la palabra "YES" o la palabra "NO", en mayúsculas, sin ningún otro texto.'
    else:
        lo, hi = question.lo, question.hi
        rango = f" entre {lo} y {hi}" if lo is not None and hi is not None else ""
        instruction = f"Respondé ÚNICAMENTE con un número{rango}, sin unidades, texto ni explicación."

    return [
        {
            "role": "system",
            "content": (
                "Sos un evaluador experto de tareas de programación. Se te muestra el "
                f"enunciado de una tarea y una pregunta sobre esa tarea. {instruction}"
            ),
        },
        {
            "role": "user",
            "content": f"Tarea:\n{task.prompt}\n\nPregunta: {question.text}",
        },
    ]


def _extract_binary_value(answer_text: str, raw_logprobs: dict | None) -> float:
    try:
        first_token = llm_client.first_token_logprob(raw_logprobs)
    except llm_client.LlmClientError as exc:
        raise LlmFeatureError(str(exc)) from exc
    answer = llm_client.normalize_yes_no(answer_text)

    if answer == "YES":
        return math.exp(first_token["logprob"])
    if answer == "NO":
        return 1 - math.exp(first_token["logprob"])

    raise LlmFeatureError(f"respuesta binaria inesperada (ni YES ni NO): {answer_text!r}")


def _extract_numeric_value(raw_logprobs: dict | None) -> float:
    try:
        first_token = llm_client.first_token_logprob(raw_logprobs)
    except llm_client.LlmClientError as exc:
        raise LlmFeatureError(str(exc)) from exc
    candidates = first_token.get("top_logprobs") or []
    if not candidates:
        raise LlmFeatureError("la respuesta del LLM no trae top_logprobs para el primer token")

    values: list[float] = []
    weights: list[float] = []
    for candidate in candidates:
        try:
            values.append(float(candidate["token"].strip()))
        except (ValueError, AttributeError) as exc:
            raise LlmFeatureError(
                f"candidato no numérico entre los top_logprobs: {candidate.get('token')!r}"
            ) from exc
        weights.append(math.exp(candidate["logprob"]))

    total_weight = sum(weights)
    if total_weight == 0:
        raise LlmFeatureError("la suma de probabilidades de los candidatos numéricos dio 0")

    return sum(v * w for v, w in zip(values, weights, strict=True)) / total_weight


def compute_llm_feature(task: Task, question: Question, model: Model) -> LlmFeatureResult:
    """Le hace `question.text` a `model` sobre `task.prompt` y extrae un valor de feature.

    Lanza LlmFeatureError si la respuesta no matchea la casuística esperada
    para `question.kind`, o LlmClientError si falló la llamada en sí; el
    caller (routers/features.py) decide qué hacer (por convención: se
    reporta en `errors` y se sigue con las demás features).
    """
    top_logprobs_count = 1 if question.kind == "binary" else 20

    started = time.perf_counter()
    response = llm_client.complete(
        model,
        _build_messages(task, question),
        logprobs=True,
        top_logprobs=top_logprobs_count,
    )
    latency_ms = (time.perf_counter() - started) * 1000

    if question.kind == "binary":
        value = _extract_binary_value(response.text, response.top_logprobs)
    else:
        value = _extract_numeric_value(response.top_logprobs)

    return LlmFeatureResult(
        value=value,
        answer_text=response.text,
        answer_logprobs=json.dumps(response.top_logprobs) if response.top_logprobs else None,
        input_tokens=response.prompt_tokens,
        output_tokens=response.completion_tokens,
        latency_ms=latency_ms,
        cost_usd=llm_client.cost_usd(model, response.prompt_tokens, response.completion_tokens),
    )
