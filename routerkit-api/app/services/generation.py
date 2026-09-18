"""Generación de soluciones candidatas: le pide a un modelo que resuelva una tarea.

Prompt fijo (mismo que usa BigCodeBench para generar candidatos): no le
agregamos nada sobre el nombre de función esperado ni ninguna otra
instrucción — el enunciado (`task.prompt`) es responsable de especificar
todo lo necesario, no lo controlamos nosotros. La respuesta del LLM se
guarda tal cual viene (`code = response.text`, sin extraer ni limpiar
bloques markdown ni nada más) — no se procesa ni se valida el código
resultante (sintaxis, nombre de función, fences, etc.); eso es
responsabilidad de la etapa de evaluación, no de esta.

Un fallo de la llamada al LLM (`LlmClientError`) no se propaga: se captura
acá y se devuelve como `GenerationResult(code=None, gen_error=...)`, para
que el caller pueda insertar igual una fila (histórico de que esa
generación puntual falló) sin abortar el resto del lote.
"""

import time
from dataclasses import dataclass

from app.clients import llm_client
from app.clients.llm_client import LlmClientError
from app.db.models import Model, Task

_PROMPT_TEMPLATE = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:\n{task_prompt}"
)


@dataclass
class GenerationResult:
    code: str | None
    tokens_in: int | None
    tokens_out: int | None
    latency_ms: float
    cost_usd: float
    gen_error: str | None


def generate_solution(task: Task, model: Model) -> GenerationResult:
    """Genera una solución candidata para `task` usando `model`.

    `model.temperature` (si está definida) la resuelve app.clients.llm_client
    solo; no se pasa temperatura por request. Nunca lanza: si la llamada al
    LLM falla, devuelve un GenerationResult con `code=None` y `gen_error` seteado.
    """
    started = time.perf_counter()
    try:
        response = llm_client.complete(
            model, [{"role": "user", "content": _PROMPT_TEMPLATE.format(task_prompt=task.prompt)}]
        )
    except LlmClientError as exc:
        return GenerationResult(
            code=None,
            tokens_in=None,
            tokens_out=None,
            latency_ms=(time.perf_counter() - started) * 1000,
            cost_usd=0.0,
            gen_error=str(exc),
        )

    return GenerationResult(
        code=response.text,
        tokens_in=response.prompt_tokens,
        tokens_out=response.completion_tokens,
        latency_ms=(time.perf_counter() - started) * 1000,
        cost_usd=llm_client.cost_usd(model, response.prompt_tokens, response.completion_tokens),
        gen_error=None,
    )
