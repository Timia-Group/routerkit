"""Generation of candidate solutions: asks a model to solve a task.

Fixed prompt (the same one BigCodeBench uses to generate candidates): we
don't add anything about the expected function name or any other
instruction — the statement (`task.prompt`) is responsible for specifying
everything necessary, we don't control that. The LLM's response is stored
exactly as it comes (`code = response.text`, without extracting or
cleaning up markdown code blocks or anything else) — the resulting code is
not processed or validated (syntax, function name, fences, etc.); that's
the responsibility of the evaluation stage, not this one.

A failure of the LLM call (`LlmClientError`) is not propagated: it's caught
here and returned as `GenerationResult(code=None, gen_error=...)`, so the
caller can still insert a row (a record that this particular generation
failed) without aborting the rest of the batch.
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
    """Generates a candidate solution for `task` using `model`.

    `model.temperature` (if defined) is resolved by app.clients.llm_client
    alone; no temperature is passed per request. Never raises: if the LLM
    call fails, returns a GenerationResult with `code=None` and `gen_error` set.
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

    latency_ms = (time.perf_counter() - started) * 1000
    return GenerationResult(
        code=response.text,
        tokens_in=response.prompt_tokens,
        tokens_out=response.completion_tokens,
        latency_ms=latency_ms,
        cost_usd=llm_client.cost_usd(
            model, response.prompt_tokens, response.completion_tokens, latency_ms=latency_ms
        ),
        gen_error=None,
    )
