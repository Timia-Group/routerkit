"""Generation of candidate solutions: asks a model to solve a task.

Fixed prompt (the same one BigCodeBench uses to generate candidates): we
don't add anything about the expected function name or any other
instruction — the statement (`task.prompt`) is responsible for specifying
everything necessary, we don't control that. The LLM's response is stored
exactly as it comes (`code = response.text`, without extracting or
cleaning up markdown code blocks or anything else) — the resulting code is
not processed or validated (syntax, function name, fences, etc.); that's
the responsibility of the evaluation stage, not this one.

A failure of the LLM call (`LlmClientError`) propagates — same convention
as services.features_llm/services.judge: the caller (routers/solutions.py)
catches it and still inserts a row (`has_error=true`, `gen_error` with the
message) without aborting the rest of the batch. Letting it propagate
(rather than catching it here and returning it as a field) is also what
lets services.concurrency.run_bounded's circuit breaker see these failures
at all — it only reacts to raised exceptions.
"""

import time
from dataclasses import dataclass

from app.clients import llm_client
from app.db.models import Model, Task

_PROMPT_TEMPLATE = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:\n{task_prompt}"
)


@dataclass
class GenerationResult:
    code: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float


def generate_solution(task: Task, model: Model) -> GenerationResult:
    """Generates a candidate solution for `task` using `model`.

    `model.temperature` (if defined) is resolved by app.clients.llm_client
    alone; no temperature is passed per request. Raises LlmClientError if
    the call fails.
    """
    started = time.perf_counter()
    response = llm_client.complete(
        model, [{"role": "user", "content": _PROMPT_TEMPLATE.format(task_prompt=task.prompt)}]
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
    )
