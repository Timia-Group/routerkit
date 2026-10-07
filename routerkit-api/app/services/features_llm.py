"""Computes features by asking an LLM `questions` about a `task`.

Used only for questions without `question.script` (see features_python.py
for the ones that have it), calling the single active model with
role='feature' via app.clients.llm_client.complete().

Value extraction depends on `question.kind` (probability = exp(logprob),
never the raw logprob):

- binary: the model is asked to answer only "YES" or "NO". The feature
  value is always "probability of YES": if it answered YES, `exp(logprob)`
  of that token; if it answered NO, `1 - exp(logprob)`.
- numeric: the model is asked to answer only a number between `lo` and
  `hi`. The ≤20 candidates for the first token (`top_logprobs`) are taken
  with their probabilities (`exp(logprob)`, not normalized to sum to 1),
  and the value is the weighted average: Σ(value_i · prob_i) / Σ(prob_i).

Only the expected cases are handled: a binary YES/NO response (tolerant of
case, trailing punctuation, and the Spanish variants "sí"/"no" — see
llm_client.normalize_yes_no), and numeric candidates parseable as a number.
Any other case (ambiguous binary response, non-numeric candidate, no
logprobs in the response) is reported as LlmFeatureError — no attempt is
made to guess beyond that.

The raw logprobs returned by the provider (`top_logprobs` in the response
from llm_client.complete) are serialized to text and stored in
`features.answer_logprobs`.

`question.repetitions` (default 1) repeats the whole single call that many
times and averages (simple mean) the resulting `value` — everything else
(tokens, cost, latency) is summed across repetitions. `answer_text`/
`answer_logprobs` are the single call's value when repetitions=1 (the
common case, unchanged), or a JSON array of each repetition's value when
repetitions>1. Meant to guard against the LLM's answer varying between
calls, at the cost of `repetitions` times the tokens/price for that (task,
question) pair.
"""

import json
import math
import time
from dataclasses import dataclass

from app.clients import llm_client
from app.db.models import Model, Question, Task


class LlmFeatureError(RuntimeError):
    """The LLM call failed, or its response doesn't match any expected case."""


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
        instruction = 'Answer ONLY with the word "YES" or the word "NO", in uppercase, with no other text.'
    else:
        lo, hi = question.lo, question.hi
        rango = f" between {lo} and {hi}" if lo is not None and hi is not None else ""
        instruction = f"Answer ONLY with a number{rango}, with no units, text, or explanation."

    return [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator of programming tasks. You are shown the "
                f"statement of a task and a question about that task. {instruction}"
            ),
        },
        {
            "role": "user",
            "content": f"Task:\n{task.prompt}\n\nQuestion: {question.text}",
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

    raise LlmFeatureError(f"unexpected binary response (neither YES nor NO): {answer_text!r}")


def _extract_numeric_value(raw_logprobs: dict | None) -> float:
    try:
        first_token = llm_client.first_token_logprob(raw_logprobs)
    except llm_client.LlmClientError as exc:
        raise LlmFeatureError(str(exc)) from exc
    candidates = first_token.get("top_logprobs") or []
    if not candidates:
        raise LlmFeatureError("the LLM response doesn't include top_logprobs for the first token")

    values: list[float] = []
    weights: list[float] = []
    for candidate in candidates:
        try:
            values.append(float(candidate["token"].strip()))
        except (ValueError, AttributeError) as exc:
            raise LlmFeatureError(
                f"non-numeric candidate among the top_logprobs: {candidate.get('token')!r}"
            ) from exc
        weights.append(math.exp(candidate["logprob"]))

    total_weight = sum(weights)
    if total_weight == 0:
        raise LlmFeatureError("the sum of probabilities of the numeric candidates was 0")

    return sum(v * w for v, w in zip(values, weights, strict=True)) / total_weight


def _compute_single_call(task: Task, question: Question, model: Model) -> LlmFeatureResult:
    """A single call to `model` asking `question.text` about `task.prompt`."""
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


def compute_llm_feature(task: Task, question: Question, model: Model) -> LlmFeatureResult:
    """Asks `model` `question.text` about `task.prompt` and extracts a
    feature value, repeating `question.repetitions` times (default 1) and
    averaging (simple mean) the resulting `value` — see module docstring
    for how repetitions combine tokens/cost/latency/answer_text/
    answer_logprobs.

    Raises LlmFeatureError if any repetition's response doesn't match the
    expected case for `question.kind`, or LlmClientError if any call itself
    failed; the caller (routers/features.py) decides what to do (by
    convention: it's reported in `errors` and processing continues with the
    other features) — a partial failure (e.g. repetition 2 of 3 fails)
    discards the whole computation for this (task, question) pair rather
    than averaging over fewer repetitions than requested.
    """
    results = [_compute_single_call(task, question, model) for _ in range(question.repetitions)]

    if len(results) == 1:
        return results[0]

    return LlmFeatureResult(
        value=sum(r.value for r in results) / len(results),
        answer_text=json.dumps([r.answer_text for r in results]),
        answer_logprobs=json.dumps([r.answer_logprobs for r in results]),
        input_tokens=sum(r.input_tokens for r in results),
        output_tokens=sum(r.output_tokens for r in results),
        latency_ms=sum(r.latency_ms for r in results),
        cost_usd=sum(r.cost_usd for r in results),
    )
