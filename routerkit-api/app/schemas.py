"""Pydantic models used to document (Swagger) and validate both requests
and responses across routerkit-api. All input/output is JSON (standardized).

Request side (`*ImportRow`): the import endpoints receive a file
(multipart), so FastAPI can't show the shape of each row in the request
body automatically; these models validate each already-parsed row and are
documented in each endpoint's docstring with an example.

Response side (the rest of this file): every endpoint declares a real
`response_model` instead of a bare `dict` — an unparameterized `dict`
return type has no field names for FastAPI to describe, so Swagger falls
back to a generic "additionalProp1" placeholder example, which isn't
useful to anyone reading the docs. The response_model doesn't change what
each endpoint actually returns (still a plain dict internally) — FastAPI
validates/serializes it against the model, which is also a second line of
defense: a field silently renamed or dropped in the implementation would
fail response validation instead of silently changing the public API.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class TaskImportRow(BaseModel):
    """A row from tasks.jsonl. `bcb_task_id` decides the type per row (not
    per file): if it comes with a value (e.g. "BigCodeBench/12") it's an
    official benchmark task, if omitted it's a custom task — a single file
    can bring both mixed together.

    `custom_task_id` is the stable key for custom tasks (analogous to
    `bcb_task_id`, but assigned by the user): if you're later going to
    reference this task from `features`/`solutions`/`validations`
    (`type=load`), assign it a `custom_task_id` here — never use the
    internal Postgres `task_id` for that, it changes depending on insertion
    order."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "Write a function that adds two integers.",
                "tests": "assert add(2, 3) == 5",
                "entry_point": "add",
                "source": "custom",
                "tags": ["math", "easy"],
                "bcb_task_id": None,
                "custom_task_id": "add-two-integers",
            }
        }
    )

    prompt: str
    tests: str | None = None
    entry_point: str | None = None
    source: str | None = None
    tags: list[str] | None = None
    bcb_task_id: str | None = None
    custom_task_id: str | None = None


class QuestionImportRow(BaseModel):
    """An entry from questions.json.

    `question_key` is the stable key (e.g. "p41") to reference this
    question later from `features.jsonl` (`type=load`) — assign it if
    you're going to need that; never use the internal Postgres
    `question_id` for that, it changes on every reload of
    `questions.json`."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Does the solution use recursion?",
                "kind": "binary",
                "lo": 0,
                "hi": 1,
                "author": "marcos",
                "version": "v1",
                "question_key": "p1",
                "script": None,
            }
        }
    )

    text: str
    # binary: yes/no question (0-1). numeric: numeric value (integer or decimal).
    kind: Literal["binary", "numeric"]
    lo: float | None = None
    hi: float | None = None
    author: str | None = None
    version: str | None = None
    question_key: str | None = None
    # None: the question is resolved by the LLM. With a value: full Python
    # function, fixed signature `def task_func(task_prompt: str) -> float`
    # (same name BigCodeBench uses as entry_point); it's executed with the
    # task's prompt as its only input (no LLM, cost 0).
    script: str | None = None
    # Times to repeat the LLM call and average (simple mean) the result.
    # Ignored for a script question (always computed once, deterministic).
    repetitions: int = 1


class ModelImportRow(BaseModel):
    """An entry from models.json."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "gpt-4o-mini",
                "type": "api",
                "provider": "openai",
                "endpoint": "https://api.openai.com/v1",
                "api_key_env": None,
                "in_token_price": 0.15,
                "out_token_price": 0.6,
                "concurrency": None,
                "timeout_seconds": 300,
                "temperature": None,
                "role": "candidate",
            }
        }
    )

    name: str
    # api: called over HTTP to an external provider. hosted: billed by its own compute time.
    type: Literal["api", "hosted"]
    provider: str | None = None
    endpoint: str | None = None
    deployment: str | None = None
    # Name of the environment variable to use for THIS model (allows having
    # several keys for the same provider, e.g. two OpenAI accounts). None =
    # use the fixed per-provider convention (see .env.example).
    api_key_env: str | None = None
    in_token_price: float | None = None
    out_token_price: float | None = None
    cost_per_second: float | None = None
    # Maximum simultaneous calls to this model. None = no limit defined.
    concurrency: int | None = None
    # Timeout of the HTTP call to this model, in seconds.
    timeout_seconds: int = 300
    # Temperature to use when calling this model (per-model config, not
    # per-request). None = provider default.
    temperature: float | None = None
    # chat_completions (default): litellm.completion(). responses: OpenAI's
    # new API (litellm.responses()) — needed by models that don't fully
    # expose Chat Completions.
    api_kind: Literal["chat_completions", "responses"] = "chat_completions"
    # Maximum output tokens to request from the model. None = not sent.
    # Some APIs require it (e.g. Anthropic Messages).
    max_tokens: int | None = None
    # candidate: solutions are routed to/generated with it (several can be
    # active at once). feature: computes features via LLM. judge: grades
    # solutions of custom tasks. There can only be ONE active model with
    # role='feature' and ONE active model with role='judge' at a time.
    role: Literal["candidate", "feature", "judge"]


class ModelPatchRow(BaseModel):
    """Body of `PATCH /models/{model_id}` — edits connection fields of a
    single already-active model in place (e.g. replacing a `<...>`
    placeholder with a real endpoint). Deliberately NOT the full
    `ModelImportRow` shape: `name`/`role`/`type` are excluded on purpose —
    changing those has structural implications (role drives the one-hot
    column a trained predictor already has baked in; name is the stable
    key solutions/targets are matched against) that a full re-import
    (with its cascade) is the right tool for, not this endpoint. Every
    field is optional and left alone unless explicitly included in the
    request body (`exclude_unset`, not merely non-null — this is how you
    deliberately clear api_key_env back to None, for instance)."""

    # extra="forbid": a typo'd or unsupported field (e.g. "name", trying to
    # rename the model here) is a clear 422, not silently dropped — Pydantic
    # ignores unknown fields by default, which would otherwise make that
    # part of the request look accepted when it was actually a no-op.
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {"endpoint": "https://my-real-resource.openai.azure.com"}},
    )

    endpoint: str | None = None
    api_key_env: str | None = None
    deployment: str | None = None


class CatalogReplaceResult(BaseModel):
    """Response of a catalog import endpoint (tasks/questions/models import
    all share this exact shape)."""

    model_config = ConfigDict(json_schema_extra={"example": {"deactivated_previous": 11, "created": 1128}})

    deactivated_previous: int
    created: int


class JobBoardEntry(BaseModel):
    """Status of one async job kind's active row. This is both what each
    async POST endpoint (`features/compute`, `solutions/generate`,
    `solutions/validate`, `train`) returns right away (202, before the job
    has necessarily finished), and the shape of each entry `GET /jobs`
    returns one of per known kind.

    `last_result` is intentionally loose (`dict[str, Any]`, not a specific
    model): its shape depends on which `kind` produced it (training's
    result looks nothing like a features-compute result) — each kind's own
    endpoint docstring documents what its `last_result` contains."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "kind": "train_predictor",
                "status": "idle",
                "message": "done",
                "started_at": "2026-10-07T20:53:05.025345Z",
                "finished_at": "2026-10-07T21:03:21.568169Z",
                "last_result": {"predictor_id": 12, "test_r2": 0.189, "artifact_path": "/data/artifacts/predictors/predictor_1791406996.joblib"},
            }
        }
    )

    kind: str
    status: Literal["empty", "running", "idle"]
    message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    last_result: dict[str, Any] | None = None


class RankingEntry(BaseModel):
    """One scored candidate in a `POST /route` response — the full list is
    already sorted best (`score` highest) first."""

    model_id: int
    model_name: str
    p: float
    price: float
    score: float


class RouteResponse(BaseModel):
    """Response of `POST /route`."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "decision_id": 42,
                "lambda_used": 0.45454545454545453,
                "ranking": [
                    {"model_id": 33, "model_name": "Grok 4.6", "p": 0.91, "price": 0.0026, "score": 0.8987},
                    {"model_id": 31, "model_name": "Claude Sonnet 5", "p": 0.95, "price": 0.0096, "score": 0.8903},
                ],
                "features_cost_usd": 0.0412,
            }
        }
    )

    decision_id: int
    lambda_used: float
    ranking: list[RankingEntry]
    features_cost_usd: float
