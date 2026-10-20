"""Catalog import (tasks.jsonl, questions.json, models.json) into Postgres.

All input is JSON (standardized). The 3 endpoints receive a file (multipart)
and return how many rows were created/deactivated. The shape and accepted
values of each row are documented in `app.schemas` (TaskImportRow,
QuestionImportRow, ModelImportRow) and are validated before inserting: an
invalid row aborts the whole import with 422 and the error detail. All 3 are
a replacement with soft delete (cascade included via app.services.cascade):
they deactivate what exists and activate only what came in the latest
upload, never touching a row already used by a previous experiment.
Additionally, each one leaves a snapshot of the resulting active state in
data/current (rotating the previous one to data/previous, see
app.services.snapshots).
"""

import json

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import ValidationError
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db.models import Model, Question
from app.schemas import CatalogReplaceResult, ModelImportRow, QuestionImportRow
from app.services import cascade
from app.services.snapshots import row_to_dict, write_snapshot
from app.services.tasks_catalog import parse_tasks_jsonl, replace_tasks

router = APIRouter(tags=["catalogs"])


def _validation_error(item: str, exc: ValidationError) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={"item": item, "errors": exc.errors(include_url=False, include_context=False)},
    )


def _parse_json_object(raw: str) -> dict | list:
    """Parses the body of an import .json file. 400 if it's not valid JSON."""
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc


@router.post(
    "/tasks/import",
    summary="Replaces ALL active tasks from a tasks.jsonl file provided by the user",
    response_model=CatalogReplaceResult,
)
async def import_tasks(
    file: UploadFile = File(description="tasks.jsonl: one JSON task per line."),
    db: Session = Depends(get_db),
) -> dict:
    """Replaces (with soft delete) ALL active tasks in `routerkit.tasks`
    with the ones in this file.

    On startup, `routerkit-api` already auto-loads the official BigCodeBench
    catalog (1128 tasks) if the table is empty (see `app.services.seed`) —
    this endpoint is for when the user wants to replace that catalog with
    their own. Each line decides its own type via `bcb_task_id`: with a
    value (e.g. "BigCodeBench/12") it's an official benchmark task, without
    one it's a custom task — a single file can freely mix both. No API
    downloads the BigCodeBench dataset on its own in this endpoint: if you
    want benchmark tasks here, the jsonl must bring them with their
    `bcb_task_id`.

    Deactivates (`active=false`) ALL existing tasks (of any type) and
    creates new rows with `active=true`; it never touches a row already
    used by a previous experiment.

    Each line must satisfy `TaskImportRow` (`prompt` required; `tests`,
    `entry_point`, `source`, `tags`, `bcb_task_id`, `custom_task_id`
    optional). If a custom task will later be referenced from
    `features`/`solutions`/`validations` (`type=load`), assign it a
    `custom_task_id` here — it's the stable key for custom tasks, analogous
    to `bcb_task_id` for benchmark ones; the internal Postgres `task_id` is
    never used for that (see `app.services.tasks_catalog`). Example:

    ```json
    {"prompt": "Write a function that adds two integers.", "tests": "assert add(2, 3) == 5", "entry_point": "add", "source": "custom", "tags": ["math", "easy"], "custom_task_id": "add-two-integers"}
    ```
    """
    raw = (await file.read()).decode("utf-8")
    rows = parse_tasks_jsonl(raw)
    return replace_tasks(db, rows)


@router.post(
    "/questions/import",
    summary="Replaces the features question catalog from questions.json",
    response_model=CatalogReplaceResult,
)
async def import_questions(
    file: UploadFile = File(description="questions.json: list of questions."),
    db: Session = Depends(get_db),
) -> dict:
    """Replaces (with soft delete) the question catalog: deactivates
    (`active=false`) ALL existing questions and creates new rows for the
    just-uploaded file, with `active=true`. Same pattern as reloading
    tasks: only the questions from the latest upload remain active.

    Each entry must satisfy `QuestionImportRow`: `text` and `kind` are
    required — `kind` only accepts `binary` (yes/no, 0-1) or `numeric`
    (integer or decimal). `lo`/`hi` bound the expected range of the answer.

    `script` is optional: if omitted, the question is answered by the LLM
    (the single active model with role='feature'); if provided, it must be
    the full text of a Python function with a fixed signature
    `def task_func(task_prompt: str) -> float` (same function name
    BigCodeBench uses as entry_point) — it's executed directly with the
    task's prompt as its only input, without calling any LLM (zero cost).

    `question_key` is optional but recommended: it's the stable key (e.g.
    "p41") to reference this question later from `features.jsonl`
    (`type=load`) — never the internal Postgres `question_id`, which
    changes on every reload of this catalog.

    ```json
    {
      "questions": [
        {"text": "Does the solution use recursion?", "kind": "binary", "lo": 0, "hi": 1, "author": "marcos", "version": "v1", "question_key": "p1"},
        {
          "text": "How many lines does the task prompt have?",
          "kind": "numeric",
          "lo": 1,
          "hi": 200,
          "version": "v1",
          "question_key": "p2",
          "script": "def task_func(task_prompt: str) -> float:\\n    return float(len(task_prompt.splitlines()))"
        }
      ]
    }
    ```
    """
    raw = (await file.read()).decode("utf-8")
    data = _parse_json_object(raw)
    raw_rows = data.get("questions", []) if isinstance(data, dict) else data

    rows: list[QuestionImportRow] = []
    for index, raw_row in enumerate(raw_rows):
        try:
            rows.append(QuestionImportRow.model_validate(raw_row))
        except ValidationError as exc:
            raise _validation_error(f"questions[{index}]", exc) from exc

    deactivated_ids = list(
        db.execute(select(Question.question_id).where(Question.active.is_(True))).scalars()
    )
    if deactivated_ids:
        db.execute(update(Question).where(Question.question_id.in_(deactivated_ids)).values(active=False))
        cascade.deactivate_for_questions(db, deactivated_ids)

    for row in rows:
        db.add(
            Question(
                text=row.text,
                kind=row.kind,
                lo=row.lo,
                hi=row.hi,
                author=row.author,
                version=row.version,
                question_key=row.question_key,
                active=True,
                script=row.script,
                repetitions=row.repetitions,
            )
        )

    db.commit()

    active_questions = db.query(Question).filter(Question.active.is_(True)).all()
    write_snapshot("questions", [row_to_dict(q) for q in active_questions])

    return {"deactivated_previous": len(deactivated_ids), "created": len(rows)}


@router.post(
    "/models/import",
    summary="Replaces the candidate/judge model catalog from models.json",
    response_model=CatalogReplaceResult,
)
async def import_models(
    file: UploadFile = File(description="models.json: list of models."),
    db: Session = Depends(get_db),
) -> dict:
    """Replaces (with soft delete) the model catalog: deactivates
    (`active=false`) ALL existing models and creates new rows for the
    just-uploaded file, with `active=true`. Same pattern as
    tasks/questions — includes cascade: deactivating a model also
    deactivates its `solutions` (and the `grades` of those solutions),
    `targets`, and `model_metrics`.

    Each entry must satisfy `ModelImportRow`: `type` only accepts `api`
    (called over HTTP to an external provider) or `hosted` (billed by its
    own compute time); `role` accepts `candidate` (routed to/generates
    solutions — several can be active at once, but never two with the same
    `name`), `feature` (computes features via LLM), or `judge` (grades
    solutions of custom tasks) — of these last two, only one can be active
    in total. `api_key_env` is optional: if omitted,
    `app.clients.llm_client` resolves the key by a fixed convention based on
    `provider` (see `.env.example`) — `OPENAI_API_KEY`,
    `AZURE_OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, or
    `CUSTOM_LLM_API_KEY`. If set, it points to a different environment
    variable of your own (useful so different models can use different
    accounts of the same provider). `concurrency` (optional, unlimited if
    omitted), `timeout_seconds` (default 300), and `temperature` (optional,
    provider default if omitted) control how the model is called —
    `temperature` is per-model config, not per-request. `concurrency`
    matters for the model with role='feature', since its responses need
    logprobs (not all providers expose them, e.g. Anthropic doesn't).

    ### Recognized `provider` values (see `app.clients.llm_client._LITELLM_PREFIXES`)

    - `openai` / `anthropic` / `openrouter`: native call to the provider,
      always Chat Completions.
    - `azure`: Azure OpenAI deployment (`endpoint` = the resource's base
      URL). Most models respond via Chat Completions, but some deployments
      (typically newer "reasoning" models) **only** expose the Responses
      API — for those, `api_kind="responses"` (see below).
    - `azure_anthropic`: Claude served from **Azure AI Foundry** (not the
      same as plain `anthropic`): litellm routes it differently (`azure_ai/`
      prefix), `endpoint` must point to the resource's
      `.../services.ai.azure.com/anthropic` sub-path. It's still
      `api_kind="chat_completions"` (Anthropic Messages, not the Responses
      API).
    - Any other value (e.g. a self-hosted gateway of your own): treated as
      an OpenAI-compatible custom endpoint (`endpoint` = api_base,
      `deployment`/`name` as-is). If that gateway carries no authentication
      at all, add the provider's name to
      `app.clients.llm_client._NO_AUTH_PROVIDERS` — `api_key` is never sent
      to it, regardless of `api_key_env` or `CUSTOM_LLM_API_KEY`.

    ### `api_kind`: which litellm method is used to call it

    - `chat_completions` (default): `litellm.completion()` — classic Chat
      Completions, what almost all the examples below already use.
    - `responses`: `litellm.responses()` — OpenAI's Responses API. Use it
      **only** when the specific deployment doesn't support Chat
      Completions (it's not a preference, it's a deployment limitation —
      see the `GPT-5.6 Sol` example below). Verified against a real
      deployment that needed it (see
      `app.clients.llm_client._complete_responses`).

    `max_tokens` is optional; some APIs require it (e.g. Anthropic Messages
    absolutely needs `max_tokens` — see the `Claude Sonnet 5` example
    below).

    ```json
    {
      "models": [
        {
          "name": "gpt-4o-mini",
          "type": "api",
          "provider": "openai",
          "endpoint": "https://api.openai.com/v1",
          "in_token_price": 0.15,
          "out_token_price": 0.60,
          "role": "candidate"
        },
        {
          "name": "GPT-5.6 Sol",
          "type": "api",
          "provider": "azure",
          "deployment": "gpt-5.6-sol",
          "endpoint": "https://<tu-recurso>.openai.azure.com",
          "api_key_env": "AZURE_FOUNDRY_API_KEY",
          "api_kind": "responses",
          "in_token_price": 5.00,
          "out_token_price": 30.00,
          "role": "candidate"
        },
        {
          "name": "Claude Sonnet 5",
          "type": "api",
          "provider": "azure_anthropic",
          "deployment": "claude-sonnet-5",
          "endpoint": "https://<tu-recurso>.services.ai.azure.com/anthropic",
          "api_key_env": "AZURE_FOUNDRY_API_KEY",
          "max_tokens": 4096,
          "role": "candidate"
        },
        {
          "name": "Qwen3.6-35B-A3B-FP8",
          "type": "hosted",
          "provider": "spheron",
          "deployment": "qwen3.6-fp8",
          "endpoint": "https://<spheron-gateway>",
          "cost_per_second": 0.00061389,
          "role": "candidate"
        },
        {
          "name": "gpt-5.4-nano",
          "type": "api",
          "provider": "azure",
          "deployment": "gpt-5.4-nano",
          "endpoint": "https://<tu-recurso>.openai.azure.com",
          "api_key_env": "AZURE_FOUNDRY_API_KEY",
          "concurrency": 5,
          "role": "feature"
        }
      ]
    }
    ```
    """
    raw = (await file.read()).decode("utf-8")
    data = _parse_json_object(raw)
    raw_rows = data.get("models", []) if isinstance(data, dict) else data

    rows: list[ModelImportRow] = []
    for index, raw_row in enumerate(raw_rows):
        try:
            rows.append(ModelImportRow.model_validate(raw_row))
        except ValidationError as exc:
            raise _validation_error(f"models[{index}]", exc) from exc

    deactivated_ids = list(db.execute(select(Model.model_id).where(Model.active.is_(True))).scalars())
    if deactivated_ids:
        db.execute(update(Model).where(Model.model_id.in_(deactivated_ids)).values(active=False))
        cascade.deactivate_for_models(db, deactivated_ids)

    for row in rows:
        db.add(
            Model(
                name=row.name,
                type=row.type,
                provider=row.provider,
                endpoint=row.endpoint,
                deployment=row.deployment,
                api_key_env=row.api_key_env,
                in_token_price=row.in_token_price,
                out_token_price=row.out_token_price,
                cost_per_second=row.cost_per_second,
                concurrency=row.concurrency,
                timeout_seconds=row.timeout_seconds,
                temperature=row.temperature,
                api_kind=row.api_kind,
                max_tokens=row.max_tokens,
                role=row.role,
                active=True,
            )
        )

    db.commit()

    active_models = db.query(Model).filter(Model.active.is_(True)).all()
    write_snapshot("models", [row_to_dict(m) for m in active_models])

    return {"deactivated_previous": len(deactivated_ids), "created": len(rows)}
