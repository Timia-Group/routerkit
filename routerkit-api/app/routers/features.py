from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import insert, text, update
from sqlalchemy.orm import Session

from app.clients.llm_client import LlmClientError
from app.db.base import get_db
from app.db.models import Feature, Model, Question, Task
from app.schemas import JobBoardEntry
from app.services import features_llm, jobs
from app.services.concurrency import run_bounded
from app.services.features_python import ScriptError, compute_python_feature
from app.services.jobs import JobConflictError, ProgressFn
from app.services.jsonl import parse_jsonl
from app.services.snapshots import row_to_dict, write_snapshot
from app.services.tasks_catalog import (
    TaskIdentifier,
    load_active_tasks_by_identifier,
    resolve_task,
    task_ref,
)

router = APIRouter(tags=["features"])

_KIND = "features_compute"
# After this many consecutive LlmClientError's (each already past its own
# internal retries in llm_client), the feature model is assumed to be down
# for now — remaining not-yet-started (task, question) pairs are skipped
# rather than each independently waiting out the same retries to confirm
# it again (see services.concurrency.run_bounded's circuit_breaker).
_CIRCUIT_BREAKER_THRESHOLD = 3


def _bulk_deactivate_features(db: Session, keys: list[tuple[int, int]]) -> None:
    """Deactivates the active feature (if any) for each (task_id,
    question_id) in `keys`, in one statement via `unnest` instead of one
    UPDATE per row (slow at scale) or a single `.in_(...)` over a flat list
    of ids (hits Postgres's ~65535 bound-parameter limit once `keys` is in
    the hundreds of thousands — see _write_grades_snapshot's docstring in
    evaluation.py for the incident this same shape caused there). Necessary
    before inserting new ones: there can only be one active row per
    (task_id, question_id) (idx_features_natural_key_active).
    """
    if not keys:
        return
    db.execute(
        text(
            "UPDATE routerkit.features f SET active = false "
            "FROM unnest(CAST(:task_ids AS BIGINT[]), CAST(:question_ids AS BIGINT[])) "
            "AS t(task_id, question_id) "
            "WHERE f.task_id = t.task_id AND f.question_id = t.question_id AND f.active"
        ),
        {"task_ids": [k[0] for k in keys], "question_ids": [k[1] for k in keys]},
    )


def _write_features_snapshot(db: Session) -> None:
    """Writes the active-features snapshot with human-facing identifiers
    (`bcb_task_id`/`custom_task_id`, `question_key`) instead of the raw
    `task_id`/`question_id` foreign keys — this file is meant to be read
    directly off disk, and those internal ids aren't stable/meaningful to
    a person (see app.services.tasks_catalog)."""
    active_features = db.query(Feature).filter(Feature.active.is_(True)).all()
    task_ids = {f.task_id for f in active_features}
    question_ids = {f.question_id for f in active_features}
    tasks_by_id = {t.task_id: t for t in db.query(Task).filter(Task.task_id.in_(task_ids)).all()}
    questions_by_id = {
        q.question_id: q for q in db.query(Question).filter(Question.question_id.in_(question_ids)).all()
    }

    def _row(f: Feature) -> dict:
        task = tasks_by_id.get(f.task_id)
        question = questions_by_id.get(f.question_id)
        extra = {"question_key": question.question_key if question else None}
        if task is not None:
            extra.update(task_ref(task))
        return row_to_dict(f, drop={"task_id", "question_id"}, extra=extra)

    write_snapshot("features", [_row(f) for f in active_features])


class FeatureLoadRow(TaskIdentifier):
    """A feature already computed externally, to load directly (type=load).

    The task is identified by `bcb_task_id` (benchmark tasks) or
    `custom_task_id` (custom tasks) — exactly one of the two. Never the
    internal Postgres `task_id` (see `app.services.tasks_catalog`). The
    question is identified by `question_key` (see `QuestionImportRow`),
    never by the internal `question_id`.
    """

    question_key: str
    source: Literal["llm", "python"]
    value: float | None = None
    answer_text: str | None = None
    answer_logprobs: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    gen_error: str | None = None
    has_error: bool = False


def _load_features_job(db: Session, progress: ProgressFn, rows: list[FeatureLoadRow]) -> dict:
    total = len(rows)
    errors: list[dict] = []

    # Bulk-resolve instead of one find_active_task()/question query per row
    # — at a few hundred thousand rows (a real type=load, not our small
    # test files), that's the difference between seconds and the better
    # part of an hour.
    tasks_by_bcb, tasks_by_custom = load_active_tasks_by_identifier(db)
    questions_by_key = {q.question_key: q for q in db.query(Question).filter(Question.active.is_(True)).all()}

    # Keyed by natural identity so a later duplicate row for the same
    # (task, question) in the file simply overwrites the earlier one in
    # this dict — same "last one wins" result as processing sequentially,
    # but safe to insert in one batch afterwards (a batch INSERT can't
    # contain two rows that violate the same partial unique index).
    to_insert: dict[tuple[int, int], dict] = {}

    for i, row in enumerate(rows, start=1):
        task = resolve_task(
            tasks_by_bcb, tasks_by_custom, bcb_task_id=row.bcb_task_id, custom_task_id=row.custom_task_id
        )
        question = questions_by_key.get(row.question_key)
        if task is None or question is None:
            errors.append(
                {
                    "bcb_task_id": row.bcb_task_id,
                    "custom_task_id": row.custom_task_id,
                    "question_key": row.question_key,
                    "error": (
                        "no active task with that bcb_task_id/custom_task_id"
                        if task is None
                        else f"no active question with question_key={row.question_key!r}"
                    ),
                }
            )
            continue

        to_insert[(task.task_id, question.question_id)] = {
            "task_id": task.task_id,
            "question_id": question.question_id,
            "source": row.source,
            "value": row.value,
            "answer_text": row.answer_text,
            "answer_logprobs": row.answer_logprobs,
            "input_tokens": row.input_tokens,
            "output_tokens": row.output_tokens,
            "latency_ms": row.latency_ms,
            "cost_usd": row.cost_usd,
            "gen_error": row.gen_error,
            "has_error": row.has_error,
        }
        if i % jobs.PROGRESS_BATCH_SIZE == 0 or i == total:
            progress(i, total, f"resolved {i}/{total}")

    _bulk_deactivate_features(db, list(to_insert.keys()))
    db.commit()
    if to_insert:
        db.execute(insert(Feature), list(to_insert.values()))
        db.commit()

    progress(total, total, f"{total}/{total} processed")
    _write_features_snapshot(db)

    return {"computed": len(to_insert), "errors": errors}


def _calculate_features_job(db: Session, progress: ProgressFn) -> dict:
    tasks = db.query(Task).filter(Task.active.is_(True)).all()
    questions = db.query(Question).filter(Question.active.is_(True)).all()

    feature_model = db.query(Model).filter(Model.role == "feature", Model.active.is_(True)).one_or_none()

    db.execute(update(Feature).where(Feature.active.is_(True)).values(active=False))

    pairs = [(task, question) for task in tasks for question in questions]
    python_pairs = [(task, question) for task, question in pairs if question.script]
    llm_pairs = [(task, question) for task, question in pairs if not question.script]

    total = len(pairs)
    processed = 0
    computed = 0
    errors: list[dict] = []

    # Python: sequential, no thread pool.
    for task, question in python_pairs:
        try:
            result = compute_python_feature(task, question)
        except ScriptError as exc:
            db.add(
                Feature(
                    task_id=task.task_id,
                    question_id=question.question_id,
                    source="python",
                    cost_usd=0,
                    gen_error=str(exc),
                    has_error=True,
                )
            )
            errors.append({**task_ref(task), "question_key": question.question_key, "error": str(exc)})
        else:
            db.add(
                Feature(
                    task_id=task.task_id,
                    question_id=question.question_id,
                    source="python",
                    value=result.value,
                    latency_ms=result.latency_ms,
                    cost_usd=0,
                )
            )
            computed += 1
        processed += 1
        progress(processed, total, f"{processed}/{total} processed")

    # LLM: concurrent, up to feature_model.concurrency calls in flight.
    if feature_model is None:
        for task, question in llm_pairs:
            error = "question without a script and no active model with role='feature'"
            db.add(
                Feature(
                    task_id=task.task_id,
                    question_id=question.question_id,
                    source="llm",
                    gen_error=error,
                    has_error=True,
                )
            )
            errors.append({**task_ref(task), "question_key": question.question_key, "error": error})
            processed += 1
            progress(processed, total, f"{processed}/{total} processed")
    else:
        llm_results = run_bounded(
            llm_pairs,
            lambda pair: features_llm.compute_llm_feature(pair[0], pair[1], feature_model),
            feature_model.concurrency,
            circuit_breaker=(LlmClientError, _CIRCUIT_BREAKER_THRESHOLD),
        )
        for (task, question), result, exc in llm_results:
            if exc is not None:
                db.add(
                    Feature(
                        task_id=task.task_id,
                        question_id=question.question_id,
                        source="llm",
                        gen_error=str(exc),
                        has_error=True,
                    )
                )
                errors.append({**task_ref(task), "question_key": question.question_key, "error": str(exc)})
            else:
                db.add(
                    Feature(
                        task_id=task.task_id,
                        question_id=question.question_id,
                        source="llm",
                        value=result.value,
                        answer_text=result.answer_text,
                        answer_logprobs=result.answer_logprobs,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        latency_ms=result.latency_ms,
                        cost_usd=result.cost_usd,
                    )
                )
                computed += 1
            processed += 1
            progress(processed, total, f"{processed}/{total} processed")

    _write_features_snapshot(db)

    return {"computed": computed, "errors": errors}


@router.post("/features/compute", status_code=202, response_model=JobBoardEntry)
async def compute_features(
    type: Literal["load", "calculate"] = "calculate",  # noqa: A002 - domain name
    file: UploadFile | None = File(
        default=None,
        description="JSONL (one FeatureLoadRow per line). Required with type=load.",
    ),
    db: Session = Depends(get_db),
) -> dict:
    """Launches (async) the load (`type=load`) or calculation
    (`type=calculate`) of features. Returns the initial state right away —
    to see progress or the final result, check `GET /jobs` (state of ALL
    kinds in a single request, including this one). 409 if a
    `kind='features_compute'` job is already running.

    - `type=calculate` (default, no `file`): for ALL active tasks x ALL
      active questions. Deactivates ALL existing active features and each
      pair computed successfully is inserted as a new active row (there can
      only be one active row per (task_id, question_id), see
      idx_features_natural_key_active). Pairs are grouped by method:
      questions with `question.script` are executed in Python,
      sequentially (no LLM, no benefit from parallelizing a local exec());
      those without a script are resolved by the single active model with
      role='feature', with up to `feature_model.concurrency` calls in
      flight at a time (see services.concurrency.run_bounded). An error
      (from the script, the LLM, or the provider call) still inserts a row
      (`has_error=true`, `gen_error` with the message, `value=NULL` — same
      convention as `solutions`/`grades`) and is reported in `errors`,
      without aborting the rest of the batch.
    - `type=load`: requires `file` (multipart), a `.jsonl` with one
      `FeatureLoadRow` per line — without calling any LLM or running any
      script. Each line identifies its task by `bcb_task_id` (benchmark) or
      `custom_task_id` (custom), and its question by `question_key` —
      in-place replacement: it only deactivates the previous active row of
      that same identity, not all existing features. If the task or the
      question doesn't exist (or isn't active), that line is reported in
      `errors` without aborting the rest of the batch.

    In both cases, when finished it leaves a snapshot of the resulting
    active features in data/current/features.json (rotating the previous
    one to data/previous).
    """
    if type == "load":
        if file is None:
            raise HTTPException(status_code=422, detail="type=load requires 'file'")
        raw = (await file.read()).decode("utf-8")
        rows = parse_jsonl(raw, FeatureLoadRow)
        target = lambda db, progress: _load_features_job(db, progress, rows)  # noqa: E731
    else:
        target = _calculate_features_job

    try:
        job = jobs.start_job(_KIND, target, db)
    except JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return jobs.to_board_entry(job, _KIND)
