from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import insert, text, update
from sqlalchemy.orm import Session

from app.clients.llm_client import LlmClientError
from app.db.base import get_db
from app.db.models import Grade, Model, Solution, Task
from app.schemas import JobBoardEntry
from app.services import generation, jobs
from app.services.concurrency import run_bounded
from app.services.jobs import JobConflictError, ProgressFn
from app.services.jsonl import parse_jsonl
from app.services.snapshots import row_to_dict, write_snapshot
from app.services.tasks_catalog import (
    TaskIdentifier,
    load_active_tasks_by_identifier,
    resolve_task,
    task_ref,
)

router = APIRouter(tags=["solutions"])

_KIND = "solutions_generate"
# After this many consecutive LlmClientError's for the SAME candidate model
# (each already past its own internal retries in llm_client), that model is
# assumed to be down for now — remaining not-yet-started (task, sample_idx)
# items for it are skipped rather than each independently waiting out the
# same retries to confirm it again (see services.concurrency.run_bounded's
# circuit_breaker — scoped per model, since this loop calls run_bounded
# once per candidate).
_CIRCUIT_BREAKER_THRESHOLD = 3

# There's no POST /solutions (manual single-row creation): seeding test
# solutions without going through generation.py is already covered by
# POST /solutions/generate type=load sending a single-element .jsonl —
# same bulk-only pattern as the rest of the system, without duplicating an
# individual-creation path nobody asked for (see evaluation.py: for the
# same reason there's also no POST /solutions/{id}/evaluate).


def _bulk_deactivate_solutions(db: Session, keys: list[tuple[int, int, int]]) -> None:
    """Deactivates the active solution (if any) for each (task_id,
    model_id, sample_idx) in `keys`, in one statement via `unnest` instead
    of one UPDATE per row (slow at scale) or a single `.in_(...)` over a
    flat list of ids (hits Postgres's ~65535 bound-parameter limit once
    `keys` is in the hundreds of thousands). Necessary before inserting new
    ones: there can only be one active row per (task_id, model_id,
    sample_idx) (idx_solutions_natural_key_active).
    """
    if not keys:
        return
    db.execute(
        text(
            "UPDATE routerkit.solutions s SET active = false "
            "FROM unnest(CAST(:task_ids AS BIGINT[]), CAST(:model_ids AS BIGINT[]), "
            "CAST(:sample_idxs AS INTEGER[])) AS t(task_id, model_id, sample_idx) "
            "WHERE s.task_id = t.task_id AND s.model_id = t.model_id "
            "AND s.sample_idx = t.sample_idx AND s.active"
        ),
        {
            "task_ids": [k[0] for k in keys],
            "model_ids": [k[1] for k in keys],
            "sample_idxs": [k[2] for k in keys],
        },
    )


def _load_active_candidates_by_name(db: Session) -> dict[str, Model]:
    """Bulk alternative to calling `_find_candidate()`-style lookups once
    per row — all active candidates fit comfortably in memory (there's
    never more than a handful), unlike the rows referencing them."""
    candidates = db.query(Model).filter(Model.role == "candidate", Model.active.is_(True)).all()
    return {m.name: m for m in candidates}


def _write_solutions_snapshot(db: Session) -> None:
    """Writes the active-solutions snapshot with human-facing identifiers
    (`bcb_task_id`/`custom_task_id`, `model_name`) instead of the raw
    `task_id`/`model_id` foreign keys — this file is meant to be read
    directly off disk, and those internal ids aren't stable/meaningful to
    a person (see app.services.tasks_catalog)."""
    active_solutions = db.query(Solution).filter(Solution.active.is_(True)).all()
    task_ids = {s.task_id for s in active_solutions}
    model_ids = {s.model_id for s in active_solutions}
    tasks_by_id = {t.task_id: t for t in db.query(Task).filter(Task.task_id.in_(task_ids)).all()}
    models_by_id = {m.model_id: m for m in db.query(Model).filter(Model.model_id.in_(model_ids)).all()}

    def _row(s: Solution) -> dict:
        task = tasks_by_id.get(s.task_id)
        model = models_by_id.get(s.model_id)
        extra = {"model_name": model.name if model else None}
        if task is not None:
            extra.update(task_ref(task))
        return row_to_dict(s, drop={"task_id", "model_id"}, extra=extra)

    write_snapshot("solutions", [_row(s) for s in active_solutions])


class SolutionLoadRow(TaskIdentifier):
    """A solution already computed externally, to load directly (type=load).

    The task is identified by `bcb_task_id` (benchmark tasks) or
    `custom_task_id` (custom tasks) — exactly one of the two. Never the
    internal Postgres `task_id` (see `app.services.tasks_catalog`).
    """

    model_name: str
    sample_idx: int = 0
    code: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    gen_error: str | None = None
    has_error: bool = False


def _load_solutions_job(db: Session, progress: ProgressFn, rows: list[SolutionLoadRow]) -> dict:
    failed = 0
    errors: list[dict] = []
    total = len(rows)

    # Bulk-resolve instead of one model/task query per row — at a few
    # hundred thousand rows (a real type=load, not our small test files),
    # that's the difference between seconds and the better part of an hour.
    candidates_by_name = _load_active_candidates_by_name(db)
    tasks_by_bcb, tasks_by_custom = load_active_tasks_by_identifier(db)

    # Keyed by natural identity so a later duplicate row for the same
    # (task, model, sample_idx) in the file simply overwrites the earlier
    # one in this dict — same "last one wins" result as processing
    # sequentially, but safe to insert in one batch afterwards (a batch
    # INSERT can't contain two rows that violate the same partial unique
    # index).
    to_insert: dict[tuple[int, int, int], dict] = {}
    # Mirrors to_insert key-for-key (None = no error) so a later duplicate
    # row overwrites this in lockstep too — never a stale error left over
    # from an earlier occurrence of the same key that a later one replaced.
    error_refs: dict[tuple[int, int, int], dict | None] = {}

    for i, row in enumerate(rows, start=1):
        model = candidates_by_name.get(row.model_name)
        task = resolve_task(
            tasks_by_bcb, tasks_by_custom, bcb_task_id=row.bcb_task_id, custom_task_id=row.custom_task_id
        )
        if model is None or task is None:
            failed += 1
            errors.append(
                {
                    "model_name": row.model_name,
                    "bcb_task_id": row.bcb_task_id,
                    "custom_task_id": row.custom_task_id,
                    "sample_idx": row.sample_idx,
                    "error": (
                        f"no active candidate model with name={row.model_name!r}"
                        if model is None
                        else "no active task with that bcb_task_id/custom_task_id"
                    ),
                }
            )
            continue

        has_error = row.has_error or row.gen_error is not None
        key = (task.task_id, model.model_id, row.sample_idx)
        to_insert[key] = {
            "task_id": task.task_id,
            "model_id": model.model_id,
            "sample_idx": row.sample_idx,
            "code": row.code,
            "tokens_in": row.tokens_in,
            "tokens_out": row.tokens_out,
            "latency_ms": row.latency_ms,
            "cost_usd": row.cost_usd,
            "gen_error": row.gen_error,
            "has_error": has_error,
        }
        error_refs[key] = (
            {"model_name": row.model_name, **task_ref(task), "sample_idx": row.sample_idx, "error": row.gen_error}
            if has_error
            else None
        )
        if i % jobs.PROGRESS_BATCH_SIZE == 0 or i == total:
            progress(i, total, f"resolved {i}/{total}")

    _bulk_deactivate_solutions(db, list(to_insert.keys()))
    db.commit()
    if to_insert:
        db.execute(insert(Solution), list(to_insert.values()))
        db.commit()

    for ref in error_refs.values():
        if ref is not None:
            failed += 1
            errors.append(ref)

    progress(total, total, f"{total}/{total} processed")
    _write_solutions_snapshot(db)

    return {"created": len(to_insert), "failed": failed, "errors": errors}


def _calculate_solutions_job(db: Session, progress: ProgressFn, n_samples: int) -> dict:
    created = 0
    failed = 0
    errors: list[dict] = []

    candidates = db.query(Model).filter(Model.role == "candidate", Model.active.is_(True)).all()
    tasks = db.query(Task).filter(Task.active.is_(True)).all()
    total = len(candidates) * len(tasks) * n_samples
    processed = 0

    # Full replace, not targeted: this run's freshly generated solutions are
    # meant to be the ONLY active ones afterwards — deactivate everything
    # upfront (same pattern as features.py's _calculate_features_job)
    # instead of only the (task, model, sample_idx) triples this run happens
    # to touch. Cascades to grades: a solution without an active grade makes
    # no sense once the solution itself is gone.
    db.execute(update(Solution).where(Solution.active.is_(True)).values(active=False))
    db.execute(update(Grade).where(Grade.active.is_(True)).values(active=False))

    for model in candidates:
        name = model.name
        items = [(task, sample_idx) for task in tasks for sample_idx in range(n_samples)]
        results = run_bounded(
            items,
            lambda item: generation.generate_solution(item[0], model),
            model.concurrency,
            circuit_breaker=(LlmClientError, _CIRCUIT_BREAKER_THRESHOLD),
        )

        for (task, sample_idx), result, exc in results:
            gen_error = str(exc) if exc is not None else None

            db.add(
                Solution(
                    task_id=task.task_id,
                    model_id=model.model_id,
                    sample_idx=sample_idx,
                    code=None if exc is not None else result.code,
                    tokens_in=None if exc is not None else result.tokens_in,
                    tokens_out=None if exc is not None else result.tokens_out,
                    latency_ms=None if exc is not None else result.latency_ms,
                    cost_usd=0 if exc is not None else result.cost_usd,
                    gen_error=gen_error,
                    has_error=gen_error is not None,
                )
            )
            created += 1
            if gen_error is not None:
                failed += 1
                errors.append(
                    {
                        "model_name": name,
                        **task_ref(task),
                        "sample_idx": sample_idx,
                        "error": gen_error,
                    }
                )
            processed += 1
            progress(processed, total, f"{processed}/{total} processed ({name})")

    _write_solutions_snapshot(db)

    return {"created": created, "failed": failed, "errors": errors}


@router.post("/solutions/generate", status_code=202, response_model=JobBoardEntry)
async def generate_solutions(
    type: Literal["load", "calculate"] = "calculate",  # noqa: A002 - domain name
    n_samples: int = Query(
        default=1, ge=1, description="Samples to generate per task (only applies to type=calculate)."
    ),
    file: UploadFile | None = File(
        default=None,
        description="JSONL (one SolutionLoadRow per line). Required with type=load.",
    ),
    db: Session = Depends(get_db),
) -> dict:
    """Launches (async) the load (`type=load`) or generation
    (`type=calculate`) of solutions for all active candidate models.
    Returns the initial state right away — to see progress or the final
    result, check `GET /jobs` (state of ALL kinds in a single request,
    including this one). 409 if a `kind='solutions_generate'` job is
    already running.

    - `type=calculate` (default, no `file`): FULL replace — deactivates ALL
      existing active solutions (and their grades, if any) first, then
      looks up ALL active models with `role='candidate'` and generates with
      `generation.generate_solution` for ALL active tasks x `n_samples`,
      for each one, with up to `model.concurrency` generations in flight at
      a time per model (see services.concurrency.run_bounded). Temperature
      is not a parameter of this request: `model.temperature` is used if
      configured. Only the solutions generated by THIS run end up active —
      running it again (even with the same candidates/tasks) regenerates
      everything from scratch, it's never incremental.
    - `type=load`: TARGETED replace — requires `file` (multipart), a
      `.jsonl` with one `SolutionLoadRow` per line — without calling any
      LLM; each line brings its own `model_name`, and its task via
      `bcb_task_id` (benchmark) or `custom_task_id` (custom) — a single
      upload can bring solutions for several different models/tasks. Only
      deactivates the previous active solution of that same (task, model,
      sample_idx) identity before inserting the new one, leaving the rest
      of the active catalog untouched.

    In both cases, an individual failure (failed generation, `model_name`
    that doesn't resolve to any active candidate, or a nonexistent task in
    `type=load`) does NOT abort the rest of the batch. When finished it
    leaves a snapshot of the resulting active solutions in
    data/current/solutions.json (rotating the previous one to
    data/previous).
    """
    if type == "load":
        if file is None:
            raise HTTPException(status_code=422, detail="type=load requires 'file'")
        raw = (await file.read()).decode("utf-8")
        rows = parse_jsonl(raw, SolutionLoadRow)
        target = lambda db, progress: _load_solutions_job(db, progress, rows)  # noqa: E731
    else:
        target = lambda db, progress: _calculate_solutions_job(db, progress, n_samples)  # noqa: E731

    try:
        job = jobs.start_job(_KIND, target, db)
    except JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return jobs.to_board_entry(job, _KIND)
