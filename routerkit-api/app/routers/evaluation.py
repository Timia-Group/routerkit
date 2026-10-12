from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import insert, text, update
from sqlalchemy.orm import Session

from app.clients import bigcodebench_client
from app.clients.bigcodebench_client import BigCodeBenchError
from app.clients.llm_client import LlmClientError
from app.db.base import get_db
from app.db.models import Grade, Model, Solution, Task
from app.schemas import JobBoardEntry
from app.services import jobs, judge
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

router = APIRouter(tags=["evaluation"])

_KIND = "solutions_validate"
# Same reasoning/scope as routers/solutions.py's own constant: after this
# many consecutive LlmClientError's from the judge model (not JudgeError —
# that's the judge responding with something unparseable, not a sign the
# service is unreachable), remaining not-yet-started gradings are skipped.
_CIRCUIT_BREAKER_THRESHOLD = 3


def _deactivate_active_grade(db: Session, *, solution_id: int) -> None:
    """Deactivates the active grade (if any) for this solution.

    Necessary before inserting a new one: there can only be one active row
    per solution_id (idx_grades_natural_key_active).
    """
    db.execute(
        update(Grade)
        .where(Grade.solution_id == solution_id, Grade.active.is_(True))
        .values(active=False)
    )


def _store_grade(
    db: Session,
    solution_id: int,
    *,
    grader: str,
    passed: bool | None,
    fail_reason: str | None,
    exec_ms: float | None,
    has_error: bool,
) -> None:
    """Replaces this solution's active grade (soft delete + insert, same
    pattern as solutions/features — see _deactivate_active_grade)."""
    _deactivate_active_grade(db, solution_id=solution_id)
    db.add(
        Grade(
            solution_id=solution_id,
            grader=grader,
            passed=passed,
            fail_reason=fail_reason,
            exec_ms=exec_ms,
            has_error=has_error,
        )
    )


def _load_active_candidates_by_name(db: Session) -> dict[str, Model]:
    """Bulk alternative to looking up a candidate by name once per row —
    all active candidates fit comfortably in memory (there's never more
    than a handful), unlike the rows referencing them."""
    candidates = db.query(Model).filter(Model.role == "candidate", Model.active.is_(True)).all()
    return {m.name: m for m in candidates}


def _load_active_solution_ids_by_identity(db: Session) -> dict[tuple[int, int, int], int]:
    """Bulk alternative to `_find_active_solution()`-style queries once per
    row: a single query for (task_id, model_id, sample_idx, solution_id) of
    every active solution — selecting only those 4 integer columns, not
    full Solution rows (which carry a `code` text column that would bloat
    memory for no reason here) — returned as a lookup dict. At a few
    hundred thousand rows (a real type=load of the project's experiment
    data), one query instead of one round trip per row is the difference
    between seconds and the better part of an hour."""
    rows = db.query(Solution.task_id, Solution.model_id, Solution.sample_idx, Solution.solution_id).filter(
        Solution.active.is_(True)
    )
    return {(task_id, model_id, sample_idx): solution_id for task_id, model_id, sample_idx, solution_id in rows}


def _bulk_deactivate_grades(db: Session, solution_ids: list[int]) -> None:
    """Deactivates the active grade (if any) for each solution_id, in one
    statement via `= ANY(...)` (a single array parameter) instead of one
    UPDATE per row (slow at scale) or a single `.in_(...)` over a flat list
    of ids (hits Postgres's ~65535 bound-parameter limit once `solution_ids`
    is in the hundreds of thousands — see _write_grades_snapshot's
    docstring for the incident this same shape caused). Necessary before
    inserting new ones: there can only be one active row per solution_id
    (idx_grades_natural_key_active).
    """
    if not solution_ids:
        return
    db.execute(
        text(
            "UPDATE routerkit.grades SET active = false "
            "WHERE solution_id = ANY(CAST(:solution_ids AS BIGINT[])) AND active"
        ),
        {"solution_ids": solution_ids},
    )


def _write_grades_snapshot(db: Session) -> None:
    """Writes the active-grades snapshot with human-facing identifiers
    (`bcb_task_id`/`custom_task_id`, `model_name`, `sample_idx`) instead of
    the raw `solution_id` — this file is meant to be read directly off
    disk, and that internal id isn't stable/meaningful to a person (see
    app.services.tasks_catalog).

    A single JOIN, not "fetch active grades, then filter Solution/Task/
    Model by their ids with .in_(...)": that second shape worked fine at
    our small test scale, but with hundreds of thousands of active grades
    (a real type=load of the project's experiment data) the solution_id
    .in_(...) alone blows past Postgres's ~65535 bound-parameter limit per
    query and the whole snapshot write fails.
    """
    rows = (
        db.query(Grade, Solution, Task, Model)
        .join(Solution, Solution.solution_id == Grade.solution_id)
        .join(Task, Task.task_id == Solution.task_id)
        .join(Model, Model.model_id == Solution.model_id)
        .filter(Grade.active.is_(True))
        .all()
    )

    def _row(grade: Grade, solution: Solution, task: Task, model: Model) -> dict:
        extra = {"model_name": model.name, "sample_idx": solution.sample_idx, **task_ref(task)}
        return row_to_dict(grade, drop={"grade_id", "solution_id"}, extra=extra)

    write_snapshot("grades", [_row(*row) for row in rows])


class GradeLoadRow(TaskIdentifier):
    """A grade already computed externally, to load directly (type=load).

    The task is identified by `bcb_task_id` (benchmark tasks) or
    `custom_task_id` (custom tasks) — exactly one of the two. Never the
    internal Postgres `task_id` (see `app.services.tasks_catalog`).
    """

    model_name: str
    sample_idx: int = 0
    grader: Literal["tests", "llm_judge"]
    passed: bool | None = None
    fail_reason: str | None = None
    exec_ms: float | None = None
    has_error: bool = False


def _load_grades_job(db: Session, progress: ProgressFn, rows: list[GradeLoadRow]) -> dict:
    failed = 0
    errors: list[dict] = []
    total = len(rows)

    # Bulk-resolve instead of one task/candidate/solution query per row —
    # at a few hundred thousand rows (a real type=load, not our small test
    # files), that's the difference between seconds and the better part of
    # an hour.
    tasks_by_bcb, tasks_by_custom = load_active_tasks_by_identifier(db)
    candidates_by_name = _load_active_candidates_by_name(db)
    solution_ids_by_identity = _load_active_solution_ids_by_identity(db)

    # Keyed by solution_id (grades' natural key) so a later duplicate row
    # for the same solution in the file simply overwrites the earlier one
    # in this dict — same "last one wins" result as processing
    # sequentially, but safe to insert in one batch afterwards (a batch
    # INSERT can't contain two rows that violate the same partial unique
    # index).
    to_insert: dict[int, dict] = {}
    # Mirrors to_insert key-for-key (None = no error) so a later duplicate
    # row overwrites this in lockstep too.
    error_refs: dict[int, dict | None] = {}

    for i, row in enumerate(rows, start=1):
        task = resolve_task(
            tasks_by_bcb, tasks_by_custom, bcb_task_id=row.bcb_task_id, custom_task_id=row.custom_task_id
        )
        model = candidates_by_name.get(row.model_name)
        solution_id = (
            solution_ids_by_identity.get((task.task_id, model.model_id, row.sample_idx))
            if task is not None and model is not None
            else None
        )
        if solution_id is None:
            failed += 1
            errors.append(
                {
                    "model_name": row.model_name,
                    "bcb_task_id": row.bcb_task_id,
                    "custom_task_id": row.custom_task_id,
                    "sample_idx": row.sample_idx,
                    "error": (
                        "no active task with that bcb_task_id/custom_task_id"
                        if task is None
                        else "no active solution for that (model_name, task_id, sample_idx)"
                    ),
                }
            )
            continue

        to_insert[solution_id] = {
            "solution_id": solution_id,
            "grader": row.grader,
            "passed": row.passed,
            "fail_reason": row.fail_reason,
            "exec_ms": row.exec_ms,
            "has_error": row.has_error,
        }
        error_refs[solution_id] = (
            {"model_name": row.model_name, **task_ref(task), "sample_idx": row.sample_idx, "error": row.fail_reason}
            if row.has_error
            else None
        )
        if i % jobs.PROGRESS_BATCH_SIZE == 0 or i == total:
            progress(i, total, f"resolved {i}/{total}")

    _bulk_deactivate_grades(db, list(to_insert.keys()))
    db.commit()
    if to_insert:
        db.execute(insert(Grade), list(to_insert.values()))
        db.commit()

    for ref in error_refs.values():
        if ref is not None:
            failed += 1
            errors.append(ref)

    progress(total, total, f"{total}/{total} processed")

    _write_grades_snapshot(db)

    return {"graded": len(to_insert), "failed": failed, "errors": errors}


def _calculate_grades_job(db: Session, progress: ProgressFn) -> dict:
    graded = 0
    failed = 0
    errors: list[dict] = []

    # Full replace, not targeted: this run's freshly computed grades are
    # meant to be the ONLY active ones afterwards (same reasoning as
    # solutions.py's _calculate_solutions_job) — deactivate everything
    # upfront instead of only the solution_ids this run happens to touch.
    db.execute(update(Grade).where(Grade.active.is_(True)).values(active=False))

    solutions = db.query(Solution).filter(Solution.active.is_(True), Solution.code.isnot(None)).all()
    task_ids = {s.task_id for s in solutions}
    model_ids = {s.model_id for s in solutions}
    tasks_by_id = {t.task_id: t for t in db.query(Task).filter(Task.task_id.in_(task_ids)).all()}
    models_by_id = {m.model_id: m for m in db.query(Model).filter(Model.model_id.in_(model_ids)).all()}

    def _solution_ref(solution: Solution) -> dict:
        model = models_by_id.get(solution.model_id)
        task = tasks_by_id.get(solution.task_id)
        ref = {"model_name": model.name if model else None, "sample_idx": solution.sample_idx}
        if task is not None:
            ref.update(task_ref(task))
        return ref

    benchmark: list[tuple[Task, Solution]] = []
    custom: list[tuple[Task, Solution]] = []
    for solution in solutions:
        task = tasks_by_id.get(solution.task_id)
        if task is None:
            continue
        (benchmark if task.bcb_task_id else custom).append((task, solution))

    total = len(benchmark) + len(custom)
    processed = 0

    # Benchmark: sequential (it's already a background async job, no need
    # to parallelize against bigcodebench-api — see conversation).
    for task, solution in benchmark:
        try:
            result = bigcodebench_client.evaluate(task.bcb_task_id, solution.code)
        except BigCodeBenchError as exc:
            _store_grade(
                db,
                solution.solution_id,
                grader="tests",
                passed=None,
                fail_reason=str(exc),
                exec_ms=None,
                has_error=True,
            )
            failed += 1
            errors.append({**_solution_ref(solution), "error": str(exc)})
        else:
            _store_grade(
                db,
                solution.solution_id,
                grader="tests",
                passed=result["passed"],
                fail_reason=result["fail_reason"],
                exec_ms=result["exec_ms"],
                has_error=False,
            )
        graded += 1
        processed += 1
        progress(processed, total, f"{processed}/{total} processed (benchmark)")

    # Custom: concurrent, up to judge_model.concurrency calls in flight.
    judge_model = db.query(Model).filter(Model.role == "judge", Model.active.is_(True)).one_or_none()
    if judge_model is None:
        for task, solution in custom:
            failed += 1
            errors.append({**_solution_ref(solution), "error": "no active model with role='judge'"})
            processed += 1
            progress(processed, total, f"{processed}/{total} processed (custom)")
    else:
        judge_results = run_bounded(
            custom,
            lambda pair: judge.grade_with_llm_judge(pair[0], pair[1], judge_model),
            judge_model.concurrency,
            circuit_breaker=(LlmClientError, _CIRCUIT_BREAKER_THRESHOLD),
        )
        for (task, solution), result, exc in judge_results:
            if exc is not None:
                _store_grade(
                    db,
                    solution.solution_id,
                    grader="llm_judge",
                    passed=None,
                    fail_reason=str(exc),
                    exec_ms=None,
                    has_error=True,
                )
                failed += 1
                errors.append({**_solution_ref(solution), "error": str(exc)})
            else:
                _store_grade(
                    db,
                    solution.solution_id,
                    grader="llm_judge",
                    passed=result.passed,
                    fail_reason=result.fail_reason,
                    exec_ms=result.exec_ms,
                    has_error=False,
                )
            graded += 1
            processed += 1
            progress(processed, total, f"{processed}/{total} processed (custom)")

    _write_grades_snapshot(db)

    return {"graded": graded, "failed": failed, "errors": errors}


@router.post("/solutions/validate", status_code=202, response_model=JobBoardEntry)
async def validate_solutions(
    type: Literal["load", "calculate"] = "calculate",  # noqa: A002 - domain name
    file: UploadFile | None = File(
        default=None,
        description="JSONL (one GradeLoadRow per line). Required with type=load.",
    ),
    db: Session = Depends(get_db),
) -> dict:
    """Launches (async) the load (`type=load`) or calculation
    (`type=calculate`) of grades for the active solutions. Returns the
    initial state right away — to see progress or the final result, check
    `GET /jobs` (state of ALL kinds in a single request, including this
    one). 409 if a `kind='solutions_validate'` job is already running.

    - `type=calculate` (default, no `file`): FULL replace — deactivates ALL
      existing active grades first, then takes ALL active solutions with
      generated code, groups them by `task.bcb_task_id` (benchmark) vs
      custom, and evaluates them:
      - benchmark -> bigcodebench-api, **sequential** (it's already a
        background async job, it doesn't block anyone, so it's not worth
        parallelizing against a service that also already has its own
        internal limit).
      - custom -> the single active model with `role='judge'`, with up to
        `judge_model.concurrency` calls in flight at a time (see
        services.concurrency.run_bounded).
      An infrastructure failure (bigcodebench-api down, timeout, judge
      failure) is saved with `has_error=true` in the grade — DIFFERENT from
      the evaluator having determined `passed=false` — and does not abort
      the rest of the batch. Only the grades computed by THIS run end up
      active — running it again regrades everything from scratch, it's
      never incremental.
    - `type=load`: TARGETED replace — requires `file` (multipart), a
      `.jsonl` with one `GradeLoadRow` per line — without calling
      bigcodebench-api or the judge; each line identifies its solution by
      `model_name` + `sample_idx` + its task (`bcb_task_id` for benchmark,
      or `custom_task_id` for custom). `grades` has PK=solution_id (1:1
      with the solution), so loading a row simply replaces that solution's
      previous grade, leaving the rest of the active catalog untouched.

    When finished it leaves a snapshot of the resulting active grades in
    data/current/grades.json (rotating the previous one to data/previous).
    """
    if type == "load":
        if file is None:
            raise HTTPException(status_code=422, detail="type=load requires 'file'")
        raw = (await file.read()).decode("utf-8")
        rows = parse_jsonl(raw, GradeLoadRow)
        target = lambda db, progress: _load_grades_job(db, progress, rows)  # noqa: E731
    else:
        target = _calculate_grades_job

    try:
        job = jobs.start_job(_KIND, target, db)
    except JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return jobs.to_board_entry(job, _KIND)
