from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.clients import bigcodebench_client
from app.clients.bigcodebench_client import BigCodeBenchError
from app.db.base import get_db
from app.db.models import Grade, Model, Solution, Task
from app.services import jobs, judge
from app.services.concurrency import run_bounded
from app.services.jobs import JobConflictError, ProgressFn
from app.services.snapshots import row_to_dict, write_snapshot

router = APIRouter(tags=["evaluation"])

_KIND = "solutions_validate"


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
    """Reemplaza el grade de esta solution (a lo sumo 1, es su PK)."""
    existing = db.get(Grade, solution_id)
    if existing:
        db.delete(existing)
        db.flush()
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


def _find_active_solution(db: Session, *, task_id: int, model_name: str, sample_idx: int) -> Solution | None:
    model = (
        db.query(Model)
        .filter(Model.name == model_name, Model.role == "candidate", Model.active.is_(True))
        .one_or_none()
    )
    if model is None:
        return None
    return (
        db.query(Solution)
        .filter(
            Solution.task_id == task_id,
            Solution.model_id == model.model_id,
            Solution.sample_idx == sample_idx,
            Solution.active.is_(True),
        )
        .one_or_none()
    )


class GradeLoadRow(BaseModel):
    """Un grade ya calculado afuera, para cargar directo (type=load)."""

    model_name: str
    task_id: int
    sample_idx: int = 0
    grader: Literal["tests", "llm_judge"]
    passed: bool | None = None
    fail_reason: str | None = None
    exec_ms: float | None = None
    has_error: bool = False


class ValidateRequest(BaseModel):
    type: Literal["load", "calculate"] = "calculate"
    grades: list[GradeLoadRow] | None = Field(
        default=None, description="Requerido con type=load; ignorado con type=calculate."
    )


def _validate_solutions_job(db: Session, progress: ProgressFn, body: ValidateRequest) -> dict:
    """Cuerpo real de /solutions/validate — corre en background (ver app.services.jobs)."""
    graded = 0
    failed = 0
    errors: list[dict] = []

    if body.type == "load":
        rows = body.grades or []
        total = len(rows)

        for i, row in enumerate(rows, start=1):
            solution = _find_active_solution(
                db, task_id=row.task_id, model_name=row.model_name, sample_idx=row.sample_idx
            )
            if solution is None:
                failed += 1
                errors.append(
                    {
                        "model_name": row.model_name,
                        "task_id": row.task_id,
                        "sample_idx": row.sample_idx,
                        "error": "no hay ninguna solution activa para esa (model_name, task_id, sample_idx)",
                    }
                )
                progress(i, total, f"{i}/{total} procesadas")
                continue

            _store_grade(
                db,
                solution.solution_id,
                grader=row.grader,
                passed=row.passed,
                fail_reason=row.fail_reason,
                exec_ms=row.exec_ms,
                has_error=row.has_error,
            )
            graded += 1
            if row.has_error:
                failed += 1
                errors.append(
                    {
                        "model_name": row.model_name,
                        "task_id": row.task_id,
                        "sample_idx": row.sample_idx,
                        "error": row.fail_reason,
                    }
                )
            progress(i, total, f"{i}/{total} procesadas")
    else:
        solutions = (
            db.query(Solution).filter(Solution.active.is_(True), Solution.code.isnot(None)).all()
        )
        task_ids = {s.task_id for s in solutions}
        tasks_by_id = {t.task_id: t for t in db.query(Task).filter(Task.task_id.in_(task_ids)).all()}

        benchmark: list[tuple[Task, Solution]] = []
        custom: list[tuple[Task, Solution]] = []
        for solution in solutions:
            task = tasks_by_id.get(solution.task_id)
            if task is None:
                continue
            (benchmark if task.bcb_task_id else custom).append((task, solution))

        total = len(benchmark) + len(custom)
        processed = 0

        # Benchmark: secuencial (ya es un job async de fondo, no hace falta
        # paralelizar contra bigcodebench-api — ver conversación).
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
                errors.append({"solution_id": solution.solution_id, "error": str(exc)})
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
            progress(processed, total, f"{processed}/{total} procesadas (benchmark)")

        # Custom: concurrente, hasta judge_model.concurrency llamadas en vuelo.
        judge_model = db.query(Model).filter(Model.role == "judge", Model.active.is_(True)).one_or_none()
        if judge_model is None:
            for task, solution in custom:
                failed += 1
                errors.append(
                    {
                        "solution_id": solution.solution_id,
                        "error": "no hay ningún model con role='judge' activo",
                    }
                )
                processed += 1
                progress(processed, total, f"{processed}/{total} procesadas (custom)")
        else:
            judge_results = run_bounded(
                custom,
                lambda pair: judge.grade_with_llm_judge(pair[0], pair[1], judge_model),
                judge_model.concurrency,
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
                    errors.append({"solution_id": solution.solution_id, "error": str(exc)})
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
                progress(processed, total, f"{processed}/{total} procesadas (custom)")

    active_grades = db.query(Grade).filter(Grade.active.is_(True)).all()
    write_snapshot("grades", [row_to_dict(g) for g in active_grades])

    return {"graded": graded, "failed": failed, "errors": errors}


@router.post("/solutions/validate", status_code=202)
def validate_solutions(body: ValidateRequest, db: Session = Depends(get_db)) -> dict:
    """Lanza (async) la carga (`type=load`) o cálculo (`type=calculate`) de
    grades para las solutions activas. Devuelve enseguida el estado inicial
    — para ver el progreso o el resultado final, consultar `GET /jobs`
    (estado de TODOS los kinds en un solo request, incluido este). 409 si
    ya hay un job de `kind='solutions_validate'` corriendo.

    - `type=calculate` (default, sin parámetros adicionales): toma TODAS
      las solutions activas con código generado, las agrupa por
      `task.bcb_task_id` (benchmark) vs custom, y las evalúa:
      - benchmark -> bigcodebench-api, **secuencial** (ya es un job async
        de fondo, no bloquea a nadie, así que no vale la pena paralelizar
        contra un servicio que además ya tiene su propio límite interno).
      - custom -> el único model con `role='judge'` activo, con hasta
        `judge_model.concurrency` llamadas en vuelo a la vez (ver
        services.concurrency.run_bounded).
      Un fallo de infraestructura (bigcodebench-api caído, timeout, falla
      del juez) se guarda con `has_error=true` en el grade — DISTINTO de
      que el evaluador haya determinado `passed=false` — y no aborta el
      resto del lote.
    - `type=load`: inserta directo los grades de `grades`, sin llamar a
      bigcodebench-api ni al juez; cada fila identifica su solution por
      `(model_name, task_id, sample_idx)`.

    Reemplazo en el lugar: `grades` tiene PK=solution_id (1:1 con la
    solution), así que reevaluar una solution simplemente reemplaza su
    grade anterior. Al terminar deja un snapshot de los grades activos
    resultantes en data/current/grades.json (rotando el anterior a
    data/previous).
    """
    if body.type == "load" and not body.grades:
        raise HTTPException(status_code=422, detail="type=load requiere 'grades'")

    try:
        job = jobs.start_job(_KIND, lambda db, progress: _validate_solutions_job(db, progress, body), db)
    except JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return jobs.to_board_entry(job, _KIND)
