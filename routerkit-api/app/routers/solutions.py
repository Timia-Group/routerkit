from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import update
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db.models import Model, Solution, Task
from app.services import generation, jobs
from app.services.concurrency import run_bounded
from app.services.jobs import JobConflictError, ProgressFn
from app.services.snapshots import row_to_dict, write_snapshot

router = APIRouter(tags=["solutions"])

_KIND = "solutions_generate"

# No hay POST /solutions (alta manual de una sola fila): sembrar solutions
# de prueba sin pasar por generation.py ya se cubre con
# POST /solutions/generate type=load mandando una lista de un solo elemento
# — mismo patrón bulk-only que el resto del sistema, sin duplicar un camino
# de alta individual que nadie pidió (ver evaluation.py: por la misma razón
# tampoco existe un POST /solutions/{id}/evaluate).


def _deactivate_active_solution(db: Session, *, task_id: int, model_id: int, sample_idx: int) -> None:
    """Desactiva la solution activa (si existe) para esta identidad natural.

    Necesario antes de insertar una nueva: solo puede haber una fila activa
    por (task_id, model_id, sample_idx) (idx_solutions_natural_key_active).
    """
    db.execute(
        update(Solution)
        .where(
            Solution.task_id == task_id,
            Solution.model_id == model_id,
            Solution.sample_idx == sample_idx,
            Solution.active.is_(True),
        )
        .values(active=False)
    )


def _find_candidate(db: Session, name: str) -> Model | None:
    """Busca el model activo con role='candidate' y ese `name` exacto.

    A lo sumo puede haber uno (idx_models_unique_name_role_active), así que
    no hay ambigüedad posible.
    """
    return (
        db.query(Model)
        .filter(Model.name == name, Model.role == "candidate", Model.active.is_(True))
        .one_or_none()
    )


class SolutionLoadRow(BaseModel):
    """Una solución ya calculada afuera, para cargar directo (type=load)."""

    model_name: str
    task_id: int
    sample_idx: int = 0
    code: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    gen_error: str | None = None
    has_error: bool = False


class GenerateSolutionsRequest(BaseModel):
    type: Literal["load", "calculate"] = "calculate"
    n_samples: int = Field(
        default=1, ge=1, description="Muestras a generar por task (solo aplica a type=calculate)."
    )
    solutions: list[SolutionLoadRow] | None = Field(
        default=None, description="Requerido con type=load; ignorado con type=calculate."
    )


def _generate_solutions_job(db: Session, progress: ProgressFn, body: GenerateSolutionsRequest) -> dict:
    """Cuerpo real de /solutions/generate — corre en background (ver app.services.jobs)."""
    created = 0
    failed = 0
    errors: list[dict] = []

    if body.type == "load":
        rows = body.solutions or []
        total = len(rows)

        for i, row in enumerate(rows, start=1):
            model = _find_candidate(db, row.model_name)
            if model is None:
                failed += 1
                errors.append(
                    {
                        "model_name": row.model_name,
                        "task_id": row.task_id,
                        "sample_idx": row.sample_idx,
                        "error": f"no hay ningún model candidate activo con name={row.model_name!r}",
                    }
                )
                progress(i, total, f"{i}/{total} procesadas")
                continue

            _deactivate_active_solution(
                db, task_id=row.task_id, model_id=model.model_id, sample_idx=row.sample_idx
            )
            has_error = row.has_error or row.gen_error is not None
            db.add(
                Solution(
                    task_id=row.task_id,
                    model_id=model.model_id,
                    sample_idx=row.sample_idx,
                    code=row.code,
                    tokens_in=row.tokens_in,
                    tokens_out=row.tokens_out,
                    latency_ms=row.latency_ms,
                    cost_usd=row.cost_usd,
                    gen_error=row.gen_error,
                    has_error=has_error,
                )
            )
            created += 1
            if has_error:
                failed += 1
                errors.append(
                    {
                        "model_name": row.model_name,
                        "task_id": row.task_id,
                        "sample_idx": row.sample_idx,
                        "error": row.gen_error,
                    }
                )
            progress(i, total, f"{i}/{total} procesadas")
    else:
        candidates = db.query(Model).filter(Model.role == "candidate", Model.active.is_(True)).all()
        tasks = db.query(Task).filter(Task.active.is_(True)).all()
        total = len(candidates) * len(tasks) * body.n_samples
        processed = 0

        for model in candidates:
            name = model.name
            items = [(task, sample_idx) for task in tasks for sample_idx in range(body.n_samples)]
            results = run_bounded(
                items, lambda item: generation.generate_solution(item[0], model), model.concurrency
            )

            for (task, sample_idx), result, exc in results:
                gen_error = str(exc) if exc is not None else result.gen_error

                _deactivate_active_solution(
                    db, task_id=task.task_id, model_id=model.model_id, sample_idx=sample_idx
                )
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
                            "task_id": task.task_id,
                            "sample_idx": sample_idx,
                            "error": gen_error,
                        }
                    )
                processed += 1
                progress(processed, total, f"{processed}/{total} procesadas ({name})")

    active_solutions = db.query(Solution).filter(Solution.active.is_(True)).all()
    write_snapshot("solutions", [row_to_dict(s) for s in active_solutions])

    return {"created": created, "failed": failed, "errors": errors}


@router.post("/solutions/generate", status_code=202)
def generate_solutions(body: GenerateSolutionsRequest, db: Session = Depends(get_db)) -> dict:
    """Lanza (async) la carga (`type=load`) o generación (`type=calculate`) de
    soluciones para todos los modelos candidate activos. Devuelve enseguida
    el estado inicial — para ver el progreso o el resultado final, consultar
    `GET /jobs` (estado de TODOS los kinds en un solo request, incluido
    este). 409 si ya hay un job de `kind='solutions_generate'` corriendo.

    - `type=calculate` (default, sin parámetros adicionales): busca TODOS
      los models activos con `role='candidate'` y genera con
      `generation.generate_solution` para TODAS las tasks activas x
      `n_samples`, por cada uno, con hasta `model.concurrency` generaciones
      en vuelo a la vez por modelo (ver services.concurrency.run_bounded).
      La temperatura no es un parámetro de este request: se usa
      `model.temperature` si está configurada.
    - `type=load`: inserta directo las filas de `solutions`, sin llamar a
      ningún LLM; cada fila trae su propio `model_name` (una sola carga
      puede traer soluciones de varios modelos distintos).

    En ambos casos, un fallo puntual (generación fallida, o `model_name`
    que no resuelve a ningún candidate activo en `type=load`) NO aborta el
    resto del lote. Reemplazo con borrado lógico: por cada (task, model,
    sample_idx) desactiva la solution activa anterior de esa misma
    identidad antes de insertar la nueva. Al terminar deja un snapshot de
    las solutions activas resultantes en data/current/solutions.json
    (rotando el anterior a data/previous).
    """
    if body.type == "load" and not body.solutions:
        raise HTTPException(status_code=422, detail="type=load requiere 'solutions'")

    try:
        job = jobs.start_job(_KIND, lambda db, progress: _generate_solutions_job(db, progress, body), db)
    except JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return jobs.to_board_entry(job, _KIND)
