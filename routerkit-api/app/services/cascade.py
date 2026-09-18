"""Borrado lógico en cascada.

Cuando una recarga desactiva filas de tasks/questions/models, todo lo que se
calculó a partir de esas filas también debe desactivarse, para que solo
quede activo lo del proceso/carga actual. Cada función recibe los IDs que
YA se marcaron `active=false` en la tabla de origen y propaga la
desactivación "hacia adelante" en el pipeline:

    tasks ─┬─> features
           ├─> solutions ─> grades
           └─> targets

    questions ─> features

    models ─┬─> solutions ─> grades
            ├─> targets
            └─> model_metrics
"""

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.models import Feature, Grade, ModelMetric, Solution, Target


def _deactivate_solutions(db: Session, solution_ids: list[int]) -> None:
    if not solution_ids:
        return
    db.execute(update(Solution).where(Solution.solution_id.in_(solution_ids)).values(active=False))
    db.execute(update(Grade).where(Grade.solution_id.in_(solution_ids)).values(active=False))


def deactivate_for_tasks(db: Session, task_ids: list[int]) -> None:
    """Desactiva features/solutions(+grades)/targets de las tasks dadas."""
    if not task_ids:
        return
    db.execute(update(Feature).where(Feature.task_id.in_(task_ids)).values(active=False))
    solution_ids = list(
        db.execute(
            select(Solution.solution_id).where(
                Solution.task_id.in_(task_ids), Solution.active.is_(True)
            )
        ).scalars()
    )
    _deactivate_solutions(db, solution_ids)
    db.execute(update(Target).where(Target.task_id.in_(task_ids)).values(active=False))


def deactivate_for_questions(db: Session, question_ids: list[int]) -> None:
    """Desactiva features de las questions dadas."""
    if not question_ids:
        return
    db.execute(update(Feature).where(Feature.question_id.in_(question_ids)).values(active=False))


def deactivate_for_models(db: Session, model_ids: list[int]) -> None:
    """Desactiva solutions(+grades)/targets/model_metrics de los models dados."""
    if not model_ids:
        return
    solution_ids = list(
        db.execute(
            select(Solution.solution_id).where(
                Solution.model_id.in_(model_ids), Solution.active.is_(True)
            )
        ).scalars()
    )
    _deactivate_solutions(db, solution_ids)
    db.execute(update(Target).where(Target.model_id.in_(model_ids)).values(active=False))
    db.execute(update(ModelMetric).where(ModelMetric.model_id.in_(model_ids)).values(active=False))
