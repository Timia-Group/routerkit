"""Cascading soft delete.

When a reload deactivates rows in tasks/questions/models, everything that
was computed from those rows must also be deactivated, so that only the
current process/load remains active. Each function receives the IDs that
were ALREADY marked `active=false` in the source table and propagates the
deactivation "forward" through the pipeline:

    tasks ─┬─> features
           ├─> solutions ─> grades
           └─> targets

    questions ─> features

    models ─┬─> solutions ─> grades
            ├─> targets
            └─> model_metrics
"""

from sqlalchemy import ColumnElement, select, update
from sqlalchemy.orm import Session

from app.db.models import Feature, Grade, ModelMetric, Solution, Target


def _deactivate_solutions_where(db: Session, condition: ColumnElement[bool]) -> None:
    """Deactivates active solutions matching `condition` (e.g.
    `Solution.task_id.in_(task_ids)` — bounded by catalog size) and their
    active grades, matched via a subquery rather than first SELECTing
    matching solution_ids into Python and passing THAT list back in a
    second `.in_(...)`: at this project's real data scale, that list can
    run into the hundreds of thousands of ids, well past Postgres's
    ~65535 bound-parameter limit per query (the exact failure
    _write_grades_snapshot hit before it was rewritten as a JOIN — this is
    the same class of bug, in the one other place it was still lurking).
    """
    solution_id_subquery = select(Solution.solution_id).where(Solution.active.is_(True), condition)
    db.execute(
        update(Grade).where(Grade.active.is_(True), Grade.solution_id.in_(solution_id_subquery)).values(active=False)
    )
    db.execute(update(Solution).where(Solution.active.is_(True), condition).values(active=False))


def deactivate_for_tasks(db: Session, task_ids: list[int]) -> None:
    """Deactivates features/solutions(+grades)/targets of the given tasks."""
    if not task_ids:
        return
    db.execute(update(Feature).where(Feature.task_id.in_(task_ids)).values(active=False))
    _deactivate_solutions_where(db, Solution.task_id.in_(task_ids))
    db.execute(update(Target).where(Target.task_id.in_(task_ids)).values(active=False))


def deactivate_for_questions(db: Session, question_ids: list[int]) -> None:
    """Deactivates features of the given questions."""
    if not question_ids:
        return
    db.execute(update(Feature).where(Feature.question_id.in_(question_ids)).values(active=False))


def deactivate_for_models(db: Session, model_ids: list[int]) -> None:
    """Deactivates solutions(+grades)/targets/model_metrics of the given models."""
    if not model_ids:
        return
    _deactivate_solutions_where(db, Solution.model_id.in_(model_ids))
    db.execute(update(Target).where(Target.model_id.in_(model_ids)).values(active=False))
    db.execute(update(ModelMetric).where(ModelMetric.model_id.in_(model_ids)).values(active=False))
