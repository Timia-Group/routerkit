"""Agregación real de targets y model_metrics a partir de solutions/grades activas.

Fórmulas basadas en los comentarios del schema (routerkit.targets,
routerkit.model_metrics):

- targets: n = soluciones generadas, k = soluciones con grades.passed=True,
  pass_rate = k/n. Una fila por (task, model candidato).
- model_metrics, por model:
  - taskpass_pct: % de tasks con al menos una solución correcta (k>0 en targets).
  - complete_pct: % de solutions cuya generación no falló (gen_error IS NULL).
  - pass_rate_avg: promedio de targets.pass_rate.
  - cost_total / cost_per_task: suma de solutions.cost_usd, dividido entre
    tasks distintas.
  - latency_avg_ms / tokens_out_avg: promedio de solutions.latency_ms / tokens_out.

Reemplazo completo con borrado lógico: desactiva todos los targets /
model_metrics existentes y crea filas nuevas activas, a partir de las
solutions/grades activas en este momento.
"""

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.db.models import Grade, ModelMetric, Solution, Target


def recompute_targets(db: Session) -> int:
    db.execute(update(Target).where(Target.active.is_(True)).values(active=False))

    rows = db.execute(
        select(
            Solution.task_id,
            Solution.model_id,
            func.count(Solution.solution_id).label("n"),
            func.count(Grade.solution_id).filter(Grade.passed.is_(True)).label("k"),
        )
        .outerjoin(Grade, (Grade.solution_id == Solution.solution_id) & Grade.active.is_(True))
        .where(Solution.active.is_(True))
        .group_by(Solution.task_id, Solution.model_id)
    ).all()

    for row in rows:
        db.add(
            Target(
                task_id=row.task_id,
                model_id=row.model_id,
                n=row.n,
                k=row.k,
                pass_rate=(row.k / row.n) if row.n else 0,
            )
        )

    db.commit()
    return len(rows)


def recompute_model_metrics(db: Session) -> int:
    db.execute(update(ModelMetric).where(ModelMetric.active.is_(True)).values(active=False))

    target_rows = db.execute(
        select(
            Target.model_id,
            func.count(Target.task_id).label("n_tasks"),
            func.count(Target.task_id).filter(Target.k > 0).label("n_tasks_passed"),
            func.avg(Target.pass_rate).label("pass_rate_avg"),
        )
        .where(Target.active.is_(True))
        .group_by(Target.model_id)
    ).all()

    solution_rows = {
        row.model_id: row
        for row in db.execute(
            select(
                Solution.model_id,
                func.count(Solution.solution_id).label("n_solutions"),
                func.count(Solution.solution_id)
                .filter(Solution.gen_error.is_(None))
                .label("n_complete"),
                func.sum(Solution.cost_usd).label("cost_total"),
                func.avg(Solution.latency_ms).label("latency_avg_ms"),
                func.avg(Solution.tokens_out).label("tokens_out_avg"),
            )
            .where(Solution.active.is_(True))
            .group_by(Solution.model_id)
        ).all()
    }

    for target_row in target_rows:
        solution_row = solution_rows.get(target_row.model_id)
        n_tasks = target_row.n_tasks or 0
        n_solutions = solution_row.n_solutions if solution_row else 0
        cost_total = (solution_row.cost_total if solution_row else 0) or 0

        db.add(
            ModelMetric(
                model_id=target_row.model_id,
                taskpass_pct=(100 * target_row.n_tasks_passed / n_tasks) if n_tasks else None,
                complete_pct=(
                    (100 * solution_row.n_complete / n_solutions)
                    if solution_row and n_solutions
                    else None
                ),
                pass_rate_avg=target_row.pass_rate_avg,
                cost_per_task=(cost_total / n_tasks) if n_tasks else None,
                cost_total=cost_total,
                latency_avg_ms=solution_row.latency_avg_ms if solution_row else None,
                tokens_out_avg=solution_row.tokens_out_avg if solution_row else None,
            )
        )

    db.commit()
    return len(target_rows)
