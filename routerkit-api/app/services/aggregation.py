"""Real aggregation of targets and model_metrics from active solutions/grades.

Formulas based on the schema comments (routerkit.targets,
routerkit.model_metrics):

- targets: n = generated solutions EXCLUDING `has_error=true` ones BY
  DEFAULT (a connection drop, exhausted rate-limit/timeout retries, a
  missing api key, a content-filter block, or any other call failure
  means no response ever came back, so it says nothing about whether the
  model can solve the task — same exclusion services.routing_calibration
  already applies to its own outcome pool). The one deliberate carve-out
  is `_TRUNCATION_GEN_ERROR_PATTERN`: a `has_error=true` row whose
  `gen_error` says the response was merely truncated
  (`finish_reason=length`/`max_tokens`) IS counted in `n` — the model did
  answer, it just ran out of its configured output budget, which is a
  real attempt that can legitimately fail evaluation, not a call failure.
  This carve-out only ever matches the historical experiment dataset
  (`solutions_exp.jsonl`, produced by an external process that flagged
  truncated-but-received responses as `has_error=true` too); the live
  generation path (services.generation) never sets `has_error=true` for
  that reason at all — it doesn't inspect `finish_reason` — so for any
  future type=calculate run or freshly loaded file, every `has_error=true`
  row is, and will remain, a genuine call failure, correctly excluded by
  the default with no pattern-matching needed. k = solutions with
  grades.passed=True, pass_rate = k/n. One row per (task, candidate
  model). `model_metrics` below, by contrast, counts ALL solutions
  including every kind of failure — it's reporting operational stats
  (cost, latency, `complete_pct`), not a training label.
- model_metrics, per model:
  - taskpass_pct: % of tasks with at least one correct solution (k>0 in targets).
  - complete_pct: % of solutions whose generation did not fail (gen_error IS NULL).
  - pass_rate_avg: average of targets.pass_rate.
  - cost_total / cost_per_task: sum of solutions.cost_usd, divided by
    the distinct number of tasks.
  - latency_avg_ms / tokens_out_avg: average of solutions.latency_ms / tokens_out.

Full replacement via soft delete: deactivates all existing targets /
model_metrics and creates new active rows, from the solutions/grades that
are active at this moment.
"""

from sqlalchemy import and_, func, not_, select, update
from sqlalchemy.orm import Session

from app.db.models import Grade, ModelMetric, Solution, Target

# Deliberately a narrow ALLOWLIST (not the reverse): has_error=true is
# excluded by default (the safe assumption for any call failure we don't
# recognize — rate limits, timeouts, missing api keys, context-window
# errors, connection drops, content-filter blocks, whatever a provider
# throws next), and only a row matching this pattern is carved back IN as
# a real attempt. See the module docstring — this only ever fires for the
# historical experiment dataset's truncation rows.
_TRUNCATION_GEN_ERROR_PATTERN = r"finish_reason=(length|max_tokens)"


def _not_infra_error():
    """True for a solution that counts as a real attempt: has_error=false,
    or has_error=true but merely truncated (see _TRUNCATION_GEN_ERROR_PATTERN)."""
    return not_(
        and_(
            Solution.has_error.is_(True),
            not_(Solution.gen_error.op("~*")(_TRUNCATION_GEN_ERROR_PATTERN)),
        )
    )


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
        .where(Solution.active.is_(True), _not_infra_error())
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
