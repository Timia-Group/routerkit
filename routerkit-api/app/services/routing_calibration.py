"""Calibrates the routing policy's cost/quality trade-off (lambda) for a
freshly trained predictor — mirrors the methodology of the project's
reference offline experiment (see conversation/README), with the one piece
that experiment never automated: picking lambda.

The policy itself (services.policy, still to be wired into production
routing) picks, at each of up to MAX_SHOTS attempts, the active candidate
model maximizing `score = Y_pred - lambda * cost/cost_max` among the
candidates not yet tried for that task. `lambda` is a single scalar per
shot count (1..MAX_SHOTS) — not learned, not per-task — trading off
predicted quality against cost.

For each shot count, LAMBDA_GRID values of lambda are swept; for each one,
a Monte Carlo simulation estimates (accuracy, avg cost per task) under that
lambda by bootstrap-resampling REAL historical pass/fail outcomes (the
project's actual solutions+grades) rather than drawing a Bernoulli sample
from Y_pred — this reflects the real variance of repeated attempts, not an
idealized one. The sweep runs on the TRAIN split's tasks only (picking
lambda using test data would leak); the winning lambda is the "elbow" of
that (cost, accuracy) curve — the point past which paying more doesn't buy
meaningfully more accuracy — found automatically with `kneed`, after
isotonic smoothing to tame the curve's inherently step-like shape (the
argmax that picks each shot's model only changes at discrete lambda
thresholds where two candidates' scores cross, not continuously — see
conversation). The winning lambda is then re-evaluated once on the TEST
split (never used to pick it) purely to report an honest expected
accuracy/cost.

Repetition counts (TRAIN_REPETITIONS/TEST_REPETITIONS) match the reference
experiment's own choice (500/1000) — already far more than enough to make
Monte-Carlo sampling noise negligible at this task-set size; the lambda
grid being reasonably fine (LAMBDA_GRID) matters far more for a clean
elbow than a larger repetition count would.
"""

import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.isotonic import IsotonicRegression
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import Grade, Model, Solution

MAX_SHOTS = 3
LAMBDA_GRID = np.linspace(0.0, 5.0, 100)
TRAIN_REPETITIONS = 500
TEST_REPETITIONS = 1000
_RNG_SEED = 42


def load_outcomes_by_task_model(db: Session, model_ids: list[int]) -> dict[tuple[int, int], np.ndarray]:
    """{(task_id, model_id): bool array of real pass/fail outcomes}, from
    active solutions + their active grade. Excludes has_error=true grades
    (an infrastructure failure, not a genuine quality signal — see
    routerkit.grades) and has_error=true solutions (no code was ever
    graded): neither belongs in a pool meant to represent how often a
    model actually solves a task. Public: services.reports reuses this
    directly rather than requerying the same join."""
    rows = (
        db.query(Solution.task_id, Solution.model_id, Grade.passed)
        .join(Grade, Grade.solution_id == Solution.solution_id)
        .filter(
            Solution.active.is_(True),
            Solution.has_error.is_(False),
            Grade.active.is_(True),
            Grade.has_error.is_(False),
            Solution.model_id.in_(model_ids),
        )
        .all()
    )
    outcomes: dict[tuple[int, int], list[bool]] = {}
    for task_id, model_id, passed in rows:
        outcomes.setdefault((task_id, model_id), []).append(bool(passed))
    return {key: np.array(vals, dtype=bool) for key, vals in outcomes.items()}


def load_model_costs(db: Session, model_ids: list[int]) -> dict[int, float]:
    """{model_id: average solutions.cost_usd}, active solutions only — the
    empirical "cost of one real generation attempt" for that candidate
    (plays the role of the reference experiment's fixed
    cost_per_model_request.csv lookup). Public: routers.routing reuses this
    directly rather than recomputing cost its own way — using a different
    cost definition there than the one lambda was calibrated against here
    would silently invalidate it (routerkit.model_metrics.cost_per_task is
    a DIFFERENT, much larger number — total historical cost across every
    sample ever generated for a task, divided by task count — not the cost
    of a single attempt; don't use it here)."""
    rows = (
        db.query(Solution.model_id, func.avg(Solution.cost_usd))
        .filter(Solution.active.is_(True), Solution.model_id.in_(model_ids))
        .group_by(Solution.model_id)
        .all()
    )
    return {model_id: float(avg_cost or 0.0) for model_id, avg_cost in rows}


def simulate_lambda(
    *,
    task_ids: list[int],
    y_pred_by_task_model: dict[tuple[int, int], float],
    outcomes_by_task_model: dict[tuple[int, int], np.ndarray],
    costs_by_model: dict[int, float],
    model_ids: list[int],
    cost_max: float,
    lam: float,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[int, tuple[float, float]]:
    """Monte Carlo for a single lambda. Returns {shots: (accuracy,
    avg_cost_per_task)} for shots in 1..MAX_SHOTS — simulated in one pass
    per task (a task's state at shot k already includes shots 1..k-1, so
    every shot count is just a slice of the same run, not a separate one).
    Public: services.reports reuses this directly for the TEST-split sweep
    it plots (the one calibrate_routing itself never needs, since it only
    evaluates its 3 already-picked lambdas on test, not the whole grid).
    """
    passed_sum = np.zeros(MAX_SHOTS)
    cost_sum = np.zeros(MAX_SHOTS)
    counted_tasks = 0

    for task_id in task_ids:
        # Rank this task's candidates by score, keeping only the ones we
        # actually have real outcome data for — can't simulate reality for
        # a candidate never tried on this task.
        scored = []
        for model_id in model_ids:
            key = (task_id, model_id)
            if key not in outcomes_by_task_model or key not in y_pred_by_task_model:
                continue
            score = y_pred_by_task_model[key] - lam * costs_by_model[model_id] / cost_max
            scored.append((score, model_id))
        if not scored:
            continue
        scored.sort(key=lambda t: -t[0])
        ranked = scored[:MAX_SHOTS]

        still_trying = np.ones(repetitions, dtype=bool)
        running_cost = np.zeros(repetitions)
        running_passed = np.zeros(repetitions, dtype=bool)
        last_shot_idx = -1
        for shot_idx, (_, model_id) in enumerate(ranked):
            pool = outcomes_by_task_model[(task_id, model_id)]
            draws = rng.choice(pool, size=repetitions, replace=True)
            running_cost = running_cost + np.where(still_trying, costs_by_model[model_id], 0.0)
            running_passed = running_passed | (still_trying & draws)
            still_trying = still_trying & ~draws
            passed_sum[shot_idx] += running_passed.mean()
            cost_sum[shot_idx] += running_cost.mean()
            last_shot_idx = shot_idx

        # A task with fewer real candidates than MAX_SHOTS has no more
        # options left to try — its last simulated shot's numbers carry
        # forward for any remaining shot counts.
        for shot_idx in range(last_shot_idx + 1, MAX_SHOTS):
            passed_sum[shot_idx] += running_passed.mean()
            cost_sum[shot_idx] += running_cost.mean()
        counted_tasks += 1

    if counted_tasks == 0:
        return dict.fromkeys(range(1, MAX_SHOTS + 1), (0.0, 0.0))
    return {
        shots: (passed_sum[shots - 1] / counted_tasks, cost_sum[shots - 1] / counted_tasks)
        for shots in range(1, MAX_SHOTS + 1)
    }


def _pick_elbow(points: list[tuple[float, float, float]]) -> float:
    """points: list of (lambda, accuracy, cost) from the train-split
    sweep. Returns the winning lambda — the knee of the (cost, accuracy)
    curve (sorted by ascending cost; isotonic-smoothed to force accuracy
    non-decreasing in cost before looking for the knee, since the real
    curve is inherently step-shaped — see module docstring)."""
    ordered = sorted(points, key=lambda t: t[2])  # by cost ascending
    costs = np.array([p[2] for p in ordered])
    accuracies = np.array([p[1] for p in ordered])
    lambdas = np.array([p[0] for p in ordered])

    if len(set(costs)) < 3:
        # Degenerate sweep (lambda barely moves cost over this grid) —
        # nothing meaningful to find a knee in, fall back to the
        # highest-accuracy point.
        return float(lambdas[int(np.argmax(accuracies))])

    smoothed = IsotonicRegression(increasing=True).fit_transform(costs, accuracies)
    kl = KneeLocator(costs, smoothed, curve="concave", direction="increasing")
    if kl.knee is None:
        return float(lambdas[int(np.argmax(smoothed))])
    knee_idx = int(np.argmin(np.abs(costs - kl.knee)))
    return float(lambdas[knee_idx])


def calibrate_routing(
    db: Session,
    *,
    model,
    x: pd.DataFrame,
    row_task_ids: pd.Series,
    row_model_ids: pd.Series,
    feature_columns: dict,
    train_task_ids: list[int],
    test_task_ids: list[int],
) -> dict:
    """Runs the full lambda calibration for a freshly trained predictor —
    see module docstring. `x`/`row_task_ids`/`row_model_ids` share the same
    row index (the training dataset, see services.training._build_dataset)
    — `model.predict(x)` gives Y_pred per row, paired with its
    (task_id, model_id) via the other two. Returns
    {"routing_lambdas": {...}, "lambda_sweep": {...}} ready to persist on
    the Predictor row.
    """
    candidates = (
        db.query(Model)
        .filter(Model.role == "candidate", Model.active.is_(True), Model.name.in_(feature_columns["models"]))
        .all()
    )
    model_ids = [m.model_id for m in candidates]

    outcomes_by_task_model = load_outcomes_by_task_model(db, model_ids)
    costs_by_model = load_model_costs(db, model_ids)
    cost_max = max(costs_by_model.values()) if costs_by_model else 1.0
    if cost_max <= 0:
        cost_max = 1.0

    predictions = model.predict(x)
    y_pred_by_task_model = {
        (int(task_id), int(model_id)): float(pred)
        for task_id, model_id, pred in zip(row_task_ids, row_model_ids, predictions, strict=True)
    }

    rng = np.random.default_rng(_RNG_SEED)

    # One Monte Carlo pass per grid point gives every shot count's
    # (accuracy, cost) at once (simulate_lambda's return already covers
    # shots 1..MAX_SHOTS) — sweep the grid ONCE here rather than once per
    # shot count, which would just redo the identical simulation 3x over.
    sweep_points_by_shots: dict[int, list[tuple[float, float, float]]] = {
        shots: [] for shots in range(1, MAX_SHOTS + 1)
    }
    for lam in LAMBDA_GRID:
        result = simulate_lambda(
            task_ids=train_task_ids,
            y_pred_by_task_model=y_pred_by_task_model,
            outcomes_by_task_model=outcomes_by_task_model,
            costs_by_model=costs_by_model,
            model_ids=model_ids,
            cost_max=cost_max,
            lam=float(lam),
            repetitions=TRAIN_REPETITIONS,
            rng=rng,
        )
        for shots in range(1, MAX_SHOTS + 1):
            accuracy, cost = result[shots]
            sweep_points_by_shots[shots].append((float(lam), accuracy, cost))

    routing_lambdas: dict[str, dict] = {}
    lambda_sweep: dict[str, list[dict]] = {}

    for shots in range(1, MAX_SHOTS + 1):
        sweep_points = sweep_points_by_shots[shots]
        winning_lambda = _pick_elbow(sweep_points)

        final = simulate_lambda(
            task_ids=test_task_ids,
            y_pred_by_task_model=y_pred_by_task_model,
            outcomes_by_task_model=outcomes_by_task_model,
            costs_by_model=costs_by_model,
            model_ids=model_ids,
            cost_max=cost_max,
            lam=winning_lambda,
            repetitions=TEST_REPETITIONS,
            rng=rng,
        )
        test_accuracy, test_cost = final[shots]

        routing_lambdas[str(shots)] = {
            "lambda": winning_lambda,
            "test_accuracy": test_accuracy,
            "test_cost_avg": test_cost,
        }
        lambda_sweep[str(shots)] = [
            {"lambda": lam, "accuracy": acc, "cost_avg": cost} for lam, acc, cost in sweep_points
        ]

    return {"routing_lambdas": routing_lambdas, "lambda_sweep": lambda_sweep}
