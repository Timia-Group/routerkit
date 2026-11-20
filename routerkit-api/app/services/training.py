"""Trains the predictor over active features/targets. Mirrors the
methodology of the project's reference offline experiment, with one
deliberate difference: we always re-run GridSearchCV over the chosen
algorithm's own hyperparameter grid instead of hardcoding the experiment's
winning hyperparameters — our own catalog varies a lot in size/distribution
from one person's data to another's (and over time, as more tasks/
questions/solutions get added), unlike the one-off academic experiment
this mirrors.

Algorithm (POST /train's `algorithms`): one or more keys from `ALGORITHMS`
below. One key -> trained as requested, no comparison (even if its own
calibration later turns out degenerate — the caller asked for it by name).
Several keys (or omitted, which defaults to just `DEFAULT_ALGORITHM` —
today's GradientBoostingRegressor, unchanged behavior) -> each is
GridSearchCV-fit AND calibrated (see services.routing_calibration), and the
winner is picked by actual downstream ROUTING performance on the test
split, not by raw cv_score/R² — see `_pick_winner`. A candidate whose own
calibration comes back with lambda=0 at any shot count is disqualified
first (that calibration found no real cost/accuracy trade-off to speak
of — see services.routing_calibration._pick_elbow's degenerate-curve
fallback — a strong sign the predictor is too weak/uninformative to
support a sensible routing policy at all, not just "a bit worse").

A single entry point (`train_predictor`, called by the one `POST /train`
endpoint) does everything end to end — the user never has to remember to
call a separate aggregation step first: it recomputes `targets`/
`model_metrics` from the currently active solutions/grades
(services.aggregation), then builds the dataset, splits it, and trains.

Dataset: wide, one row per (active task, active candidate model) pair —
columns = each active question's value (pivoted from the long `features`
table, named by `question.question_key`) + one-hot of the candidate
model's name; target = the just-recomputed `targets.pass_rate`. Every
active question needs a `question_key` assigned (same stable-id
requirement as everywhere else in this system) — training fails loudly
listing which ones don't, rather than silently guessing a column name. A
task missing any active feature value is dropped from the dataset entirely
(an incomplete row would mean feeding NaN into the regressor).

Split: GroupShuffleSplit(test_size=0.30), grouped by task_id — a task's
rows (one per candidate model) never get split across train/test, or the
model would leak information about a task it's "never seen" (same task,
different candidate row, see the project's reference experiment). Same
grouping for the inner GridSearchCV (GroupKFold), only on the train
partition; the test partition is only ever used once, for the final
reported metrics (test_r2/test_mae).

Persisted artifact (joblib): {"model": <the exact estimator selected by
GridSearchCV, fit on the train partition>, "feature_columns": {"questions":
[...], "models": [...]}} — the column order is the input contract any
future prediction (services/policy.py) must follow.
"""

import os
import time
from dataclasses import dataclass

# Must be set before sklearn/xgboost/lightgbm ever spin up their native
# thread pools (OpenMP/BLAS) — GridSearchCV's own n_jobs=-1 already
# parallelizes across the hyperparameter grid; if each estimator ALSO
# grabs every core for its own internal threading, the two layers of
# parallelism oversubscribe the CPU and a fit gets dramatically SLOWER,
# not faster (confirmed empirically: LightGBM alone went from a
# 30-minute hang to 2 minutes once this was in place). RandomForest/
# XGBoost/LightGBM also get an explicit n_jobs=1 below for the same
# reason — they're not all guaranteed to respect this env var — but
# HistGradientBoostingRegressor has no n_jobs of its own, so this is the
# only lever for it.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sqlalchemy.orm import Session
from xgboost import XGBRegressor

from app.config import settings
from app.db.models import Feature, Model, Question, Target
from app.services.aggregation import recompute_model_metrics, recompute_targets
from app.services.jobs import ProgressFn
from app.services.reports import generate_training_report
from app.services.routing_calibration import MAX_SHOTS, calibrate_routing, load_model_costs

_TEST_SIZE = 0.30
_RANDOM_STATE = 42
_MAX_CV_SPLITS = 5
_MIN_TASKS = 4  # need enough distinct tasks for a meaningful grouped train/test split

DEFAULT_ALGORITHM = "gradient_boosting"

_GB_GRID = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [2, 3, 4],
    "n_estimators": [200, 400, 600],
    "subsample": [0.8, 1.0],
}
# n_jobs=1 on every estimator below that has its own internal threading
# (RandomForest/XGBoost/LightGBM) — GridSearchCV's outer n_jobs=-1 already
# parallelizes across the grid; leaving the estimator's own n_jobs at its
# default (-1 or "all cores") makes every one of GridSearchCV's parallel
# workers ALSO try to grab every core, which oversubscribes and can make
# a fit take dramatically longer, not shorter (confirmed empirically:
# LightGBM alone went from a 30-minute hang to 2 minutes once this was
# fixed). HistGradientBoostingRegressor has no n_jobs of its own; it's
# capped by the OMP_NUM_THREADS=1 set at the top of this module instead,
# same reasoning.
ALGORITHMS: dict[str, tuple] = {
    "gradient_boosting": (
        lambda: GradientBoostingRegressor(random_state=_RANDOM_STATE),
        _GB_GRID,
    ),
    "xgboost": (
        lambda: XGBRegressor(random_state=_RANDOM_STATE, n_jobs=1, tree_method="hist"),
        _GB_GRID,
    ),
    "histgradientboosting": (
        lambda: HistGradientBoostingRegressor(random_state=_RANDOM_STATE),
        {
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 4],
            "max_iter": [200, 400, 600],
            "l2_regularization": [0.0, 1.0],
        },
    ),
    "lightgbm": (
        lambda: LGBMRegressor(random_state=_RANDOM_STATE, n_jobs=1, verbosity=-1),
        _GB_GRID,
    ),
    "random_forest": (
        lambda: RandomForestRegressor(random_state=_RANDOM_STATE, n_jobs=1),
        {
            "n_estimators": [200, 400, 600],
            "max_depth": [None, 5, 10],
            "min_samples_leaf": [1, 5],
        },
    ),
}


class TrainingError(RuntimeError):
    """Not enough active data (features/targets/tasks) to train a predictor,
    or an invalid `shots`/`algorithms` request."""


@dataclass
class TrainingResult:
    algorithm: str
    cv_score: float
    test_r2: float
    test_mae: float
    feature_columns: dict
    artifact_path: str
    shots: int
    routing_lambdas: dict
    lambda_sweep: dict
    report_dir: str
    # One summary row per algorithm actually tried (only >1 entry when
    # `algorithms` asked for more than one) — cv_score/test_r2 plus the
    # ROUTING accuracy/cost it would have gotten at the full `shots` budget,
    # and whether it was disqualified (degenerate lambda=0 calibration).
    # Never persisted on the Predictor row; just here for GET /jobs to show.
    candidates_tried: list[dict]


def _build_dataset(db: Session) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, dict]:
    """Returns (X, y, groups, row_model_ids, feature_columns) built from
    active features/targets — row_model_ids is the candidate model_id of
    each row (same index as X/y/groups), needed by
    services.routing_calibration to know which model each row's Y_pred
    belongs to. Raises TrainingError if there isn't enough active data to
    train on."""
    questions = db.query(Question).filter(Question.active.is_(True)).all()
    missing_keys = [q.question_id for q in questions if not q.question_key]
    if missing_keys:
        raise TrainingError(
            f"{len(missing_keys)} active question(s) have no question_key assigned "
            f"(question_id: {missing_keys}) — every active question needs one to train"
        )
    question_key_by_id = {q.question_id: q.question_key for q in questions}
    question_columns = sorted(question_key_by_id.values())

    features = (
        db.query(Feature)
        .filter(Feature.active.is_(True), Feature.question_id.in_(question_key_by_id.keys()))
        .all()
    )
    if not features:
        raise TrainingError("no active features to train on")

    features_long = pd.DataFrame(
        [
            {
                "task_id": f.task_id,
                "question_key": question_key_by_id[f.question_id],
                "value": float(f.value),
            }
            for f in features
            if f.value is not None
        ]
    )
    features_wide = features_long.pivot_table(index="task_id", columns="question_key", values="value")
    features_wide = features_wide.reindex(columns=question_columns)
    # A task missing any active feature's value can't be fed to the
    # regressor (NaN) — drop it entirely rather than guessing a fill value.
    features_wide = features_wide.dropna()

    candidates = db.query(Model).filter(Model.role == "candidate", Model.active.is_(True)).all()
    candidate_name_by_id = {m.model_id: m.name for m in candidates}
    if not candidate_name_by_id:
        raise TrainingError("no active candidate models")
    model_columns = sorted(candidate_name_by_id.values())

    targets = (
        db.query(Target)
        .filter(Target.active.is_(True), Target.model_id.in_(candidate_name_by_id.keys()))
        .all()
    )
    if not targets:
        raise TrainingError(
            "no active targets to train on — recompute_targets found nothing, which means "
            "there are no active grades yet (generate and evaluate solutions first)"
        )

    targets_df = pd.DataFrame(
        [
            {
                "task_id": t.task_id,
                "model_id": t.model_id,
                "model_name": candidate_name_by_id[t.model_id],
                "pass_rate": float(t.pass_rate),
            }
            for t in targets
        ]
    )
    one_hot = pd.get_dummies(targets_df["model_name"]).reindex(columns=model_columns, fill_value=0)

    dataset = targets_df[["task_id", "model_id", "pass_rate"]].join(one_hot)
    dataset = dataset.join(features_wide, on="task_id", how="inner")
    # Postgres gives no row-order guarantee without ORDER BY (targets_df
    # came straight from db.query(Target).all()) — pin it down explicitly
    # so a row-position-dependent subsampling grid (subsample<1.0, several
    # algorithms' grids use it) is reproducible run to run on the same
    # underlying data, not just "deterministic up to whatever order
    # Postgres happened to scan rows in this time."
    dataset = dataset.sort_values(["task_id", "model_id"]).reset_index(drop=True)

    n_tasks = dataset["task_id"].nunique()
    if n_tasks < _MIN_TASKS:
        raise TrainingError(
            f"only {n_tasks} distinct active task(s) with complete data — need at least "
            f"{_MIN_TASKS} to do a meaningful train/test split"
        )

    feature_columns = {"questions": question_columns, "models": model_columns}
    x = dataset[question_columns + model_columns].astype(float)
    y = dataset["pass_rate"].astype(float)
    groups = dataset["task_id"]
    row_model_ids = dataset["model_id"]
    return x, y, groups, row_model_ids, feature_columns


def _fit_algorithm(
    key: str, x_train: pd.DataFrame, y_train: pd.Series, groups_train: pd.Series, cv_splits: int
) -> tuple[object, float]:
    """GridSearchCV-fits one registered algorithm. Returns (best_estimator,
    cv_score) — cv_score is R² on the train partition only, a secondary
    diagnostic (see module docstring for why the actual winner among
    several candidates is picked by routing performance, not this)."""
    factory, grid = ALGORITHMS[key]
    search = GridSearchCV(
        factory(), param_grid=grid, cv=GroupKFold(n_splits=cv_splits), scoring="r2", n_jobs=-1
    )
    search.fit(x_train, y_train, groups=groups_train)
    return search.best_estimator_, float(search.best_score_)


def _is_degenerate(routing_lambdas: dict, shots: int) -> bool:
    """True if the elbow method found no real cost/accuracy trade-off at
    ANY shot count (lambda=0 — see routing_calibration._pick_elbow's
    degenerate-curve fallback) — a sign the predictor is too weak/
    uninformative to support a sensible routing policy at all."""
    return any(routing_lambdas[str(s)]["lambda"] == 0.0 for s in range(1, shots + 1))


def _pick_winner(candidates: list[dict], shots: int, cost_max: float) -> dict:
    """Picks by actual downstream ROUTING performance at the full `shots`
    budget — utility = test_accuracy - lambda * test_cost_avg / cost_max,
    the same cost/quality trade-off the router itself optimizes for —
    never by raw cv_score/test_r2 (see module docstring: a better-fitting
    regressor doesn't necessarily route better). Degenerate candidates
    (_is_degenerate) are disqualified first; if ALL of them are degenerate,
    falls back to comparing all of them anyway — picking something is
    better than refusing to train at all."""
    survivors = [c for c in candidates if not c["degenerate"]] or candidates

    def utility(c: dict) -> float:
        final = c["routing_lambdas"][str(shots)]
        return final["test_accuracy"] - final["lambda"] * final["test_cost_avg"] / cost_max

    return max(survivors, key=utility)


def train_predictor(
    db: Session,
    progress: ProgressFn | None = None,
    *,
    shots: int | None = None,
    algorithms: list[str] | None = None,
) -> TrainingResult:
    """Recomputes targets/model_metrics from the currently active
    solutions/grades, builds the dataset from active features/(the
    just-recomputed) targets, splits it grouped by task_id, then for each
    requested algorithm (see `ALGORITHMS`/module docstring) grid-searches
    its hyperparameters (train partition only) and calibrates the routing
    lambda (services.routing_calibration) for it. Persists the winning
    model to settings.artifacts_dir. Raises TrainingError if there isn't
    enough active data (see _build_dataset) or `shots` is out of range.

    `shots`: how many sequential routing attempts to calibrate for
    (1..number of active candidates). None defaults to
    min(routing_calibration.MAX_SHOTS, number of active candidates).
    `algorithms`: which registry keys to try; None defaults to just
    `DEFAULT_ALGORITHM` (today's behavior, unchanged).

    One entry point end to end on purpose: there's no standalone
    aggregation endpoint, and the caller (POST /train) never has to
    calibrate routing as a separate step before the predictor is actually
    usable by POST /route. `progress` (optional) is called at the 4 phases
    (data prep, training+calibration, picking the winner, report
    generation) for GET /jobs to report.
    """
    if progress:
        progress(1, 4, "preparing data")
    recompute_targets(db)
    recompute_model_metrics(db)

    x, y, groups, row_model_ids, feature_columns = _build_dataset(db)

    n_candidates = len(feature_columns["models"])
    effective_shots = shots if shots is not None else min(MAX_SHOTS, n_candidates)
    if not (1 <= effective_shots <= n_candidates):
        raise TrainingError(
            f"shots={effective_shots} is invalid — must be between 1 and {n_candidates} "
            "(the number of active candidate models)"
        )
    requested_algorithms = algorithms or [DEFAULT_ALGORITHM]

    candidate_rows = (
        db.query(Model)
        .filter(Model.role == "candidate", Model.active.is_(True), Model.name.in_(feature_columns["models"]))
        .all()
    )
    # Same cost_max calibrate_routing computes internally — reused here so
    # _pick_winner's utility uses the identical scale each candidate's own
    # lambda was calibrated against, not a possibly-different one.
    candidate_costs = load_model_costs(db, [m.model_id for m in candidate_rows])
    cost_max = max(candidate_costs.values()) if candidate_costs else 1.0
    if cost_max <= 0:
        cost_max = 1.0

    splitter = GroupShuffleSplit(n_splits=1, test_size=_TEST_SIZE, random_state=_RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(x, y, groups=groups))
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    test_task_ids = sorted(set(groups.iloc[test_idx]))
    train_task_ids = sorted(set(groups_train))
    cv_splits = max(2, min(_MAX_CV_SPLITS, groups_train.nunique()))

    if progress:
        progress(2, 4, f"training & calibrating: {', '.join(requested_algorithms)}")

    candidates_results = []
    for key in requested_algorithms:
        estimator, cv_score = _fit_algorithm(key, x_train, y_train, groups_train, cv_splits)
        y_pred_test = estimator.predict(x_test)
        calibration = calibrate_routing(
            db,
            model=estimator,
            x=x,
            row_task_ids=groups,
            row_model_ids=row_model_ids,
            feature_columns=feature_columns,
            train_task_ids=train_task_ids,
            test_task_ids=test_task_ids,
            shots=effective_shots,
        )
        candidates_results.append(
            {
                "algorithm": key,
                "estimator": estimator,
                "cv_score": cv_score,
                "test_r2": float(r2_score(y_test, y_pred_test)),
                "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
                "routing_lambdas": calibration["routing_lambdas"],
                "lambda_sweep": calibration["lambda_sweep"],
                "degenerate": _is_degenerate(calibration["routing_lambdas"], effective_shots),
            }
        )

    if progress:
        progress(
            3, 4,
            "picking the best candidate" if len(candidates_results) > 1
            else "calibrating routing strategy (lambda sweep)",
        )
    winner = (
        candidates_results[0] if len(candidates_results) == 1
        else _pick_winner(candidates_results, effective_shots, cost_max)
    )

    predictors_dir = os.path.join(settings.artifacts_dir, "predictors")
    os.makedirs(predictors_dir, exist_ok=True)
    artifact_path = os.path.join(predictors_dir, f"predictor_{int(time.time())}.joblib")
    joblib.dump({"model": winner["estimator"], "feature_columns": feature_columns}, artifact_path)

    if progress:
        progress(4, 4, "generating training report")
    report_dir = generate_training_report(
        db,
        model=winner["estimator"],
        x=x,
        row_task_ids=groups,
        row_model_ids=row_model_ids,
        feature_columns=feature_columns,
        test_task_ids=test_task_ids,
        routing_lambdas=winner["routing_lambdas"],
        shots=effective_shots,
        algorithm=winner["algorithm"],
    )

    return TrainingResult(
        algorithm=winner["algorithm"],
        cv_score=winner["cv_score"],
        test_r2=winner["test_r2"],
        test_mae=winner["test_mae"],
        feature_columns=feature_columns,
        artifact_path=artifact_path,
        shots=effective_shots,
        routing_lambdas=winner["routing_lambdas"],
        lambda_sweep=winner["lambda_sweep"],
        report_dir=report_dir,
        candidates_tried=[
            {
                "algorithm": c["algorithm"],
                "cv_score": c["cv_score"],
                "test_r2": c["test_r2"],
                "routing_accuracy": c["routing_lambdas"][str(effective_shots)]["test_accuracy"],
                "routing_cost": c["routing_lambdas"][str(effective_shots)]["test_cost_avg"],
                "degenerate": c["degenerate"],
                "is_winner": c is winner,
            }
            for c in candidates_results
        ],
    )
