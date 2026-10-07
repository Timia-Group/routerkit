"""Trains the predictor: GradientBoostingRegressor (scikit-learn), always
— no AutoML, no bake-off across other algorithms, no option exposed to the
user to change it. Mirrors the methodology of the project's reference
offline experiment, with one deliberate difference: we always re-run
GridSearchCV over GradientBoostingRegressor's own hyperparameter grid
instead of hardcoding the experiment's winning hyperparameters — our own
catalog varies a lot in size/distribution from one person's data to
another's (and over time, as more tasks/questions/solutions get added),
unlike the one-off academic experiment this mirrors.

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

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Feature, Model, Question, Target
from app.services.aggregation import recompute_model_metrics, recompute_targets
from app.services.jobs import ProgressFn
from app.services.reports import generate_training_report
from app.services.routing_calibration import calibrate_routing

_TEST_SIZE = 0.30
_RANDOM_STATE = 42
_MAX_CV_SPLITS = 5
_MIN_TASKS = 4  # need enough distinct tasks for a meaningful grouped train/test split

_PARAM_GRID = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [2, 3, 4],
    "n_estimators": [200, 400, 600],
    "subsample": [0.8, 1.0],
}


class TrainingError(RuntimeError):
    """Not enough active data (features/targets/tasks) to train a predictor."""


@dataclass
class TrainingResult:
    algorithm: str
    cv_score: float
    test_r2: float
    test_mae: float
    feature_columns: dict
    artifact_path: str
    routing_lambdas: dict
    lambda_sweep: dict
    report_dir: str


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
        raise TrainingError("no active targets to train on (did you run POST /aggregate?)")

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
    # so GradientBoostingRegressor's row-position-dependent subsampling
    # (subsample<1.0 in _PARAM_GRID) is reproducible run to run on the
    # same underlying data, not just "deterministic up to whatever order
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


def train_predictor(db: Session, progress: ProgressFn | None = None) -> TrainingResult:
    """Recomputes targets/model_metrics from the currently active
    solutions/grades, builds the dataset from active features/(the
    just-recomputed) targets, splits it grouped by task_id, grid-searches
    GradientBoostingRegressor's hyperparameters (train partition only),
    persists the winning model to settings.artifacts_dir, and calibrates
    the routing lambda (services.routing_calibration) for it. Raises
    TrainingError if there isn't enough active data (see _build_dataset).

    One entry point end to end on purpose: the caller (POST /train) never
    has to remember to call POST /aggregate first, or to calibrate routing
    as a separate step before the predictor is actually usable by
    POST /route. `progress` (optional) is called at the 4 phases (data
    prep, training, routing calibration, report generation) for GET /jobs
    to report.
    """
    if progress:
        progress(1, 4, "preparing data")
    recompute_targets(db)
    recompute_model_metrics(db)

    x, y, groups, row_model_ids, feature_columns = _build_dataset(db)

    splitter = GroupShuffleSplit(n_splits=1, test_size=_TEST_SIZE, random_state=_RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(x, y, groups=groups))
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    if progress:
        progress(2, 4, "training (grid search)")
    cv_splits = max(2, min(_MAX_CV_SPLITS, groups_train.nunique()))
    search = GridSearchCV(
        GradientBoostingRegressor(random_state=_RANDOM_STATE),
        param_grid=_PARAM_GRID,
        cv=GroupKFold(n_splits=cv_splits),
        scoring="r2",
        n_jobs=-1,
    )
    search.fit(x_train, y_train, groups=groups_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(x_test)

    if progress:
        progress(3, 4, "calibrating routing strategy (lambda sweep)")
    test_task_ids = sorted(set(groups.iloc[test_idx]))
    calibration = calibrate_routing(
        db,
        model=best_model,
        x=x,
        row_task_ids=groups,
        row_model_ids=row_model_ids,
        feature_columns=feature_columns,
        train_task_ids=sorted(set(groups_train)),
        test_task_ids=test_task_ids,
    )

    predictors_dir = os.path.join(settings.artifacts_dir, "predictors")
    os.makedirs(predictors_dir, exist_ok=True)
    artifact_path = os.path.join(predictors_dir, f"predictor_{int(time.time())}.joblib")
    joblib.dump({"model": best_model, "feature_columns": feature_columns}, artifact_path)

    if progress:
        progress(4, 4, "generating training report")
    report_dir = generate_training_report(
        db,
        model=best_model,
        x=x,
        row_task_ids=groups,
        row_model_ids=row_model_ids,
        feature_columns=feature_columns,
        test_task_ids=test_task_ids,
        routing_lambdas=calibration["routing_lambdas"],
    )

    return TrainingResult(
        algorithm="GradientBoostingRegressor",
        cv_score=float(search.best_score_),
        test_r2=float(r2_score(y_test, y_pred)),
        test_mae=float(mean_absolute_error(y_test, y_pred)),
        feature_columns=feature_columns,
        artifact_path=artifact_path,
        routing_lambdas=calibration["routing_lambdas"],
        lambda_sweep=calibration["lambda_sweep"],
        report_dir=report_dir,
    )
