from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import update
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db.models import Predictor
from app.services import jobs, training
from app.services.jobs import JobConflictError, ProgressFn

router = APIRouter(tags=["training"])

_KIND = "train_predictor"


def _train_job(db: Session, progress: ProgressFn) -> dict:
    result = training.train_predictor(db, progress)

    # Full replace, same pattern as the rest of the system: only one
    # predictor is ever active, and that's always the one POST /route uses.
    db.execute(update(Predictor).where(Predictor.active.is_(True)).values(active=False))

    predictor = Predictor(
        algorithm=result.algorithm,
        cv_score=result.cv_score,
        test_r2=result.test_r2,
        test_mae=result.test_mae,
        feature_columns=result.feature_columns,
        artifact_path=result.artifact_path,
        routing_lambdas=result.routing_lambdas,
        lambda_sweep=result.lambda_sweep,
    )
    db.add(predictor)
    db.commit()
    db.refresh(predictor)

    progress(4, 4, "done")
    return {
        "predictor_id": predictor.predictor_id,
        "algorithm": result.algorithm,
        "cv_score": result.cv_score,
        "test_r2": result.test_r2,
        "test_mae": result.test_mae,
        "artifact_path": result.artifact_path,
        "routing_lambdas": result.routing_lambdas,
        "report_dir": result.report_dir,
    }


@router.post("/train", status_code=202)
def train(db: Session = Depends(get_db)) -> dict:
    """Launches (async) training of a new predictor: recomputes
    targets/model_metrics from the currently active solutions/grades (same
    as POST /aggregate, done here automatically — no separate step
    needed), builds the dataset from active features/targets and trains,
    then calibrates the routing lambda (services.routing_calibration) for
    the resulting model and generates a training report (services.reports —
    charts + data under `{ARTIFACTS_DIR}/current/report/`, same
    current/previous rotation as the catalog snapshots) — a predictor isn't
    usable by POST /route without the calibration step, so neither of these
    is a separate call either. Returns the initial state right away — to
    see progress (4 phases: data prep, training, routing calibration,
    report generation) or the final result, check `GET /jobs` (state of ALL
    kinds in a single request, including this one). 409 if a
    `kind='train_predictor'` job is already running.

    Always the same fixed algorithm (GradientBoostingRegressor), no option
    to choose a different one — see services.training for why. If there
    isn't enough active data (missing question_key's, no active
    targets/candidates, too few distinct tasks with complete data), the job
    ends in `status='error'` with the reason in `GET /jobs`.
    """
    try:
        job = jobs.start_job(_KIND, _train_job, db)
    except JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return jobs.to_board_entry(job, _KIND)
