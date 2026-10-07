from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.services import aggregation

router = APIRouter(tags=["aggregation"])


@router.post("/aggregate")
def aggregate(db: Session = Depends(get_db)) -> dict:
    """Recomputes targets and model_metrics from the active solutions/grades.

    Full replacement (with soft delete): deactivates all existing targets and
    model_metrics and creates new active rows from the active solutions (and
    their grades).
    """
    n_targets = aggregation.recompute_targets(db)
    n_metrics = aggregation.recompute_model_metrics(db)
    return {"targets": n_targets, "model_metrics": n_metrics}
