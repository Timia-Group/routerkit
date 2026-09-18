from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.services import aggregation

router = APIRouter(tags=["aggregation"])


@router.post("/aggregate")
def aggregate(db: Session = Depends(get_db)) -> dict:
    """Recalcula targets y model_metrics a partir de las solutions/grades activas.

    Reemplazo completo (con borrado lógico): desactiva todos los targets y
    model_metrics existentes y crea filas nuevas activas a partir de las
    solutions activas (y sus grades).
    """
    n_targets = aggregation.recompute_targets(db)
    n_metrics = aggregation.recompute_model_metrics(db)
    return {"targets": n_targets, "model_metrics": n_metrics}
