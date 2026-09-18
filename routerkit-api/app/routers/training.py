from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db.models import Predictor
from app.services import training

router = APIRouter(tags=["training"])


@router.post("/train")
def train(db: Session = Depends(get_db)) -> dict:
    """Entrena el predictor (AutoML) con las features/targets activos.

    Delega en services.training.train_predictor, stub (librería de AutoML
    pendiente de definir).
    """
    result = training.train_predictor()
    if result is None:
        raise HTTPException(
            status_code=501, detail="training.train_predictor aún no está implementado"
        )

    predictor = Predictor(
        algorithm=result.algorithm,
        cv_score=result.cv_score,
        test_r2=result.test_r2,
        test_mae=result.test_mae,
        feature_columns=result.feature_columns,
        artifact_path=result.artifact_path,
    )
    db.add(predictor)
    db.commit()
    db.refresh(predictor)

    return {"predictor_id": predictor.predictor_id}
