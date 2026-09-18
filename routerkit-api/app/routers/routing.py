from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db.models import Predictor

router = APIRouter(tags=["routing"])


class RouteRequest(BaseModel):
    prompt: str


@router.post("/route")
def route(body: RouteRequest, db: Session = Depends(get_db)) -> dict:
    """Ruta una tarea nueva al mejor LLM candidato usando el predictor productivo.

    Flujo pendiente de implementar en detalle:
    1. Computar features en vivo para `body.prompt` (services.features_llm /
       features_python, hoy stub).
    2. Puntuar cada modelo candidato con el predictor productivo
       (predictor.artifact_path) -> {model_id, p, price}.
    3. Elegir ganador con services.policy.score_and_select (stub: λ, p_min,
       exclusiones y fallback aún no definidos).
    4. Registrar la decisión en routerkit.route_decisions.

    Por ahora solo valida que exista un predictor productivo y devuelve 501.
    """
    predictor = (
        db.query(Predictor)
        .filter(Predictor.is_production.is_(True), Predictor.active.is_(True))
        .one_or_none()
    )
    if predictor is None:
        raise HTTPException(
            status_code=409, detail="no hay un predictor en producción (is_production=True)"
        )

    raise HTTPException(status_code=501, detail="routing en producción aún no está implementado")
