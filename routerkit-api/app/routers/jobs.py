from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.services.jobs import status_board

router = APIRouter(tags=["jobs"])


@router.get("/jobs")
def list_jobs(db: Session = Depends(get_db)) -> list[dict]:
    """Estado de la fila ACTIVA de CADA kind conocido, en un solo request.

    Siempre devuelve una entrada por kind (features_compute,
    solutions_generate, ...), aunque nunca se haya corrido — con
    `status='empty'` y un mensaje indicándolo. Si el activo está corriendo,
    `status='running'` con `message` de progreso; si no, `status='idle'` y
    (si terminó bien) `last_result` con el resultado final. Para ver
    históricos (jobs ya desactivados), consultar la BD directamente — no
    hay endpoint de lectura para eso, igual que con tasks/questions/models.
    """
    return status_board(db)
