from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.services.jobs import status_board

router = APIRouter(tags=["jobs"])


@router.get("/jobs")
def list_jobs(db: Session = Depends(get_db)) -> list[dict]:
    """State of the ACTIVE row for EACH known kind, in a single request.

    Always returns one entry per kind (features_compute,
    solutions_generate, ...), even if it has never run — with
    `status='empty'` and a message stating so. If the active one is
    running, `status='running'` with a progress `message`; otherwise,
    `status='idle'` and (if it finished successfully) `last_result` with
    the final result. To see history (already deactivated jobs), query the
    DB directly — there's no read endpoint for that, same as with
    tasks/questions/models.
    """
    return status_board(db)
