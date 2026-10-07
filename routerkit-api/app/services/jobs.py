"""Long-running async jobs, persisted in Postgres (routerkit.jobs).

Same soft-delete pattern as the rest of the system: at most one ACTIVE row
per `kind` (idx_jobs_unique_active_kind), updated in-place
(status/message/progress) while it runs. `start_job(kind, ...)` deactivates
the previous row of that kind and creates a new one — but only if the
previous one already finished (done/error); if it's still pending/running,
it raises JobConflictError instead of overwriting a job in progress.
`reset_all_on_startup(db)` is called once when the container starts (see
lifespan in app/main.py): it deactivates any row left 'running' from a
previous instance (crash, redeploy) and creates a fresh row in
status='empty' for each known kind.

`start_job` launches `target` in a background thread with ITS OWN
SQLAlchemy session — because by the time `target` actually runs, the
request that triggered it has already returned its response and that
session is closed. `target(db, progress)` receives that new session plus a
`progress` function to report progress; whatever `target` returns ends up
in `job.result`.
"""

import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.base import SessionLocal
from app.db.models import Job

# Known kinds, so that GET /jobs always shows all of them even if one has
# never run, and so the startup reset knows which ones to create an
# 'empty' row for. Add here when a new kind of async process is added
# (e.g. "evaluation" once validation is implemented).
KNOWN_KINDS = ["features_compute", "solutions_generate", "solutions_validate", "train_predictor"]

# How often (in rows) a type=load loop calls progress() — which commits —
# instead of every single row. At hundreds of thousands of rows (a
# type=load of the real experiment's data, not our small test files),
# committing every row makes the whole job take forever for no benefit;
# every PROGRESS_BATCH_SIZE rows keeps that overhead negligible while still
# limiting how much uncommitted work a crash partway through could lose.
PROGRESS_BATCH_SIZE = 200

ProgressFn = Callable[[int, int, str | None], None]

_executor = ThreadPoolExecutor(max_workers=8)


class JobConflictError(RuntimeError):
    """There's already a pending/running job of this same kind."""


def _run(job_id: int, target: Callable[[Session, ProgressFn], dict[str, Any]]) -> None:
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        job.status = "running"
        db.commit()

        def progress(current: int, total: int, message: str | None = None) -> None:
            row = db.get(Job, job_id)
            row.progress_current = current
            row.progress_total = total
            row.message = message or f"{current}/{total}"
            db.commit()

        try:
            result = target(db, progress)
        except Exception as exc:  # noqa: BLE001 - the error is saved, not lost
            db.rollback()
            row = db.get(Job, job_id)
            row.status = "error"
            row.error = f"{exc}\n{traceback.format_exc()}"
            row.finished_at = datetime.now(UTC)
            db.commit()
            return

        row = db.get(Job, job_id)
        row.status = "done"
        row.result = result
        row.finished_at = datetime.now(UTC)
        db.commit()
    finally:
        db.close()


def start_job(
    kind: str, target: Callable[[Session, ProgressFn], dict[str, Any]], db: Session
) -> Job:
    """Deactivates the previous active row of `kind` (if it already finished)
    and creates a new one, launching `target` in the background.
    JobConflictError if the previous one is still pending/running."""
    existing = db.execute(
        select(Job).where(Job.kind == kind, Job.active.is_(True))
    ).scalar_one_or_none()

    if existing is not None:
        if existing.status in ("pending", "running"):
            raise JobConflictError(f"a '{kind}' job is already in progress (job_id={existing.job_id})")
        existing.active = False

    job = Job(kind=kind, status="pending", active=True)
    db.add(job)
    db.commit()
    db.refresh(job)

    _executor.submit(_run, job.job_id, target)
    return job


def to_board_entry(job: Job | None, kind: str) -> dict[str, Any]:
    """Builds the status entry (same shape as status_board) for a single
    row — used by both status_board and the response of the POST endpoints
    that start a job, to avoid duplicating the format."""
    if job is None or job.status == "empty":
        return {
            "kind": kind,
            "status": "empty",
            "message": "no previous runs",
            "started_at": job.started_at if job else None,
            "finished_at": None,
            "last_result": None,
        }

    if job.status in ("pending", "running"):
        display_status = "running"
        message = job.message or "starting"
    elif job.status == "error":
        display_status = "idle"
        message = f"last attempt failed: {job.error}"
    else:
        display_status = "idle"
        message = job.message or "completed"

    return {
        "kind": kind,
        "status": display_status,
        "message": message,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "last_result": job.result if job.status == "done" else None,
    }


def status_board(db: Session) -> list[dict[str, Any]]:
    """The ACTIVE row of each known kind — all present, even if they've never run."""
    board = []
    for kind in KNOWN_KINDS:
        job = db.execute(
            select(Job).where(Job.kind == kind, Job.active.is_(True))
        ).scalar_one_or_none()
        board.append(to_board_entry(job, kind))
    return board


def reset_all_on_startup(db: Session) -> None:
    """Called once when the container starts (lifespan of app/main.py).

    Any row left 'running'/'pending' from a previous instance (crash,
    redeploy) is deactivated — there's never a previous instance that's
    actually still alive competing for the same Postgres, so it's safe to
    assume that work was lost. Each known kind starts with a fresh row in
    status='empty'.
    """
    db.execute(update(Job).where(Job.kind.in_(KNOWN_KINDS), Job.active.is_(True)).values(active=False))
    for kind in KNOWN_KINDS:
        db.add(Job(kind=kind, status="empty", active=True))
    db.commit()
