"""Jobs async de larga duración, persistidos en Postgres (routerkit.jobs).

Mismo patrón de borrado lógico que el resto del sistema: a lo sumo una fila
ACTIVA por `kind` (idx_jobs_unique_active_kind), que se actualiza in-place
(status/message/progress) mientras corre. `start_job(kind, ...)` desactiva
la fila anterior de ese kind y crea una nueva — pero solo si la anterior ya
terminó (done/error); si sigue pending/running, lanza JobConflictError en
vez de pisar un proceso en curso. `reset_all_on_startup(db)` se llama una
vez al arrancar el container (ver lifespan en app/main.py): desactiva
cualquier fila que haya quedado 'running' de una instancia anterior
(crash, redeploy) y crea una fila fresca en status='empty' para cada kind
conocido.

`start_job` lanza `target` en un hilo de background con SU PROPIA sesión de
SQLAlchemy — para cuando `target` corre de verdad, el request que lo
disparó ya devolvió la respuesta y esa sesión está cerrada. `target(db,
progress)` recibe esa sesión nueva más una función `progress` para
reportar avance; lo que `target` devuelve queda en `job.result`.
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

# Kinds conocidos, para que GET /jobs siempre los muestre a todos aunque
# alguno nunca se haya corrido, y para que el reset de arranque sepa a
# cuáles crearles la fila 'empty'. Agregar acá cuando se sume un nuevo tipo
# de proceso async (ej. "evaluation" cuando se implemente validación).
KNOWN_KINDS = ["features_compute", "solutions_generate", "solutions_validate"]

ProgressFn = Callable[[int, int, str | None], None]

_executor = ThreadPoolExecutor(max_workers=8)


class JobConflictError(RuntimeError):
    """Ya hay un job pending/running de este mismo kind."""


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
        except Exception as exc:  # noqa: BLE001 - se guarda el error, no se pierde
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
    """Desactiva la fila activa anterior de `kind` (si ya terminó) y crea una
    nueva, lanzando `target` en background. JobConflictError si la anterior
    sigue pending/running."""
    existing = db.execute(
        select(Job).where(Job.kind == kind, Job.active.is_(True))
    ).scalar_one_or_none()

    if existing is not None:
        if existing.status in ("pending", "running"):
            raise JobConflictError(f"ya hay un job '{kind}' en curso (job_id={existing.job_id})")
        existing.active = False

    job = Job(kind=kind, status="pending", active=True)
    db.add(job)
    db.commit()
    db.refresh(job)

    _executor.submit(_run, job.job_id, target)
    return job


def to_board_entry(job: Job | None, kind: str) -> dict[str, Any]:
    """Arma la entrada de estado (misma forma que status_board) para una fila
    puntual — la usan tanto status_board como la respuesta de los POST que
    arrancan un job, para no duplicar el formato."""
    if job is None or job.status == "empty":
        return {
            "kind": kind,
            "status": "empty",
            "message": "sin ejecuciones previas",
            "started_at": job.started_at if job else None,
            "finished_at": None,
            "last_result": None,
        }

    if job.status in ("pending", "running"):
        display_status = "running"
        message = job.message or "iniciando"
    elif job.status == "error":
        display_status = "idle"
        message = f"último intento falló: {job.error}"
    else:
        display_status = "idle"
        message = job.message or "completado"

    return {
        "kind": kind,
        "status": display_status,
        "message": message,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "last_result": job.result if job.status == "done" else None,
    }


def status_board(db: Session) -> list[dict[str, Any]]:
    """La fila ACTIVA de cada kind conocido — todos presentes, aunque nunca hayan corrido."""
    board = []
    for kind in KNOWN_KINDS:
        job = db.execute(
            select(Job).where(Job.kind == kind, Job.active.is_(True))
        ).scalar_one_or_none()
        board.append(to_board_entry(job, kind))
    return board


def reset_all_on_startup(db: Session) -> None:
    """Se llama una vez al arrancar el container (lifespan de app/main.py).

    Cualquier fila que haya quedado 'running'/'pending' de una instancia
    anterior (crash, redeploy) se desactiva — nunca hay una instancia
    previa realmente viva compitiendo por el mismo Postgres, así que es
    seguro asumir que ese trabajo se perdió. Cada kind conocido arranca con
    una fila nueva en status='empty'.
    """
    db.execute(update(Job).where(Job.kind.in_(KNOWN_KINDS), Job.active.is_(True)).values(active=False))
    for kind in KNOWN_KINDS:
        db.add(Job(kind=kind, status="empty", active=True))
    db.commit()
