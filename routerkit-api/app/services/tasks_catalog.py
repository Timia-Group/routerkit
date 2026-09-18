"""Reemplazo (con borrado lógico) del catálogo de tasks.

Lógica compartida entre `routers.catalogs.import_tasks` (el usuario sube su
propio tasks.jsonl) y `services.seed` (la carga automática del dataset
oficial de BigCodeBench al arrancar con la DB vacía) — ambos son, en el
fondo, "reemplazar todas las tasks activas por estas filas", nada más.
"""

import json

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.models import Task
from app.schemas import TaskImportRow
from app.services import cascade
from app.services.snapshots import row_to_dict, write_snapshot


def parse_tasks_jsonl(raw: str) -> list[TaskImportRow]:
    """Parsea un tasks.jsonl (una tarea JSON por línea) a `TaskImportRow`.

    400 si alguna línea no es JSON válido, 422 si no cumple `TaskImportRow`.
    """
    rows: list[TaskImportRow] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            raw_row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400, detail=f"línea {line_no}: JSON inválido ({exc})"
            ) from exc
        try:
            rows.append(TaskImportRow.model_validate(raw_row))
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "item": f"línea {line_no}",
                    "errors": exc.errors(include_url=False, include_context=False),
                },
            ) from exc
    return rows


def replace_tasks(db: Session, rows: list[TaskImportRow]) -> dict:
    """Desactiva TODAS las tasks activas (cascada incluida) e inserta `rows`
    como las nuevas tasks activas. Deja un snapshot de las tasks activas
    resultantes en data/current/tasks.json (rotando el anterior a
    data/previous)."""
    deactivated_ids = list(
        db.execute(select(Task.task_id).where(Task.active.is_(True))).scalars()
    )
    if deactivated_ids:
        db.execute(update(Task).where(Task.task_id.in_(deactivated_ids)).values(active=False))
        cascade.deactivate_for_tasks(db, deactivated_ids)

    for row in rows:
        db.add(
            Task(
                prompt=row.prompt,
                tests=row.tests,
                entry_point=row.entry_point,
                source=row.source,
                tags=row.tags,
                bcb_task_id=row.bcb_task_id,
                active=True,
            )
        )

    db.commit()

    active_tasks = db.query(Task).filter(Task.active.is_(True)).all()
    write_snapshot("tasks", [row_to_dict(t) for t in active_tasks])

    return {"deactivated_previous": len(deactivated_ids), "created": len(rows)}
