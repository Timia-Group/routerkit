"""Snapshots JSON del estado activo, para recuperación manual.

Cada vez que se recarga/recalcula tasks, questions, models o features, se
escribe un JSON con el estado COMPLETO de lo activo en ese momento bajo
`{ARTIFACTS_DIR}/current/<nombre>.json`. Si ya había un snapshot anterior
con ese nombre, se mueve primero a `{ARTIFACTS_DIR}/previous/<nombre>.json`
(pisando lo que hubiera ahí) — un solo paso atrás, no un historial
completo. Así, si una carga sale mal, queda el JSON de la carga anterior
para poder volver a subirlo.
"""

import json
import os
import shutil
from datetime import datetime
from decimal import Decimal

from sqlalchemy import inspect

from app.config import settings


def _serialize_value(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def row_to_dict(obj) -> dict:
    """Convierte una fila del ORM a dict plano (todas sus columnas), listo para JSON."""
    mapper = inspect(obj).mapper
    return {
        column.key: _serialize_value(getattr(obj, column.key)) for column in mapper.column_attrs
    }


def write_snapshot(name: str, rows: list[dict]) -> None:
    """Escribe `rows` como `current/<name>.json`, rotando el anterior a `previous/`."""
    current_dir = os.path.join(settings.artifacts_dir, "current")
    previous_dir = os.path.join(settings.artifacts_dir, "previous")
    os.makedirs(current_dir, exist_ok=True)
    os.makedirs(previous_dir, exist_ok=True)

    current_path = os.path.join(current_dir, f"{name}.json")
    previous_path = os.path.join(previous_dir, f"{name}.json")

    if os.path.exists(current_path):
        shutil.move(current_path, previous_path)

    with open(current_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
