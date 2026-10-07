"""JSON snapshots of the active state, for manual recovery.

Every time tasks, questions, models or features are reloaded/recomputed, a
JSON is written with the COMPLETE state of what's active at that moment
under `{ARTIFACTS_DIR}/current/<name>.json`. If there was already a
previous snapshot with that name, it's first moved to
`{ARTIFACTS_DIR}/previous/<name>.json` (overwriting whatever was there) —
a single step back, not a full history. That way, if a load goes wrong,
the JSON from the previous load is still there so it can be re-uploaded.
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


def row_to_dict(obj, *, drop: set[str] | None = None, extra: dict | None = None) -> dict:
    """Converts an ORM row to a flat dict (all its columns), ready for JSON.

    `drop`: column names to omit — used to drop a foreign key column (e.g.
    `task_id`) once its value is replaced by a human-facing identifier.
    `extra`: extra key/value pairs merged in (e.g. `bcb_task_id`,
    `question_key`, `model_name` resolved from the related row) — snapshots
    are meant for a human to read directly from disk, so they should show
    the same stable identifiers the user works with, never a raw internal
    surrogate id from another table (see TaskIdentifier/question_key).
    """
    mapper = inspect(obj).mapper
    row = {
        column.key: _serialize_value(getattr(obj, column.key))
        for column in mapper.column_attrs
        if not (drop and column.key in drop)
    }
    if extra:
        row.update(extra)
    return row


def write_snapshot(name: str, rows: list[dict]) -> None:
    """Writes `rows` as `current/<name>.json`, rotating the previous one to `previous/`."""
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
