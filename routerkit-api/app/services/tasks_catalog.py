"""Replacement (via soft delete) of the task catalog, and resolution of a
task by stable key for one-off loads (`type=load` of
features/solutions/validations).

Logic shared between `routers.catalogs.import_tasks` (the user uploads
their own tasks.jsonl) and `services.seed` (the automatic load of the
official BigCodeBench dataset on startup with an empty DB) — both are, at
bottom, nothing more than "replace all active tasks with these rows".

`TaskIdentifier` + `find_active_task()` are for the other case: a one-off
load that needs to reference an already-existing task. Postgres's internal
`task_id` is NEVER exposed there — it isn't stable across reloads (it
changes depending on insertion order) and is easy to confuse with the id
assigned by the benchmark itself. Instead, every one-off load identifies
the task by a stable key: `bcb_task_id` (official benchmark tasks, e.g.
"BigCodeBench/837") or `custom_task_id` (custom tasks — assigned by the
user when creating the task in tasks.jsonl, see `TaskImportRow`) — see
`routers.features`, `routers.solutions` and `routers.evaluation`.
"""

from pydantic import BaseModel, model_validator
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.models import Task
from app.schemas import TaskImportRow
from app.services import cascade
from app.services.jsonl import parse_jsonl
from app.services.snapshots import row_to_dict, write_snapshot


class TaskIdentifier(BaseModel):
    """Mixin: identifies an existing task by its `bcb_task_id` (official
    benchmark tasks, e.g. "BigCodeBench/837") or by its `custom_task_id`
    (custom tasks, assigned by the user when creating it) — exactly one of
    the two. Postgres's internal `task_id` is never used here: it isn't
    stable across reloads (it depends on insertion order)."""

    bcb_task_id: str | None = None
    custom_task_id: str | None = None

    @model_validator(mode="after")
    def _check_exactly_one_identifier(self) -> "TaskIdentifier":
        if (self.bcb_task_id is None) == (self.custom_task_id is None):
            raise ValueError("specify exactly one of 'bcb_task_id' or 'custom_task_id'")
        return self


def find_active_task(db: Session, *, bcb_task_id: str | None, custom_task_id: str | None) -> Task | None:
    """Resolves the active task identified by `bcb_task_id` or
    `custom_task_id` (exactly one of the two — already validated by
    `TaskIdentifier`)."""
    query = db.query(Task).filter(Task.active.is_(True))
    if bcb_task_id is not None:
        query = query.filter(Task.bcb_task_id == bcb_task_id)
    else:
        query = query.filter(Task.custom_task_id == custom_task_id)
    return query.one_or_none()


def load_active_tasks_by_identifier(db: Session) -> tuple[dict[str, Task], dict[str, Task]]:
    """Bulk alternative to calling `find_active_task()` once per row: a
    single query for ALL active tasks, returned as (tasks_by_bcb_id,
    tasks_by_custom_id) lookup dicts. For callers resolving many rows at
    once (features/solutions/validations `type=load` of a large file) —
    one query instead of one round trip per row, which at a few hundred
    thousand rows is the difference between seconds and the better part of
    an hour. Pair with `resolve_task()`.
    """
    tasks = db.query(Task).filter(Task.active.is_(True)).all()
    tasks_by_bcb = {t.bcb_task_id: t for t in tasks if t.bcb_task_id is not None}
    tasks_by_custom = {t.custom_task_id: t for t in tasks if t.custom_task_id is not None}
    return tasks_by_bcb, tasks_by_custom


def resolve_task(
    tasks_by_bcb: dict[str, Task],
    tasks_by_custom: dict[str, Task],
    *,
    bcb_task_id: str | None,
    custom_task_id: str | None,
) -> Task | None:
    """In-memory equivalent of `find_active_task()`, against the lookup
    dicts from `load_active_tasks_by_identifier()`."""
    if bcb_task_id is not None:
        return tasks_by_bcb.get(bcb_task_id)
    return tasks_by_custom.get(custom_task_id)


def task_ref(task: Task) -> dict:
    """Human-facing identifier of a task (`bcb_task_id`/`custom_task_id`),
    for error payloads and snapshot files — never the raw `task_id`."""
    return {"bcb_task_id": task.bcb_task_id, "custom_task_id": task.custom_task_id}


def parse_tasks_jsonl(raw: str) -> list[TaskImportRow]:
    """Parses a tasks.jsonl (one JSON task per line) into `TaskImportRow`.

    400 if any line isn't valid JSON, 422 if it doesn't satisfy `TaskImportRow`.
    """
    return parse_jsonl(raw, TaskImportRow)


def replace_tasks(db: Session, rows: list[TaskImportRow]) -> dict:
    """Deactivates ALL active tasks (cascade included) and inserts `rows`
    as the new active tasks. Leaves a snapshot of the resulting active
    tasks in data/current/tasks.json (rotating the previous one to
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
                custom_task_id=row.custom_task_id,
                active=True,
            )
        )

    db.commit()

    active_tasks = db.query(Task).filter(Task.active.is_(True)).all()
    write_snapshot("tasks", [row_to_dict(t) for t in active_tasks])

    return {"deactivated_previous": len(deactivated_ids), "created": len(rows)}
