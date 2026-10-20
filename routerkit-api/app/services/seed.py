"""Initial seed of the task catalog with the official BigCodeBench dataset.

Runs once when the container starts (see lifespan in app/main.py): if
`routerkit.tasks` has NO rows at all (freshly created DB), it loads
`{ARTIFACTS_DIR}/examples/tasks_exp.jsonl` with `services.tasks_catalog` —
the same path a user's `POST /tasks/import` would use — so the user starts
out with the official benchmark's tasks (bigcode/bigcodebench, HuggingFace,
split v0.1.4) already active and ready to use. It lives alongside the rest
of the examples in `data/examples/` (not in its own folder) because it is,
quite simply, the actual file used in the experiment — same reasoning as
`questions_exp.json`.

That jsonl was generated once (the API doesn't build it) from the public
dataset, mapping each row as follows:
    prompt        <- instruct_prompt (the natural-language instruction;
                     same style expected by generation.py)
    tests         <- test (the official unit tests)
    entry_point   <- entry_point (always "task_func" in BigCodeBench)
    tags          <- libs (imports used by the reference solution)
    bcb_task_id   <- task_id (e.g. "BigCodeBench/12")

The official dataset has 1140 tasks; 1128 remain here. 5 were already
excluded because their tests depend on web pages that no longer exist
(BigCodeBench/101, /176, /363, /590, /1005 — can't be reliably
validated), and a further 7 (BigCodeBench/59, /215, /233, /466, /642,
/690, /734) were excluded because grading them disagreed between the
`bigcodebench-evaluate` image tag this project pins
(`bigcodebench-api/Dockerfile`) and the one originally used to produce
this experiment's solutions/validations — verified against ~39k real
comparisons, these 7 were the only ones with a meaningful (>25%)
mismatch rate, almost certainly from a dependency version difference in
whatever numeric library their official test relies on, not a flaw in
the candidate solutions themselves.

If the table already has any row (the user already loaded something, or
this seed already ran before), it does nothing — it never overwrites a
real load, and it doesn't repeat on subsequent restarts of the same
container.
"""

import os

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Task
from app.services.tasks_catalog import parse_tasks_jsonl, replace_tasks

_SEED_RELATIVE_PATH = os.path.join("examples", "tasks_exp.jsonl")


def seed_default_tasks_if_empty(db: Session) -> None:
    has_any_task = db.execute(select(Task.task_id).limit(1)).first() is not None
    if has_any_task:
        return

    seed_path = os.path.join(settings.artifacts_dir, _SEED_RELATIVE_PATH)
    if not os.path.exists(seed_path):
        return

    with open(seed_path, encoding="utf-8") as f:
        raw = f.read()

    rows = parse_tasks_jsonl(raw)
    replace_tasks(db, rows)
