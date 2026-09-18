"""Seed inicial del catálogo de tasks con el dataset oficial de BigCodeBench.

Se corre una vez al arrancar el container (ver lifespan en app/main.py):
si `routerkit.tasks` no tiene NINGUNA fila (DB recién creada), carga
`{ARTIFACTS_DIR}/seed/tasks_bigcodebench.jsonl` con `services.tasks_catalog`
— el mismo camino que usaría un `POST /tasks/import` del usuario — para que
el usuario arranque con las 1140 tasks del benchmark oficial (bigcode/
bigcodebench, HuggingFace, split v0.1.4) ya activas y listas para usar.

Ese jsonl se generó una sola vez (no lo arma la API) a partir del dataset
público, mapeando cada fila así:
    prompt        <- instruct_prompt (la instrucción en lenguaje natural;
                     mismo estilo que espera generation.py)
    tests         <- test (los unit tests oficiales)
    entry_point   <- entry_point (siempre "task_func" en BigCodeBench)
    tags          <- libs (imports que usa la solución de referencia)
    bcb_task_id   <- task_id (ej. "BigCodeBench/12")

Si la tabla ya tiene alguna fila (el usuario ya cargó algo, o ya corrió este
seed antes), no hace nada — nunca pisa una carga real, y no se repite en
reinicios subsiguientes del mismo container.
"""

import os

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Task
from app.services.tasks_catalog import parse_tasks_jsonl, replace_tasks

_SEED_RELATIVE_PATH = os.path.join("seed", "tasks_bigcodebench.jsonl")


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
