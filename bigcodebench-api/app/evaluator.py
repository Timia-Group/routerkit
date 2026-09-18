"""Wrapper sobre el sandbox de BigCodeBench.

`bigcodebench.evaluate` (el CLI oficial) está pensado para correr el benchmark
completo contra el dataset oficial. Para evaluar una sola (task, solución) a
demanda usamos directamente las piezas de más bajo nivel que expone el
paquete:

- `get_bigcodebench()` para resolver un `bcb_task_id` oficial (ej.
  "BigCodeBench/12") al `test` y `entry_point` reales de esa tarea.
- `untrusted_check(...)` para ejecutar la solución candidata contra ese test
  dentro del sandbox (subprocess con límites de recursos) que ya trae la
  imagen `bigcodebench/bigcodebench-evaluate`.
"""

import time

from bigcodebench.data import get_bigcodebench
from bigcodebench.eval import untrusted_check

from app.schemas import EvaluateResult

# Defaults tomados de los valores documentados para el CLI oficial
# (ver ADVANCED_USAGE.md de bigcodebench). Pueden overridearse por request.
DEFAULT_MAX_AS_LIMIT_MB = 30 * 1024
DEFAULT_MAX_DATA_LIMIT_MB = 30 * 1024
DEFAULT_MAX_STACK_LIMIT_MB = 10
DEFAULT_MIN_TIME_LIMIT_S = 10
DEFAULT_GT_TIME_LIMIT_S = 60

_dataset: dict | None = None


def load_dataset() -> dict:
    """Carga y cachea el dataset oficial (subset "full") en memoria del proceso.

    Se llama una sola vez al levantar el contenedor (ver app/main.py). La
    primera vez puede descargar el dataset desde Hugging Face; llamadas
    posteriores a get_bigcodebench() dentro del mismo proceso reusan el cache
    interno del paquete. No se expone ningún endpoint para listar/descargar
    este dataset: bigcodebench-api solo lo usa para resolver el `test` real
    al evaluar; si el usuario quiere el catálogo oficial para subirlo a
    routerkit-api, lo genera aparte (ver carpeta de datasets del repo).
    """
    global _dataset
    if _dataset is None:
        _dataset = get_bigcodebench(subset="full")
    return _dataset


def get_task(bcb_task_id: str) -> dict:
    """Devuelve el dict oficial de la tarea (con 'test' y 'entry_point').

    Lanza KeyError si el bcb_task_id no existe en el dataset oficial.
    """
    dataset = _dataset if _dataset is not None else load_dataset()
    return dataset[bcb_task_id]


def evaluate(
    bcb_task_id: str,
    solution: str,
    max_as_limit_mb: float | None = None,
    max_data_limit_mb: float | None = None,
    max_stack_limit_mb: float | None = None,
    min_time_limit_s: float | None = None,
    gt_time_limit_s: float | None = None,
) -> EvaluateResult:
    """Corre `untrusted_check` para (bcb_task_id, solution) y devuelve el veredicto.

    Bloqueante (hace fork de un subprocess) — debe llamarse desde un
    threadpool, no directamente en el event loop de FastAPI (ver app/jobs.py).
    """
    task = get_task(bcb_task_id)

    started = time.perf_counter()
    status, details = untrusted_check(
        solution,
        task["test"],
        task["entry_point"],
        max_as_limit_mb if max_as_limit_mb is not None else DEFAULT_MAX_AS_LIMIT_MB,
        max_data_limit_mb if max_data_limit_mb is not None else DEFAULT_MAX_DATA_LIMIT_MB,
        max_stack_limit_mb if max_stack_limit_mb is not None else DEFAULT_MAX_STACK_LIMIT_MB,
        min_time_limit_s if min_time_limit_s is not None else DEFAULT_MIN_TIME_LIMIT_S,
        gt_time_limit_s if gt_time_limit_s is not None else DEFAULT_GT_TIME_LIMIT_S,
    )
    exec_ms = (time.perf_counter() - started) * 1000

    passed = status == "pass"
    fail_reason = None if passed else status

    return EvaluateResult(
        passed=passed,
        status=status,
        fail_reason=fail_reason,
        exec_ms=exec_ms,
        details=dict(details) if details else None,
    )
