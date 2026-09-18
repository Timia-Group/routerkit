"""Ejecución concurrente acotada (sliding window) para llamadas a modelos.

`model.concurrency` define cuántas llamadas a ESE modelo pueden estar en
vuelo al mismo tiempo (None = sin límite configurado, se trata como 1,
secuencial). Un `ThreadPoolExecutor(max_workers=concurrency)` ya implementa
exactamente el comportamiento pedido: al enviar todos los trabajos, el pool
mantiene esa cantidad exacta en curso y arranca el siguiente apenas termina
uno — no hace falta lógica propia de "ventana deslizante".

Los workers NUNCA deben tocar la sesión de SQLAlchemy (no es thread-safe):
solo hacen la llamada al modelo y devuelven/lanzan; quien escribe en la BD
es siempre el hilo principal, después de recolectar los resultados.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_bounded(
    items: list[T], worker: Callable[[T], R], concurrency: int | None
) -> list[tuple[T, R | None, Exception | None]]:
    """Corre `worker(item)` para cada item, con a lo sumo `concurrency` en vuelo.

    Devuelve una lista de (item, resultado, excepción): exactamente uno de
    los dos últimos es None, según si `worker` lanzó o no. El orden de la
    lista es el de finalización, no el de `items`.
    """
    max_workers = concurrency or 1
    out: list[tuple[T, R | None, Exception | None]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                out.append((item, future.result(), None))
            except Exception as exc:  # noqa: BLE001 - se propaga como resultado, no se pierde
                out.append((item, None, exc))
    return out
