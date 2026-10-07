"""Bounded concurrent execution (sliding window) for model calls.

`model.concurrency` defines how many calls to THAT model can be in flight
at the same time (None = no configured limit, treated as 1, sequential). A
`ThreadPoolExecutor(max_workers=concurrency)` already implements exactly
the requested behavior: once all jobs are submitted, the pool keeps that
exact number running and starts the next one as soon as one finishes — no
custom "sliding window" logic is needed.

Workers must NEVER touch the SQLAlchemy session (it isn't thread-safe):
they only make the call to the model and return/raise; whoever writes to
the DB is always the main thread, after collecting the results.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_bounded(
    items: list[T], worker: Callable[[T], R], concurrency: int | None
) -> list[tuple[T, R | None, Exception | None]]:
    """Runs `worker(item)` for each item, with at most `concurrency` in flight.

    Returns a list of (item, result, exception): exactly one of the last
    two is None, depending on whether `worker` raised or not. The order of
    the list is completion order, not the order of `items`.
    """
    max_workers = concurrency or 1
    out: list[tuple[T, R | None, Exception | None]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                out.append((item, future.result(), None))
            except Exception as exc:  # noqa: BLE001 - propagated as a result, not lost
                out.append((item, None, exc))
    return out
