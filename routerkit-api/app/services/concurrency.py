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
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


class CircuitOpenError(RuntimeError):
    """This item was never attempted: the circuit breaker had already
    tripped (see `run_bounded`'s `circuit_breaker` argument) before a
    worker slot got to it."""


def run_bounded(
    items: list[T],
    worker: Callable[[T], R],
    concurrency: int | None,
    *,
    circuit_breaker: tuple[type[Exception], int] | None = None,
) -> list[tuple[T, R | None, Exception | None]]:
    """Runs `worker(item)` for each item, with at most `concurrency` in flight.

    Returns a list of (item, result, exception): exactly one of the last
    two is None, depending on whether `worker` raised or not. The order of
    the list is completion order, not the order of `items`.

    `circuit_breaker=(exc_type, threshold)`: once `threshold` items have
    raised an instance of `exc_type` (e.g. `LlmClientError` — the call
    itself failed, already past its own internal retries; NOT a
    content/parsing error like `LlmFeatureError`, which says nothing about
    whether the service is reachable), every item not yet STARTED is
    cancelled — each shows up in the result with a `CircuitOpenError`
    instead of being attempted at all. Items already running are left to
    finish normally (there's no safe way to abort an in-flight HTTP call
    mid-request): with `concurrency` calls in flight, expect up to that
    many "wasted" attempts past the threshold before cancellation takes
    effect, not zero. `items` all share one circuit, since one call to
    `run_bounded` is always scoped to calls against a single dependency
    (one candidate model, the one feature model, or the one judge model)
    by construction of every caller — never a mix.
    """
    max_workers = concurrency or 1
    out: list[tuple[T, R | None, Exception | None]] = []
    watch_exc, threshold = circuit_breaker or (Exception, 0)
    failure_count = 0
    tripped_by: Exception | None = None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                out.append((item, future.result(), None))
            except Exception as exc:  # noqa: BLE001 - propagated as a result, not lost
                if isinstance(exc, CancelledError):
                    exc = CircuitOpenError(
                        f"skipped: circuit opened after {threshold} consecutive failures "
                        f"(first: {tripped_by})"
                    )
                out.append((item, None, exc))
                if circuit_breaker is not None and isinstance(exc, watch_exc):
                    failure_count += 1
                    if tripped_by is None and failure_count >= threshold:
                        tripped_by = exc
                        for other in futures:
                            other.cancel()
    return out
