"""HTTP client for bigcodebench-api: creates a job and polls until it finishes.

bigcodebench-api only evaluates official benchmark tasks (bcb_task_id +
solution); see app/services/judge.py for custom tasks.
"""

import time
from typing import Any

import httpx

from app.config import settings


class BigCodeBenchError(RuntimeError):
    pass


class BigCodeBenchTimeout(BigCodeBenchError):
    pass


def evaluate(
    bcb_task_id: str,
    solution: str,
    poll_interval_s: float = 1.0,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Creates the evaluation job and waits for the result (blocking).

    Returns the job's `result` dict (passed/status/fail_reason/exec_ms/details).
    """
    with httpx.Client(base_url=settings.bigcodebench_api_url, timeout=30.0) as client:
        response = client.post(
            "/jobs", json={"bcb_task_id": bcb_task_id, "solution": solution}
        )
        response.raise_for_status()
        job_id = response.json()["job_id"]

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            response = client.get(f"/jobs/{job_id}")
            response.raise_for_status()
            job = response.json()

            if job["status"] == "done":
                return job["result"]
            if job["status"] == "error":
                raise BigCodeBenchError(job["error"])

            time.sleep(poll_interval_s)

    raise BigCodeBenchTimeout(f"job {job_id} did not finish within {timeout_s}s")
