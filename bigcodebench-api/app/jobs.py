"""In-process-memory job state (no Redis, deliberate decision).

A job in progress is lost if the container restarts; accepted for this
first implementation. `evaluator.evaluate` is blocking (it forks), so it
runs in a ThreadPoolExecutor to avoid blocking FastAPI's event loop.
"""

import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from app import evaluator
from app.schemas import EvaluateRequest, EvaluateResult, JobStatus

_executor = ThreadPoolExecutor(max_workers=4)


@dataclass
class Job:
    job_id: str
    status: JobStatus = JobStatus.PENDING
    result: EvaluateResult | None = None
    error: str | None = None


_jobs: dict[str, Job] = {}


def _run(job_id: str, request: EvaluateRequest) -> None:
    job = _jobs[job_id]
    job.status = JobStatus.RUNNING
    try:
        job.result = evaluator.evaluate(
            bcb_task_id=request.bcb_task_id,
            solution=request.solution,
            max_as_limit_mb=request.max_as_limit_mb,
            max_data_limit_mb=request.max_data_limit_mb,
            max_stack_limit_mb=request.max_stack_limit_mb,
            min_time_limit_s=request.min_time_limit_s,
            gt_time_limit_s=request.gt_time_limit_s,
        )
        job.status = JobStatus.DONE
    except KeyError:
        job.status = JobStatus.ERROR
        job.error = f"unknown bcb_task_id: {request.bcb_task_id!r}"
    except Exception as exc:  # noqa: BLE001 - reported as-is to the caller
        job.status = JobStatus.ERROR
        job.error = str(exc)


def create_job(request: EvaluateRequest) -> Job:
    job_id = uuid.uuid4().hex
    job = Job(job_id=job_id)
    _jobs[job_id] = job
    _executor.submit(_run, job_id, request)
    return job


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)
