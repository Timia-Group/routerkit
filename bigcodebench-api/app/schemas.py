from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class EvaluateRequest(BaseModel):
    # Official BigCodeBench task_id, e.g. "BigCodeBench/12".
    bcb_task_id: str
    # Candidate code to evaluate against that task's official test.
    solution: str

    # Optional resource limits (see bigcodebench's ADVANCED_USAGE.md for the
    # official CLI defaults); allow overriding evaluator.DEFAULT_LIMITS.
    max_as_limit_mb: float | None = Field(default=None, description="Address space limit (MB)")
    max_data_limit_mb: float | None = Field(default=None, description="Data segment limit (MB)")
    max_stack_limit_mb: float | None = Field(default=None, description="Stack limit (MB)")
    min_time_limit_s: float | None = Field(default=None, description="Minimum execution timeout (s)")
    gt_time_limit_s: float | None = Field(default=None, description="Ground-truth solution timeout (s)")


class JobCreatedResponse(BaseModel):
    job_id: str
    status: JobStatus


class EvaluateResult(BaseModel):
    passed: bool
    status: str
    fail_reason: str | None = None
    exec_ms: float
    details: dict[str, Any] | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: EvaluateResult | None = None
    error: str | None = None
