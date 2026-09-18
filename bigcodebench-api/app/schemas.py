from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class EvaluateRequest(BaseModel):
    # task_id oficial de BigCodeBench, ej. "BigCodeBench/12".
    bcb_task_id: str
    # código candidato a evaluar contra el test oficial de esa tarea.
    solution: str

    # límites de recursos opcionales (ver ADVANCED_USAGE.md de bigcodebench
    # para los defaults del CLI oficial); permiten overridear evaluator.DEFAULT_LIMITS.
    max_as_limit_mb: float | None = Field(default=None, description="Límite de address space (MB)")
    max_data_limit_mb: float | None = Field(default=None, description="Límite de data segment (MB)")
    max_stack_limit_mb: float | None = Field(default=None, description="Límite de stack (MB)")
    min_time_limit_s: float | None = Field(default=None, description="Timeout mínimo de ejecución (s)")
    gt_time_limit_s: float | None = Field(default=None, description="Timeout de la solución ground-truth (s)")


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
