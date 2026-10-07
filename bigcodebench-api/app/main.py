from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app import evaluator, jobs
from app.schemas import EvaluateRequest, JobCreatedResponse, JobStatusResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    evaluator.load_dataset()
    yield


app = FastAPI(
    title="bigcodebench-api",
    description=(
        "HTTP wrapper over BigCodeBench. `POST /jobs` + `GET /jobs/{id}` "
        "evaluate a candidate solution against the official test of a "
        "`bcb_task_id`, asynchronously (job/polling). It does not expose "
        "the official dataset: routerkit-api never requests it, the user "
        "always uploads their own task JSON."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/jobs", response_model=JobCreatedResponse, status_code=202)
def create_job(request: EvaluateRequest) -> JobCreatedResponse:
    job = jobs.create_job(request)
    return JobCreatedResponse(job_id=job.job_id, status=job.status)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        result=job.result,
        error=job.error,
    )
