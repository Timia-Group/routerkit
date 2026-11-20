from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from app.db.base import SessionLocal
from app.routers import (
    catalogs,
    evaluation,
    features,
    jobs,
    routing,
    solutions,
    training,
)
from app.services.jobs import reset_all_on_startup
from app.services.seed import seed_default_tasks_if_empty


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = SessionLocal()
    try:
        reset_all_on_startup(db)
        seed_default_tasks_if_empty(db)
    finally:
        db.close()
    yield

OPENAPI_TAGS = [
    {
        "name": "catalogs",
        "description": (
            "Catalog import: tasks (start pre-loaded with the official "
            "BigCodeBench benchmark, replaceable with the user's own, "
            "custom/benchmark/mixed), feature questions, and models."
        ),
    },
    {
        "name": "solutions",
        "description": "(Async) generation of solutions via a candidate LLM, or direct load (type=load).",
    },
    {
        "name": "evaluation",
        "description": (
            "(Async) validates the active solutions: bigcodebench-api if the task is a "
            "benchmark one, LLM judge if it's custom."
        ),
    },
    {"name": "features", "description": "(Async) computation of LLM/programmatic features over active tasks/questions."},
    {
        "name": "training",
        "description": "Training of the predictor (see POST /train for the available algorithms) with active features/targets.",
    },
    {"name": "routing", "description": "Routes a task to the best candidate LLM using the active predictor."},
    {
        "name": "jobs",
        "description": (
            "Status of long-running async processes (features/compute, "
            "solutions/generate, solutions/validate)."
        ),
    },
]

app = FastAPI(
    title="routerkit-api",
    description=(
        "RouterKit orchestrator: catalogs, features, solution generation, "
        "evaluation (bigcodebench-api or LLM judge), training, and production routing of the "
        "LLM router. No 'experiment' concept: everything operates on what's active."
    ),
    version="0.1.0",
    openapi_tags=OPENAPI_TAGS,
    lifespan=lifespan,
)

app.include_router(catalogs.router)
app.include_router(solutions.router)
app.include_router(evaluation.router)
app.include_router(features.router)
app.include_router(training.router)
app.include_router(routing.router)
app.include_router(jobs.router)


class HealthResponse(BaseModel):
    status: Literal["ok"]


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> dict:
    return {"status": "ok"}
