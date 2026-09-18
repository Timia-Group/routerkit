from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db.base import SessionLocal
from app.routers import (
    aggregation,
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
            "Import de catálogos: tareas (arrancan pre-cargadas con el benchmark "
            "oficial de BigCodeBench, reemplazables por las propias del usuario, "
            "custom/benchmark/mezcladas), preguntas de features y modelos."
        ),
    },
    {
        "name": "solutions",
        "description": "Generación (async) de soluciones vía LLM candidato, o carga directa (type=load).",
    },
    {
        "name": "evaluation",
        "description": (
            "Valida (async) las solutions activas: bigcodebench-api si la tarea es de "
            "benchmark, juez LLM si es custom."
        ),
    },
    {"name": "aggregation", "description": "Recalcula targets y model_metrics a partir de solutions/grades activos."},
    {"name": "features", "description": "Cálculo (async) de features LLM/programáticas sobre tasks/questions activas."},
    {"name": "training", "description": "Entrenamiento del predictor (AutoML, stub) con features/targets activos."},
    {"name": "routing", "description": "Ruteo en producción con el predictor productivo (stub)."},
    {
        "name": "jobs",
        "description": (
            "Estado de los procesos async de larga duración (features/compute, "
            "solutions/generate, solutions/validate)."
        ),
    },
]

app = FastAPI(
    title="routerkit-api",
    description=(
        "Orquestador de RouterKit: catálogos, features, generación de soluciones, "
        "evaluación (bigcodebench-api o juez LLM), agregación y entrenamiento del "
        "router de LLMs. Sin conceptos de 'experimento': todo opera sobre lo activo."
    ),
    version="0.1.0",
    openapi_tags=OPENAPI_TAGS,
    lifespan=lifespan,
)

app.include_router(catalogs.router)
app.include_router(solutions.router)
app.include_router(evaluation.router)
app.include_router(aggregation.router)
app.include_router(features.router)
app.include_router(training.router)
app.include_router(routing.router)
app.include_router(jobs.router)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}
