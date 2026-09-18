"""Modelos Pydantic usados para documentar (Swagger) y validar el contenido
de los archivos de import (tasks.jsonl, questions.json, models.json). Todo
input a routerkit-api es JSON (estandarizado).

Los endpoints de import reciben un archivo (multipart), así que FastAPI no
puede mostrar el shape de cada fila en el body del request automáticamente;
estos modelos se usan para validar cada fila ya parseada y quedan
documentados en el docstring de cada endpoint con un ejemplo.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class TaskImportRow(BaseModel):
    """Una fila de tasks.jsonl. `bcb_task_id` decide el tipo por fila (no por
    archivo): si viene con valor (ej. "BigCodeBench/12") es una tarea del
    benchmark oficial, si se omite es una tarea custom — un mismo archivo
    puede traer ambas mezcladas."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "Escribe una función que sume dos enteros.",
                "tests": "assert suma(2, 3) == 5",
                "entry_point": "suma",
                "source": "custom",
                "tags": ["math", "easy"],
                "bcb_task_id": None,
            }
        }
    )

    prompt: str
    tests: str | None = None
    entry_point: str | None = None
    source: str | None = None
    tags: list[str] | None = None
    bcb_task_id: str | None = None


class QuestionImportRow(BaseModel):
    """Una entrada de questions.json."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "¿La solución usa recursión?",
                "kind": "binary",
                "lo": 0,
                "hi": 1,
                "author": "marcos",
                "version": "v1",
                "script": None,
            }
        }
    )

    text: str
    # binary: pregunta de sí/no (0-1). numeric: valor numérico (entero o decimal).
    kind: Literal["binary", "numeric"]
    lo: float | None = None
    hi: float | None = None
    author: str | None = None
    version: str | None = None
    # None: la pregunta se resuelve con el LLM. Con valor: función Python
    # completa, firma fija `def task_func(task_prompt: str) -> float` (mismo
    # nombre que usa BigCodeBench como entry_point); se ejecuta con el
    # prompt de la tarea como único input (sin LLM, costo 0).
    script: str | None = None


class ModelImportRow(BaseModel):
    """Una entrada de models.json."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "gpt-4o-mini",
                "type": "api",
                "provider": "openai",
                "endpoint": "https://api.openai.com/v1",
                "api_key_env": None,
                "in_token_price": 0.15,
                "out_token_price": 0.6,
                "concurrency": None,
                "timeout_seconds": 300,
                "temperature": None,
                "role": "candidate",
            }
        }
    )

    name: str
    # api: se llama por HTTP a un proveedor externo. hosted: se factura por tiempo de cómputo propio.
    type: Literal["api", "hosted"]
    provider: str | None = None
    endpoint: str | None = None
    deployment: str | None = None
    # Nombre de la variable de entorno a usar para ESTE modelo (permite tener
    # varias keys del mismo provider, ej. dos cuentas de OpenAI). None = usar
    # la convención fija por provider (ver .env.example).
    api_key_env: str | None = None
    in_token_price: float | None = None
    out_token_price: float | None = None
    cost_per_second: float | None = None
    # Máximo de llamadas simultáneas a este modelo. None = sin límite definido.
    concurrency: int | None = None
    # Timeout de la llamada HTTP a este modelo, en segundos.
    timeout_seconds: int = 300
    # Temperatura a usar al llamar este modelo (config por modelo, no por
    # request). None = default del proveedor.
    temperature: float | None = None
    # candidate: se rutea/genera soluciones con él (puede haber varios
    # activos a la vez). feature: calcula features vía LLM. judge: califica
    # soluciones de tareas custom. Solo puede haber UN model activo con
    # role='feature' y UN model activo con role='judge' a la vez.
    role: Literal["candidate", "feature", "judge"]
