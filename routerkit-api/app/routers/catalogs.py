"""Import de catálogos (tasks.jsonl, questions.json, models.json) a Postgres.

Todo input es JSON (estandarizado). Los 3 endpoints reciben un archivo
(multipart) y devuelven cuántas filas se crearon/desactivaron. El shape y
los valores aceptados de cada fila están documentados en `app.schemas`
(TaskImportRow, QuestionImportRow, ModelImportRow) y se validan antes de
insertar: una fila inválida aborta el import completo con 422 y el detalle
del error. Los 3 son un reemplazo con borrado lógico (cascada incluida vía
app.services.cascade): desactivan lo existente y activan solo lo de la
última carga, nunca tocan una fila ya usada por un experimento anterior.
Además, cada uno deja un snapshot del estado activo resultante en
data/current (rotando el anterior a data/previous, ver app.services.snapshots).
"""

import json

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import ValidationError
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db.models import Model, Question
from app.schemas import ModelImportRow, QuestionImportRow
from app.services import cascade
from app.services.snapshots import row_to_dict, write_snapshot
from app.services.tasks_catalog import parse_tasks_jsonl, replace_tasks

router = APIRouter(tags=["catalogs"])


def _validation_error(item: str, exc: ValidationError) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={"item": item, "errors": exc.errors(include_url=False, include_context=False)},
    )


def _parse_json_object(raw: str) -> dict | list:
    """Parsea el body de un archivo .json de import. 400 si no es JSON válido."""
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"JSON inválido: {exc}") from exc


@router.post(
    "/tasks/import",
    summary="Reemplaza TODAS las tareas activas desde un tasks.jsonl provisto por el usuario",
)
async def import_tasks(
    file: UploadFile = File(description="tasks.jsonl: una tarea JSON por línea."),
    db: Session = Depends(get_db),
) -> dict:
    """Reemplaza (con borrado lógico) TODAS las tasks activas en
    `routerkit.tasks` por las de este archivo.

    Al arrancar, `routerkit-api` ya carga automáticamente el catálogo
    oficial de BigCodeBench (1140 tasks) si la tabla está vacía (ver
    `app.services.seed`) — este endpoint es para cuando el usuario quiere
    reemplazar ese catálogo por otro propio. Cada línea decide su tipo por
    sí misma vía `bcb_task_id`: con valor (ej. "BigCodeBench/12") es una
    tarea del benchmark oficial, sin valor es una tarea custom — un mismo
    archivo puede traer ambas mezcladas libremente. Ninguna API descarga el
    dataset de BigCodeBench por su cuenta en este endpoint: si se quieren
    tasks del benchmark acá, el jsonl las trae con su `bcb_task_id`.

    Desactiva (`active=false`) TODAS las tasks existentes (cualquiera sea
    su tipo) y crea filas nuevas con `active=true`; nunca toca una fila ya
    usada por un experimento anterior.

    Cada línea debe cumplir `TaskImportRow` (`prompt` obligatorio; `tests`,
    `entry_point`, `source`, `tags`, `bcb_task_id` opcionales). Ejemplo:

    ```json
    {"prompt": "Escribe una función que sume dos enteros.", "tests": "assert suma(2, 3) == 5", "entry_point": "suma", "source": "custom", "tags": ["math", "easy"]}
    ```
    """
    raw = (await file.read()).decode("utf-8")
    rows = parse_tasks_jsonl(raw)
    return replace_tasks(db, rows)


@router.post(
    "/questions/import",
    summary="Reemplaza el catálogo de preguntas de features desde questions.json",
)
async def import_questions(
    file: UploadFile = File(description="questions.json: lista de preguntas."),
    db: Session = Depends(get_db),
) -> dict:
    """Reemplaza (con borrado lógico) el catálogo de preguntas: desactiva
    (`active=false`) TODAS las preguntas existentes y crea filas nuevas para
    el archivo recién subido, con `active=true`. Mismo patrón que la recarga
    de tareas: solo quedan activas las preguntas de la última carga.

    Cada entrada debe cumplir `QuestionImportRow`: `text` y `kind` obligatorios
    — `kind` acepta solo `binary` (sí/no, 0-1) o `numeric` (entero o decimal).
    `lo`/`hi` acotan el rango esperado de la respuesta.

    `script` es opcional: si se omite, la pregunta se resuelve con el LLM
    (el único model con role='feature' activo); si se define, debe ser el
    texto de una función Python completa con firma fija
    `def task_func(task_prompt: str) -> float` (mismo nombre de función que
    usa BigCodeBench como entry_point) — se ejecuta directo con el prompt de
    la tarea como único input, sin llamar a ningún LLM (costo cero).

    ```json
    {
      "questions": [
        {"text": "¿La solución usa recursión?", "kind": "binary", "lo": 0, "hi": 1, "author": "marcos", "version": "v1"},
        {
          "text": "¿Cuántas líneas tiene el prompt de la tarea?",
          "kind": "numeric",
          "lo": 1,
          "hi": 200,
          "version": "v1",
          "script": "def task_func(task_prompt: str) -> float:\\n    return float(len(task_prompt.splitlines()))"
        }
      ]
    }
    ```
    """
    raw = (await file.read()).decode("utf-8")
    data = _parse_json_object(raw)
    raw_rows = data.get("questions", []) if isinstance(data, dict) else data

    rows: list[QuestionImportRow] = []
    for index, raw_row in enumerate(raw_rows):
        try:
            rows.append(QuestionImportRow.model_validate(raw_row))
        except ValidationError as exc:
            raise _validation_error(f"questions[{index}]", exc) from exc

    deactivated_ids = list(
        db.execute(select(Question.question_id).where(Question.active.is_(True))).scalars()
    )
    if deactivated_ids:
        db.execute(update(Question).where(Question.question_id.in_(deactivated_ids)).values(active=False))
        cascade.deactivate_for_questions(db, deactivated_ids)

    for row in rows:
        db.add(
            Question(
                text=row.text,
                kind=row.kind,
                lo=row.lo,
                hi=row.hi,
                author=row.author,
                version=row.version,
                active=True,
                script=row.script,
            )
        )

    db.commit()

    active_questions = db.query(Question).filter(Question.active.is_(True)).all()
    write_snapshot("questions", [row_to_dict(q) for q in active_questions])

    return {"deactivated_previous": len(deactivated_ids), "created": len(rows)}


@router.post(
    "/models/import",
    summary="Reemplaza el catálogo de modelos candidatos/jueces desde models.json",
)
async def import_models(
    file: UploadFile = File(description="models.json: lista de modelos."),
    db: Session = Depends(get_db),
) -> dict:
    """Reemplaza (con borrado lógico) el catálogo de modelos: desactiva
    (`active=false`) TODOS los modelos existentes y crea filas nuevas para
    el archivo recién subido, con `active=true`. Mismo patrón que
    tasks/questions — incluye cascada: al desactivar un modelo también se
    desactivan sus `solutions` (y los `grades` de esas solutions), `targets`
    y `model_metrics`.

    Cada entrada debe cumplir `ModelImportRow`: `type` acepta solo `api`
    (se llama por HTTP a un proveedor externo) u `hosted` (se factura por
    tiempo de cómputo propio); `role` acepta `candidate` (se rutea/genera
    soluciones — pueden convivir varios activos, pero nunca dos con el
    mismo `name`), `feature` (calcula features vía LLM) o `judge` (califica
    soluciones de tareas custom) — de estos dos últimos solo puede haber
    uno activo en total. `api_key_env` es opcional: si se omite,
    `app.clients.llm_client` resuelve la key por una convención fija según
    `provider` (ver `.env.example`) — `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`,
    `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, o `CUSTOM_LLM_API_KEY`. Si se
    define, apunta a otra variable de entorno propia (útil para que
    distintos modelos usen distintas cuentas del mismo provider).
    `concurrency` (opcional, sin límite si se omite), `timeout_seconds`
    (default 300) y `temperature` (opcional, default del proveedor si se
    omite) controlan cómo se llama al modelo — `temperature` es config por
    modelo, no por request. `concurrency` es importante para el model con
    role='feature', ya que sus respuestas necesitan logprobs (no todos los
    providers los exponen, ej. Anthropic no).

    ```json
    {
      "models": [
        {
          "name": "gpt-4o-mini",
          "type": "api",
          "provider": "openai",
          "endpoint": "https://api.openai.com/v1",
          "in_token_price": 0.15,
          "out_token_price": 0.60,
          "role": "candidate"
        },
        {
          "name": "gpt-4o",
          "type": "api",
          "provider": "openai",
          "api_key_env": "OPENAI_API_KEY_2",
          "in_token_price": 2.50,
          "out_token_price": 10.00,
          "role": "candidate"
        },
        {
          "name": "gpt-4o-mini",
          "type": "api",
          "provider": "openai",
          "in_token_price": 0.15,
          "out_token_price": 0.60,
          "concurrency": 5,
          "timeout_seconds": 120,
          "role": "feature"
        },
        {"name": "claude-sonnet-5", "type": "api", "provider": "anthropic", "role": "judge"}
      ]
    }
    ```
    """
    raw = (await file.read()).decode("utf-8")
    data = _parse_json_object(raw)
    raw_rows = data.get("models", []) if isinstance(data, dict) else data

    rows: list[ModelImportRow] = []
    for index, raw_row in enumerate(raw_rows):
        try:
            rows.append(ModelImportRow.model_validate(raw_row))
        except ValidationError as exc:
            raise _validation_error(f"models[{index}]", exc) from exc

    deactivated_ids = list(db.execute(select(Model.model_id).where(Model.active.is_(True))).scalars())
    if deactivated_ids:
        db.execute(update(Model).where(Model.model_id.in_(deactivated_ids)).values(active=False))
        cascade.deactivate_for_models(db, deactivated_ids)

    for row in rows:
        db.add(
            Model(
                name=row.name,
                type=row.type,
                provider=row.provider,
                endpoint=row.endpoint,
                deployment=row.deployment,
                api_key_env=row.api_key_env,
                in_token_price=row.in_token_price,
                out_token_price=row.out_token_price,
                cost_per_second=row.cost_per_second,
                concurrency=row.concurrency,
                timeout_seconds=row.timeout_seconds,
                temperature=row.temperature,
                role=row.role,
                active=True,
            )
        )

    db.commit()

    active_models = db.query(Model).filter(Model.active.is_(True)).all()
    write_snapshot("models", [row_to_dict(m) for m in active_models])

    return {"deactivated_previous": len(deactivated_ids), "created": len(rows)}
