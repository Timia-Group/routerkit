from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import update
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db.models import Feature, Model, Question, Task
from app.services import features_llm, jobs
from app.services.concurrency import run_bounded
from app.services.features_python import ScriptError, compute_python_feature
from app.services.jobs import JobConflictError, ProgressFn
from app.services.snapshots import row_to_dict, write_snapshot

router = APIRouter(tags=["features"])

_KIND = "features_compute"


def _compute_features_job(db: Session, progress: ProgressFn) -> dict:
    """Cuerpo real de /features/compute — corre en background (ver app.services.jobs)."""
    tasks = db.query(Task).filter(Task.active.is_(True)).all()
    questions = db.query(Question).filter(Question.active.is_(True)).all()

    feature_model = db.query(Model).filter(Model.role == "feature", Model.active.is_(True)).one_or_none()

    db.execute(update(Feature).where(Feature.active.is_(True)).values(active=False))

    pairs = [(task, question) for task in tasks for question in questions]
    python_pairs = [(task, question) for task, question in pairs if question.script]
    llm_pairs = [(task, question) for task, question in pairs if not question.script]

    total = len(pairs)
    processed = 0
    computed = 0
    errors: list[dict] = []

    # Python: secuencial, sin thread pool.
    for task, question in python_pairs:
        try:
            result = compute_python_feature(task, question)
        except ScriptError as exc:
            errors.append(
                {"task_id": task.task_id, "question_id": question.question_id, "error": str(exc)}
            )
        else:
            db.add(
                Feature(
                    task_id=task.task_id,
                    question_id=question.question_id,
                    source="python",
                    value=result.value,
                    latency_ms=result.latency_ms,
                    cost_usd=0,
                )
            )
            computed += 1
        processed += 1
        progress(processed, total, f"{processed}/{total} procesadas")

    # LLM: concurrente, hasta feature_model.concurrency llamadas en vuelo.
    if feature_model is None:
        for task, question in llm_pairs:
            errors.append(
                {
                    "task_id": task.task_id,
                    "question_id": question.question_id,
                    "error": "question sin script y no hay ningún model con role='feature' activo",
                }
            )
            processed += 1
            progress(processed, total, f"{processed}/{total} procesadas")
    else:
        llm_results = run_bounded(
            llm_pairs,
            lambda pair: features_llm.compute_llm_feature(pair[0], pair[1], feature_model),
            feature_model.concurrency,
        )
        for (task, question), result, exc in llm_results:
            if exc is not None:
                errors.append(
                    {"task_id": task.task_id, "question_id": question.question_id, "error": str(exc)}
                )
            else:
                db.add(
                    Feature(
                        task_id=task.task_id,
                        question_id=question.question_id,
                        source="llm",
                        value=result.value,
                        answer_text=result.answer_text,
                        answer_logprobs=result.answer_logprobs,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        latency_ms=result.latency_ms,
                        cost_usd=result.cost_usd,
                    )
                )
                computed += 1
            processed += 1
            progress(processed, total, f"{processed}/{total} procesadas")

    active_features = db.query(Feature).filter(Feature.active.is_(True)).all()
    write_snapshot("features", [row_to_dict(f) for f in active_features])

    return {"computed": computed, "errors": errors}


@router.post("/features/compute", status_code=202)
def compute_features(db: Session = Depends(get_db)) -> dict:
    """Lanza (async) el cálculo de features para TODAS las tasks activas x
    TODAS las questions activas. Devuelve enseguida el estado inicial
    (`status='running'`, viene de status='pending' recién creado) — para
    ver el progreso o el resultado final, consultar `GET /jobs` (estado de
    TODOS los kinds en un solo request, incluido este). 409 si ya hay un
    job de `kind='features_compute'` corriendo.

    Reemplazo con borrado lógico: al arrancar, desactiva TODAS las features
    activas existentes; cada par (task, question) calculado con éxito se
    inserta como fila nueva activa (solo puede haber una fila activa por
    (task_id, question_id), ver idx_features_natural_key_active).

    Los pares se agrupan por método: las preguntas con `question.script` se
    ejecutan en Python, secuencial (no LLM, no hay beneficio en paralelizar
    exec() local); las que no tienen script se resuelven con el único model
    con role='feature' activo, con hasta `feature_model.concurrency`
    llamadas en vuelo a la vez (ver services.concurrency.run_bounded). Un
    error (de script, de LLM, o de la llamada al proveedor) se reporta en
    `errors` sin abortar el resto del lote. Al terminar deja un snapshot de
    las features activas resultantes en data/current/features.json
    (rotando el anterior a data/previous).
    """
    try:
        job = jobs.start_job(_KIND, _compute_features_job, db)
    except JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return jobs.to_board_entry(job, _KIND)
