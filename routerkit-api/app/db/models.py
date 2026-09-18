"""ORM 1:1 con postgres/init/001_schema.sql (schema `routerkit`).

Cualquier cambio de columnas/tablas debe reflejarse en ambos lugares. No hay
tabla `experiments`: todo trabaja sobre lo activo (ver comentario al inicio
del schema SQL).
"""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import BigInteger, Boolean, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Task(Base):
    __tablename__ = "tasks"

    task_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    tests: Mapped[str | None] = mapped_column(Text)
    entry_point: Mapped[str | None] = mapped_column(String(255))
    source: Mapped[str | None] = mapped_column(String(255))
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(String))
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.current_timestamp()
    )
    # NULL = tarea custom (juez LLM). Con valor = tarea oficial del benchmark
    # (bigcodebench-api la evalúa contra el dataset oficial).
    bcb_task_id: Mapped[str | None] = mapped_column(String(255))
    # Borrado lógico: una recarga del benchmark desactiva las filas viejas y
    # crea filas nuevas en vez de actualizar (ver postgres/init/001_schema.sql).
    # Todo proceso que busque tareas debe filtrar active=True.
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class Question(Base):
    __tablename__ = "questions"

    question_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)  # binary | numeric
    lo: Mapped[Decimal | None] = mapped_column(Numeric)
    hi: Mapped[Decimal | None] = mapped_column(Numeric)
    author: Mapped[str | None] = mapped_column(String(255))
    version: Mapped[str | None] = mapped_column(String(100))
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # NULL = se resuelve con LLM (el model con role='feature' activo). Con
    # valor = texto de `def task_func(task_prompt): ...` (mismo nombre que
    # el entry_point de BigCodeBench); se ejecuta directo (sin LLM) con el
    # prompt de la tarea como input.
    script: Mapped[str | None] = mapped_column(Text)


class Model(Base):
    __tablename__ = "models"

    model_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[str] = mapped_column(String(50), nullable=False)  # api | hosted
    provider: Mapped[str | None] = mapped_column(String(255))
    endpoint: Mapped[str | None] = mapped_column(Text)
    deployment: Mapped[str | None] = mapped_column(String(255))
    # Nombre de var. de entorno a usar para este modelo (permite varias keys
    # del mismo provider). NULL = convención fija por provider (llm_client.py).
    api_key_env: Mapped[str | None] = mapped_column(String(255))
    in_token_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    out_token_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    cost_per_second: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    # Máximo de llamadas simultáneas a este modelo. NULL = sin límite definido.
    concurrency: Mapped[int | None] = mapped_column(Integer)
    # Timeout de la llamada HTTP a este modelo, en segundos.
    timeout_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=300)
    # Temperatura a usar al llamar este modelo. NULL = default del proveedor.
    temperature: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))
    # candidate (varios activos a la vez) | feature | judge (a lo sumo uno
    # activo de cada uno, ver idx_models_singleton_role_active).
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class Feature(Base):
    __tablename__ = "features"

    feature_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    task_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("routerkit.tasks.task_id"), nullable=False)
    question_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("routerkit.questions.question_id"), nullable=False
    )
    source: Mapped[str | None] = mapped_column(String(50))  # llm | python
    value: Mapped[Decimal | None] = mapped_column(Numeric)
    answer_text: Mapped[str | None] = mapped_column(Text)
    # Logprobs crudos de la respuesta (solo source='llm'), serializados como texto.
    answer_logprobs: Mapped[str | None] = mapped_column(Text)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[Decimal | None] = mapped_column(Numeric(18, 3))
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    # Borrado lógico: a lo sumo una fila activa por (task_id, question_id)
    # (ver idx_features_natural_key_active). Se desactiva en cascada cuando
    # se desactiva su task o su question (services/cascade.py).
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class Solution(Base):
    __tablename__ = "solutions"

    solution_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    task_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("routerkit.tasks.task_id"), nullable=False)
    model_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("routerkit.models.model_id"), nullable=False
    )
    sample_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    code: Mapped[str | None] = mapped_column(Text)
    tokens_in: Mapped[int | None] = mapped_column(Integer)
    tokens_out: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[Decimal | None] = mapped_column(Numeric(18, 3))
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    gen_error: Mapped[str | None] = mapped_column(Text)
    # Indicador explícito de error (generación, servicio, o carga externa),
    # independiente de si gen_error trae texto o no.
    has_error: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Borrado lógico: a lo sumo una fila activa por (task_id, model_id,
    # sample_idx) (ver idx_solutions_natural_key_active). Se desactiva en
    # cascada cuando se desactiva su task o su model.
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class Grade(Base):
    __tablename__ = "grades"

    solution_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("routerkit.solutions.solution_id", ondelete="CASCADE"),
        primary_key=True,
    )
    grader: Mapped[str] = mapped_column(String(50), nullable=False)  # tests | llm_judge
    passed: Mapped[bool | None] = mapped_column(Boolean)
    fail_reason: Mapped[str | None] = mapped_column(String(255))
    exec_ms: Mapped[Decimal | None] = mapped_column(Numeric(18, 3))
    # Falló la validación en sí (infra), distinto de passed=false.
    has_error: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Borrado lógico en cascada: se desactiva cuando se desactiva su solution.
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class Target(Base):
    __tablename__ = "targets"

    target_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    task_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("routerkit.tasks.task_id"), nullable=False)
    model_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("routerkit.models.model_id"), nullable=False
    )
    n: Mapped[int] = mapped_column(Integer, nullable=False)
    k: Mapped[int] = mapped_column(Integer, nullable=False)
    pass_rate: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    # Borrado lógico: a lo sumo una fila activa por (task_id, model_id) (ver
    # idx_targets_natural_key_active). Se recalcula por completo cada vez.
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    model_metric_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    model_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("routerkit.models.model_id"), nullable=False
    )
    taskpass_pct: Mapped[Decimal | None] = mapped_column(Numeric)
    complete_pct: Mapped[Decimal | None] = mapped_column(Numeric)
    pass_rate_avg: Mapped[Decimal | None] = mapped_column(Numeric)
    cost_per_task: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    cost_total: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    latency_avg_ms: Mapped[Decimal | None] = mapped_column(Numeric(18, 3))
    tokens_out_avg: Mapped[Decimal | None] = mapped_column(Numeric(18, 3))
    # Borrado lógico: a lo sumo una fila activa por model_id (ver
    # idx_model_metrics_natural_key_active). Se recalcula por completo.
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class Predictor(Base):
    __tablename__ = "predictors"

    predictor_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    algorithm: Mapped[str] = mapped_column(String(255), nullable=False)
    cv_score: Mapped[Decimal | None] = mapped_column(Numeric)
    test_r2: Mapped[Decimal | None] = mapped_column(Numeric)
    test_mae: Mapped[Decimal | None] = mapped_column(Numeric)
    feature_columns: Mapped[dict] = mapped_column(JSONB, nullable=False)
    artifact_path: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.current_timestamp()
    )
    is_production: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # No participa de la cascada automática (entrenar es deliberado).
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class RouteDecision(Base):
    __tablename__ = "route_decisions"

    decision_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    predictor_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("routerkit.predictors.predictor_id"), nullable=False
    )
    winner_model_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("routerkit.models.model_id"), nullable=False
    )
    ts: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.current_timestamp()
    )
    prompt_hash: Mapped[str | None] = mapped_column(String(255))
    features: Mapped[dict | None] = mapped_column(JSONB)
    ranking: Mapped[dict | None] = mapped_column(JSONB)
    used_fallback: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    outcome_success: Mapped[bool | None] = mapped_column(Boolean)
    outcome_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))


class Job(Base):
    __tablename__ = "jobs"

    job_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    # features_compute | solutions_generate | evaluation (a futuro)
    kind: Mapped[str] = mapped_column(String(50), nullable=False)
    # empty | pending | running | done | error
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="empty")
    message: Mapped[str | None] = mapped_column(Text)
    progress_current: Mapped[int | None] = mapped_column(Integer)
    progress_total: Mapped[int | None] = mapped_column(Integer)
    result: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.current_timestamp()
    )
    finished_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    # Borrado lógico: a lo sumo una fila activa por kind (ver
    # idx_jobs_unique_active_kind). Arrancar un job nuevo desactiva la
    # anterior (si ya terminó) y crea una nueva; al iniciar el container se
    # resetean todas a status='empty' (ver app.services.jobs).
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
