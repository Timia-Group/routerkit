"""ORM 1:1 with postgres/init/001_schema.sql (schema `routerkit`).

Any column/table change must be reflected in both places. There's no
`experiments` table: everything works on what's active (see the comment at
the top of the SQL schema).
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
    # NULL = custom task (LLM judge). With a value = official benchmark task
    # (bigcodebench-api evaluates it against the official dataset).
    bcb_task_id: Mapped[str | None] = mapped_column(String(255))
    # Stable key for custom tasks (analogous to bcb_task_id) — to reference
    # them from features/solutions/validations without depending on the
    # internal task_id (see app/services/tasks_catalog.py).
    custom_task_id: Mapped[str | None] = mapped_column(String(255))
    # Soft delete: reloading the benchmark deactivates the old rows and
    # creates new ones instead of updating (see postgres/init/001_schema.sql).
    # Any process that looks up tasks must filter on active=True.
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
    # Stable key (e.g. "p41") to reference this question from
    # features/solutions/validations without depending on the internal question_id.
    question_key: Mapped[str | None] = mapped_column(String(255))
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # NULL = resolved by LLM (the active model with role='feature'). With a
    # value = text of `def task_func(task_prompt): ...` (same name as
    # BigCodeBench's entry_point); it's executed directly (no LLM) with the
    # task's prompt as input.
    script: Mapped[str | None] = mapped_column(Text)
    # Times to repeat the LLM call and average (simple mean) the result.
    # Ignored for a script question (always computed once, deterministic).
    repetitions: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class Model(Base):
    __tablename__ = "models"

    model_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[str] = mapped_column(String(50), nullable=False)  # api | hosted
    provider: Mapped[str | None] = mapped_column(String(255))
    endpoint: Mapped[str | None] = mapped_column(Text)
    deployment: Mapped[str | None] = mapped_column(String(255))
    # Name of the env var to use for this model (allows several keys for
    # the same provider). NULL = fixed per-provider convention (llm_client.py).
    api_key_env: Mapped[str | None] = mapped_column(String(255))
    in_token_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    out_token_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    cost_per_second: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    # Maximum simultaneous calls to this model. NULL = no limit defined.
    concurrency: Mapped[int | None] = mapped_column(Integer)
    # Timeout of the HTTP call to this model, in seconds.
    timeout_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=300)
    # Temperature to use when calling this model. NULL = provider default.
    temperature: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))
    # chat_completions (default, litellm.completion()) | responses
    # (litellm.responses(), OpenAI's new API).
    api_kind: Mapped[str] = mapped_column(String(50), nullable=False, default="chat_completions")
    # Maximum output tokens to request from the model. NULL = not sent.
    max_tokens: Mapped[int | None] = mapped_column(Integer)
    # candidate (several active at once) | feature | judge (at most one
    # active of each, see idx_models_singleton_role_active).
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
    # Raw logprobs of the response (source='llm' only), serialized as text.
    answer_logprobs: Mapped[str | None] = mapped_column(Text)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[Decimal | None] = mapped_column(Numeric(18, 3))
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    gen_error: Mapped[str | None] = mapped_column(Text)
    has_error: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Soft delete: at most one active row per (task_id, question_id)
    # (see idx_features_natural_key_active). Deactivated in cascade when
    # its task or its question is deactivated (services/cascade.py).
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
    # Explicit error flag (generation, service, or external load),
    # independent of whether gen_error carries text or not.
    has_error: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Soft delete: at most one active row per (task_id, model_id,
    # sample_idx) (see idx_solutions_natural_key_active). Deactivated in
    # cascade when its task or its model is deactivated.
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class Grade(Base):
    __tablename__ = "grades"

    grade_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    solution_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("routerkit.solutions.solution_id", ondelete="CASCADE"),
        nullable=False,
    )
    grader: Mapped[str] = mapped_column(String(50), nullable=False)  # tests | llm_judge
    passed: Mapped[bool | None] = mapped_column(Boolean)
    fail_reason: Mapped[str | None] = mapped_column(Text)
    exec_ms: Mapped[Decimal | None] = mapped_column(Numeric(18, 3))
    # The validation itself failed (infra), different from passed=false.
    has_error: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Cascading soft delete: deactivated when its solution is deactivated.
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
    # Soft delete: at most one active row per (task_id, model_id) (see
    # idx_targets_natural_key_active). Fully recomputed each time.
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
    # Soft delete: at most one active row per model_id (see
    # idx_model_metrics_natural_key_active). Fully recomputed.
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
    # Routing policy calibrated for this predictor's own Y_pred, keyed by
    # shot count: {lambda, test_accuracy, test_cost_avg} per key (see
    # services/routing_calibration.py).
    routing_lambdas: Mapped[dict | None] = mapped_column(JSONB)
    # Full (lambda, accuracy, cost) sweep curve, keyed by shot count — kept
    # for later audit/plotting without re-running the Monte Carlo sweep.
    lambda_sweep: Mapped[dict | None] = mapped_column(JSONB)
    # Full replace: POST /train deactivates every existing predictor before
    # inserting the new one — there's only ever one active predictor, and
    # that's always the one POST /route uses (see services/training.py).
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
    # features_compute | solutions_generate | evaluation (future)
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
    # Soft delete: at most one active row per kind (see
    # idx_jobs_unique_active_kind). Starting a new job deactivates the
    # previous one (if it already finished) and creates a new one; on
    # container startup all of them are reset to status='empty' (see
    # app.services.jobs).
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
