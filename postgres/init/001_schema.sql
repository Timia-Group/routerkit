BEGIN;

-- ============================================================
-- ROUTERKIT
-- PostgreSQL schema
--
-- There is no "experiments" table: the system does not run parallel
-- configurations. Each load (tasks/questions/models) replaces the previous
-- data with a logical delete (active=false) and activates only the new
-- data, cascading "forward" (see app/services/cascade.py) to everything
-- that would have been computed from those rows. Every process always
-- works on "what's active" — there is never a need to pass an id to know
-- what to use, except to specifically reference an already-generated
-- record (e.g. to save it as an FK for the next step).
-- ============================================================

CREATE SCHEMA IF NOT EXISTS routerkit;


-- ============================================================
-- 1. TASKS
-- Catalog of programming tasks
-- ============================================================

CREATE TABLE routerkit.tasks (
    task_id         BIGSERIAL PRIMARY KEY,
    prompt          TEXT NOT NULL,
    tests           TEXT,
    entry_point     VARCHAR(255),
    source          VARCHAR(255),
    tags            TEXT[],
    created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- NULL = custom task, evaluated with an LLM judge (services/judge.py).
    -- With a value (e.g. "BigCodeBench/12") = official benchmark task,
    -- evaluated entirely inside bigcodebench-api against the official dataset.
    bcb_task_id     VARCHAR(255),

    -- Stable key the user assigns to a custom task (analogous to
    -- bcb_task_id, but for tasks that don't come from the benchmark) — so
    -- that features/solutions/validations can reference it without
    -- depending on Postgres's internal task_id (which changes depending on
    -- insertion order, see app/services/tasks_catalog.py). NULL if the task
    -- is from the benchmark (bcb_task_id is used there) or if this custom
    -- task doesn't need anything to reference it later.
    custom_task_id  VARCHAR(255),

    -- Logical delete. On startup, routerkit-api preloads the official
    -- BigCodeBench catalog here if the table is empty (see
    -- app/services/seed.py). Each `POST /tasks/import` (the user always
    -- uploads the jsonl, no API downloads it on its own) deactivates ALL
    -- existing tasks (regardless of bcb_task_id) and creates new rows
    -- (never updates in place) — the file can carry benchmark, custom, or
    -- both mixed, each row decides by its own bcb_task_id. This way each
    -- row remains a historical snapshot of what the task looked like when
    -- something used it, and the FKs of features/solutions/targets that
    -- already point to a task_id from a previous load are not broken.
    -- Every process that looks for tasks to work on must filter on
    -- active = true.
    active          BOOLEAN NOT NULL DEFAULT TRUE
);


-- ============================================================
-- 2. QUESTIONS
-- Catalog of questions used to obtain features
-- ============================================================

CREATE TABLE routerkit.questions (
    question_id     BIGSERIAL PRIMARY KEY,

    text            TEXT NOT NULL,

    -- binary | numeric
    kind            VARCHAR(20) NOT NULL,

    -- expected response range
    lo              NUMERIC,
    hi              NUMERIC,

    author          VARCHAR(255),
    version         VARCHAR(100),

    -- Stable key the user assigns to the question (e.g. "p41"), so that
    -- features/solutions/validations (type=load) reference it without
    -- depending on Postgres's internal question_id (which changes on
    -- every reload of questions.json, see app/services/tasks_catalog.py
    -- for the same pattern applied to tasks).
    question_key    VARCHAR(255),

    -- Logical delete: each POST /questions/import replaces the ENTIRE
    -- catalog (deactivates all existing ones, activates only the new ones).
    active          BOOLEAN NOT NULL DEFAULT TRUE,

    -- NULL = the question is resolved with the LLM (the active model with
    -- role='feature'). With a value = text of a complete Python function
    -- with fixed signature `def task_func(task_prompt: str) -> float`
    -- (same function name BigCodeBench uses as entry_point); it runs with
    -- the task prompt as its only input and does not go through any LLM
    -- (zero cost).
    script          TEXT,

    -- How many times to repeat this question's LLM call and average
    -- (simple mean) the resulting value — guards against the LLM's answer
    -- varying between calls, at the cost of repetitions x the
    -- tokens/price. Ignored for a script question (deterministic, always
    -- computed once regardless of this value — see features_python.py).
    repetitions     INTEGER NOT NULL DEFAULT 1
);


-- ============================================================
-- 3. MODELS
-- Catalog of models: routing candidates, or judges (features/grading)
-- ============================================================

CREATE TABLE routerkit.models (
    model_id            BIGSERIAL PRIMARY KEY,

    name                VARCHAR(255) NOT NULL,

    -- api | hosted
    type                VARCHAR(50) NOT NULL,

    provider            VARCHAR(255),
    endpoint            TEXT,
    deployment          VARCHAR(255),

    -- name of the environment variable to use for this model (allows
    -- having several keys for the same provider and choosing which one
    -- each model uses). NULL = resolved by the fixed provider convention
    -- (see app/clients/llm_client.py and .env.example).
    api_key_env         VARCHAR(255),

    -- prices USD / 1M tokens
    in_token_price      NUMERIC(18, 8),
    out_token_price     NUMERIC(18, 8),

    -- for hosted models
    cost_per_second     NUMERIC(18, 8),

    -- maximum concurrent calls to this model (provider/account-specific
    -- rate limit). NULL = no explicit limit defined yet.
    concurrency         INTEGER,

    -- timeout for the HTTP call to this model, in seconds.
    timeout_seconds     INTEGER NOT NULL DEFAULT 300,

    -- temperature to use when calling this model. NULL = provider default.
    -- Configured per model, not per request (see app/clients/llm_client.py).
    temperature         NUMERIC(8, 4),

    -- chat_completions (default, litellm.completion()) | responses (OpenAI's
    -- new API, litellm.responses() — needed by some models that don't
    -- expose (or don't fully support) Chat Completions).
    api_kind            VARCHAR(50) NOT NULL DEFAULT 'chat_completions',

    -- Maximum output tokens to request from the model. Some APIs require
    -- it (e.g. Anthropic Messages), others take it as an optional limit
    -- (Responses API: max_output_tokens). NULL = not sent.
    max_tokens          INTEGER,

    -- candidate: used to route/generate solutions (several can be active
    -- at once, but each with a distinct name). feature: computes features
    -- via LLM. judge: grades solutions for custom tasks. There can only be
    -- ONE active model with role='feature' and ONE active model with
    -- role='judge' in total; for candidate, the limit is "at most one
    -- active per name" (see the partial unique indexes below).
    role                VARCHAR(50) NOT NULL,

    -- Logical delete: each POST /models/import replaces the ENTIRE catalog
    -- (deactivates all existing ones, activates only the new ones). It
    -- cascades to also deactivate solutions/grades/targets/model_metrics
    -- of the deactivated models (see app/services/cascade.py).
    active              BOOLEAN NOT NULL DEFAULT TRUE
);


-- ============================================================
-- 4. FEATURES
--
-- One row per (task, question): the question is run against the task
-- (LLM or script) and produces a feature value.
-- ============================================================

CREATE TABLE routerkit.features (
    feature_id          BIGSERIAL PRIMARY KEY,

    task_id             BIGINT NOT NULL,
    question_id         BIGINT NOT NULL,

    -- llm | python
    source              VARCHAR(50),

    -- value obtained from the question
    value               NUMERIC,

    -- literal model response before processing
    answer_text         TEXT,

    -- raw logprobs of the response (only when source='llm'), serialized as
    -- text (e.g. JSON of the {token, logprob} list returned by the provider).
    answer_logprobs     TEXT,

    -- token consumption of the call (only when source='llm')
    input_tokens        INTEGER,
    output_tokens       INTEGER,

    -- call telemetry
    latency_ms          NUMERIC(18, 3),
    cost_usd            NUMERIC(18, 8),

    -- Logical delete: recomputing features deactivates the previous active
    -- row for the same (task_id, question_id) and creates a new one. There
    -- can only be one active row per (task_id, question_id) at a time
    -- (partial unique index below).
    active              BOOLEAN NOT NULL DEFAULT TRUE,

    FOREIGN KEY (task_id)
        REFERENCES routerkit.tasks(task_id),

    FOREIGN KEY (question_id)
        REFERENCES routerkit.questions(question_id)
);


-- ============================================================
-- 5. SOLUTIONS
--
-- One row per generated solution attempt (task + candidate model + sample).
-- ============================================================

CREATE TABLE routerkit.solutions (
    solution_id         BIGSERIAL PRIMARY KEY,

    task_id             BIGINT NOT NULL,
    model_id            BIGINT NOT NULL,

    -- 0 ... n_samples - 1
    sample_idx          INTEGER NOT NULL,

    -- generated code
    code                TEXT,

    -- consumption
    tokens_in           INTEGER,
    tokens_out          INTEGER,

    -- telemetry
    latency_ms          NUMERIC(18, 3),
    cost_usd            NUMERIC(18, 8),

    -- NULL if generation succeeded
    gen_error           TEXT,

    -- Explicit indicator that this row had some error (generation failed,
    -- the service failed, or any other reason during loading). Simpler to
    -- filter than null-checks on gen_error, and also useful for loaded
    -- rows (type=load) that don't necessarily carry a text gen_error.
    has_error           BOOLEAN NOT NULL DEFAULT FALSE,

    -- Logical delete: regenerating solutions for a model (POST
    -- /solutions/generate, always by model_id) deactivates its previous
    -- active rows for that model and creates new rows. There can only be
    -- one active row per (task_id, model_id, sample_idx) at a time
    -- (partial unique index below).
    active              BOOLEAN NOT NULL DEFAULT TRUE,

    FOREIGN KEY (task_id)
        REFERENCES routerkit.tasks(task_id),

    FOREIGN KEY (model_id)
        REFERENCES routerkit.models(model_id)
);


-- ============================================================
-- 6. GRADES
--
-- Verdict for a solution. At most one ACTIVE grade per solution at a time
-- (partial unique index below) — but, same as solutions/features, its own
-- surrogate PK (not solution_id) so re-evaluating a solution follows the
-- same logical-delete pattern as the rest of the system (deactivate +
-- insert) instead of a one-off hard delete.
-- ============================================================

CREATE TABLE routerkit.grades (
    grade_id            BIGSERIAL PRIMARY KEY,

    solution_id         BIGINT NOT NULL,

    -- tests | llm_judge
    grader              VARCHAR(50) NOT NULL,

    passed              BOOLEAN,

    -- Free-form: a short status word (bigcodebench-api's `status`, e.g.
    -- "fail"/"timeout"), or a full per-test-case traceback blob (JSON,
    -- can run to several KB for real BigCodeBench runs) — unbounded on
    -- purpose, whoever writes it decides the level of detail.
    fail_reason         TEXT,

    -- execution time of the evaluation process
    exec_ms             NUMERIC(18, 3),

    -- Indicates that the validation itself failed (timeout/crash of
    -- bigcodebench-api, failure calling the LLM judge, etc.) — DIFFERENT
    -- from the evaluator having determined that the solution doesn't pass
    -- (that's passed=false + fail_reason). With has_error=true, the
    -- infrastructure failure message still goes into fail_reason.
    has_error           BOOLEAN NOT NULL DEFAULT FALSE,

    -- Logical delete: re-evaluating a solution (POST /solutions/validate)
    -- deactivates its previous active grade (if any) and inserts a new row.
    -- Also deactivated together with its solution when that solution is
    -- deactivated (see app/services/cascade.py).
    active              BOOLEAN NOT NULL DEFAULT TRUE,

    FOREIGN KEY (solution_id)
        REFERENCES routerkit.solutions(solution_id)
        ON DELETE CASCADE
);


-- ============================================================
-- 7. TARGETS
--
-- Aggregated result used as Y for training.
-- One row per (task, candidate model).
-- ============================================================

CREATE TABLE routerkit.targets (
    target_id           BIGSERIAL PRIMARY KEY,

    task_id             BIGINT NOT NULL,
    model_id            BIGINT NOT NULL,

    -- number of solutions generated
    n                   INTEGER NOT NULL,

    -- number of correct solutions
    k                   INTEGER NOT NULL,

    -- k / n
    pass_rate           NUMERIC NOT NULL,

    -- Logical delete: fully recomputed (deactivates everything, inserts
    -- anew) every time targets are aggregated from active solutions/grades.
    active              BOOLEAN NOT NULL DEFAULT TRUE,

    FOREIGN KEY (task_id)
        REFERENCES routerkit.tasks(task_id),

    FOREIGN KEY (model_id)
        REFERENCES routerkit.models(model_id)
);


-- ============================================================
-- 8. MODEL_METRICS
--
-- Aggregated metrics for a candidate model, over the active targets.
-- ============================================================

CREATE TABLE routerkit.model_metrics (
    model_metric_id     BIGSERIAL PRIMARY KEY,

    model_id            BIGINT NOT NULL,

    -- % tasks with at least one correct solution
    taskpass_pct        NUMERIC,

    -- % solutions that compile and finish
    complete_pct        NUMERIC,

    -- average pass_rate
    pass_rate_avg       NUMERIC,

    -- costs
    cost_per_task       NUMERIC(18, 8),
    cost_total          NUMERIC(18, 8),

    -- aggregated telemetry
    latency_avg_ms      NUMERIC(18, 3),
    tokens_out_avg      NUMERIC(18, 3),

    -- Logical delete: fully recomputed, same as targets.
    active              BOOLEAN NOT NULL DEFAULT TRUE,

    FOREIGN KEY (model_id)
        REFERENCES routerkit.models(model_id)
);


-- ============================================================
-- 9. PREDICTORS
--
-- Models trained via AutoML, over active features/targets.
-- ============================================================

CREATE TABLE routerkit.predictors (
    predictor_id        BIGSERIAL PRIMARY KEY,

    -- winning algorithm/model
    algorithm           VARCHAR(255) NOT NULL,

    -- model metrics
    cv_score            NUMERIC,
    test_r2             NUMERIC,
    test_mae            NUMERIC,

    -- model's input contract:
    -- expected questions/features and associated configuration
    feature_columns     JSONB NOT NULL,

    -- location of the serialized model (.joblib, etc.)
    artifact_path       TEXT,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Routing policy calibrated for THIS predictor's own Y_pred (score =
    -- Y_pred - lambda * cost/cost_max, see app/services/routing_calibration.py).
    -- Keyed by shot count ("1".."max shots"): {lambda, test_accuracy,
    -- test_cost_avg} per key — the lambda that survived the elbow method
    -- on the train-split Monte Carlo sweep, re-evaluated once on the test
    -- split for the reported accuracy/cost.
    routing_lambdas     JSONB,

    -- Full (lambda, accuracy, cost) sweep curve used to pick the winners
    -- above, keyed by shot count — kept so the curve can be audited or
    -- plotted later without re-running the whole Monte Carlo sweep.
    lambda_sweep        JSONB,

    -- Full replace, same pattern as the rest of the system: each
    -- POST /train deactivates every existing predictor before inserting
    -- the new one (see services/training.py) — there's only ever one
    -- active predictor, and that's always the one POST /route uses. To go
    -- back to an older one, retrain or reactivate it by hand.
    active              BOOLEAN NOT NULL DEFAULT TRUE
);


-- ============================================================
-- 10. ROUTE_DECISIONS
--
-- Each router decision in production + feedback.
-- ============================================================

CREATE TABLE routerkit.route_decisions (
    decision_id         BIGSERIAL PRIMARY KEY,

    predictor_id        BIGINT NOT NULL,

    -- finally selected model
    winner_model_id     BIGINT NOT NULL,

    -- moment of the decision
    ts                  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- hash of the prompt/request
    prompt_hash         VARCHAR(255),

    -- features computed for this request
    features            JSONB,

    -- predictions/scores per candidate
    ranking             JSONB,

    -- indicates whether the fallback model was used
    used_fallback       BOOLEAN NOT NULL DEFAULT FALSE,

    -- actual subsequent feedback
    outcome_success     BOOLEAN,
    outcome_cost_usd    NUMERIC(18, 8),

    FOREIGN KEY (predictor_id)
        REFERENCES routerkit.predictors(predictor_id),

    FOREIGN KEY (winner_model_id)
        REFERENCES routerkit.models(model_id)
);


-- ============================================================
-- 11. JOBS
--
-- Tracking of long-running async processes (features/compute,
-- solutions/generate, evaluation...). Logical delete like the rest of the
-- system: at most one ACTIVE row per `kind` (partial unique index below),
-- which gets updated in-place (status/message/progress) while it runs.
-- Starting a new job of the same kind deactivates the previous row and
-- creates a new one — but only if the previous one already finished
-- (done/error); if it's still pending/running, it's rejected (409 at the
-- API level) instead of overwriting an in-progress process. On container
-- startup (lifespan in app/main.py) ALL known kinds are reset: any row
-- left 'running' from a previous instance (crash, redeploy) is
-- deactivated, and each kind starts with a new row in status='empty'.
-- ============================================================

CREATE TABLE routerkit.jobs (
    job_id              BIGSERIAL PRIMARY KEY,

    -- features_compute | solutions_generate | evaluation (future)
    kind                VARCHAR(50) NOT NULL,

    -- empty (just reset, hasn't run anything yet) | pending (just created,
    -- about to start) | running | done | error
    status              VARCHAR(20) NOT NULL DEFAULT 'empty',

    -- human-readable progress, e.g. "4/1000 processed"
    message             TEXT,
    progress_current    INTEGER,
    progress_total      INTEGER,

    -- final result (same shape the synchronous endpoint used to return
    -- before this became async), or the error if status='error'
    result              JSONB,
    error               TEXT,

    started_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at         TIMESTAMPTZ,

    active              BOOLEAN NOT NULL DEFAULT TRUE
);


-- ============================================================
-- INDEXES
--
-- The PK already generates an index automatically. The partial unique
-- indexes (WHERE active) are what replace the composite UNIQUE/PK
-- constraints that used to include experiment_id: they guarantee that only
-- one active row exists per "natural identity" (e.g. a task+question, or
-- a task+model+sample), allowing inactive historical rows with that same
-- identity to coexist.
-- ============================================================

-- tasks
CREATE UNIQUE INDEX idx_tasks_bcb_task_id_active
    ON routerkit.tasks(bcb_task_id)
    WHERE bcb_task_id IS NOT NULL AND active;

CREATE UNIQUE INDEX idx_tasks_custom_task_id_active
    ON routerkit.tasks(custom_task_id)
    WHERE custom_task_id IS NOT NULL AND active;

CREATE INDEX idx_tasks_active
    ON routerkit.tasks(active);


-- questions
CREATE INDEX idx_questions_active
    ON routerkit.questions(active);

CREATE UNIQUE INDEX idx_questions_question_key_active
    ON routerkit.questions(question_key)
    WHERE question_key IS NOT NULL AND active;


-- models
CREATE INDEX idx_models_active
    ON routerkit.models(active);

-- At most one active model with role='feature' and one active model with
-- role='judge'. role='candidate' has no such limit (many candidates can
-- be active at the same time).
CREATE UNIQUE INDEX idx_models_singleton_role_active
    ON routerkit.models(role)
    WHERE active AND role IN ('feature', 'judge');

-- At most one active model per (name, role): allows several candidates to
-- be active at once (distinct names), but never two with the SAME name in
-- the same role — needed to be able to look up candidates by name without
-- ambiguity (see POST /solutions/generate). Having 2 accounts of the same
-- model active simultaneously as candidate is no longer supported; to do
-- that, deactivate one and activate the other.
CREATE UNIQUE INDEX idx_models_unique_name_role_active
    ON routerkit.models(name, role)
    WHERE active;


-- features
CREATE INDEX idx_features_task_id
    ON routerkit.features(task_id);

CREATE INDEX idx_features_question_id
    ON routerkit.features(question_id);

CREATE INDEX idx_features_active
    ON routerkit.features(active);

CREATE UNIQUE INDEX idx_features_natural_key_active
    ON routerkit.features(task_id, question_id)
    WHERE active;


-- solutions
CREATE INDEX idx_solutions_task_id
    ON routerkit.solutions(task_id);

CREATE INDEX idx_solutions_model_id
    ON routerkit.solutions(model_id);

CREATE INDEX idx_solutions_active
    ON routerkit.solutions(active);

CREATE UNIQUE INDEX idx_solutions_natural_key_active
    ON routerkit.solutions(task_id, model_id, sample_idx)
    WHERE active;


-- grades
CREATE INDEX idx_grades_active
    ON routerkit.grades(active);

CREATE UNIQUE INDEX idx_grades_natural_key_active
    ON routerkit.grades(solution_id)
    WHERE active;


-- targets
CREATE INDEX idx_targets_task_id
    ON routerkit.targets(task_id);

CREATE INDEX idx_targets_model_id
    ON routerkit.targets(model_id);

CREATE INDEX idx_targets_active
    ON routerkit.targets(active);

CREATE UNIQUE INDEX idx_targets_natural_key_active
    ON routerkit.targets(task_id, model_id)
    WHERE active;


-- model_metrics
CREATE INDEX idx_model_metrics_model_id
    ON routerkit.model_metrics(model_id);

CREATE INDEX idx_model_metrics_active
    ON routerkit.model_metrics(active);

CREATE UNIQUE INDEX idx_model_metrics_natural_key_active
    ON routerkit.model_metrics(model_id)
    WHERE active;


-- predictors
CREATE INDEX idx_predictors_active
    ON routerkit.predictors(active);

-- At most one active predictor at a time (unlike most tables, there's no
-- natural key to scope this by — it's a single global singleton, same
-- trick as idx_models_singleton_role_active but over a constant instead of
-- a role column).
CREATE UNIQUE INDEX idx_predictors_singleton_active
    ON routerkit.predictors((1))
    WHERE active;


-- route_decisions
CREATE INDEX idx_route_decisions_predictor_id
    ON routerkit.route_decisions(predictor_id);

CREATE INDEX idx_route_decisions_winner_model_id
    ON routerkit.route_decisions(winner_model_id);

CREATE INDEX idx_route_decisions_ts
    ON routerkit.route_decisions(ts);


-- jobs
CREATE INDEX idx_jobs_kind_started_at
    ON routerkit.jobs(kind, started_at DESC);

-- At most one active row per kind (the "current" one for that process type).
CREATE UNIQUE INDEX idx_jobs_unique_active_kind
    ON routerkit.jobs(kind)
    WHERE active;


COMMIT;
