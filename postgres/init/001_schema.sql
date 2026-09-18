BEGIN;

-- ============================================================
-- ROUTERKIT
-- PostgreSQL schema
--
-- No hay tabla "experiments": el sistema no corre configuraciones
-- paralelas. Cada carga (tasks/questions/models) reemplaza con borrado
-- lógico lo anterior (active=false) y activa solo lo nuevo, en cascada
-- "hacia adelante" (ver app/services/cascade.py) hacia todo lo que se
-- hubiera calculado a partir de esas filas. Todo proceso siempre trabaja
-- sobre "lo activo" — nunca hace falta pasar un id para saber qué usar,
-- salvo para referenciar puntualmente un registro ya generado (ej. para
-- guardarlo como FK del siguiente paso).
-- ============================================================

CREATE SCHEMA IF NOT EXISTS routerkit;


-- ============================================================
-- 1. TASKS
-- Catálogo de tareas de programación
-- ============================================================

CREATE TABLE routerkit.tasks (
    task_id         BIGSERIAL PRIMARY KEY,
    prompt          TEXT NOT NULL,
    tests           TEXT,
    entry_point     VARCHAR(255),
    source          VARCHAR(255),
    tags            TEXT[],
    created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- NULL = tarea custom, se evalúa con juez LLM (services/judge.py).
    -- Con valor (ej. "BigCodeBench/12") = tarea oficial del benchmark, se
    -- evalúa 100% dentro de bigcodebench-api contra el dataset oficial.
    bcb_task_id     VARCHAR(255),

    -- Borrado lógico. Al arrancar, routerkit-api precarga acá el catálogo
    -- oficial de BigCodeBench si la tabla está vacía (ver
    -- app/services/seed.py). Cada `POST /tasks/import` (el usuario siempre
    -- sube el jsonl, ninguna API lo descarga sola) desactiva TODAS las
    -- tasks existentes (sin importar bcb_task_id) y crea filas nuevas
    -- (nunca actualiza en el lugar) — el archivo puede traer benchmark,
    -- custom, o ambas mezcladas, cada fila decide por su propio
    -- bcb_task_id. Así cada fila queda como una foto histórica de cómo era
    -- la tarea cuando algo la usó, y no se rompen las FK de
    -- features/solutions/targets que ya apuntan a un task_id de una carga
    -- anterior. Todo proceso que busque tareas para trabajar debe filtrar
    -- active = true.
    active          BOOLEAN NOT NULL DEFAULT TRUE
);


-- ============================================================
-- 2. QUESTIONS
-- Catálogo de preguntas utilizadas para obtener features
-- ============================================================

CREATE TABLE routerkit.questions (
    question_id     BIGSERIAL PRIMARY KEY,

    text            TEXT NOT NULL,

    -- binary | numeric
    kind            VARCHAR(20) NOT NULL,

    -- rango esperado de respuesta
    lo              NUMERIC,
    hi              NUMERIC,

    author          VARCHAR(255),
    version         VARCHAR(100),

    -- Borrado lógico: cada POST /questions/import reemplaza TODO el
    -- catálogo (desactiva todas las existentes, activa solo las nuevas).
    active          BOOLEAN NOT NULL DEFAULT TRUE,

    -- NULL = la pregunta se resuelve con el LLM (el model con role='feature'
    -- activo). Con valor = texto de una función Python completa con firma
    -- fija `def task_func(task_prompt: str) -> float` (mismo nombre de
    -- función que usa BigCodeBench como entry_point); se ejecuta con el
    -- prompt de la tarea como único input y no pasa por ningún LLM (costo
    -- cero).
    script          TEXT
);


-- ============================================================
-- 3. MODELS
-- Catálogo de modelos: candidatos a rutear, o jueces (features/grading)
-- ============================================================

CREATE TABLE routerkit.models (
    model_id            BIGSERIAL PRIMARY KEY,

    name                VARCHAR(255) NOT NULL,

    -- api | hosted
    type                VARCHAR(50) NOT NULL,

    provider            VARCHAR(255),
    endpoint            TEXT,
    deployment          VARCHAR(255),

    -- nombre de la variable de entorno a usar para este modelo (permite
    -- tener varias keys del mismo provider y elegir cuál usa cada modelo).
    -- NULL = se resuelve por la convención fija de provider (ver
    -- app/clients/llm_client.py y .env.example).
    api_key_env         VARCHAR(255),

    -- precios USD / 1M tokens
    in_token_price      NUMERIC(18, 8),
    out_token_price     NUMERIC(18, 8),

    -- para modelos hosted
    cost_per_second     NUMERIC(18, 8),

    -- máximo de llamadas simultáneas a este modelo (rate limit propio del
    -- proveedor/cuenta). NULL = sin límite explícito definido todavía.
    concurrency         INTEGER,

    -- timeout de la llamada HTTP a este modelo, en segundos.
    timeout_seconds     INTEGER NOT NULL DEFAULT 300,

    -- temperatura a usar al llamar este modelo. NULL = default del proveedor.
    -- Config por modelo, no por request (ver app/clients/llm_client.py).
    temperature         NUMERIC(8, 4),

    -- candidate: se rutea/genera soluciones con él (pueden convivir varios
    -- activos, pero con name distinto entre sí). feature: calcula features
    -- vía LLM. judge: califica soluciones de tareas custom. Solo puede
    -- haber UN modelo activo con role='feature' y UN modelo activo con
    -- role='judge' en total; para candidate, el límite es "a lo sumo uno
    -- activo por name" (ver índices únicos parciales más abajo).
    role                VARCHAR(50) NOT NULL,

    -- Borrado lógico: cada POST /models/import reemplaza TODO el catálogo
    -- (desactiva todos los existentes, activa solo los nuevos). En cascada
    -- desactiva también solutions/grades/targets/model_metrics de los
    -- modelos desactivados (ver app/services/cascade.py).
    active              BOOLEAN NOT NULL DEFAULT TRUE
);


-- ============================================================
-- 4. FEATURES
--
-- Una fila por (task, question): la pregunta se ejecuta sobre la tarea
-- (LLM o script) y genera un valor de feature.
-- ============================================================

CREATE TABLE routerkit.features (
    feature_id          BIGSERIAL PRIMARY KEY,

    task_id             BIGINT NOT NULL,
    question_id         BIGINT NOT NULL,

    -- llm | python
    source              VARCHAR(50),

    -- valor obtenido de la pregunta
    value               NUMERIC,

    -- respuesta literal del modelo antes del procesamiento
    answer_text         TEXT,

    -- logprobs crudos de la respuesta (solo cuando source='llm'), serializados
    -- como texto (ej. JSON del listado de {token, logprob} devuelto por el proveedor).
    answer_logprobs     TEXT,

    -- consumo de tokens de la llamada (solo cuando source='llm')
    input_tokens        INTEGER,
    output_tokens       INTEGER,

    -- telemetría de la llamada
    latency_ms          NUMERIC(18, 3),
    cost_usd            NUMERIC(18, 8),

    -- Borrado lógico: recalcular features desactiva la fila activa anterior
    -- de la misma (task_id, question_id) y crea una nueva. Solo puede haber
    -- una fila activa por (task_id, question_id) a la vez (índice único
    -- parcial más abajo).
    active              BOOLEAN NOT NULL DEFAULT TRUE,

    FOREIGN KEY (task_id)
        REFERENCES routerkit.tasks(task_id),

    FOREIGN KEY (question_id)
        REFERENCES routerkit.questions(question_id)
);


-- ============================================================
-- 5. SOLUTIONS
--
-- Una fila por intento de solución generado (task + model candidato + sample).
-- ============================================================

CREATE TABLE routerkit.solutions (
    solution_id         BIGSERIAL PRIMARY KEY,

    task_id             BIGINT NOT NULL,
    model_id            BIGINT NOT NULL,

    -- 0 ... n_samples - 1
    sample_idx          INTEGER NOT NULL,

    -- código generado
    code                TEXT,

    -- consumo
    tokens_in           INTEGER,
    tokens_out          INTEGER,

    -- telemetría
    latency_ms          NUMERIC(18, 3),
    cost_usd            NUMERIC(18, 8),

    -- NULL si la generación fue correcta
    gen_error           TEXT,

    -- Indicador explícito de que esta fila tuvo algún error (falló la
    -- generación, falló el servicio, o cualquier otro motivo al cargarla).
    -- Más simple de filtrar que null-checks sobre gen_error, y sirve
    -- también para filas cargadas (type=load) que no necesariamente traen
    -- un gen_error de texto.
    has_error           BOOLEAN NOT NULL DEFAULT FALSE,

    -- Borrado lógico: regenerar soluciones de un model (POST
    -- /solutions/generate, siempre por model_id) desactiva sus filas
    -- activas anteriores para ese modelo y crea filas nuevas. Solo puede
    -- haber una fila activa por (task_id, model_id, sample_idx) a la vez
    -- (índice único parcial más abajo).
    active              BOOLEAN NOT NULL DEFAULT TRUE,

    FOREIGN KEY (task_id)
        REFERENCES routerkit.tasks(task_id),

    FOREIGN KEY (model_id)
        REFERENCES routerkit.models(model_id)
);


-- ============================================================
-- 6. GRADES
--
-- Veredicto de una solución.
-- Relación 1:1 con solutions.
-- ============================================================

CREATE TABLE routerkit.grades (
    solution_id         BIGINT PRIMARY KEY,

    -- tests | llm_judge
    grader              VARCHAR(50) NOT NULL,

    passed              BOOLEAN,

    -- syntax | assert | timeout | judge_no | etc.
    fail_reason         VARCHAR(255),

    -- tiempo de ejecución del proceso de evaluación
    exec_ms             NUMERIC(18, 3),

    -- Indicador de que la validación en sí falló (timeout/caída de
    -- bigcodebench-api, falla en la llamada al juez LLM, etc.) —
    -- DISTINTO de que el evaluador haya determinado que la solución no
    -- pasa (eso es passed=false + fail_reason). Con has_error=true, el
    -- mensaje de la falla de infraestructura queda en fail_reason igual.
    has_error           BOOLEAN NOT NULL DEFAULT FALSE,

    -- Se desactiva junto con su solution (ver app/services/cascade.py).
    active              BOOLEAN NOT NULL DEFAULT TRUE,

    FOREIGN KEY (solution_id)
        REFERENCES routerkit.solutions(solution_id)
        ON DELETE CASCADE
);


-- ============================================================
-- 7. TARGETS
--
-- Resultado agregado utilizado como Y para entrenamiento.
-- Una fila por (task, model candidato).
-- ============================================================

CREATE TABLE routerkit.targets (
    target_id           BIGSERIAL PRIMARY KEY,

    task_id             BIGINT NOT NULL,
    model_id            BIGINT NOT NULL,

    -- cantidad de soluciones generadas
    n                   INTEGER NOT NULL,

    -- cantidad de soluciones correctas
    k                   INTEGER NOT NULL,

    -- k / n
    pass_rate           NUMERIC NOT NULL,

    -- Borrado lógico: se recalcula por completo (desactiva todo, inserta de
    -- nuevo) cada vez que se agregan targets a partir de solutions/grades
    -- activos.
    active              BOOLEAN NOT NULL DEFAULT TRUE,

    FOREIGN KEY (task_id)
        REFERENCES routerkit.tasks(task_id),

    FOREIGN KEY (model_id)
        REFERENCES routerkit.models(model_id)
);


-- ============================================================
-- 8. MODEL_METRICS
--
-- Métricas agregadas de un modelo candidato, sobre los targets activos.
-- ============================================================

CREATE TABLE routerkit.model_metrics (
    model_metric_id     BIGSERIAL PRIMARY KEY,

    model_id            BIGINT NOT NULL,

    -- % tareas con al menos una solución correcta
    taskpass_pct        NUMERIC,

    -- % soluciones que compilan y terminan
    complete_pct        NUMERIC,

    -- promedio de pass_rate
    pass_rate_avg       NUMERIC,

    -- costos
    cost_per_task       NUMERIC(18, 8),
    cost_total          NUMERIC(18, 8),

    -- telemetría agregada
    latency_avg_ms      NUMERIC(18, 3),
    tokens_out_avg      NUMERIC(18, 3),

    -- Borrado lógico: se recalcula por completo, igual que targets.
    active              BOOLEAN NOT NULL DEFAULT TRUE,

    FOREIGN KEY (model_id)
        REFERENCES routerkit.models(model_id)
);


-- ============================================================
-- 9. PREDICTORS
--
-- Modelos entrenados mediante AutoML, sobre features/targets activos.
-- ============================================================

CREATE TABLE routerkit.predictors (
    predictor_id        BIGSERIAL PRIMARY KEY,

    -- algoritmo/modelo ganador
    algorithm           VARCHAR(255) NOT NULL,

    -- métricas del modelo
    cv_score            NUMERIC,
    test_r2             NUMERIC,
    test_mae            NUMERIC,

    -- contrato de entrada del modelo:
    -- questions/features esperadas y configuración asociada
    feature_columns     JSONB NOT NULL,

    -- ubicación del modelo serializado (.joblib, etc.)
    artifact_path       TEXT,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    is_production       BOOLEAN NOT NULL DEFAULT FALSE,

    -- No participa de la cascada automática (entrenar es un paso deliberado,
    -- no se invalida solo porque cambiaron los datos de origen).
    active              BOOLEAN NOT NULL DEFAULT TRUE
);


-- ============================================================
-- 10. ROUTE_DECISIONS
--
-- Cada decisión del router en producción + feedback.
-- ============================================================

CREATE TABLE routerkit.route_decisions (
    decision_id         BIGSERIAL PRIMARY KEY,

    predictor_id        BIGINT NOT NULL,

    -- modelo finalmente seleccionado
    winner_model_id     BIGINT NOT NULL,

    -- momento de la decisión
    ts                  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- hash del prompt/request
    prompt_hash         VARCHAR(255),

    -- features calculadas para esta request
    features            JSONB,

    -- predicciones/scores por candidato
    ranking             JSONB,

    -- indica si se utilizó el modelo fallback
    used_fallback       BOOLEAN NOT NULL DEFAULT FALSE,

    -- feedback real posterior
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
-- Seguimiento de procesos async de larga duración (features/compute,
-- solutions/generate, evaluación...). Borrado lógico igual que el resto
-- del sistema: a lo sumo una fila ACTIVA por `kind` (índice único parcial
-- más abajo), que se va actualizando in-place (status/message/progress)
-- mientras corre. Arrancar un job nuevo del mismo kind desactiva la fila
-- anterior y crea una nueva — pero solo si la anterior ya terminó
-- (done/error); si sigue pending/running, se rechaza (409 a nivel de API)
-- en vez de pisar un proceso en curso. Al iniciar el container (lifespan
-- de app/main.py) se resetean TODOS los kinds conocidos: cualquier fila
-- que haya quedado 'running' de una instancia anterior (crash, redeploy)
-- se desactiva, y cada kind arranca con una fila nueva en status='empty'.
-- ============================================================

CREATE TABLE routerkit.jobs (
    job_id              BIGSERIAL PRIMARY KEY,

    -- features_compute | solutions_generate | evaluation (a futuro)
    kind                VARCHAR(50) NOT NULL,

    -- empty (recién reseteado, nunca corrió nada todavía) | pending (recién
    -- creado, por arrancar) | running | done | error
    status              VARCHAR(20) NOT NULL DEFAULT 'empty',

    -- progreso legible, ej. "4/1000 procesadas"
    message             TEXT,
    progress_current    INTEGER,
    progress_total      INTEGER,

    -- resultado final (mismo shape que devolvía el endpoint síncrono antes
    -- de que esto fuera async), o el error si status='error'
    result              JSONB,
    error               TEXT,

    started_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at         TIMESTAMPTZ,

    active              BOOLEAN NOT NULL DEFAULT TRUE
);


-- ============================================================
-- INDEXES
--
-- PK ya genera índice automáticamente. Los índices únicos parciales
-- (WHERE active) son los que reemplazan las UNIQUE/PK compuestas que
-- tenían experiment_id: garantizan que solo exista una fila activa por
-- "identidad natural" (ej. una task+question, o una task+model+sample),
-- permitiendo que convivan filas históricas inactivas con esa misma
-- identidad.
-- ============================================================

-- tasks
CREATE UNIQUE INDEX idx_tasks_bcb_task_id_active
    ON routerkit.tasks(bcb_task_id)
    WHERE bcb_task_id IS NOT NULL AND active;

CREATE INDEX idx_tasks_active
    ON routerkit.tasks(active);


-- questions
CREATE INDEX idx_questions_active
    ON routerkit.questions(active);


-- models
CREATE INDEX idx_models_active
    ON routerkit.models(active);

-- A lo sumo un model activo con role='feature' y un model activo con
-- role='judge'. role='candidate' no tiene este límite (puede haber
-- muchos candidatos activos a la vez).
CREATE UNIQUE INDEX idx_models_singleton_role_active
    ON routerkit.models(role)
    WHERE active AND role IN ('feature', 'judge');

-- A lo sumo un model activo por (name, role): permite varios candidates
-- activos a la vez (distintos name), pero nunca dos con el MISMO name en
-- el mismo role — necesario para poder buscar candidatos por nombre sin
-- ambigüedad (ver POST /solutions/generate). Ya no se soporta tener 2
-- cuentas del mismo modelo activas simultáneamente como candidate; para
-- eso hay que desactivar una y activar la otra.
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
CREATE INDEX idx_predictors_is_production
    ON routerkit.predictors(is_production);

CREATE INDEX idx_predictors_active
    ON routerkit.predictors(active);


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

-- A lo sumo una fila activa por kind (el "actual" de ese tipo de proceso).
CREATE UNIQUE INDEX idx_jobs_unique_active_kind
    ON routerkit.jobs(kind)
    WHERE active;


COMMIT;
