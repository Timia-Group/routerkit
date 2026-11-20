import hashlib
from dataclasses import dataclass

import joblib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db.models import Model, Predictor, Question, RouteDecision
from app.schemas import RouteResponse
from app.services import features_llm, policy
from app.services.concurrency import run_bounded
from app.services.features_python import compute_python_feature
from app.services.routing_calibration import MAX_SHOTS, load_model_costs

# Predictors trained before `shots` existed as a column have it NULL —
# fall back to the old fixed default rather than crash on them.
_FALLBACK_SHOTS = MAX_SHOTS

router = APIRouter(tags=["routing"])


@dataclass
class _AdHocTask:
    """Stand-in for a real `Task` row, for a prompt that isn't (and won't
    be) cataloged — services.features_llm/features_python only ever read
    `.prompt` off the `task` they're given, so this is all either needs.
    """

    prompt: str


class RouteRequest(BaseModel):
    prompt: str


def _resolve_question(question_key: str, db: Session) -> Question:
    question = (
        db.query(Question).filter(Question.question_key == question_key, Question.active.is_(True)).one_or_none()
    )
    if question is None:
        raise HTTPException(
            status_code=500,
            detail=f"predictor expects question_key={question_key!r}, no longer an active question",
        )
    return question


def _compute_live_feature(task: _AdHocTask, question: Question, feature_model: Model | None) -> float:
    """Raises ScriptError/LlmFeatureError/LlmClientError/RuntimeError on
    failure — left for the caller (run_bounded collects it per-question
    instead of letting the first failure abort the rest) to turn into an
    HTTPException. Returns the full result (not just .value) so the caller
    can add up cost_usd across every question — a script question has none
    (0.0), an LLM one carries whatever services.features_llm actually
    spent on it."""
    if question.script:
        return compute_python_feature(task, question)
    if feature_model is None:
        raise RuntimeError("no active model with role='feature' to compute features")
    return features_llm.compute_llm_feature(task, question, feature_model)


@router.post("/route", response_model=RouteResponse)
def route(body: RouteRequest, db: Session = Depends(get_db)) -> dict:
    """Routes a task (`body.prompt`) to the candidate LLMs, using the
    active predictor (POST /train) and the routing policy
    (services.policy). Returns exactly `predictor.shots` candidates, in the
    order they should be tried — the caller executes them in that order
    (generate a solution, evaluate it, however they choose to) until one
    works; this endpoint only decides the order, it doesn't generate or
    evaluate anything itself. `shots` is fixed at train time (POST
    /train's `shots`), not something this endpoint takes as input — see
    services.training.

    The sequence is produced in one shot, with a DIFFERENT lambda per
    position (routerkit.predictors.routing_lambdas[str(1)]..[str(shots)],
    each calibrated by services.routing_calibration for that specific shot
    count — see services.policy.schedule_and_select) — not re-scored call
    by call as earlier attempts fail, so there's nothing to pass back in
    about what's already been tried.

    Computing live features for `prompt` (services.features_llm /
    features_python) is the only part of this that can call an LLM
    (whichever active questions aren't a pure `script`) — everything else
    is local. Every decision is recorded in routerkit.route_decisions
    (features/ranking included) whether or not it's ever followed up on.

    409 if there's no active predictor, or no active candidate the
    predictor was trained with a one-hot column for.
    """
    predictor = db.query(Predictor).filter(Predictor.active.is_(True)).one_or_none()
    if predictor is None:
        raise HTTPException(status_code=409, detail="no active predictor exists (POST /train first)")

    artifact = joblib.load(predictor.artifact_path)
    model = artifact["model"]
    feature_columns: dict = artifact["feature_columns"]
    shots = predictor.shots or _FALLBACK_SHOTS
    lambdas_by_position = [predictor.routing_lambdas[str(p)]["lambda"] for p in range(1, shots + 1)]

    # Only candidates the predictor was actually trained with a one-hot
    # column for (feature_columns["models"]) can be scored at all — an
    # active candidate added after training has no column to set to 1.
    candidates = db.query(Model).filter(Model.role == "candidate", Model.active.is_(True)).all()
    known_candidates = [c for c in candidates if c.name in feature_columns["models"]]
    if not known_candidates:
        raise HTTPException(status_code=409, detail="no active candidates to route to")

    # Same cost definition services.routing_calibration used to calibrate
    # lambda (average cost of a single real generation attempt) — NOT
    # routerkit.model_metrics.cost_per_task, a different and much larger
    # number (total historical cost across every sample ever generated for
    # a task, divided by task count). Using a different cost scale here
    # than the one lambda was tuned against would silently invalidate it.
    known_model_ids = [c.model_id for c in known_candidates]
    cost_per_model_id = load_model_costs(db, known_model_ids)
    cost_by_name = {c.name: cost_per_model_id.get(c.model_id, 0.0) for c in known_candidates}
    cost_max = max(cost_by_name.values(), default=1.0) or 1.0

    task = _AdHocTask(prompt=body.prompt)
    feature_model = db.query(Model).filter(Model.role == "feature", Model.active.is_(True)).one_or_none()
    questions = [_resolve_question(key, db) for key in feature_columns["questions"]]

    # Concurrent (up to feature_model.concurrency calls in flight), same as
    # routers/features.py's calculate path — sequential would mean waiting
    # on every one of potentially 100+ live LLM calls one at a time, in a
    # single synchronous HTTP request.
    results = run_bounded(
        questions,
        lambda q: _compute_live_feature(task, q, feature_model),
        feature_model.concurrency if feature_model else None,
    )
    live_features: dict[str, float] = {}
    features_cost_usd = 0.0
    for question, result, exc in results:
        if exc is not None:
            raise HTTPException(status_code=502, detail=f"question {question.question_key}: {exc}") from exc
        live_features[question.question_key] = result.value
        features_cost_usd += getattr(result, "cost_usd", 0.0) or 0.0

    rows = []
    for candidate in known_candidates:
        row = dict(live_features)
        for name in feature_columns["models"]:
            row[name] = 1.0 if name == candidate.name else 0.0
        rows.append(row)
    x = pd.DataFrame(rows, columns=feature_columns["questions"] + feature_columns["models"])
    predictions = model.predict(x)

    scored_candidates = [
        {
            "model_id": candidate.model_id,
            "model_name": candidate.name,
            "p": float(pred),
            "price": cost_by_name.get(candidate.name, 0.0),
        }
        for candidate, pred in zip(known_candidates, predictions, strict=True)
    ]

    decision = policy.schedule_and_select(scored_candidates, lambdas_by_position=lambdas_by_position, cost_max=cost_max)

    route_decision = RouteDecision(
        predictor_id=predictor.predictor_id,
        winner_model_id=decision.winner_model_id,
        prompt_hash=hashlib.sha256(body.prompt.encode("utf-8")).hexdigest(),
        features=live_features,
        ranking=decision.ranking,
    )
    db.add(route_decision)
    db.commit()
    db.refresh(route_decision)

    return {
        "decision_id": route_decision.decision_id,
        "lambdas_used": lambdas_by_position,
        "ranking": decision.ranking,
        "features_cost_usd": features_cost_usd,
    }
