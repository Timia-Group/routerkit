"""Production routing policy: picks the winning candidate given the
predictor's predictions. Mirrors the reference experiment's rule exactly:

    score(m) = Y_pred(task, m) - lambda * cost(m) / cost_max

`lambda` is chosen per shot count by services.routing_calibration at
training time (routerkit.predictors.routing_lambdas) — this function just
applies it, it doesn't pick it. No `p_min` (minimum eligibility threshold)
or manual exclusions: the reference experiment doesn't have either. The
routing endpoint (routers/routing.py) always scores every active
candidate in one pass and returns the full ranking as "the suggested
execution order" — there's no notion of excluding already-tried models
here, since nothing is tried yet when this runs.
"""

from dataclasses import dataclass


@dataclass
class RoutingDecision:
    winner_model_id: int
    ranking: list[dict]


def score_and_select(candidates: list[dict], *, lam: float, cost_max: float) -> RoutingDecision:
    """Scores and ranks `candidates` (each `{model_id, model_name, p,
    price}` — `p` = Y_pred, `price` = cost per task), returns the winner
    (highest score) plus the full ranking, best first. Raises ValueError
    if `candidates` is empty.
    """
    if not candidates:
        raise ValueError("no candidates to score")

    ranking = [
        {**c, "score": c["p"] - lam * c["price"] / cost_max} for c in candidates
    ]
    ranking.sort(key=lambda c: -c["score"])

    return RoutingDecision(
        winner_model_id=ranking[0]["model_id"],
        ranking=ranking,
    )
