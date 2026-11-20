"""Production routing policy: picks the sequence of candidates to try given
the predictor's predictions. Mirrors the reference experiment's scoring
rule exactly:

    score(m) = Y_pred(task, m) - lambda * cost(m) / cost_max

but with one lambda PER POSITION instead of one lambda for the whole
sequence: position 1 is scored with `lambdas_by_position[0]`, position 2
(among whatever's left once position 1 is removed from the pool) with
`lambdas_by_position[1]`, and so on — each position's lambda comes from
services.routing_calibration's calibration for that shot count
(routerkit.predictors.routing_lambdas, keyed "1"..str(predictor.shots)).
This function just applies the schedule, it doesn't calibrate it. No
`p_min` (minimum eligibility threshold) or manual exclusions: the
reference experiment doesn't have either.

The routing endpoint (routers/routing.py) returns exactly
`len(lambdas_by_position)` candidates (the predictor's own `shots`) — not
every active candidate — since there's no calibrated lambda for any
position beyond that, and the caller already said how many attempts it
plans to make by however it configured training.
"""

from dataclasses import dataclass


@dataclass
class RoutingDecision:
    winner_model_id: int
    ranking: list[dict]


def schedule_and_select(candidates: list[dict], *, lambdas_by_position: list[float], cost_max: float) -> RoutingDecision:
    """Picks the sequence to try: scores the full pool with
    `lambdas_by_position[0]` and takes the best as position 1, removes it,
    rescans the remainder with `lambdas_by_position[1]` for position 2, and
    so on for `len(lambdas_by_position)` positions (or until candidates run
    out, whichever comes first). Returns the winner (position 1) plus the
    ordered sequence actually picked — shorter than `lambdas_by_position`
    only if there aren't enough distinct candidates. Each entry's `score`
    is under that position's OWN lambda — scores across positions aren't
    on the same scale, since each reflects a different cost/quality
    trade-off, not "how good is this one compared to the others here."
    Raises ValueError if `candidates` is empty.
    """
    if not candidates:
        raise ValueError("no candidates to score")

    pool = list(candidates)
    ranking: list[dict] = []
    for lam in lambdas_by_position:
        if not pool:
            break
        scored = [{**c, "score": c["p"] - lam * c["price"] / cost_max} for c in pool]
        scored.sort(key=lambda c: -c["score"])
        winner = scored[0]
        ranking.append(winner)
        pool = [c for c in pool if c["model_id"] != winner["model_id"]]

    return RoutingDecision(
        winner_model_id=ranking[0]["model_id"],
        ranking=ranking,
    )
