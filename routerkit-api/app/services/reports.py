"""Generates the training report: a folder of charts (plotly, standalone
HTML — no server/network needed to view them) plus their underlying data
(CSV) for a human to inspect directly, every time `POST /train` runs.
Mirrors services.snapshots' current/previous rotation (a single slot that
gets overwritten each run, with one step back for recovery), just at
folder granularity instead of per-file, since a report is a bundle of
several artifacts rather than one JSON.

Every chart here is built strictly from THIS project's own data (real
solutions/grades + the freshly trained predictor) — it mirrors the shape
of the reference offline experiment's own report (same chart types, same
"shots" framing) purely so the two are easy to compare side by side, not
because any number is copied from it.

Five charts, each answering a different question about the freshly
calibrated router:
  - sankey: at lambda=0 (not the production lambda — fixed at 0 so this
    is directly comparable to the reference experiment's own sankey,
    which is also at lambda=0), for EVERY test task, which candidate
    ranks #1/#2/#3? Purely the
    deterministic ranking structure (score = Y_pred - lambda*cost/cost_max,
    no Monte Carlo at all) — ribbon width = how many tasks share that pair
    of ranks. "Shot 2"/"shot 3" here means "who the task's ranking would
    go to next if it needed to", not an actual simulated outcome — that's
    also why every column sums to the same total task count.
  - sensitivity_table: for a curated set of lambda values (REPORT_LAMBDAS)
    PLUS the exact production lambda(s) (marked with a star, however many
    decimals that takes — never rounded to the curated grid), what
    accuracy/cost would each shot budget (1/2/3) have gotten, on the test
    split?
  - frontier: at the production shot budget (MAX_SHOTS), how does the
    router's (cost, accuracy) compare to always using just one candidate
    model by itself for every attempt? Cost is the TOTAL accumulated cost
    across every test task (not the per-task average) — same unit the
    reference experiment's own frontier chart uses.
  - tradeoff_curve: tracing the full lambda grid (same LAMBDA_GRID
    routing_calibration already sweeps on train), how does the
    (cost, accuracy) curve move for each shot budget on the TEST split?
    (routing_calibration itself never needs this — it only re-evaluates
    its 3 already-picked lambdas on test, not the whole grid). X axis runs
    high-cost-to-low-cost (reversed) — lambda=0 (priciest) on the left,
    lambda=max (cheapest) on the right, same convention as the reference.
  - lambda_grid: across ALL test tasks (not a sample), which model wins
    shot 1 as lambda grows from 0 to 4 — a proportional stacked bar per
    lambda (segment width = how many of the test tasks that lambda sends
    to that model), illustrating the router "abandoning" the single
    smartest-but-priciest model as cost gets penalized harder.
"""

import os
import shutil

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Model
from app.services.routing_calibration import (
    LAMBDA_GRID,
    MAX_SHOTS,
    TEST_REPETITIONS,
    load_model_costs,
    load_outcomes_by_task_model,
    simulate_lambda,
)

REPORT_LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 2.0, 3.0, 4.0]
_REPORT_RNG_SEED = 99  # separate stream from routing_calibration's own — reporting never perturbs the persisted lambdas
# Minimum vertical share (of its column) a sankey node is allowed, even if
# its real share of the flow is smaller — otherwise a 1-task node's label
# overlaps its neighbors once node height is proportional to value (see
# _build_sankey). With up to ~7 nodes per column, 0.06 leaves enough room
# for one line of label text without flattening the big nodes' proportions.
_SANKEY_MIN_NODE_FRACTION = 0.06
_PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#EECA3B", "#9D755D",
]


def _rotate_report_dir() -> str:
    current_dir = os.path.join(settings.artifacts_dir, "current", "report")
    previous_dir = os.path.join(settings.artifacts_dir, "previous", "report")
    os.makedirs(os.path.dirname(previous_dir), exist_ok=True)
    if os.path.isdir(current_dir):
        if os.path.isdir(previous_dir):
            shutil.rmtree(previous_dir)
        shutil.move(current_dir, previous_dir)
    os.makedirs(current_dir, exist_ok=True)
    return current_dir


def _simulate_fixed_model(
    *,
    task_ids: list[int],
    outcomes_by_task_model: dict[tuple[int, int], np.ndarray],
    costs_by_model: dict[int, float],
    model_id: int,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[int, tuple[float, float]]:
    """Same bootstrap mechanics as routing_calibration.simulate_lambda, but
    with a FIXED model tried at every shot instead of a lambda-ranked one —
    "what if you just always used this one candidate, retrying it up to
    MAX_SHOTS times?" (the baseline the frontier chart compares the router
    against)."""
    passed_sum = np.zeros(MAX_SHOTS)
    cost_sum = np.zeros(MAX_SHOTS)
    counted_tasks = 0

    for task_id in task_ids:
        key = (task_id, model_id)
        if key not in outcomes_by_task_model:
            continue
        pool = outcomes_by_task_model[key]
        cost = costs_by_model[model_id]

        still_trying = np.ones(repetitions, dtype=bool)
        running_cost = np.zeros(repetitions)
        running_passed = np.zeros(repetitions, dtype=bool)
        for shot_idx in range(MAX_SHOTS):
            draws = rng.choice(pool, size=repetitions, replace=True)
            running_cost = running_cost + np.where(still_trying, cost, 0.0)
            running_passed = running_passed | (still_trying & draws)
            still_trying = still_trying & ~draws
            passed_sum[shot_idx] += running_passed.mean()
            cost_sum[shot_idx] += running_cost.mean()
        counted_tasks += 1

    if counted_tasks == 0:
        return dict.fromkeys(range(1, MAX_SHOTS + 1), (0.0, 0.0))
    return {
        shots: (passed_sum[shots - 1] / counted_tasks, cost_sum[shots - 1] / counted_tasks)
        for shots in range(1, MAX_SHOTS + 1)
    }


def _rank(task_id: int, y_pred_by_task_model: dict, costs_by_model: dict, model_ids: list[int], cost_max: float, lam: float):
    """Candidates for `task_id` ranked best-first by score, restricted to
    ones with a prediction available. Returns [(score, model_id), ...]."""
    scored = [
        (y_pred_by_task_model[(task_id, m)] - lam * costs_by_model[m] / cost_max, m)
        for m in model_ids
        if (task_id, m) in y_pred_by_task_model
    ]
    scored.sort(key=lambda t: -t[0])
    return scored


def _build_sankey(
    *,
    test_task_ids: list[int],
    y_pred_by_task_model: dict,
    costs_by_model: dict,
    model_ids: list[int],
    model_name_by_id: dict[int, str],
    cost_max: float,
    lam: float,
) -> tuple[go.Figure, pd.DataFrame]:
    """Deterministic ranking structure, no Monte Carlo: for every test
    task, who ranks #1/#2/#3 at this lambda. A ribbon from a shot-k node to
    a shot-(k+1) node is "how many tasks whose rank-k pick is A also have
    rank-(k+1) pick B" — since the ranking is a pure function of
    (task, lambda), this is exact, not sampled, and every column's nodes
    add up to the same total (every task has a well-defined full ranking,
    whether or not it would ever actually need shot 2/3)."""
    rows = []
    flow: dict[tuple[str, str], int] = {}

    for task_id in test_task_ids:
        ranked = [m for _, m in _rank(task_id, y_pred_by_task_model, costs_by_model, model_ids, cost_max, lam)]
        ranked = ranked[:MAX_SHOTS]
        if not ranked:
            continue

        row = {"task_id": task_id}
        nodes_for_task = []
        for shot_idx, model_id in enumerate(ranked):
            name = model_name_by_id[model_id]
            node = f"{name} · shot {shot_idx + 1}"
            row[f"shot{shot_idx + 1}_model"] = name
            nodes_for_task.append(node)
        for a, b in zip(nodes_for_task, nodes_for_task[1:], strict=False):
            flow[(a, b)] = flow.get((a, b), 0) + 1
        rows.append(row)

    # Explicit columns (arrangement="fixed") instead of plotly's default
    # force-directed layout, which draws long ribbons that visually cross
    # over unrelated columns and made the first version of this chart
    # unreadable.
    model_names_sorted = sorted({model_name_by_id[m] for m in model_ids})
    color_by_model = {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(model_names_sorted)}
    n_tasks = len(rows)

    # A node's "size" is however many tasks pass through it — its outgoing
    # flow for shot 1/2 (nothing flows OUT of a shot-3 node, it's the last
    # column), its incoming flow otherwise. Ordering each column by size
    # (largest first), not alphabetically, is what actually makes a Sankey
    # readable — it's why the reference chart has almost no ribbon
    # crossings and this one, before this fix, looked like a tangle.
    node_outgoing: dict[str, int] = {}
    node_incoming: dict[str, int] = {}
    for (a, b), v in flow.items():
        node_outgoing[a] = node_outgoing.get(a, 0) + v
        node_incoming[b] = node_incoming.get(b, 0) + v
    all_nodes = {n for pair in flow for n in pair}
    node_size = {n: node_outgoing.get(n, node_incoming.get(n, 0)) for n in all_nodes}

    columns: list[list[str]] = [[] for _ in range(MAX_SHOTS)]
    for shot_idx in range(1, MAX_SHOTS + 1):
        col_nodes = [n for n in all_nodes if n.endswith(f"· shot {shot_idx}")]
        columns[shot_idx - 1] = sorted(col_nodes, key=lambda n: -node_size[n])

    nodes = [n for col in columns for n in col]
    node_index = {n: i for i, n in enumerate(nodes)}
    node_x, node_y = [], []
    for col_idx, col in enumerate(columns):
        x = 0.001 if col_idx == 0 else 0.999 if col_idx == len(columns) - 1 else col_idx / (len(columns) - 1)
        # Position each node's center by its cumulative share of the
        # column's total flow, not by plain row index — stacking by value
        # (like a conventional Sankey) packs large nodes tightly instead of
        # giving a 1-task node the same vertical slot as a 338-task one,
        # which is what made the first version of this chart look sprawled
        # out compared to the reference's compact layout. A pure proportional
        # share, though, squeezes a 1-2 task node into a sliver too thin for
        # its own label to fit without overlapping its neighbors — so each
        # node's share is floored to _MIN_NODE_FRACTION before stacking
        # (then the column's shares are renormalized back to 1 — this only
        # meaningfully shrinks the dominant node(s), everything already
        # above the floor is practically unaffected).
        col_total = sum(node_size[n] for n in col) or 1
        raw_fractions = [node_size[n] / col_total for n in col]
        fractions = [max(_SANKEY_MIN_NODE_FRACTION, f) for f in raw_fractions]
        fractions_total = sum(fractions) or 1
        cumulative = 0.0
        for _n, frac in zip(col, fractions, strict=True):
            y = (cumulative + frac / 2) / fractions_total
            node_x.append(x)
            node_y.append(min(0.999, max(0.001, y)))
            cumulative += frac

    def _node_color(n: str) -> str:
        return color_by_model[n.split(" · ")[0]]

    def _hex_to_rgba(hex_color: str, alpha: float) -> str:
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"

    link_colors = [_hex_to_rgba(_node_color(a), 0.45) for a, _ in flow]

    # Count + percentage right in the node label (same as the reference)
    # instead of only on hover — "shot N" is already conveyed by the
    # column headers below, so the label itself is just name + share.
    def _node_label(n: str) -> str:
        name = n.split(" · ")[0]
        count = node_size[n]
        pct = round(100 * count / n_tasks)
        return f"{name} · {count} ({pct}%)"

    # Custom hover for both nodes and links — Plotly's own defaults are
    # generic and inconsistent with the labels actually shown: a node's
    # default hover is "incoming/outgoing flow count" (meaningless jargon,
    # and always 0 on one side for a shot-1/shot-3 node since nothing
    # flows in/out of the chart's edges), and a link's default hover is
    # "source"/"target"/"value" instead of the real model names.
    link_hover_text = [
        f"{a.split(' · ')[0]} → {b.split(' · ')[0]}: {v} tasks ({round(100 * v / n_tasks)}%)"
        for (a, b), v in flow.items()
    ]

    fig = go.Figure(
        go.Sankey(
            arrangement="fixed",
            node={
                "label": [_node_label(n) for n in nodes],
                "color": [_node_color(n) for n in nodes],
                "x": node_x,
                "y": node_y,
                "pad": 14,
                "thickness": 16,
                "hovertemplate": "%{label}<extra></extra>",
            },
            link={
                "source": [node_index[a] for a, _ in flow],
                "target": [node_index[b] for _, b in flow],
                "value": list(flow.values()),
                "color": link_colors,
                "customdata": link_hover_text,
                "hovertemplate": "%{customdata}<extra></extra>",
            },
        )
    )
    for shot_idx in range(1, MAX_SHOTS + 1):
        x = 0.001 if shot_idx == 1 else 0.999 if shot_idx == MAX_SHOTS else (shot_idx - 1) / (MAX_SHOTS - 1)
        fig.add_annotation(
            x=x, y=1.0, xref="paper", yref="paper", yanchor="bottom", showarrow=False,
            text=f"<b>SHOT {shot_idx}</b>", font={"size": 13},
        )
    fig.update_layout(
        title=(
            f"Router paths (λ={lam:.0f}): where each attempt goes"
            f"<br><sub>{n_tasks} test tasks · deterministic ranking "
            "(no Monte Carlo, not the production λ — fixed at λ=0 to compare directly against "
            "the reference experiment) · ribbon width = tasks sharing that pair of ranking "
            "positions</sub>"
        ),
        font_size=12,
        margin={"t": 150},
    )
    return fig, pd.DataFrame(rows)


def _build_sensitivity_table(
    *,
    test_task_ids: list[int],
    y_pred_by_task_model: dict,
    outcomes_by_task_model: dict,
    costs_by_model: dict,
    model_ids: list[int],
    cost_max: float,
    routing_lambdas: dict,
    rng: np.random.Generator,
) -> tuple[go.Figure, pd.DataFrame]:
    # The curated grid (REPORT_LAMBDAS) alone never lands exactly on the
    # real calibrated lambda for any shot count — plus the exact production
    # value(s), never rounded to the grid, each marked with which shot(s)
    # it was actually picked for.
    production_by_lambda: dict[float, list[int]] = {}
    for shots in range(1, MAX_SHOTS + 1):
        production_by_lambda.setdefault(routing_lambdas[str(shots)]["lambda"], []).append(shots)
    all_lambdas = sorted(set(REPORT_LAMBDAS) | set(production_by_lambda))

    n_tasks = len(test_task_ids)
    rows = []
    for lam in all_lambdas:
        result = simulate_lambda(
            task_ids=test_task_ids,
            y_pred_by_task_model=y_pred_by_task_model,
            outcomes_by_task_model=outcomes_by_task_model,
            costs_by_model=costs_by_model,
            model_ids=model_ids,
            cost_max=cost_max,
            lam=lam,
            repetitions=TEST_REPETITIONS,
            rng=rng,
        )
        row = {"lambda": lam, "is_production_for_shots": production_by_lambda.get(lam, [])}
        for shots in range(1, MAX_SHOTS + 1):
            accuracy, cost_avg = result[shots]
            row[f"shot{shots}_accuracy"] = accuracy
            row[f"shot{shots}_cost_total"] = cost_avg * n_tasks
        rows.append(row)
    df = pd.DataFrame(rows)

    def _lambda_label(lam: float, production_shots: list[int]) -> str:
        if production_shots:
            shots_label = ",".join(str(s) for s in production_shots)
            return f"★ λ = {lam:.6f} (shot {shots_label})"
        return f"λ = {lam:.2f}"

    header = ["λ"] + [
        f"Shot {s}" + (" (cumulative)" if s == 2 else " (total)" if s == 3 else "") for s in range(1, MAX_SHOTS + 1)
    ]
    cell_text = [
        [_lambda_label(lam, shots) for lam, shots in zip(df["lambda"], df["is_production_for_shots"], strict=True)]
    ]
    # Production-lambda rows get a distinct pale-blue fill in the λ column
    # ONLY — kept visually separate from the per-column "best accuracy"
    # gold below, since those are two unrelated claims (this row is what
    # was actually picked; that cell is simply the highest raw accuracy
    # in its column, ignoring cost) that happened to collide on the same
    # color before, which read as "these cells all claim the same thing."
    cell_fill = [
        ["#A6CEE3" if shots else "white" for shots in df["is_production_for_shots"]]
    ]
    for shots in range(1, MAX_SHOTS + 1):
        acc = df[f"shot{shots}_accuracy"]
        cost = df[f"shot{shots}_cost_total"]
        cell_text.append([f"<b>{a * 100:.1f}%</b><br>${c:.2f}" for a, c in zip(acc, cost)])
        best_idx = acc.idxmax()
        lo, hi = acc.min(), acc.max()
        span = (hi - lo) or 1.0
        colors = []
        for i, a in enumerate(acc):
            if i == best_idx:
                colors.append("#F4C430")  # highlight: best RAW accuracy in this column, ignoring cost
            else:
                t = (a - lo) / span
                colors.append(f"rgba(76,120,168,{0.15 + 0.55 * t:.2f})")
        cell_fill.append(colors)

    fig = go.Figure(
        go.Table(
            header={"values": header, "fill_color": "#2b2b2b", "font": {"color": "white"}, "align": "center"},
            cells={"values": cell_text, "fill_color": cell_fill, "align": "center", "height": 28},
        )
    )
    fig.update_layout(
        title=f"λ sensitivity on TEST ({n_tasks} tasks, λ across the {MAX_SHOTS} shots)<br>"
        "<sub>% solved and total cost (all tasks) — gold: best raw accuracy in that "
        "column (cost aside) · blue: exact production λ (★, not rounded to the grid — "
        "the same value marked in tradeoff_curve). These are two different things: the "
        "production one isn't meant to also be the best-accuracy one, since it deliberately "
        "trades some accuracy for much lower cost.</sub>"
    )
    return fig, df


def _build_frontier(
    *,
    test_task_ids: list[int],
    outcomes_by_task_model: dict,
    costs_by_model: dict,
    model_ids: list[int],
    model_name_by_id: dict[int, str],
    routing_lambdas: dict,
    rng: np.random.Generator,
) -> tuple[go.Figure, pd.DataFrame]:
    rows = []
    for model_id in model_ids:
        result = _simulate_fixed_model(
            task_ids=test_task_ids,
            outcomes_by_task_model=outcomes_by_task_model,
            costs_by_model=costs_by_model,
            model_id=model_id,
            repetitions=TEST_REPETITIONS,
            rng=rng,
        )
        accuracy, cost_avg = result[MAX_SHOTS]
        n_tasks = len(test_task_ids)
        rows.append(
            {
                "name": model_name_by_id[model_id],
                "accuracy": accuracy,
                "cost_total": cost_avg * n_tasks,
                "is_router": False,
            }
        )

    router = routing_lambdas[str(MAX_SHOTS)]
    rows.append(
        {
            "name": "Router GB",
            "accuracy": router["test_accuracy"],
            "cost_total": router["test_cost_avg"] * len(test_task_ids),
            "is_router": True,
        }
    )
    df = pd.DataFrame(rows)

    fig = go.Figure()
    for _, r in df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[r["cost_total"]],
                y=[r["accuracy"] * 100],
                mode="markers+text",
                marker={"size": 22 if r["is_router"] else 16, "color": "#E45756" if r["is_router"] else "#4C78A8"},
                text=[f"<b>{r['name']}</b><br>{r['accuracy'] * 100:.1f}% · ${r['cost_total']:.2f}"],
                textposition="top center",
                showlegend=False,
            )
        )
    fig.update_layout(
        title=(
            f"Cost-quality frontier TEST — shot {MAX_SHOTS} (total) · λ = {router['lambda']:.4f} "
            f"({len(test_task_ids)} tasks, Monte Carlo {TEST_REPETITIONS} replicates)"
        ),
        xaxis={"title": "Total cost (USD, all tasks) →", "tickformat": ".2f"},
        yaxis_title="Quality (% solved) →",
    )
    return fig, df


def _build_tradeoff_curve(
    *,
    test_task_ids: list[int],
    y_pred_by_task_model: dict,
    outcomes_by_task_model: dict,
    costs_by_model: dict,
    model_ids: list[int],
    cost_max: float,
    routing_lambdas: dict,
    rng: np.random.Generator,
) -> tuple[go.Figure, pd.DataFrame]:
    n_tasks = len(test_task_ids)
    rows = []
    for lam in LAMBDA_GRID:
        result = simulate_lambda(
            task_ids=test_task_ids,
            y_pred_by_task_model=y_pred_by_task_model,
            outcomes_by_task_model=outcomes_by_task_model,
            costs_by_model=costs_by_model,
            model_ids=model_ids,
            cost_max=cost_max,
            lam=float(lam),
            repetitions=TEST_REPETITIONS,
            rng=rng,
        )
        for shots in range(1, MAX_SHOTS + 1):
            accuracy, cost_avg = result[shots]
            rows.append({"lambda": float(lam), "shots": shots, "accuracy": accuracy, "cost_total": cost_avg * n_tasks})
    df = pd.DataFrame(rows)

    fig = go.Figure()
    for shots in range(1, MAX_SHOTS + 1):
        sub = df[df["shots"] == shots].sort_values("lambda")
        fig.add_trace(
            go.Scatter(
                x=sub["cost_total"],
                y=sub["accuracy"] * 100,
                mode="lines+markers",
                name=f"Shot {shots}",
                marker={"size": 5},
                customdata=sub["lambda"],
                hovertemplate="λ=%{customdata:.3f}<br>$%{x:.2f} · %{y:.1f}%<extra></extra>",
            )
        )

    # Marks exactly where the chosen lambda landed — this is the
    # routing_lambdas["test_accuracy"/"test_cost_avg"] point computed on
    # ITS OWN separate Monte Carlo pass. Without this marker the curve
    # gives no clue which point was actually picked, and the elbow was
    # never chosen from THIS curve anyway (see below), so eyeballing this
    # curve's own peak for it is misleading.
    for shots in range(1, MAX_SHOTS + 1):
        picked = routing_lambdas[str(shots)]
        fig.add_trace(
            go.Scatter(
                x=[picked["test_cost_avg"] * n_tasks],
                y=[picked["test_accuracy"] * 100],
                mode="markers+text",
                marker={"size": 14, "symbol": "star", "color": "black"},
                text=[f"λ chosen (shot {shots})" if shots == MAX_SHOTS else ""],
                textposition="bottom center",
                showlegend=False,
                hovertemplate=f"<b>λ chosen, shot {shots}</b><br>λ=%{{customdata:.3f}}<br>$%{{x:.2f}} · %{{y:.1f}}%<extra></extra>",
                customdata=[picked["lambda"]],
            )
        )

    fig.update_layout(
        title=(
            f"Router trade-off on TEST: tasks solved vs. cost, swept across λ<br>"
            f"<sub>{len(test_task_ids)} tasks · Monte Carlo {TEST_REPETITIONS} replicates · "
            f"each curve runs from λ={LAMBDA_GRID[0]:.0f} to λ={LAMBDA_GRID[-1]:.0f} (only up to {MAX_SHOTS} shots, "
            "which is what's calibrated for production) · ★ = λ chosen for that shot.<br>"
            "λ is picked on a separate sweep over the TRAIN split (not this curve, which is TEST-only) — "
            "that's why the ★ doesn't have to land exactly on this curve's own peak.</sub>"
        ),
        xaxis={
            "title": "Total cost (USD, all tasks) → cheaper",
            "tickformat": ".2f",
            "autorange": "reversed",
        },
        yaxis_title="% of tasks solved →",
    )
    return fig, df


def _build_lambda_grid_heatmap(
    *,
    test_task_ids: list[int],
    y_pred_by_task_model: dict,
    costs_by_model: dict,
    model_ids: list[int],
    model_name_by_id: dict[int, str],
    cost_max: float,
) -> tuple[go.Figure, pd.DataFrame]:
    """A grid of boxes: one row per lambda (REPORT_LAMBDAS), one column per
    shot (1..MAX_SHOTS) — mirrors the reference experiment's own "15 DEL
    ENRUTADOR" chart, just MAX_SHOTS columns instead of its 6. Each box is
    a condensed proportional mosaic (every model's segment width = its
    share of ALL test tasks at that (lambda, shot)) — deterministic
    ranking, no Monte Carlo, no sampling."""
    names = sorted({model_name_by_id[m] for m in model_ids})
    color_by_model = {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(names)}
    n_tasks = len(test_task_ids)

    keys = [(lam, shots) for lam in REPORT_LAMBDAS for shots in range(1, MAX_SHOTS + 1)]
    counts = {key: dict.fromkeys(names, 0) for key in keys}
    for lam in REPORT_LAMBDAS:
        for task_id in test_task_ids:
            ranked = _rank(task_id, y_pred_by_task_model, costs_by_model, model_ids, cost_max, lam)
            for shots in range(1, MAX_SHOTS + 1):
                if shots > len(ranked):
                    continue
                winner_name = model_name_by_id[ranked[shots - 1][1]]
                counts[(lam, shots)][winner_name] += 1

    df = pd.DataFrame(counts).T  # index=(lambda, shots), columns=model name, values=task count
    df.index = [f"λ={lam:.2f} · shot {shots}" for lam, shots in keys]

    n_rows = len(REPORT_LAMBDAS)
    box_w, row_gap, col_gap = 1.0, 0.18, 0.10
    col_stride = box_w + col_gap

    def _box_x(shots: int) -> tuple[float, float]:
        x0 = (shots - 1) * col_stride
        return x0, x0 + box_w

    def _box_y(row_idx: int) -> tuple[float, float]:
        # row 0 (lambda=0) at the TOP: highest y band first.
        y1 = n_rows - row_idx - row_gap / 2
        y0 = n_rows - row_idx - 1 + row_gap / 2
        return y0, y1

    short_name = {
        "GPT-5.6 Sol": "Sol", "GPT-5.6 Luna": "Luna", "GPT-OSS-120B": "OSS",
        "Claude Sonnet 5": "Claude", "Grok 4.6": "Grok",
        "Qwen3.6-35B-A3B-FP8": "Qwen", "Devstral-Small-2-24B": "Dev",
    }

    fig = go.Figure()
    for row_idx, lam in enumerate(REPORT_LAMBDAS):
        y0, y1 = _box_y(row_idx)
        for shots in range(1, MAX_SHOTS + 1):
            x0, x1 = _box_x(shots)
            cum = 0.0
            for name in names:
                count = counts[(lam, shots)][name]
                if count == 0:
                    continue
                frac = count / n_tasks
                seg_x0, seg_x1 = x0 + cum * box_w, x0 + (cum + frac) * box_w
                fig.add_shape(
                    type="rect", x0=seg_x0, x1=seg_x1, y0=y0, y1=y1,
                    fillcolor=color_by_model[name], line={"width": 0.5, "color": "white"},
                )
                if frac > 0.05:
                    fig.add_annotation(
                        x=(seg_x0 + seg_x1) / 2, y=(y0 + y1) / 2, showarrow=False,
                        text=short_name.get(name, name.split()[0]), font={"size": 10, "color": "white"},
                    )
                cum += frac

    for shots in range(1, MAX_SHOTS + 1):
        x0, x1 = _box_x(shots)
        fig.add_annotation(
            x=(x0 + x1) / 2, y=n_rows + 0.3, showarrow=False,
            text=f"<b>SHOT {shots}</b>", font={"size": 13},
        )

    row_centers = [(_box_y(i)[0] + _box_y(i)[1]) / 2 for i in range(n_rows)]
    fig.update_layout(
        title=(
            f"Who the router assigns each shot to, by λ (full grid: {n_rows} λ × {MAX_SHOTS} shots, "
            f"{n_tasks} test tasks)<br>"
            "<sub>deterministic ranking (no Monte Carlo) — each box represents all test tasks: a "
            f"segment's width is the fraction of the {n_tasks} tasks going to that model at that "
            "(λ, shot)</sub>"
        ),
        xaxis={"visible": False, "range": [-0.05, (MAX_SHOTS - 1) * col_stride + box_w + 0.05]},
        yaxis={
            "tickvals": row_centers,
            "ticktext": [f"λ={lam:.2f}" for lam in REPORT_LAMBDAS],
            "range": [0, n_rows + 0.6],
        },
        showlegend=False,
        height=max(700, 45 * n_rows),
        plot_bgcolor="white",
    )
    return fig, df


def generate_training_report(
    db: Session,
    *,
    model,
    x: pd.DataFrame,
    row_task_ids: pd.Series,
    row_model_ids: pd.Series,
    feature_columns: dict,
    test_task_ids: list[int],
    routing_lambdas: dict,
) -> str:
    """Builds the 5-chart training report (see module docstring) into
    `{ARTIFACTS_DIR}/current/report/` (rotating any previous one to
    `previous/report/`), from the just-trained `model` and the just-picked
    `routing_lambdas`. Returns the output directory. Never touches the
    persisted predictor/calibration — purely a read-only visualization
    pass over the same data `calibrate_routing` already used."""
    out_dir = _rotate_report_dir()

    candidates = (
        db.query(Model)
        .filter(Model.role == "candidate", Model.active.is_(True), Model.name.in_(feature_columns["models"]))
        .all()
    )
    model_ids = [m.model_id for m in candidates]
    model_name_by_id = {m.model_id: m.name for m in candidates}

    outcomes_by_task_model = load_outcomes_by_task_model(db, model_ids)
    costs_by_model = load_model_costs(db, model_ids)
    cost_max = max(costs_by_model.values()) if costs_by_model else 1.0
    if cost_max <= 0:
        cost_max = 1.0

    predictions = model.predict(x)
    y_pred_by_task_model = {
        (int(t), int(m)): float(p) for t, m, p in zip(row_task_ids, row_model_ids, predictions, strict=True)
    }

    rng = np.random.default_rng(_REPORT_RNG_SEED)

    charts = [
        (
            "sankey",
            lambda: _build_sankey(
                test_task_ids=test_task_ids,
                y_pred_by_task_model=y_pred_by_task_model,
                costs_by_model=costs_by_model,
                model_ids=model_ids,
                model_name_by_id=model_name_by_id,
                cost_max=cost_max,
                # Fixed at 0 (pure quality ranking, no cost penalty) to
                # match the reference experiment's own sankey, which is
                # also at lambda=0 — not the production lambda, so this
                # one is directly comparable rather than showing a
                # different baseline than what's being compared against.
                lam=0.0,
            ),
        ),
        (
            "sensitivity_table",
            lambda: _build_sensitivity_table(
                test_task_ids=test_task_ids,
                y_pred_by_task_model=y_pred_by_task_model,
                outcomes_by_task_model=outcomes_by_task_model,
                costs_by_model=costs_by_model,
                model_ids=model_ids,
                cost_max=cost_max,
                routing_lambdas=routing_lambdas,
                rng=rng,
            ),
        ),
        (
            "frontier",
            lambda: _build_frontier(
                test_task_ids=test_task_ids,
                outcomes_by_task_model=outcomes_by_task_model,
                costs_by_model=costs_by_model,
                model_ids=model_ids,
                model_name_by_id=model_name_by_id,
                routing_lambdas=routing_lambdas,
                rng=rng,
            ),
        ),
        (
            "tradeoff_curve",
            lambda: _build_tradeoff_curve(
                test_task_ids=test_task_ids,
                y_pred_by_task_model=y_pred_by_task_model,
                outcomes_by_task_model=outcomes_by_task_model,
                costs_by_model=costs_by_model,
                model_ids=model_ids,
                cost_max=cost_max,
                routing_lambdas=routing_lambdas,
                rng=rng,
            ),
        ),
        (
            "lambda_grid",
            lambda: _build_lambda_grid_heatmap(
                test_task_ids=test_task_ids,
                y_pred_by_task_model=y_pred_by_task_model,
                costs_by_model=costs_by_model,
                model_ids=model_ids,
                model_name_by_id=model_name_by_id,
                cost_max=cost_max,
            ),
        ),
    ]

    index_rows = []
    for name, builder in charts:
        fig, df = builder()
        # Full inline embed (not include_plotlyjs="cdn"): these need to
        # open straight from disk with no internet access — a "cdn" embed
        # would leave the page blank offline, defeating the point of
        # writing it to a local folder in the first place.
        fig.write_html(os.path.join(out_dir, f"{name}.html"), include_plotlyjs=True)
        # A chart that set its own layout height (e.g. lambda_grid, with
        # MAX_SHOTS*len(REPORT_LAMBDAS) rows) keeps it instead of being
        # squashed into the 900px default every other chart uses.
        png_height = fig.layout.height or 900
        fig.write_image(os.path.join(out_dir, f"{name}.png"), width=1400, height=png_height, scale=2)
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
        index_rows.append(name)

    index_html = "<html><body><h1>Training report</h1><ul>" + "".join(
        f'<li><a href="{name}.html">{name}</a> '
        f'(<a href="{name}.png">png</a>, <a href="{name}.csv">data</a>)</li>'
        for name in index_rows
    ) + "</ul></body></html>"
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)

    return out_dir
