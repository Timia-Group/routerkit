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
  - sankey: for a REAL prompt going through the router (lambda = the one
    actually used in production, routing_lambdas[str(MAX_SHOTS)]), where
    does each attempt actually go, and how often does it take 1 vs 2 vs 3
    shots? (one reproducible Monte Carlo realization per test task, not an
    aggregate — see _build_sankey)
  - sensitivity_table: for a curated set of lambda values (REPORT_LAMBDAS,
    the same list for all 3 charts that use it), what accuracy/cost would
    each shot budget (1/2/3) have gotten, on the test split?
  - frontier: at the production shot budget (MAX_SHOTS), how does the
    router's (cost, accuracy) compare to always using just one candidate
    model by itself for every attempt?
  - tradeoff_curve: tracing the full lambda grid (same LAMBDA_GRID
    routing_calibration already sweeps on train), how does the
    (cost, accuracy) curve move for each shot budget on the TEST split?
    (routing_calibration itself never needs this — it only re-evaluates
    its 3 already-picked lambdas on test, not the whole grid)
  - lambda_grid: for a small sample of test tasks, which model would shot 1
    actually go to as lambda grows from 0 to 4? (deterministic ranking,
    no Monte Carlo — illustrates the router "abandoning" the single
    smartest-but-priciest model as cost gets penalized harder)
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
HEATMAP_SAMPLE_SIZE = 15
_REPORT_RNG_SEED = 99  # separate stream from routing_calibration's own — reporting never perturbs the persisted lambdas
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
    outcomes_by_task_model: dict,
    costs_by_model: dict,
    model_ids: list[int],
    model_name_by_id: dict[int, str],
    cost_max: float,
    lam: float,
    rng: np.random.Generator,
) -> tuple[go.Figure, pd.DataFrame]:
    rows = []
    flow: dict[tuple[str, str], int] = {}

    for task_id in test_task_ids:
        ranked = [m for _, m in _rank(task_id, y_pred_by_task_model, costs_by_model, model_ids, cost_max, lam)]
        ranked = [m for m in ranked if (task_id, m) in outcomes_by_task_model][:MAX_SHOTS]
        if not ranked:
            continue

        row = {"task_id": task_id}
        prev_node = None
        resolved = False
        for shot_idx, model_id in enumerate(ranked):
            name = model_name_by_id[model_id]
            node = f"{name} · shot {shot_idx + 1}"
            passed = bool(rng.choice(outcomes_by_task_model[(task_id, model_id)]))
            row[f"shot{shot_idx + 1}_model"] = name
            row[f"shot{shot_idx + 1}_passed"] = passed

            if prev_node is not None:
                flow[(prev_node, node)] = flow.get((prev_node, node), 0) + 1
            if passed:
                terminal = f"Resuelto en shot {shot_idx + 1}"
                flow[(node, terminal)] = flow.get((node, terminal), 0) + 1
                resolved = True
                break
            prev_node = node
        if not resolved and ranked:
            last_node = f"{model_name_by_id[ranked[-1]]} · shot {len(ranked)}"
            flow[(last_node, "Sin resolver")] = flow.get((last_node, "Sin resolver"), 0) + 1
        rows.append(row)

    # Explicit columns (arrangement="fixed") instead of plotly's default
    # force-directed layout — left as automatic, the few tasks resolved
    # right at shot 1 still draw a ribbon spanning the FULL width straight
    # to the terminal nodes, visually crossing over the shot2/shot3
    # columns as if passing "through" them, which is what made the first
    # version of this chart unreadable.
    model_names_sorted = sorted({model_name_by_id[m] for m in model_ids})
    color_by_model = {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(model_names_sorted)}
    terminal_order = [f"Resuelto en shot {s}" for s in range(1, MAX_SHOTS + 1)] + ["Sin resolver"]

    columns: list[list[str]] = [[] for _ in range(MAX_SHOTS + 1)]
    for shot_idx in range(1, MAX_SHOTS + 1):
        columns[shot_idx - 1] = sorted(n for n in {p for pair in flow for p in pair} if n.endswith(f"· shot {shot_idx}"))
    columns[MAX_SHOTS] = [t for t in terminal_order if any(t in pair for pair in flow)]

    nodes = [n for col in columns for n in col]
    node_index = {n: i for i, n in enumerate(nodes)}
    node_x, node_y = [], []
    for col_idx, col in enumerate(columns):
        x = 0.001 if col_idx == 0 else 0.999 if col_idx == len(columns) - 1 else col_idx / (len(columns) - 1)
        for row_idx, _n in enumerate(col):
            node_x.append(x)
            node_y.append((row_idx + 0.5) / max(len(col), 1))

    def _node_color(n: str) -> str:
        if n.startswith("Resuelto"):
            return "#54A24B"
        if n == "Sin resolver":
            return "#B0B0B0"
        return color_by_model[n.split(" · ")[0]]

    def _hex_to_rgba(hex_color: str, alpha: float) -> str:
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"

    link_colors = [_hex_to_rgba(_node_color(a), 0.45) for a, _ in flow]

    fig = go.Figure(
        go.Sankey(
            arrangement="fixed",
            node={
                "label": nodes,
                "color": [_node_color(n) for n in nodes],
                "x": node_x,
                "y": node_y,
                "pad": 14,
                "thickness": 16,
            },
            link={
                "source": [node_index[a] for a, _ in flow],
                "target": [node_index[b] for _, b in flow],
                "value": list(flow.values()),
                "color": link_colors,
            },
        )
    )
    n_tasks = len(rows)
    fig.update_layout(
        title=(
            f"Caminos del router (λ={lam:.4f}, el de producción para {MAX_SHOTS} shots): "
            f"a quién va cada intento<br><sub>{n_tasks} tareas de test · el shot k solo se ejecuta si el anterior falló "
            "(una realización de Monte Carlo)</sub>"
        ),
        font_size=12,
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
    rng: np.random.Generator,
) -> tuple[go.Figure, pd.DataFrame]:
    rows = []
    for lam in REPORT_LAMBDAS:
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
        row = {"lambda": lam}
        for shots in range(1, MAX_SHOTS + 1):
            accuracy, cost = result[shots]
            row[f"shot{shots}_accuracy"] = accuracy
            row[f"shot{shots}_cost_avg"] = cost
        rows.append(row)
    df = pd.DataFrame(rows)

    header = ["λ"] + [f"Shot {s} ({'acum.' if s == 2 else 'total' if s == 3 else ''})" for s in range(1, MAX_SHOTS + 1)]
    cell_text = [[f"λ = {lam:.2f}" for lam in df["lambda"]]]
    cell_fill = [["white"] * len(df)]
    for shots in range(1, MAX_SHOTS + 1):
        acc = df[f"shot{shots}_accuracy"]
        cost = df[f"shot{shots}_cost_avg"]
        cell_text.append([f"<b>{a * 100:.1f}%</b><br>${c:.4f}" for a, c in zip(acc, cost)])
        best_idx = acc.idxmax()
        lo, hi = acc.min(), acc.max()
        span = (hi - lo) or 1.0
        colors = []
        for i, a in enumerate(acc):
            if i == best_idx:
                colors.append("#F4C430")  # highlight: best accuracy in this column
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
        title=f"Sensibilidad de λ en TEST ({len(test_task_ids)} tareas, λ en los {MAX_SHOTS} shots)<br>"
        "<sub>% resueltas y costo acumulado por tarea — resaltado: mejor accuracy por columna</sub>"
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
        accuracy, cost = result[MAX_SHOTS]
        rows.append({"name": model_name_by_id[model_id], "accuracy": accuracy, "cost_avg": cost, "is_router": False})

    router = routing_lambdas[str(MAX_SHOTS)]
    rows.append(
        {
            "name": "Router GB",
            "accuracy": router["test_accuracy"],
            "cost_avg": router["test_cost_avg"],
            "is_router": True,
        }
    )
    df = pd.DataFrame(rows)

    fig = go.Figure()
    for _, r in df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[r["cost_avg"]],
                y=[r["accuracy"] * 100],
                mode="markers+text",
                marker={"size": 22 if r["is_router"] else 16, "color": "#E45756" if r["is_router"] else "#4C78A8"},
                text=[f"<b>{r['name']}</b><br>{r['accuracy'] * 100:.1f}% · ${r['cost_avg']:.4f}"],
                textposition="top center",
                showlegend=False,
            )
        )
    fig.update_layout(
        title=(
            f"Frontera costo-calidad TEST — shot {MAX_SHOTS} (total) · λ = {router['lambda']:.4f} "
            f"({len(test_task_ids)} tareas, Monte Carlo {TEST_REPETITIONS} réplicas)"
        ),
        xaxis={"title": "Costo acumulado (USD) →", "tickformat": ".4f"},
        yaxis_title="Calidad (% resueltas) →",
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
            accuracy, cost = result[shots]
            rows.append({"lambda": float(lam), "shots": shots, "accuracy": accuracy, "cost_avg": cost})
    df = pd.DataFrame(rows)

    fig = go.Figure()
    for shots in range(1, MAX_SHOTS + 1):
        sub = df[df["shots"] == shots].sort_values("lambda")
        fig.add_trace(
            go.Scatter(
                x=sub["cost_avg"],
                y=sub["accuracy"] * 100,
                mode="lines+markers",
                name=f"Shot {shots}",
                marker={"size": 5},
                customdata=sub["lambda"],
                hovertemplate="λ=%{customdata:.3f}<br>$%{x:.4f} · %{y:.1f}%<extra></extra>",
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
                x=[picked["test_cost_avg"]],
                y=[picked["test_accuracy"] * 100],
                mode="markers+text",
                marker={"size": 14, "symbol": "star", "color": "black"},
                text=[f"λ elegido (shot {shots})" if shots == MAX_SHOTS else ""],
                textposition="bottom center",
                showlegend=False,
                hovertemplate=f"<b>λ elegido, shot {shots}</b><br>λ=%{{customdata:.3f}}<br>$%{{x:.4f}} · %{{y:.1f}}%<extra></extra>",
                customdata=[picked["lambda"]],
            )
        )

    fig.update_layout(
        title=(
            f"Trade-off del router en TEST: tareas resueltas vs costo, recorrido por λ<br>"
            f"<sub>{len(test_task_ids)} tareas · Monte Carlo {TEST_REPETITIONS} réplicas · "
            f"cada curva se recorre de λ={LAMBDA_GRID[0]:.0f} a λ={LAMBDA_GRID[-1]:.0f} (solo hasta {MAX_SHOTS} shots, "
            "que es lo calibrado para producción) · ★ = λ elegido para ese shot.<br>"
            "λ se elige con un sweep aparte sobre el split de TRAIN (no esta curva, que es solo TEST) — "
            "por eso la ★ no tiene por qué caer justo en el pico de esta curva.</sub>"
        ),
        xaxis={"title": "Costo acumulado (USD) →", "tickformat": ".4f"},
        yaxis_title="% de tareas resueltas →",
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
    rng: np.random.Generator,
) -> tuple[go.Figure, pd.DataFrame]:
    eligible = [t for t in test_task_ids if any((t, m) in y_pred_by_task_model for m in model_ids)]
    sample_size = min(HEATMAP_SAMPLE_SIZE, len(eligible))
    sample = sorted(rng.choice(eligible, size=sample_size, replace=False).tolist())

    names = sorted({model_name_by_id[m] for m in model_ids})
    name_code = {n: i for i, n in enumerate(names)}

    z, text = [], []
    for lam in REPORT_LAMBDAS:
        z_row, text_row = [], []
        for task_id in sample:
            ranked = _rank(task_id, y_pred_by_task_model, costs_by_model, model_ids, cost_max, lam)
            winner_name = model_name_by_id[ranked[0][1]] if ranked else None
            z_row.append(name_code[winner_name] if winner_name else None)
            text_row.append(winner_name or "")
        z.append(z_row)
        text.append(text_row)

    n = len(names)
    # discrete colorscale: one flat-colored band per model, two stops each
    # (plotly colorscales are continuous, so a "categorical" one is just a
    # step function built from repeated stops)
    colorscale = []
    for i, _name in enumerate(names):
        lo, hi = i / n, (i + 1) / n
        colorscale.append([lo, _PALETTE[i % len(_PALETTE)]])
        colorscale.append([hi, _PALETTE[i % len(_PALETTE)]])

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[f"task {t}" for t in sample],
            y=[f"λ={lam:.2f}" for lam in REPORT_LAMBDAS],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 9},
            colorscale=colorscale,
            zmin=0,
            zmax=n,
            colorbar={"tickvals": [i + 0.5 for i in range(n)], "ticktext": names},
            showscale=True,
        )
    )
    fig.update_layout(
        title=(
            f"A quién asigna el router el shot 1, por λ ({sample_size} tareas de test de muestra)<br>"
            "<sub>ranking determinístico (sin Monte Carlo) — muestra cómo el router abandona el modelo "
            "más capaz-pero-caro a medida que λ penaliza más el costo</sub>"
        ),
        xaxis={"tickangle": -45},
    )
    df_named = pd.DataFrame(text, index=[f"{lam:.2f}" for lam in REPORT_LAMBDAS], columns=[f"task_{t}" for t in sample])
    return fig, df_named


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
    lam3 = routing_lambdas[str(MAX_SHOTS)]["lambda"]

    charts = [
        (
            "sankey",
            lambda: _build_sankey(
                test_task_ids=test_task_ids,
                y_pred_by_task_model=y_pred_by_task_model,
                outcomes_by_task_model=outcomes_by_task_model,
                costs_by_model=costs_by_model,
                model_ids=model_ids,
                model_name_by_id=model_name_by_id,
                cost_max=cost_max,
                lam=lam3,
                rng=rng,
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
                rng=rng,
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
        fig.write_image(os.path.join(out_dir, f"{name}.png"), width=1400, height=900, scale=2)
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
