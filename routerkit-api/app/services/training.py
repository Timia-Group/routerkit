"""Entrenamiento del predictor (AutoML) con las features/targets activos.

Pendiente de definir: librería de AutoML a usar (a decidir más adelante).
El resultado final (.joblib u otro formato serializado) debe guardarse bajo
settings.artifacts_dir para que quede en el volumen persistente.
"""

from dataclasses import dataclass


@dataclass
class TrainingResult:
    algorithm: str
    cv_score: float
    test_r2: float | None
    test_mae: float | None
    feature_columns: dict
    artifact_path: str


def train_predictor() -> TrainingResult | None:
    """Entrena un predictor con routerkit.features + routerkit.targets activos.

    Entrada: ninguna — junta las features/targets ACTIVOS en este momento;
    arma el dataset largo -1 fila por (task, model candidato)-, con el
    modelo candidato en one-hot como predictora, y split agrupado por task.
    Salida esperada: TrainingResult con el algoritmo ganador, sus métricas
    (cv_score/test_r2/test_mae), el contrato de columnas esperado
    (feature_columns) y la ruta del artefacto serializado bajo
    settings.artifacts_dir.
    Devuelve None mientras no esté implementado.
    """
    pass
