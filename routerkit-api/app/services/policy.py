"""Política de ruteo en producción: elige el candidato ganador dado el predictor.

Pendiente de definir (más adelante, junto con el usuario): λ (trade-off
calidad/costo), p_min (umbral mínimo de elegibilidad), exclusiones manuales
de candidatos, y la lógica de fallback si el primer intento falla.
"""

from dataclasses import dataclass


@dataclass
class RoutingDecision:
    winner_model_id: int
    ranking: list[dict]
    used_fallback: bool


def score_and_select(
    task_features: dict, candidates: list[dict]
) -> RoutingDecision | None:
    """Aplica la política de ruteo (score = p - λ*precio, p_min, exclusiones, fallback).

    Entrada: task_features (vector de features ya calculado para la tarea
    nueva), candidates (lista de {model_id, p, price} con la probabilidad de
    éxito predicha por el predictor y el precio normalizado de cada
    candidato).
    Salida esperada: RoutingDecision (modelo ganador, ranking completo de
    candidatos con sus scores, si se recurrió al fallback).
    Devuelve None mientras no esté implementado.
    """
    pass
