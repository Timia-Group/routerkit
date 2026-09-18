"""Cliente LLM genérico sobre litellm.

Usado por services.features_llm, services.generation (ya implementados) y
services.judge (todavía stub) para llamar al modelo que corresponda en cada
caso (el model con role='feature', un candidate, o el model con
role='judge'). La api key se resuelve así: si `model.api_key_env`
está definido, se usa esa variable de entorno (permite tener varias cuentas
del mismo provider y elegir cuál usa cada modelo); si no, se cae a una
convención fija por `model.provider` (ver `.env.example`). La llamada se
arma con `model.provider` + `model.deployment` + `model.endpoint` (siempre
Chat Completions; no soportamos la Responses API de OpenAI).

No usamos el estimador de costo de litellm: `cost_usd()` calcula aparte,
con `model.in_token_price`/`out_token_price` y el `usage` (tokens) que
litellm sí devuelve normalizado, igual para cualquier proveedor (no cubre
`cost_per_second` de modelos hosted, eso lo calcula el caller aparte).

`model.timeout_seconds` se pasa directo a litellm en cada llamada.
`model.temperature` es config del modelo (no un parámetro por request): se
usa siempre que esté definido, salvo que el caller pase `temperature`
explícito a `complete()` para pisarla puntualmente.
`model.concurrency` (máximo de llamadas simultáneas a este modelo) lo
aplican los callers (routers/features.py, routers/solutions.py) vía
services.concurrency.run_bounded, no esta función — acá solo se hace UNA
llamada.

`normalize_yes_no()` + `first_token_logprob()` son la técnica compartida
para preguntas de sí/no con probabilidad calibrada (`exp(logprob)`, nunca
el logprob crudo): las usan tanto services.features_llm (preguntas
kind='binary') como services.judge (veredicto del juez LLM), para no
duplicar la extracción.
"""

import os
import re
import unicodedata
from dataclasses import dataclass

import litellm

from app.db.models import Model

# Variantes aceptadas de una respuesta de sí/no, ya normalizadas (mayúsculas,
# sin acentos): inglés y español. "SI"/"SÍ" cuentan como YES.
_YES_WORDS = {"YES", "SI"}
_NO_WORDS = {"NO"}

# Prefijo de modelo litellm por provider (ver "Provider Route" en los docs
# de litellm). Si `model.provider` no está en este mapa, se asume un
# endpoint OpenAI-compatible custom (self-hosted, vLLM, etc.) usando
# `model.endpoint` como api_base y `model.deployment`/`model.name` tal cual.
_LITELLM_PREFIXES = {
    "openai": "openai",
    "azure": "azure",
    "anthropic": "anthropic",
    "openrouter": "openrouter",
}

# Variable de entorno fija por provider (una sola cuenta por proveedor).
# Cualquier provider fuera de este mapa (custom/self-hosted) usa
# CUSTOM_LLM_API_KEY si está definida, o ninguna key si el endpoint no
# requiere auth.
_PROVIDER_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
_CUSTOM_API_KEY_ENV = "CUSTOM_LLM_API_KEY"

# Azure necesita api_version además de endpoint/deployment; no hay columna
# propia en `models` para esto (todas las deployments de Azure comparten
# versión de API), así que se toma de una única variable de entorno global.
_AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")


class LlmClientError(RuntimeError):
    """Falta la api key requerida, o litellm devolvió un error de la llamada."""


@dataclass
class LlmResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    top_logprobs: dict | None
    raw: dict


def _resolve_api_key(model: Model) -> str | None:
    if model.api_key_env:
        api_key = os.environ.get(model.api_key_env)
        if not api_key:
            raise LlmClientError(
                f"model {model.model_id} ({model.name}): falta la variable de entorno "
                f"'{model.api_key_env}' (api_key_env)"
            )
        return api_key

    provider = (model.provider or "").lower()
    env_var = _PROVIDER_API_KEY_ENV.get(provider)

    if env_var is None:
        # Provider custom/self-hosted: puede no requerir auth en absoluto.
        return os.environ.get(_CUSTOM_API_KEY_ENV) or None

    api_key = os.environ.get(env_var)
    if not api_key:
        raise LlmClientError(
            f"model {model.model_id} ({model.name}): falta la variable de entorno "
            f"'{env_var}' (requerida para provider='{model.provider}')"
        )
    return api_key


def _litellm_model_name(model: Model) -> str:
    prefix = _LITELLM_PREFIXES.get((model.provider or "").lower())
    identifier = model.deployment or model.name
    return f"{prefix}/{identifier}" if prefix else identifier


def normalize_yes_no(text: str) -> str | None:
    """Normaliza una respuesta a "YES"/"NO".

    Tolera mayúsculas/minúsculas, puntuación al final ("Yes.", "no!"),
    acentos y las variantes en español ("sí"/"no") además de inglés
    ("yes"/"no"). Devuelve None si no matchea ninguna variante conocida.
    """
    match = re.search(r"[A-Za-zÀ-ÿ]+", text)
    if not match:
        return None
    word = unicodedata.normalize("NFKD", match.group(0))
    word = "".join(c for c in word if not unicodedata.combining(c)).upper()
    if word in _YES_WORDS:
        return "YES"
    if word in _NO_WORDS:
        return "NO"
    return None


def first_token_logprob(top_logprobs: dict | None) -> dict:
    """Info del primer token de logprobs (token/logprob/top_logprobs).

    Lanza LlmClientError si la respuesta no trae logprobs (proveedor sin
    soporte, o no se pidieron con `complete(..., logprobs=True)`).
    """
    content = (top_logprobs or {}).get("content") or []
    if not content:
        raise LlmClientError("la respuesta del LLM no trae logprobs (content vacío)")
    return content[0]


def cost_usd(model: Model, prompt_tokens: int, completion_tokens: int) -> float:
    """Costo en USD de una llamada, según `model.in_token_price`/`out_token_price`
    (USD / 1M tokens). No cubre `cost_per_second` (modelos hosted) — eso no
    depende del `usage` de la llamada, lo calcula el caller aparte."""
    in_price = float(model.in_token_price) if model.in_token_price is not None else 0.0
    out_price = float(model.out_token_price) if model.out_token_price is not None else 0.0
    return (prompt_tokens / 1_000_000) * in_price + (completion_tokens / 1_000_000) * out_price


def complete(
    model: Model,
    messages: list[dict],
    *,
    temperature: float | None = None,
    logprobs: bool = False,
    top_logprobs: int | None = None,
) -> LlmResponse:
    """Llama a `model` (chat completions) con `messages` y normaliza la respuesta.

    `messages` sigue el formato de chat de OpenAI:
    `[{"role": "system"|"user"|"assistant", "content": "..."}]`.
    `logprobs`/`top_logprobs` solo tienen efecto en proveedores que los
    soportan (ej. OpenAI); no todos lo hacen (ej. Anthropic). `temperature`
    es un override puntual — si se omite, se usa `model.temperature`.
    Lanza LlmClientError si falta la api key o si litellm falla la llamada.
    """
    kwargs: dict = {
        "model": _litellm_model_name(model),
        "messages": messages,
        "timeout": model.timeout_seconds,
    }

    api_key = _resolve_api_key(model)
    if api_key:
        kwargs["api_key"] = api_key
    if model.endpoint:
        kwargs["api_base"] = model.endpoint
    if (model.provider or "").lower() == "azure":
        kwargs["api_version"] = _AZURE_API_VERSION
    effective_temperature = temperature if temperature is not None else model.temperature
    if effective_temperature is not None:
        kwargs["temperature"] = float(effective_temperature)
    if logprobs:
        kwargs["logprobs"] = True
        if top_logprobs is not None:
            kwargs["top_logprobs"] = top_logprobs

    try:
        response = litellm.completion(**kwargs)
    except Exception as exc:  # noqa: BLE001 - se homogeneiza como LlmClientError para el caller
        raise LlmClientError(f"model {model.model_id} ({model.name}): {exc}") from exc

    choice = response.choices[0]
    logprobs_payload = None
    if logprobs and getattr(choice, "logprobs", None):
        logprobs_payload = choice.logprobs.model_dump()

    return LlmResponse(
        text=choice.message.content or "",
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        top_logprobs=logprobs_payload,
        raw=response.model_dump(),
    )
