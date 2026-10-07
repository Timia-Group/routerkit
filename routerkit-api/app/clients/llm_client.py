"""Generic LLM client on top of litellm.

Used by services.features_llm, services.generation and services.judge to
call whichever model is relevant in each case (the model with
role='feature', a candidate, or the model with role='judge'). The api key
is resolved as follows: if `model.api_key_env` is defined, that
environment variable is used (this allows having several accounts for the
same provider and choosing which one each model uses); otherwise, it falls
back to a fixed convention per `model.provider` (see `.env.example`) —
except for the providers in `_NO_AUTH_PROVIDERS` (in-house gateways with no
Authorization header), which never carry an api_key. The call is built
from `model.provider` + `model.deployment` + `model.endpoint`.

`model.api_kind` decides the actual call mechanism: `chat_completions`
(default, `litellm.completion()`) or `responses` (`litellm.responses()`,
OpenAI's newer API — needed by models that don't fully expose Chat
Completions). Both paths return the same normalized `LlmResponse`, logprobs
included (see `_extract_responses_logprobs`), so the rest of the system
doesn't need to know which API the response came from.

We don't use litellm's cost estimator: `cost_usd()` computes it separately
— from `model.in_token_price`/`out_token_price` and the call's token usage
for `type="api"` models, or from `model.cost_per_second`/`model.concurrency`
and the call's latency for `type="hosted"` models (see its docstring).

`model.timeout_seconds` is passed straight through to litellm on every
call. `model.temperature` is model config (not a per-request parameter):
it's used whenever it's defined, unless the caller passes an explicit
`temperature` to `complete()` to override it for that call.
`model.concurrency` (maximum simultaneous calls to this model) is applied
by the callers (routers/features.py, routers/solutions.py) via
services.concurrency.run_bounded, not by this function — here only a
SINGLE call is made.

`normalize_yes_no()` + `first_token_logprob()` are the shared technique for
yes/no questions with a calibrated probability (`exp(logprob)`, never the
raw logprob): used by both services.features_llm (kind='binary' questions)
and services.judge (the LLM judge's verdict), so the extraction isn't
duplicated.
"""

import os
import re
import unicodedata
from dataclasses import dataclass

import litellm

from app.db.models import Model

# Accepted variants of a yes/no response, already normalized (uppercase,
# no accents): English and Spanish. "SI"/"SÍ" count as YES.
_YES_WORDS = {"YES", "SI"}
_NO_WORDS = {"NO"}

# litellm model prefix per provider (see "Provider Route" in litellm's
# docs). If `model.provider` isn't in this map, a custom OpenAI-compatible
# endpoint is assumed (self-hosted, vLLM, etc.), using `model.endpoint` as
# api_base and `model.deployment`/`model.name` as-is.
#
# "azure_anthropic": Claude served from Azure AI Foundry (AnthropicFoundry
# SDK / Messages API), NOT the same as a regular Azure OpenAI deployment
# ("azure") — litellm routes it with its own "azure_ai/" prefix, pointing
# `model.endpoint` at the resource's ".../services.ai.azure.com/anthropic"
# sub-path.
_LITELLM_PREFIXES = {
    "openai": "openai",
    "azure": "azure",
    "azure_anthropic": "azure_ai",
    "anthropic": "anthropic",
    "openrouter": "openrouter",
}

# Providers that explicitly carry no authentication (in-house gateway with
# no Authorization header) — they are never sent an api_key, regardless of
# api_key_env or CUSTOM_LLM_API_KEY.
_NO_AUTH_PROVIDERS = {"spheron"}

# Fixed environment variable per provider (a single account per provider).
# Any provider outside this map (custom/self-hosted) uses
# CUSTOM_LLM_API_KEY if defined, or no key at all if the endpoint doesn't
# require auth.
_PROVIDER_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
_CUSTOM_API_KEY_ENV = "CUSTOM_LLM_API_KEY"

# Azure needs api_version in addition to endpoint/deployment; there's no
# dedicated column in `models` for this (all Azure deployments share the
# same API version), so it's taken from a single global environment
# variable.
_AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")


class LlmClientError(RuntimeError):
    """The required api key is missing, or litellm returned a call error."""


@dataclass
class LlmResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    top_logprobs: dict | None
    raw: dict


def _resolve_api_key(model: Model) -> str | None:
    if (model.provider or "").lower() in _NO_AUTH_PROVIDERS:
        return None

    if model.api_key_env:
        api_key = os.environ.get(model.api_key_env)
        if not api_key:
            raise LlmClientError(
                f"model {model.model_id} ({model.name}): missing environment variable "
                f"'{model.api_key_env}' (api_key_env)"
            )
        return api_key

    provider = (model.provider or "").lower()
    env_var = _PROVIDER_API_KEY_ENV.get(provider)

    if env_var is None:
        # Custom/self-hosted provider: it may not require auth at all.
        return os.environ.get(_CUSTOM_API_KEY_ENV) or None

    api_key = os.environ.get(env_var)
    if not api_key:
        raise LlmClientError(
            f"model {model.model_id} ({model.name}): missing environment variable "
            f"'{env_var}' (required for provider='{model.provider}')"
        )
    return api_key


def _is_azure_v1(model: Model) -> bool:
    """True for Azure AI Foundry's newer "v1" API (api_base ending in
    ".../openai/v1"). That surface is wire-compatible with the plain OpenAI
    API (deployment name goes in the request body, not the URL path, and
    there's no api-version query param) — unlike the classic Azure OpenAI
    API (".../openai/deployments/{id}/...?api-version=..."), which needs
    litellm's "azure/" prefix to build that different URL shape."""
    return (model.provider or "").lower() == "azure" and "/openai/v1" in (model.endpoint or "")


def _litellm_model_name(model: Model) -> str:
    provider = (model.provider or "").lower()
    # Azure's v1 API mimics OpenAI's own wire format exactly, so it's routed
    # the same way as a custom OpenAI-compatible endpoint (litellm's
    # "openai/" prefix + api_base override) — NOT through "azure/", which
    # would build the classic (and incompatible) deployments-path URL.
    prefix = "openai" if _is_azure_v1(model) else _LITELLM_PREFIXES.get(provider)
    identifier = model.deployment or model.name
    return f"{prefix}/{identifier}" if prefix else identifier


def normalize_yes_no(text: str) -> str | None:
    """Normalizes a response to "YES"/"NO".

    Tolerant of case, trailing punctuation ("Yes.", "no!"), accents, and
    the Spanish variants ("sí"/"no") in addition to English ("yes"/"no").
    Returns None if it doesn't match any known variant.
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
    """Info for the first logprobs token (token/logprob/top_logprobs).

    Raises LlmClientError if the response doesn't include logprobs
    (unsupported provider, or they weren't requested with
    `complete(..., logprobs=True)`).
    """
    content = (top_logprobs or {}).get("content") or []
    if not content:
        raise LlmClientError("the LLM response doesn't include logprobs (empty content)")
    return content[0]


def cost_usd(
    model: Model, prompt_tokens: int, completion_tokens: int, latency_ms: float | None = None
) -> float:
    """Cost in USD of a call.

    `type="api"`: based on `model.in_token_price`/`out_token_price` (USD /
    1M tokens) and the call's token usage.

    `type="hosted"` (e.g. Spheron): based on `model.cost_per_second` and
    `latency_ms` instead — there's no per-token price, our own
    infrastructure is billed by compute time. Divided by `model.concurrency`
    (treated as 1 if undefined): we run up to that many generations in
    parallel on the same paid compute (see services.concurrency.run_bounded),
    so a single call's fair share of the bill is latency / concurrency, not
    the full wall-clock latency — that would massively over-count cost
    whenever several calls overlap. Requires `latency_ms`; without it (or
    without `cost_per_second` configured), returns 0.0.
    """
    if model.type == "hosted":
        if model.cost_per_second is None or latency_ms is None:
            return 0.0
        concurrency = model.concurrency or 1
        return (latency_ms / 1000) * float(model.cost_per_second) / concurrency

    in_price = float(model.in_token_price) if model.in_token_price is not None else 0.0
    out_price = float(model.out_token_price) if model.out_token_price is not None else 0.0
    return (prompt_tokens / 1_000_000) * in_price + (completion_tokens / 1_000_000) * out_price


def _common_kwargs(model: Model) -> dict:
    kwargs: dict = {"model": _litellm_model_name(model), "timeout": model.timeout_seconds}
    api_key = _resolve_api_key(model)
    if api_key:
        kwargs["api_key"] = api_key
    if model.endpoint:
        kwargs["api_base"] = model.endpoint
    # Azure AI Foundry's newer "v1" API doesn't use api_version as a query
    # param — that's only for the classic API
    # (".../openai/deployments/{id}/...?api-version=..."). Sending it there
    # anyway can break the call (see _is_azure_v1).
    if (model.provider or "").lower() == "azure" and not _is_azure_v1(model):
        kwargs["api_version"] = _AZURE_API_VERSION
    return kwargs


def complete(
    model: Model,
    messages: list[dict],
    *,
    temperature: float | None = None,
    logprobs: bool = False,
    top_logprobs: int | None = None,
) -> LlmResponse:
    """Calls `model` with `messages` and normalizes the response.

    `messages` follows OpenAI's chat format:
    `[{"role": "system"|"user"|"assistant", "content": "..."}]`.
    `logprobs`/`top_logprobs` only have an effect on providers that support
    them (e.g. OpenAI); not all do (e.g. Anthropic). `temperature` is a
    one-off override — if omitted, `model.temperature` is used.
    Raises LlmClientError if the api key is missing or if litellm's call fails.

    `model.api_kind` decides the actual mechanism: `chat_completions`
    (default, `litellm.completion()`) or `responses` (`litellm.responses()`,
    for models that don't fully expose the Chat Completions API).
    """
    effective_temperature = temperature if temperature is not None else model.temperature
    if model.api_kind == "responses":
        return _complete_responses(model, messages, effective_temperature, logprobs, top_logprobs)
    return _complete_chat_completions(model, messages, effective_temperature, logprobs, top_logprobs)


def _complete_chat_completions(
    model: Model,
    messages: list[dict],
    temperature: float | None,
    logprobs: bool,
    top_logprobs: int | None,
) -> LlmResponse:
    kwargs = _common_kwargs(model)
    kwargs["messages"] = messages
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if model.max_tokens is not None:
        kwargs["max_tokens"] = model.max_tokens
    if logprobs:
        kwargs["logprobs"] = True
        if top_logprobs is not None:
            kwargs["top_logprobs"] = top_logprobs

    try:
        response = litellm.completion(**kwargs)
    except Exception as exc:  # noqa: BLE001 - normalized into LlmClientError for the caller
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


def _complete_responses(
    model: Model,
    messages: list[dict],
    temperature: float | None,
    logprobs: bool,
    top_logprobs: int | None,
) -> LlmResponse:
    """Same as `_complete_chat_completions` but via `litellm.responses()`
    (OpenAI's Responses API) — needed by models with
    `model.api_kind == "responses"` (see the `models` schema).

    NOT VERIFIED against the real API: there are no credentials or sandbox
    in this development environment to test it. The shape of the response
    (and in particular the structure of `logprobs`) was built based on the
    public OpenAI/litellm documentation — before relying on this for a real
    run of the experiment, it's worth validating it with an actual call and
    adjusting `_extract_responses_logprobs` if needed.
    """
    kwargs = _common_kwargs(model)
    kwargs["input"] = messages
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if model.max_tokens is not None:
        kwargs["max_output_tokens"] = model.max_tokens
    if logprobs:
        kwargs["top_logprobs"] = top_logprobs if top_logprobs is not None else 1
        kwargs["include"] = ["message.output_text.logprobs"]

    try:
        response = litellm.responses(**kwargs)
    except Exception as exc:  # noqa: BLE001 - normalized into LlmClientError for the caller
        raise LlmClientError(f"model {model.model_id} ({model.name}): {exc}") from exc

    text = getattr(response, "output_text", None)
    if not text:
        text = ""
        for item in getattr(response, "output", None) or []:
            for content in getattr(item, "content", None) or []:
                if getattr(content, "text", None):
                    text = content.text
                    break

    logprobs_payload = _extract_responses_logprobs(response) if logprobs else None
    usage = getattr(response, "usage", None)

    return LlmResponse(
        text=text or "",
        prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
        completion_tokens=getattr(usage, "output_tokens", 0) or 0,
        top_logprobs=logprobs_payload,
        raw=response.model_dump() if hasattr(response, "model_dump") else dict(response),
    )


def _extract_responses_logprobs(response) -> dict | None:
    """Converts the Responses API's logprobs to the same shape we already
    use for Chat Completions (`{"content": [{"token", "logprob",
    "top_logprobs": [...]}]}`), so that `first_token_logprob()` and
    `normalize_yes_no()` don't need to know which API the response came
    from."""
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            raw_logprobs = getattr(content, "logprobs", None)
            if not raw_logprobs:
                continue
            content_tokens = [
                {
                    "token": getattr(entry, "token", None),
                    "logprob": getattr(entry, "logprob", None),
                    "top_logprobs": [
                        {"token": c.token, "logprob": c.logprob}
                        for c in (getattr(entry, "top_logprobs", None) or [])
                    ],
                }
                for entry in raw_logprobs
            ]
            if content_tokens:
                return {"content": content_tokens}
    return None
