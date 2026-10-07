# Catalog guide — `data/examples/`

This folder has two kinds of files:

- **Small examples** (`tasks.jsonl`, `questions.json`, `models.json`, `features.jsonl`, `solutions.jsonl`, `validations.jsonl`): illustrate the format with a handful of rows. Useful to test the system and as a template.
- **Real experiment data** (`*_exp.json` / `*_exp.jsonl`, e.g. `tasks_exp.jsonl`, `questions_exp.json`, `models_exp.json`): the actual content used to run the real experiment — not an example, this is the project's production data.

This guide explains, field by field, what can go in each file — so you can build your own `<name>_exp.json`/`.jsonl` (the way `models_exp.json` was built) starting from the matching small example, without having to guess what values each field accepts.

## Downloading `solutions_exp.jsonl` / `validations_exp.jsonl`

These two are real experiment results, several hundred MB uncompressed — too heavy to live in git history, so they're **not** part of `git clone` (see [`.gitignore`](../../.gitignore)). They're published instead as compressed assets on the repo's [GitHub Releases](https://github.com/Timia-Group/routerkit/releases) page. Download and decompress them into this same folder before using them with `type=load`:

```bash
curl -L -o data/examples/solutions_exp.jsonl.gz \
  https://github.com/Timia-Group/routerkit/releases/latest/download/solutions_exp.jsonl.gz
curl -L -o data/examples/validations_exp.jsonl.gz \
  https://github.com/Timia-Group/routerkit/releases/latest/download/validations_exp.jsonl.gz

gunzip data/examples/solutions_exp.jsonl.gz data/examples/validations_exp.jsonl.gz
```

(`releases/latest/download/<file>` always resolves to that asset on the most recent release — no version tag to keep up to date here. If you specifically need an older release's version, replace `latest` with its tag, e.g. `releases/download/v1.0.0/solutions_exp.jsonl.gz`.)

Once decompressed, use them exactly like any other `type=load` file (see the `solutions.jsonl`/`validations.jsonl` sections below) — `type=load` expects plain `.jsonl`, not the `.gz`:

```bash
curl -X POST "localhost:8000/solutions/generate?type=load" -F file=@data/examples/solutions_exp.jsonl
curl -X POST "localhost:8000/solutions/validate?type=load" -F file=@data/examples/validations_exp.jsonl
```

## General rules

- **Full replace vs. targeted replace.** `tasks.jsonl`, `questions.json`, and `models.json` replace **the entire** active catalog of that table when uploaded (`POST /tasks/import`, `/questions/import`, `/models/import`) — anything not in the file stops being active. `features.jsonl`, `solutions.jsonl`, and `validations.jsonl` (the `type=load` of `/features/compute`, `/solutions/generate`, and `/solutions/validate`) replace **only the rows they bring** (identified by their natural key — see each section), the rest of the active catalog is left untouched.
- **`.jsonl` vs. `.json`.** `tasks`, `features`, `solutions`, and `validations` are `.jsonl` (one JSON row per line, no brackets or commas between lines) because they scale to thousands of records. `questions` and `models` are a single JSON object (`{"questions": [...]}` / `{"models": [...]}`) because they're small catalogs (dozens of rows). Don't mix the formats.
- **Unsupported values.** Any field not in the tables below is ignored if you send it anyway, and any field marked required that's missing causes the import to be rejected entirely with a 422 error (the message points to which row and which field).
- **Placeholders.** If a value (your own resource's URL, a deployment name) isn't ready yet or will differ per person/account, write it between `< >` (e.g. `"https://<YOUR_AZURE_FOUNDRY_RESOURCE>.openai.azure.com"`) — that makes it clear it needs to be filled in before use, and whoever uploads it knows exactly what to replace. Never put a real API key inside the JSON — keys always go in environment variables (`.env`), the JSON only references the *name* of that variable (`api_key_env`).

---

## `tasks.jsonl` — task catalog

Endpoint: `POST /tasks/import` (multipart, replaces ALL active tasks).

One JSON line per task:

| Field | Type | Required | Description |
|---|---|:-:|---|
| `prompt` | string | **yes** | The task statement. This is what's sent to the candidate LLM to generate a solution — it has to be self-contained (everything the candidate needs to know goes here: expected function name, input/output format, etc.). |
| `tests` | string \| null | no | Test/validation-criteria code. If the task is custom (no `bcb_task_id`), this is what the LLM judge sees. If it's a benchmark task, it isn't used (bigcodebench-api already has the official test). |
| `entry_point` | string \| null | no | Name of the function the solution is expected to define. Informational only — never told to the candidate LLM during generation. |
| `source` | string \| null | no | Free-form origin tag (e.g. `"custom"`, `"bigcodebench"`). Doesn't affect any behavior. |
| `tags` | array of string \| null | no | Free-form tags (e.g. `["math", "easy"]`, or the imports used by the reference solution). |
| `bcb_task_id` | string \| null | no | **Defines the task's type.** With a value (e.g. `"BigCodeBench/12"`) → official benchmark task, evaluated against `bigcodebench-api`. Without a value (`null` or omitted) → custom task, evaluated with the LLM judge. A single file can freely mix rows with and without `bcb_task_id`. |
| `custom_task_id` | string \| null | no | Stable key for **custom** tasks (analogous to `bcb_task_id`, but you choose it). Assign it if you're later going to reference this task from `features.jsonl`/`solutions.jsonl`/`validations.jsonl` — see the section for those three files below. Doesn't apply to benchmark tasks (those use `bcb_task_id`). |

```json
{"prompt": "Write a function that adds two integers.", "tests": "assert add(2, 3) == 5", "entry_point": "add", "source": "custom", "tags": ["math", "easy"], "custom_task_id": "add-two-integers"}
{"prompt": "Official instruct prompt of the task...", "tests": "import unittest\n...", "entry_point": "task_func", "source": "bigcodebench", "tags": ["random", "itertools"], "bcb_task_id": "BigCodeBench/0"}
```

---

## `questions.json` — feature question catalog

Endpoint: `POST /questions/import` (multipart, replaces the ENTIRE question catalog).

File shape: `{"questions": [ {...}, {...} ]}`.

| Field | Type | Required | Description |
|---|---|:-:|---|
| `text` | string | **yes** | The question as it's asked to the LLM (if it has no `script`) about each task's `prompt`. |
| `kind` | `"binary"` \| `"numeric"` | **yes** | `binary`: yes/no answer, stored as the probability of "yes" (0–1). `numeric`: a number (integer or decimal). No other value is valid. |
| `lo` / `hi` | number \| null | no | Expected range of the answer (informational for `numeric`; use `0`/`1` for `binary`). Not validated against the real answer. |
| `author` | string \| null | no | Who wrote it. Free-form. |
| `version` | string \| null | no | Free-form version/tag (e.g. `"v1"`). Doesn't affect behavior, it's for your own tracking. |
| `question_key` | string \| null | no | Stable key (e.g. `"p41"`, or simply `"q1"`, `"q2"`...) to reference this question later from `features.jsonl` (`type=load`). If you're not going to load precomputed features for this question, you can omit it — but if you will need it, assign it now: never use the internal Postgres `question_id` for that, it changes on every reload of this catalog. |
| `script` | string \| null | no | If omitted, the question is resolved by calling the LLM (`role=feature`). If defined, **it has to be the complete text of a Python function** with this exact signature: `def task_func(task_prompt: str) -> float:` — it's run directly (no LLM, $0 cost) with the task's `prompt` as its only argument. If `kind="binary"`, the script must return `0.0` or `1.0` (any other value is an error). |
| `repetitions` | integer | no (default `1`) | How many times to repeat the LLM call and average (simple mean) the resulting value — guards against the LLM's answer varying between calls, at the cost of `repetitions` times the tokens/price for that question. Ignored for a `script` question (deterministic, always computed once regardless of this value). |

```json
{
  "questions": [
    {"text": "Does the expected solution likely use recursion?", "kind": "binary", "lo": 0, "hi": 1, "author": "team", "version": "v1", "question_key": "q1"},
    {"text": "How difficult is the task, from 1 to 10?", "kind": "numeric", "lo": 1, "hi": 10, "version": "v1", "question_key": "q2", "repetitions": 5},
    {
      "text": "How many lines does the task's prompt have?",
      "kind": "numeric",
      "version": "v1",
      "question_key": "q3",
      "script": "def task_func(task_prompt: str) -> float:\n    return float(len(task_prompt.splitlines()))"
    }
  ]
}
```

### Available packages for `script`

A `script` runs inside `routerkit-api`'s own container (`exec()` in `app/services/features_python.py`) — **not** inside `bigcodebench-api`'s sandbox, which is a separate, much heavier image used only to grade generated solutions against the benchmark's official tests. Besides the standard library (`re`, `ast`, `gzip`, etc., always available), `routerkit-api`'s image ships:

- `nltk` (with the `punkt`/`punkt_tab` tokenizer data already downloaded at build time — `nltk.tokenize.sent_tokenize` works offline, no runtime download)
- `numpy`
- `tiktoken`

If your script needs something else, add it to [`routerkit-api/requirements.txt`](../../routerkit-api/requirements.txt) (and, if it needs offline data like nltk's, a download step in [`routerkit-api/Dockerfile`](../../routerkit-api/Dockerfile)) and rebuild the image (`docker compose build routerkit-api`) — a script that imports a package that isn't installed fails with `ModuleNotFoundError` at compute time, reported per-row in `errors`, it doesn't abort the rest of the batch.

---

## `models.json` — model catalog (candidates, feature, judge)

Endpoint: `POST /models/import` (multipart, replaces the ENTIRE model catalog).

File shape: `{"models": [ {...}, {...} ]}`.

### Fields

| Field | Type | Required | Description |
|---|---|:-:|---|
| `name` | string | **yes** | Name used to identify the model across the whole system (reports, `/solutions/generate` type=load looks up candidates by this name). Free-form, but unique per `role` (see below). |
| `type` | `"api"` \| `"hosted"` | **yes** | `api`: billed per token, to an external provider (OpenAI, Anthropic, Azure, etc.) — use `in_token_price`/`out_token_price`. `hosted`: your own/rented infrastructure billed by compute time — use `cost_per_second`. |
| `provider` | string \| null | no | See the recognized providers table below. If it's none of the recognized ones, it's treated as your own OpenAI-compatible endpoint (`endpoint` = base URL). |
| `endpoint` | string \| null | no | Base URL of the API. Required in practice for Azure (any variant) and for custom/hosted providers; not needed for native `openai`/`anthropic`/`openrouter` (they already have a fixed URL). |
| `deployment` | string \| null | no | The actual model identifier to send to the API (if it differs from `name` — e.g. `name="Claude Sonnet 5"` but `deployment="claude-sonnet-5"`). If omitted, `name` is used as-is. |
| `api_key_env` | string \| null | no | Name of the environment variable (in `.env`) holding this model's API key. If omitted, it's resolved by a fixed convention based on `provider` — see `.env.example` (`OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `CUSTOM_LLM_API_KEY`). Use this whenever two models share a `provider` but use different accounts. |
| `in_token_price` / `out_token_price` | number \| null | no | USD per **1 million** input/output tokens. Only for `type="api"`. |
| `cost_per_second` | number \| null | no | USD per second of compute. Only for `type="hosted"`. If your provider gives you a cost per hour, divide by 3600. |
| `concurrency` | integer \| null | no | Max simultaneous calls to this model. `null` = no limit defined (treated as 1, sequential). |
| `timeout_seconds` | integer | no (default `300`) | Timeout for each HTTP call to this model. |
| `temperature` | number \| null | no | Fixed temperature for this model. `null` = provider default. |
| `api_kind` | `"chat_completions"` \| `"responses"` | no (default `"chat_completions"`) | See the dedicated section below. |
| `max_tokens` | integer \| null | no | Max output tokens to request from the model. Some APIs **require** it (see Anthropic Messages below). |
| `role` | `"candidate"` \| `"feature"` \| `"judge"` | **yes** | `candidate`: model to route/compare — you can have several active at once, but never two with the same `name`. `feature`: computes features via LLM — **at most one can be active in the whole system**, and it needs to support logprobs (see below). `judge`: grades solutions for custom tasks — **at most one can be active in the whole system**, doesn't need logprobs. |

### Recognized `provider` values

| `provider` | When to use it | Notes |
|---|---|---|
| `openai` | OpenAI directly | — |
| `anthropic` | Anthropic directly | Doesn't support logprobs — can't be used for `role="feature"`. |
| `openrouter` | OpenRouter | — |
| `azure` | Azure OpenAI deployment (includes deployments of other models served via Azure AI Foundry through the OpenAI-compatible endpoint, e.g. Grok, GPT-5.x) | `endpoint` = base URL of the Azure resource. |
| `azure_anthropic` | Claude served from **Azure AI Foundry** (not the same as plain `anthropic` — uses a different route/auth) | `endpoint` must point to the resource's `.../services.ai.azure.com/anthropic` sub-path. |
| *any other value* | Your own gateway / self-hosted OpenAI-compatible (e.g. Spheron, vLLM, etc.) | `endpoint` = the gateway's URL. If that gateway carries **no** authentication at all, let whoever maintains the code know (the provider name needs to be added to an explicit list in `llm_client.py` so it never gets sent an API key). |

### `api_kind`: which method is used to call the model

- **`chat_completions`** (default): the "classic" method (Chat Completions). Works for almost everything.
- **`responses`**: OpenAI's Responses API — use it **only** if that specific deployment doesn't respond to Chat Completions (this isn't a style preference, it's a real limitation of the deployment — happens with some newer "reasoning" models served via Azure). Verified against a real deployment that needed it (also what resolved that model's `top_logprobs` cap — Chat Completions limits it to 5, Responses doesn't).

### Logprobs (matters for `role="feature"`)

The model with `role="feature"` needs the provider to expose logprobs in the response. **Anthropic (directly or via `azure_anthropic`) never exposes them** — it can't be used for this role. OpenAI (directly or via Azure, Chat Completions or Responses) does support them.

### Examples (one per real combination we've already used)

```json
{
  "models": [
    { "name": "gpt-4o-mini", "type": "api", "provider": "openai", "in_token_price": 0.15, "out_token_price": 0.60, "role": "candidate" },

    { "name": "GPT-5.6 Sol", "type": "api", "provider": "azure", "deployment": "gpt-5.6-sol",
      "endpoint": "https://<your-resource>.openai.azure.com", "api_key_env": "AZURE_FOUNDRY_API_KEY",
      "api_kind": "responses", "in_token_price": 5.00, "out_token_price": 30.00, "role": "candidate" },

    { "name": "Claude Sonnet 5", "type": "api", "provider": "azure_anthropic", "deployment": "claude-sonnet-5",
      "endpoint": "https://<your-resource>.services.ai.azure.com/anthropic", "api_key_env": "AZURE_FOUNDRY_API_KEY",
      "max_tokens": 4096, "role": "candidate" },

    { "name": "Qwen3.6-35B-A3B-FP8", "type": "hosted", "provider": "spheron", "deployment": "qwen3.6-fp8",
      "endpoint": "https://<spheron-gateway>", "cost_per_second": 0.00061389, "role": "candidate" },

    { "name": "gpt-5.4-nano", "type": "api", "provider": "azure", "deployment": "gpt-5.4-nano",
      "endpoint": "https://<your-resource>.openai.azure.com", "api_key_env": "AZURE_FOUNDRY_API_KEY",
      "role": "feature" }
  ]
}
```

See `models_exp.json` for the complete real set used in the experiment.

---

## `features.jsonl` / `solutions.jsonl` / `validations.jsonl` — targeted loads (`type=load`)

These three **don't replace the whole catalog** — each line identifies one targeted row by its natural key, and only that row gets replaced. They're uploaded as a file (multipart) to their endpoint with `?type=load` in the URL:

```bash
curl -X POST "localhost:8000/features/compute?type=load"   -F file=@my_file.jsonl
curl -X POST "localhost:8000/solutions/generate?type=load" -F file=@my_file.jsonl
curl -X POST "localhost:8000/solutions/validate?type=load" -F file=@my_file.jsonl
```

### How to identify the task in these three files

Every row in `features.jsonl`/`solutions.jsonl`/`validations.jsonl` needs to say **which task it refers to**, and there are **two ways to do that, mutually exclusive** (exactly one of the two, never both or neither). **The internal Postgres `task_id` is never used here** — it isn't stable across reloads (it changes based on insertion order), and mixing it in with these two just creates confusion:

| Field | When to use it |
|---|---|
| `bcb_task_id` | For **official benchmark** tasks (e.g. `"BigCodeBench/837"`). It's the key the benchmark itself assigns. |
| `custom_task_id` | For **custom** tasks. It's the key you yourself assigned to that task when creating it in `tasks.jsonl` (`custom_task_id` field — see that section). If your custom task doesn't have one assigned, you won't be able to reference it from these files: go back to `tasks.jsonl` and add it. |

⚠️ Common mistake: using the number from `"BigCodeBench/N"` as if it were a Postgres id, or making up a number for a custom task. Neither exists as a field here — always `bcb_task_id` or `custom_task_id`, never a plain integer.

Same criterion for **questions** in `features.jsonl`: they're identified by `question_key` (the one you assigned in `questions.json`, see that section) — never by the internal Postgres `question_id`, which also changes on every reload of `questions.json`.

### `features.jsonl`

Identifies each row by `(bcb_task_id|custom_task_id, question_key)`.

| Field | Type | Required | Description |
|---|---|:-:|---|
| `bcb_task_id` | string | one of the two | See table above. |
| `custom_task_id` | string | one of the two | See table above. |
| `question_key` | string | **yes** | The question's stable key (the one you assigned in `questions.json`). |
| `source` | `"llm"` \| `"python"` | **yes** | How it was computed externally. |
| `value` | number | **yes** | The feature's value (0–1 if it came from a `binary` question). |
| `answer_text` | string \| null | no | The LLM's literal answer, if applicable. |
| `answer_logprobs` | string \| null | no | Raw serialized logprobs, if you have them. |
| `input_tokens` / `output_tokens` | integer \| null | no | Token usage, if `source="llm"`. |
| `latency_ms` | number \| null | no | Telemetry. |
| `cost_usd` | number \| null | no | Telemetry. |

```json
{"custom_task_id": "add-two-integers", "question_key": "q15", "source": "llm", "value": 0.87, "answer_text": "YES", "input_tokens": 120, "output_tokens": 1, "latency_ms": 430.5, "cost_usd": 0.000018}
{"bcb_task_id": "BigCodeBench/0", "question_key": "q60", "source": "python", "value": 54.0, "latency_ms": 0.2, "cost_usd": 0}
```

### `solutions.jsonl`

Identifies each row by `(model_name, bcb_task_id|custom_task_id, sample_idx)` — `model_name` has to exactly match the `name` of an active model with `role="candidate"`.

| Field | Type | Required | Description |
|---|---|:-:|---|
| `model_name` | string | **yes** | The candidate's name (must exist and be active). |
| `bcb_task_id` | string | one of the two | See table above. |
| `custom_task_id` | string | one of the two | See table above. |
| `sample_idx` | integer | no (default `0`) | Sample number, if you generated several solutions for the same (task, model). |
| `code` | string \| null | no | The generated code, exactly as it came from the LLM (with markdown fences if it included them, unprocessed). |
| `tokens_in` / `tokens_out` | integer \| null | no | Token usage. |
| `latency_ms` / `cost_usd` | number \| null | no | Telemetry. |
| `gen_error` | string \| null | no | Error message, if generation failed. |
| `has_error` | boolean | no (default `false`) | Explicit flag that this row had an error (independent of `gen_error`). |

```json
{"model_name": "gpt-4o-mini", "custom_task_id": "add-two-integers", "sample_idx": 0, "code": "```python\ndef add(a, b):\n    return a + b\n```", "tokens_in": 85, "tokens_out": 22, "latency_ms": 610.4, "cost_usd": 0.0000271}
{"model_name": "gpt-4o", "bcb_task_id": "BigCodeBench/0", "sample_idx": 0, "gen_error": "rate limit exceeded", "has_error": true}
```

### `validations.jsonl`

Identifies each row by `(model_name, bcb_task_id|custom_task_id, sample_idx)` — there has to be an active `solution` for that combination.

| Field | Type | Required | Description |
|---|---|:-:|---|
| `model_name` | string | **yes** | Name of the candidate that owns the solution being graded. |
| `bcb_task_id` | string | one of the two | See table above. |
| `custom_task_id` | string | one of the two | See table above. |
| `sample_idx` | integer | no (default `0`) | Same as in solutions. |
| `grader` | `"tests"` \| `"llm_judge"` | **yes** | Who produced this verdict externally. |
| `passed` | boolean \| null | no | Binary result. |
| `fail_reason` | string \| null | no | Reason if `passed=false` (or error message if `has_error=true`). |
| `exec_ms` | number \| null | no | Telemetry. |
| `has_error` | boolean | no (default `false`) | The validation itself failed (infra) — different from `passed=false`. |

```json
{"model_name": "gpt-4o-mini", "custom_task_id": "add-two-integers", "sample_idx": 0, "grader": "llm_judge", "passed": true, "exec_ms": 120.3}
{"model_name": "gpt-4o", "bcb_task_id": "BigCodeBench/0", "sample_idx": 0, "grader": "tests", "passed": false, "fail_reason": "The solution does not handle the empty list case."}
```

---

## Checklist for building your `<name>_exp.json`/`.jsonl`

1. Start from the matching small example (`tasks.jsonl` → `tasks_exp.jsonl`, etc.) and keep exactly the same field names.
2. Check that section's table: fill in the required fields, leave the ones that don't apply as `null`/omitted.
3. Respect the allowed values for "enum" fields (`kind`, `type`, `role`, `api_kind`, `grader`, `source`) — any other value rejects the whole import with a 422.
4. If the file is `tasks`/`features`/`solutions`/`validations`: one row per line, no arrays or commas between lines (`.jsonl`). If it's `questions`/`models`: a single object with the `"questions"`/`"models"` key containing the array.
5. In `features`/`solutions`/`validations`: identify the task with `bcb_task_id` (benchmark) or `custom_task_id` (custom) — **never a bare number or the internal Postgres `task_id`**, neither is a valid field here (see the section above). If a custom task doesn't have a `custom_task_id` yet, assign it in `tasks.jsonl` before you can reference it.
6. In `features`: identify the question with `question_key` (the one you assigned in `questions.json`) — never the internal `question_id`. If a question doesn't have a `question_key` yet, assign it in `questions.json` before you can reference it.
7. Never a real API key in the JSON — always `api_key_env` pointing to a variable in `.env`.
8. If something doesn't quite fit these tables (a case not covered), ask before forcing it in — better to confirm than to guess.
