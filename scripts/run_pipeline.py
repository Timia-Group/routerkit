#!/usr/bin/env python3
"""Runs the full routerkit pipeline (catalogs -> features -> solutions ->
validations -> train) against an already-running routerkit-api, one step
at a time, waiting for each async job to finish and checking it for errors
before moving on. Stdlib only — no pip install needed.

scripts/wizard.py imports this module directly and reaches into a few of
its underscore-prefixed names (_STEP_DEFAULT_FILE, _PLACEHOLDER_RE,
_resolve_step_file) rather than duplicating them — grep wizard.py for a
name here before renaming or removing it.

Surfaces every problem instead of silently tolerating it the way the API
itself does: the step's own HTTP call failing or the job itself raising an
exception always stops the run immediately. A batch that finished with
per-row errors (e.g. a few solutions failed to generate) instead PAUSES
and asks whether to continue — those rows are already excluded from
training/routing-calibration on their own (service-side failures aren't a
quality signal), so continuing is normally safe; say no (or run with
--yes to always continue non-interactively) if you'd rather stop and swap
in a different file. Either way, resume later without repeating earlier
steps with --start-from pointed at wherever you left off.

Default behaviour (no arguments): loads the real experiment's catalogs/
data with `type=load` (NOT `type=calculate`) and trains a predictor —
zero LLM calls, zero cost. 'tasks' is skipped by default (the official
BigCodeBench catalog is already auto-seeded by routerkit-api on first
startup) unless you explicitly pass --tasks.

Examples:
  # Full default run (safe, free, ~15 min including training):
  python scripts/run_pipeline.py

  # Resume after fixing whatever made 'solutions' fail last time:
  python scripts/run_pipeline.py --start-from=solutions

  # Use your own models.json (with real endpoints), default data for the rest:
  python scripts/run_pipeline.py --models=data/examples/models.json

  # Actually call LLMs to generate solutions instead of loading precomputed ones:
  python scripts/run_pipeline.py --start-from=solutions --solutions=calculate --validations=calculate

  # Just retrain: questions/models/features/solutions/validations are
  # already active from a previous run, no catalog/data step needs to repeat:
  python scripts/run_pipeline.py --start-from=train

  # Small/custom catalog: compute live, but stop before training (too few
  # rows to train anything meaningful on) — swaps in your own tasks/
  # questions/models and real-calls every step up to validations only:
  python scripts/run_pipeline.py --tasks=data/examples/tasks.jsonl \\
      --questions=data/examples/questions.json --models=data/examples/models_test_small.json \\
      --features=calculate --solutions=calculate --validations=calculate --until=validations
"""

import argparse
import json
import mimetypes
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "data" / "examples"

STEP_ORDER = ["tasks", "questions", "models", "features", "solutions", "validations", "train"]

# (endpoint path, default data file or None, job "kind" for polling or None if synchronous)
_STEP_DEFAULT_FILE = {
    "tasks": None,  # opt-in only, see module docstring
    "questions": "questions_exp.json",
    "models": "models_exp.json",
    "features": "features_exp.jsonl",
    "solutions": "solutions_exp.jsonl",
    "validations": "validations_exp.jsonl",
}
_STEP_IMPORT_PATH = {"tasks": "/tasks/import", "questions": "/questions/import", "models": "/models/import"}
_STEP_ASYNC_PATH = {
    "features": "/features/compute",
    "solutions": "/solutions/generate",
    "validations": "/solutions/validate",
}
_STEP_JOB_KIND = {
    "features": "features_compute",
    "solutions": "solutions_generate",
    "validations": "solutions_validate",
}

_PLACEHOLDER_RE = re.compile(r"<[A-Z][A-Z0-9_]*>")

# Job messages carry their progress as "N/M ..." (see progress() in
# routerkit-api's services/jobs.py) — that's all GET /jobs exposes of it.
_FRACTION_RE = re.compile(r"(\d+)/(\d+)")
_BAR_WIDTH = 30
# Redraw one line in place only on a real terminal; piped/redirected output
# (CI, a log file) keeps the one-line-per-change behaviour instead.
_LIVE_PROGRESS = sys.stdout.isatty()
_progress_line_open = False


class PipelineError(RuntimeError):
    """Raised to stop the whole run — caught once, at the top level."""


def _end_progress_line() -> None:
    global _progress_line_open
    if _progress_line_open:
        print(flush=True)
        _progress_line_open = False


def _log(msg: str) -> None:
    _end_progress_line()
    print(f"[pipeline] {msg}", flush=True)


def _fmt_elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def _show_progress(message: str, elapsed: float) -> None:
    """Live mode: redraws a single line — a bar when the message carries an
    "N/M" fraction, plus elapsed time so a long phase with no countable
    progress (training's grid search) still visibly ticks."""
    global _progress_line_open
    if not _LIVE_PROGRESS:
        _log(f"  ... {message}")
        return

    bar = ""
    match = _FRACTION_RE.search(message)
    if match and int(match.group(2)) > 0:
        ratio = min(int(match.group(1)) / int(match.group(2)), 1.0)
        filled = round(_BAR_WIDTH * ratio)
        bar = f"[{'#' * filled}{'-' * (_BAR_WIDTH - filled)}] {ratio * 100:5.1f}%  "

    width = shutil.get_terminal_size().columns - 1
    line = f"[pipeline]   {bar}{message}  ({_fmt_elapsed(elapsed)})"
    print("\r" + line[:width].ljust(width), end="", flush=True)
    _progress_line_open = True


def _http_json(method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed local API
            return json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise PipelineError(f"HTTP {exc.code} calling {method} {url}: {detail}") from exc


def _http_multipart(url: str, file_path: Path) -> dict:
    boundary = uuid.uuid4().hex
    content_type, _ = mimetypes.guess_type(str(file_path))
    content_type = content_type or "application/octet-stream"
    file_bytes = file_path.read_bytes()

    parts = [
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'.encode(),
        f"Content-Type: {content_type}\r\n\r\n".encode(),
        file_bytes,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    body = b"".join(parts)

    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310 - fixed local API
            return json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise PipelineError(f"HTTP {exc.code} calling POST {url}: {detail}") from exc


def _resolve_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _resolve_step_file(step: str, override: str | None) -> Path:
    """A user-supplied --<step> value is a path relative to the repo root
    (or absolute) — they might point anywhere. The step's own default
    (a bare filename, e.g. "questions_exp.json") always means that file
    inside data/examples/ specifically, never the repo root."""
    if override:
        return _resolve_path(override)
    return EXAMPLES_DIR / _STEP_DEFAULT_FILE[step]


def _confirm(prompt: str, assume_yes: bool) -> bool:
    """Interactive y/N gate, bypassed by --yes. EOFError (no stdin attached
    — e.g. run non-interactively without --yes) counts as "no": fails
    closed rather than silently assuming consent."""
    if assume_yes:
        _log("--yes passed, continuing anyway.")
        return True
    try:
        answer = input(f"[pipeline] {prompt} [y/N]: ").strip().lower()
    except EOFError:
        answer = ""
    return answer in ("y", "yes")


def _check_models_placeholders(file_path: Path, assume_yes: bool) -> None:
    text = file_path.read_text(encoding="utf-8")
    found = sorted(set(_PLACEHOLDER_RE.findall(text)))
    if not found:
        return

    print()
    print(f"[pipeline] WARNING: {file_path} still has placeholder value(s): {', '.join(found)}")
    print("[pipeline] This is fine for everything up to 'train' (no LLM is actually called), but")
    print("[pipeline] POST /route and any '=calculate' step WILL fail until you replace them with")
    print("[pipeline] real endpoints/credentials in that file (see data/examples/README.md).")
    if not _confirm("Continue anyway?", assume_yes):
        raise PipelineError("Aborted by user at the 'models' confirmation prompt.")
    print()


def _run_sync_import(base_url: str, step: str, file_path: Path) -> None:
    _log(f"{step}: importing {file_path.relative_to(REPO_ROOT)} ...")
    result = _http_multipart(f"{base_url}{_STEP_IMPORT_PATH[step]}", file_path)
    _log(f"{step}: done — deactivated_previous={result.get('deactivated_previous')}, "
         f"created={result.get('created')}")


def _poll_job(base_url: str, kind: str) -> dict:
    """Polls GET /jobs every 2s. On a terminal, progress is one line redrawn
    in place; otherwise a new line is printed only when the job's message
    actually changes (not on every poll) — we don't want to print 100
    near-identical lines for one job."""
    last_printed = None
    started = time.monotonic()
    while True:
        jobs = _http_json("GET", f"{base_url}/jobs")
        entry = next((j for j in jobs if j["kind"] == kind), None)
        if entry is None:
            raise PipelineError(f"GET /jobs never returned a '{kind}' entry — unexpected API response.")

        if entry["status"] == "running":
            if _LIVE_PROGRESS or entry["message"] != last_printed:
                _show_progress(entry["message"], time.monotonic() - started)
                last_printed = entry["message"]
            time.sleep(2)
            continue
        _end_progress_line()

        # status == "idle": job finished (successfully or not — see
        # to_board_entry in services/jobs.py: both show as "idle", the only
        # way to tell them apart is last_result being None on failure).
        if entry["last_result"] is None:
            raise PipelineError(f"job '{kind}' failed: {entry['message']}")
        return entry["last_result"]


def _check_batch_errors(step: str, result: dict, assume_yes: bool) -> None:
    """Surfaces every error (never silently tolerated), but doesn't assume
    they're necessarily fatal: service-side failures (has_error=true —
    a dropped connection, a content-filter block, a truncated response)
    are already excluded from training/routing-calibration on their own
    (see services/aggregation.py, services/routing_calibration.py) — they
    don't need the file cleaned up by hand first. Lets you decide: keep
    going with this file as-is, or stop and swap it for a different one."""
    errors = result.get("errors") or []
    if not errors:
        return
    _log(f"{step}: {len(errors)} error(s) reported:")
    for err in errors[:20]:
        _log(f"    {err}")
    if len(errors) > 20:
        _log(f"    ... and {len(errors) - 20} more (see GET /jobs for the full list)")
    _log(f"{step}: these rows are already loaded (has_error=true) — training/routing-calibration "
         "already excludes them on their own, nothing to clean up by hand first.")
    if not _confirm(f"Continue past '{step}' anyway?", assume_yes):
        raise PipelineError(f"{step} finished with {len(errors)} error(s) — stopped by user.")


def _run_async_step(base_url: str, step: str, *, file_path: Path | None, calculate: bool, assume_yes: bool) -> None:
    kind = _STEP_JOB_KIND[step]
    path = _STEP_ASYNC_PATH[step]

    if calculate:
        _log(f"{step}: starting type=calculate (this calls real LLMs — real cost/time) ...")
        _http_json("POST", f"{base_url}{path}?type=calculate")
    else:
        _log(f"{step}: starting type=load from {file_path.relative_to(REPO_ROOT)} ...")
        _http_multipart(f"{base_url}{path}?type=load", file_path)

    result = _poll_job(base_url, kind)
    _check_batch_errors(step, result, assume_yes)
    summary = {k: v for k, v in result.items() if k != "errors"}
    _log(f"{step}: done — {summary}")


def _run_train(base_url: str) -> None:
    _log("train: starting (grid search + lambda calibration + report — can take several "
         "minutes on real data) ...")
    _http_json("POST", f"{base_url}/train")
    result = _poll_job(base_url, "train_predictor")
    _log(f"train: done — predictor_id={result.get('predictor_id')}, test_r2={result.get('test_r2')}, "
         f"report_dir={result.get('report_dir')}")
    _log(f"train: routing_lambdas={result.get('routing_lambdas')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default="http://localhost:8000", help="routerkit-api base URL.")
    parser.add_argument(
        "--start-from", choices=STEP_ORDER, default="tasks",
        help="Skip every step before this one (e.g. after fixing an error and wanting to "
        "resume without repeating earlier steps). Default: tasks (the whole sequence).",
    )
    parser.add_argument(
        "--until", choices=STEP_ORDER, default="train",
        help="Stop after this step instead of running the whole sequence — e.g. "
        "--until=validations to skip training entirely (too little data to train on with a "
        "small/custom catalog, or you just want to inspect the loaded/computed rows first). "
        "Default: train (the whole sequence).",
    )
    parser.add_argument("--tasks", metavar="FILE", help="Import this tasks.jsonl. Omit to skip the step entirely.")
    parser.add_argument("--questions", metavar="FILE", help=f"Default: {_STEP_DEFAULT_FILE['questions']}")
    parser.add_argument("--models", metavar="FILE", help=f"Default: {_STEP_DEFAULT_FILE['models']}")
    parser.add_argument(
        "--features", metavar="FILE|calculate",
        help=f"Default: {_STEP_DEFAULT_FILE['features']} (type=load). Pass 'calculate' to call the "
        "feature model live instead (real cost).",
    )
    parser.add_argument(
        "--solutions", metavar="FILE|calculate",
        help=f"Default: {_STEP_DEFAULT_FILE['solutions']} (type=load). Pass 'calculate' to generate "
        "live with the candidate models instead (real cost).",
    )
    parser.add_argument(
        "--validations", metavar="FILE|calculate",
        help=f"Default: {_STEP_DEFAULT_FILE['validations']} (type=load). Pass 'calculate' to grade "
        "live (bigcodebench-api / LLM judge) instead (real cost).",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Don't pause for any confirmation prompt (models-placeholder warning, or a batch "
        "finishing with per-row errors) — always continue. For non-interactive use.",
    )
    args = parser.parse_args()

    start_idx = STEP_ORDER.index(args.start_from)
    until_idx = STEP_ORDER.index(args.until)
    if until_idx < start_idx:
        parser.error(f"--until={args.until} comes before --start-from={args.start_from}")
    overrides = {
        "tasks": args.tasks,
        "questions": args.questions,
        "models": args.models,
        "features": args.features,
        "solutions": args.solutions,
        "validations": args.validations,
    }

    steps = STEP_ORDER[start_idx : until_idx + 1]
    run_started = time.monotonic()
    try:
        for n, step in enumerate(steps, start=1):
            print(flush=True)
            _log(f"=== step {n}/{len(steps)}: {step} ===")
            step_started = time.monotonic()
            _run_step(args, step, overrides[step] if step != "train" else None)
            _log(f"{step}: finished in {_fmt_elapsed(time.monotonic() - step_started)}")

    except PipelineError as exc:
        _log(f"STOPPED: {exc}")
        _log(f"Fix the issue above, then resume with: --start-from={step}")
        sys.exit(1)

    print(flush=True)
    total = _fmt_elapsed(time.monotonic() - run_started)
    if args.until == "train":
        _log(f"Pipeline complete in {total} — predictor trained and ready for POST /route.")
    else:
        _log(f"Pipeline complete in {total} — stopped after '{args.until}' as requested (--until), "
             "no training run.")


def _run_step(args: argparse.Namespace, step: str, override: str | None) -> None:
    if step == "train":
        _run_train(args.base_url)
        return

    if step in _STEP_IMPORT_PATH:
        if override is None and _STEP_DEFAULT_FILE[step] is None:
            _log(f"{step}: skipped (no --{step} given).")
            return
        file_path = _resolve_step_file(step, override)
        if not file_path.exists():
            raise PipelineError(f"{step}: file not found: {file_path}")
        if step == "models":
            _check_models_placeholders(file_path, assume_yes=args.yes)
        _run_sync_import(args.base_url, step, file_path)
        return

    # features / solutions / validations
    if override == "calculate":
        _run_async_step(args.base_url, step, file_path=None, calculate=True, assume_yes=args.yes)
    else:
        file_path = _resolve_step_file(step, override)
        if not file_path.exists():
            raise PipelineError(
                f"{step}: file not found: {file_path} — if this is one of the heavy "
                f"*_exp.jsonl datasets, run scripts/start.py first to download it."
            )
        _run_async_step(args.base_url, step, file_path=file_path, calculate=False, assume_yes=args.yes)


if __name__ == "__main__":
    main()
