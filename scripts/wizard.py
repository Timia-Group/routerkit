#!/usr/bin/env python3
"""Interactive, step-by-step installer and guide for RouterKit.

Asks every decision up front, then checks that this machine has what
those answers need (Python, Docker, disk space, datasets, API keys and
endpoints — offering to fill in what's missing), shows the exact commands
that fit, and runs them one step at a time with their progress. Stdlib
only — no pip install needed.

Usage:
    python scripts/wizard.py

It never talks to the API on its own: every step is a plain command (the
other scripts — start.py, run_pipeline.py, test_route.py — or docker),
printed before it runs, so next time you can run the same thing yourself
without the wizard.

Imports run_pipeline/start as modules (not just subprocess targets) to
reuse their own constants/helpers instead of re-deriving them here. Of
run_pipeline's, only STEP_ORDER is actually public; _STEP_DEFAULT_FILE,
_PLACEHOLDER_RE and _resolve_step_file are its own internals, reused
anyway rather than duplicated. If you rename or restructure any of those
four in run_pipeline.py, grep this file for the name first — nothing
else enforces that contract.
"""

import sys

if sys.version_info < (3, 10):
    sys.exit(f"RouterKit's scripts need Python 3.10 or newer (this is {sys.version.split()[0]}).")

import getpass
import json
import re
import shutil
import subprocess
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import run_pipeline
import start

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "data" / "examples"
ENV_FILE = REPO_ROOT / ".env"
REPORT_INDEX = REPO_ROOT / "data" / "current" / "report" / "index.html"
# What's REALLY active in the database right now — rewritten by the API on
# every catalog import (app/services/snapshots.py), independent of
# whichever file a human remembers loading, or whether a .local.json copy
# made afterwards was ever actually imported. Used to check readiness for
# 'route' (nothing gets imported for that goal, so there's no other way to
# know what the database currently holds — see _check_live_calls).
ACTIVE_MODELS_SNAPSHOT = REPO_ROOT / "data" / "current" / "models.json"
BCB_DOCKERFILE = REPO_ROOT / "bigcodebench-api" / "Dockerfile"
DEFAULT_MODELS = EXAMPLES_DIR / run_pipeline._STEP_DEFAULT_FILE["models"]
# Imported explicitly rather than relying on routerkit-api's startup seed:
# the seed only runs on an empty database, so after anyone loads their own
# tasks, "the official catalog" would otherwise silently not be what's active.
OFFICIAL_TASKS_FILE = EXAMPLES_DIR / "tasks_exp.jsonl"
OFFICIAL_TASKS_ARG = f"--tasks={OFFICIAL_TASKS_FILE.relative_to(REPO_ROOT).as_posix()}"

# Free disk space to ask for before the first build: the bigcodebench base
# image alone is ~9.5GB unpacked, plus its compressed layers while pulling,
# the routerkit-api image, Postgres and the loaded experiment data. Running
# out mid-pull leaves Docker Desktop's engine unresponsive until restarted.
_MIN_FREE_GB_FIRST_INSTALL = 20
_MIN_FREE_GB = 5

# Mirrors routerkit-api's app/clients/llm_client.py: which .env variable a
# model's key comes from when its row has no explicit api_key_env, and the
# providers that are never sent a key at all.
_PROVIDER_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
_NO_AUTH_PROVIDERS = {"spheron"}

_GOAL_TITLES = {
    "reproduce": "Load the experiment data and train the predictor",
    "custom": "Run the pipeline",
    "retrain": "Train the predictor",
}


class WizardQuit(Exception):
    """The user chose to stop — caught once, at the top level."""


@dataclass
class Command:
    """One step of the plan. `program` is shown as-is; a leading "python" is
    swapped for this interpreter when it actually runs."""

    title: str
    program: list[str]

    def argv(self) -> list[str]:
        if self.program[0] == "python":
            return [sys.executable, *self.program[1:]]
        return self.program

    def display(self) -> str:
        return " ".join(f'"{p}"' if " " in p else p for p in self.program)

    @property
    def is_pipeline(self) -> bool:
        return "scripts/run_pipeline.py" in self.program

    @property
    def is_start(self) -> bool:
        return "scripts/start.py" in self.program


@dataclass
class Plan:
    """Everything decided in the questions phase, before anything runs."""

    goal: str
    pipeline_args: list[str] | None = None  # None = no run_pipeline.py step
    models_file: Path | None = None
    live_roles: frozenset[str] = frozenset()  # model roles that will really be called
    needs_datasets: bool = False
    want_routing: bool = False

    @property
    def needs_existing_install(self) -> bool:
        """Retrain/route work on what a previous run already loaded/trained."""
        return self.goal in ("retrain", "route")


# Commands not finished yet — read by the top-level handler, so stopping
# (a menu's Quit, or Ctrl+C) can say exactly what's left to run by hand.
_remaining: list[Command] = []
_started = False


# --------------------------------------------------------------------------
# Terminal UI
# --------------------------------------------------------------------------

def _phase(n: int, title: str) -> None:
    print()
    print("#" * 72)
    print(f"#  Phase {n}/4: {title}")
    print("#" * 72)


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _say(msg: str = "") -> None:
    print(f"  {msg}" if msg else "", flush=True)


def _ok(msg: str) -> None:
    _say(f"[ OK ] {msg}")


def _todo(msg: str) -> None:
    _say(f"[ -- ] {msg}")


def _fail(msg: str) -> None:
    _say(f"[ !! ] {msg}")


def _detail(msg: str) -> None:
    _say(f"       {msg}")


def _input(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError as exc:
        raise WizardQuit("no input available") from exc


def _ask_choice(question: str, options: list[tuple[str, str]], default: int = 1) -> int:
    """Numbered menu; returns the 0-based index of the chosen option."""
    print()
    print(f"? {question}")
    for i, (label, detail) in enumerate(options, start=1):
        print(f"  {i}) {label}{'  (default)' if i == default else ''}")
        if detail:
            print(f"       {detail}")
    while True:
        raw = _input(f"  Choose 1-{len(options)} [{default}]: ")
        if not raw:
            return default - 1
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        _say("Please type one of the numbers above.")


def _ask_yes_no(question: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        raw = _input(f"? {question} [{hint}]: ").lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        _say("Please answer y or n.")


def _ask_file(question: str) -> str:
    """A path that exists, as typed (relative to the repo root, or absolute) —
    run_pipeline.py resolves it the same way."""
    while True:
        raw = _input(f"? {question}\n  Path (relative to the routerkit folder, or absolute): ").strip('"')
        if raw and _resolve(raw).is_file():
            return raw
        _say(f"File not found: {raw or '(empty)'} - try again.")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _retry_or_quit(reason: str, retry_label: str = "Check again") -> None:
    if _ask_choice("What now?", [(retry_label, ""), ("Quit", "")]) == 1:
        raise WizardQuit(reason)


def _continue_or_quit(reason: str, continue_detail: str) -> None:
    if _ask_choice("What now?", [("Continue anyway", continue_detail), ("Quit", "")], default=2) == 1:
        raise WizardQuit(reason)


# --------------------------------------------------------------------------
# Phase 1: questions
# --------------------------------------------------------------------------

def _warn_resets_active_experiment() -> None:
    """Both 'Reproduce' and 'Custom run' start from the 'tasks' step, and
    tasks/questions/models are always a full replace (see catalogs.py) —
    never incremental, regardless of how small the actual change is. That
    cascades (app/services/cascade.py) to deactivate whatever
    features/solutions/grades/targets were already active, and the
    currently active predictor along with them. Silent only because
    ACTIVE_MODELS_SNAPSHOT existing is the one cheap, already-available
    signal that a previous run left something active to lose — a brand
    new install has nothing to warn about."""
    if not ACTIVE_MODELS_SNAPSHOT.exists():
        return
    print()
    _say("NOTE: this reloads tasks/questions/models from scratch, which resets whatever")
    _say("was already active here (including any trained predictor) - even if the change")
    _say("you want feels small (e.g. just recomputing one step). To train again on data")
    _say("that's already loaded without resetting it, quit and choose 'Retrain only' instead.")


def _ask_unattended(calls_llms: bool) -> list[str]:
    """Only a run that calls LLMs gets the question: there, many failed rows
    usually mean a misconfigured model burning money, worth pausing for.
    Otherwise a few failed rows are expected (they come from the loaded data)
    and already skipped from training, so the run just keeps going (--yes)."""
    if not calls_llms:
        return ["--yes"]
    print()
    _say("Some rows of a step that calls LLMs can fail (a call errors or times out).")
    _say("They're skipped from training automatically, but many failures usually mean")
    _say("a misconfigured model that is costing money for nothing.")
    choice = _ask_choice(
        "When a step reports failed rows, should the run pause and ask you?",
        [
            ("Pause and ask me", "you'll need to watch the screen and type 'y' to continue"),
            ("Keep going on its own", "the failures are still shown, you don't have to answer anything"),
        ],
    )
    return ["--yes"] if choice == 1 else []


def _ask_want_routing(feature_called: bool, other_llms_called: bool) -> bool:
    """`feature_called`: the feature model's endpoint/key are already being
    asked for (live features). `other_llms_called`: some other model is."""
    print()
    _say("OPTIONAL EXTRA - not needed for the run itself.")
    if feature_called:
        _say("After training, you can also test the router with your own prompts. It uses")
        _say("the same 'feature' model as the live features step (small cost per prompt),")
        _say("so nothing extra will be asked for.")
    else:
        _say("After training, you can also test the router with your own prompts. That")
        _say("calls an LLM (the 'feature' model, small cost per prompt), so it needs its API")
        _say("key and endpoint - set now, before the models are loaded (adding them after")
        _say("training would mean training again).")
        _say("Answer 'n' to skip it: " + ("no extra API key will be asked for." if other_llms_called
                                          else "no API key will be asked for."))
    return _ask_yes_no("Do you want this optional prompt testing too?", default=False)


def _questions_reproduce() -> Plan:
    _section("Reproduce the published experiment")
    _say("Loads the experiment's precomputed features, solutions and grades and trains a")
    _say("predictor on them. No LLM calls, no cost, no API keys. About 15 minutes, plus a")
    _say("one-off download of ~10 GB of Docker images on the first install.")
    _warn_resets_active_experiment()
    want_routing = _ask_want_routing(feature_called=False, other_llms_called=False)
    return Plan(
        goal="reproduce",
        pipeline_args=[OFFICIAL_TASKS_ARG, *_ask_unattended(calls_llms=False)],
        models_file=DEFAULT_MODELS,
        live_roles=frozenset({"feature"} if want_routing else ()),
        needs_datasets=True,
        want_routing=want_routing,
    )


def _ask_catalog(step: str, label: str) -> str | None:
    default = run_pipeline._STEP_DEFAULT_FILE[step]
    choice = _ask_choice(
        f"Which {label} should be used?",
        [(f"The experiment's ({default})", ""), ("My own file", "")],
    )
    return None if choice == 0 else _ask_file(f"Your {label} file:")


def _ask_data_step(label: str, *, allow_experiment: bool, default: str | None, live_detail: str) -> str | None:
    """Returns run_pipeline's --<step> value: None (the experiment's file),
    a file path, or 'calculate'. `default` is one of those three kinds
    ('file' standing for "my own file")."""
    options, values = [], []
    if allow_experiment:
        options.append((f"Load the experiment's precomputed {label}", "free, no LLM calls"))
        values.append(None)
    options.append((f"Load my own {label} file", "free, no LLM calls"))
    values.append("file")
    options.append((f"Compute {label} live", live_detail))
    values.append("calculate")

    default_index = values.index(default) + 1 if default in values else len(values)
    value = values[_ask_choice(f"How should {label} be obtained?", options, default=default_index)]
    if value == "file":
        return _ask_file(f"Your {label} file (.jsonl):")
    return value


def _source_kind(value: str | None) -> str | None:
    """None / 'calculate' as-is; any file path collapses to 'file'."""
    return value if value in (None, "calculate") else "file"


def _questions_custom() -> Plan:
    _section("Custom run")
    _say("One question per pipeline step. The experiment's files are the default;")
    _say("anything you change replaces just that part.")
    _warn_resets_active_experiment()

    args: list[str] = []
    tasks_count = sum(1 for _ in OFFICIAL_TASKS_FILE.open(encoding="utf-8"))
    custom_tasks = _ask_choice(
        "Which tasks should the router learn from?",
        [
            (f"Official BigCodeBench catalog ({tasks_count} tasks)", "graded by their official tests"),
            ("My own tasks file", "replaces the whole task catalog"),
        ],
    ) == 1
    args.append(f"--tasks={_ask_file('Your tasks file (.jsonl):')}" if custom_tasks else OFFICIAL_TASKS_ARG)

    custom_catalogs = custom_tasks
    models_value = None
    for step, label in (("questions", "feature questions"), ("models", "models catalog")):
        value = _ask_catalog(step, label)
        if value:
            custom_catalogs = True
            args.append(f"--{step}={value}")
            if step == "models":
                models_value = value

    if custom_catalogs:
        print()
        _say("Note: with your own catalogs, the experiment's precomputed results only match")
        _say("the rows that also exist in yours - computing live is usually what you want.")

    data_default = "calculate" if custom_catalogs else None
    features = _ask_data_step(
        "features", allow_experiment=True, default=data_default,
        live_detail="the feature LLM answers every question that has no script: real cost and time",
    )
    solutions = _ask_data_step(
        "solutions", allow_experiment=True, default=data_default,
        live_detail="every candidate LLM solves every task: real cost and time",
    )
    # Grades belong to specific solutions: the experiment's only fit the
    # experiment's solutions, and they should come from the same kind of source.
    validations = _ask_data_step(
        "validations (grades)", allow_experiment=solutions is None, default=_source_kind(solutions),
        live_detail="official tasks run their tests in a sandbox (free, slow)"
        + ("; your own tasks are graded by the judge LLM (real cost)" if custom_tasks else ""),
    )
    choices = {"features": features, "solutions": solutions, "validations": validations}
    args += [f"--{step}={value}" for step, value in choices.items() if value]

    live_roles = set()
    if features == "calculate":
        live_roles.add("feature")
    if solutions == "calculate":
        live_roles.add("candidate")
    if validations == "calculate" and custom_tasks:
        live_roles.add("judge")  # official tasks never reach the judge

    train = _ask_yes_no("Train the predictor at the end?", default=True)
    if not train:
        args.append("--until=validations")

    want_routing = train and _ask_want_routing(
        feature_called="feature" in live_roles, other_llms_called=bool(live_roles - {"feature"})
    )
    calls_llms = bool(live_roles)
    if want_routing:
        live_roles.add("feature")

    return Plan(
        goal="custom",
        pipeline_args=args + _ask_unattended(calls_llms=calls_llms),
        models_file=run_pipeline._resolve_step_file("models", models_value),
        live_roles=frozenset(live_roles),
        # Only the experiment's solutions/validations are the heavy downloads.
        needs_datasets=solutions is None or validations is None,
        want_routing=want_routing,
    )


def _questions_retrain() -> Plan:
    _section("Retrain on the data already loaded")
    _say("Trains again on whatever questions/models/features/solutions/grades a previous")
    _say("run loaded (they stay in the database between restarts). No LLM calls.")
    return Plan(goal="retrain", pipeline_args=["--start-from=train"])


def _questions_route() -> Plan:
    _section("Try the router on prompts")
    _say("Uses the predictor trained by a previous run. Each prompt calls the 'feature'")
    _say("model (small cost), with the endpoint stored when the models catalog was loaded,")
    _say("so that catalog must have had its real endpoint, and its API key must be in .env.")
    filled = _filled_copy_path(DEFAULT_MODELS)
    options = [(f"The experiment's ({DEFAULT_MODELS.name})", "")]
    if filled.exists():
        options.insert(0, (f"The experiment's, with my values filled in ({filled.name})", ""))
    options.append(("My own file", ""))
    choice = _ask_choice("Which models catalog did that previous run load?", options)
    if choice == len(options) - 1:
        models_file = _resolve(_ask_file("Your models catalog file:"))
    else:
        models_file = filled if filled.exists() and choice == 0 else DEFAULT_MODELS
    return Plan(goal="route", models_file=models_file, live_roles=frozenset({"feature"}), want_routing=True)


def _ask_questions() -> Plan:
    _say("A few questions first - nothing runs until you've seen and confirmed the plan.")
    goal = _ask_choice(
        "What do you want to do?",
        [
            ("Reproduce the published experiment",
             "free, no API keys, ~15 minutes plus a one-off ~10 GB download (recommended first)"),
            ("Custom run", "your own tasks/questions/models, and/or call real LLMs"),
            ("Retrain only", "train again on the data a previous run loaded"),
            ("Try the router on prompts", "needs a previous run's trained predictor and the feature model's API key"),
        ],
    )
    return [_questions_reproduce, _questions_custom, _questions_retrain, _questions_route][goal]()


# --------------------------------------------------------------------------
# Phase 2: requirements check
# --------------------------------------------------------------------------

def _run_quiet(argv: list[str]) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(argv, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None


def _check_docker() -> None:
    while True:
        result = _run_quiet(["docker", "version", "--format", "{{.Server.Version}}"])
        if result is None:
            _fail("Docker is not installed. Install Docker Desktop: https://docs.docker.com/get-docker/")
        elif result.returncode != 0 or not result.stdout.strip():
            _fail("Docker is installed but its engine isn't responding. Start Docker Desktop and wait")
            _detail("until it says 'Engine running'. If it already does, restart it (on Windows,")
            _detail("'wsl --shutdown' in PowerShell first, then reopen Docker Desktop).")
        elif (compose := _run_quiet(["docker", "compose", "version"])) is None or compose.returncode != 0:
            _fail("Docker Compose v2 ('docker compose') is missing - update Docker Desktop.")
        else:
            _ok(f"Docker engine {result.stdout.strip()} is running, with Compose v2.")
            return
        _retry_or_quit("Docker not available")


def _base_image() -> str:
    match = re.search(r"^FROM\s+(\S+)", BCB_DOCKERFILE.read_text(encoding="utf-8"), re.MULTILINE)
    return match.group(1)


def _image_present(image: str) -> bool:
    result = _run_quiet(["docker", "image", "inspect", image])
    return result is not None and result.returncode == 0


def _check_disk(first_install: bool) -> None:
    """Docker Desktop keeps its images under the user's home drive by default;
    the datasets land in the repo. Both need room."""
    needed = _MIN_FREE_GB_FIRST_INSTALL if first_install else _MIN_FREE_GB
    while True:
        drives = {Path(p.anchor or "/") for p in (Path.home(), REPO_ROOT)}
        free = {d: shutil.disk_usage(d).free / 1e9 for d in drives}
        low = {d: gb for d, gb in free.items() if gb < needed}
        if not low:
            _ok("Free disk space: " + ", ".join(f"{d} {gb:.0f} GB" for d, gb in free.items())
                + f" (needs ~{needed} GB).")
            return
        for drive, gb in low.items():
            _fail(f"Only {gb:.1f} GB free on {drive} - about {needed} GB are needed"
                  + (" to download the Docker images." if first_install else "."))
        _detail("Free up space, or move Docker's data to another drive in Docker Desktop:")
        _detail("Settings > Resources > Advanced > Disk image location.")
        choice = _ask_choice(
            "What now?",
            [("Check again", ""), ("Continue anyway", "the install may fail and stall Docker"), ("Quit", "")],
        )
        if choice == 1:
            return
        if choice == 2:
            raise WizardQuit("not enough disk space")


def _release_available(name: str) -> bool:
    req = urllib.request.Request(f"{start.RELEASE_BASE}/{name}.gz", method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 - fixed GitHub Releases URL
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def _check_datasets() -> None:
    while True:
        missing = [n for n in start.HEAVY_DATASETS if not (EXAMPLES_DIR / n).exists()]
        if not missing:
            _ok("Experiment datasets are in place.")
            return
        not_downloadable = [n for n in missing if not _release_available(n)]
        if not not_downloadable:
            _todo(f"Experiment datasets will be downloaded: {', '.join(missing)} (a few hundred MB).")
            return
        _fail(f"Missing and not downloadable right now: {', '.join(not_downloadable)}")
        _detail("(the repo or its release may still be private). If you have the files,")
        _detail(f"copy them, uncompressed (.jsonl, not .gz), into {EXAMPLES_DIR}")
        _retry_or_quit("experiment datasets missing", "I've copied them, check again")


def _read_env() -> dict[str, str]:
    path = ENV_FILE if ENV_FILE.exists() else REPO_ROOT / ".env.example"
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip("'\"")
    return values


def _write_env_value(key: str, value: str) -> None:
    """Sets KEY=value in .env (created from .env.example if missing),
    replacing an existing KEY= line or appending one."""
    if not ENV_FILE.exists():
        shutil.copy(REPO_ROOT / ".env.example", ENV_FILE)
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    pattern = re.compile(rf"^\s*#?\s*{re.escape(key)}\s*=")
    for i, line in enumerate(lines):
        if pattern.match(line):
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _key_env_for(model: dict) -> str | None:
    provider = (model.get("provider") or "").lower()
    if provider in _NO_AUTH_PROVIDERS:
        return None
    if model.get("api_key_env"):
        return model["api_key_env"]
    # A custom/self-hosted provider may need no key at all, so it's not checked.
    return _PROVIDER_API_KEY_ENV.get(provider)


def _filled_copy_path(models_file: Path) -> Path:
    """Where the wizard writes a models file with its placeholders filled
    in — never over the original (it's versioned), and *.local.json is
    gitignored so real endpoints don't get committed by accident."""
    if models_file.name.endswith(".local.json"):
        return models_file
    return models_file.with_name(f"{models_file.stem}.local.json")


def _load_models_list(models_file: Path) -> list[dict]:
    return json.loads(models_file.read_text(encoding="utf-8")).get("models", [])


def _load_active_models_snapshot() -> list[dict] | None:
    """None if nothing has ever been imported here (the snapshot doesn't
    exist yet) — see ACTIVE_MODELS_SNAPSHOT."""
    if not ACTIVE_MODELS_SNAPSHOT.exists():
        return None
    return json.loads(ACTIVE_MODELS_SNAPSHOT.read_text(encoding="utf-8"))


def _missing_roles(models: list[dict], roles: frozenset[str]) -> list[str]:
    """Roles this run calls that have no model at all in the catalog (e.g.
    models_exp.json has no 'judge') — every call for them would fail."""
    present = {m.get("role") for m in models}
    return sorted(roles - present)


def _live_call_gaps(models: list[dict], roles: frozenset[str]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """For the models with one of `roles`: placeholders still in their rows
    and API-key variables still empty in .env, each mapped to the models
    that need it."""
    env = _read_env()
    placeholders: dict[str, list[str]] = {}
    keys: dict[str, list[str]] = {}
    for model in models:
        if model.get("role") not in roles:
            continue
        label = f"{model.get('name')} ({model.get('role')})"
        for placeholder in sorted(set(run_pipeline._PLACEHOLDER_RE.findall(json.dumps(model)))):
            placeholders.setdefault(placeholder, []).append(label)
        env_var = _key_env_for(model)
        if env_var and not env.get(env_var):
            keys.setdefault(env_var, []).append(label)
    return placeholders, keys


def _fill_in_keys(keys: dict[str, list[str]]) -> None:
    """Asks for each missing key (typed hidden) and saves it in .env."""
    for env_var, users in keys.items():
        print()
        _say(f"{env_var} - used by {', '.join(users)}")
        value = getpass.getpass("  Paste its value (hidden, Enter to skip): ").strip()
        if value:
            _write_env_value(env_var, value)
            _ok(f"Saved {env_var} in .env")


def _fill_in_placeholders(models_file: Path, placeholders: dict[str, list[str]]) -> Path:
    """Asks for each placeholder's real value and writes them to the models
    file's .local.json copy, which becomes the file to load. Returns the
    models file to use from now on."""
    text = models_file.read_text(encoding="utf-8")
    for placeholder, users in placeholders.items():
        print()
        _say(f"{placeholder} - used by {', '.join(users)}")
        value = _input("  Real value (e.g. your resource name or gateway URL, Enter to skip): ")
        if value:
            text = text.replace(placeholder, value)
    target = _filled_copy_path(models_file)
    target.write_text(text, encoding="utf-8")
    _ok(f"Wrote {_relative(target)} (gitignored) - it will be loaded instead of {models_file.name}.")
    return target


def _check_live_calls(plan: Plan) -> None:
    """Loops until the models that will really be called look usable, or the
    user explicitly chooses to go on anyway (or quit).

    For 'route', nothing gets imported by this run — the catalog is
    whatever a *previous* run already loaded into the database, so this
    checks data/current/models.json (ACTIVE_MODELS_SNAPSHOT: the live
    snapshot the API rewrites on every catalog import, see
    app/services/snapshots.py) instead of plan.models_file. Checking the
    file instead would be checking the wrong thing twice over: filling its
    placeholders now wouldn't reach the database either way, AND the file
    the user remembers loading may not even be what's active any more (a
    `.local.json` copy filled in after the fact, but never actually
    (re)loaded, would make this check falsely report "ready"). Every other
    goal checks plan.models_file, correctly: nothing has been imported yet
    at that point, so the file IS the thing about to become real."""
    if not plan.live_roles:
        return
    placeholders_fixable = plan.goal != "route"
    checking_snapshot = plan.goal == "route" and ACTIVE_MODELS_SNAPSHOT.exists()
    if plan.goal == "route" and not checking_snapshot:
        _detail(f"No record of what's actually active ({_relative(ACTIVE_MODELS_SNAPSHOT)} doesn't exist "
                "yet, so nothing looks to have been loaded here before) - checking "
                f"{_relative(plan.models_file)} instead, which may not match the database.")
    while True:
        models = _load_active_models_snapshot() if checking_snapshot else _load_models_list(plan.models_file)
        source_label = "the database's active catalog" if checking_snapshot else _relative(plan.models_file)
        missing_roles = _missing_roles(models, plan.live_roles)
        placeholders, keys = _live_call_gaps(models, plan.live_roles)
        if not missing_roles and not placeholders and not keys:
            _ok(f"Endpoints and API keys are filled in for: {', '.join(sorted(plan.live_roles))}"
                f" ({source_label}).")
            return
        _fail(f"Models that will be called aren't ready ({source_label}):")
        for role in missing_roles:
            _detail(f"- no model with role '{role}' in this catalog"
                    + (" (it grades your own tasks)" if role == "judge" else "")
                    + " - add one to the file (see data/examples/models.json for an example)")
        for placeholder, users in placeholders.items():
            _detail(f"- placeholder {placeholder}, used by {', '.join(users)}")
        for env_var, users in keys.items():
            _detail(f"- {env_var} empty in .env, used by {', '.join(users)}")
        if placeholders and not placeholders_fixable:
            _detail("That catalog was loaded with these placeholders, so the database has them too -")
            _detail("filling them in now wouldn't help. Run the wizard again with 'Reproduce' or")
            _detail("'Custom run' and answer 'y' to the prompt testing: it asks for them first.")

        fill_placeholders = bool(placeholders) and placeholders_fixable
        options, actions = [], []
        if keys or fill_placeholders:
            options.append((f"Enter the missing {'values' if fill_placeholders else 'API keys'} now",
                            "keys are typed hidden and saved in .env"))
            actions.append("fill")
        if keys or fill_placeholders or missing_roles:
            edited = "the files" if fill_placeholders or missing_roles else ".env"
            options.append((f"I've edited {edited} myself, check again", ""))
            actions.append("recheck")
        options.append(("Continue anyway", "the steps that call these models will fail"))
        actions.append("continue")
        options.append(("Quit", ""))
        actions.append("quit")

        # Nothing fixable from here left: continuing is a guaranteed failure.
        default = 1 if actions[0] in ("fill", "recheck") else len(options)
        action = actions[_ask_choice("What now?", options, default=default)]
        if action == "fill":
            _fill_in_keys(keys)
            if placeholders and placeholders_fixable:
                plan.models_file = _fill_in_placeholders(plan.models_file, placeholders)
        elif action == "continue":
            return
        elif action == "quit":
            raise WizardQuit("models not ready to be called")


def _check_requirements(plan: Plan) -> bool:
    """Returns whether the base image still has to be downloaded."""
    _ok(f"Python {sys.version.split()[0]}.")
    _check_docker()
    image = _base_image()
    needs_pull = not _image_present(image)
    if needs_pull and plan.needs_existing_install:
        _fail("RouterKit doesn't seem to be installed here yet (its Docker images are missing),")
        _detail("so there's no " + ("loaded data to train on." if plan.goal == "retrain"
                                    else "trained predictor to route with."))
        _detail("Quit, then run the wizard again and choose 'Reproduce' or 'Custom run' first.")
        _continue_or_quit("nothing installed yet", "download and start everything, then try anyway")
    if needs_pull:
        _todo(f"Base image {image} isn't downloaded yet (one-off, ~10 GB).")
    else:
        _ok(f"Base image {image} is already downloaded.")
    _check_disk(first_install=needs_pull)
    if plan.needs_datasets:
        _check_datasets()
    _check_live_calls(plan)
    return needs_pull


# --------------------------------------------------------------------------
# Phase 3 / 4: plan and run
# --------------------------------------------------------------------------

def _build_commands(plan: Plan, needs_pull: bool) -> list[Command]:
    commands = []
    if needs_pull:
        commands.append(Command("Download the BigCodeBench base image (~10 GB, one-off)",
                                ["docker", "pull", _base_image()]))
    commands.append(Command(
        "Build and start the services (Postgres, bigcodebench-api, routerkit-api)",
        ["python", "scripts/start.py", *([] if plan.needs_datasets else ["--skip-datasets"])],
    ))
    if plan.pipeline_args is not None:
        args = [a for a in plan.pipeline_args if not a.startswith("--models=")]
        if plan.goal != "retrain" and plan.models_file and plan.models_file != DEFAULT_MODELS:
            args.insert(0, f"--models={_relative(plan.models_file)}")
        commands.append(Command(_GOAL_TITLES[plan.goal], ["python", "scripts/run_pipeline.py", *args]))
    return commands


def _placeholder_note(plan: Plan, pipeline: Command | None) -> list[str]:
    """What to tell the user about run_pipeline's own placeholder prompt —
    only when it will actually stop to ask (no --yes) and the models file it
    loads still has placeholders. Those are harmless unless they belong to a
    model this run calls (possible only if the user chose 'Continue anyway')."""
    if pipeline is None or "--yes" in pipeline.program or plan.goal == "retrain":
        return []
    if not run_pipeline._PLACEHOLDER_RE.search(plan.models_file.read_text(encoding="utf-8")):
        return []
    if _live_call_gaps(_load_models_list(plan.models_file), plan.live_roles)[0]:
        return ["Note: the pipeline will warn that the models file still has <...> placeholders.",
                "Some belong to models this run calls (you chose to continue anyway), so the",
                "steps that call them will fail - answer 'n' there if you'd rather stop first."]
    return ["Note: the pipeline will warn that the models file still has <...> placeholders.",
            "They belong to models this run doesn't call - answer 'y' to continue."]


def _option(program: list[str], name: str) -> str | None:
    return next((a.split("=", 1)[1] for a in program if a.startswith(f"--{name}=")), None)


def _resume_program(program: list[str]) -> list[str]:
    """Same pipeline command, with --start-from set to the step the user
    picks — only among the steps this command actually runs, so a retry
    doesn't repeat the ones that already succeeded."""
    steps = run_pipeline.STEP_ORDER
    first = steps.index(_option(program, "start-from") or steps[0])
    last = steps.index(_option(program, "until") or steps[-1])
    own_steps = steps[first : last + 1]
    if len(own_steps) == 1:
        return program
    choice = _ask_choice(
        "Resume from which step? (the error above names the one that failed)",
        [(f"{s} (from the beginning)" if i == 0 else s, "") for i, s in enumerate(own_steps)],
    )
    kept = [a for a in program if not a.startswith("--start-from=")]
    chosen = own_steps[choice]
    return kept if chosen == steps[0] else [*kept, f"--start-from={chosen}"]


def _execute(commands: list[Command], total: int) -> None:
    global _started
    _started = True
    _remaining[:] = commands
    for n, command in enumerate(commands, start=1):
        while True:
            _section(f"Step {n}/{total}: {command.title}")
            _say(f"$ {command.display()}")
            print(flush=True)
            result = subprocess.run(command.argv(), cwd=REPO_ROOT, check=False)
            print()
            if result.returncode == 0:
                _ok(f"Step {n}/{total} done.")
                _remaining.pop(0)
                break

            _fail(f"Step {n}/{total} failed (exit code {result.returncode}) - see the messages above.")
            if command.program[0] == "docker":
                _detail("If the disk filled up or Docker stopped responding, free space, restart")
                _detail("Docker Desktop, then retry - layers already downloaded are kept.")
            if command.is_start:
                _detail("If a dataset download looked incomplete or corrupted above, answer 'y' next")
                _detail("to redownload it instead of reusing the partial file.")
            _retry_or_quit(f"step {n}/{total} failed", "Retry this step")
            if command.is_pipeline:
                command.program = _resume_program(command.program)
            elif command.is_start and "--force-download" not in command.program:
                if _ask_yes_no("Redownload the experiment datasets instead of reusing what's there?",
                                default=False):
                    command.program = [*command.program, "--force-download"]


def _route_loop(n: int, total: int) -> None:
    _section(f"Step {n}/{total}: Try the router on your prompts")
    _say("Type a task prompt to see which models the router recommends, best first.")
    _say("Leave it empty to finish.")
    while True:
        print()
        prompt = _input("? Prompt: ")
        if not prompt:
            _ok(f"Step {n}/{total} done.")
            return
        command = Command("Route a prompt", ["python", "scripts/test_route.py", prompt])
        _say(f"$ {command.display()}")
        subprocess.run(command.argv(), cwd=REPO_ROOT, check=False)


def main() -> None:
    _section("RouterKit setup wizard")
    _say("Installs and runs RouterKit step by step. Press Ctrl+C at any time to stop.")

    _phase(1, "Questions")
    plan = _ask_questions()

    _phase(2, "Checking requirements")
    needs_pull = _check_requirements(plan)

    _phase(3, "Plan")
    commands = _build_commands(plan, needs_pull)
    total = len(commands) + plan.want_routing
    for n, command in enumerate(commands, start=1):
        _say(f"{n}. {command.title}")
        _say(f"     $ {command.display()}")
    if plan.want_routing:
        _say(f"{total}. Try the router on your prompts")
        _say('     $ python scripts/test_route.py "<your prompt>"')
    pipeline = commands[-1] if commands[-1].is_pipeline else None
    note = _placeholder_note(plan, pipeline)
    if note:
        print()
        for line in note:
            _say(line)
    print()
    if not _ask_yes_no("Start now?", default=True):
        raise WizardQuit("plan not confirmed")

    _phase(4, "Running")
    _execute(commands, total)
    trained = pipeline is not None and "--until=validations" not in pipeline.program
    if trained and REPORT_INDEX.exists():
        print()
        _say(f"Training report: {REPORT_INDEX}")
        if _ask_yes_no("Open it in your browser now?", default=True):
            webbrowser.open(REPORT_INDEX.as_uri())
    if plan.want_routing:
        _route_loop(total, total)

    _section("Finished")
    _say("All done. To repeat this later without the wizard, run the commands shown in the plan.")


def _report_stop(reason: str) -> None:
    print()
    _say(f"Stopped{f' ({reason})' if reason else ''}.")
    if not _started:
        _say("Nothing has been run yet.")
        return
    if not _remaining:
        _say("Every step of the plan had already finished.")
        return
    _say("Finished steps and downloads are kept. To continue, run the wizard again,")
    _say("or run what's left yourself:")
    for command in _remaining:
        _say(f"  $ {command.display()}")
    if any(c.is_pipeline for c in _remaining):
        _say("(to skip the pipeline steps that already finished, set --start-from=<step> -")
        _say("the pipeline's last message names the step to resume from)")


if __name__ == "__main__":
    try:
        main()
    except WizardQuit as exc:
        _report_stop(str(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        _report_stop("Ctrl+C")
        sys.exit(1)
