#!/usr/bin/env python3
"""Bootstraps the environment: downloads the heavy experiment datasets (if
missing), brings up the docker-compose stack, and waits until the API
answers. Stdlib only — no pip install needed to run this.

Usage:
    python scripts/start.py [--force-download] [--base-url http://localhost:8000]

Whether the local bigcodebench-api container (~25GB image) gets built and
started depends on .env's BIGCODEBENCH_API_URL: left at its default (or
unset), it's built locally like every other service; pointed at a
separately-deployed instance instead, it's skipped entirely (docker
compose's "local-eval" profile) — routerkit-api only ever needs that URL
reachable over HTTP, nothing else changes.

After this finishes, use scripts/run_pipeline.py to load catalogs/data and
train a predictor through the API itself.
"""

import argparse
import gzip
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "data" / "examples"
RELEASE_BASE = "https://github.com/Timia-Group/routerkit/releases/latest/download"
HEAVY_DATASETS = ["solutions_exp.jsonl", "validations_exp.jsonl"]
# Must match docker-compose.yml's routerkit-api environment and .env.example.
_DEFAULT_BIGCODEBENCH_URL = "http://bigcodebench-api:8000"


def _log(msg: str) -> None:
    print(f"[start] {msg}", flush=True)


def ensure_env_file() -> None:
    env_path = REPO_ROOT / ".env"
    example_path = REPO_ROOT / ".env.example"
    if env_path.exists():
        return
    shutil.copy(example_path, env_path)
    _log(".env created from .env.example — fill in at least one LLM API key if you plan to "
         "use 'calculate' mode or POST /route later (not needed for the default load+train path).")


def _wants_local_eval_container() -> bool:
    """True unless .env points BIGCODEBENCH_API_URL somewhere other than
    the local container — in which case there's nothing to build/start for
    it (see docker-compose.yml's "local-eval" profile)."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return True
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "BIGCODEBENCH_API_URL":
            value = value.strip()
            return value in ("", _DEFAULT_BIGCODEBENCH_URL)
    return True


def ensure_dataset(name: str, force: bool) -> None:
    target = EXAMPLES_DIR / name
    if target.exists() and not force:
        _log(f"{name}: already present, skipping download.")
        return

    gz_name = f"{name}.gz"
    url = f"{RELEASE_BASE}/{gz_name}"
    gz_path = EXAMPLES_DIR / gz_name
    _log(f"{name}: downloading {url} ...")
    try:
        urllib.request.urlretrieve(url, gz_path)  # noqa: S310 - fixed, hardcoded GitHub Releases URL
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"[start] Failed to download {url} ({exc.code}). If this repo is still private, "
            f"you need push/read access and the release must exist — see "
            f"data/examples/README.md for the manual download command, or ask whoever "
            f"maintains the repo for the files directly."
        ) from exc

    _log(f"{name}: decompressing ...")
    with gzip.open(gz_path, "rb") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
    gz_path.unlink()
    _log(f"{name}: ready ({target.stat().st_size:,} bytes).")


def compose_up(local_eval: bool) -> None:
    cmd = ["docker", "compose"]
    if local_eval:
        cmd += ["--profile", "local-eval"]
    cmd += ["up", "-d", "--build"]
    _log(f"{' '.join(cmd)} ...")
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(f"[start] docker compose up failed (exit code {result.returncode}).")


def wait_for_api(base_url: str, timeout_seconds: int = 180) -> None:
    _log(f"waiting for {base_url} to respond ...")
    deadline = time.monotonic() + timeout_seconds
    last_print = 0.0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/openapi.json", timeout=3) as resp:  # noqa: S310
                if resp.status == 200:
                    _log("API is up.")
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        now = time.monotonic()
        if now - last_print >= 10:
            _log("still waiting ...")
            last_print = now
        time.sleep(2)
    raise SystemExit(f"[start] API did not respond at {base_url} within {timeout_seconds}s.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--force-download", action="store_true",
        help="Re-download the heavy datasets even if they already exist locally.",
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000",
        help="Where routerkit-api will be reachable once up (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-datasets", action="store_true",
        help="Don't download/check the heavy datasets at all (e.g. you only plan to use "
        "'calculate' mode, which doesn't need them).",
    )
    args = parser.parse_args()

    ensure_env_file()
    if not args.skip_datasets:
        for name in HEAVY_DATASETS:
            ensure_dataset(name, force=args.force_download)
    local_eval = _wants_local_eval_container()
    if local_eval:
        _log("BIGCODEBENCH_API_URL is unset/local — building and starting the local eval container too.")
    else:
        _log("BIGCODEBENCH_API_URL points elsewhere in .env — skipping the local eval container.")
    compose_up(local_eval)
    wait_for_api(args.base_url)
    _log("Done. Next: python scripts/run_pipeline.py --help")


if __name__ == "__main__":
    main()
